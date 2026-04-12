package process

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type State string

const (
	StateStopped  State = "stopped"
	StateStarting State = "starting"
	StateRunning  State = "running"
	StateStopping State = "stopping"
	StateError    State = "error"
)

type Status struct {
	State     State     `json:"state"`
	ModelID   string    `json:"model_id,omitempty"`
	PID       int       `json:"pid,omitempty"`
	StartedAt time.Time `json:"started_at,omitempty"`
	Uptime    string    `json:"uptime,omitempty"`
	Error     string    `json:"error,omitempty"`
}

// Manager manages a single vLLM process.
type Manager struct {
	mu          sync.RWMutex
	cmd         *exec.Cmd
	state       State
	modelID     string
	pid         int
	startedAt   time.Time
	lastError   string
	cancelFunc  context.CancelFunc
	vllmHost    string
	vllmPort    int

	// Log ring buffer
	logMu  sync.Mutex
	logBuf []string
	logMax int

	// Log subscribers
	subMu sync.Mutex
	subs  map[chan string]struct{}
}

func NewManager(vllmHost string, vllmPort int) *Manager {
	return &Manager{
		state:    StateStopped,
		vllmHost: vllmHost,
		vllmPort: vllmPort,
		logBuf:   make([]string, 0, 1000),
		logMax:   1000,
		subs:     make(map[chan string]struct{}),
	}
}

func (m *Manager) GetStatus() Status {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s := Status{
		State:   m.state,
		ModelID: m.modelID,
		PID:     m.pid,
	}
	if m.state == StateRunning || m.state == StateStarting {
		s.StartedAt = m.startedAt
		s.Uptime = time.Since(m.startedAt).Truncate(time.Second).String()
	}
	if m.lastError != "" {
		s.Error = m.lastError
	}
	return s
}

// Start launches vLLM with the given model path and config flags.
func (m *Manager) Start(modelID, modelPath string, args []string, env []string) error {
	m.mu.Lock()
	if m.state == StateRunning || m.state == StateStarting {
		m.mu.Unlock()
		return fmt.Errorf("vLLM is already running (state: %s)", m.state)
	}
	m.state = StateStarting
	m.modelID = modelID
	m.lastError = ""
	m.startedAt = time.Now()
	m.mu.Unlock()

	// Clear log buffer
	m.logMu.Lock()
	m.logBuf = m.logBuf[:0]
	m.logMu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancelFunc = cancel
	m.mu.Unlock()

	cmdArgs := append([]string{"serve", modelPath,
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(m.vllmPort),
	}, args...)

	cmd := exec.CommandContext(ctx, "vllm", cmdArgs...)
	cmd.Env = append(os.Environ(), env...)

	// Capture stdout and stderr
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()

	slog.Info("starting vLLM", "model", modelID, "args", cmdArgs)

	if err := cmd.Start(); err != nil {
		m.mu.Lock()
		m.state = StateError
		m.lastError = err.Error()
		m.mu.Unlock()
		cancel()
		return fmt.Errorf("start vllm: %w", err)
	}

	m.mu.Lock()
	m.cmd = cmd
	m.pid = cmd.Process.Pid
	m.mu.Unlock()

	// Stream logs
	go m.streamOutput(stdout)
	go m.streamOutput(stderr)

	// Wait for process in background
	go m.waitForExit(cmd, cancel)

	// Poll for readiness
	go m.waitForReady()

	return nil
}

// Stop gracefully stops vLLM.
func (m *Manager) Stop() error {
	m.mu.Lock()
	if m.state != StateRunning && m.state != StateStarting {
		m.mu.Unlock()
		return fmt.Errorf("vLLM is not running (state: %s)", m.state)
	}
	m.state = StateStopping
	cancel := m.cancelFunc
	m.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	// Wait for process to exit (up to 30s)
	deadline := time.After(30 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline:
			m.mu.Lock()
			if m.cmd != nil && m.cmd.Process != nil {
				m.cmd.Process.Kill()
			}
			m.state = StateStopped
			m.mu.Unlock()
			return nil
		case <-ticker.C:
			m.mu.RLock()
			state := m.state
			m.mu.RUnlock()
			if state == StateStopped || state == StateError {
				return nil
			}
		}
	}
}

// Restart stops and starts vLLM with the same model.
func (m *Manager) Restart(modelID, modelPath string, args []string, env []string) error {
	if m.GetStatus().State != StateStopped {
		if err := m.Stop(); err != nil {
			return err
		}
		// Brief pause for cleanup
		time.Sleep(500 * time.Millisecond)
	}
	return m.Start(modelID, modelPath, args, env)
}

// ClearLogs empties the log buffer.
func (m *Manager) ClearLogs() {
	m.logMu.Lock()
	m.logBuf = m.logBuf[:0]
	m.logMu.Unlock()
}

// RecentLogs returns the most recent log lines.
func (m *Manager) RecentLogs(n int) []string {
	m.logMu.Lock()
	defer m.logMu.Unlock()

	if n <= 0 || n > len(m.logBuf) {
		n = len(m.logBuf)
	}
	start := len(m.logBuf) - n
	out := make([]string, n)
	copy(out, m.logBuf[start:])
	return out
}

// SubscribeLogs returns a channel that receives log lines.
func (m *Manager) SubscribeLogs() chan string {
	ch := make(chan string, 64)
	m.subMu.Lock()
	m.subs[ch] = struct{}{}
	m.subMu.Unlock()
	return ch
}

// UnsubscribeLogs removes a log subscriber.
func (m *Manager) UnsubscribeLogs(ch chan string) {
	m.subMu.Lock()
	delete(m.subs, ch)
	m.subMu.Unlock()
}

func (m *Manager) streamOutput(r io.ReadCloser) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 256*1024)
	for scanner.Scan() {
		line := scanner.Text()
		m.appendLog(line)

		// Detect ready signal
		if strings.Contains(line, "Uvicorn running on") || strings.Contains(line, "Application startup complete") {
			m.mu.Lock()
			if m.state == StateStarting {
				m.state = StateRunning
			}
			m.mu.Unlock()
		}
	}
}

func (m *Manager) appendLog(line string) {
	m.logMu.Lock()
	if len(m.logBuf) >= m.logMax {
		m.logBuf = m.logBuf[1:]
	}
	m.logBuf = append(m.logBuf, line)
	m.logMu.Unlock()

	// Fan out to subscribers
	m.subMu.Lock()
	for ch := range m.subs {
		select {
		case ch <- line:
		default:
		}
	}
	m.subMu.Unlock()
}

func (m *Manager) waitForExit(cmd *exec.Cmd, cancel context.CancelFunc) {
	err := cmd.Wait()
	cancel()

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.state == StateStopping {
		m.state = StateStopped
		return
	}

	if err != nil {
		m.state = StateError
		m.lastError = err.Error()
		slog.Error("vLLM process exited", "error", err)
	} else {
		m.state = StateStopped
	}
}

func (m *Manager) waitForReady() {
	healthURL := fmt.Sprintf("http://%s:%d/health", m.vllmHost, m.vllmPort)
	timeout := time.After(10 * time.Minute) // large models take a while
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			m.mu.Lock()
			if m.state == StateStarting {
				m.state = StateError
				m.lastError = "startup timeout (10 minutes)"
			}
			m.mu.Unlock()
			return
		case <-ticker.C:
			m.mu.RLock()
			state := m.state
			m.mu.RUnlock()

			if state != StateStarting && state != StateRunning {
				return
			}

			resp, err := http.Get(healthURL)
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					m.mu.Lock()
					if m.state == StateStarting {
						m.state = StateRunning
						slog.Info("vLLM is ready", "model", m.modelID)
					}
					m.mu.Unlock()
					return
				}
			}
		}
	}
}

// BuildArgs constructs vLLM CLI arguments from a model config.
func BuildArgs(cfg VLLMStartConfig) []string {
	var args []string

	if cfg.Dtype != "" && cfg.Dtype != "auto" {
		args = append(args, "--dtype", cfg.Dtype)
	}
	if cfg.MaxModelLen > 0 {
		args = append(args, "--max-model-len", strconv.Itoa(cfg.MaxModelLen))
	}
	if cfg.TensorParallelSize > 1 {
		args = append(args, "--tensor-parallel-size", strconv.Itoa(cfg.TensorParallelSize))
	}
	if cfg.GPUMemoryUtilization > 0 && cfg.GPUMemoryUtilization != 0.90 {
		args = append(args, "--gpu-memory-utilization", fmt.Sprintf("%.2f", cfg.GPUMemoryUtilization))
	}
	if cfg.EnforceEager {
		args = append(args, "--enforce-eager")
	}
	if cfg.TrustRemoteCode {
		args = append(args, "--trust-remote-code")
	}
	if cfg.MaxNumSeqs > 0 && cfg.MaxNumSeqs != 16 {
		args = append(args, "--max-num-seqs", strconv.Itoa(cfg.MaxNumSeqs))
	}
	if cfg.Quantization != "" {
		args = append(args, "--quantization", cfg.Quantization)
	}
	if cfg.LoadFormat != "" && cfg.LoadFormat != "auto" {
		args = append(args, "--load-format", cfg.LoadFormat)
	}
	if cfg.EnablePrefixCaching {
		args = append(args, "--enable-prefix-caching")
	}
	if cfg.KVCacheDtype != "" && cfg.KVCacheDtype != "auto" {
		args = append(args, "--kv-cache-dtype", cfg.KVCacheDtype)
	}
	if cfg.EnableChunkedPrefill {
		args = append(args, "--enable-chunked-prefill")
	}
	if cfg.MaxNumBatchedTokens > 0 {
		args = append(args, "--max-num-batched-tokens", strconv.Itoa(cfg.MaxNumBatchedTokens))
	}
	if cfg.EnableAutoToolChoice && cfg.ToolCallParser != "" {
		args = append(args, "--enable-auto-tool-choice", "--tool-call-parser", cfg.ToolCallParser)
	}
	if cfg.Tokenizer != "" {
		args = append(args, "--tokenizer", cfg.Tokenizer)
	}
	if cfg.ChatTemplate != "" {
		args = append(args, "--chat-template", cfg.ChatTemplate)
	}
	if cfg.ExtraFlags != "" {
		for _, f := range strings.Fields(cfg.ExtraFlags) {
			args = append(args, f)
		}
	}

	return args
}

// BuildEnv constructs environment variables for the vLLM process.
func BuildEnv(quantMethod string) []string {
	env := []string{
		"VLLM_TARGET_DEVICE=rocm",
		"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1",
		"FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE",
	}
	if quantMethod == "awq" {
		env = append(env, "VLLM_USE_TRITON_AWQ=1")
	}
	return env
}

// ResolveModelPath determines the model path for vLLM.
func ResolveModelPath(localPath string) string {
	// Check for GGUF files
	matches, _ := filepath.Glob(filepath.Join(localPath, "*.gguf"))
	if len(matches) == 1 {
		return matches[0] // single GGUF file
	}
	return localPath // directory-based model
}

// VLLMStartConfig mirrors the config fields needed to build the command.
type VLLMStartConfig struct {
	Dtype                string
	MaxModelLen          int
	TensorParallelSize   int
	GPUMemoryUtilization float64
	EnforceEager         bool
	TrustRemoteCode      bool
	MaxNumSeqs           int
	Quantization         string
	LoadFormat           string
	EnablePrefixCaching  bool
	KVCacheDtype         string
	EnableChunkedPrefill bool
	MaxNumBatchedTokens  int
	EnableAutoToolChoice bool
	ToolCallParser       string
	Tokenizer            string
	ChatTemplate         string
	ExtraFlags           string
}

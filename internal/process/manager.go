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
	"syscall"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/ansi"
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

	// Args is the flag list the running engine was launched with, after the
	// model path and host/port. Recorded because "is the right thing running?"
	// cannot be answered by the model id alone: a benchmark sweep serves one
	// model at several context lengths, and every one of those is the same
	// model with different arguments.
	Args []string `json:"args,omitempty"`
}

// Launcher is the argv prefix used to start a server. The generic image runs
// `vllm serve <model> …`; the radiance image runs its entrypoint script, which
// takes the same arguments, prints the arch/P2P/version banner, optionally
// applies NUMA binding, and then execs `vllm serve` itself.
type Launcher struct {
	Bin  string   // executable
	Args []string // argv prefix inserted before the model path
}

// DefaultLauncher is plain `vllm serve`.
var DefaultLauncher = Launcher{Bin: "vllm", Args: []string{"serve"}}

// Manager manages a single vLLM process.
type Manager struct {
	mu      sync.RWMutex
	cmd     *exec.Cmd
	state   State
	modelID string
	// args is the flag list the running process was launched with; see Status.
	args       []string
	pid        int
	startedAt  time.Time
	lastError  string
	cancelFunc context.CancelFunc
	vllmHost   string
	vllmPort   int
	launcher   Launcher

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
		launcher: DefaultLauncher,
		logBuf:   make([]string, 0, 5000),
		logMax:   5000,
		subs:     make(map[chan string]struct{}),
	}
}

// SetLauncher overrides how the server process is spawned. Called once at boot
// from the detected image environment.
func (m *Manager) SetLauncher(l Launcher) {
	if l.Bin == "" {
		return
	}
	m.mu.Lock()
	m.launcher = l
	m.mu.Unlock()
}

func (m *Manager) GetStatus() Status {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s := Status{
		State:   m.state,
		ModelID: m.modelID,
		PID:     m.pid,
		Args:    append([]string(nil), m.args...),
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
	m.args = append([]string(nil), args...)
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

	m.mu.RLock()
	launcher := m.launcher
	m.mu.RUnlock()
	if launcher.Bin == "" {
		launcher = DefaultLauncher
	}

	cmdArgs := append(append([]string{}, launcher.Args...), modelPath,
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(m.vllmPort),
	)
	cmdArgs = append(cmdArgs, args...)

	cmd := exec.CommandContext(ctx, launcher.Bin, cmdArgs...)
	cmd.Env = append(os.Environ(), env...)

	// Put the server in its own process group so the whole tree can be
	// signalled at once.
	//
	// vLLM is not one process: it forks an EngineCore and one Worker per
	// tensor-parallel rank, and those are our grandchildren. Signalling only
	// the direct child leaves them running, still holding their HIP contexts
	// -- which is to say still holding the GPU memory. The symptom is a later
	// start failing with "Free memory on device cuda:0 (2.82/31.86 GiB) on
	// startup is less than desired", on exactly the GPUs the previous serve
	// used, while the UI reports nothing is running.
	//
	// A new group is what makes this safe: the workers inherit it, so
	// kill(-pgid) reaches every one of them without also signalling vllmctl,
	// which shares its own group with them otherwise.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return killProcessGroup(cmd.Process.Pid, syscall.SIGKILL)
	}
	// Bound how long Wait blocks on the output pipes: a surviving grandchild
	// holds them open, and without this the reaper never returns.
	cmd.WaitDelay = 10 * time.Second

	// Capture stdout and stderr
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()

	slog.Info("starting vLLM", "model", modelID, "bin", launcher.Bin, "args", cmdArgs)

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

// Stop gracefully stops vLLM and everything it spawned.
//
// SIGTERM to the process group first: vLLM shuts down cleanly on it, releasing
// GPU memory and tearing down NCCL. Only if that does not finish in time does
// this escalate to SIGKILL. Either way the whole group is signalled, not just
// the process we launched -- see the comment in Start.
func (m *Manager) Stop() error {
	m.mu.Lock()
	if m.state != StateRunning && m.state != StateStarting {
		m.mu.Unlock()
		return fmt.Errorf("vLLM is not running (state: %s)", m.state)
	}
	m.state = StateStopping
	cancel := m.cancelFunc
	pgid := m.pid
	m.mu.Unlock()

	if pgid > 0 {
		if err := killProcessGroup(pgid, syscall.SIGTERM); err != nil {
			slog.Warn("signalling vLLM process group", "pgid", pgid, "error", err)
		}
	}

	// finish sweeps up anything in the group that outlived the leader, then
	// releases the context.
	finish := func() {
		if pgid > 0 {
			// Only sweep while the group still exists, so a group id recycled
			// after everything exited cannot be signalled by mistake.
			if killProcessGroup(pgid, 0) == nil {
				slog.Info("reaping vLLM workers that outlived the server", "pgid", pgid)
				killProcessGroup(pgid, syscall.SIGKILL)
			}
		}
		if cancel != nil {
			cancel()
		}
	}

	deadline := time.After(30 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline:
			slog.Warn("vLLM did not exit on SIGTERM; killing the process group", "pgid", pgid)
			finish()
			m.mu.Lock()
			m.state = StateStopped
			m.mu.Unlock()
			return nil
		case <-ticker.C:
			m.mu.RLock()
			state := m.state
			m.mu.RUnlock()
			if state == StateStopped || state == StateError {
				finish()
				return nil
			}
		}
	}
}

// killProcessGroup signals every process in the group led by pid. Signal 0
// tests for the group's existence without delivering anything.
func killProcessGroup(pid int, sig syscall.Signal) error {
	if pid <= 0 {
		return nil
	}
	return syscall.Kill(-pid, sig)
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
		// Strip before anything else looks at the line: vLLM and the radiance
		// banner colour their output, and an embedded escape would both
		// corrupt the display and break the readiness match below.
		line := ansi.Strip(scanner.Text())
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
	if cfg.ReasoningParser != "" {
		args = append(args, "--reasoning-parser", cfg.ReasoningParser)
	}
	if cfg.AttentionBackend != "" {
		args = append(args, "--attention-backend", cfg.AttentionBackend)
	}
	// Hybrid (gated-delta-net / mamba) models leave automatic prefix caching
	// off unless the cache mode is set too; "align" makes the linear-attention
	// layers prefix-cacheable by snapshotting their state at block boundaries.
	if cfg.MambaCacheMode != "" {
		args = append(args, "--mamba-cache-mode", cfg.MambaCacheMode)
	}
	if cfg.SpeculativeConfig != "" {
		args = append(args, "--speculative-config="+cfg.SpeculativeConfig)
	}
	if cfg.CompilationConfig != "" {
		args = append(args, "--compilation-config="+cfg.CompilationConfig)
	}
	// vLLM's memory profiler measures headroom through torch, which
	// under-reports free VRAM; an explicitly sized pool recovers it. Pins the
	// KV cache, so it must be cleared before measuring anything memory-related.
	if cfg.KVCacheMemory > 0 {
		args = append(args, "--kv-cache-memory", strconv.FormatInt(cfg.KVCacheMemory, 10))
	}
	// disable_padded_drafter_batch (common in speculative configs) is
	// incompatible with async scheduling; vLLM would otherwise auto-enable it
	// and then disable it again with a runtime warning.
	if cfg.DisableAsyncScheduling {
		args = append(args, "--no-async-scheduling")
	}
	if cfg.LanguageModelOnly {
		args = append(args, "--language-model-only")
	}
	if cfg.Tokenizer != "" {
		args = append(args, "--tokenizer", cfg.Tokenizer)
	}
	if cfg.ChatTemplate != "" {
		args = append(args, "--chat-template", cfg.ChatTemplate)
	}
	args = append(args, SplitFlags(cfg.ExtraFlags)...)

	return args
}

// SplitFlags splits a raw extra-flags string into argv entries.
//
// Whitespace separates, but quotes group — vLLM's structured flags carry JSON,
// and plain field splitting tears a spaced JSON object into broken fragments:
//
//	--speculative-config='{"method": "mtp", "num_speculative_tokens": 8}'
//
// Quoting follows shell rules closely enough not to surprise anyone pasting a
// command line: single quotes are literal, double quotes let a backslash escape
// a quote or another backslash, and outside quotes a backslash escapes whatever
// follows. An unterminated quote or a trailing backslash runs to end of input
// rather than erroring, so a half-typed flag degrades instead of vanishing.
func SplitFlags(s string) []string {
	var (
		out  []string
		cur  strings.Builder
		open bool // a quote was seen in this token, so "" yields an empty arg
	)
	flush := func() {
		if open || cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
			open = false
		}
	}

	r := []rune(s)
	for i := 0; i < len(r); i++ {
		switch c := r[i]; c {
		case '\'':
			open = true
			for i++; i < len(r) && r[i] != '\''; i++ {
				cur.WriteRune(r[i]) // single quotes: everything is literal
			}
		case '"':
			open = true
			for i++; i < len(r) && r[i] != '"'; i++ {
				if r[i] == '\\' && i+1 < len(r) && (r[i+1] == '"' || r[i+1] == '\\') {
					i++
				}
				cur.WriteRune(r[i])
			}
		case '\\':
			if i+1 < len(r) {
				i++
				cur.WriteRune(r[i])
			}
		case ' ', '\t', '\n', '\r':
			flush()
		default:
			cur.WriteRune(c)
		}
	}
	flush()
	return out
}

// BuildEnv constructs environment variables for the vLLM process.
// ROCm-specific vars (VLLM_TARGET_DEVICE, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL,
// FLASH_ATTENTION_TRITON_AMD_ENABLE) are set in the container's Dockerfile and
// inherited via os.Environ(), so they are not duplicated here.
//
// extra carries image-variant overrides (the RADIANCE_* switches). They are
// appended last so they win over anything inherited from the image.
func BuildEnv(quantMethod string, extra ...string) []string {
	env := []string{
		// Force vLLM to use spawn start method so child process logs are captured
		"VLLM_WORKER_MULTIPROC_METHOD=spawn",
		// Disable log buffering so errors appear immediately
		"PYTHONUNBUFFERED=1",
	}
	if quantMethod == "awq" {
		env = append(env, "VLLM_USE_TRITON_AWQ=1")
	}
	return append(env, extra...)
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
	Dtype                  string
	MaxModelLen            int
	TensorParallelSize     int
	GPUMemoryUtilization   float64
	EnforceEager           bool
	TrustRemoteCode        bool
	MaxNumSeqs             int
	Quantization           string
	LoadFormat             string
	EnablePrefixCaching    bool
	KVCacheDtype           string
	EnableChunkedPrefill   bool
	MaxNumBatchedTokens    int
	EnableAutoToolChoice   bool
	ToolCallParser         string
	ReasoningParser        string
	AttentionBackend       string
	MambaCacheMode         string
	SpeculativeConfig      string
	CompilationConfig      string
	KVCacheMemory          int64
	DisableAsyncScheduling bool
	LanguageModelOnly      bool
	Tokenizer              string
	ChatTemplate           string
	ExtraFlags             string
}

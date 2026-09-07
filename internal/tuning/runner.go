package tuning

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/ansi"
)

type JobState string

const (
	StateQueued    JobState = "queued"
	StateRunning   JobState = "running"
	StateCompleted JobState = "completed"
	StateFailed    JobState = "failed"
	StateCancelled JobState = "cancelled"
)

type Job struct {
	ID         string    `json:"id"`
	ModelID    string    `json:"model_id"`
	DeviceName string    `json:"device_name"`
	TPSize     int       `json:"tp_size"`
	BlockN     int       `json:"block_n"`
	BlockK     int       `json:"block_k"`
	Shapes     []Shape   `json:"shapes"`
	State      JobState  `json:"state"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at,omitempty"`
	Error      string    `json:"error,omitempty"`
}

// VLLMStopper lets the runner ask the process manager to stop vLLM before
// running. Decouples this package from internal/process to avoid cycles.
type VLLMStopper interface {
	Stop() error
}

type Manager struct {
	dataDir    string
	deviceName string
	tunerPath  string // path to tune_fp8_wrapper.py inside container
	python     string // interpreter that owns the vLLM install
	configsDir string // vLLM's block-FP8 config dir, for hot-installing results
	vllmStop   VLLMStopper

	mu     sync.RWMutex
	active *Job

	logMu  sync.Mutex
	logBuf []string

	subMu sync.Mutex
	subs  map[chan string]struct{}

	cancel context.CancelFunc
}

const logMax = 5000

func NewManager(dataDir, deviceName, tunerPath string, vllmStop VLLMStopper) *Manager {
	return &Manager{
		dataDir:    dataDir,
		deviceName: deviceName,
		tunerPath:  tunerPath,
		python:     "python",
		configsDir: DefaultVLLMConfigsDir,
		vllmStop:   vllmStop,
		logBuf:     make([]string, 0, 256),
		subs:       map[chan string]struct{}{},
	}
}

// SetPython points the tuner at the interpreter that owns the vLLM install.
func (m *Manager) SetPython(python string) {
	if python == "" {
		return
	}
	m.mu.Lock()
	m.python = python
	m.mu.Unlock()
}

// SetConfigsDir points the tuner at vLLM's block-FP8 config directory.
func (m *Manager) SetConfigsDir(dir string) {
	if dir == "" {
		return
	}
	m.mu.Lock()
	m.configsDir = dir
	m.mu.Unlock()
}

// SetDeviceName updates the device name tuned configs are keyed by. Called
// once the boot-time probe resolves what the running vLLM actually reports;
// until then the manager holds the architecture-derived fallback.
func (m *Manager) SetDeviceName(name string) {
	if name == "" {
		return
	}
	m.mu.Lock()
	m.deviceName = name
	m.mu.Unlock()
}

func (m *Manager) DeviceName() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.deviceName
}

// TunedDir returns the directory where tuned config JSONs are persisted on
// the data volume, keyed by GPU architecture so multi-host setups don't
// clobber each other.
func (m *Manager) TunedDir() string {
	return filepath.Join(m.dataDir, "tuned-kernels", m.DeviceName())
}

// IsTuned reports whether a config JSON exists for the given shape.
func (m *Manager) IsTuned(s Shape, blockN, blockK int) bool {
	path := filepath.Join(m.TunedDir(), ConfigFilename(s, m.DeviceName(), blockN, blockK))
	_, err := os.Stat(path)
	return err == nil
}

// ActiveJob returns a snapshot of the currently running (or last) job.
func (m *Manager) ActiveJob() *Job {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.active == nil {
		return nil
	}
	cp := *m.active
	return &cp
}

func (m *Manager) Cancel() {
	m.mu.Lock()
	cancel := m.cancel
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// LogBuffer returns a copy of recent log lines.
func (m *Manager) LogBuffer() []string {
	m.logMu.Lock()
	defer m.logMu.Unlock()
	out := make([]string, len(m.logBuf))
	copy(out, m.logBuf)
	return out
}

// Subscribe returns a channel that receives new log lines until Unsubscribe
// is called or the job ends. The channel is buffered; slow consumers drop.
func (m *Manager) Subscribe() chan string {
	ch := make(chan string, 256)
	m.subMu.Lock()
	m.subs[ch] = struct{}{}
	m.subMu.Unlock()
	return ch
}

func (m *Manager) Unsubscribe(ch chan string) {
	m.subMu.Lock()
	if _, ok := m.subs[ch]; ok {
		delete(m.subs, ch)
		close(ch)
	}
	m.subMu.Unlock()
}

func (m *Manager) appendLog(line string) {
	m.logMu.Lock()
	m.logBuf = append(m.logBuf, line)
	if len(m.logBuf) > logMax {
		m.logBuf = m.logBuf[len(m.logBuf)-logMax:]
	}
	m.logMu.Unlock()

	m.subMu.Lock()
	for ch := range m.subs {
		select {
		case ch <- line:
		default:
			// slow consumer — drop
		}
	}
	m.subMu.Unlock()
}

func (m *Manager) fanoutClose() {
	m.subMu.Lock()
	for ch := range m.subs {
		close(ch)
		delete(m.subs, ch)
	}
	m.subMu.Unlock()
}

// StartJob spawns the tuner subprocess for the given shapes. Returns the job
// ID. Only one job can run at a time; second call while a job is active
// returns an error.
func (m *Manager) StartJob(modelID string, shapes []Shape, tpSize, blockN, blockK int) (*Job, error) {
	m.mu.Lock()
	if m.active != nil && m.active.State == StateRunning {
		m.mu.Unlock()
		return nil, fmt.Errorf("tuning job already running: %s", m.active.ID)
	}

	id := newJobID()
	job := &Job{
		ID:         id,
		ModelID:    modelID,
		DeviceName: m.deviceName,
		TPSize:     tpSize,
		BlockN:     blockN,
		BlockK:     blockK,
		Shapes:     shapes,
		State:      StateRunning,
		StartedAt:  time.Now(),
	}
	m.active = job

	// Reset log buffer for the new job.
	m.logMu.Lock()
	m.logBuf = m.logBuf[:0]
	m.logMu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m.mu.Unlock()

	go m.runJob(ctx, job, shapes, tpSize, blockN, blockK)
	return job, nil
}

func (m *Manager) runJob(ctx context.Context, job *Job, shapes []Shape, tpSize, blockN, blockK int) {
	defer m.fanoutClose()

	finish := func(state JobState, errMsg string) {
		m.mu.Lock()
		job.State = state
		job.FinishedAt = time.Now()
		job.Error = errMsg
		m.cancel = nil
		m.mu.Unlock()
	}

	// Stop vLLM first — the tuner needs full GPU access.
	if m.vllmStop != nil {
		m.appendLog("[tuner] stopping vLLM (if running)")
		if err := m.vllmStop.Stop(); err != nil {
			m.appendLog(fmt.Sprintf("[tuner] stop returned: %v (continuing)", err))
		}
	}

	if err := os.MkdirAll(m.TunedDir(), 0o755); err != nil {
		m.appendLog(fmt.Sprintf("[tuner] mkdir failed: %v", err))
		finish(StateFailed, err.Error())
		return
	}

	if len(shapes) == 0 {
		finish(StateFailed, "no shapes to tune")
		return
	}

	// Encode shapes for the wrapper script.
	shapeStrs := make([]string, len(shapes))
	for i, s := range shapes {
		shapeStrs[i] = fmt.Sprintf("%d:%d", s.N, s.K)
	}

	args := []string{
		m.tunerPath,
		"--shapes", strings.Join(shapeStrs, ","),
		"--tp-size", fmt.Sprintf("%d", tpSize),
		"--block-n", fmt.Sprintf("%d", blockN),
		"--block-k", fmt.Sprintf("%d", blockK),
		"--save-path", m.TunedDir(),
	}

	m.mu.RLock()
	python, configsDir := m.python, m.configsDir
	m.mu.RUnlock()

	m.appendLog(fmt.Sprintf("[tuner] running: %s %s", python, strings.Join(args, " ")))
	m.appendLog(fmt.Sprintf("[tuner] shapes: %s", strings.Join(shapeStrs, ", ")))
	m.appendLog(fmt.Sprintf("[tuner] output dir: %s", m.TunedDir()))

	cmd := exec.CommandContext(ctx, python, args...)
	cmd.Env = append(os.Environ(),
		"PYTHONUNBUFFERED=1",
		// tqdm updates its bar with \r overwrites many times per second.
		// Throttle so we get a useful progress line in the UI without a
		// flood. Combined with the \r-aware scanner below, each surviving
		// update becomes its own log line.
		"TQDM_MININTERVAL=2",
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		finish(StateFailed, err.Error())
		return
	}
	cmd.Stderr = cmd.Stdout // merge

	if err := cmd.Start(); err != nil {
		m.appendLog(fmt.Sprintf("[tuner] start failed: %v", err))
		finish(StateFailed, err.Error())
		return
	}

	go m.pumpReader(stdout)

	err = cmd.Wait()
	if ctx.Err() == context.Canceled {
		m.appendLog("[tuner] cancelled")
		finish(StateCancelled, "cancelled")
		return
	}
	if err != nil {
		m.appendLog(fmt.Sprintf("[tuner] exited with error: %v", err))
		finish(StateFailed, err.Error())
		return
	}

	tuned := 0
	for _, s := range shapes {
		if m.IsTuned(s, blockN, blockK) {
			tuned++
		}
	}
	m.appendLog(fmt.Sprintf("[tuner] done — %d/%d shapes have JSONs on disk", tuned, len(shapes)))

	// Link the fresh results into vLLM's config dir so the next launch picks
	// them up, rather than making the operator restart the container to get
	// the boot-time install.
	if n, warn, err := InstallTunedConfigs(m.dataDir, m.DeviceName(), configsDir); err != nil {
		m.appendLog(fmt.Sprintf("[tuner] installing configs failed: %v", err))
	} else {
		m.appendLog(fmt.Sprintf("[tuner] installed %d config(s) into %s", n, configsDir))
		if warn != "" {
			m.appendLog("[tuner] " + warn)
		}
	}
	finish(StateCompleted, "")
}

func (m *Manager) pumpReader(r io.ReadCloser) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	scanner.Split(splitCROrLF)
	for scanner.Scan() {
		// tqdm repaints with colour and erase-line sequences; strip them so
		// the progress lines are readable in the UI.
		line := strings.TrimRight(ansi.Strip(scanner.Text()), " \t")
		if line == "" {
			continue
		}
		m.appendLog(line)
	}
}

// splitCROrLF is a bufio.SplitFunc that treats either '\n' or '\r' as a
// line boundary. tqdm (used by vLLM's tuner) updates progress bars in-place
// using bare '\r' overwrites; the default ScanLines accumulates the whole
// bar-update history into one giant line until '\n' arrives, which in our
// UI renders as a blank pane with horizontal-overflow whitespace. This
// split emits each bar update as its own line — combined with
// TQDM_MININTERVAL=2 on the subprocess env, the user sees a useful
// progress trickle.
func splitCROrLF(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	for i, b := range data {
		if b == '\n' || b == '\r' {
			return i + 1, data[:i], nil
		}
	}
	if atEOF {
		return len(data), data, nil
	}
	return 0, nil, nil
}

func newJobID() string {
	b := make([]byte, 6)
	rand.Read(b)
	return hex.EncodeToString(b)
}

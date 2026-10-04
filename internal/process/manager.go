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

	"github.com/tmac1973/vllm-toolchest/internal/advice"
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

	// Overdue is set while a start has outlasted the startup timeout and its
	// process is still alive. The state stays "starting": nothing has failed,
	// and a first start that compiles and tunes kernels is routinely slower
	// than any start after it.
	Overdue bool `json:"overdue,omitempty"`
	// StartFailed is set while the engine has said it could not start and
	// its process has not exited. It is the other thing a start that never
	// finishes can be, and it must not read as a slow one.
	StartFailed bool `json:"start_failed,omitempty"`
	// Notice is what to tell the operator about either of those.
	Notice string `json:"notice,omitempty"`

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
	// lifecycle serialises Start and Stop against each other. Without it a
	// Start could run while a Stop was still waiting for the old server to
	// exit, and the old server's exit was then recorded against the new one;
	// see waitForExit.
	lifecycle sync.Mutex

	mu      sync.RWMutex
	cmd     *exec.Cmd
	state   State
	modelID string
	// args is the flag list the running process was launched with; see Status.
	args []string
	pid  int
	// run numbers each launch, so that the goroutines watching one cannot
	// write state that belongs to the next.
	run        int
	startedAt  time.Time
	lastError  string
	cancelFunc context.CancelFunc
	vllmHost   string
	vllmPort   int
	launcher   Launcher
	// startupTimeout is how long a launch may take before it is reported as
	// overdue. The watch continues past it; see waitForReady.
	startupTimeout time.Duration
	// overdue and startFailed describe a start still in progress; see Status.
	overdue     bool
	startFailed bool
	// pollInterval is how often /health is checked. A field so tests do not
	// wait two seconds per state transition; nothing else sets it.
	pollInterval time.Duration
	// stopGrace is how long the server has to exit on SIGTERM, killWait how
	// long the group then has to go after SIGKILL, and portWait how long a
	// freshly reaped server's port gets to be released. Fields for the same
	// reason as pollInterval.
	stopGrace time.Duration
	killWait  time.Duration
	portWait  time.Duration

	// Log ring buffer
	logMu  sync.Mutex
	logBuf []string
	logMax int

	// Log subscribers
	subMu sync.Mutex
	subs  map[chan string]struct{}

	// What the engine said about this run, read off the log stream as it
	// arrives. Cleared on start: advice from the previous run describes a
	// configuration that may no longer be the one loaded.
	adviceMu sync.Mutex
	advice   []advice.Item
	measured advice.Measurements
}

// adviceMax caps the list. A pathological start can repeat a warning per layer,
// and the panel is not improved by the four hundredth copy.
const adviceMax = 64

// DefaultStartupTimeout is used when a caller passes nothing sensible. It is
// generous on purpose: passing it only marks a start overdue, and a label that
// says so about every first start of a large model tells nobody anything.
const DefaultStartupTimeout = 30 * time.Minute

func NewManager(vllmHost string, vllmPort int, startupTimeout time.Duration) *Manager {
	if startupTimeout <= 0 {
		startupTimeout = DefaultStartupTimeout
	}
	return &Manager{
		state:          StateStopped,
		vllmHost:       vllmHost,
		vllmPort:       vllmPort,
		launcher:       DefaultLauncher,
		startupTimeout: startupTimeout,
		pollInterval:   2 * time.Second,
		stopGrace:      30 * time.Second,
		killWait:       10 * time.Second,
		portWait:       5 * time.Second,
		logBuf:         make([]string, 0, 5000),
		logMax:         5000,
		subs:           make(map[chan string]struct{}),
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
	if m.state == StateStarting {
		switch {
		case m.startFailed:
			s.StartFailed = true
			s.Notice = "The engine failed to start but has not exited. See the log; Stop clears it."
		case m.overdue:
			s.Overdue = true
			s.Notice = fmt.Sprintf("Still starting after %s, past the startup timeout. Nothing has failed; the log shows progress.", m.startupTimeout)
		}
	}
	if m.lastError != "" {
		s.Error = m.lastError
	}
	return s
}

// Start launches vLLM with the given model path and config flags.
func (m *Manager) Start(modelID, modelPath string, args []string, env []string) error {
	m.lifecycle.Lock()
	defer m.lifecycle.Unlock()

	m.mu.Lock()
	if m.state != StateStopped && m.state != StateError {
		state := m.state
		m.mu.Unlock()
		return fmt.Errorf("vLLM is already running (state: %s)", state)
	}
	stale := m.pid
	m.mu.Unlock()

	if err := m.clearStale(stale); err != nil {
		m.mu.Lock()
		m.state = StateError
		m.lastError = err.Error()
		m.mu.Unlock()
		return err
	}

	m.mu.Lock()
	m.run++
	run := m.run
	m.state = StateStarting
	m.modelID = modelID
	m.args = append([]string(nil), args...)
	m.lastError = ""
	m.overdue = false
	m.startFailed = false
	m.startedAt = time.Now()
	m.mu.Unlock()

	// Clear log buffer, and with it what the last run's output said. Advice
	// describing a configuration that is no longer loaded is worse than none.
	m.logMu.Lock()
	m.logBuf = m.logBuf[:0]
	m.logMu.Unlock()
	m.resetAdvice()

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
	go m.streamOutput(stdout, run)
	go m.streamOutput(stderr, run)

	// Wait for process in background
	go m.waitForExit(cmd, cancel, run)

	// Poll for readiness
	go m.waitForReady(run)

	return nil
}

// clearStale makes sure nothing is left of a previous run before a new one
// binds the port: the recorded process group is reaped if any of it is still
// running, and the port must then be free.
//
// The recorded group is reaped rather than refused. It is provably ours, the
// state has already said it is not running, and it is holding the GPUs the
// new start needs -- refusing would only hand the operator the same kill to
// do by hand. A port held by anything else is refused: that is not ours to
// kill, and vLLM would only fail on it with "Address already in use" after
// spending a minute importing torch.
func (m *Manager) clearStale(pgid int) error {
	if groupAlive(pgid) {
		slog.Warn("a vLLM process group from a previous run is still running; reaping it before starting", "pgid", pgid)
		if err := terminateGroup(pgid, m.stopGrace, m.killWait); err != nil {
			return fmt.Errorf("a previous vLLM is still running and could not be stopped: %w", err)
		}
	}
	if !waitUntil(func() bool { return !portInUse(m.vllmPort) }, m.portWait) {
		return fmt.Errorf("port %d is already in use by a process vllmctl did not start "+
			"(or no longer tracks); stop it, or restart the container, before starting vLLM", m.vllmPort)
	}
	return nil
}

// Stop gracefully stops vLLM and everything it spawned.
//
// SIGTERM to the process group first: vLLM shuts down cleanly on it, releasing
// GPU memory and tearing down NCCL. Only if that does not finish in time does
// this escalate to SIGKILL. Either way the whole group is signalled, not just
// the process we launched -- see the comment in Start.
//
// It also stops a recorded group the state says is not running. That is the
// state being wrong, not the server being gone: on compute a server stayed up
// for fifteen minutes, holding the port and all four GPUs, under a status of
// "stopped", and Stop refusing on the strength of that status left no way to
// clear it short of restarting the container.
//
// Stop returns once nothing in the group is running. If something survives
// SIGKILL the state is error, not stopped, and the pid stays recorded so a
// later Stop or Start can try again.
func (m *Manager) Stop() error {
	m.lifecycle.Lock()
	defer m.lifecycle.Unlock()

	m.mu.Lock()
	state := m.state
	pgid := m.pid
	cancel := m.cancelFunc
	if state != StateRunning && state != StateStarting {
		if state == StateStopping || !groupAlive(pgid) {
			m.mu.Unlock()
			return fmt.Errorf("vLLM is not running (state: %s)", state)
		}
		slog.Warn("stopping a vLLM process group the status had lost track of", "pgid", pgid, "state", state)
	}
	m.state = StateStopping
	m.mu.Unlock()

	err := terminateGroup(pgid, m.stopGrace, m.killWait)
	if cancel != nil {
		cancel()
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		m.state = StateError
		m.lastError = err.Error()
		return err
	}
	m.state = StateStopped
	return nil
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
//
// Only a live server is stopped first. Start accepts a stopped or errored
// manager as it is, and reaps anything left of the last run itself.
func (m *Manager) Restart(modelID, modelPath string, args []string, env []string) error {
	if st := m.GetStatus().State; st == StateRunning || st == StateStarting {
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

func (m *Manager) streamOutput(r io.ReadCloser, run int) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 256*1024)
	for scanner.Scan() {
		// Strip before anything else looks at the line: vLLM and the radiance
		// banner colour their output, and an embedded escape would both
		// corrupt the display and break the readiness match below.
		line := ansi.Strip(scanner.Text())
		m.appendLog(line)
		m.observe(line)

		if advice.Ready(line) {
			m.mu.Lock()
			if m.run == run && m.state == StateStarting {
				m.state = StateRunning
			}
			m.mu.Unlock()
		}
		if advice.StartFailed(line) {
			m.mu.Lock()
			if m.run == run {
				m.startFailed = true
			}
			m.mu.Unlock()
		}
	}
}

// observe reads what the engine is telling us as it streams, rather than
// leaving it to be dredged out of the ring buffer afterwards.
//
// Live matters: on a start that fails, the advice is wanted at the moment the
// log panel becomes least readable, and on a start that succeeds the
// measurements are gone as soon as the buffer rolls over.
func (m *Manager) observe(line string) {
	item := advice.Scan(line)

	m.adviceMu.Lock()
	defer m.adviceMu.Unlock()

	advice.Observe(&m.measured, line)
	if item == nil {
		return
	}
	// A failing start prints the same complaint from every rank -- and the
	// ranks do not always agree on the number. Each one measures its own KV
	// pool, so a four-card start produced four notes differing in their last
	// few digits:
	//
	//	kv_cache_memory -> 4545302242
	//	kv_cache_memory -> 4532719330
	//	kv_cache_memory -> 4536913634
	//
	// The old key included Suggested, so those read as three separate pieces
	// of advice about one thing. Identity is the rule that fired -- severity,
	// field and message -- and the number is reconciled rather than repeated.
	//
	// The free-memory rule already solved this for itself by rounding to a
	// 0.05 step "so every rank agrees on one number"; kv_cache_memory cannot,
	// because it hands over an exact byte count the engine reported.
	for i, existing := range m.advice {
		if existing.Severity != item.Severity ||
			existing.Field != item.Field ||
			existing.Message != item.Message {
			continue
		}
		if smallerSuggestion(item.Suggested, existing.Suggested) {
			m.advice[i] = *item
		}
		return
	}
	if len(m.advice) < adviceMax {
		m.advice = append(m.advice, *item)
	}
}

// smallerSuggestion reports whether a is the more conservative of two values
// for the same piece of advice.
//
// Every applicable suggestion is a ceiling: the KV pool to pin, the context
// length that fits, the fraction that clears a shortfall. A figure measured
// per rank has to hold on the tightest rank, so the smallest is the one that
// is safe everywhere -- taking the largest would hand back a value that the
// most crowded card cannot honour.
//
// Anything non-numeric, or a missing value on either side, expresses no
// preference and the first one seen stands.
func smallerSuggestion(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	av, err := strconv.ParseFloat(a, 64)
	if err != nil {
		return false
	}
	bv, err := strconv.ParseFloat(b, 64)
	if err != nil {
		return false
	}
	return av < bv
}

// Advice returns what the engine has said worth acting on during this run.
func (m *Manager) Advice() []advice.Item {
	m.adviceMu.Lock()
	defer m.adviceMu.Unlock()
	out := make([]advice.Item, len(m.advice))
	copy(out, m.advice)
	return out
}

// Measured returns what the engine reported about this run: the KV pool it
// claimed, what the weights actually took, whether offload worked.
//
// These are the figures the VRAM estimator infers. Reported here so the panel
// can put the estimate beside the measurement rather than asking anyone to
// read a terminal.
func (m *Manager) Measured() advice.Measurements {
	m.adviceMu.Lock()
	defer m.adviceMu.Unlock()
	return m.measured
}

// resetAdvice clears both, so a new run is never described by the last one's
// output.
func (m *Manager) resetAdvice() {
	m.adviceMu.Lock()
	m.advice = nil
	m.measured = advice.Measurements{}
	m.adviceMu.Unlock()
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

// waitForExit records how a run ended -- if it is still the current run.
//
// That condition is how a live server came to be reported as stopped. On
// compute a Stop was sent and, while it waited for the old server to exit, a
// Start was accepted for the same model; Start refused only "running" and
// "starting", and the state was "stopping". The new server came up. Then the
// old one's Wait returned and this function, seeing a state that was no longer
// "stopping", wrote "stopped" over the new run. Stop saw that, swept the old
// group, and returned. What was left was a status of "stopped" naming the new
// server's pid while that server held the port and every GPU, a Stop that
// refused to touch it, and a next Start that died on "Address already in use".
//
// Start and Stop are now serialised, so that interleaving cannot recur, but a
// run's watcher still has no business writing another run's state.
func (m *Manager) waitForExit(cmd *exec.Cmd, cancel context.CancelFunc, run int) {
	err := cmd.Wait()
	cancel()

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.run != run {
		slog.Info("a replaced vLLM run exited", "pid", cmd.Process.Pid, "error", err)
		return
	}
	// Stop may already have finished: it waits on the processes, not on this.
	if m.state == StateStopping || m.state == StateStopped {
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

// waitForReady polls vLLM's /health until it answers, for as long as the
// process is alive.
//
// The startup deadline does not end the watch and does not fail the start. It
// marks the start overdue, which the status reports beside a state that is
// still "starting".
//
// It used to do more than that, twice. First it returned at the deadline, and
// a model that became healthy one second late stayed marked failed until
// someone restarted it -- which happened on a 125B MoE whose engine printed
// "Application startup complete" at 9m26s, inside the old 10-minute deadline,
// while the poll had already given up. Then it kept watching but set the state
// to error in the meantime, and that was wrong in a quieter way: an operator
// reading "Error" beside a first start that was simply slow would reasonably
// stop it. It also made the process unmanageable. Stop refuses a state of
// error, and Start accepts one, so a live engine past its deadline could not
// be stopped and could have a second one launched on top of it.
//
// The health GET carries its own timeout. With http.Get's default of none, a
// request issued just before the deadline could hang past it, and the loop
// would sit in that call rather than ever polling again.
func (m *Manager) waitForReady(run int) {
	healthURL := fmt.Sprintf("http://%s:%d/health", m.vllmHost, m.vllmPort)
	client := &http.Client{Timeout: 5 * time.Second}

	deadline := time.After(m.startupTimeout)
	ticker := time.NewTicker(m.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-deadline:
			m.mu.Lock()
			if m.run == run && m.state == StateStarting {
				m.overdue = true
				slog.Warn("vLLM is taking longer than the startup timeout; still polling",
					"model", m.modelID, "timeout", m.startupTimeout)
			}
			m.mu.Unlock()

		case <-ticker.C:
			m.mu.RLock()
			state, current := m.state, m.run == run
			m.mu.RUnlock()

			// Anything but starting means there is nothing left to wait for:
			// the process exited, is being stopped, or the log stream already
			// saw it come up. Nor is there once another run has replaced this
			// one; its own watcher is polling.
			if !current || state != StateStarting {
				return
			}

			resp, err := client.Get(healthURL)
			if err != nil {
				continue
			}
			resp.Body.Close()
			if resp.StatusCode != 200 {
				continue
			}

			m.mu.Lock()
			if m.run == run && m.state == StateStarting {
				if m.overdue {
					slog.Info("vLLM came up after the startup timeout",
						"model", m.modelID, "after", time.Since(m.startedAt).Truncate(time.Second))
				} else {
					slog.Info("vLLM is ready", "model", m.modelID)
				}
				m.state = StateRunning
				m.lastError = ""
			}
			m.mu.Unlock()
			return
		}
	}
}

// BuildArgs constructs vLLM CLI arguments from a model config.
func BuildArgs(cfg VLLMStartConfig) []string {
	var args []string

	// Without this vLLM names the model by the path it was loaded from, so
	// /v1/models answers "/data/models/owner/repo" and a client has to send
	// that container-local path as its model id. Naming it explicitly makes
	// the served name the same HuggingFace repo id the rest of the tool uses.
	if cfg.ServedModelName != "" {
		args = append(args, "--served-model-name", cfg.ServedModelName)
	}
	if cfg.Dtype != "" && cfg.Dtype != "auto" {
		args = append(args, "--dtype", cfg.Dtype)
	}
	if cfg.MaxModelLen > 0 {
		args = append(args, "--max-model-len", strconv.Itoa(cfg.MaxModelLen))
	}
	if cfg.TensorParallelSize > 1 {
		args = append(args, "--tensor-parallel-size", strconv.Itoa(cfg.TensorParallelSize))
	}
	// Always passed, never left to the engine's default: the default is not
	// 0.90 everywhere. vLLM 0.29 on compute defaults to 0.92, so every model
	// configured at 0.90 ran at 0.92 while the UI, the planner and the
	// measurements all took it for 0.90.
	if cfg.GPUMemoryUtilization > 0 {
		args = append(args, "--gpu-memory-utilization", fmt.Sprintf("%.2f", cfg.GPUMemoryUtilization))
	}
	if cfg.EnforceEager {
		args = append(args, "--enforce-eager")
	}
	if cfg.TrustRemoteCode {
		args = append(args, "--trust-remote-code")
	}
	// Always passed. This used to be left out at 16 on the grounds that 16
	// was vLLM's default, which it is not: left unset, the OpenAI server picks
	// 256 on a card under 70 GB and 1024 above that. So the one value the
	// tool itself defaults a new model to was the one value it never applied,
	// and a model saved at 16 ran sixteen times wider than its panel said.
	if cfg.MaxNumSeqs > 0 {
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
	// ServedModelName is what vLLM will call this model in /v1/models and
	// what clients pass in a request's "model" field. Empty leaves vLLM's
	// own default, which is the model path.
	ServedModelName        string
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

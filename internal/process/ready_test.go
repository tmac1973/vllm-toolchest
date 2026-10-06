package process

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// readyHarness is a Manager pointed at a health endpoint the test controls,
// with a real live child process standing in for the engine.
func readyHarness(t *testing.T, healthy *atomic.Bool, timeout time.Duration) *Manager {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" && healthy.Load() {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	host, portStr, ok := strings.Cut(strings.TrimPrefix(srv.URL, "http://"), ":")
	if !ok {
		t.Fatalf("unexpected test server URL %q", srv.URL)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}

	m := NewManager(host, port, timeout)
	m.pollInterval = 20 * time.Millisecond

	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start a helper process: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})

	m.mu.Lock()
	m.cmd = cmd
	m.state = StateStarting
	m.modelID = "org/model"
	m.startedAt = time.Now()
	m.mu.Unlock()

	return m
}

// status is a failure-message argument that reads the manager's state when
// the message is printed, so a wait that gives up reports where it stopped.
type status struct{ m *Manager }

func (s status) String() string {
	st := s.m.GetStatus()
	return fmt.Sprintf("state %s, error %q, notice %q", st.State, st.Error, st.Notice)
}

func becomes(m *Manager, want State) func() bool {
	return func() bool { return m.GetStatus().State == want }
}

func isOverdue(m *Manager) func() bool {
	return func() bool { return m.GetStatus().Overdue }
}

// The case this was written for. A 125B MoE printed "Application startup
// complete" at 9m26s while the poll had already given up, so a serving engine
// was reported as failed until someone restarted it.
func TestAServerThatComesUpLateIsNotLeftMarkedFailed(t *testing.T) {
	var healthy atomic.Bool
	m := readyHarness(t, &healthy, 60*time.Millisecond)

	go m.waitForReady(m.run)
	testutil.Eventually(t, 2*time.Second, isOverdue(m), "the start was never marked overdue: %v", status{m})

	// The engine finishes loading well after the deadline.
	healthy.Store(true)

	testutil.Eventually(t, 2*time.Second, becomes(m, StateRunning), "never running: %v", status{m})
	st := m.GetStatus()
	if st.Error != "" || st.Notice != "" || st.Overdue {
		t.Errorf("a late start that came good still carries error %q, notice %q, overdue %v",
			st.Error, st.Notice, st.Overdue)
	}
}

// Past the deadline with the process alive is a slow start, not a failed one.
// It was reported as "error" while the watch went on, and an operator reading
// that beside a first start would reasonably stop it.
func TestASlowStartIsNotReportedAsAnError(t *testing.T) {
	var healthy atomic.Bool
	m := readyHarness(t, &healthy, 60*time.Millisecond)

	go m.waitForReady(m.run)
	testutil.Eventually(t, 2*time.Second, isOverdue(m), "the start was never marked overdue: %v", status{m})

	st := m.GetStatus()
	if st.State != StateStarting {
		t.Errorf("state = %s past the startup timeout, want it still %s", st.State, StateStarting)
	}
	if st.Error != "" {
		t.Errorf("error = %q for a start that has not failed", st.Error)
	}
	for _, want := range []string{"Still starting", "Nothing has failed"} {
		if !strings.Contains(st.Notice, want) {
			t.Errorf("notice = %q, want it to say %q", st.Notice, want)
		}
	}
	if st.Uptime == "" {
		t.Error("no elapsed time reported for a start still in progress")
	}
}

// The error state was also unmanageable while the process lived: Stop refuses
// it and Start accepts it, so a slow engine could not be stopped and could
// have a second one launched on top of it, onto GPUs it was still holding.
func TestAnOverdueStartCannotBeStartedOver(t *testing.T) {
	var healthy atomic.Bool
	m := readyHarness(t, &healthy, 60*time.Millisecond)

	go m.waitForReady(m.run)
	testutil.Eventually(t, 2*time.Second, isOverdue(m), "the start was never marked overdue: %v", status{m})

	err := m.Start("org/other", "/nowhere", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "already running") {
		t.Errorf("Start during an overdue start returned %v, want a refusal", err)
	}
	if got := m.GetStatus().ModelID; got != "org/model" {
		t.Errorf("model = %q, the overdue start was replaced", got)
	}
}

// The other start that never finishes. The engine says it failed and the
// process does not exit -- seen on a checkpoint newer than its image, where
// the API server raised and then sat there. That must not read as slow.
func TestAStartTheEngineGaveUpOnIsNotCalledSlow(t *testing.T) {
	var healthy atomic.Bool
	m := readyHarness(t, &healthy, 60*time.Millisecond)

	m.streamOutput(io.NopCloser(strings.NewReader(
		"(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above.\n")), m.run)

	st := m.GetStatus()
	if !st.StartFailed || !strings.Contains(st.Notice, "failed to start") {
		t.Errorf("start_failed = %v, notice = %q, want the failure reported", st.StartFailed, st.Notice)
	}

	// Still so once the deadline passes: the failure outranks the lateness.
	go m.waitForReady(m.run)
	testutil.Eventually(t, 2*time.Second, func() bool {
		m.mu.RLock()
		defer m.mu.RUnlock()
		return m.overdue
	}, "the startup deadline never passed")
	st = m.GetStatus()
	if !st.StartFailed || st.Overdue || strings.Contains(st.Notice, "Nothing has failed") {
		t.Errorf("a failed start reads as a slow one: %+v", st)
	}
}

func TestReadyBeforeTheDeadlineNeverReportsAnError(t *testing.T) {
	var healthy atomic.Bool
	healthy.Store(true)
	m := readyHarness(t, &healthy, 10*time.Second)

	go m.waitForReady(m.run)

	testutil.Eventually(t, 2*time.Second, becomes(m, StateRunning), "never running: %v", status{m})
	if e := m.GetStatus().Error; e != "" {
		t.Errorf("error text = %q, want none", e)
	}
}

// A process that has exited is never coming up, so the watch must end rather
// than poll a dead engine forever.
func TestTheWatchStopsWhenTheProcessIsGone(t *testing.T) {
	var healthy atomic.Bool
	m := readyHarness(t, &healthy, time.Hour)

	done := make(chan struct{})
	go func() { m.waitForReady(m.run); close(done) }()

	// waitForExit would do this for real; set it directly so the test does not
	// depend on process teardown timing.
	m.mu.Lock()
	m.state = StateStopped
	m.mu.Unlock()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("waitForReady kept polling after the process stopped")
	}
}

// The timeout is configuration now, not a constant buried in the poll. Zero
// means the default rather than an instant timeout, which is what a config
// file with the field absent would otherwise produce.
func TestStartupTimeoutComesFromTheCaller(t *testing.T) {
	if got := NewManager("127.0.0.1", 0, 45*time.Minute).startupTimeout; got != 45*time.Minute {
		t.Errorf("startupTimeout = %s, want 45m", got)
	}
	if got := NewManager("127.0.0.1", 0, 0).startupTimeout; got != DefaultStartupTimeout {
		t.Errorf("startupTimeout = %s with none given, want the default %s", got, DefaultStartupTimeout)
	}
}

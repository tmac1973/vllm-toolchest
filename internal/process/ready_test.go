package process

import (
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// readyHarness is a Manager pointed at a health endpoint the test controls,
// with a real live child process so processAlive() means something.
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

func waitForState(t *testing.T, m *Manager, want State, within time.Duration) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if got := m.GetStatus().State; got == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("state = %s after %s, want %s (error: %q)",
		m.GetStatus().State, within, want, m.GetStatus().Error)
}

// The case this was written for. A 125B MoE printed "Application startup
// complete" at 9m26s while the poll had already given up, so a serving engine
// was reported as failed until someone restarted it.
func TestAServerThatComesUpLateIsNotLeftMarkedFailed(t *testing.T) {
	var healthy atomic.Bool
	m := readyHarness(t, &healthy, 60*time.Millisecond)

	go m.waitForReady()

	waitForState(t, m, StateError, 2*time.Second)
	if e := m.GetStatus().Error; !strings.Contains(e, "still watching") {
		t.Errorf("error text = %q, want it to say the watch continues", e)
	}

	// The engine finishes loading well after the deadline.
	healthy.Store(true)

	waitForState(t, m, StateRunning, 2*time.Second)
	if e := m.GetStatus().Error; e != "" {
		t.Errorf("error text = %q, want it cleared once healthy", e)
	}
}

func TestReadyBeforeTheDeadlineNeverReportsAnError(t *testing.T) {
	var healthy atomic.Bool
	healthy.Store(true)
	m := readyHarness(t, &healthy, 10*time.Second)

	go m.waitForReady()

	waitForState(t, m, StateRunning, 2*time.Second)
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
	go func() { m.waitForReady(); close(done) }()

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

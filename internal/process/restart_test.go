package process

import (
	"reflect"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// fakeEngine returns a manager whose engine is script, with stop timings
// short enough for a test. The engine is killed when the test ends.
func fakeEngine(t *testing.T, script string) *Manager {
	t.Helper()
	m := NewManager("127.0.0.1", 0, 0)
	m.SetLauncher(Launcher{Bin: testutil.WriteScript(t, script)})
	m.stopGrace = 5 * time.Second
	m.killWait = 2 * time.Second
	m.portWait = 200 * time.Millisecond
	t.Cleanup(func() {
		if pid := m.GetStatus().PID; pid > 0 {
			_ = killProcessGroup(pid, syscall.SIGKILL)
		}
	})
	return m
}

// An engine that announces its arguments and then runs until signalled, as
// vLLM does once it is up.
const echoAndWait = `echo "args: $*"
sleep 120 &
wait
`

func logged(m *Manager, want string) bool {
	return slices.ContainsFunc(m.RecentLogs(0), func(l string) bool { return strings.Contains(l, want) })
}

// Restart is how a changed config is applied to a live engine: the old process
// tree must be gone -- it holds the port and the GPUs -- and a new one running
// with the new arguments.
func TestRestartStopsALiveEngineAndStartsANewOne(t *testing.T) {
	m := fakeEngine(t, echoAndWait)
	if err := m.Start("org/model", "/models/org/model", []string{"--max-model-len", "8192"}, nil); err != nil {
		t.Fatal(err)
	}
	old := m.GetStatus().PID
	testutil.Eventually(t, 5*time.Second, func() bool { return logged(m, "--max-model-len 8192") },
		"first engine never printed its arguments")

	if err := m.Restart("org/model", "/models/org/model", []string{"--max-model-len", "32768"}, nil); err != nil {
		t.Fatalf("restart: %v", err)
	}

	st := m.GetStatus()
	if st.State != StateStarting || st.PID == old || st.PID <= 0 {
		t.Fatalf("after restart: state %s pid %d (old pid %d)", st.State, st.PID, old)
	}
	if !reflect.DeepEqual(st.Args, []string{"--max-model-len", "32768"}) {
		t.Errorf("status args = %v, want the new ones", st.Args)
	}
	if processAlive(old) {
		t.Errorf("old engine %d survived the restart", old)
	}
	testutil.Eventually(t, 5*time.Second, func() bool { return logged(m, "--max-model-len 32768") },
		"new engine never printed its arguments")
	// The log is the new run's. The old run's lines describe a config that
	// is no longer loaded.
	if logged(m, "--max-model-len 8192") {
		t.Error("the old engine's output is still in the log after restart")
	}
}

// With nothing running there is nothing to stop. Restart must start rather
// than fail on Stop's "not running".
func TestRestartWhenStoppedJustStarts(t *testing.T) {
	m := fakeEngine(t, echoAndWait)
	if err := m.Restart("org/model", "", nil, nil); err != nil {
		t.Fatalf("restart from stopped: %v", err)
	}
	if st := m.GetStatus(); st.State != StateStarting || !processAlive(st.PID) {
		t.Errorf("state %s pid %d, want a live starting engine", st.State, st.PID)
	}
}

// An engine that crashed leaves the manager in error. Restart is the obvious
// button to press then, and it must start again rather than refuse.
func TestRestartAfterACrashStartsAgain(t *testing.T) {
	m := fakeEngine(t, `if [ -e "$0.ran" ]; then sleep 120 & wait; fi
touch "$0.ran"
echo boom >&2
exit 1
`)
	if err := m.Start("org/model", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	testutil.Eventually(t, 5*time.Second, func() bool { return m.GetStatus().State == StateError },
		"engine never failed; state %s", m.GetStatus().State)

	if err := m.Restart("org/model", "", nil, nil); err != nil {
		t.Fatalf("restart after crash: %v", err)
	}
	if st := m.GetStatus(); st.State != StateStarting || st.Error != "" {
		t.Errorf("state %s error %q, want starting with the old error cleared", st.State, st.Error)
	}
}

func TestRecentLogsClampsNToWhatIsBuffered(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	for _, l := range []string{"one", "two", "three"} {
		m.appendLog(l)
	}
	cases := []struct {
		n    int
		want []string
	}{
		{2, []string{"two", "three"}},
		{3, []string{"one", "two", "three"}},
		// More than there are, zero, and negative all mean everything.
		// The log endpoint passes a fixed 5000 against a buffer that may
		// hold three lines.
		{5000, []string{"one", "two", "three"}},
		{0, []string{"one", "two", "three"}},
		{-1, []string{"one", "two", "three"}},
	}
	for _, c := range cases {
		if got := m.RecentLogs(c.n); !reflect.DeepEqual(got, c.want) {
			t.Errorf("RecentLogs(%d) = %v, want %v", c.n, got, c.want)
		}
	}

	// The result is a copy: a caller appending to it must not write into
	// the buffer.
	got := m.RecentLogs(1)
	got[0] = "changed"
	if m.RecentLogs(1)[0] != "three" {
		t.Error("RecentLogs returned the buffer itself")
	}
}

func TestRecentLogsOfAnEmptyBufferIsEmpty(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	for _, n := range []int{-1, 0, 1, 10} {
		if got := m.RecentLogs(n); len(got) != 0 {
			t.Errorf("RecentLogs(%d) = %v on an empty buffer", n, got)
		}
	}
}

// The log panel's Clear button empties the server's buffer too, or a reload
// brings every cleared line back. Lines after the clear are kept.
func TestClearLogsEmptiesTheBufferAndKeepsLaterLines(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	m.appendLog("before")
	m.ClearLogs()
	if got := m.RecentLogs(0); len(got) != 0 {
		t.Fatalf("after ClearLogs: %v", got)
	}
	m.appendLog("after")
	if got := m.RecentLogs(0); !reflect.DeepEqual(got, []string{"after"}) {
		t.Errorf("got %v, want only the line appended after clearing", got)
	}
}

// Clearing the buffer must not cut off a live log stream: subscribers keep
// receiving lines.
func TestClearLogsLeavesSubscribersConnected(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	ch := m.SubscribeLogs()
	defer m.UnsubscribeLogs(ch)
	m.ClearLogs()
	m.appendLog("still streaming")
	select {
	case got := <-ch:
		if got != "still streaming" {
			t.Errorf("got %q", got)
		}
	case <-time.After(2 * time.Second):
		t.Error("subscriber received nothing after ClearLogs")
	}
}

package process

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// vLLM is a process tree, not a process: an EngineCore and one Worker per rank
// hang off the launched command. Killing only the leader leaves those holding
// their HIP contexts and the GPU memory with them, which is what made a later
// start fail with "Free memory on device cuda:0 ... less than desired".
//
// This stands in a shell tree of the same shape for that, and asserts the
// grandchildren are gone once Stop returns.
func TestStopKillsTheWholeProcessTree(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "grandchild.pid")

	// leader -> child -> grandchild, the grandchild recording its own pid and
	// then sleeping. It ignores SIGTERM so that only a group-wide signal, not
	// a polite one to the leader, can end it.
	script := fmt.Sprintf(`
child() {
    trap '' TERM
    echo $$ > %q
    sleep 120
}
child &
sleep 120
`, marker)

	fake := testutil.WriteScript(t, script)

	m := NewManager("127.0.0.1", 0, 0)
	m.SetLauncher(Launcher{Bin: fake})
	if err := m.Start("test", "", nil, nil); err != nil {
		t.Fatalf("start: %v", err)
	}

	// Wait for the grandchild to announce itself.
	var gpid int
	testutil.Eventually(t, 5*time.Second, func() bool {
		b, _ := os.ReadFile(marker)
		gpid, _ = strconv.Atoi(strings.TrimSpace(string(b)))
		return gpid > 0
	}, "grandchild never started")
	// It ignores SIGTERM; if Stop fails to end it, nothing else will.
	t.Cleanup(func() { _ = syscall.Kill(gpid, syscall.SIGKILL) })
	if !processAlive(gpid) {
		t.Fatalf("grandchild %d should be alive before Stop", gpid)
	}

	if err := m.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}

	// Give the signal a moment to land.
	testutil.Eventually(t, 2*time.Second, func() bool { return !processAlive(gpid) },
		"grandchild %d survived Stop — it would still hold its GPU context", gpid)
}

func TestStopOnStoppedManagerErrors(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	if err := m.Stop(); err == nil {
		t.Error("expected an error stopping a manager that is not running")
	}
}

func TestKillProcessGroupIgnoresNonPositivePID(t *testing.T) {
	if err := killProcessGroup(0, 9); err != nil {
		t.Errorf("pid 0 must be a no-op, got %v", err)
	}
	if err := killProcessGroup(-1, 9); err != nil {
		t.Errorf("negative pid must be a no-op, got %v", err)
	}
}

// fakeServer writes script as an executable and returns a manager that
// launches it, with timings short enough for a test.
func fakeServer(t *testing.T, script string) *Manager {
	t.Helper()
	fake := testutil.WriteScript(t, script)
	m := NewManager("127.0.0.1", 0, 0)
	m.SetLauncher(Launcher{Bin: fake})
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

func startedPID(t *testing.T, m *Manager) int {
	t.Helper()
	pid := m.GetStatus().PID
	if pid <= 0 || !processAlive(pid) {
		t.Fatalf("expected a live server after Start, got pid %d", pid)
	}
	return pid
}

// The incident on compute, reproduced: Stop sent, and while it waits for the
// old server to exit, Start sent for the same model. The old server's exit
// used to be recorded against the new one, leaving the new server running --
// holding the port and the GPUs -- under a status of "stopped".
func TestStartDuringStopDoesNotLoseTheNewServer(t *testing.T) {
	// Takes a second to exit on SIGTERM, and exits 0, as vLLM does.
	trapped := filepath.Join(t.TempDir(), "trapped")
	m := fakeServer(t, fmt.Sprintf(`
trap 'sleep 1; exit 0' TERM
touch %q
sleep 120 &
wait
`, trapped))
	if err := m.Start("model", "", nil, nil); err != nil {
		t.Fatalf("first start: %v", err)
	}
	first := startedPID(t, m)

	// Start returns once the shell is spawned, not once it has run its first
	// line. A SIGTERM sent before the trap is set ends it at once, Stop is
	// over before the test can see it stopping, and the overlap this test is
	// about never happens.
	testutil.Eventually(t, 5*time.Second, func() bool {
		_, err := os.Stat(trapped)
		return err == nil
	}, "the fake server never set its SIGTERM trap")

	stopped := make(chan error, 1)
	go func() { stopped <- m.Stop() }()
	testutil.Eventually(t, 2*time.Second, func() bool { return m.GetStatus().State == StateStopping },
		"Stop never reached stopping")

	if err := m.Start("model", "", nil, nil); err != nil {
		t.Fatalf("start during stop: %v", err)
	}
	if err := <-stopped; err != nil {
		t.Fatalf("stop: %v", err)
	}
	second := startedPID(t, m)
	if second == first {
		t.Fatalf("second start reports the first server's pid %d", first)
	}
	if processAlive(first) {
		t.Errorf("first server %d is still alive", first)
	}

	// Give a stale watcher every chance to write over the new run. There is
	// nothing to wait for: the bug is a write that should never come. Both of
	// the first run's watchers have to have had their turn -- its exit
	// watcher, and its health poll, which ticks every 2s from the first Start.
	// The SIGTERM trap took a second, so 1.5s more puts that tick well behind.
	time.Sleep(1500 * time.Millisecond)
	if st := m.GetStatus(); st.State != StateStarting || st.PID != second {
		t.Errorf("status = %s pid %d, want starting pid %d", st.State, st.PID, second)
	}
	if err := m.Stop(); err != nil {
		t.Fatalf("stopping the second server: %v", err)
	}
	if processAlive(second) {
		t.Errorf("second server %d survived Stop", second)
	}
}

// The state can be wrong about the server being gone. When it is, Stop has to
// be able to clear what is actually running rather than refuse on the word of
// the status.
func TestStopReapsAGroupTheStatusLostTrackOf(t *testing.T) {
	m := fakeServer(t, "sleep 120\n")
	if err := m.Start("model", "", nil, nil); err != nil {
		t.Fatalf("start: %v", err)
	}
	pid := startedPID(t, m)

	// What compute showed: a live server under a status of "stopped".
	m.mu.Lock()
	m.state = StateStopped
	m.mu.Unlock()

	if err := m.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if groupAlive(pid) {
		t.Fatalf("group %d survived Stop", pid)
	}
	if st := m.GetStatus().State; st != StateStopped {
		t.Errorf("state after reaping = %s, want stopped", st)
	}

	// Nothing left: now the refusal is right.
	if err := m.Stop(); err == nil {
		t.Error("expected an error stopping a manager with nothing running")
	}
}

// Start reaps a recorded group that is still running before launching
// another server on top of it.
func TestStartReapsALeftoverGroup(t *testing.T) {
	m := fakeServer(t, "sleep 120\n")
	if err := m.Start("model", "", nil, nil); err != nil {
		t.Fatalf("start: %v", err)
	}
	first := startedPID(t, m)

	m.mu.Lock()
	m.state = StateStopped
	m.mu.Unlock()

	if err := m.Start("other", "", nil, nil); err != nil {
		t.Fatalf("start over a leftover group: %v", err)
	}
	if groupAlive(first) {
		t.Errorf("leftover group %d survived the next Start", first)
	}
	if second := startedPID(t, m); second == first {
		t.Errorf("status still names the leftover server %d", first)
	}
}

// A port held by something that is not ours is refused, before vLLM is
// spawned to fail on it.
func TestStartRefusesAPortHeldByAStranger(t *testing.T) {
	l, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	dir := t.TempDir()
	marker := filepath.Join(dir, "launched")
	m := fakeServer(t, fmt.Sprintf("touch %q\nsleep 120\n", marker))
	m.vllmPort = l.Addr().(*net.TCPAddr).Port

	err = m.Start("model", "", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "already in use") {
		t.Fatalf("Start = %v, want the port refused", err)
	}
	if st := m.GetStatus(); st.State != StateError || !strings.Contains(st.Error, "already in use") {
		t.Errorf("status = %s %q, want error naming the port", st.State, st.Error)
	}
	// Start returned without launching, so there is no event to wait for;
	// give a server that was spawned regardless the moment it needs to show
	// itself by touching the marker.
	time.Sleep(100 * time.Millisecond)
	if _, err := os.Stat(marker); err == nil {
		t.Error("the server was launched anyway")
	}
}

// vllmctl is PID 1 in its container, so orphaned workers are reparented to it
// and, never waited on, stay zombies. A group whose only members are zombies
// is gone for every purpose that matters here; kill -0 says otherwise.
func TestGroupAliveIgnoresZombies(t *testing.T) {
	cmd := exec.Command("sh", "-c", "exit 0")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid
	defer cmd.Wait()

	// Not waited on yet, so it lingers as a zombie in its own group.
	if !waitUntil(func() bool {
		st, err := readProcStat(strconv.Itoa(pid))
		return err == nil && st.state == 'Z'
	}, 2*time.Second) {
		t.Skip("child did not become a zombie we could observe")
	}
	if syscall.Kill(-pid, 0) != nil {
		t.Fatal("expected kill -0 to still see the zombie's group")
	}
	if groupAlive(pid) {
		t.Error("groupAlive counted a group of zombies as running")
	}
	if processAlive(pid) {
		t.Error("processAlive counted a zombie as running")
	}
}

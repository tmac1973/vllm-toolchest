package process

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
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

	fake := filepath.Join(dir, "fakevllm")
	if err := os.WriteFile(fake, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatal(err)
	}

	m := NewManager("127.0.0.1", 0)
	m.SetLauncher(Launcher{Bin: fake})
	if err := m.Start("test", "", nil, nil); err != nil {
		t.Fatalf("start: %v", err)
	}

	// Wait for the grandchild to announce itself.
	var gpid int
	for i := 0; i < 100; i++ {
		if b, err := os.ReadFile(marker); err == nil {
			if gpid, _ = strconv.Atoi(strings.TrimSpace(string(b))); gpid > 0 {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	if gpid == 0 {
		t.Fatal("grandchild never started")
	}
	if !processAlive(gpid) {
		t.Fatalf("grandchild %d should be alive before Stop", gpid)
	}

	if err := m.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}

	// Give the signal a moment to land.
	for i := 0; i < 40 && processAlive(gpid); i++ {
		time.Sleep(50 * time.Millisecond)
	}
	if processAlive(gpid) {
		_ = killProcessGroup(gpid, 9) // don't leak it out of the test
		t.Errorf("grandchild %d survived Stop — it would still hold its GPU context", gpid)
	}
}

func TestStopOnStoppedManagerErrors(t *testing.T) {
	m := NewManager("127.0.0.1", 0)
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

func processAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(os.Signal(nil)) == nil || signalZero(pid) == nil
}

func signalZero(pid int) error {
	out, err := exec.Command("kill", "-0", strconv.Itoa(pid)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v: %s", err, out)
	}
	return nil
}

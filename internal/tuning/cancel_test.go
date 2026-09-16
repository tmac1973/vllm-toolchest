package tuning

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// The tuner is a process tree, not a process: python spawns ROCm workers that
// hold the GPU. Cancelling with exec.CommandContext alone kills the direct
// child and leaves those workers running, still holding their contexts.
//
// That is not theoretical. A cancelled tuning job on 2026-09-16 left 1.2 GiB
// on one card and 1.0 on another, and an unrelated model then would not launch:
// vLLM compares gpu_memory_utilization against *free* memory and refused with
// "Free memory on device cuda:0 (27.28/31.86 GiB) ... less than desired". The
// UI said the job was cancelled and nothing said the GPUs were still occupied.
//
// This stands a shell tree of the same shape in for it, and asserts the
// grandchild is gone once the job reports cancelled.
func TestCancelKillsTheWholeProcessTree(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "grandchild.pid")

	// leader -> child, the child recording its own pid and then sleeping. It
	// ignores SIGTERM, so only a group-wide signal can end it -- a polite one
	// aimed at the leader will not.
	script := fmt.Sprintf(`
child() {
    trap '' TERM
    echo $$ > %q
    sleep 120
}
child &
sleep 120
`, marker)

	fake := filepath.Join(dir, "faketuner")
	if err := os.WriteFile(fake, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatal(err)
	}

	m := NewManager(dir, "gfx1201", "tuner.py", nil)
	m.SetPython(fake)

	if _, err := m.StartJob("test/model", []Shape{{N: 128, K: 128}}, 1, 128, 128); err != nil {
		t.Fatalf("start job: %v", err)
	}

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
		t.Skip("the fake tuner never started a child; nothing to assert about")
	}
	if !processAlive(gpid) {
		t.Fatalf("child %d should be alive before Cancel", gpid)
	}

	m.Cancel()

	for i := 0; i < 60 && processAlive(gpid); i++ {
		time.Sleep(50 * time.Millisecond)
	}
	if processAlive(gpid) {
		_ = syscall.Kill(-gpid, syscall.SIGKILL) // don't leak it out of the test
		t.Errorf("child %d survived Cancel -- it would still be holding GPU memory", gpid)
	}
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	// Signal 0 tests for existence without delivering anything.
	return syscall.Kill(pid, 0) == nil
}

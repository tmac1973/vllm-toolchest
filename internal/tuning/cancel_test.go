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

	"github.com/tmac1973/vllm-toolchest/internal/testutil"
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

	fake := testutil.WriteScript(t, script)

	m := NewManager(dir, "gfx1201", "tuner.py", nil)
	m.SetPython(fake)

	if _, err := m.StartJob("test/model", []Shape{{N: 128, K: 128}}, 1, 128, 128); err != nil {
		t.Fatalf("start job: %v", err)
	}

	var gpid int
	testutil.Eventually(t, 5*time.Second, func() bool {
		b, _ := os.ReadFile(marker)
		gpid, _ = strconv.Atoi(strings.TrimSpace(string(b)))
		return gpid > 0
	}, "the fake tuner never started its child")
	// It ignores SIGTERM; if Cancel fails to end it, nothing else will.
	t.Cleanup(func() { _ = syscall.Kill(gpid, syscall.SIGKILL) })
	if !processAlive(gpid) {
		t.Fatalf("child %d should be alive before Cancel", gpid)
	}

	m.Cancel()

	testutil.Eventually(t, 3*time.Second, func() bool { return !processAlive(gpid) },
		"child %d survived Cancel -- it would still be holding GPU memory", gpid)
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	// Signal 0 tests for existence without delivering anything.
	return syscall.Kill(pid, 0) == nil
}

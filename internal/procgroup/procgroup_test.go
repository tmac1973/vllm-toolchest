package procgroup

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Cancelling the command must reach a grandchild, not only the shell that
// started it, and must not leave Wait blocked on the pipe the grandchild
// holds open.
func TestCancelKillsTheWholeGroup(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "grandchild.pid")
	ctx, cancel := context.WithCancel(context.Background())

	// The shell backgrounds a sleeper that inherits stdout, records its pid,
	// then waits on it.
	cmd := Command(ctx, syscall.SIGKILL, 2*time.Second, "sh", "-c",
		"sleep 60 & echo $! > "+pidFile+"; wait")
	var out strings.Builder
	cmd.Stdout = &out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	var grandchild int
	deadline := time.Now().Add(5 * time.Second)
	for grandchild == 0 && time.Now().Before(deadline) {
		if b, err := os.ReadFile(pidFile); err == nil {
			grandchild, _ = strconv.Atoi(strings.TrimSpace(string(b)))
		}
		if grandchild == 0 {
			time.Sleep(20 * time.Millisecond)
		}
	}
	if grandchild == 0 {
		t.Fatal("grandchild never recorded its pid")
	}

	cancel()
	waited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(waited) }()
	select {
	case <-waited:
	case <-time.After(5 * time.Second):
		t.Fatal("Wait still blocked after cancel")
	}

	// Signal 0 probes for existence. The sleeper was a child of the shell,
	// which is gone, so init reaps it; allow a moment for that.
	deadline = time.Now().Add(2 * time.Second)
	for syscall.Kill(grandchild, 0) == nil {
		if time.Now().After(deadline) {
			t.Fatalf("grandchild %d survived the cancel", grandchild)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestKillBeforeStartIsANoOp(t *testing.T) {
	cmd := Command(context.Background(), syscall.SIGKILL, time.Second, "true")
	if err := Kill(cmd, syscall.SIGKILL); err != nil {
		t.Fatalf("Kill on an unstarted command: %v", err)
	}
}

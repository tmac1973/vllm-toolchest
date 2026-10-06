package testutil

import (
	"os/exec"
	"testing"
	"time"
)

func TestWriteScriptIsRunnable(t *testing.T) {
	out, err := exec.Command(WriteScript(t, "echo hi\n")).Output()
	if err != nil || string(out) != "hi\n" {
		t.Fatalf("got %q, %v", out, err)
	}
}

func TestEventuallyReturnsOnceTheConditionHolds(t *testing.T) {
	start := time.Now()
	n := 0
	Eventually(t, 5*time.Second, func() bool { n++; return n == 3 }, "n never reached 3")
	if time.Since(start) > time.Second {
		t.Errorf("took %s", time.Since(start))
	}
}

package process

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// An engine that prints and dies at once must still have every line read
// by the time its exit is recorded. Reading a StdoutPipe while Wait ran used
// to drop the tail, which is where a failing start prints its error.
func TestAnEngineThatDiesAtOnceKeepsItsOutput(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "fakevllm")
	script := "#!/bin/sh\nfor i in $(seq 1 200); do echo line $i; done\necho LAST LINE >&2\nexit 1\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	for attempt := 0; attempt < 20; attempt++ {
		m := NewManager("127.0.0.1", 0, 0)
		m.SetLauncher(Launcher{Bin: fake})
		if err := m.Start("org/model", dir, nil, nil); err != nil {
			t.Fatal(err)
		}
		waitForState(t, m, StateError, 5*time.Second)

		logs := m.RecentLogs(1000)
		var sawLast, sawTail bool
		for _, l := range logs {
			sawLast = sawLast || l == "LAST LINE"
			sawTail = sawTail || l == "line 200"
		}
		if !sawLast || !sawTail {
			t.Fatalf("attempt %d: exit recorded before the output was read (%d lines, stderr tail %v, stdout tail %v)",
				attempt, len(logs), sawLast, sawTail)
		}
	}
}

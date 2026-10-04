package api

import (
	"bytes"
	"os"
	"strconv"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// The container stop path has to take vLLM down itself. Leaving it to the
// container's teardown SIGKILLs it with no chance to release the GPUs cleanly.
func TestShutdownStopsARunningServer(t *testing.T) {
	s := leaseServer(t)
	serve(t, s, "org/served")
	pid := s.process.GetStatus().PID

	s.Shutdown()

	if st := s.process.GetStatus().State; st != process.StateStopped {
		t.Fatalf("state after Shutdown = %s, want stopped", st)
	}
	if processRunning(pid) {
		t.Errorf("vLLM pid %d is still running after Shutdown", pid)
	}
}

// With nothing running, Shutdown has nothing to stop and must not hang or
// complain.
func TestShutdownWithNothingRunning(t *testing.T) {
	s := leaseServer(t)
	s.Shutdown()
	if st := s.process.GetStatus().State; st != process.StateStopped {
		t.Errorf("state = %s, want stopped", st)
	}
}

// processRunning reports whether pid is alive and not a zombie: the fake is a
// child of the test binary, and Stop can return before cmd.Wait reaps it.
func processRunning(pid int) bool {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false
	}
	i := bytes.LastIndexByte(b, ')')
	return i >= 0 && i+2 < len(b) && b[i+2] != 'Z'
}

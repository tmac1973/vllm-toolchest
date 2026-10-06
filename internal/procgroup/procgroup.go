// Package procgroup starts a child in its own process group, so that
// cancelling it reaches everything it spawned rather than only the direct
// child.
//
// vLLM and the tools built on it are never one process: importing vLLM forks
// an EngineCore and per-rank workers. Those are grandchildren. Killing only
// the direct child orphans them, and an orphan that reached the GPU keeps its context -- and its VRAM --
// alive, so the next start fails on free memory with nothing on screen
// saying why. A new group is what makes kill(-pgid) safe: the workers inherit
// it, and vllmctl, which is not in it, is not signalled.
package procgroup

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// ownSecrets are vllmctl's own credentials, which reach it through the
// container environment. No child needs them under these names: the engine
// is handed its key as VLLM_API_KEY, and HF_TOKEN, which vLLM and the Hub
// tools read, is a separate name and is passed through.
var ownSecrets = []string{"VLLMCTL_API_KEY=", "VLLMCTL_HF_TOKEN="}

// Environ is os.Environ without vllmctl's own secrets, the base environment
// for every child it starts.
func Environ() []string {
	env := os.Environ()
	out := env[:0:0]
	for _, kv := range env {
		secret := false
		for _, prefix := range ownSecrets {
			if strings.HasPrefix(kv, prefix) {
				secret = true
				break
			}
		}
		if !secret {
			out = append(out, kv)
		}
	}
	return out
}

// Command is exec.CommandContext with the child placed in a new process
// group. When ctx is done the whole group gets sig. If the child has still
// not exited wait later, Go kills it and stops waiting on its output: a
// surviving grandchild holds the output pipes open, and without that bound
// Wait (and Output, CombinedOutput) blocks long past the cancel.
func Command(ctx context.Context, sig syscall.Signal, wait time.Duration, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return Kill(cmd, sig)
	}
	cmd.WaitDelay = wait
	return cmd
}

// Kill sends sig to cmd's whole process group. It is a no-op for a command
// that never started.
func Kill(cmd *exec.Cmd, sig syscall.Signal) error {
	if cmd.Process == nil || cmd.Process.Pid <= 0 {
		return nil
	}
	return syscall.Kill(-cmd.Process.Pid, sig)
}

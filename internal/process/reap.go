package process

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strconv"
	"syscall"
	"time"
)

// procState is one process's run state and process group, read from
// /proc/<pid>/stat.
type procState struct {
	state byte // R, S, D, Z, ...
	pgrp  int
}

// readProcStat parses /proc/<pid>/stat. The command name is parenthesised and
// may itself contain spaces and parentheses, so the fields are found after the
// last ')' rather than by splitting the whole line.
func readProcStat(pid string) (procState, error) {
	b, err := os.ReadFile("/proc/" + pid + "/stat")
	if err != nil {
		return procState{}, err
	}
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return procState{}, fmt.Errorf("malformed stat for %s", pid)
	}
	// After the name: state ppid pgrp ...
	f := bytes.Fields(b[i+1:])
	if len(f) < 3 || len(f[0]) != 1 {
		return procState{}, fmt.Errorf("malformed stat for %s", pid)
	}
	pgrp, err := strconv.Atoi(string(f[2]))
	if err != nil {
		return procState{}, err
	}
	return procState{state: f[0][0], pgrp: pgrp}, nil
}

// processAlive reports whether pid is running. A zombie counts as exited: it
// holds no memory, no GPU context and no socket, only its pid until someone
// waits on it.
func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	st, err := readProcStat(strconv.Itoa(pid))
	if err == nil {
		return st.state != 'Z'
	}
	if errors.Is(err, os.ErrNotExist) {
		return false
	}
	return syscall.Kill(pid, 0) == nil
}

// groupAlive reports whether any process in group pgid is still running,
// ignoring zombies.
//
// kill(-pgid, 0) cannot answer this. vllmctl is PID 1 in its container, so a
// worker orphaned by its server is reparented to it -- and nothing waits on
// those, so they stay zombies forever. On compute a dead server's group still
// had one, and kill -0 said the group existed for as long as the container
// did. Asked that way, a stale group would never read as gone.
//
// A pid number cannot be reused while it is still some group's id, so a group
// found here is the one that was recorded, not a recycled one.
func groupAlive(pgid int) bool {
	if pgid <= 0 {
		return false
	}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return syscall.Kill(-pgid, 0) == nil
	}
	for _, e := range entries {
		name := e.Name()
		if name[0] < '0' || name[0] > '9' {
			continue
		}
		st, err := readProcStat(name)
		if err != nil {
			continue // exited while we looked
		}
		if st.pgrp == pgid && st.state != 'Z' {
			return true
		}
	}
	return false
}

// waitUntil polls cond until it holds or d passes, and reports which.
func waitUntil(cond func() bool, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for {
		if cond() {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// terminateGroup ends the process group led by pgid: SIGTERM to all of it, a
// grace period for the leader to shut down cleanly, then SIGKILL to anything
// left. It returns once nothing in the group is running, or an error if
// something survived even SIGKILL.
//
// It waits on the processes themselves, not on the manager's state or on
// cmd.Wait. Wait also waits for the output pipes, which a surviving worker
// holds open, so waiting on it put the WaitDelay in front of every sweep.
func terminateGroup(pgid int, grace, killWait time.Duration) error {
	if !groupAlive(pgid) {
		return nil
	}
	if err := killProcessGroup(pgid, syscall.SIGTERM); err != nil {
		slog.Warn("signalling vLLM process group", "pgid", pgid, "error", err)
	}
	if !waitUntil(func() bool { return !processAlive(pgid) }, grace) {
		slog.Warn("vLLM did not exit on SIGTERM; killing the process group", "pgid", pgid)
	}
	if groupAlive(pgid) {
		slog.Info("reaping vLLM workers that outlived the server", "pgid", pgid)
		killProcessGroup(pgid, syscall.SIGKILL)
	}
	if !waitUntil(func() bool { return !groupAlive(pgid) }, killWait) {
		return fmt.Errorf("vLLM process group %d is still running after SIGKILL; "+
			"a process stuck in the GPU driver cannot be killed until it returns, "+
			"so check dmesg and the GPUs before starting anything else", pgid)
	}
	return nil
}

// portInUse reports whether something is listening on the port vLLM is about
// to bind. It binds the same address vLLM does, so the answer is the one
// vLLM's own bind would get.
func portInUse(port int) bool {
	if port <= 0 {
		return false
	}
	l, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", port))
	if err != nil {
		return true
	}
	l.Close()
	return false
}

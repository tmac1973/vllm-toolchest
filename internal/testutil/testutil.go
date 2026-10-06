// Package testutil holds helpers shared by tests across packages. It is
// imported only from _test files.
package testutil

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// WriteScript writes body as an executable /bin/sh script in a fresh temp
// directory and returns its path. body is everything after the #! line. It
// stands in for vLLM, the tuner or any other child a test needs to start.
func WriteScript(t testing.TB, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fake")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// Eventually polls cond until it holds, failing the test if it has not
// within the given time. msg and args describe what never happened.
//
// It replaces fixed sleeps: a test waits only as long as it must, and a slow
// machine under the full suite's load gets the whole window rather than a
// guess at it.
func Eventually(t testing.TB, within time.Duration, cond func() bool, msg string, args ...any) {
	t.Helper()
	deadline := time.Now().Add(within)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("after %s: %s", within, fmt.Sprintf(msg, args...))
		}
		time.Sleep(10 * time.Millisecond)
	}
}

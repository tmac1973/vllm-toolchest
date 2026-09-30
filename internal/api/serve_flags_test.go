package api

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

func TestFlagSupport(t *testing.T) {
	var unknown flagSupport
	if unknown.Known() || !unknown.Has("--made-up") {
		t.Error("an unknown list must pass every flag")
	}
	known := newFlagSupport([]string{"--max-model-len", "--kv-cache-dtype"})
	if !known.Known() || known.Has("--made-up") {
		t.Error("a known list passed a flag it does not have")
	}
	if !known.Has("--max-model-len=4096") {
		t.Error("a =value suffix was not ignored")
	}
}

// fakeServeLauncher answers `serve --help` with a listing of n flags, and
// counts how often it was asked.
func fakeServeLauncher(t *testing.T, n int, fail bool) (vllmenv.Env, string) {
	t.Helper()
	dir := t.TempDir()
	count := filepath.Join(dir, "calls")
	var b strings.Builder
	b.WriteString("usage: vllm serve\n  --max-model-len N\n  --tensor-parallel-size N\n")
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "  --flag-%d\n", i)
	}
	listing := filepath.Join(dir, "listing.txt")
	os.WriteFile(listing, []byte(b.String()), 0o644)
	body := "#!/bin/sh\necho x >> '" + count + "'\ncat '" + listing + "'\n"
	if fail {
		body = "#!/bin/sh\necho x >> '" + count + "'\necho 'ImportError: no GPU' >&2\nexit 1\n"
	}
	script := filepath.Join(dir, "vllm")
	os.WriteFile(script, []byte(body), 0o755)
	return vllmenv.Env{Variant: "rocm", VariantVersion: "v1", Launcher: []string{script, "serve"}}, count
}

func calls(path string) int {
	data, _ := os.ReadFile(path)
	return strings.Count(string(data), "x")
}

func TestRefreshServeFlagsProbesOnceAndCaches(t *testing.T) {
	env, count := fakeServeLauncher(t, 30, false)
	dir := t.TempDir()
	s := &Server{cfg: &config.Config{DataDir: dir}, vllmEnv: env}

	if s.serveFlags().Known() {
		t.Fatal("known before any probe")
	}
	s.RefreshServeFlags()
	if f := s.serveFlags(); !f.Known() || !f.Has("--flag-3") || f.Has("--nope") {
		t.Fatalf("after the probe: %+v", f)
	}
	var c serveFlagsCache
	data, err := os.ReadFile(filepath.Join(dir, "cache", "serve-flags.json"))
	if err != nil || json.Unmarshal(data, &c) != nil || c.Variant != "rocm" || c.VariantVersion != "v1" {
		t.Fatalf("cache not written: %v %+v", err, c)
	}

	// The next boot reads the file and does not probe.
	next := &Server{cfg: &config.Config{DataDir: dir}, vllmEnv: env}
	if !next.serveFlags().Known() {
		t.Error("the cache was not read before any probe")
	}
	next.RefreshServeFlags()
	if n := calls(count); n != 1 {
		t.Errorf("probed %d times, want 1", n)
	}

	// Another image version retires the file.
	env.VariantVersion = "v2"
	other := &Server{cfg: &config.Config{DataDir: dir}, vllmEnv: env}
	if other.serveFlags().Known() {
		t.Error("a list for another image version was used")
	}
}

// An image with no version stamp gets a list for this process only.
func TestRefreshServeFlagsWithoutAStamp(t *testing.T) {
	env, _ := fakeServeLauncher(t, 30, false)
	env.VariantVersion = ""
	dir := t.TempDir()
	s := &Server{cfg: &config.Config{DataDir: dir}, vllmEnv: env}
	s.RefreshServeFlags()
	if !s.serveFlags().Known() {
		t.Error("the probe result was not kept in memory")
	}
	if _, err := os.Stat(filepath.Join(dir, "cache", "serve-flags.json")); err == nil {
		t.Error("an unstamped image's list was written to the cache")
	}
}

func TestRefreshServeFlagsFailureLeavesItUnknown(t *testing.T) {
	env, _ := fakeServeLauncher(t, 0, true)
	dir := t.TempDir()
	s := &Server{cfg: &config.Config{DataDir: dir}, vllmEnv: env}
	s.RefreshServeFlags()
	if s.serveFlags().Known() {
		t.Error("a failed probe produced a list")
	}
	if _, err := os.Stat(filepath.Join(dir, "cache", "serve-flags.json")); err == nil {
		t.Error("a failed probe wrote a cache file")
	}
}

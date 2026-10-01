package vllmenv

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// A real listing: `vllm serve --help=all` from the 0.27 radiance image,
// start-up chatter and all.
func TestParseServeFlagsOnARealListing(t *testing.T) {
	data, err := os.ReadFile("testdata/serve-help-0.27.txt")
	if err != nil {
		t.Fatal(err)
	}
	flags := parseServeFlags(string(data))
	if !looksLikeServeHelp(flags) {
		t.Fatalf("a real listing was not recognised: %d flags", len(flags))
	}
	for _, want := range []string{
		"--max-model-len", "--kv-cache-dtype", "--enable-auto-tool-choice",
		"--async-scheduling", "--no-async-scheduling", "--speculative-config",
	} {
		if !slices.Contains(flags, want) {
			t.Errorf("%s not found", want)
		}
	}
	for _, f := range flags {
		if !strings.HasPrefix(f, "--") || strings.HasSuffix(f, "-") {
			t.Errorf("%q is not a flag", f)
		}
	}
}

// An image's start-up lines come before "usage:" and are not read: a JIT
// build echoing its compiler flags must not add them to the serve flags.
func TestParseServeFlagsIgnoresStartUpChatter(t *testing.T) {
	flags := parseServeFlags("[aiter] build with --offload-arch=gfx1100\nusage: vllm serve\n  --max-model-len N\n")
	if slices.Contains(flags, "--offload-arch") || !slices.Contains(flags, "--max-model-len") {
		t.Errorf("flags = %v", flags)
	}
}

// launcher writes a stand-in for `vllm serve` that answers each help request
// with the given body.
func launcher(t *testing.T, summary, all string) Env {
	t.Helper()
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	s, a := write("summary.txt", summary), write("all.txt", all)
	script := filepath.Join(dir, "vllm")
	body := "#!/bin/sh\n" +
		`for arg; do case "$arg" in --help=all) cat '` + a + `'; exit 0;; --help) cat '` + s + `'; exit 0;; esac; done` + "\nexit 2\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return Env{Launcher: []string{script, "serve"}}
}

func fakeListing(n int) string {
	var b strings.Builder
	b.WriteString("usage: vllm serve [model_tag] [options]\n  --max-model-len N\n  --tensor-parallel-size N\n")
	for i := 0; i < n; i++ {
		b.WriteString("  --flag-" + string(rune('a'+i%26)) + string(rune('a'+i/26)) + "\n")
	}
	return b.String()
}

func TestProbeServeFlagsAsksForTheFullListing(t *testing.T) {
	summary := "usage: vllm serve\n  Use `--help=all` to show all available flags at once.\n"
	env := launcher(t, summary, fakeListing(30))
	flags, err := env.ProbeServeFlags(10 * time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if len(flags) < 30 || !slices.Contains(flags, "--max-model-len") {
		t.Errorf("got %d flags; the full listing was not read", len(flags))
	}
}

func TestProbeServeFlagsTakesAPlainListing(t *testing.T) {
	env := launcher(t, fakeListing(25), "should not be read")
	flags, err := env.ProbeServeFlags(10 * time.Second)
	if err != nil || len(flags) < 25 {
		t.Errorf("flags=%d err=%v", len(flags), err)
	}
}

func TestProbeServeFlagsRefusesWhatIsNotAListing(t *testing.T) {
	env := launcher(t, "usage: something else\n  --only --three --flags\n", "")
	if _, err := env.ProbeServeFlags(10 * time.Second); err == nil {
		t.Error("three flags were accepted as a serve listing")
	}
}

func TestProbeServeFlagsTimesOut(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "vllm")
	os.WriteFile(script, []byte("#!/bin/sh\nsleep 30\n"), 0o755)
	env := Env{Launcher: []string{script, "serve"}}

	start := time.Now()
	if _, err := env.ProbeServeFlags(300 * time.Millisecond); err == nil {
		t.Error("a probe that never answered returned no error")
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Errorf("the probe took %s to give up", d)
	}
}

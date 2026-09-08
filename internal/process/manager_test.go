package process

import (
	"reflect"
	"strings"
	"testing"
)

func TestSplitFlags(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"plain", "--disable-log-requests --swap-space 4",
			[]string{"--disable-log-requests", "--swap-space", "4"}},
		{"collapses runs of whitespace", "  --a\t\t--b\n--c ",
			[]string{"--a", "--b", "--c"}},

		// The reason this function exists: vLLM's structured flags carry JSON,
		// and field splitting tears a spaced JSON object into broken fragments.
		{"single-quoted json stays one arg",
			`--speculative-config='{"method": "mtp", "num_speculative_tokens": 8}'`,
			[]string{`--speculative-config={"method": "mtp", "num_speculative_tokens": 8}`}},
		// Shell semantics: inside double quotes, \" unescapes to a bare quote.
		{"double-quoted json stays one arg, escapes resolved",
			`--compilation-config="{\"x\": 1}"`,
			[]string{`--compilation-config={"x": 1}`}},
		{"backslash outside quotes escapes the next char",
			`--chat-template /work/my\ template.jinja`,
			[]string{"--chat-template", "/work/my template.jinja"}},
		{"single quotes are literal", `--a '\n"x"'`, []string{"--a", `\n"x"`}},
		{"quoted value with following flag",
			`--chat-template '/work/my template.jinja' --trust-remote-code`,
			[]string{"--chat-template", "/work/my template.jinja", "--trust-remote-code"}},
		{"empty quoted value survives", `--foo=""`, []string{"--foo="}},

		// A half-typed flag should degrade, not vanish.
		{"unterminated quote runs to end", `--a 'bc`, []string{"--a", "bc"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := SplitFlags(tc.in); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("SplitFlags(%q)\n got %#v\nwant %#v", tc.in, got, tc.want)
			}
		})
	}
}

func TestBuildArgsOmitsUnsetOptionalFlags(t *testing.T) {
	args := strings.Join(BuildArgs(VLLMStartConfig{}), " ")
	for _, flag := range []string{
		"--attention-backend", "--reasoning-parser", "--mamba-cache-mode",
		"--speculative-config", "--compilation-config", "--kv-cache-memory",
		"--no-async-scheduling", "--language-model-only",
	} {
		if strings.Contains(args, flag) {
			t.Errorf("zero config emitted %s (got: %s)", flag, args)
		}
	}
}

func TestBuildArgsEmitsRadianceFlags(t *testing.T) {
	args := BuildArgs(VLLMStartConfig{
		AttentionBackend:       "ROCM_AITER_UNIFIED_ATTN",
		ReasoningParser:        "qwen3",
		EnablePrefixCaching:    true,
		MambaCacheMode:         "align",
		SpeculativeConfig:      `{"method":"mtp","num_speculative_tokens":8}`,
		CompilationConfig:      `{"cudagraph_capture_sizes":[1,2,4]}`,
		KVCacheMemory:          15989735424,
		DisableAsyncScheduling: true,
		LanguageModelOnly:      true,
	})
	joined := strings.Join(args, " ")

	for _, want := range []string{
		"--attention-backend ROCM_AITER_UNIFIED_ATTN",
		"--reasoning-parser qwen3",
		"--enable-prefix-caching",
		"--mamba-cache-mode align",
		`--speculative-config={"method":"mtp","num_speculative_tokens":8}`,
		`--compilation-config={"cudagraph_capture_sizes":[1,2,4]}`,
		"--kv-cache-memory 15989735424",
		"--no-async-scheduling",
		"--language-model-only",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in: %s", want, joined)
		}
	}

	// The JSON flags must survive as a single argv entry — splitting them is
	// exactly the failure this guards against.
	var sawSpec bool
	for _, a := range args {
		if strings.HasPrefix(a, "--speculative-config=") {
			sawSpec = true
			if !strings.HasSuffix(a, "}") {
				t.Errorf("speculative config was split: %q", a)
			}
		}
	}
	if !sawSpec {
		t.Error("no --speculative-config argument emitted")
	}
}

func TestBuildEnvAppendsExtras(t *testing.T) {
	env := BuildEnv("awq", "RADIANCE_USE_R4D=0", "RADIANCE_DRAFT_TAU=0.35")
	joined := strings.Join(env, " ")
	for _, want := range []string{
		"VLLM_WORKER_MULTIPROC_METHOD=spawn",
		"VLLM_USE_TRITON_AWQ=1",
		"RADIANCE_USE_R4D=0",
		"RADIANCE_DRAFT_TAU=0.35",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %v", want, env)
		}
	}

	// Extras go last so they win over anything inherited from the image.
	if got := env[len(env)-1]; got != "RADIANCE_DRAFT_TAU=0.35" {
		t.Errorf("extras must be appended last, got %q", got)
	}
}

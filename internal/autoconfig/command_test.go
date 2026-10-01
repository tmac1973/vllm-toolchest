package autoconfig

import (
	"os"
	"slices"
	"testing"
)

func readCard(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestExtractFromTheTwentySevenBCard(t *testing.T) {
	raw := readCard(t, "thinkingcap-27b-paro5.md")
	cmds := ExtractCommands(raw)
	if len(cmds) != 3 {
		t.Fatalf("got %d commands, want the docker one and the base card's two", len(cmds))
	}

	d := cmds[0]
	if !d.Docker || d.Partial || d.Model != "/app/models" {
		t.Errorf("first command: docker=%v partial=%v model=%q", d.Docker, d.Partial, d.Model)
	}
	if len(d.Env) != 9 || !slices.Contains(d.Env, "GPU_MAX_HW_QUEUES=2") || !slices.Contains(d.Env, "ROCR_VISIBLE_DEVICES=0,1,2,3") {
		t.Errorf("env = %q", d.Env)
	}
	s := d.Settings()
	st := s.Start
	if st.TensorParallelSize != 4 || st.MaxModelLen != 262144 || st.KVCacheDtype != "fp8" ||
		st.ToolCallParser != "qwen3_coder" || !st.EnableAutoToolChoice || st.ReasoningParser != "qwen3" ||
		st.MaxNumBatchedTokens != 8192 || !st.EnableChunkedPrefill || st.MaxNumSeqs != 8 ||
		st.GPUMemoryUtilization != 0.92 || st.CompilationConfig == "" {
		t.Errorf("known fields: %+v", st)
	}
	if st.ServedModelName != "" {
		t.Error("the card's served name was carried")
	}
	if len(s.Rest) != 1 || s.Rest[0][0] != "--override-generation-config" || len(s.Rest[0]) != 2 {
		t.Errorf("rest = %q, want only the generation override with its value", s.Rest)
	}
	for _, g := range s.Rest {
		if g[0] == "--port" || g[0] == "--host" {
			t.Errorf("%s was carried", g[0])
		}
	}

	// The base model's card: two direct commands for another repository.
	if cmds[1].Docker || cmds[1].Model != "bottlecapai/ThinkingCap-Qwen3.8-27B" {
		t.Errorf("second command: %+v", cmds[1])
	}
	if cmds[2].Settings().Start.SpeculativeConfig == "" {
		t.Error("the MTP variant lost its speculative config")
	}

	inline := InlineFlags(raw)
	if len(inline) != 1 || inline[0][0] != "--speculative-config" || len(inline[0]) != 2 {
		t.Fatalf("inline flags = %q", inline)
	}
	if want := `{"method": "dflash", "model": "/app/draft", "num_speculative_tokens": 7}`; inline[0][1] != want {
		t.Errorf("inline value = %s", inline[0][1])
	}
}

func TestExtractFromTheMoECard(t *testing.T) {
	cmds := ExtractCommands(readCard(t, "flash-next-mxfp4.md"))
	var full, partial []Command
	for _, c := range cmds {
		if c.Partial {
			partial = append(partial, c)
		} else {
			full = append(full, c)
		}
	}
	if len(full) != 2 {
		t.Fatalf("%d full commands, want the four-card and two-card recipes", len(full))
	}
	if !slices.Contains(full[0].Env, "ROCR_VISIBLE_DEVICES=0,1,2,3") || !slices.Contains(full[1].Env, "ROCR_VISIBLE_DEVICES=0,1") {
		t.Errorf("the two recipes' devices: %q / %q", full[0].Env, full[1].Env)
	}
	if full[0].Settings().Start.TensorParallelSize != 4 || full[1].Settings().Start.TensorParallelSize != 2 {
		t.Error("widths not read")
	}
	var offload bool
	for _, g := range full[1].Settings().Rest {
		offload = offload || g[0] == "--enable-expert-offload"
	}
	if !offload {
		t.Error("the image-specific offload flag was not kept")
	}
	if len(partial) != 1 || !slices.Equal(partial[0].Env, []string{"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"}) {
		t.Errorf("partial commands: %+v", partial)
	}
}

func TestExtractFromTheAWQCard(t *testing.T) {
	cmds := ExtractCommands(readCard(t, "qwen3-14b-awq.md"))
	if len(cmds) != 2 {
		t.Fatalf("got %d commands", len(cmds))
	}
	s := cmds[0].Settings()
	if cmds[0].Partial || s.Start.ReasoningParser != "deepseek_r1" {
		t.Errorf("first: partial=%v %+v", cmds[0].Partial, s.Start)
	}
	if len(s.Rest) != 1 || s.Rest[0][0] != "--enable-reasoning" {
		t.Errorf("rest = %q", s.Rest)
	}
	if !cmds[1].Partial {
		t.Error("`vllm serve ... --rope-scaling` is a fragment, not a command")
	}
}

func TestExtractCommandShapes(t *testing.T) {
	for _, tc := range []struct {
		name, card string
		want       int
		check      func(t *testing.T, c Command)
	}{
		{
			name: "continuations and an export",
			card: "```bash\nexport HF_HOME=/x\nexport VLLM_USE_V1=1\nvllm serve org/m \\\n  --max-model-len 4096 \\\n  --enforce-eager\n```",
			want: 1,
			check: func(t *testing.T, c Command) {
				if !slices.Equal(c.Env, []string{"HF_HOME=/x", "VLLM_USE_V1=1"}) || c.Settings().Start.MaxModelLen != 4096 {
					t.Errorf("%+v", c)
				}
			},
		},
		{
			name: "docker with vllm serve after the image",
			card: "```\ndocker run --gpus all -e A=1 --env=B=2 -p 8000:8000 vllm/vllm-openai:latest vllm serve org/m --max-model-len 8192 --kv-cache-dtype fp8\n```",
			want: 1,
			check: func(t *testing.T, c Command) {
				if c.Model != "org/m" || !slices.Equal(c.Env, []string{"A=1", "B=2"}) || c.Settings().Start.KVCacheDtype != "fp8" {
					t.Errorf("%+v", c)
				}
			},
		},
		{name: "another container", card: "```\ndocker run -p 80:80 nginx\n```", want: 0},
		{name: "an install line", card: "```\npip install vllm\n```", want: 0},
		{
			name: "piped and backgrounded",
			card: "```\nvllm serve org/m --max-model-len 2048 > log.txt 2>&1 &\n```",
			want: 1,
			check: func(t *testing.T, c Command) {
				if len(c.Settings().Rest) != 0 {
					t.Errorf("shell redirection was read as flags: %q", c.Settings().Rest)
				}
			},
		},
		{
			name: "the same command twice",
			card: "```\nvllm serve org/m --enforce-eager\n```\n\n```\nvllm serve org/m --enforce-eager\n```",
			want: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtractCommands(tc.card)
			if len(got) != tc.want {
				t.Fatalf("got %d commands: %+v", len(got), got)
			}
			if tc.check != nil && len(got) > 0 {
				tc.check(t, got[0])
			}
		})
	}
}

func TestInlineFlagsIgnoresTheUnknown(t *testing.T) {
	got := InlineFlags("Pass `--help` to see more, or `--max-model-len=8192` for less. SGLang uses `--context-length 8192`.")
	if len(got) != 1 || !slices.Equal(got[0], []string{"--max-model-len", "8192"}) {
		t.Errorf("got %q", got)
	}
}

// A quantized repository's card and its base model's often give the same
// recipes, differing only in the repository named. Each is listed once, as
// the model's own card gives it.
func TestExtractCommandsListsARecipeOnce(t *testing.T) {
	raw := "# Quant\n\n```\nvllm serve org/quant --tensor-parallel-size 4 --reasoning-parser qwen3\n```\n\n" +
		"# Base\n\n```\nvllm serve org/base --tensor-parallel-size 4 --reasoning-parser qwen3\n```\n\n" +
		"```\nvllm serve org/base --tensor-parallel-size 8\n```\n"
	cmds := ExtractCommands(raw)
	if len(cmds) != 2 {
		t.Fatalf("got %d commands, want 2: %+v", len(cmds), cmds)
	}
	if cmds[0].Model != "org/quant" {
		t.Errorf("the base card's copy was kept, not the model's own: %s", cmds[0].Model)
	}
}

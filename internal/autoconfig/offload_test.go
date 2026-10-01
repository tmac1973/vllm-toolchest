package autoconfig

import (
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// Choosing the experts-in-RAM width turns offload on and leaves out a ticked
// MTP config, which this image cannot start with offload; the other widths
// leave both alone.
func TestTheOffloadWidth(t *testing.T) {
	mtp := `{"method": "mtp", "num_speculative_tokens": 3}`
	all := models.WidthPlan{TP: 4, ContextTokens: 65536, Config: models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 65536}}
	off := models.WidthPlan{TP: 4, ContextTokens: 262144, Offload: true, FullContextRequests: 1,
		Config: models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 262144, ExtraFlags: models.ExpertOffloadFlag}}
	r := &Result{
		Base: models.VLLMConfig{ExtraFlags: "--max-num-batched-tokens 8192"},
		Rows: []Row{{Key: "field:speculative_config", Kind: RowField, Field: "speculative_config", Value: mtp, Quote: "q", Ticked: true}},
		Plan: models.FitPlan{Known: true, All: all, Offload: &off},
	}

	cfg, _, err := r.Config("offload", nil)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MaxModelLen != 262144 || !process.HasFlag(cfg.ExtraFlags, models.ExpertOffloadFlag) || cfg.SpeculativeConfig != "" ||
		!strings.Contains(cfg.ExtraFlags, "--max-num-batched-tokens 8192") {
		t.Errorf("offload: %+v", cfg)
	}
	if w := ChosenWidth(r.Plan, "offload"); w == nil || !w.Offload {
		t.Error("the offload width is not chosen")
	}

	cfg, _, _ = r.Config("all", nil)
	if cfg.MaxModelLen != 65536 || process.HasFlag(cfg.ExtraFlags, models.ExpertOffloadFlag) || cfg.SpeculativeConfig != mtp {
		t.Errorf("all cards: %+v", cfg)
	}

	// When offload is the only plan, it is the all-cards width.
	r.Plan = models.FitPlan{Known: true, All: off}
	if cfg, _, _ := r.Config("all", nil); !process.HasFlag(cfg.ExtraFlags, models.ExpertOffloadFlag) || cfg.SpeculativeConfig != "" {
		t.Errorf("offload only: %+v", cfg)
	}
}

// On a re-run after the first was applied, the live config already holds what
// the rows propose. Unticking one then removes it -- Gemma 4's speculative
// config could not otherwise be got rid of. A row proposing something other
// than what is there still leaves it alone when unticked.
func TestUntickingWhatIsAlreadySetRemovesIt(t *testing.T) {
	spec := `{"method": "eagle3", "model": "/m/d", "num_speculative_tokens": 3}`
	base := models.VLLMConfig{
		SpeculativeConfig: spec, EnableAutoToolChoice: true, ToolCallParser: "gemma4",
		ExtraFlags: process.SetFlag("--keep 1", []string{"--limit-mm-per-prompt", `{"image":10}`}), Env: "FOO=1\nBAR=2",
	}
	rows := []Row{
		{Key: "field:speculative_config", Kind: RowField, Field: "speculative_config", Value: spec},
		{Key: "field:enable_auto_tool_choice", Kind: RowField, Field: "enable_auto_tool_choice", Value: "true"},
		{Key: "flag:--limit-mm-per-prompt", Kind: RowFlag, Flag: []string{"--limit-mm-per-prompt", `{"image":10}`}},
		{Key: "env:FOO", Kind: RowEnv, Env: "FOO=1"},
		{Key: "field:tool_call_parser", Kind: RowField, Field: "tool_call_parser", Value: "pythonic"},
	}
	c, err := Apply(base, rows, map[string]bool{})
	if err != nil {
		t.Fatal(err)
	}
	if c.SpeculativeConfig != "" || c.EnableAutoToolChoice || process.HasFlag(c.ExtraFlags, "--limit-mm-per-prompt") ||
		!process.HasFlag(c.ExtraFlags, "--keep") || c.Env != "BAR=2" {
		t.Errorf("unticked rows that were set: %+v", c)
	}
	if c.ToolCallParser != "gemma4" {
		t.Errorf("an unticked change replaced what was there: %q", c.ToolCallParser)
	}
}

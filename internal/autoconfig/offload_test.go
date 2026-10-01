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

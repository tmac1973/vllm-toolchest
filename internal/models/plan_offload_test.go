package models

import (
	"strings"
	"testing"
)

// flashNext is tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ as registered on
// compute: 512 experts in 48 layers, a 47.7 GB n-gram table the image keeps in
// RAM, and an MTP head.
func flashNext() *Model {
	return &Model{
		ID:             "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ",
		TotalSizeBytes: 116989200000,
		HFConfig: HFConfig{
			Architectures: []string{"Qwen4ExpForConditionalGeneration"}, ModelType: "qwen4_exp",
			NumHiddenLayers: 48, HiddenSize: 2560, NumAttentionHeads: 24, NumKeyValueHeads: 2, HeadDim: 256,
			MaxPositionEmbeddings: 262144, AttentionLayers: 12, NumExperts: 512, NumExpertsPerTok: 10,
			MoEIntermediate: 640, SharedExpertInter: 640, PLELayers: 1, MTPLayers: 1, VocabSize: 248320,
			MetaVersion: hfMetaVersion,
		},
		Quantization: QuantMeta{Method: "compressed-tensors", Bits: 4, BytesPerParam: 0.5625},
		VLLMConfig: VLLMConfig{
			KVCacheDtype: "fp8", Env: "VLLM_PLE_CPU_OFFLOAD=1",
			SpeculativeConfig: `{"method": "mtp", "num_speculative_tokens": 3}`,
		},
	}
}

func offloadInput(m *Model, cards int, perCardGB float64) PlanInput {
	return PlanInput{
		Model: m, Base: m.VLLMConfig, Class: ContextMax,
		Inventory: GPUInventory{Count: cards, PerCardGB: perCardGB, Known: true},
		Defaults:  PlanDefaults{GPUMemoryUtilization: 0.90, MaxNumSeqs: 16},
		Estimate: func(c VLLMConfig) VRAMEstimate {
			x := *m
			x.VLLMConfig = c
			return EstimateVRAM(&x, nil)
		},
		ExpertOffload: true, HostRAMGB: 188.5,
	}
}

// Its experts at MXFP4 come to what the engine counted: 59.9 GiB.
func TestExpertWeight(t *testing.T) {
	if got := expertWeightGB(flashNext().HFConfig); got < 59 || got > 61 {
		t.Errorf("experts %.1f GiB", got)
	}
}

// On two cards it fits only with its experts in RAM, as it served on compute:
// the plan is offload, at the context asked for, one request at a time, with
// the flag set and MTP left out.
func TestOffloadWhenNothingElseFits(t *testing.T) {
	p := PlanFit(offloadInput(flashNext(), 2, 31.86))
	if !p.Known || !p.All.Offload || p.All.TP != 2 || p.All.ContextTokens != 262144 || p.All.FullContextRequests != 1 {
		t.Fatalf("plan: known=%v %s offload=%v TP=%d ctx=%d", p.Known, p.Why, p.All.Offload, p.All.TP, p.All.ContextTokens)
	}
	if !strings.Contains(p.All.Config.ExtraFlags, ExpertOffloadFlag) || p.All.Config.SpeculativeConfig != "" {
		t.Errorf("config: flags %q spec %q", p.All.Config.ExtraFlags, p.All.Config.SpeculativeConfig)
	}
	if !hasNote(p.All.Notes, "speculative_config") || !hasNote(p.All.Notes, "extra_flags") {
		t.Errorf("notes: %+v", p.All.Notes)
	}

	// Four cards hold it without: no offload at all.
	if p := PlanFit(offloadInput(flashNext(), 4, 31.86)); !p.Known || p.All.Offload || p.Offload != nil {
		t.Errorf("four cards: offload=%v alternative=%v", p.All.Offload, p.Offload != nil)
	}
}

func TestNoOffloadWhereItCannotHelp(t *testing.T) {
	// An image without it.
	in := offloadInput(flashNext(), 2, 31.86)
	in.ExpertOffload = false
	if p := PlanFit(in); p.Known {
		t.Error("planned offload on an image without it")
	}
	// Not enough RAM for the experts beside the table.
	in = offloadInput(flashNext(), 2, 31.86)
	in.HostRAMGB = 96
	if p := PlanFit(in); p.Known || !strings.Contains(p.Why, "host's memory") {
		t.Errorf("small host: %q", p.Why)
	}
	// Experts of 8 bits or more are not what the image offloads.
	m := flashNext()
	m.Quantization = QuantMeta{Method: "fp8", BytesPerParam: 1.0625}
	if p := PlanFit(offloadInput(m, 2, 31.86)); p.Known && p.All.Offload {
		t.Error("offloaded an 8-bit MoE")
	}
	// Nor is a dense model.
	m = flashNext()
	m.HFConfig.NumExperts, m.HFConfig.MoEIntermediate = 0, 0
	if offloadEligible(m) {
		t.Error("a dense model is eligible")
	}
}

// When the cards hold the model but cut its context, offload is offered
// beside the plan, not in place of it.
func TestOffloadAsAnAlternativeWhenTheContextIsCut(t *testing.T) {
	m := flashNext()
	m.HFConfig.AttentionLayers, m.HFConfig.NumKeyValueHeads = 48, 16 // a cache forty times the size
	m.VLLMConfig.Env = "VLLM_PLE_CPU_OFFLOAD=1"
	for _, per := range []float64{36, 40, 44, 48} {
		p := PlanFit(offloadInput(m, 4, per))
		if !p.Known || p.All.Offload || !p.All.ContextReduced {
			continue
		}
		if p.Offload == nil || !p.Offload.Offload || p.Offload.ContextTokens <= p.All.ContextTokens {
			t.Fatalf("%.0f GB cards: context cut to %d, alternative %+v", per, p.All.ContextTokens, p.Offload)
		}
		return
	}
	t.Fatal("no card size gave a plan that fits with its context cut")
}

func TestWithoutFlagKeepsQuotedValues(t *testing.T) {
	in := `--override-generation-config '{"max_tokens":  65536}' --enable-expert-offload`
	if got := withoutFlag(in, ExpertOffloadFlag); got != `--override-generation-config '{"max_tokens":  65536}'` {
		t.Errorf("got %q", got)
	}
}

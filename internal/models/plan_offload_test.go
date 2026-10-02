package models

import (
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/process"
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

func TestHeadsSplit(t *testing.T) {
	for _, c := range []struct {
		q, kv, tp int
		want      bool
	}{
		{16, 2, 4, true},  // Qwen3.5: two KV heads copied onto four cards
		{32, 8, 4, true},  // split
		{24, 6, 4, false}, // six KV heads do not split four ways or divide four
		{24, 6, 2, true},
		{40, 8, 8, true},
		{30, 6, 4, false}, // the query heads do not divide
		{0, 0, 4, true},   // unknown passes
	} {
		if got := headsSplit(HFConfig{NumAttentionHeads: c.q, NumKeyValueHeads: c.kv}, c.tp); got != c.want {
			t.Errorf("heads %d/%d at TP=%d: %v", c.q, c.kv, c.tp, got)
		}
	}
}

// A seeded config is marked so, until a change makes it the operator's; an
// autosave of the same values does not clear the mark. Entries written before
// the field read as unmarked.
func TestConfigSource(t *testing.T) {
	r := NewRegistry(t.TempDir(), t.TempDir())
	r.Register(&Model{ID: "org/m", LocalPath: t.TempDir()})
	m, _ := r.Get("org/m")
	if m.ConfigSource != "" {
		t.Errorf("a new entry is %q", m.ConfigSource)
	}
	cfg := m.VLLMConfig
	cfg.TensorParallelSize = 4
	r.SetSeededConfig("org/m", cfg)
	if m, _ := r.Get("org/m"); m.ConfigSource != ConfigSeeded || m.VLLMConfig.TensorParallelSize != 4 {
		t.Errorf("seeded: %q %d", m.ConfigSource, m.VLLMConfig.TensorParallelSize)
	}
	r.UpdateConfig("org/m", cfg)
	if m, _ := r.Get("org/m"); m.ConfigSource != ConfigSeeded {
		t.Error("an unchanged save cleared the mark")
	}
	cfg.MaxModelLen = 4096
	r.UpdateConfig("org/m", cfg)
	if m, _ := r.Get("org/m"); m.ConfigSource != "" {
		t.Error("a change kept the mark")
	}
}

// Applying a profile whose config differs clears the seeded mark: found when
// autoconfigure's Save and apply left a seeded model still claiming its
// settings were the feed's.
func TestApplyingAProfileClearsTheSeededMark(t *testing.T) {
	r := NewRegistry(t.TempDir(), t.TempDir())
	r.Register(&Model{ID: "org/m", LocalPath: t.TempDir()})
	m, _ := r.Get("org/m")
	seeded := m.VLLMConfig
	seeded.TensorParallelSize = 4
	r.SetSeededConfig("org/m", seeded)

	r.SaveProfileFrom("org/m", "Same", ConfigProfile{Config: seeded})
	r.ApplyProfile("org/m", "Same")
	if m, _ := r.Get("org/m"); m.ConfigSource != ConfigSeeded {
		t.Error("a profile holding the same config cleared the mark")
	}
	other := seeded
	other.ReasoningParser = "gemma4"
	r.SaveProfileFrom("org/m", AutoconfigProfileName, ConfigProfile{Config: other})
	r.ApplyProfile("org/m", AutoconfigProfileName)
	if m, _ := r.Get("org/m"); m.ConfigSource != "" {
		t.Error("an applied profile left the seeded mark")
	}
}

// Flash-Next on two R9700s: with compute's 188 GB of RAM the experts and the
// table both fit in RAM; with 128 GB only by serving the table from NVMe, as
// its card's two-card recipe does (82 GiB of RAM at runtime); with 64 GB not
// at all. Without the image's NVMe support, 128 GB is not enough.
func TestOffloadWithThePLETableOnNVMe(t *testing.T) {
	plan := func(ram float64, nvme bool) FitPlan {
		in := offloadInput(flashNext(), 2, 31.86)
		in.HostRAMGB, in.PLENVMe = ram, nvme
		return PlanFit(in)
	}
	if p := plan(188, true); !p.Known || !p.All.Offload || process.HasFlag(p.All.Config.ExtraFlags, "--ple-nvme-offload") {
		t.Errorf("188 GB: known=%v flags %q", p.Known, p.All.Config.ExtraFlags)
	}
	p := plan(128, true)
	if !p.Known || !p.All.Offload || !process.HasFlag(p.All.Config.ExtraFlags, "--ple-nvme-offload") ||
		!strings.Contains(p.All.Config.ExtraFlags, "--ple-cache-gb 8") {
		t.Errorf("128 GB with NVMe: known=%v %s flags %q", p.Known, p.Why, p.All.Config.ExtraFlags)
	}
	if p := plan(128, false); p.Known {
		t.Error("128 GB without NVMe support was planned")
	}
	if p := plan(64, true); p.Known {
		t.Error("64 GB was planned")
	}
}

func TestCarryOffloadFlags(t *testing.T) {
	planned := "--enable-expert-offload --ple-nvme-offload --ple-cache-gb 8 --ple-cache-reuse true --other 1"
	got := CarryOffloadFlags("--keep 2", planned)
	for _, want := range []string{"--keep 2", "--enable-expert-offload", "--ple-nvme-offload", "--ple-cache-gb 8", "--ple-cache-reuse true"} {
		if !strings.Contains(got, want) {
			t.Errorf("%q missing %q", got, want)
		}
	}
	if strings.Contains(got, "--other") {
		t.Errorf("a flag that is not offload's was carried: %q", got)
	}
}

// An offload plan's KV cache is fp8: the image reserves for one, and with
// the default cache Flash-Next's two-card start fell short at 262,144.
func TestOffloadPlansAnFP8Cache(t *testing.T) {
	in := offloadInput(flashNext(), 2, 31.86)
	in.Base.KVCacheDtype = "auto"
	p := PlanFit(in)
	if !p.Known || !p.All.Offload || p.All.Config.KVCacheDtype != "fp8" || !hasNote(p.All.Notes, "kv_cache_dtype") {
		t.Errorf("offload plan: known=%v dtype=%q", p.Known, p.All.Config.KVCacheDtype)
	}
}

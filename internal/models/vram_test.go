package models

import (
	"encoding/json"
	"testing"
)

func TestCountAttentionLayers(t *testing.T) {
	raw := func(s string) map[string]json.RawMessage {
		var m map[string]json.RawMessage
		if err := json.Unmarshal([]byte(s), &m); err != nil {
			t.Fatal(err)
		}
		return m
	}

	for _, tc := range []struct {
		name   string
		config string
		total  int
		want   int
	}{
		{
			// Qwen3.8-27B: 16 of 64 layers are full attention. Counting all 64
			// overstates the KV cache fourfold.
			name:   "gdn hybrid via layer_types",
			config: `{"layer_types":["full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention"]}`,
			total:  8,
			want:   2,
		},
		{
			name:   "mamba hybrid",
			config: `{"layer_types":["mamba","mamba","attention","mamba"]}`,
			total:  4,
			want:   1,
		},
		{
			name:   "sliding + global are both attention and both cache",
			config: `{"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"]}`,
			total:  4,
			want:   4,
		},
		{
			// Fallback when only the stride is published.
			name:   "full_attention_interval",
			config: `{"full_attention_interval":4}`,
			total:  64,
			want:   16,
		},
		{
			name:   "dense model has no markers",
			config: `{}`,
			total:  32,
			want:   32,
		},
		{
			name:   "unknown layer count stays unknown",
			config: `{}`,
			total:  0,
			want:   0,
		},
		{
			// An empty list must not be read as "zero attention layers".
			name:   "empty layer_types falls through",
			config: `{"layer_types":[]}`,
			total:  32,
			want:   32,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := countAttentionLayers(raw(tc.config), tc.total); got != tc.want {
				t.Errorf("countAttentionLayers() = %d, want %d", got, tc.want)
			}
		})
	}
}

// The numbers here are the real ones from Qwen3.8-27B-FP8, which is what
// exposed the bug: the panel reported 131072 bytes/token where the true figure
// is a quarter of that.
func TestKVCachePerTokenUsesAttentionLayersOnly(t *testing.T) {
	newModel := func(attnLayers int) *Model {
		return &Model{
			HFConfig: HFConfig{
				NumHiddenLayers:  64,
				HiddenSize:       5120,
				NumKeyValueHeads: 4,
				HeadDim:          256,
				AttentionLayers:  attnLayers,
			},
			Quantization: QuantMeta{Method: "fp8", BytesPerParam: 1},
			VLLMConfig:   VLLMConfig{KVCacheDtype: "fp8"},
		}
	}

	// fp8 KV: 2 (K+V) * 16 layers * 4 heads * 256 dim * 1 byte
	const wantHybrid = 2 * 16 * 4 * 256 * 1
	if got := EstimateVRAM(newModel(16), nil); got.KVCachePerTokenB != wantHybrid {
		t.Errorf("hybrid KV/token = %d, want %d", got.KVCachePerTokenB, wantHybrid)
	}

	// A dense model of the same shape must be unaffected by the change.
	const wantDense = 2 * 64 * 4 * 256 * 1
	if got := EstimateVRAM(newModel(64), nil); got.KVCachePerTokenB != wantDense {
		t.Errorf("dense KV/token = %d, want %d", got.KVCachePerTokenB, wantDense)
	}

	// And a model registered before this field existed (0 = unknown) must fall
	// back to the old behaviour rather than collapsing to zero.
	if got := EstimateVRAM(newModel(0), nil); got.KVCachePerTokenB != wantDense {
		t.Errorf("legacy KV/token = %d, want the dense fallback %d",
			got.KVCachePerTokenB, wantDense)
	}
}

// Real shapes, read from each checkpoint's config.json, checked against the
// size each model is published as. Modelling an expert layer as one dense MLP
// was the original defect, and these are what catch it coming back -- an alias
// regression in ParseHFConfig shows up here as a model a fifth of its size.
func mixtralShape() HFConfig {
	return HFConfig{
		NumHiddenLayers: 32, HiddenSize: 4096, IntermediateSize: 14336,
		NumAttentionHeads: 32, NumKeyValueHeads: 8, HeadDim: 128,
		VocabSize: 32000, NumExperts: 8, NumExpertsPerTok: 2,
		// Mixtral states no moe_intermediate_size; intermediate_size is the
		// expert width.
		MoEIntermediate: 14336,
	}
}

func TestEstimateParamCountMoE(t *testing.T) {
	for _, tc := range []struct {
		name      string
		cfg       HFConfig
		want, tol float64 // billions
	}{
		{
			name: "qwen3-30b-a3b, 128 experts top-8, no shared",
			cfg: HFConfig{
				NumHiddenLayers: 48, HiddenSize: 2048, IntermediateSize: 6144,
				NumAttentionHeads: 32, NumKeyValueHeads: 4, HeadDim: 128,
				VocabSize: 151936, NumExperts: 128, NumExpertsPerTok: 8,
				MoEIntermediate: 768,
			},
			want: 30.5, tol: 0.1,
		},
		{
			// Hybrid: 12 of 48 layers are full attention. The attention term
			// is billed to every layer, which is why the tolerance is wider —
			// it is under 1% of this model.
			name: "qwen3-next-80b, 512 experts top-10, shared 512",
			cfg: HFConfig{
				NumHiddenLayers: 48, HiddenSize: 2048, IntermediateSize: 5120,
				NumAttentionHeads: 16, NumKeyValueHeads: 2, HeadDim: 256,
				VocabSize: 151936, NumExperts: 512, NumExpertsPerTok: 10,
				MoEIntermediate: 512, SharedExpertInter: 512, AttentionLayers: 12,
			},
			want: 79.0, tol: 0.5,
		},
		{
			name: "mixtral-8x7b",
			cfg:  mixtralShape(),
			want: 46.7, tol: 0.1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := float64(estimateParamCount(tc.cfg)) / 1e9
			if diff := got - tc.want; diff > tc.tol || diff < -tc.tol {
				t.Errorf("estimateParamCount = %.2fB, want %.2fB +/- %.2f", got, tc.want, tc.tol)
			}
		})
	}
}

// The defect itself, stated as a test: the same shape without its expert
// fields reads as a small dense model.
func TestMissingExpertFieldsUnderstateTheModel(t *testing.T) {
	moe := estimateParamCount(mixtralShape())

	dense := mixtralShape()
	dense.NumExperts, dense.NumExpertsPerTok, dense.MoEIntermediate = 0, 0, 0
	got := estimateParamCount(dense)

	if got*4 > moe {
		t.Errorf("dense reading %.1fB is not far enough below the true %.1fB to be the known defect",
			float64(got)/1e9, float64(moe)/1e9)
	}
}

func TestActiveParamCount(t *testing.T) {
	// Mixtral routes 2 of 8 experts, and is published as ~12.9B active.
	active := float64(ActiveParamCount(mixtralShape())) / 1e9
	if diff := active - 12.9; diff > 0.1 || diff < -0.1 {
		t.Errorf("active = %.2fB, want 12.9B", active)
	}

	// A dense model has no separate active count to report.
	dense := mixtralShape()
	dense.NumExperts, dense.NumExpertsPerTok, dense.MoEIntermediate = 0, 0, 0
	if got := ActiveParamCount(dense); got != 0 {
		t.Errorf("dense active = %d, want 0", got)
	}
}

// The regression that prompted the rework: a checkpoint serving on four cards
// was labelled too large for the machine it was running on, because the fit
// was judged against an invented single 32 GiB card.
func TestFitDoesNotRejectAModelThatRuns(t *testing.T) {
	est := VRAMEstimate{
		CheckpointGB:        108.5,
		StructuralGB:        69.7,
		DeviceWeightsGB:     69.7,
		DeviceWeightsHighGB: 69.7,
		KVCachePerTokenB:    8046,
		ActivationBaseGB:    0.4,
	}
	cfg := VLLMConfig{TensorParallelSize: 4, GPUMemoryUtilization: 0.90, MaxModelLen: 32768}
	inv := GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}

	fit := Fit(est, cfg, inv)
	if !fit.Known {
		t.Fatal("inventory was supplied but the fit reports none")
	}
	if fit.Configured == nil {
		t.Fatal("TP=4 was configured and the host has four cards, but no option matched")
	}
	if !fit.Configured.ServesConfigured {
		t.Errorf("TP=4 reported as not serving: weights %.1f, load %.1f, budget %.1f, max tokens %d",
			fit.Configured.WeightsPerGPUGB, fit.Configured.LoadGB,
			fit.Configured.BudgetGB, fit.Configured.MaxTokens)
	}
	// ~19.1 GiB per rank is what the engine itself reported for this model.
	if w := fit.Configured.WeightsPerGPUGB; w < 18.1 || w > 20.1 {
		t.Errorf("per-rank weights %.2f GB, want 19.1 +/- 1.0", w)
	}
}

// One rank shards nothing, so it pays no replication surcharge.
func TestSingleRankPaysNoReplicationOverhead(t *testing.T) {
	est := VRAMEstimate{
		CheckpointGB: 23, DeviceWeightsGB: 23, DeviceWeightsHighGB: 23,
		KVCachePerTokenB: 131072,
	}
	cfg := VLLMConfig{TensorParallelSize: 1, GPUMemoryUtilization: 0.85, MaxModelLen: 8192}
	fit := Fit(est, cfg, GPUInventory{Count: 1, PerCardGB: 32, Known: true})

	if fit.Configured == nil {
		t.Fatal("no TP=1 option")
	}
	if w := fit.Configured.WeightsPerGPUGB; w < 22.95 || w > 23.05 {
		t.Errorf("TP=1 weights %.2f GB, want the checkpoint's own 23.0", w)
	}
}

// Without an inventory there is arithmetic to show but nothing to judge it
// against, and inventing a card is what went wrong before.
func TestFitWithoutInventoryWithholdsTheVerdict(t *testing.T) {
	est := VRAMEstimate{CheckpointGB: 23, DeviceWeightsGB: 23, DeviceWeightsHighGB: 23}
	fit := Fit(est, VLLMConfig{TensorParallelSize: 2}, GPUInventory{})

	if fit.Known {
		t.Error("reported a verdict with no cards to judge against")
	}
	if len(fit.Options) != 0 {
		t.Errorf("offered %d tensor-parallel options with no inventory", len(fit.Options))
	}
	if fit.PerGPUGB <= 0 {
		t.Error("withheld the figure as well as the verdict")
	}
}

// Offload is estimated, never read, so the figure is a band. The verdict then
// follows from whether the band agrees with itself: a verdict is offered when
// every point in it lands the same side of the budget, and withheld exactly
// when the bounds straddle.
func TestOffloadBandDecidesTheVerdict(t *testing.T) {
	newModel := func(perCardGB float64) (VRAMEstimate, VLLMConfig, GPUInventory) {
		m := &Model{
			HFConfig:       mixtralShape(),
			Quantization:   QuantMeta{Method: "awq", BytesPerParam: 0.5},
			TotalSizeBytes: 24_700_000_000,
			VLLMConfig: VLLMConfig{
				TensorParallelSize: 1, GPUMemoryUtilization: 0.90, MaxModelLen: 8192,
				Env: "VLLM_PLE_CPU_OFFLOAD=1",
			},
		}
		est := EstimateVRAM(m, m.OwnEnvPairs())
		return est, m.VLLMConfig, GPUInventory{Count: 1, PerCardGB: perCardGB, Known: true}
	}

	est, _, _ := newModel(32)
	if !est.Ranged() {
		t.Fatal("an estimated offload produced a single figure rather than a band")
	}
	if est.HostResidentMinGB >= est.HostResidentGB {
		t.Errorf("host-resident band is inverted or empty: %.1f..%.1f",
			est.HostResidentMinGB, est.HostResidentGB)
	}

	// The verdict itself is the three relationships a band can have to a
	// budget. Built explicitly rather than derived from a checkpoint: this
	// fixture's residual spans barely a gigabyte, so a straddle case drawn
	// from it would hang on a half-gigabyte window of card sizes and would be
	// testing that coincidence rather than the rule.
	banded := VRAMEstimate{
		CheckpointGB:        100,
		StructuralGB:        60,
		HostResidentGB:      40,
		HostResidentMinGB:   0,
		DeviceWeightsGB:     60,
		DeviceWeightsHighGB: 100,
		KVCachePerTokenB:    1024,
		ActivationBaseGB:    0.1,
		Offload:             Offload{PLE: true},
	}
	// At TP=1 that is a load of 61.0 GB optimistic, 101.0 GB pessimistic.
	cfg := VLLMConfig{TensorParallelSize: 1, GPUMemoryUtilization: 1, MaxModelLen: 1024}

	for _, tc := range []struct {
		name                     string
		perCardGB                float64
		wantLoads, wantUncertain bool
	}{
		{"band entirely inside the budget is a firm fit", 120, true, false},
		{"band straddling the budget withholds the verdict", 80, false, true},
		{"band entirely above the budget is a plain refusal", 40, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := Fit(banded, cfg, GPUInventory{Count: 1, PerCardGB: tc.perCardGB, Known: true}).Configured
			if o == nil {
				t.Fatal("no TP=1 option")
			}
			if o.Loads != tc.wantLoads || o.Uncertain != tc.wantUncertain {
				t.Errorf("loads=%v uncertain=%v, want loads=%v uncertain=%v (load %.1f-%.1f against budget %.1f)",
					o.Loads, o.Uncertain, tc.wantLoads, tc.wantUncertain,
					o.LoadGB, o.LoadHighGB, o.BudgetGB)
			}
		})
	}
}

// The checkpoint measured on compute: 108.5 GB on disk, 65.0 GB of structure,
// PLE offload on, served at TP=4 across four 31.86 GiB cards. The engine
// reported 19.07 GiB per rank and the residual method put the table at 43.6
// against 47.68 measured.
//
// Before this, every row read "depends on offload" — a 65–108 GB band on the
// one model where the sizing demonstrably works.
func TestMeasuredCheckpointGetsAVerdict(t *testing.T) {
	est := VRAMEstimate{
		ParamCountBillion:  124.0,
		ActiveParamBillion: 5.57,
		StructuralGB:       64.96,
		CheckpointGB:       108.54,
		KVCachePerTokenB:   12288,
		ActivationBaseGB:   0.25,
		Offload:            Offload{PLE: true},
	}
	// Reproduce what EstimateVRAM derives from those, rather than restating it.
	maxHost, minHost := hostResidentRange(est, HFConfig{})
	est.HostResidentGB, est.HostResidentMinGB = maxHost, minHost
	est.DeviceWeightsGB = est.CheckpointGB - maxHost
	est.DeviceWeightsHighGB = est.CheckpointGB - minHost

	cfg := VLLMConfig{
		TensorParallelSize: 4, GPUMemoryUtilization: 0.97,
		MaxModelLen: 262144, MaxNumBatchedTokens: 8192,
	}
	fit := Fit(est, cfg, GPUInventory{Count: 4, PerCardGB: 31.86, Known: true})

	o := fit.Configured
	if o == nil {
		t.Fatal("no TP=4 option on a four-card host")
	}
	if o.Uncertain || !o.ServesConfigured {
		t.Errorf("TP=4 should be a firm fit; got uncertain=%v serves=%v load %.1f-%.1f budget %.1f",
			o.Uncertain, o.ServesConfigured, o.LoadGB, o.LoadHighGB, o.BudgetGB)
	}
	// The engine reported 19.07 GiB per rank; the band must contain it.
	if o.WeightsPerGPUGB > 19.07 || o.WeightsPerGPUHighGB < 19.07 {
		t.Errorf("per-rank band %.1f-%.1f GB does not contain the measured 19.07",
			o.WeightsPerGPUGB, o.WeightsPerGPUHighGB)
	}

	// The narrow widths are plainly too large — not "depends on offload".
	for _, opt := range fit.Options {
		if opt.TP < 4 && opt.Uncertain {
			t.Errorf("TP=%d: optimistic bound %.1f already exceeds the %.1f budget, so this is a refusal, not uncertainty",
				opt.TP, opt.LoadGB, opt.BudgetGB)
		}
	}
}

// An unrecognised architecture with nothing on disk has no answer, and saying
// so beats working backwards from a fabricated bytes-per-param.
func TestUnknownWithholdsRatherThanGuessing(t *testing.T) {
	m := &Model{Quantization: QuantMeta{BytesPerParam: 0}}
	est := EstimateVRAM(m, nil)

	if !est.Unknown {
		t.Error("produced an estimate from no architecture and no size on disk")
	}
	if est.UnknownWhy == "" {
		t.Error("withheld the estimate without saying why")
	}
	if fit := Fit(est, VLLMConfig{}, GPUInventory{Count: 4, PerCardGB: 32, Known: true}); fit.Label != "—" {
		t.Errorf("fit label = %q, want a dash for an unknown estimate", fit.Label)
	}
}

// FP8 names its own width. A block-quantized FP8 checkpoint leaves bits unset,
// and reading that as "unknown" left the structural figure at zero — which
// makes the offload residual (checkpoint − structural) the entire checkpoint,
// so enabling PLE on such a model would report ~100% of it offloaded.
func TestBlockFP8HasAKnownWidth(t *testing.T) {
	if got := bytesPerParam("fp8", 0, 0); got < 1.0 || got > 1.2 {
		t.Errorf("bytesPerParam(fp8, bits unset) = %v, want ~1 byte per weight", got)
	}
	if got := bytesPerParam("compressed-tensors", 0, 0); got != 0 {
		t.Errorf("a genuinely unknown scheme returned %v; it must stay unknown", got)
	}

	// End to end: the 27B block-FP8 checkpoint on compute reported a
	// structural size of exactly zero.
	m := &Model{
		HFConfig: HFConfig{
			NumHiddenLayers: 64, HiddenSize: 5120, IntermediateSize: 17408,
			NumAttentionHeads: 24, NumKeyValueHeads: 4, HeadDim: 256,
			VocabSize: 248320, AttentionLayers: 16,
		},
		Quantization:   QuantMeta{Method: "fp8", BytesPerParam: bytesPerParam("fp8", 0, 0)},
		TotalSizeBytes: 30_886_627_093,
	}
	if est := EstimateVRAM(m, nil); est.StructuralGB <= 0 {
		t.Errorf("structural size %.2f GB; a residual against this is the whole checkpoint", est.StructuralGB)
	}
}

func TestDetectOffload(t *testing.T) {
	for _, tc := range []struct {
		name    string
		env     []string
		flags   string
		want    Offload
		isSized bool
	}{
		{name: "nothing set"},
		{
			name: "ple via env",
			env:  []string{"VLLM_PLE_CPU_OFFLOAD=1"},
			want: Offload{PLE: true},
		},
		{
			// A later layer must be able to turn it back off, not only on.
			name: "later layer overrides earlier",
			env:  []string{"VLLM_PLE_CPU_OFFLOAD=1", "VLLM_PLE_CPU_OFFLOAD=0"},
			want: Offload{},
		},
		{
			name:  "on-card staging cache",
			flags: "--enable-expert-offload --expert-cache-gb 18",
			want:  Offload{Experts: true, ExpertCacheGB: 18, ExpertCacheSet: true},
		},
		{
			name:  "host ceiling, with an equals sign and a unit",
			flags: "--expert-offload-mem=24GiB",
			want:  Offload{Experts: true, ExpertHostCapGB: 24, ExpertCapSet: true},
		},
		{
			// Both at once, as the checkpoint on compute is configured. These
			// are different quantities — how much may leave the card, and how
			// much is staged on it — and parsing them into one field had the
			// second silently overwrite the first.
			name:  "ceiling and cache are not the same number",
			flags: "--enable-expert-offload --expert-offload-mem 46 --expert-cache-gb 5.5",
			want: Offload{
				Experts:         true,
				ExpertHostCapGB: 46, ExpertCapSet: true,
				ExpertCacheGB: 5.5, ExpertCacheSet: true,
			},
		},
		{
			name:  "expert offload with no size at all",
			flags: "--enable-expert-offload",
			want:  Offload{Experts: true},
		},
		{
			name:  "nvme implies ple and can never be bounded",
			flags: "--ple-nvme-offload",
			want:  Offload{PLE: true, NVMe: true},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := DetectOffload(tc.env, tc.flags)
			if got != tc.want {
				t.Errorf("DetectOffload() = %+v, want %+v", got, tc.want)
			}
			// NVMe is the one offload with no bound at all.
			if got.Bounded() == got.NVMe {
				t.Errorf("Bounded() = %v with NVMe = %v", got.Bounded(), got.NVMe)
			}
		})
	}
}

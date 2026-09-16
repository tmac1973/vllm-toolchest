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

// An offload whose size the configuration does not state must never produce a
// firm verdict, at any width — including where both bounds land the same side
// of the budget, which is the case most likely to be believed.
func TestUnsizedOffloadIsNeverFirm(t *testing.T) {
	m := &Model{
		HFConfig:       mixtralShape(),
		Quantization:   QuantMeta{Method: "awq", BytesPerParam: 0.5},
		TotalSizeBytes: 24_700_000_000,
		VLLMConfig: VLLMConfig{
			TensorParallelSize: 2, GPUMemoryUtilization: 0.90, MaxModelLen: 8192,
			Env: "VLLM_PLE_CPU_OFFLOAD=1",
		},
	}

	est := EstimateVRAM(m, m.OwnEnvPairs())
	if !est.OffloadUnsized {
		t.Fatal("PLE offload was not recorded as unsized")
	}
	if !est.Ranged() {
		t.Error("unsized offload produced a single figure rather than a band")
	}

	fit := Fit(est, m.VLLMConfig, GPUInventory{Count: 4, PerCardGB: 32, Known: true})
	for _, o := range fit.Options {
		if !o.Uncertain {
			t.Errorf("TP=%d gave a firm verdict on an unsized offload", o.TP)
		}
	}
	// It should still recommend something: the cheapest width that works even
	// if nothing moves off the cards at all.
	if fit.RecommendedTP == 0 {
		t.Error("no recommendation offered; the worst-case fallback should supply one")
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
			name:  "expert offload, sized with a separate argument",
			flags: "--enable-expert-offload --expert-cache-gb 18",
			want:  Offload{Experts: true, ExpertGB: 18, ExpertSized: true},
		},
		{
			name:  "expert offload, sized with an equals sign and a unit",
			flags: "--expert-offload-mem=24GiB",
			want:  Offload{Experts: true, ExpertGB: 24, ExpertSized: true},
		},
		{
			name:  "expert offload with no size at all",
			flags: "--enable-expert-offload",
			want:  Offload{Experts: true},
		},
		{
			name:  "nvme implies ple and is never sized",
			flags: "--ple-nvme-offload",
			want:  Offload{PLE: true, NVMe: true},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := DetectOffload(tc.env, tc.flags)
			if got != tc.want {
				t.Errorf("DetectOffload() = %+v, want %+v", got, tc.want)
			}
			// PLE is never sized, so anything carrying it must band.
			if got.PLE && got.Sized() {
				t.Error("PLE reported as sized; it is inferred, never measured")
			}
		})
	}
}

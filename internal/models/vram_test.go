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
	// Not rejected, which is what the regression was. Whether it is a firm fit
	// is another matter: with every band at its pessimistic end at once this
	// configuration needs more than the host has, so the projection says
	// "uncertain" and leaves the verdict to a real start.
	if !fit.Configured.Fits && !fit.Configured.Uncertain {
		t.Errorf("TP=4 rejected: required %.1f (worst %.1f) of %.1f available",
			fit.Configured.RequiredGB, fit.Configured.RequiredHigh,
			fit.Configured.AvailableGB)
	}
	// The engine reported ~19.1 GiB per rank, so ~76.3 GiB across the four.
	if w := fit.Configured.WeightsGB; w < 72.4 || w > 80.4 {
		t.Errorf("total weights %.2f GB, want %.1f (19.1 x 4) +/- 4", w, 19.1*4)
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
	if w := fit.Configured.WeightsGB; w < 22.95 || w > 23.05 {
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
	if fit.Why == "" {
		t.Error("withheld the comparison without saying why")
	}
	// The requirement itself does not depend on the cards, so it survives.
	if est.TotalRequiredGB <= 0 {
		est = EstimateVRAM(&Model{
			HFConfig:       mixtralShape(),
			Quantization:   QuantMeta{Method: "awq", BytesPerParam: 0.5},
			TotalSizeBytes: 24_700_000_000,
			VLLMConfig:     VLLMConfig{TensorParallelSize: 2, MaxModelLen: 8192},
		}, nil)
		if est.TotalRequiredGB <= 0 {
			t.Error("no requirement computed, though it does not depend on the hardware")
		}
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
	// At TP=1 that is a load of 60.2 GB optimistic, 104.1 GB pessimistic.
	cfg := VLLMConfig{TensorParallelSize: 1, GPUMemoryUtilization: 1, MaxModelLen: 1024}

	for _, tc := range []struct {
		name                    string
		perCardGB               float64
		wantFits, wantUncertain bool
	}{
		{"band entirely inside what is available is a firm fit", 120, true, false},
		{"band straddling what is available withholds the verdict", 80, false, true},
		{"band entirely above what is available is a plain refusal", 40, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := Fit(banded, cfg, GPUInventory{Count: 1, PerCardGB: tc.perCardGB, Known: true}).Configured
			if o == nil {
				t.Fatal("no TP=1 option")
			}
			if o.Fits != tc.wantFits || o.Uncertain != tc.wantUncertain {
				t.Errorf("fits=%v uncertain=%v, want fits=%v uncertain=%v (required %.1f, worst %.1f, available %.1f)",
					o.Fits, o.Uncertain, tc.wantFits, tc.wantUncertain,
					o.RequiredGB, o.RequiredHigh, o.AvailableGB)
			}
		})
	}
}

// The checkpoint measured on compute: 108.5 GB on disk, 65.0 GB of structure,
// PLE offload on, served at TP=4 across four 31.86 GiB cards. The engine
// reported 19.07 GiB per rank and the residual method put the table at 43.6
// against 47.68 measured.
//
// A real start of this configuration needed 113.9 GB in all. The projection
// once said 83.3 and called it a firm fit, which was right about the verdict
// for the wrong reason: it had no term for what a rank consumes beyond its
// weights. Counting that, the headline lands within a few percent and the
// pessimistic end passes what the host has -- so it no longer promises.
func TestMeasuredCheckpointIsInsideItsBand(t *testing.T) {
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
	addTotals(&est, HFConfig{}, cfg)
	fit := Fit(est, cfg, GPUInventory{Count: 4, PerCardGB: 31.86, Known: true})

	o := fit.Configured
	if o == nil {
		t.Fatal("no TP=4 option on a four-card host")
	}
	if !o.Fits && !o.Uncertain {
		t.Errorf("TP=4 refused for a model that runs at it: required %.1f (worst %.1f) of %.1f",
			o.RequiredGB, o.RequiredHigh, o.AvailableGB)
	}
	const measuredRequired = 113.9
	if est.TotalRequiredLowGB > measuredRequired || est.TotalRequiredHighGB < measuredRequired {
		t.Errorf("projected %.1f-%.1f GB does not hold the %.1f a real start needed",
			est.TotalRequiredLowGB, est.TotalRequiredHighGB, measuredRequired)
	}
	if diff := est.TotalRequiredGB - measuredRequired; diff > measuredRequired*0.1 || diff < -measuredRequired*0.1 {
		t.Errorf("headline %.1f GB is more than a tenth off the measured %.1f", est.TotalRequiredGB, measuredRequired)
	}

	// The engine reported 19.07 GiB per rank, so 76.28 across the four cards.
	// The band is on the total, because the total is what is reported now.
	const measuredTotal = 19.07 * 4
	lo := weightsTotalGB(est.DeviceWeightsGB, 4)
	hi := weightsTotalGB(est.DeviceWeightsHighGB, 4)
	if lo > measuredTotal || hi < measuredTotal {
		t.Errorf("weights band %.1f-%.1f GB does not contain the measured %.1f", lo, hi, measuredTotal)
	}
	// Containing it is not enough. An earlier version passed while reporting a
	// band whose lower end claimed a 124B model occupies less than it possibly
	// can -- it took the whole residual as offloadable and the configured
	// ceiling as the amount offloaded, both at once. A band wider than a
	// quarter of the figure is not an estimate.
	if spread := hi - lo; spread > measuredTotal*0.25 {
		t.Errorf("weights band %.1f-%.1f GB spans %.1f GB, too wide to be useful", lo, hi, spread)
	}

	// The whole point of the rework: the headline is a total, and it is far
	// larger than any single card's share of it.
	if est.TotalRequiredGB < lo {
		t.Errorf("total required %.1f GB is below the weights alone (%.1f)", est.TotalRequiredGB, lo)
	}
	if est.KVAtContextGB <= 0 {
		t.Error("KV cache at the configured context is not counted in the total")
	}
}

// The other checkpoint measured on compute, and the one whose panel read
// 2.3 GB per card: 117.2 GB on disk, PLE offload plus expert offload with a
// 46 GB ceiling and a 5.5 GB on-card cache. The engine reported 14.15 GiB per
// rank at TP=4.
//
// The ceiling is the trap. Treating it as the amount offloaded assumed the
// maximum permitted offload and the largest plausible table simultaneously,
// which nothing observed does.
func TestExpertOffloadCeilingIsNotTheAmount(t *testing.T) {
	m := &Model{
		HFConfig: HFConfig{
			NumHiddenLayers: 48, HiddenSize: 2560, NumAttentionHeads: 24,
			NumKeyValueHeads: 2, HeadDim: 256, VocabSize: 248320,
			AttentionLayers: 12, NumExperts: 512, NumExpertsPerTok: 10,
			MoEIntermediate: 640, SharedExpertInter: 640,
		},
		Quantization:   QuantMeta{Method: "compressed-tensors", Bits: 4, BytesPerParam: 0.5625},
		TotalSizeBytes: 125_810_393_909,
		VLLMConfig: VLLMConfig{
			TensorParallelSize: 4, GPUMemoryUtilization: 0.92,
			MaxModelLen: 262144, MaxNumBatchedTokens: 4096, KVCacheDtype: "fp8",
			Env:        "VLLM_PLE_CPU_OFFLOAD=1",
			ExtraFlags: "--enable-expert-offload --expert-offload-mem 46 --expert-cache-gb 5.5",
		},
	}

	est := EstimateVRAM(m, m.OwnEnvPairs())
	if est.HostResidentGB >= est.CheckpointGB*0.75 {
		t.Errorf("host-resident %.1f of a %.1f GB checkpoint: the ceiling is being read as the amount",
			est.HostResidentGB, est.CheckpointGB)
	}
	// The cache is staged on the card, so it is not offload.
	if est.DeviceCacheGB != 5.5 {
		t.Errorf("on-card cache = %.1f GB, want 5.5", est.DeviceCacheGB)
	}

	fit := Fit(est, m.VLLMConfig, GPUInventory{Count: 4, PerCardGB: 31.859375, Known: true})
	o := fit.Configured
	if o == nil {
		t.Fatal("no TP=4 option on a four-card host")
	}
	// The engine reported 14.15 GiB per rank, so 56.6 across the four cards.
	const measuredTotal = 14.15 * 4
	lo := weightsTotalGB(est.DeviceWeightsGB, 4)
	hi := weightsTotalGB(est.DeviceWeightsHighGB, 4)
	if lo > measuredTotal || hi < measuredTotal {
		t.Errorf("weights band %.1f-%.1f GB does not contain the measured %.1f", lo, hi, measuredTotal)
	}
	// The figure that prompted this: 2.3 GB per card, i.e. 9.2 GB in total,
	// for a 124B checkpoint.
	if lo < 20 {
		t.Errorf("weights floor %.1f GB is below anything a 124B checkpoint can occupy", lo)
	}
	// It runs at TP=4, so it must not be refused. It is not promised either:
	// four ranks of on-card cache and every band at its worst pass the host.
	if !o.Fits && !o.Uncertain {
		t.Errorf("TP=4 refused for a model that runs at it: required %.1f (worst %.1f) of %.1f available",
			o.RequiredGB, o.RequiredHigh, o.AvailableGB)
	}
	// The on-card expert cache is charged to every rank, so it grows with the
	// width rather than being divided by it.
	if o.OverheadGB < est.DeviceCacheGB*4 {
		t.Errorf("overhead %.1f GB does not carry the 4 x %.1f GB of on-card cache",
			o.OverheadGB, est.DeviceCacheGB)
	}

	// A single card cannot hold this checkpoint under any offload assumption,
	// and available memory now scales with the width, so narrow splits are
	// plain refusals. (Straddling is covered by TestOffloadBandDecidesTheVerdict
	// against a fixture built for it, rather than by whichever real checkpoint
	// happens to land near a boundary.)
	for _, opt := range fit.Options {
		if opt.TP == 1 && (opt.Fits || opt.Uncertain) {
			t.Errorf("TP=1: %.1f GB required against %.1f available is a refusal, not a maybe",
				opt.RequiredGB, opt.AvailableGB)
		}
	}
}

// The defect: a measurement stored its working set as a total across the
// ranks, a projection was charged once whatever the width, and the field's own
// comment described a third behaviour that nothing implemented. One function
// now reads the field, for both sources.
func TestActivationIsCarriedAcrossWidthsOneWay(t *testing.T) {
	near := func(got, want float64) bool { return got > want-1e-9 && got < want+1e-9 }

	// A projection describes the model unsplit, so at one rank it is a single
	// figure, and wider it lies between all-sharded and all-replicated.
	projected := VRAMEstimate{ActivationBaseGB: 0.5}
	for _, tc := range []struct {
		tp        int
		low, high float64
	}{
		{1, 0.5, 0.5},
		{2, 0.5, 1.0},
		{4, 0.5, 2.0},
	} {
		if low, high := activationAt(projected, tc.tp); !near(low, tc.low) || !near(high, tc.high) {
			t.Errorf("projected, TP=%d: %.2f-%.2f GB, want %.2f-%.2f", tc.tp, low, high, tc.low, tc.high)
		}
	}

	// A measurement describes the width it was taken at: 1.46 GiB on each of
	// four ranks. There it is exact. Narrower, the replicated reading is the
	// smaller one; wider, the larger.
	measured := VRAMEstimate{Source: SourceMeasured, MeasuredTP: 4, ActivationBaseGB: 1.46 * 4}
	for _, tc := range []struct {
		tp        int
		low, high float64
	}{
		{1, 1.46, 5.84},
		{2, 2.92, 5.84},
		{4, 5.84, 5.84},
		{8, 5.84, 11.68},
	} {
		if low, high := activationAt(measured, tc.tp); !near(low, tc.low) || !near(high, tc.high) {
			t.Errorf("measured at TP=4, read at TP=%d: %.2f-%.2f GB, want %.2f-%.2f",
				tc.tp, low, high, tc.low, tc.high)
		}
	}

	// The one model measured at two widths. The 27B held 1.27 GiB a rank at
	// TP=4 and 1.90 at TP=2: 5.08 in total against 3.80. Neither reading is
	// right alone -- all-sharded says 5.08 at two ranks, all-replicated 2.54 --
	// and the band between them holds what the engine reported.
	at4 := VRAMEstimate{Source: SourceMeasured, MeasuredTP: 4, ActivationBaseGB: 1.27 * 4}
	if low, high := activationAt(at4, 2); low > 1.90*2 || high < 1.90*2 {
		t.Errorf("27B measured at TP=4, read at TP=2: %.2f-%.2f GB does not hold the 3.80 measured there",
			low, high)
	}
	at2 := VRAMEstimate{Source: SourceMeasured, MeasuredTP: 2, ActivationBaseGB: 1.90 * 2}
	if low, high := activationAt(at2, 4); low > 1.27*4 || high < 1.27*4 {
		t.Errorf("27B measured at TP=2, read at TP=4: %.2f-%.2f GB does not hold the 5.08 measured there",
			low, high)
	}

	// And the requirement carries the band rather than the bare field.
	r := RequiredAt(projected, VLLMConfig{EnforceEager: true}, 4)
	if !near(r.ActivationGB, 0.5) || !near(r.ActivationHighGB, 2.0) {
		t.Errorf("RequiredAt at TP=4: activation %.2f-%.2f GB, want 0.50-2.00",
			r.ActivationGB, r.ActivationHighGB)
	}
	// With no graphs captured, what separates the totals is this band and the
	// per-rank overhead's.
	if want := 1.5 + (r.RankOverheadHighGB - r.RankOverheadGB); !near(r.TotalHighGB-r.TotalGB, want) {
		t.Errorf("the totals differ by %.2f GB, want %.2f", r.TotalHighGB-r.TotalGB, want)
	}
}

// The graph pool was a flat 0.9 GiB per rank. Starts have reported anything
// from 0.07 to 3.74, so it is a band, and the band has to hold every one of
// them or it is the same mistake with a second number.
func TestGraphPoolBandHoldsEveryMeasurement(t *testing.T) {
	for _, perRank := range []float64{0.07, 0.49, 0.93, 1.21, 2.47, 3.74} {
		if perRank < graphPoolLowPerRankGB || perRank > graphPoolHighPerRankGB {
			t.Errorf("a start measured %.2f GiB per rank, outside the band %.2f-%.2f",
				perRank, graphPoolLowPerRankGB, graphPoolHighPerRankGB)
		}
	}

	est := VRAMEstimate{CheckpointGB: 20, DeviceWeightsGB: 20, DeviceWeightsHighGB: 20}

	// Every rank captures its own ladder, at both ends of the band.
	one, four := RequiredAt(est, VLLMConfig{}, 1), RequiredAt(est, VLLMConfig{}, 4)
	if one.GraphsHighGB <= one.GraphsGB {
		t.Fatalf("graph pool at TP=1 is %.2f-%.2f GB, not a band", one.GraphsGB, one.GraphsHighGB)
	}
	if four.GraphsGB != one.GraphsGB*4 || four.GraphsHighGB != one.GraphsHighGB*4 {
		t.Errorf("graph pool at TP=4 is %.2f-%.2f GB, want four times TP=1's %.2f-%.2f",
			four.GraphsGB, four.GraphsHighGB, one.GraphsGB, one.GraphsHighGB)
	}

	// Eager mode captures nothing.
	if eager := RequiredAt(est, VLLMConfig{EnforceEager: true}, 4); eager.GraphsGB != 0 || eager.GraphsHighGB != 0 {
		t.Errorf("eager mode is charged %.2f-%.2f GB of graph pool", eager.GraphsGB, eager.GraphsHighGB)
	}

	// The verdict is drawn at the pessimistic end. 20 GB of weights and a pool
	// of 0.05-4.0 is a load of 20.05-24.0: a card offering 22 straddles it.
	cfg := VLLMConfig{TensorParallelSize: 1, GPUMemoryUtilization: 1}
	for _, tc := range []struct {
		name                    string
		perCardGB               float64
		wantFits, wantUncertain bool
	}{
		{"room for the largest pool measured", 25, true, false},
		{"room for the smallest but not the largest", 22, false, true},
		{"no room for the weights", 19, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := Fit(est, cfg, GPUInventory{Count: 1, PerCardGB: tc.perCardGB, Known: true}).Configured
			if o == nil {
				t.Fatal("no TP=1 option")
			}
			if o.Fits != tc.wantFits || o.Uncertain != tc.wantUncertain {
				t.Errorf("fits=%v uncertain=%v, want fits=%v uncertain=%v (required %.1f, worst %.1f, available %.1f)",
					o.Fits, o.Uncertain, tc.wantFits, tc.wantUncertain,
					o.RequiredGB, o.RequiredHigh, o.AvailableGB)
			}
		})
	}
}

// What a rank of a split consumes beyond its projected weights is a band by
// the width of the weights, across what the engine has reported, charged from
// two ranks up.
func TestRankOverheadBandsHoldEveryMeasurement(t *testing.T) {
	for _, c := range []struct {
		name          string
		bytesPerParam float64
		perRank       float64
	}{
		{"Mistral Small 3.2 BF16", 2, -0.06},
		{"Qwen3.5-35B-A3B FP8", 1.0625, 1.40},
		{"Mistral Small 3.2 FP8", 1.0625, 2.03},
		{"Flash-Next MXFP4", 0.5625, 5.10},
		{"ThinkingCap PARO, width unknown", 0, 6.64},
		{"27B TP=4, older build", 0, 11.67 - 7.05},
		{"27B TP=2, older build", 0, 19.89 - 13.40},
		{"MoE TP=4, older build", 0.5625, 24.57 - 19.07},
	} {
		// An unstated width is charged the top on purpose, so only the
		// ceiling is checked for it.
		low, high := rankOverheadFor(c.bytesPerParam)
		if c.bytesPerParam == 0 {
			low = -1
		}
		if c.perRank > high || c.perRank < low-0.1 {
			t.Errorf("%s measured %.2f GiB a rank, outside its band %.2f-%.2f", c.name, c.perRank, low, high)
		}
	}

	est := VRAMEstimate{CheckpointGB: 20, DeviceWeightsGB: 20, DeviceWeightsHighGB: 20, WeightBytesPerParam: 1}
	if one := RequiredAt(est, VLLMConfig{}, 1); one.RankOverheadGB != 0 || one.RankOverheadHighGB != 0 {
		t.Errorf("a single rank is charged %.2f-%.2f GB for being split", one.RankOverheadGB, one.RankOverheadHighGB)
	}
	low, high := rankOverheadFor(1)
	four := RequiredAt(est, VLLMConfig{}, 4)
	if four.RankOverheadGB != low*4 || four.RankOverheadHighGB != high*4 {
		t.Errorf("four ranks are charged %.2f-%.2f GB, want %.2f-%.2f", four.RankOverheadGB, four.RankOverheadHighGB, low*4, high*4)
	}
}

// The model measured at two widths, end to end: the 27B hybrid with its DFlash
// drafter, as configured on compute. The engine needed 57.1 GB at TP=2 and
// 68.3 at TP=4 -- consumed, activation and graphs on every rank, plus the KV
// cache at the configured context. Before the drafter's cache and the
// per-rank overhead were counted the projection's pessimistic end was 44.8
// and 53.7.
func TestProjectionHoldsTheModelMeasuredAtTwoWidths(t *testing.T) {
	draft := writeModelDir(t, `{
  "architectures": ["DFlash2DraftModel"], "model_type": "qwen3",
  "num_hidden_layers": 5, "hidden_size": 5120, "intermediate_size": 17408,
  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
  "vocab_size": 248320
}`, 2_118_893_271)

	for _, tc := range []struct {
		tp       int
		measured float64
	}{
		{2, 57.1},
		{4, 68.3},
	} {
		m := &Model{
			HFConfig: HFConfig{
				NumHiddenLayers: 64, HiddenSize: 5120, IntermediateSize: 17408,
				NumAttentionHeads: 24, NumKeyValueHeads: 4, HeadDim: 256,
				VocabSize: 248320, AttentionLayers: 16, MaxPositionEmbeddings: 262144,
			},
			TotalSizeBytes: 25_247_995_844,
			VLLMConfig: VLLMConfig{
				TensorParallelSize: tc.tp, GPUMemoryUtilization: 0.92,
				MaxModelLen: 262144, MaxNumBatchedTokens: 8192, MaxNumSeqs: 8,
				KVCacheDtype:      "fp8",
				SpeculativeConfig: `{"method": "dflash", "model": "` + draft + `", "num_speculative_tokens": 7}`,
			},
		}
		est := EstimateVRAM(m, nil)

		// The target's sixteen attention layers and the drafter's five.
		if want := int64(2*16*4*256 + 2*5*8*128); est.KVCachePerTokenB != want {
			t.Errorf("TP=%d: KV/token = %d, want %d with the drafter's own cache", tc.tp, est.KVCachePerTokenB, want)
		}
		if est.TotalRequiredLowGB > tc.measured || est.TotalRequiredHighGB < tc.measured {
			t.Errorf("TP=%d: projected %.1f-%.1f GB does not hold the %.1f the engine needed",
				tc.tp, est.TotalRequiredLowGB, est.TotalRequiredHighGB, tc.measured)
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
	fit := Fit(est, VLLMConfig{}, GPUInventory{Count: 4, PerCardGB: 32, Known: true})
	if !fit.Unknown {
		t.Error("compared an unknown estimate against the hardware anyway")
	}
	if len(fit.Options) != 0 {
		t.Errorf("offered %d tensor-parallel options for a model it cannot size", len(fit.Options))
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

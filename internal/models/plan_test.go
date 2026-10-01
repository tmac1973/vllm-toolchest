package models

import "testing"

// thinkingCap27B is the 27B hybrid measured at two widths on compute, as its
// config reads, with no draft configured.
func thinkingCap27B() *Model {
	return &Model{
		ID: "tcclaviger/ThinkingCap-3.8-27B-PARO5",
		HFConfig: HFConfig{
			NumHiddenLayers: 64, HiddenSize: 5120, IntermediateSize: 17408,
			NumAttentionHeads: 24, NumKeyValueHeads: 4, HeadDim: 256,
			VocabSize: 248320, AttentionLayers: 16, MaxPositionEmbeddings: 262144,
		},
		TotalSizeBytes: 25_247_995_844,
		VLLMConfig: VLLMConfig{
			TensorParallelSize: 1, MaxModelLen: 8192, GPUMemoryUtilization: 0.90,
			MaxNumSeqs: 16, MaxNumBatchedTokens: 8192, KVCacheDtype: "auto",
		},
	}
}

// estimator is the rule the server uses: a measurement when one applies to
// the candidate, the projection otherwise.
func estimator(m *Model) func(VLLMConfig) VRAMEstimate {
	return func(c VLLMConfig) VRAMEstimate {
		cp := *m
		cp.VLLMConfig = c
		if est, ok := MeasuredEstimate(&cp, EngineIdentity{}); ok {
			return est
		}
		est := EstimateVRAM(&cp, cp.OwnEnvPairs())
		est.Source = SourceProjected
		return est
	}
}

var fourCards = GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}

func planFor(m *Model, inv GPUInventory, class ContextClass, cardKV string) FitPlan {
	return PlanFit(PlanInput{
		Model: m, Base: m.VLLMConfig, Inventory: inv, Class: class,
		Defaults:    PlanDefaults{GPUMemoryUtilization: 0.90, MaxNumSeqs: 16},
		CardKVDtype: cardKV, Estimate: estimator(m),
	})
}

// qwen14B is a dense 14B on a single 24 GiB card: the case where the
// context has to give.
func qwen14B() *Model {
	return &Model{
		ID: "Qwen/Qwen3-14B",
		HFConfig: HFConfig{
			NumHiddenLayers: 40, HiddenSize: 5120, IntermediateSize: 17408,
			NumAttentionHeads: 40, NumKeyValueHeads: 8, HeadDim: 128,
			VocabSize: 151936, MaxPositionEmbeddings: 40960,
		},
		TotalSizeBytes: 20_000_000_000,
		VLLMConfig:     VLLMConfig{TensorParallelSize: 1, MaxModelLen: 8192, KVCacheDtype: "auto"},
	}
}

var oneCard = GPUInventory{Count: 1, PerCardGB: 24, Known: true}

func hasNote(notes []ProfileNote, field string) bool {
	for _, n := range notes {
		if n.Field == field {
			return true
		}
	}
	return false
}

// The model that drove the design, never run: all four cards hold its whole
// context with the engine's default dtype, so nothing is changed to fit, and
// the result is labelled a first guess.
func TestPlanAllCardsForAModelThatHasNotRun(t *testing.T) {
	p := planFor(thinkingCap27B(), fourCards, ContextMax, "")
	if !p.Known {
		t.Fatalf("no plan: %s", p.Why)
	}
	if p.All.TP != 4 || p.All.ContextTokens != 262144 || p.All.ContextReduced {
		t.Errorf("all cards: TP=%d context=%d reduced=%v, want 4, 262144, false",
			p.All.TP, p.All.ContextTokens, p.All.ContextReduced)
	}
	if p.All.Config.KVCacheDtype != "auto" || hasNote(p.Notes, "kv_cache_dtype") {
		t.Errorf("fp8 was chosen although the context fits without it: %q", p.All.Config.KVCacheDtype)
	}
	if !p.FirstGuess || p.All.Measured {
		t.Error("a projection was not labelled a first guess")
	}
	// Two cards cannot hold 262,144 tokens at the default dtype, so there is
	// no narrower option that gives what was asked for.
	if p.Narrow != nil {
		t.Errorf("narrow option at TP=%d with context %d; none should reach the target",
			p.Narrow.TP, p.Narrow.ContextTokens)
	}
	for _, f := range []string{"tensor_parallel_size", "max_model_len", "gpu_memory_utilization", "max_num_seqs", "kv_cache_dtype"} {
		if !hasNote(p.All.Notes, f) {
			t.Errorf("no note explaining %s", f)
		}
	}
}

// With the card's fp8, two cards reach the target too, and that is offered
// as the narrow option. The card's dtype is used at both widths, without the
// "chosen to fit" note.
func TestPlanNarrowOptionWithTheCardsKVDtype(t *testing.T) {
	p := planFor(thinkingCap27B(), fourCards, ContextMax, "fp8")
	if p.Narrow == nil {
		t.Fatal("no narrow option")
	}
	if p.Narrow.TP != 2 || p.Narrow.ContextTokens != 262144 {
		t.Errorf("narrow: TP=%d context=%d, want 2 and 262144", p.Narrow.TP, p.Narrow.ContextTokens)
	}
	if p.All.Config.KVCacheDtype != "fp8" || p.Narrow.Config.KVCacheDtype != "fp8" {
		t.Error("the card's KV dtype was not used at both widths")
	}
	if hasNote(p.Notes, "kv_cache_dtype") {
		t.Error("the card's dtype was explained as a fallback")
	}
	if p.All.FullContextRequests <= p.Narrow.FullContextRequests {
		t.Errorf("all cards hold %d full requests and two hold %d; more cards should hold more",
			p.All.FullContextRequests, p.Narrow.FullContextRequests)
	}
}

// One card, a context it cannot hold at the default dtype: fp8 is chosen
// because it gives more, and the context is still reduced.
func TestPlanReducesTheContextOnOneCard(t *testing.T) {
	p := planFor(qwen14B(), oneCard, ContextLong, "")
	if !p.Known {
		t.Fatalf("no plan: %s", p.Why)
	}
	if p.All.TP != 1 || p.Narrow != nil {
		t.Errorf("TP=%d narrow=%v, want 1 and none", p.All.TP, p.Narrow)
	}
	// ContextLong is capped at the model's own 40,960.
	if !p.All.ContextReduced || p.All.ContextTokens >= 40960 || p.All.ContextTokens < minPlanContext {
		t.Errorf("context %d reduced=%v; want below 40960 and reduced", p.All.ContextTokens, p.All.ContextReduced)
	}
	if p.All.ContextTokens%contextStep != 0 {
		t.Errorf("context %d is not a multiple of %d", p.All.ContextTokens, contextStep)
	}
	if p.All.Config.KVCacheDtype != "fp8" || !hasNote(p.Notes, "kv_cache_dtype") {
		t.Errorf("dtype %q; fp8 gives more context here and should be chosen with a note", p.All.Config.KVCacheDtype)
	}

	// A card's own dtype is used even where fp8 would give more.
	p = planFor(qwen14B(), oneCard, ContextLong, "fp8_e5m2")
	if p.All.Config.KVCacheDtype != "fp8_e5m2" || hasNote(p.Notes, "kv_cache_dtype") {
		t.Errorf("card dtype not honoured: %q", p.All.Config.KVCacheDtype)
	}
}

// Refinement plans the width that ran, on the measurement, and changes
// nothing but the context.
func TestPlanFixedWidthUsesTheMeasurement(t *testing.T) {
	m := measuredModel()
	m.VLLMConfig.MaxModelLen = 131072 // the measurement survives: context is not in the fingerprint
	p := PlanFit(PlanInput{
		Model: m, Base: m.VLLMConfig, Inventory: fourCards, Class: ContextMax,
		Defaults: PlanDefaults{GPUMemoryUtilization: 0.5, MaxNumSeqs: 99}, // must be ignored
		FixedTP:  4, Estimate: estimator(m),
	})
	if !p.Known || !p.All.Measured || p.FirstGuess {
		t.Fatalf("known=%v measured=%v firstGuess=%v, want a measured plan (%s)", p.Known, p.All.Measured, p.FirstGuess, p.Why)
	}
	if p.Narrow != nil {
		t.Error("a fixed width offered a narrow option")
	}
	want := m.VLLMConfig
	want.MaxModelLen = p.All.ContextTokens
	if p.All.Config != want {
		t.Errorf("fixed-width plan changed more than the context:\n got %+v\nwant %+v", p.All.Config, want)
	}
	if p.All.ContextTokens != 262144 {
		t.Errorf("context %d; the measured pool holds the model's full 262,144", p.All.ContextTokens)
	}
}

func TestPlanWithoutAnswer(t *testing.T) {
	m := thinkingCap27B()
	if p := planFor(m, GPUInventory{}, ContextMax, ""); p.Known || p.Why == "" {
		t.Error("planned against an unknown inventory")
	}
	unknown := &Model{ID: "x/y"} // nothing to size it by
	if p := planFor(unknown, fourCards, ContextMax, ""); p.Known || p.Why == "" {
		t.Error("planned a model whose size is unknown")
	}
	huge := thinkingCap27B()
	huge.TotalSizeBytes = 500_000_000_000
	if p := planFor(huge, fourCards, ContextMax, ""); p.Known || p.Why != "does not fit on this machine at any width" {
		t.Errorf("a 500 GB checkpoint: known=%v why=%q", p.Known, p.Why)
	}
}

func TestPlanUsesTheDefaultsAndClearsAPinnedPool(t *testing.T) {
	m := thinkingCap27B()
	m.VLLMConfig.KVCacheMemory = 1 << 30
	m.VLLMConfig.MaxNumSeqs = 3
	p := planFor(m, fourCards, ContextMedium, "")
	for _, w := range []*WidthPlan{&p.All, p.Narrow} {
		if w == nil {
			continue
		}
		if w.Config.KVCacheMemory != 0 || w.Config.MaxNumSeqs != 16 || w.Config.GPUMemoryUtilization != 0.90 {
			t.Errorf("TP=%d: pool=%d seqs=%d util=%.2f; want 0, 16, 0.90",
				w.TP, w.Config.KVCacheMemory, w.Config.MaxNumSeqs, w.Config.GPUMemoryUtilization)
		}
	}
	// Medium is 32,768, which a single card holds for this checkpoint, only
	// just: the narrow option is the fewest cards that reach the target.
	if p.Narrow == nil || p.Narrow.TP != 1 || p.Narrow.ContextTokens != 32768 {
		t.Errorf("narrow = %+v, want one card holding 32768", p.Narrow)
	}
}

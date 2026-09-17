package models

import (
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
)

// A model that has run, with the figures a real start reported.
func measuredModel() *Model {
	m := &Model{
		ID:             "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ",
		TotalSizeBytes: 116_549_178_737,
		Quantization:   QuantMeta{Method: "compressed-tensors", BytesPerParam: 0.5625},
		HFConfig: HFConfig{
			NumHiddenLayers: 48, HiddenSize: 2560, NumAttentionHeads: 24,
			NumKeyValueHeads: 2, HeadDim: 256, VocabSize: 248320,
			AttentionLayers: 12, NumExperts: 512, NumExpertsPerTok: 10,
			MoEIntermediate: 640, SharedExpertInter: 640,
			MaxPositionEmbeddings: 262144,
		},
		VLLMConfig: VLLMConfig{
			TensorParallelSize: 4, MaxModelLen: 262144, KVCacheDtype: "fp8",
			MaxNumBatchedTokens: 8192, GPUMemoryUtilization: 0.97,
			Env: "VLLM_PLE_CPU_OFFLOAD=1",
		},
	}
	m.Measured = &RunMeasurement{
		At:            time.Date(2026, 9, 17, 19, 15, 0, 0, time.UTC),
		TP:            4,
		ContextTokens: 262144,
		Fingerprint:   MeasurementFingerprint(m),
		Engine: advice.Measurements{
			WeightsPerRankGB: 19.07,
			ConsumedGB:       24.57,
			NonTorchGB:       2.36,
			PeakActivationGB: 1.46,
			GraphPoolGB:      0.49,
			KVCacheGB:        4.87,
			KVCacheTokens:    651859,
			PLEOffloadGB:     38.8,
		},
	}
	return m
}

func TestMeasuredEstimateUsesTheEngineNotTheFormula(t *testing.T) {
	m := measuredModel()
	est, ok := MeasuredEstimate(m)
	if !ok {
		t.Fatal("a model that has run produced no measured estimate")
	}
	if est.Source != SourceMeasured {
		t.Errorf("source = %q, want measured", est.Source)
	}

	// The figure the whole approach turns on. Counting attention layers gives
	// 12,288; the engine's own allocation says ~32,087.
	if est.KVCachePerTokenB < 32000 || est.KVCachePerTokenB > 32200 {
		t.Errorf("KV bytes/token = %d, want ~32087 (the formula's answer, 12288, is what this replaces)",
			est.KVCachePerTokenB)
	}

	// Offload is measured rather than inferred from an unaccounted residual.
	if est.HostResidentGB != 38.8 {
		t.Errorf("host-resident = %.1f, want the measured 38.8", est.HostResidentGB)
	}
	if est.Ranged() {
		t.Error("a measured figure is not a band")
	}

	// The total is what the run would need, not what the formula projects.
	// The projected path said 83.3 GB for this configuration.
	if est.TotalRequiredGB < 110 || est.TotalRequiredGB > 118 {
		t.Errorf("total = %.1f GB, want ~114", est.TotalRequiredGB)
	}

	// Params still come from the checkpoint's shape: they are a property of
	// the file, not of a run.
	if est.ParamCountBillion < 120 || est.ParamCountBillion > 128 {
		t.Errorf("params = %.1fB, want ~124", est.ParamCountBillion)
	}
}

func TestMeasuredEstimateRescalesOnlyTheCache(t *testing.T) {
	m := measuredModel()
	full, _ := MeasuredEstimate(m)

	m.VLLMConfig.MaxModelLen = 131072
	half, ok := MeasuredEstimate(m)
	if !ok {
		t.Fatal("halving the context retired the measurement; it scales exactly")
	}

	if half.WeightsTotalGB != full.WeightsTotalGB || half.GraphPoolGB != full.GraphPoolGB {
		t.Error("something other than the cache moved with the context length")
	}
	if diff := (full.TotalRequiredGB - half.TotalRequiredGB) - full.KVAtContextGB/2; diff > 0.05 || diff < -0.05 {
		t.Errorf("halving the context moved %.2f GB, want %.2f -- half the cache and nothing else",
			full.TotalRequiredGB-half.TotalRequiredGB, full.KVAtContextGB/2)
	}
}

func TestMeasuredEstimateStandsDownWhenItNoLongerApplies(t *testing.T) {
	for _, tc := range []struct {
		name  string
		apply func(*Model)
	}{
		{"never run", func(m *Model) { m.Measured = nil }},
		{"the run died partway", func(m *Model) { m.Measured.Engine.KVCacheTokens = 0 }},
		{"the split changed", func(m *Model) { m.VLLMConfig.TensorParallelSize = 2 }},
		{"the cache dtype changed", func(m *Model) { m.VLLMConfig.KVCacheDtype = "auto" }},
		{"offload was turned off", func(m *Model) { m.VLLMConfig.Env = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := measuredModel()
			tc.apply(m)
			if _, ok := MeasuredEstimate(m); ok {
				t.Error("a measurement was reused for a configuration it never described")
			}
		})
	}
}

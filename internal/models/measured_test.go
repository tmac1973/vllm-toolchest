package models

import (
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
)

// The figures are from a real start of Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ on
// four R9700s, 2026-09-17. Nothing here is invented; that is the whole point of
// the approach these types exist to serve.
func liveRun() RunMeasurement {
	return RunMeasurement{
		At:            time.Date(2026, 9, 17, 19, 15, 0, 0, time.UTC),
		TP:            4,
		ContextTokens: 262144,
		Fingerprint:   "deadbeefdeadbeef",
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
}

func TestKVBytesPerTokenFromARealRun(t *testing.T) {
	// Counted from the architecture this came out 12,288. Two real starts a
	// day apart measured 32,126 and 32,087 -- a tenth of a percent, which is
	// what makes it worth carrying forward to a different context length.
	got := liveRun().KVBytesPerToken()
	if got < 32000 || got > 32200 {
		t.Errorf("KV bytes/token = %.0f, want ~32087", got)
	}
}

func TestTotalScalesWithContextAndNothingElseDoes(t *testing.T) {
	r := liveRun()

	full := r.TotalRequiredGB(262144)
	half := r.TotalRequiredGB(131072)

	// Load-and-serve-one-request at the configured context. The projected
	// estimate said 83.3 for this configuration.
	if full < 110 || full > 118 {
		t.Errorf("total at 262144 = %.1f GB, want ~114", full)
	}

	// Only the KV term moves. Halving the context halves it and leaves the
	// weights, allocator overhead, working set and graphs exactly where they
	// were -- which is why one run can answer a question about a context
	// length it never ran at.
	kvFull := r.KVBytesPerToken() * 262144 / (1024 * 1024 * 1024)
	if diff := (full - half) - kvFull/2; diff > 0.05 || diff < -0.05 {
		t.Errorf("halving the context moved %.2f GB, want %.2f (half the KV term)",
			full-half, kvFull/2)
	}
}

// A start that died partway has a weights figure and nothing else. Basing an
// estimate on that would be worse than having none.
func TestIncompleteRunsAreNotUsable(t *testing.T) {
	partial := RunMeasurement{TP: 4, Engine: advice.Measurements{WeightsPerRankGB: 19.07}}
	if partial.Complete() {
		t.Error("a run that only got as far as loading weights reports itself complete")
	}
	if !liveRun().Complete() {
		t.Error("a full run reports itself incomplete")
	}
}

// The fingerprint errs strict on purpose: reporting a measurement stale when it
// might still hold is visible and mildly annoying, while silently reusing one
// that no longer applies is the failure this whole approach exists to escape.
func TestFingerprintInvalidation(t *testing.T) {
	base := func() *Model {
		return &Model{
			TotalSizeBytes: 116_549_178_737,
			Quantization:   QuantMeta{Method: "compressed-tensors", BytesPerParam: 0.5625},
			VLLMConfig: VLLMConfig{
				TensorParallelSize: 4, KVCacheDtype: "fp8", MaxModelLen: 262144,
				MaxNumBatchedTokens: 8192, Env: "VLLM_PLE_CPU_OFFLOAD=1",
			},
		}
	}
	original := MeasurementFingerprint(base())

	t.Run("context length varies freely", func(t *testing.T) {
		m := base()
		m.VLLMConfig.MaxModelLen = 131072
		if MeasurementFingerprint(m) != original {
			t.Error("changing max_model_len invalidated the measurement; the KV term scales with it exactly")
		}
	})

	for _, tc := range []struct {
		name  string
		apply func(*Model)
	}{
		{"tensor-parallel width", func(m *Model) { m.VLLMConfig.TensorParallelSize = 2 }},
		{"kv cache dtype", func(m *Model) { m.VLLMConfig.KVCacheDtype = "auto" }},
		{"batch size, which activation is a peak over", func(m *Model) { m.VLLMConfig.MaxNumBatchedTokens = 2048 }},
		{"offload environment", func(m *Model) { m.VLLMConfig.Env = "" }},
		{"extra flags", func(m *Model) { m.VLLMConfig.ExtraFlags = "--enable-expert-offload" }},
		{"eager mode, which skips the graph pool", func(m *Model) { m.VLLMConfig.EnforceEager = true }},
		{"quantization", func(m *Model) { m.Quantization.BytesPerParam = 1.0 }},
		{"the checkpoint itself", func(m *Model) { m.TotalSizeBytes = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := base()
			tc.apply(m)
			if MeasurementFingerprint(m) == original {
				t.Errorf("changing %s left the fingerprint unchanged", tc.name)
			}
		})
	}
}

func TestAppliesNeedsBothCompletenessAndAMatch(t *testing.T) {
	m := &Model{
		TotalSizeBytes: 116_549_178_737,
		Quantization:   QuantMeta{Method: "compressed-tensors", BytesPerParam: 0.5625},
		VLLMConfig: VLLMConfig{
			TensorParallelSize: 4, KVCacheDtype: "fp8", MaxModelLen: 262144,
			MaxNumBatchedTokens: 8192, Env: "VLLM_PLE_CPU_OFFLOAD=1",
		},
	}

	r := liveRun()
	r.Fingerprint = MeasurementFingerprint(m)
	if !r.Applies(m) {
		t.Error("a complete measurement of this very configuration does not apply to it")
	}

	m.VLLMConfig.TensorParallelSize = 2
	if r.Applies(m) {
		t.Error("a measurement taken at TP=4 still claims to describe TP=2")
	}
}

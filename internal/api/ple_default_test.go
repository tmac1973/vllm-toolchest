package api

import (
	"slices"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

// The 125B MoE as autoconfigure found it on compute: a default config, no
// offload variable, on an image that offloads its PLE table anyway. Without
// knowing that, the planner said it did not fit at any width.
func moeWithPLE() *models.Model {
	return &models.Model{
		ID:             "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ",
		TotalSizeBytes: 116_549_178_737,
		Quantization:   models.QuantMeta{Method: "compressed-tensors", BytesPerParam: 0.5625},
		HFConfig: models.HFConfig{
			NumHiddenLayers: 48, HiddenSize: 2560, NumAttentionHeads: 24,
			NumKeyValueHeads: 2, HeadDim: 256, VocabSize: 248320,
			AttentionLayers: 12, NumExperts: 512, NumExpertsPerTok: 10,
			MoEIntermediate: 640, SharedExpertInter: 640,
			MaxPositionEmbeddings: 262144, PLELayers: 1,
		},
		VLLMConfig: models.VLLMConfig{TensorParallelSize: 1, MaxModelLen: 8192, MaxNumSeqs: 16},
	}
}

func TestAnImageThatOffloadsPLEByDefault(t *testing.T) {
	s := newGoldenServer(t, vllmenv.Env{Variant: "rdna4-clav"})
	s.gpuInvOverride = &models.GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}
	m := moeWithPLE()

	if !slices.Contains(s.configuredEnvPairs(m), "VLLM_PLE_CPU_OFFLOAD=1") {
		t.Fatal("a PLE checkpoint on this image was not given the offload variable")
	}
	plan := models.PlanFit(s.planInput(m, m.VLLMConfig, models.ContextMax, "fp8"))
	if !plan.Known || plan.All.TP != 4 || plan.All.ContextTokens != 262144 {
		t.Errorf("plan: known=%v why=%q TP=%d context=%d", plan.Known, plan.Why, plan.All.TP, plan.All.ContextTokens)
	}

	// A model can still turn it off.
	off := moeWithPLE()
	off.VLLMConfig.Env = "VLLM_PLE_CPU_OFFLOAD=0"
	if o := models.DetectOffload(s.configuredEnvPairs(off), ""); o.PLE {
		t.Error("the model's own VLLM_PLE_CPU_OFFLOAD=0 did not win")
	}

	// A checkpoint without a PLE table, or another image, is untouched.
	plain := moeWithPLE()
	plain.HFConfig.PLELayers = 0
	if slices.Contains(s.configuredEnvPairs(plain), "VLLM_PLE_CPU_OFFLOAD=1") {
		t.Error("a checkpoint with no PLE table was given the offload variable")
	}
	other := newGoldenServer(t, vllmenv.Env{Variant: "rocm"})
	if slices.Contains(other.configuredEnvPairs(m), "VLLM_PLE_CPU_OFFLOAD=1") {
		t.Error("an image that does not offload by default was assumed to")
	}
}

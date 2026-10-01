package recommend

import (
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// The seed is the planner's five hardware fields with a pinned pool cleared,
// and nothing else: every other field stays as registration defaulted it.
func TestSeedConfig(t *testing.T) {
	base := models.VLLMConfig{
		Dtype: "auto", MaxModelLen: 8192, TensorParallelSize: 1, GPUMemoryUtilization: 0.9, MaxNumSeqs: 16,
		KVCacheDtype: "auto", KVCacheMemory: 4 << 30, ToolCallParser: "hermes", EnableAutoToolChoice: true,
		ExtraFlags: "--keep 1", SpeculativeConfig: `{"method": "mtp", "num_speculative_tokens": 3}`,
	}
	planned := models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 262144, KVCacheDtype: "fp8",
		GPUMemoryUtilization: 0.85, MaxNumSeqs: 8, ToolCallParser: "something-else", Dtype: "bfloat16"}
	got := SeedConfig(base, planned)

	want := base
	want.TensorParallelSize, want.MaxModelLen, want.KVCacheDtype = 4, 262144, "fp8"
	want.GPUMemoryUtilization, want.MaxNumSeqs, want.KVCacheMemory = 0.85, 8, 0
	if got != want {
		t.Errorf("seeded\n%+v\nwant\n%+v", got, want)
	}

	// An offload plan brings its flag, and leaves MTP out.
	planned.ExtraFlags = models.ExpertOffloadFlag
	got = SeedConfig(base, planned)
	if !process.HasFlag(got.ExtraFlags, models.ExpertOffloadFlag) || !process.HasFlag(got.ExtraFlags, "--keep") || got.SpeculativeConfig != "" {
		t.Errorf("offload seed: %+v", got)
	}
}

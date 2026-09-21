package api

import (
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Most suggestions are not settings. Acting on them would be worse than
// ignoring them, so applicability is declared per rule rather than inferred
// from an item merely having a value.
func TestOnlyRealSettingsAreApplicable(t *testing.T) {
	for _, tc := range []struct {
		name       string
		line       string
		applicable bool
		ours       bool
	}{
		{
			name:       "the engine states the cache's real ceiling",
			line:       `ValueError: The model's max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache (47328).`,
			applicable: true,
		},
		{
			name:       "the engine hands over its own pool size",
			line:       `Replace gpu_memory_utilization config with --kv-cache-memory=4545545954 (4.23 GiB) to fit into requested memory.`,
			applicable: true,
		},
		{
			// A value worth applying, but arithmetic done here rather than a
			// figure the engine named.
			name:       "the fraction that would clear a shortfall is ours",
			line:       `ValueError: Free memory on device cuda:0 (27.28/31.86 GiB) on startup is less than desired GPU memory utilization (0.97, 30.9 GiB). Decrease GPU memory utilization.`,
			applicable: true,
			ours:       true,
		},
		{
			// The flag named is what went wrong, not what to set.
			name: "an unrecognised flag is the problem, not the fix",
			line: `vllm: error: unrecognized arguments: --enable-expert-offload`,
		},
		{
			// Reports back what is already configured.
			name: "chunked prefill echoes the current setting",
			line: `INFO Chunked prefill is enabled with max_num_batched_tokens=8192.`,
		},
		{
			// An equivalence figure; applying it chases an accounting artefact.
			name: "the graph-accounting note is not a setting",
			line: `CUDA graph memory profiling is enabled. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9826.`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			it := advice.Scan(tc.line)
			if it == nil {
				t.Fatalf("matched nothing:\n  %.110s", tc.line)
			}
			if it.Applicable != tc.applicable {
				t.Errorf("applicable = %v, want %v (suggested %q for %s)",
					it.Applicable, tc.applicable, it.Suggested, it.Field)
			}
			if it.Ours != tc.ours {
				t.Errorf("ours = %v, want %v", it.Ours, tc.ours)
			}
			if it.Applicable && it.Suggested == "" {
				t.Error("marked applicable with nothing to apply")
			}
		})
	}
}

// Applying changes the named field and leaves the rest of the configuration
// exactly as it was.
func TestApplyingTouchesOneFieldOnly(t *testing.T) {
	base := models.VLLMConfig{
		MaxModelLen: 262144, TensorParallelSize: 4, GPUMemoryUtilization: 0.97,
		KVCacheDtype: "fp8", MaxNumBatchedTokens: 8192,
		Env: "VLLM_PLE_CPU_OFFLOAD=1", ExtraFlags: "--override-generation-config '{}'",
	}

	got, err := applyToConfig(base, "kv_cache_memory", "4545545954")
	if err != nil {
		t.Fatalf("refused a value the engine reported: %v", err)
	}
	if got.KVCacheMemory != 4545545954 {
		t.Errorf("kv_cache_memory = %d, want 4545545954", got.KVCacheMemory)
	}

	// Everything else must survive untouched.
	want := base
	want.KVCacheMemory = got.KVCacheMemory
	if got != want {
		t.Error("applying one suggestion changed something else in the config")
	}
}

func TestApplyRefusesWhatItCannotWrite(t *testing.T) {
	base := models.VLLMConfig{MaxModelLen: 262144, GPUMemoryUtilization: 0.97}

	for _, tc := range []struct {
		name, field, value string
	}{
		{"a field no rule can name", "tensor_parallel_size", "2"},
		{"the offending flag from an unrecognised-arguments item", "extra_flags", "--enable-expert-offload"},
		{"a context length that is not a number", "max_model_len", "lots"},
		{"a negative context length", "max_model_len", "-1"},
		{"a utilization above one", "gpu_memory_utilization", "1.4"},
		{"a utilization of zero", "gpu_memory_utilization", "0"},
		{"a pool size that is not bytes", "kv_cache_memory", "4.23 GiB"},
		{"nothing at all", "max_model_len", "   "},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyToConfig(base, tc.field, tc.value)
			if err == nil {
				t.Errorf("accepted %s=%q", tc.field, tc.value)
			}
			if got != base {
				t.Error("refused the value but changed the config anyway")
			}
		})
	}
}

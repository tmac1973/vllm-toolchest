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
	if got := EstimateVRAM(newModel(16)); got.KVCachePerTokenB != wantHybrid {
		t.Errorf("hybrid KV/token = %d, want %d", got.KVCachePerTokenB, wantHybrid)
	}

	// A dense model of the same shape must be unaffected by the change.
	const wantDense = 2 * 64 * 4 * 256 * 1
	if got := EstimateVRAM(newModel(64)); got.KVCachePerTokenB != wantDense {
		t.Errorf("dense KV/token = %d, want %d", got.KVCachePerTokenB, wantDense)
	}

	// And a model registered before this field existed (0 = unknown) must fall
	// back to the old behaviour rather than collapsing to zero.
	if got := EstimateVRAM(newModel(0)); got.KVCachePerTokenB != wantDense {
		t.Errorf("legacy KV/token = %d, want the dense fallback %d",
			got.KVCachePerTokenB, wantDense)
	}
}

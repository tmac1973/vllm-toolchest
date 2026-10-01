package models

import (
	"strings"
	"testing"
)

// Gemma 4 31B: fifty sliding layers at a 1,024-token window, ten global ones
// with their own shape. Counted as sixty full layers it came to 983,040 bytes
// a token, and was planned at a third of its context with an fp8 cache.
func TestSlidingWindowLayersStopAtTheWindow(t *testing.T) {
	types := strings.Repeat(`"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",`, 10)
	dir := writeConfig(t, `{"architectures":["Gemma4ForConditionalGeneration"],"text_config":{
		"num_hidden_layers":60,"hidden_size":5376,"num_attention_heads":32,"num_key_value_heads":16,"head_dim":256,
		"num_global_key_value_heads":4,"global_head_dim":512,"attention_k_eq_v":true,"sliding_window":1024,
		"max_position_embeddings":262144,"layer_types":[`+strings.TrimSuffix(types, ",")+`]}}`)
	cfg := ParseHFConfig(dir)
	if cfg.AttentionLayers != 10 || cfg.SlidingLayers != 50 || cfg.SlidingWindow != 1024 || cfg.GlobalKVHeads != 4 || cfg.GlobalHeadDim != 512 {
		t.Fatalf("parsed %+v", cfg)
	}

	// Ten global layers, 4 heads of 512, keys and values: 81,920 a token.
	// Fifty sliding layers, 16 heads of 256, keys and values, for 1,024 of
	// the 262,144 tokens: 819,200 / 256 = 3,200 more.
	if got := kvCachePerToken(cfg, VLLMConfig{MaxModelLen: 262144}); got != 81920+3200 {
		t.Errorf("at 262,144: %d, want %d", got, 81920+3200)
	}
	// Within the window every sliding layer holds the whole context.
	if got := kvCachePerToken(cfg, VLLMConfig{MaxModelLen: 1024}); got != 81920+819200 {
		t.Errorf("at 1,024: %d", got)
	}

	// A "sliding" layer with no window to cap it at is a full one.
	dir = writeConfig(t, `{"num_hidden_layers":2,"hidden_size":64,"num_attention_heads":4,"layer_types":["sliding_attention","full_attention"]}`)
	if cfg := ParseHFConfig(dir); cfg.AttentionLayers != 2 || cfg.SlidingLayers != 0 {
		t.Errorf("without a window: %+v", cfg)
	}
}

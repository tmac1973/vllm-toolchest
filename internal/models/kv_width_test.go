package models

import (
	"strconv"
	"testing"
)

func TestKVReplication(t *testing.T) {
	for _, c := range []struct {
		heads, tp int
		want      float64
	}{
		{8, 1, 1}, {8, 4, 1}, {8, 8, 1}, {2, 2, 1}, {2, 4, 2}, {1, 4, 4}, {4, 8, 2}, {0, 4, 1},
	} {
		if got := kvReplication(c.heads, c.tp); got != c.want {
			t.Errorf("kvReplication(%d, %d) = %g, want %g", c.heads, c.tp, got, c.want)
		}
	}
}

// qwen35MoE is Qwen3.5-35B-A3B's attention shape: ten full-attention layers
// of forty, two KV heads, and one MTP layer.
func qwen35MoE() *Model {
	return &Model{
		ID: "Qwen/Qwen3.5-35B-A3B-FP8",
		HFConfig: HFConfig{
			NumHiddenLayers: 40, AttentionLayers: 10, HiddenSize: 2048, NumAttentionHeads: 16,
			NumKeyValueHeads: 2, HeadDim: 256, MaxPositionEmbeddings: 262144, MTPLayers: 1,
			VocabSize: 248320, NumExperts: 256, NumExpertsPerTok: 8, MoEIntermediate: 512,
		},
		VLLMConfig: VLLMConfig{TensorParallelSize: 4, MaxModelLen: 262144},
	}
}

// What the engine allocated on compute: 17.92 GiB a card at four cards for
// 1,666,259 tokens, 46,190 bytes a token across the cards. Ten layers at two
// heads were counted as 20,480; the heads are held twice at four cards and
// the MTP layer has its own cache, which comes to 45,056.
func TestKVPerTokenCountsCopiedHeadsAndTheMTPLayer(t *testing.T) {
	m := qwen35MoE()
	if got := EstimateVRAM(m, nil).KVCachePerTokenB; got != 40960 {
		t.Errorf("without MTP: %d, want 40960", got)
	}
	m.VLLMConfig.SpeculativeConfig = `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`
	est := EstimateVRAM(m, nil)
	if est.KVCachePerTokenB != 45056 {
		t.Errorf("with MTP: %d, want 45056", est.KVCachePerTokenB)
	}
	// At two cards the heads are not copied: half as much a token.
	if s := est.KVScale(2); s != 0.5 {
		t.Errorf("KVScale(2) = %g", s)
	}
	if r2, r4 := RequiredAt(est, m.VLLMConfig, 2), RequiredAt(est, m.VLLMConfig, 4); r2.KVGB*2 != r4.KVGB {
		t.Errorf("KV at two cards %.2f, at four %.2f", r2.KVGB, r4.KVGB)
	}

	// A drafter of its own is not MTP, and a model without MTP layers adds none.
	m.HFConfig.MTPLayers = 0
	if got := EstimateVRAM(m, nil).KVCachePerTokenB; got != 40960 {
		t.Errorf("no MTP layers: %d", got)
	}
}

// An estimate written before the fields existed carries its figures as they
// were, at every width.
func TestKVScaleWithoutHeads(t *testing.T) {
	if s := (VRAMEstimate{KVWidth: 4}).KVScale(8); s != 1 {
		t.Errorf("KVScale = %g", s)
	}
}

func TestParseMTPLayers(t *testing.T) {
	for field, want := range map[string]int{"mtp_num_hidden_layers": 1, "num_nextn_predict_layers": 3} {
		dir := writeConfig(t, `{"architectures":["X"],"num_hidden_layers":4,"hidden_size":64,"num_attention_heads":4,"`+field+`":`+strconv.Itoa(want)+`}`)
		if got := ParseHFConfig(dir).MTPLayers; got != want {
			t.Errorf("%s: %d, want %d", field, got, want)
		}
	}
}

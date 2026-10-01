package huggingface

// Safetensors is the Hub's per-dtype count for a repository, from
// expand[]=safetensors, kept on search results for the recommendation feed to
// rank by. It is not a size, and nothing here makes one from it.
//
// What the counts are depends on whether the Hub recognises the quantization.
// For a format it knows, they are parameters, labelled with the dtype that
// stores them packed: Qwen3.5-35B-A3B-GPTQ-Int4 reports "I32: 32.2B", which is
// 32.2B 4-bit parameters in 4B int32s -- about 16 GB, not 129. For one it does
// not, they are stored elements: tcclaviger's MXFP416 Gemma reports "U8:
// 10.4B", which is 10.4B bytes. Multiplying by the dtype's width is exact only
// for unquantized and FP8 repositories, and was wrong by eight times on GPTQ.
// The exact size of any repository is its weight files', from the file tree:
// see ModelDetail.WeightsBytes.
type Safetensors struct {
	Parameters map[string]int64 `json:"parameters,omitempty"`
	Total      int64            `json:"total,omitempty"`
}

package tuning

import (
	"fmt"
	"sort"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Shape is a single (N, K) weight matmul for the block-FP8 kernel.
// Tuning produces one JSON file per Shape per (block_n, block_k).
type Shape struct {
	N int `json:"n"`
	K int `json:"k"`
}

// DeriveShapes computes the unique block-FP8 GEMM shapes a model's transformer
// decoder layers will use under a given tensor-parallel factor.
//
// For each layer the relevant matmuls are:
//
//	Q proj:    (heads * head_dim / tp,    hidden)
//	K, V proj: (kv_heads * head_dim / tp, hidden)   (only if kv_heads % tp == 0)
//	O proj:    (hidden,                   heads * head_dim / tp)
//	Gate, Up:  (intermediate / tp,        hidden)
//	Down:      (hidden,                   intermediate / tp)
//
// We only emit shapes where both N and K are divisible by the kernel's block
// size (default 128) — the FP8 block kernel only fires on those, and the
// fallback path (used otherwise) doesn't need tuning.
func DeriveShapes(cfg models.HFConfig, tp int, blockN, blockK int) []Shape {
	if cfg.HiddenSize == 0 {
		return nil
	}

	h := cfg.HiddenSize
	inter := cfg.IntermediateSize
	if inter == 0 {
		inter = 4 * h
	}
	heads := cfg.NumAttentionHeads
	kvHeads := cfg.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = heads
	}
	headDim := cfg.HeadDim
	if headDim == 0 && heads > 0 {
		headDim = h / heads
	}

	if tp < 1 {
		tp = 1
	}

	pairs := map[Shape]struct{}{}
	add := func(n, k int) {
		if n <= 0 || k <= 0 {
			return
		}
		if n%blockN != 0 || k%blockK != 0 {
			return
		}
		pairs[Shape{N: n, K: k}] = struct{}{}
	}

	if heads > 0 && headDim > 0 {
		add(heads*headDim/tp, h) // Q
		add(h, heads*headDim/tp) // O
	}
	if kvHeads > 0 && headDim > 0 && kvHeads%tp == 0 {
		add(kvHeads*headDim/tp, h) // K, V (only when GQA group is divisible)
	} else if kvHeads > 0 && headDim > 0 {
		// GQA groups not divisible — vLLM replicates KV heads. Same shape as tp=1.
		add(kvHeads*headDim, h)
	}
	add(inter/tp, h) // gate, up
	add(h, inter/tp) // down

	out := make([]Shape, 0, len(pairs))
	for s := range pairs {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].N != out[j].N {
			return out[i].N < out[j].N
		}
		return out[i].K < out[j].K
	})
	return out
}

// ConfigFilename matches vLLM's expected filename for block-FP8 kernel
// configs. The runtime looks for files of exactly this shape in
// vllm/model_executor/layers/quantization/utils/configs/.
func ConfigFilename(s Shape, deviceName string, blockN, blockK int) string {
	return fmt.Sprintf(
		"N=%d,K=%d,device_name=%s,dtype=fp8_w8a8,block_shape=[%d,%d].json",
		s.N, s.K, deviceName, blockN, blockK,
	)
}

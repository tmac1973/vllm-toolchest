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
// For each layer the matmuls the block-FP8 kernel runs are, as vLLM lays
// them out. vLLM fuses q, k and v into one QKVParallelLinear and gate and up
// into one MergedColumnParallelLinear, and looks a config up by the weight's
// own N and K -- so the fused widths are the ones to tune, and a file for an
// unfused width is never read:
//
//	QKV proj:  ((heads/tp + 2*kv_per_rank) * head_dim, hidden)
//	O proj:    (hidden, heads * head_dim / tp)
//	Gate+Up:   (2 * intermediate / tp, hidden)
//	Down:      (hidden, intermediate / tp)
//
// kv_per_rank is kv_heads/tp, or 1 when there are fewer KV heads than ranks
// and vLLM replicates them. The MLP pair is emitted for the dense MLP only
// when the model has one -- every layer of a dense model, the leading dense
// layers of an MoE one -- and again at the shared expert's width when there
// is a shared expert. Routed experts run vLLM's fused-MoE kernel, which has
// configs of its own and is not tuned here.
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

	// vLLM refuses a head count the ranks do not divide, and KV heads that
	// outnumber the ranks without dividing by them; such a split never runs.
	if heads > 0 && headDim > 0 && heads%tp == 0 {
		kvPerRank := 0
		switch {
		case kvHeads >= tp && kvHeads%tp == 0:
			kvPerRank = kvHeads / tp
		case kvHeads < tp:
			kvPerRank = 1
		}
		if kvPerRank > 0 {
			add((heads/tp+2*kvPerRank)*headDim, h) // QKV
		}
		add(h, heads*headDim/tp) // O
	}
	mlp := func(width int) {
		add(2*width/tp, h) // gate+up
		add(h, width/tp)   // down
	}
	if cfg.NumExperts == 0 || cfg.DenseLayers > 0 {
		mlp(inter)
	}
	if cfg.SharedExpertInter > 0 {
		mlp(cfg.SharedExpertInter)
	}

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

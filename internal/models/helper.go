package models

import (
	"fmt"

	"github.com/tmac1973/vllm-toolchest/internal/config"
)

// The helper model is the app's own: a small model autoconfigure loads to
// read a model card, and stops again. It is one fixed model on every host,
// chosen so that every image variant can load it -- unquantized, because
// quantized formats are not portable across the images (block FP8 cannot run
// on RDNA3 at all), and a plain dense architecture old enough for the oldest
// pinned engine. It is an instruct model with no thinking mode, so it answers
// a form directly with nothing to switch off.
const (
	HelperRepo = "Qwen/Qwen3-4B-Instruct-2507"
	// HelperServedName is what the helper is served as while it runs.
	HelperServedName = "vllmctl-helper"
	// HelperContext holds a trimmed model card, the instructions and the
	// answer.
	HelperContext = 16384
)

// helperShape is the helper's config.json as published, checked 2026-09-30.
// It is written out here so that whether the helper fits can be answered
// before it has been downloaded.
var helperShape = HFConfig{
	Architectures:     []string{"Qwen3ForCausalLM"},
	ModelType:         "qwen3",
	NumHiddenLayers:   36,
	HiddenSize:        2560,
	IntermediateSize:  9728,
	NumAttentionHeads: 32,
	NumKeyValueHeads:  8,
	HeadDim:           128,
	VocabSize:         151936,
	TieWordEmbeddings: true,
}

// helperWeightBytes is 4,022,468,096 BF16 parameters.
const helperWeightBytes = 8_044_936_192

// HelperUtil is the memory fraction the helper runs at: the machine-wide
// setting when it is in range, and the default otherwise, so an unset
// setting can never make the helper look as though it fits in nothing.
func HelperUtil(configured float64) float64 {
	return config.GPUUtilOrDefault(configured)
}

// HelperConfig is the helper's launch config. It is fixed, and ignores
// whatever is stored on the helper's record, so a config edited by hand
// cannot break the feature: one card, a short context, eager mode (no graph
// pool and nothing to compile) and one sequence, which keeps its footprint
// close to its weights.
func HelperConfig(util float64) VLLMConfig {
	return VLLMConfig{
		Dtype:                "auto",
		LoadFormat:           "auto",
		KVCacheDtype:         "auto",
		TensorParallelSize:   1,
		MaxModelLen:          HelperContext,
		EnforceEager:         true,
		MaxNumSeqs:           1,
		MaxNumBatchedTokens:  2048,
		GPUMemoryUtilization: HelperUtil(util),
	}
}

// HelperFits reports whether the helper fits on the smallest card, and when
// it does not, says by how much. It is judged against the card's size rather
// than its free memory, because whatever is serving is stopped before the
// helper starts. An unknown inventory counts as a fit: refusing on no
// information would disable the feature on a host whose cards have simply
// not been read yet.
func HelperFits(inv GPUInventory, util float64) (ok bool, why string) {
	if !inv.Known || inv.PerCardGB <= 0 {
		return true, ""
	}
	cfg := HelperConfig(util)
	m := &Model{
		ID:             HelperRepo,
		HFConfig:       helperShape,
		TotalSizeBytes: helperWeightBytes,
		Quantization:   QuantMeta{BytesPerParam: 2},
		VLLMConfig:     cfg,
	}
	est := EstimateVRAM(m, nil)
	need := RequiredAt(est, cfg, 1).TotalHighGB
	have := inv.PerCardGB * cfg.GPUMemoryUtilization
	if need <= have {
		return true, ""
	}
	return false, fmt.Sprintf("The helper model needs about %.1f GB and the smallest card offers %.1f GB.", need, have)
}

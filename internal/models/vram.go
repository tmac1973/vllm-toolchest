package models

import "fmt"

// VRAMEstimate holds computed VRAM requirements.
type VRAMEstimate struct {
	ParamCountBillion float64 `json:"param_count_billion"`
	WeightMemoryGB    float64 `json:"weight_memory_gb"`
	KVCachePerTokenB  int64   `json:"kv_cache_per_token_bytes"`
	ActivationGB      float64 `json:"activation_overhead_gb"`
	TotalSingleGPUGB  float64 `json:"total_single_gpu_gb"`
	TotalPerGPUTP2GB  float64 `json:"total_per_gpu_tp2_gb"`
	FitsSingleGPU     bool    `json:"fits_single_gpu"`
	NeedsTP2          bool    `json:"needs_tp2"`
	TooLarge          bool    `json:"too_large"`
	RecommendedTP     int     `json:"recommended_tp"`
	FitLabel          string  `json:"fit_label"`
}

// EstimateVRAM computes VRAM requirements for a model.
func EstimateVRAM(m *Model) VRAMEstimate {
	est := VRAMEstimate{}

	params := estimateParamCount(m.HFConfig)
	if params == 0 {
		// Fallback: estimate from file size
		if m.TotalSizeBytes > 0 {
			est.WeightMemoryGB = float64(m.TotalSizeBytes) / (1024 * 1024 * 1024)
			bpp := m.Quantization.BytesPerParam
			if bpp <= 0 {
				bpp = 2.0
			}
			est.ParamCountBillion = est.WeightMemoryGB / bpp
		}
		// Still compute overhead and totals from weight estimate
		if est.WeightMemoryGB > 0 {
			est.ActivationGB = activationOverhead(est.ParamCountBillion)
			est.TotalSingleGPUGB = est.WeightMemoryGB + est.ActivationGB
			est.TotalPerGPUTP2GB = est.WeightMemoryGB/2 + est.ActivationGB
		}
		computeFitLabels(&est, m.VLLMConfig.GPUMemoryUtilization)
		return est
	}

	est.ParamCountBillion = float64(params) / 1e9
	est.WeightMemoryGB = float64(params) * m.Quantization.BytesPerParam / (1024 * 1024 * 1024)

	// KV cache per token
	cfg := m.HFConfig
	headDim := cfg.HeadDim
	if headDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		headDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}

	kvDtypeBytes := 2 // FP16 default
	if m.VLLMConfig.KVCacheDtype == "fp8" || m.VLLMConfig.KVCacheDtype == "fp8_e5m2" || m.VLLMConfig.KVCacheDtype == "fp8_e4m3" {
		kvDtypeBytes = 1
	}

	kvHeads := cfg.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = cfg.NumAttentionHeads
	}

	if cfg.NumHiddenLayers > 0 && kvHeads > 0 && headDim > 0 {
		est.KVCachePerTokenB = int64(2 * cfg.NumHiddenLayers * kvHeads * headDim * kvDtypeBytes)
	}

	// Activation overhead estimate
	est.ActivationGB = activationOverhead(est.ParamCountBillion)

	// Total for single GPU (weights + activation, KV cache is dynamic)
	est.TotalSingleGPUGB = est.WeightMemoryGB + est.ActivationGB

	// Per-GPU for TP=2
	est.TotalPerGPUTP2GB = est.WeightMemoryGB/2 + est.ActivationGB

	computeFitLabels(&est, m.VLLMConfig.GPUMemoryUtilization)
	return est
}

func computeFitLabels(est *VRAMEstimate, gpuMemUtil float64) {
	gpuBudget := 32.0 * 0.90
	if gpuMemUtil > 0 {
		gpuBudget = 32.0 * gpuMemUtil
	}

	if est.TotalSingleGPUGB <= gpuBudget {
		est.FitsSingleGPU = true
		est.RecommendedTP = 1
		est.FitLabel = "Fits single GPU"
	} else if est.TotalPerGPUTP2GB <= gpuBudget {
		est.NeedsTP2 = true
		est.RecommendedTP = 2
		est.FitLabel = "Needs TP=2"
	} else {
		est.TooLarge = true
		est.RecommendedTP = 0
		est.FitLabel = "Too large"
	}
}

// VRAMFitLabel returns a short fit label given an estimated VRAM in GB.
func VRAMFitLabel(estimatedGB, perGPUGB float64, numGPUs int) string {
	budget := perGPUGB * 0.90
	if estimatedGB <= budget {
		return "fits"
	}
	if numGPUs >= 2 && estimatedGB/2 <= budget {
		return "TP=2"
	}
	return "too_large"
}

// BytesToGB converts bytes to GB for display.
func BytesToGB(b int64) float64 {
	return float64(b) / (1024 * 1024 * 1024)
}

// FormatParamCount formats parameter count for display.
func FormatParamCount(billions float64) string {
	if billions >= 1 {
		return fmt.Sprintf("%.1fB", billions)
	}
	return fmt.Sprintf("%.0fM", billions*1000)
}

func estimateParamCount(cfg HFConfig) int64 {
	if cfg.HiddenSize == 0 || cfg.NumHiddenLayers == 0 {
		return 0
	}

	h := int64(cfg.HiddenSize)
	l := int64(cfg.NumHiddenLayers)
	v := int64(cfg.VocabSize)
	inter := int64(cfg.IntermediateSize)
	kvHeads := int64(cfg.NumKeyValueHeads)
	heads := int64(cfg.NumAttentionHeads)
	headDim := int64(cfg.HeadDim)

	if inter == 0 {
		inter = 4 * h
	}
	if kvHeads == 0 {
		kvHeads = heads
	}
	if headDim == 0 && heads > 0 {
		headDim = h / heads
	}

	// Embeddings
	embedding := v * h
	outputEmbedding := int64(0)
	if !cfg.TieWordEmbeddings {
		outputEmbedding = v * h
	}

	// Per-layer attention
	qProj := h * heads * headDim
	kProj := h * kvHeads * headDim
	vProj := h * kvHeads * headDim
	oProj := heads * headDim * h
	attn := qProj + kProj + vProj + oProj

	// Per-layer MLP (3x for gate/up/down)
	mlp := 3 * h * inter

	// Per-layer norms
	norms := 2 * h

	layerTotal := attn + mlp + norms
	finalNorm := h

	return embedding + outputEmbedding + l*layerTotal + finalNorm
}

func activationOverhead(paramBillions float64) float64 {
	switch {
	case paramBillions < 3:
		return 0.3
	case paramBillions < 13:
		return 0.5
	case paramBillions < 34:
		return 1.0
	case paramBillions < 72:
		return 1.5
	default:
		return 2.0
	}
}

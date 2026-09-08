package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// modelRow is one row of the models table.
type modelRow struct {
	ID          string
	SafeID      string
	DisplayName string
	Orphaned    bool
	Quant       quantBadge
	SizeLabel   string
	VRAM        vramLabel
	// ToolParser is the parser tool calling would use, or "" when the model
	// has no tool support — which is how the template decides whether to
	// render the badge at all.
	ToolParser string
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	list := s.registry.List()

	if !isHTMX(r) {
		respondJSON(w, list)
		return
	}

	rows := make([]modelRow, 0, len(list))
	for _, m := range list {
		row := modelRow{
			ID:          m.ID,
			SafeID:      safeID(m.ID),
			DisplayName: m.DisplayName,
			Orphaned:    m.Orphaned,
			Quant:       newQuantBadge(m.Quantization),
			SizeLabel:   huggingface.FormatBytes(m.TotalSizeBytes),
			VRAM:        newVRAMLabel(m.VRAMEstimate),
		}
		if m.ToolUse.HasToolSupport {
			row.ToolParser = m.ToolUse.ToolCallParser
		}
		rows = append(rows, row)
	}

	respondHTML(w)
	s.renderPartial(w, "model_list", struct{ Rows []modelRow }{rows})
}

func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	respondJSON(w, m)
}

func (s *Server) handleModelConfigPanel(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	respondHTML(w)
	s.renderPartial(w, "model_config", s.newModelConfigView(m))
}

func (s *Server) handleUpdateModelConfig(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	var cfg models.VLLMConfig
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "json") {
		json.NewDecoder(r.Body).Decode(&cfg)
	} else {
		r.ParseForm()
		cfg = models.VLLMConfig{
			Dtype:                  r.FormValue("dtype"),
			MaxModelLen:            formInt(r, "max_model_len"),
			TensorParallelSize:     formInt(r, "tensor_parallel_size"),
			GPUMemoryUtilization:   formFloat(r, "gpu_memory_utilization"),
			EnforceEager:           r.FormValue("enforce_eager") == "on",
			EnablePrefixCaching:    r.FormValue("enable_prefix_caching") == "on",
			EnableChunkedPrefill:   r.FormValue("enable_chunked_prefill") == "on",
			MaxNumSeqs:             formInt(r, "max_num_seqs"),
			MaxNumBatchedTokens:    formInt(r, "max_num_batched_tokens"),
			Quantization:           r.FormValue("quantization"),
			LoadFormat:             r.FormValue("load_format"),
			KVCacheDtype:           r.FormValue("kv_cache_dtype"),
			TrustRemoteCode:        r.FormValue("trust_remote_code") == "on",
			EnableAutoToolChoice:   r.FormValue("enable_auto_tool_choice") == "on",
			ToolCallParser:         r.FormValue("tool_call_parser"),
			ReasoningParser:        r.FormValue("reasoning_parser"),
			AttentionBackend:       r.FormValue("attention_backend"),
			MambaCacheMode:         r.FormValue("mamba_cache_mode"),
			SpeculativeConfig:      strings.TrimSpace(r.FormValue("speculative_config")),
			CompilationConfig:      strings.TrimSpace(r.FormValue("compilation_config")),
			KVCacheMemory:          int64(formInt(r, "kv_cache_memory")),
			DisableAsyncScheduling: r.FormValue("disable_async_scheduling") == "on",
			LanguageModelOnly:      r.FormValue("language_model_only") == "on",
			Tokenizer:              r.FormValue("tokenizer"),
			ChatTemplate:           r.FormValue("chat_template"),
			ExtraFlags:             r.FormValue("extra_flags"),
		}
		if cfg.LoadFormat == "" {
			cfg.LoadFormat = "auto"
		}
		if cfg.KVCacheDtype == "" {
			cfg.KVCacheDtype = "auto"
		}
		if cfg.Dtype == "" {
			cfg.Dtype = "auto"
		}
	}

	// Auto-fill parser from model detection when tool use is enabled
	if cfg.EnableAutoToolChoice && cfg.ToolCallParser == "" {
		cfg.ToolCallParser = m.ToolUse.ToolCallParser
	}
	// Clear parser when tool use is disabled
	if !cfg.EnableAutoToolChoice {
		cfg.ToolCallParser = ""
	}

	if err := s.registry.UpdateConfig(id, cfg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Recompute VRAM estimate
	m.VLLMConfig = cfg
	m.VRAMEstimate = models.EstimateVRAM(m)
	s.registry.Register(m)

	if isHTMX(r) {
		// Re-render the full config panel so effective command updates
		s.handleModelConfigPanel(w, r)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if err := s.registry.Delete(id, true); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		return // empty response removes the row
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleScanModels(w http.ResponseWriter, r *http.Request) {
	s.registry.Maintenance()
	list := s.registry.List()

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "ok_message", fmt.Sprintf("Scan complete. %d models registered.", len(list)))
		return
	}
	respondJSON(w, map[string]int{"count": len(list)})
}

type quantOption struct {
	val   string
	label string
}

// compatibleQuantOptions returns quantization methods compatible with the model's format.
func compatibleQuantOptions(detectedMethod string, sym bool, bits int) []quantOption {
	switch detectedMethod {
	case "awq":
		opts := []quantOption{
			{"", "auto-detect (AWQ)"},
			{"awq", "AWQ"},
		}
		// Marlin works with 4-bit symmetric AWQ
		if bits == 4 {
			opts = append(opts, quantOption{"marlin", "Marlin (optimized AWQ kernel)"})
		}
		return opts

	case "gptq":
		opts := []quantOption{
			{"", "auto-detect (GPTQ)"},
			{"gptq", "GPTQ"},
		}
		// Marlin works with 4-bit symmetric GPTQ without desc_act
		if bits == 4 && sym {
			opts = append(opts, quantOption{"marlin", "Marlin (optimized GPTQ kernel)"})
		}
		return opts

	case "fp8":
		return []quantOption{
			{"", "auto-detect (FP8)"},
			{"fp8", "FP8"},
		}

	case "bitsandbytes":
		return []quantOption{
			{"", "auto-detect (BitsAndBytes)"},
			{"bitsandbytes", "BitsAndBytes"},
		}

	case "compressed_tensors":
		return []quantOption{
			{"", "auto-detect (compressed_tensors)"},
			{"compressed_tensors", "compressed_tensors"},
		}

	case "squeezellm":
		return []quantOption{
			{"", "auto-detect (SqueezeLLM)"},
			{"squeezellm", "SqueezeLLM"},
		}

	case "gguf":
		return []quantOption{
			{"", "auto-detect (GGUF)"},
			{"gguf", "GGUF"},
		}

	default:
		// FP16/BF16 unquantized model -- can do dynamic quantization
		return []quantOption{
			{"", "None (full precision)"},
			{"bitsandbytes", "BitsAndBytes (dynamic 4/8-bit at load)"},
			{"fp8", "FP8 (dynamic 8-bit, needs GPU support)"},
		}
	}
}

// quantBadge is a model's quantization rendered as a coloured pill by the
// "quant_badge" template.
type quantBadge struct {
	Label string
	Color string
}

func newQuantBadge(q models.QuantMeta) quantBadge {
	label := strings.ToUpper(q.Method)
	if label == "NONE" || label == "" {
		label = "FP16"
	}
	if q.Bits > 0 {
		label += fmt.Sprintf(" %d-bit", q.Bits)
	}
	if q.GGUFQuantType != "" {
		label = "GGUF " + q.GGUFQuantType
	}
	return quantBadge{Label: label, Color: quantBadgeColor(q.Method)}
}

// vramLabel is a model's VRAM estimate and fit verdict, rendered by the
// "vram_label" template. Unknown means there is no estimate to show.
type vramLabel struct {
	Unknown  bool
	TotalGB  float64
	FitLabel string
	Color    string
}

func newVRAMLabel(est models.VRAMEstimate) vramLabel {
	if est.WeightMemoryGB == 0 {
		return vramLabel{Unknown: true}
	}
	color := "#2d8a4e"
	if est.NeedsTP2 {
		color = "#b86e00"
	}
	if est.TooLarge {
		color = "#b83d3d"
	}
	return vramLabel{
		TotalGB:  est.TotalSingleGPUGB,
		FitLabel: est.FitLabel,
		Color:    color,
	}
}

func safeID(id string) string {
	r := strings.NewReplacer(
		"/", "--",
		".", "-",
		":", "-",
		" ", "-",
	)
	return r.Replace(id)
}

func formInt(r *http.Request, key string) int {
	v := r.FormValue(key)
	if v == "" {
		return 0
	}
	var n int
	fmt.Sscanf(v, "%d", &n)
	return n
}

func formFloat(r *http.Request, key string) float64 {
	v := r.FormValue(key)
	if v == "" {
		return 0
	}
	var f float64
	fmt.Sscanf(v, "%f", &f)
	return f
}

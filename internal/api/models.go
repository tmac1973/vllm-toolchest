package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// modelRow is one card on the models page.
type modelRow struct {
	ID          string
	SafeID      string
	DisplayName string
	URL         string
	Orphaned    bool
	Quant       quantBadge
	SizeLabel   string
	VRAM        vramLabel
	// ToolParser is the parser tool calling would use, or "" when the model
	// has no tool support — which is how the template decides whether to
	// render the badge at all.
	ToolParser string
	// Active is the model Start will launch; Serving is the one the running
	// process actually has loaded. They differ after a change that has not
	// been restarted into, which is what NeedsRestart marks.
	Active       bool
	Serving      bool
	NeedsRestart bool
	// SearchText is what the filter box matches against, lowercased once here
	// rather than on every keystroke.
	SearchText string
}

func (s *Server) modelRows() []modelRow {
	list := s.registry.List()

	servingID := ""
	if st := s.process.GetStatus(); st.State == process.StateRunning || st.State == process.StateStarting {
		servingID = st.ModelID
	}

	rows := make([]modelRow, 0, len(list))
	for _, m := range list {
		row := modelRow{
			ID:          m.ID,
			SafeID:      safeID(m.ID),
			DisplayName: displayNameOf(m),
			URL:         hfModelURL(m.ID),
			Orphaned:    m.Orphaned,
			Quant:       newQuantBadge(m.Quantization),
			SizeLabel:   huggingface.FormatBytes(m.TotalSizeBytes),
			VRAM:        newVRAMLabel(effectiveVRAM(m)),
			Active:      m.ID == s.cfg.ActiveModel,
			Serving:     m.ID == servingID,
		}
		if m.ToolUse.HasToolSupport {
			row.ToolParser = m.ToolUse.ToolCallParser
		}
		// Only worth flagging while something is actually running: with the
		// server stopped, Start will pick up the choice anyway.
		row.NeedsRestart = row.Active && servingID != "" && !row.Serving
		row.SearchText = strings.ToLower(strings.Join([]string{
			m.ID, row.DisplayName, row.Quant.Label(), m.HFConfig.ModelType,
		}, " "))
		rows = append(rows, row)
	}
	return rows
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	if !isHTMX(r) {
		respondJSON(w, s.registry.List())
		return
	}

	respondHTML(w)
	s.renderPartial(w, "model_list", struct{ Rows []modelRow }{s.modelRows()})
}

// handleActivateModel records which model the Start button launches. The whole
// list comes back: the previously-active card has to lose its radio, and the
// restart markers move with the choice.
func (s *Server) handleActivateModel(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	if m.Orphaned {
		http.Error(w, "model files are missing", http.StatusConflict)
		return
	}

	s.cfg.ActiveModel = id
	if err := s.cfg.Save(""); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if !isHTMX(r) {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "model_list", struct{ Rows []modelRow }{s.modelRows()})
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
	// Removing the registry entry and deleting tens of gigabytes are separate
	// decisions, so the caller has to say which one it meant.
	keepFiles := r.URL.Query().Get("keep_files") == "true"
	if err := s.registry.Delete(id, !keepFiles); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	// Nothing should still be pointed at a model that is gone.
	if s.cfg.ActiveModel == id {
		s.cfg.ActiveModel = ""
		_ = s.cfg.Save("")
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

// compatibleQuantOptions returns the quantization methods that both suit the
// model's format and exist in this image.
//
// hasBNB gates BitsAndBytes: not every image ships it, and vLLM only discovers
// that at load time, so offering it where it is absent turns a two-second
// choice into a launch that dies on an import several minutes later.
func compatibleQuantOptions(detectedMethod string, sym bool, bits int, hasBNB bool) []quantOption {
	// A model already stored in a format the image cannot load is not a
	// choice to hide — it is the reason the launch will fail, and saying so
	// here is the only place the operator will see it before trying.
	unavailable := func(label string) string {
		if hasBNB {
			return label
		}
		return label + " — not installed in this image"
	}

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
			{"", unavailable("auto-detect (BitsAndBytes)")},
			{"bitsandbytes", unavailable("BitsAndBytes")},
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
		// FP16/BF16 unquantized model -- can do dynamic quantization. Here
		// BitsAndBytes is one option among several rather than the model's
		// own format, so an image without it simply does not offer it.
		opts := []quantOption{{"", "None (full precision)"}}
		if hasBNB {
			opts = append(opts, quantOption{"bitsandbytes", "BitsAndBytes (dynamic 4/8-bit at load)"})
		}
		return append(opts, quantOption{"fp8", "FP8 (dynamic 8-bit, needs GPU support)"})
	}
}

// effectiveVRAM is the estimate to display for a model.
//
// A record written before the estimate was stored carries none, and the value
// is arithmetic over fields already loaded — so compute it rather than showing
// a dash on the card and zeros in the config panel.
func effectiveVRAM(m *models.Model) models.VRAMEstimate {
	if m.VRAMEstimate.WeightMemoryGB == 0 && m.HFConfig.HiddenSize > 0 {
		return models.EstimateVRAM(m)
	}
	return m.VRAMEstimate
}

// quantBadge is a model's quantization. Method and Width are separate because
// the label has to be allowed to wrap — "COMPRESSED_TENSORS 4-bit" is too wide
// for the column — and the only acceptable place to break it is the space
// between them. Left as one string, the browser also breaks at the hyphen in
// "4-bit", which reads as badly as the overflow it replaced.
type quantBadge struct {
	Method string
	Width  string
}

// Label is the two parts joined, for anything that wants the plain text.
func (q quantBadge) Label() string {
	if q.Width == "" {
		return q.Method
	}
	return q.Method + " " + q.Width
}

func newQuantBadge(q models.QuantMeta) quantBadge {
	method := strings.ToUpper(q.Method)
	if method == "NONE" || method == "" {
		method = "FP16"
	}
	badge := quantBadge{Method: method}
	if q.Bits > 0 {
		badge.Width = fmt.Sprintf("%d-bit", q.Bits)
	}
	if q.GGUFQuantType != "" {
		badge = quantBadge{Method: "GGUF", Width: q.GGUFQuantType}
	}
	return badge
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

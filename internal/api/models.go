package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/tuning"
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
	// Tunable reports whether kernel tuning could do anything for this model:
	// a block-quantized FP8 checkpoint with at least one shape the block
	// kernel will take. Derived from the model alone, deliberately — the
	// button appears for a property of the checkpoint, not for the state of
	// the tuner, and the start endpoint rejects a second concurrent job.
	Tunable bool
	// TunableShapes is how many distinct matmul shapes tuning would measure,
	// shown in the button's tooltip so the cost is visible before clicking.
	TunableShapes int
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
			VRAM:        s.newVRAMLabel(m),
			Active:      m.ID == s.cfg.ActiveModel,
			Serving:     m.ID == servingID,
		}
		if m.ToolUse.HasToolSupport {
			row.ToolParser = m.ToolUse.ToolCallParser
		}
		// Same test the Tuning page applies, so the two pages cannot disagree
		// about which models are worth tuning.
		if m.Quantization.IsBlockFP8() {
			tp := m.VLLMConfig.TensorParallelSize
			if tp < 1 {
				tp = 1
			}
			row.TunableShapes = len(tuning.DeriveShapes(m.HFConfig, tp, 128, 128))
			row.Tunable = row.TunableShapes > 0
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
	s.renderConfigPanel(w, m, panelBanner{})
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
			Env:                    strings.TrimSpace(r.FormValue("env")),
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

	// The attention backend named in the speculative config or the extra
	// flags gets the same check the picker does. Refused rather than saved
	// with a warning: unlike a risky environment variable, this one does not
	// degrade, it aborts the engine minutes into a load.
	d, known := s.vllmEnv.Descriptor()
	if err := validateNamedBackends(d, known, cfg.SpeculativeConfig, cfg.ExtraFlags); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "error_message", err.Error())
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.registry.UpdateConfig(id, cfg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Recompute VRAM estimate
	m.VLLMConfig = cfg
	m.VRAMEstimate = models.EstimateVRAM(m, s.configuredEnvPairs(m))
	s.registry.Register(m)

	if isHTMX(r) {
		// Re-render the full config panel so effective command updates
		s.renderConfigPanel(w, m, panelBanner{Warning: envBlockWarning(cfg.Env)})
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// envBlockWarning is what to say about a model's environment block, or "" when
// there is nothing to say.
//
// It warns and never refuses, for two reasons. The panel autosaves on every
// change, so refusing a half-typed line would decline to save the whole config
// and lose the operator's other edits. And a variable that silently fails to
// apply is the worse outcome: the launch drops a malformed line either way,
// so the only question is whether anyone is told.
func envBlockWarning(env string) string {
	if strings.TrimSpace(env) == "" {
		return ""
	}
	set := config.EnvSet{Extra: env}
	var notes []string
	if err := set.Validate(); err != nil {
		notes = append(notes, err.Error()+" — that line is ignored")
	}
	notes = append(notes, set.Warnings()...)
	return strings.Join(notes, " · ")
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
// It recomputes rather than reading the stored value, because the stored one
// was written with the model's own environment alone: the registry has no way
// to resolve the machine-wide and variant-knob layers that sit under it. A
// variable like VLLM_PLE_CPU_OFFLOAD set machine-wide changes what the engine
// keeps in host RAM, and an estimate blind to it is wrong by tens of gigabytes.
//
// Passing the resolved environment is also what stops the estimate and the
// launch command from drifting apart — they read the same layers, through the
// same parser, in the same order.
func (s *Server) effectiveVRAM(m *models.Model) models.VRAMEstimate {
	return models.EstimateVRAM(m, s.configuredEnvPairs(m))
}

// vramFit judges that estimate against this host's cards.
func (s *Server) vramFit(m *models.Model) (models.VRAMEstimate, models.VRAMFit) {
	est := s.effectiveVRAM(m)
	return est, models.Fit(est, m.VLLMConfig, s.gpuInventory())
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
// "vram_label" template.
//
// Three states, because there are three genuinely different things to say:
//
//	Unknown   nothing can be computed; Why says what is missing
//	!Verdict  a figure, but no cards to judge it against
//	default   a figure and a verdict
//
// The middle one is new. The old label had no way to express it and so judged
// every host against an invented single 32 GiB card, which is how a model
// serving happily across four cards came to be labelled "Too large".
type vramLabel struct {
	Unknown bool
	Why     string
	// PerGPUGB is what one card holds at the configured width, not the whole
	// model: on a four-way split the total was never the number that mattered.
	PerGPUGB float64
	FitLabel string
	Color    string
}

func (s *Server) newVRAMLabel(m *models.Model) vramLabel {
	est, fit := s.vramFit(m)

	if est.Unknown {
		return vramLabel{Unknown: true, Why: est.UnknownWhy}
	}
	if !fit.Known {
		return vramLabel{
			PerGPUGB: fit.PerGPUGB,
			FitLabel: fit.Label,
			Color:    "#6b7280",
		}
	}

	color := "#2d8a4e"
	switch {
	case fit.Configured == nil, fit.RecommendedTP == 0:
		color = "#b83d3d"
	case fit.Configured.Uncertain:
		color = "#6b7280"
	case !fit.Configured.ServesConfigured:
		color = "#b86e00"
	}

	return vramLabel{
		PerGPUGB: fit.PerGPUGB,
		FitLabel: fit.Label,
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

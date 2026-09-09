package api

import (
	"fmt"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// selectOption is one <option>. Selected is computed here rather than in the
// template so the "which value is current" rule for each control lives beside
// the list it applies to.
type selectOption struct {
	Value    string
	Label    string
	Selected bool
}

// optGroup is one <optgroup> of the tool-call-parser picker.
type optGroup struct {
	Label   string
	Options []selectOption
}

// modelConfigView is everything the "model_config" template renders. Fields
// are ordered as they appear in the panel.
type modelConfigView struct {
	ID          string
	SafeID      string
	DisplayName string

	// VRAM estimate banner
	WeightGB   float64
	OverheadGB float64
	TotalGB    float64
	FitLabel   string

	// Core
	DtypeOptions []selectOption
	MaxCtx       int
	ModelLen     int
	CtxOptions   []selectOption
	CtxIsCustom  bool
	// CtxCustomName is the custom input's name attribute: the picker and the
	// custom box share one form field, so exactly one of them carries the name
	// at a time and the other submits nothing.
	CtxCustomName string
	TPOptions     []selectOption
	GPUMemUtil    float64

	// Performance
	EnforceEager         bool
	EnablePrefixCaching  bool
	MaxNumSeqsOptions    []selectOption
	EnableChunkedPrefill bool
	MaxNumBatchedTokens  int
	// BatchedAdvice is the all-reduce token ceiling for this model and TP
	// size. It depends on the hidden size, so it cannot live in a static
	// tooltip. Empty when there is nothing to say.
	BatchedAdvice     string
	BatchedAdviceWarn bool

	// Quantization
	QuantOptions   []selectOption
	KVCacheOptions []selectOption

	// Tool use
	HasToolSupport     bool
	DetectedParser     string
	DetectionMethod    string
	ToolEnabled        bool
	ParserAutoSelected bool
	ParserGroups       []optGroup

	// Advanced
	TrustRemoteCode        bool
	BackendOptions         []selectOption
	ReasoningParser        string
	MambaOptions           []selectOption
	KVCacheMemory          int64
	SpeculativeConfig      string
	CompilationConfig      string
	DisableAsyncScheduling bool
	LanguageModelOnly      bool
	ChatTemplate           string
	Tokenizer              string
	ExtraFlags             string

	EffectiveCommand string
}

// newModelConfigView assembles the config panel's data for one model.
func (s *Server) newModelConfigView(m *models.Model) modelConfigView {
	c := m.VLLMConfig

	maxCtx := m.HFConfig.MaxPositionEmbeddings
	if maxCtx == 0 {
		maxCtx = 4096
	}
	modelLen := c.MaxModelLen
	if modelLen == 0 {
		modelLen = maxCtx
	}

	est := effectiveVRAM(m)

	v := modelConfigView{
		ID:          m.ID,
		SafeID:      safeID(m.ID),
		DisplayName: displayNameOf(m),

		WeightGB:   est.WeightMemoryGB,
		OverheadGB: est.ActivationGB,
		TotalGB:    est.TotalSingleGPUGB,
		FitLabel:   est.FitLabel,

		MaxCtx:     maxCtx,
		ModelLen:   modelLen,
		GPUMemUtil: c.GPUMemoryUtilization,

		EnforceEager:         c.EnforceEager,
		EnablePrefixCaching:  c.EnablePrefixCaching,
		EnableChunkedPrefill: c.EnableChunkedPrefill,
		MaxNumBatchedTokens:  c.MaxNumBatchedTokens,

		HasToolSupport:  m.ToolUse.HasToolSupport,
		DetectedParser:  m.ToolUse.ToolCallParser,
		DetectionMethod: m.ToolUse.DetectionMethod,

		TrustRemoteCode:        c.TrustRemoteCode,
		ReasoningParser:        c.ReasoningParser,
		KVCacheMemory:          c.KVCacheMemory,
		SpeculativeConfig:      c.SpeculativeConfig,
		CompilationConfig:      c.CompilationConfig,
		DisableAsyncScheduling: c.DisableAsyncScheduling,
		LanguageModelOnly:      c.LanguageModelOnly,
		ChatTemplate:           c.ChatTemplate,
		Tokenizer:              c.Tokenizer,
		ExtraFlags:             c.ExtraFlags,
	}

	for _, opt := range []string{"auto", "float16", "bfloat16", "float32"} {
		v.DtypeOptions = append(v.DtypeOptions, selectOption{opt, opt, c.Dtype == opt})
	}

	v.CtxOptions, v.CtxIsCustom = contextLengthOptions(maxCtx, modelLen)
	if v.CtxIsCustom {
		v.CtxCustomName = "max_model_len"
	}

	for _, tp := range []int{1, 2, 4, 8} {
		label := fmt.Sprintf("%d GPU", tp)
		if tp > 1 {
			label += "s"
		}
		v.TPOptions = append(v.TPOptions, selectOption{
			Value: fmt.Sprintf("%d", tp), Label: label, Selected: c.TensorParallelSize == tp,
		})
	}

	for _, n := range []int{1, 4, 8, 16, 32, 64, 128, 256} {
		label := fmt.Sprintf("%d", n)
		v.MaxNumSeqsOptions = append(v.MaxNumSeqsOptions, selectOption{label, label, c.MaxNumSeqs == n})
	}

	v.BatchedAdvice, v.BatchedAdviceWarn = batchedTokenAdvice(
		s.vllmEnv.IsRadiance(), m.HFConfig.HiddenSize,
		c.TensorParallelSize, c.MaxNumBatchedTokens,
	)

	for _, opt := range compatibleQuantOptions(m.Quantization.Method, m.Quantization.Sym, m.Quantization.Bits, s.vllmEnv.HasBitsAndBytes) {
		v.QuantOptions = append(v.QuantOptions, selectOption{opt.val, opt.label, c.Quantization == opt.val})
	}
	for _, opt := range []string{"auto", "fp8", "fp8_e5m2", "fp8_e4m3"} {
		v.KVCacheOptions = append(v.KVCacheOptions, selectOption{opt, opt, c.KVCacheDtype == opt})
	}

	// Tool use is on only when a parser is actually available to name — the
	// switch sets --enable-auto-tool-choice and --tool-call-parser together,
	// and vLLM rejects the first without the second.
	effectiveParser := c.ToolCallParser
	if effectiveParser == "" {
		effectiveParser = m.ToolUse.ToolCallParser
	}
	v.ToolEnabled = c.EnableAutoToolChoice && effectiveParser != ""
	v.ParserAutoSelected = c.ToolCallParser == "" || c.ToolCallParser == m.ToolUse.ToolCallParser
	for _, grp := range toolParserGroups {
		g := optGroup{Label: grp.Label}
		for _, opt := range grp.Options {
			g.Options = append(g.Options, selectOption{
				Value: opt.Value, Label: opt.Label,
				// An explicit override is only "selected" when it differs from
				// what detection found; otherwise the "(auto: …)" entry is.
				Selected: c.ToolCallParser == opt.Value && c.ToolCallParser != m.ToolUse.ToolCallParser,
			})
		}
		v.ParserGroups = append(v.ParserGroups, g)
	}

	for _, opt := range attentionBackendOptions(s.vllmEnv.IsRadiance()) {
		v.BackendOptions = append(v.BackendOptions, selectOption{opt.Val, opt.Label, c.AttentionBackend == opt.Val})
	}
	for _, opt := range []struct{ val, label string }{
		{"", "(vLLM default)"},
		{"align", "align — makes hybrid models prefix-cacheable"},
	} {
		v.MambaOptions = append(v.MambaOptions, selectOption{opt.val, opt.label, c.MambaCacheMode == opt.val})
	}

	v.EffectiveCommand = s.effectiveServeCommand(m, modelLen)
	return v
}

// contextLengthOptions builds the context-length picker: the preset ladder,
// clamped to what the model supports, with the model's own maximum inserted if
// it is not already one of the rungs. The second return reports whether the
// configured length is off the ladder, in which case the custom input takes
// over.
func contextLengthOptions(maxCtx, modelLen int) ([]selectOption, bool) {
	values := []int{2048, 4096, 8192, 16384, 32768, 65536, 131072}

	hasMax := false
	for _, v := range values {
		if v == maxCtx {
			hasMax = true
		}
	}
	if !hasMax && maxCtx > 0 {
		var merged []int
		inserted := false
		for _, v := range values {
			if !inserted && maxCtx < v {
				merged = append(merged, maxCtx)
				inserted = true
			}
			merged = append(merged, v)
		}
		if !inserted {
			merged = append(merged, maxCtx)
		}
		values = merged
	}
	if maxCtx > 0 {
		var capped []int
		for _, v := range values {
			if v <= maxCtx {
				capped = append(capped, v)
			}
		}
		values = capped
	}

	isCustom := true
	for _, v := range values {
		if v == modelLen {
			isCustom = false
		}
	}

	opts := make([]selectOption, 0, len(values))
	for _, v := range values {
		label := fmt.Sprintf("%d", v)
		if v >= 1024 {
			label = fmt.Sprintf("%dK", v/1024)
		}
		if v == maxCtx {
			label += " (max)"
		}
		opts = append(opts, selectOption{
			Value: fmt.Sprintf("%d", v), Label: label, Selected: !isCustom && v == modelLen,
		})
	}
	return opts, isCustom
}

// effectiveServeCommand is the argv the operator would see if they started
// this model now.
//
// It mirrors process.Manager.Start exactly: this box is what gets read to
// reason about a failed launch, so a preview that differs from the real
// command is worse than no preview.
func (s *Server) effectiveServeCommand(m *models.Model, modelLen int) string {
	c := m.VLLMConfig
	parser := c.ToolCallParser
	if parser == "" && c.EnableAutoToolChoice {
		parser = m.ToolUse.ToolCallParser
	}

	cfg := c.StartConfig()
	// The preview has to match what a start actually runs, and the served
	// name is part of that command.
	cfg.ServedModelName = m.ID
	cfg.MaxModelLen = modelLen
	cfg.ToolCallParser = parser

	args := process.BuildArgs(cfg)
	bin, argv := s.vllmEnv.ServeCommand(process.ResolveModelPath(m.LocalPath), append(
		[]string{"--host", "0.0.0.0", "--port", fmt.Sprintf("%d", s.cfg.VLLMPort)}, args...))

	cmd := bin
	for _, a := range argv {
		cmd += " " + a
	}
	return cmd
}

// toolParserGroups is vLLM's tool-call parser roster, grouped most-common
// first. The values must match the registration names in
// vllm/tool_parsers/__init__.py.
var toolParserGroups = []optGroup{
	{Label: "Common", Options: []selectOption{
		{Value: "hermes", Label: "hermes — Hermes, NousResearch, Qwen 2.5, plain Qwen 3"},
		{Value: "qwen3_xml", Label: "qwen3_xml — Qwen 3.5+, Qwen thinking variants, MiMo"},
		{Value: "qwen3_coder", Label: "qwen3_coder — Qwen 3 Coder"},
		{Value: "llama3_json", Label: "llama3_json — Llama 3.1 / 3.2 / 3.3"},
		{Value: "llama4_pythonic", Label: "llama4_pythonic — Llama 4"},
		{Value: "llama4_json", Label: "llama4_json — Llama 4 (json output)"},
		{Value: "mistral", Label: "mistral — Mistral, Mixtral"},
		{Value: "pythonic", Label: "pythonic — Python-style function calls"},
		{Value: "openai", Label: "openai — OpenAI-compatible JSON"},
	}},
	{Label: "DeepSeek", Options: []selectOption{
		{Value: "deepseek_v3", Label: "deepseek_v3 — DeepSeek V3, R1"},
		{Value: "deepseek_v31", Label: "deepseek_v31 — DeepSeek V3.1"},
		{Value: "deepseek_v32", Label: "deepseek_v32 — DeepSeek V3.2"},
		{Value: "deepseek_v4", Label: "deepseek_v4 — DeepSeek V4"},
	}},
	{Label: "Granite", Options: []selectOption{
		{Value: "granite", Label: "granite — IBM Granite"},
		{Value: "granite4", Label: "granite4 — IBM Granite 4"},
		{Value: "granite-20b-fc", Label: "granite-20b-fc — IBM Granite 20B FC"},
	}},
	{Label: "GLM", Options: []selectOption{
		{Value: "glm45", Label: "glm45 — GLM 4.5 MoE"},
		{Value: "glm47", Label: "glm47 — GLM 4.7 MoE"},
	}},
	{Label: "Cohere", Options: []selectOption{
		{Value: "cohere_command3", Label: "cohere_command3 — Command R / R+"},
		{Value: "cohere_command4", Label: "cohere_command4 — Command R 4"},
	}},
	{Label: "Other", Options: []selectOption{
		{Value: "gemma4", Label: "gemma4 — Gemma 4"},
		{Value: "functiongemma", Label: "functiongemma — FunctionGemma"},
		{Value: "phi4_mini_json", Label: "phi4_mini_json — Phi-4 Mini"},
		{Value: "internlm", Label: "internlm — InternLM"},
		{Value: "jamba", Label: "jamba — Jamba (AI21)"},
		{Value: "kimi_k2", Label: "kimi_k2 — Moonshot Kimi K2"},
		{Value: "minimax", Label: "minimax — MiniMax"},
		{Value: "minimax_m2", Label: "minimax_m2 — MiniMax M2"},
		{Value: "hunyuan_a13b", Label: "hunyuan_a13b — Tencent Hunyuan A13B"},
		{Value: "hy_v3", Label: "hy_v3 — Tencent Hunyuan V3"},
		{Value: "olmo3", Label: "olmo3 — AI2 OLMo 3"},
		{Value: "longcat", Label: "longcat — LongCat Flash"},
		{Value: "ernie45", Label: "ernie45 — Baidu Ernie 4.5"},
		{Value: "lfm2", Label: "lfm2 — Liquid LFM-2"},
		{Value: "xlam", Label: "xlam — Salesforce xLAM"},
		{Value: "seed_oss", Label: "seed_oss — ByteDance Seed-OSS"},
		{Value: "step3", Label: "step3 — StepFun Step3"},
		{Value: "step3p5", Label: "step3p5 — StepFun Step3.5"},
		{Value: "mimo", Label: "mimo — Xiaomi MiMo (uses Qwen3 XML)"},
		{Value: "apertus", Label: "apertus — Apertus"},
		{Value: "gigachat3", Label: "gigachat3 — GigaChat 3"},
		{Value: "poolside_v1", Label: "poolside_v1 — Poolside V1"},
	}},
}

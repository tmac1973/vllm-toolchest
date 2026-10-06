package models

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// hfMetaVersion is bumped whenever ParseHFConfig learns to read a new field.
// A stored record carrying an older version is re-parsed on the next scan.
//
// This exists because the usual trick does not generalise. AttentionLayers can
// treat zero as "written before the field existed" because a real model always
// has at least one attention layer; NumExperts cannot, because zero is the
// correct answer for every dense model. Versioning the parse makes the next
// field a one-line bump instead of another sentinel special case.
//
// 1: MoE shape -- NumExperts, NumExpertsPerTok, MoEIntermediate,
// SharedExpertInter, DenseLayers.
// 2: Draft -- whether the checkpoint is a speculative-decoding drafter.
// 3: PLELayers.
// 4: MTPLayers.
// 5: SlidingLayers, SlidingWindow, GlobalKVHeads, GlobalHeadDim.
// 6: Draft method and tokens of a speculators-format drafter.
const hfMetaVersion = 6

// ParseHFConfig reads config.json and extracts key architecture fields.
func ParseHFConfig(modelDir string) HFConfig {
	cfg := HFConfig{}

	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return cfg
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return cfg
	}

	// Top-level fields
	jsonFieldFrom(raw, &cfg.Architectures, "architectures")
	jsonFieldFrom(raw, &cfg.ModelType, "model_type")

	// For multimodal models (Qwen VL, LLaVA, etc.), architecture fields
	// are nested under text_config. Try top-level first, fall back to text_config.
	src := raw
	if _, ok := raw["text_config"]; ok {
		var textCfg map[string]json.RawMessage
		if json.Unmarshal(raw["text_config"], &textCfg) == nil {
			// Check if the text_config has the fields we need
			if _, has := textCfg["hidden_size"]; has {
				src = textCfg
			}
		}
	}

	jsonFieldFrom(src, &cfg.VocabSize, "vocab_size")
	jsonFieldFrom(src, &cfg.TorchDtype, "torch_dtype")
	jsonFieldFrom(src, &cfg.TieWordEmbeddings, "tie_word_embeddings")

	// Hidden size (with aliases)
	if !jsonFieldFrom(src, &cfg.HiddenSize, "hidden_size") {
		if !jsonFieldFrom(src, &cfg.HiddenSize, "d_model") {
			jsonFieldFrom(src, &cfg.HiddenSize, "n_embd")
		}
	}

	// Num hidden layers (with aliases)
	if !jsonFieldFrom(src, &cfg.NumHiddenLayers, "num_hidden_layers") {
		if !jsonFieldFrom(src, &cfg.NumHiddenLayers, "n_layer") {
			if !jsonFieldFrom(src, &cfg.NumHiddenLayers, "num_layers") {
				jsonFieldFrom(src, &cfg.NumHiddenLayers, "n_layers")
			}
		}
	}

	// Hybrid models list a type per layer. Only the full-attention ones carry
	// a KV cache; linear-attention / mamba / GDN layers keep a fixed-size
	// recurrent state instead, which does not scale with context.
	jsonFieldFrom(src, &cfg.SlidingWindow, "sliding_window")
	cfg.AttentionLayers, cfg.SlidingLayers = countAttentionLayers(src, cfg.NumHiddenLayers, cfg.SlidingWindow > 0)
	jsonFieldFrom(src, &cfg.GlobalKVHeads, "num_global_key_value_heads")
	jsonFieldFrom(src, &cfg.GlobalHeadDim, "global_head_dim")

	// Intermediate size
	if !jsonFieldFrom(src, &cfg.IntermediateSize, "intermediate_size") {
		jsonFieldFrom(src, &cfg.IntermediateSize, "ffn_dim")
	}

	// Mixture-of-experts shape. Every family spells these differently, so the
	// cascade matters: Qwen uses num_experts, DeepSeek n_routed_experts,
	// Mixtral num_local_experts.
	if !jsonFieldFrom(src, &cfg.NumExperts, "num_experts") {
		if !jsonFieldFrom(src, &cfg.NumExperts, "n_routed_experts") {
			if !jsonFieldFrom(src, &cfg.NumExperts, "num_local_experts") {
				jsonFieldFrom(src, &cfg.NumExperts, "moe_num_experts")
			}
		}
	}
	if !jsonFieldFrom(src, &cfg.NumExpertsPerTok, "num_experts_per_tok") {
		jsonFieldFrom(src, &cfg.NumExpertsPerTok, "moe_topk")
	}
	if !jsonFieldFrom(src, &cfg.MoEIntermediate, "moe_intermediate_size") {
		jsonFieldFrom(src, &cfg.MoEIntermediate, "intermediate_size_moe")
	}
	// A shared expert may be stated as a width, or as a count of experts at
	// the MoE width (DeepSeek's n_shared_experts).
	if !jsonFieldFrom(src, &cfg.SharedExpertInter, "shared_expert_intermediate_size") {
		var nShared int
		if jsonFieldFrom(src, &nShared, "n_shared_experts") && nShared > 0 {
			cfg.SharedExpertInter = nShared * cfg.MoEIntermediate
		}
	}
	cfg.DenseLayers = countDenseLayers(src, cfg.NumHiddenLayers, cfg.NumExperts)

	// Attention heads
	if !jsonFieldFrom(src, &cfg.NumAttentionHeads, "num_attention_heads") {
		if !jsonFieldFrom(src, &cfg.NumAttentionHeads, "n_head") {
			jsonFieldFrom(src, &cfg.NumAttentionHeads, "num_heads")
		}
	}

	// KV heads
	if !jsonFieldFrom(src, &cfg.NumKeyValueHeads, "num_key_value_heads") {
		if !jsonFieldFrom(src, &cfg.NumKeyValueHeads, "num_kv_heads") {
			cfg.NumKeyValueHeads = cfg.NumAttentionHeads // MHA default
		}
	}

	// Head dim
	if !jsonFieldFrom(src, &cfg.HeadDim, "head_dim") {
		if cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
			cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
		}
	}

	// Max position embeddings
	if !jsonFieldFrom(src, &cfg.MaxPositionEmbeddings, "max_position_embeddings") {
		if !jsonFieldFrom(src, &cfg.MaxPositionEmbeddings, "max_sequence_length") {
			jsonFieldFrom(src, &cfg.MaxPositionEmbeddings, "n_positions")
		}
	}

	// Also check top-level for fields that might only be there
	if cfg.VocabSize == 0 {
		jsonFieldFrom(raw, &cfg.VocabSize, "vocab_size")
	}
	if cfg.TorchDtype == "" {
		jsonFieldFrom(raw, &cfg.TorchDtype, "torch_dtype")
	}

	cfg.Draft = parseDraft(raw, cfg.Architectures)

	// A per-layer embedding table (the n-gram "engram" tables some Qwen-Next
	// checkpoints carry) is what PLE offload moves to host RAM. Knowing the
	// checkpoint has one lets an image that offloads it by default be
	// estimated as doing so.
	var pleLayers []int
	if jsonFieldFrom(src, &pleLayers, "ple_layer_ids") {
		cfg.PLELayers = len(pleLayers)
	}

	// Qwen3.5 names its MTP layers mtp_num_hidden_layers; DeepSeek and GLM,
	// num_nextn_predict_layers.
	if !jsonFieldFrom(src, &cfg.MTPLayers, "mtp_num_hidden_layers") {
		jsonFieldFrom(src, &cfg.MTPLayers, "num_nextn_predict_layers")
	}

	// Stamped only on the success path: a config.json we could not read or
	// parse stays stale and gets retried, rather than being recorded as
	// fully parsed at this version.
	cfg.MetaVersion = hfMetaVersion

	return cfg
}

// DetectQuantization detects the quantization method from model files.
func DetectQuantization(modelDir, modelID string) QuantMeta {
	q := QuantMeta{Method: "none", BytesPerParam: 2.0}

	// 1. Check quantize_config.json
	if data, err := os.ReadFile(filepath.Join(modelDir, "quantize_config.json")); err == nil {
		var qc struct {
			QuantMethod string `json:"quant_method"`
			Bits        int    `json:"bits"`
			GroupSize   int    `json:"group_size"`
			DescAct     bool   `json:"desc_act"`
			Sym         bool   `json:"sym"`
		}
		if json.Unmarshal(data, &qc) == nil && qc.QuantMethod != "" {
			q.Method = strings.ToLower(qc.QuantMethod)
			q.Bits = qc.Bits
			q.GroupSize = qc.GroupSize
			q.DescAct = qc.DescAct
			q.Sym = qc.Sym
			q.BytesPerParam = bytesPerParam(q.Method, q.Bits, q.GroupSize)
			return q
		}
	}

	// 2. Check config.json quantization_config
	if data, err := os.ReadFile(filepath.Join(modelDir, "config.json")); err == nil {
		var cfg struct {
			QuantizationConfig *struct {
				QuantMethod  string                     `json:"quant_method"`
				Bits         int                        `json:"bits"`
				GroupSize    int                        `json:"group_size"`
				Format       string                     `json:"format"`
				ConfigGroups map[string]json.RawMessage `json:"config_groups"`
				// Native fp8 checkpoints state their block shape here.
				// compressed-tensors puts it inside config_groups instead.
				WeightBlockSize []int `json:"weight_block_size"`
			} `json:"quantization_config"`
		}
		if json.Unmarshal(data, &cfg) == nil && cfg.QuantizationConfig != nil && cfg.QuantizationConfig.QuantMethod != "" {
			q.Method = strings.ToLower(cfg.QuantizationConfig.QuantMethod)
			q.Bits = cfg.QuantizationConfig.Bits
			q.GroupSize = cfg.QuantizationConfig.GroupSize
			// compressed-tensors (RedHatAI / llm-compressor format) stores
			// the actual quant scheme inside config_groups[*].weights.
			// Parse it so we get the right bytes/param for FP8, INT8, INT4.
			q.WeightBlockSize = cfg.QuantizationConfig.WeightBlockSize
			if q.Method == "compressed-tensors" || q.Method == "compressed_tensors" {
				q.Bits, _ = compressedTensorsBits(cfg.QuantizationConfig.ConfigGroups, cfg.QuantizationConfig.Format)
				if len(q.WeightBlockSize) == 0 {
					q.WeightBlockSize = compressedTensorsBlockSize(cfg.QuantizationConfig.ConfigGroups)
				}
			}
			q.BytesPerParam = bytesPerParam(q.Method, q.Bits, q.GroupSize)
			return q
		}
	}

	// 3. Check for GGUF files
	if hasGGUFFiles(modelDir) {
		q.Method = "gguf"
		q.GGUFQuantType = detectGGUFQuantType(modelDir)
		q.BytesPerParam = ggufBytesPerParam(q.GGUFQuantType)
		return q
	}

	// 4. Heuristic from model ID
	lower := strings.ToLower(modelID)
	switch {
	case strings.Contains(lower, "-awq"):
		q.Method = "awq"
		q.Bits = 4
		q.BytesPerParam = 0.5625
	case strings.Contains(lower, "-gptq"):
		q.Method = "gptq"
		q.Bits = 4
		q.BytesPerParam = 0.5625
	case strings.Contains(lower, "-fp8"):
		q.Method = "fp8"
		q.Bits = 8
		q.BytesPerParam = 1.0
	case strings.Contains(lower, "-bnb-4bit"):
		q.Method = "bitsandbytes"
		q.Bits = 4
		q.BytesPerParam = 0.5625
	case strings.Contains(lower, "-bnb-8bit"):
		q.Method = "bitsandbytes"
		q.Bits = 8
		q.BytesPerParam = 1.0625
	}

	return q
}

// DetectToolUse checks if the model supports tool/function calling.
// Detection order: architecture (most reliable for new model families) →
// chat template regex → model-name patterns. The architecture check goes
// first because newer model families (Qwen3.5+ hybrid, Llama 4, DeepSeek
// V3.x, GLM 4.x MoE, Granite 4) often share chat-template markers with
// older relatives but emit a different tool-call format at runtime.
func DetectToolUse(modelDir, modelID string, hfCfg HFConfig) ToolUseMeta {
	t := ToolUseMeta{}

	if parser := detectToolParserFromArch(hfCfg.Architectures); parser != "" {
		t.HasToolSupport = true
		t.ToolCallParser = parser
		t.DetectionMethod = "architecture"
		return t
	}

	if template := chatTemplate(modelDir); template != "" {
		if parser := detectToolParser(template); parser != "" {
			t.HasToolSupport = true
			t.ToolCallParser = parser
			t.DetectionMethod = "chat_template_regex"
			return t
		}
	}

	parser := detectToolParserFromName(modelID)
	if parser != "" {
		t.HasToolSupport = true
		t.ToolCallParser = parser
		t.DetectionMethod = "known_model_family"
	}

	return t
}

// chatTemplate is the model's chat template: tokenizer_config.json's
// chat_template -- a string, or a list of named templates, all of which are
// read -- or, when that has none, chat_template.jinja beside it, where newer
// repositories keep it and where vLLM looks too.
func chatTemplate(modelDir string) string {
	if data, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json")); err == nil {
		var tc struct {
			ChatTemplate any `json:"chat_template"`
		}
		if json.Unmarshal(data, &tc) == nil {
			switch v := tc.ChatTemplate.(type) {
			case string:
				if v != "" {
					return v
				}
			case []any:
				template := ""
				for _, item := range v {
					if m, ok := item.(map[string]any); ok {
						if s, ok := m["template"].(string); ok {
							template += s + "\n"
						}
					}
				}
				if template != "" {
					return template
				}
			}
		}
	}
	if data, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil {
		return string(data)
	}
	return ""
}

// detectToolParserFromArch picks a parser by HF architecture string.
// Architectures are a more reliable signal than name patterns for
// distinguishing modern model families that share lineage with older
// ones (Qwen3.5+ vs Qwen3, Llama 4 vs Llama 3, DeepSeek V3.x revisions).
func detectToolParserFromArch(archs []string) string {
	for _, a := range archs {
		lower := strings.ToLower(a)
		switch {
		// Qwen3.5+ hybrid (Mamba/GDN + thinking blocks) — class names look
		// like Qwen3_5ForConditionalGeneration, Qwen3_6*, MiMo*.
		case strings.HasPrefix(lower, "qwen3_5"),
			strings.HasPrefix(lower, "qwen3_6"),
			strings.Contains(lower, "mimo"):
			return "qwen3_xml"
		case strings.HasPrefix(lower, "llama4"):
			return "llama4_pythonic"
		// DeepSeek V3.x / V4 share the V3 base class, so check most specific first.
		case strings.Contains(lower, "deepseekv4"):
			return "deepseek_v4"
		case strings.Contains(lower, "deepseekv32"):
			return "deepseek_v32"
		case strings.Contains(lower, "deepseekv31"):
			return "deepseek_v31"
		case strings.Contains(lower, "deepseekv3"):
			return "deepseek_v3"
		case strings.HasPrefix(lower, "glm47"):
			return "glm47"
		case strings.HasPrefix(lower, "glm4moe"), strings.HasPrefix(lower, "glm45"):
			return "glm45"
		case strings.HasPrefix(lower, "granite4"):
			return "granite4"
		case strings.HasPrefix(lower, "gemma4"):
			return "gemma4"
		case strings.Contains(lower, "minimaxm2"):
			return "minimax_m2"
		case strings.Contains(lower, "minimax"):
			return "minimax"
		case strings.Contains(lower, "hunyuana13b"):
			return "hunyuan_a13b"
		case strings.HasPrefix(lower, "olmo3"):
			return "olmo3"
		case strings.HasPrefix(lower, "apertus"):
			return "apertus"
		case strings.HasPrefix(lower, "kimik2"), strings.Contains(lower, "kimi_k2"):
			return "kimi_k2"
		}
	}
	return ""
}

func detectToolParser(template string) string {
	// Qwen's <tool_call> wraps two different formats. Qwen3.5 and Qwen3
	// Coder put XML inside it (<function=name><parameter=...>), which the
	// XML parsers read; Qwen 2.5 and Qwen3, thinking variants included, put
	// Hermes-style JSON inside it. <think> says nothing about which: plain
	// Qwen3's template has it and calls in JSON. The XML marker does.
	if strings.Contains(template, "<tool_call>") && strings.Contains(template, "<function=") {
		if strings.Contains(template, "<think>") {
			return "qwen3_xml" // Qwen3.5: XML calls after a thinking block
		}
		return "qwen3_coder"
	}

	patterns := []struct {
		pattern *regexp.Regexp
		parser  string
	}{
		{regexp.MustCompile(`<\|?tool_call\|?>`), "hermes"},
		{regexp.MustCompile(`\[TOOL_CALLS\]|\[AVAILABLE_TOOLS\]`), "mistral"},
		{regexp.MustCompile(`<function=`), "granite"},
		// Llama 3.x. <|python_tag|> only prefixes its built-in ipython tools
		// (brave_search, wolfram_alpha); the tools a client defines are
		// called in JSON, which llama3_json parses. Llama 4, the family that
		// does call tools pythonically, is caught by its architecture first.
		{regexp.MustCompile(`<\|python_tag\|>`), "llama3_json"},
		{regexp.MustCompile(`<\|plugin\|>`), "internlm"},
		{regexp.MustCompile(`"type":\s*"function"`), "llama3_json"},
		{regexp.MustCompile(`tool_calls`), "hermes"},
	}

	for _, p := range patterns {
		if p.pattern.MatchString(template) {
			return p.parser
		}
	}
	return ""
}

// detectToolParserFromName is the last-resort fallback when neither the
// architecture nor the chat template gave us a hit. Order matters: more
// specific patterns (qwen3-coder, deepseek-r1) must come before broader
// ones (qwen3, deepseek).
func detectToolParserFromName(modelID string) string {
	lower := strings.ToLower(modelID)
	switch {
	case strings.Contains(lower, "hermes"):
		return "hermes"
	case strings.Contains(lower, "qwen3") && strings.Contains(lower, "coder"):
		return "qwen3_coder"
	case strings.Contains(lower, "qwen3.5"),
		strings.Contains(lower, "qwen3.6"),
		strings.Contains(lower, "qwen-3.5"),
		strings.Contains(lower, "qwen-3.6"),
		strings.Contains(lower, "thinking") && strings.Contains(lower, "qwen"):
		return "qwen3_xml"
	case strings.Contains(lower, "qwen2.5"), strings.Contains(lower, "qwen3"):
		return "hermes"
	case strings.Contains(lower, "llama-4"), strings.Contains(lower, "llama4"):
		return "llama4_pythonic"
	case strings.Contains(lower, "llama-3.1"),
		strings.Contains(lower, "llama-3.2"),
		strings.Contains(lower, "llama-3.3"):
		return "llama3_json"
	case strings.Contains(lower, "deepseek-v4"), strings.Contains(lower, "deepseek_v4"):
		return "deepseek_v4"
	case strings.Contains(lower, "deepseek-v3.2"), strings.Contains(lower, "deepseek-v32"):
		return "deepseek_v32"
	case strings.Contains(lower, "deepseek-v3.1"), strings.Contains(lower, "deepseek-v31"):
		return "deepseek_v31"
	case strings.Contains(lower, "deepseek-v3"),
		strings.Contains(lower, "deepseek-r1"),
		strings.Contains(lower, "deepseek_r1"):
		return "deepseek_v3"
	case strings.Contains(lower, "mistral"), strings.Contains(lower, "mixtral"):
		return "mistral"
	case strings.Contains(lower, "granite-4"), strings.Contains(lower, "granite4"):
		return "granite4"
	case strings.Contains(lower, "granite-20b") && strings.Contains(lower, "fc"):
		return "granite-20b-fc"
	case strings.Contains(lower, "granite"):
		return "granite"
	case strings.Contains(lower, "glm-4.7"), strings.Contains(lower, "glm-47"):
		return "glm47"
	case strings.Contains(lower, "glm-4.5"), strings.Contains(lower, "glm-45"):
		return "glm45"
	case strings.Contains(lower, "gemma-4"), strings.Contains(lower, "gemma4"):
		return "gemma4"
	case strings.Contains(lower, "phi-4") && strings.Contains(lower, "mini"):
		return "phi4_mini_json"
	case strings.Contains(lower, "command-r4"), strings.Contains(lower, "command-r-4"):
		return "cohere_command4"
	case strings.Contains(lower, "command-r"), strings.Contains(lower, "command-r3"):
		return "cohere_command3"
	case strings.Contains(lower, "kimi"):
		return "kimi_k2"
	case strings.Contains(lower, "minimax-m2"), strings.Contains(lower, "minimax_m2"):
		return "minimax_m2"
	case strings.Contains(lower, "minimax"):
		return "minimax"
	case strings.Contains(lower, "hunyuan"):
		return "hunyuan_a13b"
	case strings.Contains(lower, "olmo-3"), strings.Contains(lower, "olmo3"):
		return "olmo3"
	case strings.Contains(lower, "longcat"):
		return "longcat"
	case strings.Contains(lower, "step3.5"), strings.Contains(lower, "step-3.5"):
		return "step3p5"
	case strings.Contains(lower, "step3"), strings.Contains(lower, "step-3"):
		return "step3"
	case strings.Contains(lower, "seed-oss"), strings.Contains(lower, "seed_oss"):
		return "seed_oss"
	case strings.Contains(lower, "ernie"):
		return "ernie45"
	case strings.Contains(lower, "lfm-2"), strings.Contains(lower, "lfm2"):
		return "lfm2"
	case strings.Contains(lower, "xlam"):
		return "xlam"
	case strings.Contains(lower, "internlm"):
		return "internlm"
	case strings.Contains(lower, "jamba"):
		return "jamba"
	case strings.Contains(lower, "mimo"):
		return "qwen3_xml"
	case strings.Contains(lower, "apertus"):
		return "apertus"
	}
	return ""
}

// DetectVision checks if the model is a vision model.
func DetectVision(modelDir string, hfCfg HFConfig) VisionMeta {
	v := VisionMeta{}

	if processesImages(filepath.Join(modelDir, "processor_config.json")) ||
		processesImages(filepath.Join(modelDir, "preprocessor_config.json")) {
		v.IsVisionModel = true
		return v
	}

	// Check config.json for vision_config
	data, _ := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if data != nil {
		var raw map[string]json.RawMessage
		if json.Unmarshal(data, &raw) == nil {
			if _, ok := raw["vision_config"]; ok {
				v.IsVisionModel = true
				return v
			}
		}
	}

	// Check architecture name
	for _, arch := range hfCfg.Architectures {
		lower := strings.ToLower(arch)
		if strings.Contains(lower, "vision") || strings.Contains(lower, "vl") ||
			strings.Contains(lower, "pix") || strings.Contains(lower, "llava") {
			v.IsVisionModel = true
			return v
		}
	}

	return v
}

// processesImages reports a processor config that handles images. Having one
// is not enough: audio models (Whisper, Qwen2-Audio) ship a
// preprocessor_config.json for their feature extractor. One that handles
// images says so in its keys -- image_processor_type, image_mean,
// image_token, vision_feature_select_strategy -- which an audio-only one
// never has.
func processesImages(path string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var raw map[string]json.RawMessage
	if json.Unmarshal(data, &raw) != nil {
		return false
	}
	for key := range raw {
		if k := strings.ToLower(key); strings.Contains(k, "image") || strings.Contains(k, "vision") {
			return true
		}
	}
	return false
}

// ParseGenDefaults reads generation_config.json.
func ParseGenDefaults(modelDir string) GenDefaults {
	g := GenDefaults{}
	data, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		return g
	}
	// Each field is read on its own: one value of an unexpected type -- a
	// top_k written as 20.0, a temperature as a string -- costs that value,
	// not every default the file states.
	var raw map[string]json.RawMessage
	if json.Unmarshal(data, &raw) != nil {
		return g
	}
	g.Temperature = genFloat(raw["temperature"])
	g.TopP = genFloat(raw["top_p"])
	g.TopK = genInt(raw["top_k"])
	g.RepetitionPenalty = genFloat(raw["repetition_penalty"])
	g.MaxNewTokens = genInt(raw["max_new_tokens"])
	return g
}

// genFloat is a generation_config number, or nil when it is absent, null or
// not a number.
func genFloat(v json.RawMessage) *float64 {
	var f *float64
	if len(v) == 0 || json.Unmarshal(v, &f) != nil {
		return nil
	}
	return f
}

// genInt is a generation_config count. A whole number written as a float
// (20.0) is read as the integer it is; a fractional one is not a count.
func genInt(v json.RawMessage) *int {
	f := genFloat(v)
	if f == nil || *f != math.Trunc(*f) {
		return nil
	}
	n := int(*f)
	return &n
}

func bytesPerParam(method string, bits, groupSize int) float64 {
	if bits == 0 && strings.Contains(method, "fp8") {
		// FP8 names its own width. A block-quantized FP8 checkpoint leaves
		// bits unset, and returning "unknown" for it left the structural
		// figure at zero -- which is survivable on its own, since the size on
		// disk carries the estimate, but it makes the offload residual
		// (checkpoint - structural) the *entire* checkpoint. Enabling PLE
		// offload on such a model would then report ~100% of it offloaded.
		bits = 8
	}
	if bits == 0 {
		// Defaulting silently to 4-bit was an old bug — newer formats
		// (compressed-tensors, bitsandbytes) leave bits unset. 0 means
		// "unknown"; let EstimateVRAM fall back to disk-size rather
		// than fabricate a quantization level.
		return 0
	}
	base := float64(bits) / 8.0
	if groupSize > 0 && (method == "gptq" || method == "awq") {
		overhead := 2.0 / float64(groupSize) * 2 // scales + zeros
		return base + overhead
	}
	return base + 0.0625 // small overhead for metadata
}

// compressedTensorsBits inspects the config_groups of a compressed-tensors
// (llm-compressor / RedHatAI) checkpoint to recover the weight bit-width.
// Each group has a weights.num_bits and weights.type ("float" = FP8,
// "int" = INT8/INT4). Returns 0 if the scheme can't be determined.
func compressedTensorsBits(groups map[string]json.RawMessage, format string) (int, string) {
	for _, raw := range groups {
		var grp struct {
			Weights struct {
				NumBits int    `json:"num_bits"`
				Type    string `json:"type"`
			} `json:"weights"`
		}
		if json.Unmarshal(raw, &grp) == nil && grp.Weights.NumBits > 0 {
			return grp.Weights.NumBits, strings.ToLower(grp.Weights.Type)
		}
	}
	// Format hint as a secondary signal.
	switch strings.ToLower(format) {
	case "float-quantized":
		return 8, "float" // assume FP8 — the only float quantization in vLLM today
	case "pack-quantized":
		return 4, "int" // INT4 packed
	case "int-quantized":
		return 8, "int"
	}
	return 0, ""
}

// compressedTensorsBlockSize reads the weight block shape from a
// compressed-tensors config, which states it per group rather than at the top
// level.
//
// Only a "block" strategy counts. The other strategies name the axis the
// scales apply over -- "tensor", "channel", "group", "token" -- and none of
// them is a blockwise scheme.
func compressedTensorsBlockSize(groups map[string]json.RawMessage) []int {
	for _, raw := range groups {
		var grp struct {
			Weights struct {
				Strategy       string `json:"strategy"`
				BlockStructure []int  `json:"block_structure"`
			} `json:"weights"`
		}
		if json.Unmarshal(raw, &grp) != nil {
			continue
		}
		if strings.ToLower(grp.Weights.Strategy) != "block" {
			continue
		}
		if len(grp.Weights.BlockStructure) >= 2 {
			return grp.Weights.BlockStructure
		}
	}
	return nil
}

func detectGGUFQuantType(dir string) string {
	matches, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if len(matches) == 0 {
		return ""
	}
	name := filepath.Base(matches[0])
	re := regexp.MustCompile(`[.-]([Qq]\d+_[Kk]?_?[A-Za-z]*|[Ff]\d+)\.gguf$`)
	if m := re.FindStringSubmatch(name); len(m) > 1 {
		return strings.ToUpper(m[1])
	}
	return ""
}

func ggufBytesPerParam(quantType string) float64 {
	switch strings.ToUpper(quantType) {
	case "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M":
		return 0.5625
	case "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M":
		return 0.6875
	case "Q6_K":
		return 0.8125
	case "Q8_0":
		return 1.0625
	case "Q2_K":
		return 0.3125
	case "Q3_K_S", "Q3_K_M", "Q3_K_L":
		return 0.4375
	case "F16":
		return 2.0
	default:
		return 0.5625 // default to ~4bit
	}
}

// jsonFieldFrom tries to unmarshal a field from a raw JSON map.
func jsonFieldFrom[T any](raw map[string]json.RawMessage, dst *T, key string) bool {
	v, ok := raw[key]
	if !ok {
		return false
	}
	return json.Unmarshal(v, dst) == nil
}

// countAttentionLayers works out how many layers hold a KV cache that grows
// with the context, and how many hold one capped at a sliding window.
//
// Three shapes appear in the wild, in decreasing order of reliability:
//
//	layer_types: ["full_attention", "linear_attention", ...]   one entry per layer
//	full_attention_interval: 4                                 every Nth layer
//	(neither)                                                  assume dense
//
// A "sliding" layer is counted apart only when the config gives a window
// (windowed); without one there is nothing to cap it at, and it is counted
// as full. Returns 0 full layers when the layer count itself is unknown, so
// callers can tell "no information" from "genuinely zero".
func countAttentionLayers(src map[string]json.RawMessage, totalLayers int, windowed bool) (full, sliding int) {
	if totalLayers <= 0 {
		return 0, 0
	}

	var types []string
	if jsonFieldFrom(src, &types, "layer_types") && len(types) > 0 {
		for _, t := range types {
			// Match on the absence of a linear/recurrent marker rather than a
			// fixed list of attention spellings: new hybrids keep inventing
			// names for their recurrent layers, and mistaking one for
			// attention overstates memory, while the reverse understates it.
			switch {
			case strings.Contains(t, "linear"),
				strings.Contains(t, "mamba"),
				strings.Contains(t, "recurrent"),
				strings.Contains(t, "gdn"),
				strings.Contains(t, "conv"):
				// recurrent layer: no KV cache
			case windowed && strings.Contains(t, "sliding"):
				sliding++
			default:
				full++
			}
		}
		return full, sliding
	}

	// Some configs give only the stride between full-attention layers.
	var interval int
	if jsonFieldFrom(src, &interval, "full_attention_interval") && interval > 1 {
		return totalLayers / interval, 0
	}

	return totalLayers, 0
}

// countDenseLayers works out how many layers of an MoE model keep an ordinary
// MLP rather than an expert block. Returns 0 for "every layer is MoE", which
// is also the right answer for a model with no MoE at all -- the caller only
// consults this once it has a usable expert shape.
//
// Two conventions, and they disagree about what they describe:
//
//	first_k_dense_replace: 3      DeepSeek -- the leading N layers are dense
//	mlp_only_layers + decoder_sparse_step   Qwen -- a layer is MoE only when
//	                                        it is off the exception list and
//	                                        lands on the stride
func countDenseLayers(src map[string]json.RawMessage, totalLayers, numExperts int) int {
	if totalLayers <= 0 || numExperts <= 0 {
		return 0
	}

	var firstKDense int
	if jsonFieldFrom(src, &firstKDense, "first_k_dense_replace") && firstKDense > 0 {
		if firstKDense > totalLayers {
			return totalLayers
		}
		return firstKDense
	}

	var mlpOnly []int
	haveMLPOnly := jsonFieldFrom(src, &mlpOnly, "mlp_only_layers")
	var sparseStep int
	haveStep := jsonFieldFrom(src, &sparseStep, "decoder_sparse_step")
	if !haveMLPOnly && !haveStep {
		return 0
	}
	if sparseStep <= 0 {
		sparseStep = 1
	}

	only := make(map[int]bool, len(mlpOnly))
	for _, i := range mlpOnly {
		only[i] = true
	}

	dense := 0
	for i := 0; i < totalLayers; i++ {
		if only[i] || (i+1)%sparseStep != 0 {
			dense++
		}
	}
	return dense
}

package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

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
	cfg.AttentionLayers = countAttentionLayers(src, cfg.NumHiddenLayers)

	// Intermediate size
	if !jsonFieldFrom(src, &cfg.IntermediateSize, "intermediate_size") {
		jsonFieldFrom(src, &cfg.IntermediateSize, "ffn_dim")
	}

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
			} `json:"quantization_config"`
		}
		if json.Unmarshal(data, &cfg) == nil && cfg.QuantizationConfig != nil && cfg.QuantizationConfig.QuantMethod != "" {
			q.Method = strings.ToLower(cfg.QuantizationConfig.QuantMethod)
			q.Bits = cfg.QuantizationConfig.Bits
			q.GroupSize = cfg.QuantizationConfig.GroupSize
			// compressed-tensors (RedHatAI / llm-compressor format) stores
			// the actual quant scheme inside config_groups[*].weights.
			// Parse it so we get the right bytes/param for FP8, INT8, INT4.
			if q.Method == "compressed-tensors" || q.Method == "compressed_tensors" {
				q.Bits, _ = compressedTensorsBits(cfg.QuantizationConfig.ConfigGroups, cfg.QuantizationConfig.Format)
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

	data, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json"))
	if err == nil {
		var tc struct {
			ChatTemplate interface{} `json:"chat_template"`
		}
		if json.Unmarshal(data, &tc) == nil && tc.ChatTemplate != nil {
			template := ""
			switch v := tc.ChatTemplate.(type) {
			case string:
				template = v
			case []interface{}:
				for _, item := range v {
					if m, ok := item.(map[string]interface{}); ok {
						if s, ok := m["template"].(string); ok {
							template += s + "\n"
						}
					}
				}
			}

			if template != "" {
				parser := detectToolParser(template)
				if parser != "" {
					t.HasToolSupport = true
					t.ToolCallParser = parser
					t.DetectionMethod = "chat_template_regex"
					return t
				}
			}
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
	patterns := []struct {
		pattern *regexp.Regexp
		parser  string
	}{
		// Qwen3-XML emits tool calls wrapped in <think>...</think> blocks
		// then XML; matching both markers in the same template is a strong
		// indicator the model is a Qwen3 thinking variant even when the
		// arch field doesn't make it obvious.
		{regexp.MustCompile(`<think>[\s\S]*<tool_call>|<tool_call>[\s\S]*<think>`), "qwen3_xml"},
		{regexp.MustCompile(`<\|?tool_call\|?>`), "hermes"},
		{regexp.MustCompile(`\[TOOL_CALLS\]|\[AVAILABLE_TOOLS\]`), "mistral"},
		{regexp.MustCompile(`<function=`), "granite"},
		{regexp.MustCompile(`<\|python_tag\|>`), "pythonic"},
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

	if fileExists(filepath.Join(modelDir, "processor_config.json")) ||
		fileExists(filepath.Join(modelDir, "preprocessor_config.json")) {
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

// ParseGenDefaults reads generation_config.json.
func ParseGenDefaults(modelDir string) GenDefaults {
	g := GenDefaults{}
	data, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		return g
	}
	var raw struct {
		Temperature       *float64 `json:"temperature"`
		TopP              *float64 `json:"top_p"`
		TopK              *int     `json:"top_k"`
		RepetitionPenalty *float64 `json:"repetition_penalty"`
		MaxNewTokens      *int     `json:"max_new_tokens"`
	}
	if json.Unmarshal(data, &raw) == nil {
		g.Temperature = raw.Temperature
		g.TopP = raw.TopP
		g.TopK = raw.TopK
		g.RepetitionPenalty = raw.RepetitionPenalty
		g.MaxNewTokens = raw.MaxNewTokens
	}
	return g
}

func bytesPerParam(method string, bits, groupSize int) float64 {
	if bits == 0 {
		// Defaulting silently to 4-bit was an old bug — newer formats
		// (compressed-tensors FP8, raw FP8) leave bits unset. 0 means
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

// countAttentionLayers works out how many layers hold a KV cache.
//
// Three shapes appear in the wild, in decreasing order of reliability:
//
//	layer_types: ["full_attention", "linear_attention", ...]   one entry per layer
//	full_attention_interval: 4                                 every Nth layer
//	(neither)                                                  assume dense
//
// Returns 0 when the layer count itself is unknown, so callers can tell
// "no information" from "genuinely zero".
func countAttentionLayers(src map[string]json.RawMessage, totalLayers int) int {
	if totalLayers <= 0 {
		return 0
	}

	var types []string
	if jsonFieldFrom(src, &types, "layer_types") && len(types) > 0 {
		n := 0
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
			default:
				n++
			}
		}
		return n
	}

	// Some configs give only the stride between full-attention layers.
	var interval int
	if jsonFieldFrom(src, &interval, "full_attention_interval") && interval > 1 {
		return totalLayers / interval
	}

	return totalLayers
}

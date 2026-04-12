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

	// Architectures
	jsonField(raw, &cfg.Architectures, "architectures")
	jsonField(raw, &cfg.ModelType, "model_type")
	jsonField(raw, &cfg.VocabSize, "vocab_size")
	jsonField(raw, &cfg.TorchDtype, "torch_dtype")
	jsonField(raw, &cfg.TieWordEmbeddings, "tie_word_embeddings")

	// Hidden size (with aliases)
	if !jsonField(raw, &cfg.HiddenSize, "hidden_size") {
		if !jsonField(raw, &cfg.HiddenSize, "d_model") {
			jsonField(raw, &cfg.HiddenSize, "n_embd")
		}
	}

	// Num hidden layers (with aliases)
	if !jsonField(raw, &cfg.NumHiddenLayers, "num_hidden_layers") {
		if !jsonField(raw, &cfg.NumHiddenLayers, "n_layer") {
			if !jsonField(raw, &cfg.NumHiddenLayers, "num_layers") {
				jsonField(raw, &cfg.NumHiddenLayers, "n_layers")
			}
		}
	}

	// Intermediate size
	if !jsonField(raw, &cfg.IntermediateSize, "intermediate_size") {
		jsonField(raw, &cfg.IntermediateSize, "ffn_dim")
	}

	// Attention heads
	if !jsonField(raw, &cfg.NumAttentionHeads, "num_attention_heads") {
		if !jsonField(raw, &cfg.NumAttentionHeads, "n_head") {
			jsonField(raw, &cfg.NumAttentionHeads, "num_heads")
		}
	}

	// KV heads
	if !jsonField(raw, &cfg.NumKeyValueHeads, "num_key_value_heads") {
		if !jsonField(raw, &cfg.NumKeyValueHeads, "num_kv_heads") {
			cfg.NumKeyValueHeads = cfg.NumAttentionHeads // MHA default
		}
	}

	// Head dim
	if !jsonField(raw, &cfg.HeadDim, "head_dim") {
		if cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
			cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
		}
	}

	// Max position embeddings
	if !jsonField(raw, &cfg.MaxPositionEmbeddings, "max_position_embeddings") {
		if !jsonField(raw, &cfg.MaxPositionEmbeddings, "max_sequence_length") {
			jsonField(raw, &cfg.MaxPositionEmbeddings, "n_positions")
		}
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
				QuantMethod string `json:"quant_method"`
				Bits        int    `json:"bits"`
				GroupSize   int    `json:"group_size"`
			} `json:"quantization_config"`
		}
		if json.Unmarshal(data, &cfg) == nil && cfg.QuantizationConfig != nil && cfg.QuantizationConfig.QuantMethod != "" {
			q.Method = strings.ToLower(cfg.QuantizationConfig.QuantMethod)
			q.Bits = cfg.QuantizationConfig.Bits
			q.GroupSize = cfg.QuantizationConfig.GroupSize
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
func DetectToolUse(modelDir, modelID string) ToolUseMeta {
	t := ToolUseMeta{}

	// Read chat_template from tokenizer_config.json
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
				// List of templates -- check each
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

	// Fallback: known model families
	parser := detectToolParserFromName(modelID)
	if parser != "" {
		t.HasToolSupport = true
		t.ToolCallParser = parser
		t.DetectionMethod = "known_model_family"
	}

	return t
}

func detectToolParser(template string) string {
	patterns := []struct {
		pattern *regexp.Regexp
		parser  string
	}{
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

func detectToolParserFromName(modelID string) string {
	lower := strings.ToLower(modelID)
	switch {
	case strings.Contains(lower, "hermes"):
		return "hermes"
	case strings.Contains(lower, "llama-3.1") || strings.Contains(lower, "llama-3.2") || strings.Contains(lower, "llama-3.3"):
		return "llama3_json"
	case strings.Contains(lower, "mistral") || strings.Contains(lower, "mixtral"):
		return "mistral"
	case strings.Contains(lower, "granite"):
		return "granite"
	case strings.Contains(lower, "internlm"):
		return "internlm"
	case strings.Contains(lower, "qwen2.5") || strings.Contains(lower, "qwen3"):
		return "hermes"
	case strings.Contains(lower, "jamba"):
		return "jamba"
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
		bits = 4
	}
	base := float64(bits) / 8.0
	if groupSize > 0 && (method == "gptq" || method == "awq") {
		overhead := 2.0 / float64(groupSize) * 2 // scales + zeros
		return base + overhead
	}
	return base + 0.0625 // small overhead for metadata
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

// jsonField tries to unmarshal a field from a raw JSON map.
func jsonField[T any](raw map[string]json.RawMessage, dst *T, key string) bool {
	v, ok := raw[key]
	if !ok {
		return false
	}
	return json.Unmarshal(v, dst) == nil
}

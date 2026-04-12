package models

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Model represents a registered model in the inventory.
type Model struct {
	ID            string         `json:"id"`
	DisplayName   string         `json:"display_name"`
	LocalPath     string         `json:"local_path"`
	Enabled       bool           `json:"enabled"`
	DownloadDate  time.Time      `json:"download_date"`
	TotalSizeBytes int64         `json:"total_size_bytes"`
	Orphaned      bool           `json:"orphaned,omitempty"`

	HFConfig      HFConfig       `json:"hf_config"`
	Quantization  QuantMeta      `json:"quantization"`
	ToolUse       ToolUseMeta    `json:"tool_use"`
	Vision        VisionMeta     `json:"vision"`
	GenDefaults   GenDefaults    `json:"generation_defaults,omitempty"`
	VRAMEstimate  VRAMEstimate   `json:"vram_estimate"`
	VLLMConfig    VLLMConfig     `json:"vllm_config"`
}

// HFConfig holds key fields from the model's config.json.
type HFConfig struct {
	Architectures         []string `json:"architectures,omitempty"`
	ModelType             string   `json:"model_type,omitempty"`
	NumHiddenLayers       int      `json:"num_hidden_layers,omitempty"`
	HiddenSize            int      `json:"hidden_size,omitempty"`
	IntermediateSize      int      `json:"intermediate_size,omitempty"`
	NumAttentionHeads     int      `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int      `json:"num_key_value_heads,omitempty"`
	HeadDim               int      `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings,omitempty"`
	VocabSize             int      `json:"vocab_size,omitempty"`
	TorchDtype            string   `json:"torch_dtype,omitempty"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings,omitempty"`
}

// QuantMeta holds quantization information.
type QuantMeta struct {
	Method       string  `json:"method"` // awq, gptq, fp8, gguf, bitsandbytes, marlin, squeezellm, compressed_tensors, none
	Bits         int     `json:"bits,omitempty"`
	GroupSize    int     `json:"group_size,omitempty"`
	DescAct      bool    `json:"desc_act,omitempty"`
	Sym          bool    `json:"sym,omitempty"`
	GGUFQuantType string `json:"gguf_quant_type,omitempty"`
	BytesPerParam float64 `json:"bytes_per_param"`
}

// ToolUseMeta holds tool/function calling detection info.
type ToolUseMeta struct {
	HasToolSupport  bool   `json:"has_tool_support"`
	ToolCallParser  string `json:"tool_call_parser,omitempty"`
	DetectionMethod string `json:"detection_method,omitempty"`
}

// VisionMeta holds vision model detection info.
type VisionMeta struct {
	IsVisionModel bool `json:"is_vision_model"`
}

// GenDefaults holds default generation parameters from generation_config.json.
type GenDefaults struct {
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"top_p,omitempty"`
	TopK              *int     `json:"top_k,omitempty"`
	RepetitionPenalty *float64 `json:"repetition_penalty,omitempty"`
	MaxNewTokens      *int     `json:"max_new_tokens,omitempty"`
}

// VLLMConfig holds per-model vLLM serve configuration.
type VLLMConfig struct {
	Dtype                string  `json:"dtype"`
	MaxModelLen          int     `json:"max_model_len"`
	TensorParallelSize   int     `json:"tensor_parallel_size"`
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization"`
	EnforceEager         bool    `json:"enforce_eager"`
	EnablePrefixCaching  bool    `json:"enable_prefix_caching"`
	EnableChunkedPrefill bool    `json:"enable_chunked_prefill"`
	MaxNumBatchedTokens  int     `json:"max_num_batched_tokens,omitempty"`
	MaxNumSeqs           int     `json:"max_num_seqs"`
	Quantization         string  `json:"quantization"`
	LoadFormat           string  `json:"load_format"`
	KVCacheDtype         string  `json:"kv_cache_dtype"`
	TrustRemoteCode      bool    `json:"trust_remote_code"`
	EnableAutoToolChoice bool    `json:"enable_auto_tool_choice"`
	ToolCallParser       string  `json:"tool_call_parser,omitempty"`
	Tokenizer            string  `json:"tokenizer,omitempty"`
	ChatTemplate         string  `json:"chat_template,omitempty"`
	ExtraFlags           string  `json:"extra_flags,omitempty"`
}

type registryFile struct {
	Models        map[string]*Model `json:"models"`
	SchemaVersion int               `json:"schema_version"`
	LastScan      time.Time         `json:"last_scan"`
}

// Registry manages the model inventory.
type Registry struct {
	mu       sync.RWMutex
	models   map[string]*Model
	dataDir  string
	filePath string
}

func NewRegistry(dataDir string) *Registry {
	r := &Registry{
		models:   make(map[string]*Model),
		dataDir:  dataDir,
		filePath: filepath.Join(dataDir, "config", "models.json"),
	}
	r.load()
	return r
}

func (r *Registry) load() {
	data, err := os.ReadFile(r.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return
		}
		slog.Error("failed to load models.json", "error", err)
		return
	}

	var rf registryFile
	if err := json.Unmarshal(data, &rf); err != nil {
		slog.Error("failed to parse models.json", "error", err)
		return
	}
	if rf.Models != nil {
		r.models = rf.Models
	}
}

func (r *Registry) save() error {
	rf := registryFile{
		Models:        r.models,
		SchemaVersion: 2,
		LastScan:      time.Now(),
	}
	data, err := json.MarshalIndent(rf, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(r.filePath)
	os.MkdirAll(dir, 0o755)
	return os.WriteFile(r.filePath, data, 0o644)
}

// List returns all registered models.
func (r *Registry) List() []*Model {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Model, 0, len(r.models))
	for _, m := range r.models {
		out = append(out, m)
	}
	return out
}

// Get returns a specific model by ID.
func (r *Registry) Get(id string) (*Model, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[id]
	return m, ok
}

// Register adds or updates a model in the registry.
func (r *Registry) Register(m *Model) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.models[m.ID] = m
	return r.save()
}

// Delete removes a model from the registry and optionally its files.
func (r *Registry) Delete(id string, deleteFiles bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	m, ok := r.models[id]
	if !ok {
		return fmt.Errorf("model not found: %s", id)
	}

	if deleteFiles && m.LocalPath != "" {
		os.RemoveAll(m.LocalPath)
	}

	delete(r.models, id)
	return r.save()
}

// UpdateConfig updates the vLLM config for a model.
func (r *Registry) UpdateConfig(id string, cfg VLLMConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	m, ok := r.models[id]
	if !ok {
		return fmt.Errorf("model not found: %s", id)
	}
	m.VLLMConfig = cfg
	return r.save()
}

// SetEnabled toggles a model's enabled state.
func (r *Registry) SetEnabled(id string, enabled bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	m, ok := r.models[id]
	if !ok {
		return fmt.Errorf("model not found: %s", id)
	}
	m.Enabled = enabled
	return r.save()
}

// RegisterFromDownload creates a new registry entry from a downloaded model.
func (r *Registry) RegisterFromDownload(modelID, modelDir string) error {
	// Parse config files
	hfCfg := ParseHFConfig(modelDir)
	quantMeta := DetectQuantization(modelDir, modelID)
	toolMeta := DetectToolUse(modelDir, modelID)
	visionMeta := DetectVision(modelDir, hfCfg)
	genDefaults := ParseGenDefaults(modelDir)

	// Calculate total size
	var totalSize int64
	filepath.Walk(modelDir, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			totalSize += info.Size()
		}
		return nil
	})

	// Set default vLLM config based on quant method
	vllmCfg := defaultVLLMConfig(quantMeta, toolMeta, hfCfg)

	// Auto-enable trust_remote_code for models that need it
	if needsTrustRemoteCode(modelDir, hfCfg) {
		vllmCfg.TrustRemoteCode = true
	}

	// Compute display name
	displayName := filepath.Base(modelDir)
	if parts := strings.SplitN(modelID, "/", 2); len(parts) == 2 {
		displayName = parts[1]
	}

	m := &Model{
		ID:             modelID,
		DisplayName:    displayName,
		LocalPath:      modelDir,
		Enabled:        true,
		DownloadDate:   time.Now(),
		TotalSizeBytes: totalSize,
		HFConfig:       hfCfg,
		Quantization:   quantMeta,
		ToolUse:        toolMeta,
		Vision:         visionMeta,
		GenDefaults:    genDefaults,
		VLLMConfig:     vllmCfg,
	}

	// Compute VRAM estimate
	m.VRAMEstimate = EstimateVRAM(m)

	return r.Register(m)
}

// Maintenance runs startup maintenance tasks.
func (r *Registry) Maintenance() {
	r.scanForNewModels()
	r.verifyModelPaths()
	r.backfillMetadata()

	r.mu.Lock()
	r.save()
	r.mu.Unlock()
}

func (r *Registry) scanForNewModels() {
	modelsDir := filepath.Join(r.dataDir, "models")
	orgs, _ := os.ReadDir(modelsDir)
	for _, org := range orgs {
		if !org.IsDir() {
			continue
		}
		repos, _ := os.ReadDir(filepath.Join(modelsDir, org.Name()))
		for _, repo := range repos {
			if !repo.IsDir() {
				continue
			}
			modelID := org.Name() + "/" + repo.Name()
			modelDir := filepath.Join(modelsDir, org.Name(), repo.Name())

			r.mu.RLock()
			_, exists := r.models[modelID]
			r.mu.RUnlock()

			if exists {
				continue
			}

			// Check if it looks like a model directory
			hasConfig := fileExists(filepath.Join(modelDir, "config.json"))
			hasGGUF := hasGGUFFiles(modelDir)
			hasSafetensors := hasSafetensorsFiles(modelDir)

			if hasConfig || hasGGUF || hasSafetensors {
				slog.Info("discovered unregistered model", "id", modelID)
				r.RegisterFromDownload(modelID, modelDir)
			}
		}
	}
}

func (r *Registry) verifyModelPaths() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, m := range r.models {
		if m.LocalPath == "" {
			continue
		}
		if _, err := os.Stat(m.LocalPath); os.IsNotExist(err) {
			if !m.Orphaned {
				slog.Warn("model path missing", "id", m.ID, "path", m.LocalPath)
				m.Orphaned = true
			}
		} else {
			m.Orphaned = false
		}
	}
}

func (r *Registry) backfillMetadata() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, m := range r.models {
		if m.Orphaned || m.LocalPath == "" {
			continue
		}
		if m.HFConfig.HiddenSize == 0 && fileExists(filepath.Join(m.LocalPath, "config.json")) {
			slog.Info("backfilling metadata", "id", m.ID)
			m.HFConfig = ParseHFConfig(m.LocalPath)
			m.Quantization = DetectQuantization(m.LocalPath, m.ID)
			m.ToolUse = DetectToolUse(m.LocalPath, m.ID)
			m.Vision = DetectVision(m.LocalPath, m.HFConfig)
			m.GenDefaults = ParseGenDefaults(m.LocalPath)
			m.VRAMEstimate = EstimateVRAM(m)
		}
	}
}

// needsTrustRemoteCode checks if a model requires --trust-remote-code.
func needsTrustRemoteCode(modelDir string, hfCfg HFConfig) bool {
	// Models with custom tokenizer classes need trust_remote_code
	data, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json"))
	if err == nil {
		s := string(data)
		// Custom tokenizer backends not in standard transformers
		if strings.Contains(s, "TokenizersBackend") ||
			strings.Contains(s, "AutoTokenizer") == false && strings.Contains(s, "tokenizer_class") {
			// Check if the tokenizer_class is non-standard
			var tc struct {
				TokenizerClass string `json:"tokenizer_class"`
			}
			if json.Unmarshal(data, &tc) == nil && tc.TokenizerClass != "" {
				standardClasses := map[string]bool{
					"PreTrainedTokenizerFast": true,
					"GPT2Tokenizer":          true,
					"GPT2TokenizerFast":      true,
					"LlamaTokenizer":         true,
					"LlamaTokenizerFast":     true,
					"T5Tokenizer":            true,
					"T5TokenizerFast":        true,
				}
				if !standardClasses[tc.TokenizerClass] {
					return true
				}
			}
		}
	}

	// Known model families that need it
	for _, arch := range hfCfg.Architectures {
		lower := strings.ToLower(arch)
		if strings.Contains(lower, "qwen") ||
			strings.Contains(lower, "internlm") ||
			strings.Contains(lower, "yi") ||
			strings.Contains(lower, "chatglm") ||
			strings.Contains(lower, "baichuan") {
			return true
		}
	}

	return false
}

func defaultVLLMConfig(q QuantMeta, t ToolUseMeta, h HFConfig) VLLMConfig {
	// Pick a sensible default context length rather than the model max.
	// Many models support 128K+ but that requires enormous KV cache.
	defaultCtx := 8192
	if h.MaxPositionEmbeddings > 0 && h.MaxPositionEmbeddings < defaultCtx {
		defaultCtx = h.MaxPositionEmbeddings
	}

	cfg := VLLMConfig{
		Dtype:                "auto",
		MaxModelLen:          defaultCtx,
		TensorParallelSize:   1,
		GPUMemoryUtilization: 0.90,
		MaxNumSeqs:           16,
		LoadFormat:           "auto",
		KVCacheDtype:         "auto",
	}

	switch q.Method {
	case "awq":
		cfg.Quantization = "awq"
	case "gptq":
		cfg.Quantization = "gptq"
	case "fp8":
		cfg.Quantization = "fp8"
		cfg.KVCacheDtype = "fp8"
	case "gguf":
		cfg.Quantization = "gguf"
		cfg.LoadFormat = "gguf"
	case "bitsandbytes":
		cfg.Quantization = "bitsandbytes"
		cfg.Dtype = "float16"
		cfg.EnforceEager = true
	case "marlin":
		cfg.Quantization = "marlin"
	case "squeezellm":
		cfg.Quantization = "squeezellm"
	case "compressed_tensors":
		cfg.Quantization = "compressed_tensors"
	}

	if t.HasToolSupport {
		cfg.EnableAutoToolChoice = true
		cfg.ToolCallParser = t.ToolCallParser
	}

	return cfg
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func hasGGUFFiles(dir string) bool {
	matches, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
	return len(matches) > 0
}

func hasSafetensorsFiles(dir string) bool {
	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	return len(matches) > 0
}

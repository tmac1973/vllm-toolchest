package models

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Model represents a registered model in the inventory.
type Model struct {
	ID             string    `json:"id"`
	DisplayName    string    `json:"display_name"`
	LocalPath      string    `json:"local_path"`
	Enabled        bool      `json:"enabled"`
	DownloadDate   time.Time `json:"download_date"`
	TotalSizeBytes int64     `json:"total_size_bytes"`
	Orphaned       bool      `json:"orphaned,omitempty"`

	HFConfig     HFConfig     `json:"hf_config"`
	Quantization QuantMeta    `json:"quantization"`
	ToolUse      ToolUseMeta  `json:"tool_use"`
	Vision       VisionMeta   `json:"vision"`
	GenDefaults  GenDefaults  `json:"generation_defaults,omitempty"`
	VRAMEstimate VRAMEstimate `json:"vram_estimate"`
	VLLMConfig   VLLMConfig   `json:"vllm_config"`
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

	// AttentionLayers is how many of NumHiddenLayers are full-attention and
	// therefore hold a KV cache. On a dense model this equals NumHiddenLayers;
	// on a hybrid it does not, and using the total there overstates the KV
	// cache badly -- Qwen3.8-27B is 16 attention layers out of 64, so counting
	// all of them inflates the per-token figure fourfold.
	//
	// Zero means "unknown", and callers fall back to NumHiddenLayers.
	AttentionLayers   int    `json:"attention_layers,omitempty"`
	VocabSize         int    `json:"vocab_size,omitempty"`
	TorchDtype        string `json:"torch_dtype,omitempty"`
	TieWordEmbeddings bool   `json:"tie_word_embeddings,omitempty"`
}

// QuantMeta holds quantization information.
type QuantMeta struct {
	Method        string  `json:"method"` // awq, gptq, fp8, gguf, bitsandbytes, marlin, squeezellm, compressed_tensors, none
	Bits          int     `json:"bits,omitempty"`
	GroupSize     int     `json:"group_size,omitempty"`
	DescAct       bool    `json:"desc_act,omitempty"`
	Sym           bool    `json:"sym,omitempty"`
	GGUFQuantType string  `json:"gguf_quant_type,omitempty"`
	BytesPerParam float64 `json:"bytes_per_param"`

	// WeightBlockSize is the [block_n, block_k] the weight scales are applied
	// over, for checkpoints quantized blockwise. Empty for per-tensor,
	// per-channel and per-group schemes.
	//
	// This is what decides whether kernel tuning can do anything: the tuner
	// targets the block-FP8 GEMM, and only a blockwise checkpoint reaches it.
	// "FP8" alone is not enough — a per-channel FP8 model takes a different
	// code path entirely.
	WeightBlockSize []int `json:"weight_block_size,omitempty"`
}

// IsBlockFP8 reports whether this checkpoint's linear layers run the
// block-quantized FP8 GEMM, which is the only thing kernel tuning affects.
//
// Requires both a block shape and 8-bit float weights. A blockwise INT8
// checkpoint has the first and not the second, and would gain nothing.
func (q QuantMeta) IsBlockFP8() bool {
	if len(q.WeightBlockSize) < 2 {
		return false
	}
	for _, d := range q.WeightBlockSize {
		if d <= 0 {
			return false
		}
	}
	switch q.Method {
	case "fp8":
		return true
	case "compressed-tensors", "compressed_tensors":
		// Bits is resolved from config_groups during detection.
		return q.Bits == 8
	}
	return false
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
	ReasoningParser      string  `json:"reasoning_parser,omitempty"`
	Tokenizer            string  `json:"tokenizer,omitempty"`
	ChatTemplate         string  `json:"chat_template,omitempty"`
	ExtraFlags           string  `json:"extra_flags,omitempty"`

	// AttentionBackend overrides vLLM's backend choice, e.g.
	// ROCM_AITER_UNIFIED_ATTN or R4D on the radiance image. Empty = vLLM picks.
	AttentionBackend string `json:"attention_backend,omitempty"`

	// MambaCacheMode is required alongside EnablePrefixCaching on hybrid
	// linear-attention models; "align" makes their recurrent state cacheable.
	MambaCacheMode string `json:"mamba_cache_mode,omitempty"`

	// SpeculativeConfig is the raw JSON passed to --speculative-config,
	// e.g. {"method":"mtp","num_speculative_tokens":8}.
	SpeculativeConfig string `json:"speculative_config,omitempty"`

	// CompilationConfig is the raw JSON passed to --compilation-config,
	// most usefully to trim the CUDA-graph capture ladder.
	CompilationConfig string `json:"compilation_config,omitempty"`

	// KVCacheMemory pins the KV cache pool size in bytes (0 = let vLLM size
	// it). Pinned, so clear it before measuring anything memory-related.
	KVCacheMemory int64 `json:"kv_cache_memory,omitempty"`

	// DisableAsyncScheduling passes --no-async-scheduling; required with
	// speculative configs that set disable_padded_drafter_batch.
	DisableAsyncScheduling bool `json:"disable_async_scheduling,omitempty"`

	// LanguageModelOnly serves a vision-language checkpoint text-only.
	LanguageModelOnly bool `json:"language_model_only,omitempty"`
}

// schemaVersion is the envelope version this build writes.
//
//	1  the original file
//	2  pending_configs, and the first version whose builds refuse to overwrite
//	   a file they did not fully understand
const schemaVersion = 2

type registryFile struct {
	Models        map[string]*Model `json:"models"`
	SchemaVersion int               `json:"schema_version"`
	LastScan      time.Time         `json:"last_scan"`
	// PendingConfigs are launch configs restored from a backup for models
	// that aren't installed here; see pending.go.
	PendingConfigs []PendingConfig `json:"pending_configs,omitempty"`
}

// Registry manages the model inventory.
type Registry struct {
	mu      sync.RWMutex
	models  map[string]*Model
	dataDir string
	// modelsDir is where model files live; see config.ModelsPath. Kept apart
	// from dataDir so the files can sit on another disk while models.json
	// stays with the rest of the registry state.
	modelsDir string
	// pending holds configs waiting for their model to arrive.
	pending  []PendingConfig
	filePath string

	// readOnlyReason is set when load() could not take responsibility for the
	// file it found. save() rewrites the whole file, so a load that returned
	// early used to mean the next save truncated a registry it had failed to
	// read — a corrupt file, or one from a newer build, destroyed the lot.
	// Set only during construction, so reading it needs no more than the lock
	// the caller already holds.
	readOnlyReason string
}

func NewRegistry(dataDir, modelsDir string) *Registry {
	r := &Registry{
		models:    make(map[string]*Model),
		dataDir:   dataDir,
		modelsDir: modelsDir,
		filePath:  filepath.Join(dataDir, "config", "models.json"),
	}
	r.load()
	return r
}

func (r *Registry) load() {
	data, err := os.ReadFile(r.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return // a fresh install; the first save creates it
		}
		// Unreadable is not empty.
		r.readOnlyReason = fmt.Sprintf("could not be read (%v)", err)
		slog.Error("failed to load models.json — the registry is read-only", "error", err, "path", r.filePath)
		return
	}

	var rf registryFile
	if err := json.Unmarshal(data, &rf); err != nil {
		r.readOnlyReason = fmt.Sprintf("could not be parsed (%v)", err)
		slog.Error("failed to parse models.json — the registry is read-only", "error", err, "path", r.filePath)
		return
	}

	// A newer envelope carries fields this build has no struct for. They are
	// already gone from rf, and rewriting would make that permanent. Version 0
	// is not newer: it is a file written before the field existed, or by
	// hand, and is read as current.
	if rf.SchemaVersion > schemaVersion {
		r.readOnlyReason = fmt.Sprintf("is schema version %d, and this build writes %d",
			rf.SchemaVersion, schemaVersion)
		slog.Error("models.json is newer than this build — the registry is read-only",
			"file_version", rf.SchemaVersion, "build_version", schemaVersion, "path", r.filePath)
	}

	// Everything that did parse is loaded, read-only or not: the operator
	// should see their models and the reason, not an empty page.
	if rf.Models != nil {
		r.models = rf.Models
	}
	r.pending = rf.PendingConfigs
}

// ReadOnly reports why the registry refuses to write, or "" when it does not.
func (r *Registry) ReadOnly() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.readOnlyReason
}

// writableLocked is the check every mutator makes before it changes anything.
// save() makes it again, but that alone is too late: the mutators change the
// in-memory state first, so a refused write would leave this process launching
// a config the panel reported as not saved — and Delete removes a model's
// files before it ever reaches save().
func (r *Registry) writableLocked() error {
	if r.readOnlyReason == "" {
		return nil
	}
	return fmt.Errorf("refusing to write %s: it %s — move it aside or fix it, then restart",
		r.filePath, r.readOnlyReason)
}

func (r *Registry) save() error {
	if err := r.writableLocked(); err != nil {
		return err
	}
	rf := registryFile{
		Models:         r.models,
		SchemaVersion:  schemaVersion,
		LastScan:       time.Now(),
		PendingConfigs: r.pending,
	}
	data, err := json.MarshalIndent(rf, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(r.filePath), 0o755); err != nil {
		return err
	}
	// Write-then-rename, as the benchmark store does. os.WriteFile truncates
	// first, so a crash or a full disk mid-write left a half-file that the
	// next load could not parse.
	tmp := r.filePath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, r.filePath)
}

// List returns every registered model, ordered by ID.
//
// The order matters: the registry is a map, so without the sort every caller
// got Go's randomized iteration order. The models table reshuffled its rows on
// each htmx refresh, and the benchmark and probe model pickers reordered their
// options between openings.
func (r *Registry) List() []*Model {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Model, 0, len(r.models))
	for _, m := range r.models {
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
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
	if err := r.writableLocked(); err != nil {
		return err
	}
	// Both the download path and the directory scan land here, which makes it
	// the one place a restored-but-unmatched config can be claimed.
	r.claimPendingLocked(m)
	r.models[m.ID] = m
	return r.save()
}

// Delete removes a model from the registry and optionally its files.
func (r *Registry) Delete(id string, deleteFiles bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return err
	}

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
	if err := r.writableLocked(); err != nil {
		return err
	}

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
	if err := r.writableLocked(); err != nil {
		return err
	}

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
	toolMeta := DetectToolUse(modelDir, modelID, hfCfg)
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
	if err := r.save(); err != nil {
		slog.Error("registry maintenance was not saved", "error", err)
	}
	r.mu.Unlock()
}

func (r *Registry) scanForNewModels() {
	modelsDir := r.modelsDir
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
		if !fileExists(filepath.Join(m.LocalPath, "config.json")) {
			continue
		}
		// Re-derive when the metadata was never parsed, and also when it
		// predates a field we have since started reading. AttentionLayers is
		// only ever zero on a record written before it existed -- a fresh
		// parse always sets it, falling back to the total layer count on a
		// dense model -- and leaving it zero would keep showing the old,
		// fourfold-too-high KV estimate for every hybrid already registered.
		stale := m.HFConfig.HiddenSize == 0 ||
			(m.HFConfig.AttentionLayers == 0 && m.HFConfig.NumHiddenLayers > 0)
		if stale {
			slog.Info("backfilling metadata", "id", m.ID)
			m.HFConfig = ParseHFConfig(m.LocalPath)
			m.Quantization = DetectQuantization(m.LocalPath, m.ID)
			m.ToolUse = DetectToolUse(m.LocalPath, m.ID, m.HFConfig)
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
					"GPT2Tokenizer":           true,
					"GPT2TokenizerFast":       true,
					"LlamaTokenizer":          true,
					"LlamaTokenizerFast":      true,
					"T5Tokenizer":             true,
					"T5TokenizerFast":         true,
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

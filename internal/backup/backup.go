// Package backup defines the versioned configuration backup file format and
// the engines that produce (Assemble) and consume (Parse/Apply) it.
//
// The backup carries intent, not artifacts: server preference settings, the
// runtime environment, the Radiance switches, and per-model vLLM launch
// configs keyed by HuggingFace repo ID. It deliberately excludes anything
// machine-specific or re-derivable — model weights, benchmark history, tuned
// kernels, registry metadata, and deployment-identity settings, which are
// exported for reference only and never applied.
package backup

import (
	"encoding/json"
	"sort"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Version is the current backup schema version. Parse rejects any other value.
const Version = 1

// File is the top-level backup document. Every field carries an explicit
// snake_case JSON tag: this is a versioned wire format, and the Settings
// page's client-side preview reads these exact keys.
type File struct {
	Version      int                    `json:"version"`
	ExportedAt   time.Time              `json:"exported_at"`
	Source       SourceInfo             `json:"source"` // reference only, never applied
	Settings     *Settings              `json:"settings,omitempty"`
	RuntimeEnv   *RuntimeEnv            `json:"runtime_env,omitempty"`
	Radiance     *config.RadianceConfig `json:"radiance,omitempty"`
	ModelConfigs []ModelConfigExport    `json:"model_configs,omitempty"`
}

// SourceInfo documents the origin server. Restore ignores it entirely —
// deployment identity is exported so the file describes where it came from,
// never so it can be applied to a target.
type SourceInfo struct {
	ListenAddr  string   `json:"listen_addr,omitempty"`
	VLLMPort    int      `json:"vllm_port,omitempty"`
	ExternalURL string   `json:"external_url,omitempty"`
	DataDir     string   `json:"data_dir,omitempty"`
	ModelsDir   string   `json:"models_dir,omitempty"`
	Variant     string   `json:"variant,omitempty"`
	GPUArch     string   `json:"gpu_arch,omitempty"`
	NumGPUs     int      `json:"num_gpus"`
	GPUs        []string `json:"gpus,omitempty"` // marketing names, for topology warnings
}

// Settings holds the preference fields a restore may apply. All pointers: an
// absent field means "leave the target untouched", while a present zero
// (auto_start: false) is a real value. The merge-never-deletes guarantee
// depends on that distinction.
type Settings struct {
	LogLevel            *string  `json:"log_level,omitempty"`
	GPUMemoryUtil       *float64 `json:"gpu_memory_util,omitempty"`
	MaxModelLen         *int     `json:"max_model_len,omitempty"`
	TensorParallelSize  *int     `json:"tensor_parallel_size,omitempty"`
	MaxNumSeqs          *int     `json:"max_num_seqs,omitempty"`
	DefaultDtype        *string  `json:"default_dtype,omitempty"`
	AttentionBackend    *string  `json:"attention_backend,omitempty"`
	EnforceEager        *bool    `json:"enforce_eager,omitempty"`
	EnablePrefixCache   *bool    `json:"enable_prefix_cache,omitempty"`
	ToolUseEnabled      *bool    `json:"tool_use_enabled,omitempty"`
	DefaultToolParser   *string  `json:"default_tool_parser,omitempty"`
	PreferMarlin        *bool    `json:"prefer_marlin,omitempty"`
	DefaultKVCacheDtype *string  `json:"default_kv_cache_dtype,omitempty"`
	AutoRestart         *bool    `json:"auto_restart,omitempty"`
	AutoStart           *bool    `json:"auto_start,omitempty"`
	Theme               *string  `json:"theme,omitempty"`

	// Secrets travel only with the explicit opt-in, and only when non-empty.
	HFToken *string `json:"hf_token,omitempty"`
	APIKey  *string `json:"api_key,omitempty"`
}

// RuntimeEnv mirrors config.EnvSet: curated variable values plus the
// free-form extra block.
type RuntimeEnv struct {
	Curated map[string]string `json:"curated,omitempty"`
	Extra   string            `json:"extra,omitempty"`
}

// ModelConfigExport is one model's launch config keyed by HuggingFace repo ID.
// That ID is both the registry key and the identity a restore matches on, so
// unlike llama-toolchest's equivalent there is nothing else to carry.
type ModelConfigExport struct {
	ModelID string            `json:"model_id"`
	Config  models.VLLMConfig `json:"config"`
}

// Assemble builds a backup of the current configuration state. Output is
// deterministic: identical state produces byte-identical JSON but for
// exported_at.
func Assemble(cfg *config.Config, reg *models.Registry, variant string, gpus []string, includeSecrets bool) File {
	f := File{
		Version:    Version,
		ExportedAt: time.Now().UTC(),
		Source: SourceInfo{
			ListenAddr:  cfg.ListenAddr,
			VLLMPort:    cfg.VLLMPort,
			ExternalURL: cfg.ExternalURL,
			DataDir:     cfg.DataDir,
			ModelsDir:   cfg.ModelsPath(),
			Variant:     variant,
			GPUArch:     cfg.GPUArch,
			NumGPUs:     len(gpus),
			GPUs:        gpus,
		},
	}

	f.Settings = &Settings{
		LogLevel:            ptr(cfg.LogLevel),
		GPUMemoryUtil:       ptr(cfg.GPUMemoryUtil),
		MaxModelLen:         ptr(cfg.MaxModelLen),
		TensorParallelSize:  ptr(cfg.TensorParallelSize),
		MaxNumSeqs:          ptr(cfg.MaxNumSeqs),
		DefaultDtype:        ptr(cfg.DefaultDtype),
		AttentionBackend:    ptr(cfg.AttentionBackend),
		EnforceEager:        ptr(cfg.EnforceEager),
		EnablePrefixCache:   ptr(cfg.EnablePrefixCache),
		ToolUseEnabled:      ptr(cfg.ToolUseEnabled),
		DefaultToolParser:   ptr(cfg.DefaultToolParser),
		PreferMarlin:        ptr(cfg.PreferMarlin),
		DefaultKVCacheDtype: ptr(cfg.DefaultKVCacheDtype),
		AutoRestart:         ptr(cfg.AutoRestart),
		AutoStart:           ptr(cfg.AutoStart),
		Theme:               ptr(cfg.Theme),
	}
	// Secrets: only on the explicit opt-in, and never as empty strings. An
	// emitted empty secret could blank a target's credential on restore, so
	// absence of the key is the only representation of "no secret".
	if includeSecrets {
		if cfg.HFToken != "" {
			f.Settings.HFToken = ptr(cfg.HFToken)
		}
		if cfg.APIKey != "" {
			f.Settings.APIKey = ptr(cfg.APIKey)
		}
	}

	f.RuntimeEnv = &RuntimeEnv{Curated: cfg.RuntimeEnv, Extra: cfg.RuntimeEnvExtra}

	// The Radiance switches are only meaningful on the radiance image, but
	// they travel regardless: a backup taken on a generic box and restored
	// onto a radiance one should carry whatever was configured, and every
	// value defaults to "" (leave the image's own default alone) so an
	// all-empty section applies as a no-op.
	rad := cfg.Radiance
	f.Radiance = &rad

	f.ModelConfigs = assembleModelConfigs(reg)
	return f
}

// assembleModelConfigs emits one entry per registered model, ordered by ID so
// two exports of the same state compare equal.
func assembleModelConfigs(reg *models.Registry) []ModelConfigExport {
	list := reg.List()
	out := make([]ModelConfigExport, 0, len(list))
	for _, m := range list {
		out = append(out, ModelConfigExport{ModelID: m.ID, Config: m.VLLMConfig})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ModelID < out[j].ModelID })
	return out
}

// Marshal renders the file as indented JSON.
func (f File) Marshal() ([]byte, error) {
	return json.MarshalIndent(f, "", "  ")
}

func ptr[T any](v T) *T { return &v }

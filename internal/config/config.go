package config

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	// Server
	ListenAddr string `yaml:"listen_addr"`
	DataDir    string `yaml:"data_dir"`
	LogLevel   string `yaml:"log_level"`

	// API
	APIKey      string `yaml:"api_key"`
	ExternalURL string `yaml:"external_url"`

	// HuggingFace
	HFToken string `yaml:"hf_token"`

	// vLLM connection
	VLLMPort int    `yaml:"vllm_port"`
	VLLMHost string `yaml:"vllm_host"`

	// vLLM defaults
	GPUMemoryUtil      float64 `yaml:"gpu_memory_util"`
	MaxModelLen        int     `yaml:"max_model_len"`
	TensorParallelSize int     `yaml:"tensor_parallel_size"`
	EnforceEager       bool    `yaml:"enforce_eager"`
	EnablePrefixCache  bool    `yaml:"enable_prefix_cache"`
	MaxNumSeqs         int     `yaml:"max_num_seqs"`
	DefaultDtype       string  `yaml:"default_dtype"`
	AttentionBackend   string  `yaml:"attention_backend"`

	// Tool use
	ToolUseEnabled    bool   `yaml:"tool_use_enabled"`
	DefaultToolParser string `yaml:"default_tool_parser"`

	// Quantization
	DefaultQuantFormat string `yaml:"default_quant_format"`
	PreferMarlin       bool   `yaml:"prefer_marlin"`
	DefaultKVCacheDtype string `yaml:"default_kv_cache_dtype"`

	// Process management
	AutoRestart      bool `yaml:"auto_restart"`
	StartupTimeoutS  int  `yaml:"startup_timeout_s"`
	ShutdownTimeoutS int  `yaml:"shutdown_timeout_s"`

	// Theme
	Theme string `yaml:"theme"`

	// Model storage
	ModelDir string `yaml:"model_dir"`

	// GPU architecture — used to namespace tuned kernel configs and to
	// construct vLLM's device-name filename suffix (e.g. "AMD-gfx1201").
	// Defaults to the GPU_ARCH env var the Dockerfile sets at build time.
	GPUArch string `yaml:"gpu_arch"`

	// Internal: path this config was loaded from (not serialized)
	configPath string `yaml:"-"`
}

func Load(path string) (*Config, error) {
	cfg := defaults()
	cfg.configPath = path

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			applyEnvOverrides(cfg)
			return cfg, nil
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	applyEnvOverrides(cfg)
	return cfg, nil
}

func defaults() *Config {
	return &Config{
		ListenAddr:         ":3000",
		DataDir:            "/data",
		LogLevel:           "info",
		ExternalURL:        "http://localhost:3000",
		VLLMPort:           8000,
		VLLMHost:           "127.0.0.1",
		GPUMemoryUtil:      0.90,
		TensorParallelSize: 1,
		MaxNumSeqs:         16,
		DefaultDtype:       "auto",
		AttentionBackend:   "TRITON_FLASH_ATTN",
		ToolUseEnabled:     true,
		DefaultToolParser:  "hermes",
		PreferMarlin:       true,
		DefaultKVCacheDtype: "auto",
		AutoRestart:        true,
		StartupTimeoutS:    300,
		ShutdownTimeoutS:   30,
		EnablePrefixCache:  false,
		Theme:              "dark",
	}
}

func applyEnvOverrides(cfg *Config) {
	envStr(&cfg.ListenAddr, "VLLMCTL_LISTEN_ADDR")
	envStr(&cfg.DataDir, "VLLMCTL_DATA_DIR")
	envStr(&cfg.LogLevel, "VLLMCTL_LOG_LEVEL")
	envStr(&cfg.APIKey, "VLLMCTL_API_KEY")
	envStr(&cfg.ExternalURL, "VLLMCTL_EXTERNAL_URL")

	// HF token: check both our prefix and the standard HF_TOKEN
	envStr(&cfg.HFToken, "VLLMCTL_HF_TOKEN")
	envStr(&cfg.HFToken, "HF_TOKEN")

	envInt(&cfg.VLLMPort, "VLLMCTL_VLLM_PORT")
	envStr(&cfg.VLLMHost, "VLLMCTL_VLLM_HOST")
	envFloat(&cfg.GPUMemoryUtil, "VLLMCTL_GPU_MEMORY_UTIL")
	envInt(&cfg.MaxModelLen, "VLLMCTL_MAX_MODEL_LEN")
	envInt(&cfg.TensorParallelSize, "VLLMCTL_TENSOR_PARALLEL_SIZE")
	envBool(&cfg.EnforceEager, "VLLMCTL_ENFORCE_EAGER")
	envBool(&cfg.EnablePrefixCache, "VLLMCTL_ENABLE_PREFIX_CACHE")
	envInt(&cfg.MaxNumSeqs, "VLLMCTL_MAX_NUM_SEQS")
	envStr(&cfg.DefaultDtype, "VLLMCTL_DEFAULT_DTYPE")
	envStr(&cfg.AttentionBackend, "VLLMCTL_ATTENTION_BACKEND")
	envBool(&cfg.ToolUseEnabled, "VLLMCTL_TOOL_USE_ENABLED")
	envStr(&cfg.DefaultToolParser, "VLLMCTL_TOOL_CALL_PARSER")
	envStr(&cfg.DefaultQuantFormat, "VLLMCTL_DEFAULT_QUANT_FORMAT")
	envBool(&cfg.PreferMarlin, "VLLMCTL_PREFER_MARLIN")
	envStr(&cfg.DefaultKVCacheDtype, "VLLMCTL_DEFAULT_KV_CACHE_DTYPE")
	envBool(&cfg.AutoRestart, "VLLMCTL_AUTO_RESTART")
	envInt(&cfg.StartupTimeoutS, "VLLMCTL_STARTUP_TIMEOUT_S")
	envInt(&cfg.ShutdownTimeoutS, "VLLMCTL_SHUTDOWN_TIMEOUT_S")
	envStr(&cfg.Theme, "VLLMCTL_THEME")
	envStr(&cfg.ModelDir, "VLLMCTL_MODEL_DIR")
	// The Dockerfile sets GPU_ARCH as a build arg → ENV. Honor it as the
	// default so the operator doesn't have to duplicate it in vllmctl.yaml.
	envStr(&cfg.GPUArch, "VLLMCTL_GPU_ARCH")
	envStr(&cfg.GPUArch, "GPU_ARCH")
}

// DeviceNameSuffix is the value vLLM expects in tuned kernel config
// filenames (e.g. "AMD-gfx1201"). It matches what kyuz0's RDNA4 patches
// make rocm.py's get_device_name return.
func (c *Config) DeviceNameSuffix() string {
	if c.GPUArch == "" {
		return ""
	}
	return "AMD-" + c.GPUArch
}

func (c *Config) Save(path string) error {
	if path == "" {
		path = c.configPath
	}
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func (c *Config) ConfigPath() string {
	return c.configPath
}

func envStr(dst *string, key string) {
	if v := os.Getenv(key); v != "" {
		*dst = v
	}
}

func envInt(dst *int, key string) {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			*dst = n
		}
	}
}

func envFloat(dst *float64, key string) {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			*dst = f
		}
	}
}

func envBool(dst *bool, key string) {
	if v := os.Getenv(key); v != "" {
		*dst = strings.EqualFold(v, "true") || v == "1"
	}
}

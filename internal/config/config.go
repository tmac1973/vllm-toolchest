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

	// VLLMDeviceName is the exact string vLLM interpolates into tuned-kernel
	// filenames. Left empty it is probed from the running vLLM at boot and
	// cached here; the GPUArch-derived value is only a last-resort fallback.
	//
	// This must not be guessed: the generic image patches get_device_name to
	// return "AMD-gfx1201", while the radiance image leaves it reporting the
	// marketing name ("AMD_Radeon_R9700"). Tuning against the wrong one
	// produces correctly-formatted files vLLM never reads.
	VLLMDeviceName string `yaml:"vllm_device_name"`

	// Radiance holds the RDNA4 image's feature switches. Only meaningful on
	// the radiance image variant; empty values leave the image default alone.
	Radiance RadianceConfig `yaml:"radiance"`

	// Internal: path this config was loaded from (not serialized)
	configPath string `yaml:"-"`
}

// RadianceConfig mirrors the RADIANCE_* environment knobs the vllm-radiance
// image reads. Every field is a string with "" meaning "don't set it, use the
// image's baked-in default" — the image is the source of truth for defaults,
// and hardcoding them here would silently drift as radiance is updated.
//
// See the vllm-radiance DOCKERHUB.md for the full reference.
type RadianceConfig struct {
	// UseR4D is the master switch for the hand-written gfx1201 kernel
	// library (attention, gated delta net, vision attention, all-reduce,
	// skinny GEMM). "0" reverts every one of them to the stock path.
	UseR4D string `yaml:"use_r4d"`

	// UseR4DAllReduce toggles the TP=2 P2P one-shot all-reduce; "0" keeps
	// RCCL. Bit-identical to RCCL either way.
	UseR4DAllReduce string `yaml:"use_r4d_ar"`

	// AllReduceQuant compresses large (prefill) all-reduce payloads. Not
	// bit-identical to RCCL; "0" gives the exact bf16 all-reduce.
	AllReduceQuant string `yaml:"use_r4d_ar_quant"`

	// Preshuffle enables the preshuffled block-FP8 GEMM weight layout.
	Preshuffle string `yaml:"preshuffle"`

	// FuseRMSQuant enables the fused RMSNorm + group-FP8-quant path.
	FuseRMSQuant string `yaml:"fuse_rms_quant"`

	// SkinnyGEMM routes small-M bf16 projections to the R4D split-K kernel.
	// "all" adds shapes that differ from rocBLAS at a bf16 ULP.
	SkinnyGEMM string `yaml:"skinny_gemm"`

	// DynamicDraft varies MTP draft depth per request by confidence. "0" is
	// byte-identical stock MTP.
	DynamicDraft string `yaml:"dynamic_draft"`

	// DraftSchedule caps serial MTP forwards by batch size, e.g.
	// "1:8,2:7,4:6,8:5,16:4".
	DraftSchedule string `yaml:"draft_schedule"`

	// DraftTau is the per-request confidence gate. Keep in sync with
	// FastDraft: radiance tunes 0.28 for the 2-bit head and 0.35 for the
	// stock bf16 one.
	DraftTau string `yaml:"draft_tau"`

	// FastDraft enables the 2-bit draft head with exact rerank, plus 4-bit
	// dflash drafter weights.
	FastDraft string `yaml:"fast_draft"`

	// RunBWTest runs the startup topology + bandwidth sweep. Backgrounded,
	// about a second; "0" skips it.
	RunBWTest string `yaml:"run_bwtest"`

	// NumaBind pins the vLLM fleet to the GPU-local NUMA node(s):
	// "auto", explicit nodes ("0" / "0,1"), or "" for off.
	NumaBind string `yaml:"numa_bind"`
}

// Env renders the non-empty knobs as KEY=VALUE strings for the vLLM process.
func (r RadianceConfig) Env() []string {
	pairs := []struct{ key, val string }{
		{"RADIANCE_USE_R4D", r.UseR4D},
		{"RADIANCE_USE_R4D_AR", r.UseR4DAllReduce},
		{"RADIANCE_USE_R4D_AR_QUANT", r.AllReduceQuant},
		{"RADIANCE_PRESHUFFLE", r.Preshuffle},
		{"RADIANCE_FUSE_RMS_QUANT", r.FuseRMSQuant},
		{"RADIANCE_SKINNY_GEMM", r.SkinnyGEMM},
		{"RADIANCE_DYNAMIC_DRAFT", r.DynamicDraft},
		{"RADIANCE_DRAFT_SCHEDULE", r.DraftSchedule},
		{"RADIANCE_DRAFT_TAU", r.DraftTau},
		{"RADIANCE_FAST_DRAFT", r.FastDraft},
		{"RADIANCE_RUN_BWTEST", r.RunBWTest},
		{"RADIANCE_NUMA_BIND", r.NumaBind},
	}
	var env []string
	for _, p := range pairs {
		if p.val != "" {
			env = append(env, p.key+"="+p.val)
		}
	}
	return env
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
		// Empty = let vLLM pick. Only emitted as --attention-backend when
		// explicitly set, so the default install keeps vLLM's own choice.
		AttentionBackend:   "",
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
	envStr(&cfg.VLLMDeviceName, "VLLMCTL_VLLM_DEVICE_NAME")

	// Radiance knobs. The image already exports sane RADIANCE_* defaults, so
	// these exist to let an operator override them from .env / compose
	// without editing vllmctl.yaml.
	envStr(&cfg.Radiance.UseR4D, "RADIANCE_USE_R4D")
	envStr(&cfg.Radiance.UseR4DAllReduce, "RADIANCE_USE_R4D_AR")
	envStr(&cfg.Radiance.AllReduceQuant, "RADIANCE_USE_R4D_AR_QUANT")
	envStr(&cfg.Radiance.Preshuffle, "RADIANCE_PRESHUFFLE")
	envStr(&cfg.Radiance.FuseRMSQuant, "RADIANCE_FUSE_RMS_QUANT")
	envStr(&cfg.Radiance.SkinnyGEMM, "RADIANCE_SKINNY_GEMM")
	envStr(&cfg.Radiance.DynamicDraft, "RADIANCE_DYNAMIC_DRAFT")
	envStr(&cfg.Radiance.DraftSchedule, "RADIANCE_DRAFT_SCHEDULE")
	envStr(&cfg.Radiance.DraftTau, "RADIANCE_DRAFT_TAU")
	envStr(&cfg.Radiance.FastDraft, "RADIANCE_FAST_DRAFT")
	envStr(&cfg.Radiance.RunBWTest, "RADIANCE_RUN_BWTEST")
	envStr(&cfg.Radiance.NumaBind, "RADIANCE_NUMA_BIND")
}

// DeviceNameSuffix is the value vLLM expects in tuned kernel config
// filenames (e.g. "AMD-gfx1201" on the generic image, "AMD_Radeon_R9700" on
// radiance).
//
// A name probed from the running vLLM always wins. The GPUArch-derived value
// is the fallback, and is only correct on images carrying kyuz0's RDNA4 patch
// that makes rocm.py's get_device_name return the gfx target.
func (c *Config) DeviceNameSuffix() string {
	if c.VLLMDeviceName != "" {
		return c.VLLMDeviceName
	}
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

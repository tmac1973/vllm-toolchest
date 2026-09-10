package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/tmac1973/vllm-toolchest/variants"
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
	DefaultQuantFormat  string `yaml:"default_quant_format"`
	PreferMarlin        bool   `yaml:"prefer_marlin"`
	DefaultKVCacheDtype string `yaml:"default_kv_cache_dtype"`

	// Process management
	AutoRestart      bool `yaml:"auto_restart"`
	StartupTimeoutS  int  `yaml:"startup_timeout_s"`
	ShutdownTimeoutS int  `yaml:"shutdown_timeout_s"`

	// AutoStart launches the active model when the container starts, without
	// waiting for someone to press Start. Off by default: a model that fails
	// to load takes a while to fail, and a container that does nothing until
	// asked is easier to diagnose than one that is busy on boot.
	AutoStart bool `yaml:"auto_start"`

	// ActiveModel is the registry ID the Start button launches, and what a
	// restart brings back. Empty means nothing has been chosen yet.
	ActiveModel string `yaml:"active_model"`

	// Theme
	Theme string `yaml:"theme"`

	// Model storage
	ModelDir string `yaml:"model_dir"`

	// RuntimeEnv holds the curated environment variables applied to the vLLM
	// process, keyed by variable name. RuntimeEnvExtra is the free-form
	// KEY=VALUE block for anything outside the curated set. See
	// runtime_env.go — an unset curated value means "leave the environment
	// alone", which is not the same as setting it empty.
	RuntimeEnv      map[string]string `yaml:"runtime_env,omitempty"`
	RuntimeEnvExtra string            `yaml:"runtime_env_extra,omitempty"`

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

	// Knobs holds the image-variant feature switches, keyed by variant id
	// and then by knob id, as declared in variants/<id>.conf. An absent
	// value leaves the image's own default alone, which is not the same as
	// setting it empty.
	//
	// Keyed by variant rather than flat because a config outlives the image
	// it was written on: rebuilding onto a different variant, or restoring a
	// backup taken on one, must not discard the switches belonging to the
	// variant that is not currently running.
	Knobs map[string]map[string]string `yaml:"knobs,omitempty"`

	// LegacyRadiance is the pre-manifest `radiance:` block. It is read on
	// load, folded into Knobs["radiance"], and never written back — see
	// migrateLegacyKnobs.
	LegacyRadiance map[string]string `yaml:"radiance,omitempty"`

	// Internal: path this config was loaded from (not serialized)
	configPath string `yaml:"-"`
}

// KnobEnv renders the knobs configured for variantID as KEY=VALUE strings, in
// manifest order, skipping any that are unset.
//
// Unset means "leave the image's own default in place". That is deliberately
// not the same as setting the variable empty: several images read an empty
// RADIANCE_*-style switch as "off", so emitting one would silently disable the
// feature the image exists for.
//
// A variant with no manifest — an image built before its manifest existed, or
// an operator override naming something we do not ship — contributes nothing
// rather than erroring. There is no environment we could correctly emit for a
// variant we cannot describe.
func (c *Config) KnobEnv(variantID string) []string {
	d, ok := variants.Get(variantID)
	if !ok {
		return nil
	}
	vals := c.Knobs[variantID]
	var env []string
	for _, k := range d.Knobs {
		if v := strings.TrimSpace(vals[k.ID]); v != "" {
			env = append(env, k.Env+"="+v)
		}
	}
	return env
}

// KnobValues returns one variant's configured knobs. The returned map is a
// copy, so a caller rendering the Settings page cannot mutate stored config.
func (c *Config) KnobValues(variantID string) map[string]string {
	out := map[string]string{}
	for id, v := range c.Knobs[variantID] {
		out[id] = v
	}
	return out
}

// SetKnobs replaces one variant's knobs, dropping empty values so that "image
// default" round-trips as an absent key rather than an empty string.
func (c *Config) SetKnobs(variantID string, vals map[string]string) {
	cleaned := map[string]string{}
	for id, v := range vals {
		if v = strings.TrimSpace(v); v != "" {
			cleaned[id] = v
		}
	}
	if len(cleaned) == 0 {
		delete(c.Knobs, variantID)
		return
	}
	if c.Knobs == nil {
		c.Knobs = map[string]map[string]string{}
	}
	c.Knobs[variantID] = cleaned
}

// KnobSet is one variant's worth of knob values, for validation before save.
type KnobSet struct {
	Variant string
	Values  map[string]string
}

// Validate rejects a knob this variant does not declare, and a select knob set
// to a value outside its option list.
//
// Text knobs accept anything: the image is the authority on what it parses,
// and refusing a value here because we do not recognise it would make a knob
// unusable the moment upstream extends it.
func (s KnobSet) Validate() error {
	d, ok := variants.Get(s.Variant)
	if !ok {
		if len(s.Values) == 0 {
			return nil
		}
		return fmt.Errorf("no manifest describes variant %q, so it has no knobs to set", s.Variant)
	}
	for id, v := range s.Values {
		k, known := d.Knob(id)
		if !known {
			return fmt.Errorf("%s has no knob %q", s.Variant, id)
		}
		if k.Kind != variants.KindSelect || v == "" {
			continue
		}
		valid := false
		var allowed []string
		for _, o := range k.Options {
			if o.Value == "" {
				continue
			}
			allowed = append(allowed, o.Value)
			if o.Value == v {
				valid = true
			}
		}
		if !valid {
			return fmt.Errorf("%s: %q is not a valid value (want one of %s, or empty for the image default)",
				k.Label, v, strings.Join(allowed, ", "))
		}
	}
	return nil
}

// migrateLegacyKnobs folds the pre-manifest `radiance:` block into
// knobs.radiance.
//
// The old block's yaml keys are the knob ids — that is not a coincidence, the
// manifest slugs were named after them — so this is a reparent, key for key,
// with no renaming and no value transformation. It cannot lose a setting.
//
// A value already under knobs wins: a file written by a current build is
// authoritative over a stale legacy block someone left behind by hand. Keys
// the manifest no longer declares are dropped rather than carried, which is
// why LegacyRadiance is a map and not the old struct — unmarshalling into the
// struct would have failed outright on a field that had been removed.
//
// Nilling LegacyRadiance is what retires the old shape: `omitempty` means the
// next Save writes no `radiance:` key at all. The reading side stays for good,
// though. It is a few lines, and dropping it would make an old config lose
// every knob silently rather than fail loudly.
func migrateLegacyKnobs(cfg *Config) {
	if len(cfg.LegacyRadiance) == 0 {
		return
	}
	defer func() { cfg.LegacyRadiance = nil }()

	d, ok := variants.Get("radiance")
	if !ok {
		return
	}
	vals := cfg.KnobValues(d.ID)
	for _, k := range d.Knobs {
		if _, exists := vals[k.ID]; exists {
			continue
		}
		if v := strings.TrimSpace(cfg.LegacyRadiance[k.ID]); v != "" {
			vals[k.ID] = v
		}
	}
	cfg.SetKnobs(d.ID, vals)
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

	// Before the env overrides, so a knob set in the container environment
	// still beats one stored in the file — the precedence the hand-written
	// version had.
	migrateLegacyKnobs(cfg)

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
		AttentionBackend:    "",
		ToolUseEnabled:      true,
		DefaultToolParser:   "hermes",
		PreferMarlin:        true,
		DefaultKVCacheDtype: "auto",
		AutoRestart:         true,
		StartupTimeoutS:     300,
		ShutdownTimeoutS:    30,
		EnablePrefixCache:   false,
		Theme:               "dark",
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
	envBool(&cfg.AutoStart, "VLLMCTL_AUTO_START")
	envInt(&cfg.StartupTimeoutS, "VLLMCTL_STARTUP_TIMEOUT_S")
	envInt(&cfg.ShutdownTimeoutS, "VLLMCTL_SHUTDOWN_TIMEOUT_S")
	envStr(&cfg.Theme, "VLLMCTL_THEME")
	envStr(&cfg.ModelDir, "VLLMCTL_MODEL_DIR")
	// The Dockerfile sets GPU_ARCH as a build arg → ENV. Honor it as the
	// default so the operator doesn't have to duplicate it in vllmctl.yaml.
	envStr(&cfg.GPUArch, "VLLMCTL_GPU_ARCH")
	envStr(&cfg.GPUArch, "GPU_ARCH")
	envStr(&cfg.VLLMDeviceName, "VLLMCTL_VLLM_DEVICE_NAME")

	applyKnobEnvOverrides(cfg)
}

// applyKnobEnvOverrides lets an operator set a variant's feature knobs from
// .env or compose without editing vllmctl.yaml. The images export their own
// sane defaults, so this is an override path, not a source of defaults.
//
// It walks every variant rather than only the running one: config.Load has no
// business knowing which image it is inside, and env names are unique across
// manifests by lint, so a variable can always be attributed to exactly one
// knob. Setting a knob for a variant you are not running is harmless — nothing
// reads it until you rebuild onto that image.
func applyKnobEnvOverrides(cfg *Config) {
	for _, d := range variants.All() {
		vals := cfg.KnobValues(d.ID)
		changed := false
		for _, k := range d.Knobs {
			if v := os.Getenv(k.Env); v != "" {
				vals[k.ID] = v
				changed = true
			}
		}
		if changed {
			cfg.SetKnobs(d.ID, vals)
		}
	}
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

// DefaultModelsPath is where model files live when no override is set: a
// "models" directory under the data directory.
func (c *Config) DefaultModelsPath() string {
	return filepath.Join(c.DataDir, "models")
}

// ModelsPath is the directory model files are downloaded into and scanned
// from. An override is only honoured when it is an absolute path — a relative
// one would resolve against the process's working directory, which is not
// something the operator can see or reason about from the Settings page.
func (c *Config) ModelsPath() string {
	if d := strings.TrimSpace(c.ModelDir); d != "" && filepath.IsAbs(d) {
		return filepath.Clean(d)
	}
	return c.DefaultModelsPath()
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

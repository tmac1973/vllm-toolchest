package config

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type Config struct {
	ListenAddr         string  `yaml:"listen_addr"`
	DataDir            string  `yaml:"data_dir"`
	VLLMPort           int     `yaml:"vllm_port"`
	VLLMHost           string  `yaml:"vllm_host"`
	ExternalURL        string  `yaml:"external_url"`
	HFToken            string  `yaml:"hf_token"`
	APIKey             string  `yaml:"api_key"`
	LogLevel           string  `yaml:"log_level"`
	ToolUseEnabled     bool    `yaml:"tool_use_enabled"`
	DefaultToolParser  string  `yaml:"default_tool_parser"`
	DefaultQuantFormat string  `yaml:"default_quant_format"`
	MaxModelLen        int     `yaml:"max_model_len"`
	TensorParallelSize int     `yaml:"tensor_parallel_size"`
	GPUMemoryUtil      float64 `yaml:"gpu_memory_util"`
	EnforceEager       bool    `yaml:"enforce_eager"`
}

func Load(path string) (*Config, error) {
	cfg := &Config{
		ListenAddr:         ":3000",
		DataDir:            "/data",
		VLLMPort:           8000,
		VLLMHost:           "127.0.0.1",
		ExternalURL:        "http://localhost:3000",
		LogLevel:           "info",
		ToolUseEnabled:     true,
		DefaultToolParser:  "hermes",
		TensorParallelSize: 1,
		GPUMemoryUtil:      0.90,
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Config) Save(path string) error {
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

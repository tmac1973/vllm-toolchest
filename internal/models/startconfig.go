package models

import "github.com/tmac1973/vllm-toolchest/internal/process"

// StartConfig projects a stored per-model config onto the struct BuildArgs
// consumes.
//
// This exists because the same projection was open-coded at every call site
// (start, restart, benchmark jobs, the context probe, the effective-command
// preview) and they had already drifted apart — restart silently dropped
// chunked prefill, the batched-token budget, the tokenizer and the chat
// template. Callers that vary a field (the probe sweeps context length; a
// benchmark job overrides from its snapshot) should call this and then assign
// over the handful of fields they own.
func (c VLLMConfig) StartConfig() process.VLLMStartConfig {
	return process.VLLMStartConfig{
		Dtype:                  c.Dtype,
		MaxModelLen:            c.MaxModelLen,
		TensorParallelSize:     c.TensorParallelSize,
		GPUMemoryUtilization:   c.GPUMemoryUtilization,
		EnforceEager:           c.EnforceEager,
		TrustRemoteCode:        c.TrustRemoteCode,
		MaxNumSeqs:             c.MaxNumSeqs,
		Quantization:           c.Quantization,
		LoadFormat:             c.LoadFormat,
		EnablePrefixCaching:    c.EnablePrefixCaching,
		KVCacheDtype:           c.KVCacheDtype,
		EnableChunkedPrefill:   c.EnableChunkedPrefill,
		MaxNumBatchedTokens:    c.MaxNumBatchedTokens,
		EnableAutoToolChoice:   c.EnableAutoToolChoice,
		ToolCallParser:         c.ToolCallParser,
		ReasoningParser:        c.ReasoningParser,
		AttentionBackend:       c.AttentionBackend,
		MambaCacheMode:         c.MambaCacheMode,
		SpeculativeConfig:      c.SpeculativeConfig,
		CompilationConfig:      c.CompilationConfig,
		KVCacheMemory:          c.KVCacheMemory,
		DisableAsyncScheduling: c.DisableAsyncScheduling,
		LanguageModelOnly:      c.LanguageModelOnly,
		Tokenizer:              c.Tokenizer,
		ChatTemplate:           c.ChatTemplate,
		ExtraFlags:             c.ExtraFlags,
	}
}

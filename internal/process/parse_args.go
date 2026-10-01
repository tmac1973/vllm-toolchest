package process

import (
	"strconv"
	"strings"
)

// serveFlagFields is every flag BuildArgs writes, by the field it comes from.
// ParseArgs reads them back; the round-trip test keeps the two in step.
var (
	serveBoolFlags = map[string]func(*VLLMStartConfig){
		"--enforce-eager":           func(c *VLLMStartConfig) { c.EnforceEager = true },
		"--trust-remote-code":       func(c *VLLMStartConfig) { c.TrustRemoteCode = true },
		"--enable-prefix-caching":   func(c *VLLMStartConfig) { c.EnablePrefixCaching = true },
		"--enable-chunked-prefill":  func(c *VLLMStartConfig) { c.EnableChunkedPrefill = true },
		"--enable-auto-tool-choice": func(c *VLLMStartConfig) { c.EnableAutoToolChoice = true },
		"--no-async-scheduling":     func(c *VLLMStartConfig) { c.DisableAsyncScheduling = true },
		"--language-model-only":     func(c *VLLMStartConfig) { c.LanguageModelOnly = true },
	}
	serveValueFlags = map[string]func(*VLLMStartConfig, string) bool{
		"--dtype":                  func(c *VLLMStartConfig, v string) bool { c.Dtype = v; return true },
		"--max-model-len":          func(c *VLLMStartConfig, v string) bool { return parseInt(v, &c.MaxModelLen) },
		"--tensor-parallel-size":   func(c *VLLMStartConfig, v string) bool { return parseInt(v, &c.TensorParallelSize) },
		"-tp":                      func(c *VLLMStartConfig, v string) bool { return parseInt(v, &c.TensorParallelSize) },
		"--gpu-memory-utilization": func(c *VLLMStartConfig, v string) bool { return parseFloat(v, &c.GPUMemoryUtilization) },
		"--max-num-seqs":           func(c *VLLMStartConfig, v string) bool { return parseInt(v, &c.MaxNumSeqs) },
		"--quantization":           func(c *VLLMStartConfig, v string) bool { c.Quantization = v; return true },
		"--load-format":            func(c *VLLMStartConfig, v string) bool { c.LoadFormat = v; return true },
		"--kv-cache-dtype":         func(c *VLLMStartConfig, v string) bool { c.KVCacheDtype = v; return true },
		"--max-num-batched-tokens": func(c *VLLMStartConfig, v string) bool { return parseInt(v, &c.MaxNumBatchedTokens) },
		"--tool-call-parser":       func(c *VLLMStartConfig, v string) bool { c.ToolCallParser = v; return true },
		"--reasoning-parser":       func(c *VLLMStartConfig, v string) bool { c.ReasoningParser = v; return true },
		"--attention-backend":      func(c *VLLMStartConfig, v string) bool { c.AttentionBackend = v; return true },
		"--mamba-cache-mode":       func(c *VLLMStartConfig, v string) bool { c.MambaCacheMode = v; return true },
		"--speculative-config":     func(c *VLLMStartConfig, v string) bool { c.SpeculativeConfig = v; return true },
		"--compilation-config":     func(c *VLLMStartConfig, v string) bool { c.CompilationConfig = v; return true },
		"--kv-cache-memory": func(c *VLLMStartConfig, v string) bool {
			n, err := strconv.ParseInt(v, 10, 64)
			c.KVCacheMemory = n
			return err == nil
		},
		"--tokenizer":     func(c *VLLMStartConfig, v string) bool { c.Tokenizer = v; return true },
		"--chat-template": func(c *VLLMStartConfig, v string) bool { c.ChatTemplate = v; return true },
	}
	// serveOwnedFlags are set by this tool, or describe the author's machine,
	// and are dropped rather than carried: a card's port or served name is
	// not this host's.
	serveOwnedFlags = map[string]bool{
		"--host": true, "--port": true, "--served-model-name": true, "--api-key": true,
		"--model": true, "--download-dir": true, "--uvicorn-log-level": true,
	}
)

func parseInt(v string, dst *int) bool {
	n, err := strconv.Atoi(v)
	*dst = n
	return err == nil
}

func parseFloat(v string, dst *float64) bool {
	f, err := strconv.ParseFloat(v, 64)
	*dst = f
	return err == nil
}

// ParseArgs reads serve arguments back into the fields BuildArgs writes
// them from: the inverse of BuildArgs, for a command someone else wrote.
//
// Flags this tool owns (--port, --served-model-name and the like) are dropped
// with their values. Everything else -- a flag with no field here, a known
// flag whose value does not parse, a stray positional -- is returned in rest,
// in order, each flag followed by its value.
//
// A flag takes a value when it is written --flag=value, or when the next
// argument does not begin with "-". That is the only rule available without
// the engine's own parser, and it is right for every flag BuildArgs writes.
func ParseArgs(args []string) (cfg VLLMStartConfig, rest []string) {
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !strings.HasPrefix(a, "-") {
			rest = append(rest, a)
			continue
		}
		name, value, inline := strings.Cut(a, "=")
		takeValue := func() (string, bool) {
			if inline {
				return value, true
			}
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				i++
				return args[i], true
			}
			return "", false
		}

		switch {
		case serveOwnedFlags[name]:
			takeValue()
		case serveBoolFlags[name] != nil && !inline:
			serveBoolFlags[name](&cfg)
		case serveValueFlags[name] != nil:
			v, ok := takeValue()
			if ok && serveValueFlags[name](&cfg, v) {
				continue
			}
			// Known but unreadable: kept as written, so nothing is lost.
			rest = append(rest, name)
			if ok {
				rest = append(rest, v)
			}
		default:
			rest = append(rest, name)
			if v, ok := takeValue(); ok {
				rest = append(rest, v)
			}
		}
	}
	return cfg, rest
}

// KnownServeFlags is every flag ParseArgs recognises, mapped or owned. A run
// of arguments with two or more of these in it is a vLLM serve command, which
// is how one is told apart from any other container's arguments.
func KnownServeFlags() []string {
	var out []string
	for f := range serveBoolFlags {
		out = append(out, f)
	}
	for f := range serveValueFlags {
		out = append(out, f)
	}
	for f := range serveOwnedFlags {
		out = append(out, f)
	}
	return out
}

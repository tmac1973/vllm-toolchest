package autoconfig

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// DefaultTicks is each row's own starting tick, by key.
func DefaultTicks(rows []Row) map[string]bool {
	out := make(map[string]bool, len(rows))
	for _, r := range rows {
		out[r.Key] = r.Ticked
	}
	return out
}

// Apply writes the ticked rows onto base. A nil ticked means each row's own
// default.
//
// Field rows set their field; flag rows go into the extra flags, replacing
// the same flag if base had it and keeping the rest; env rows go into the env
// block by name. Only the fields named here can be set, which is what keeps a
// helper's answer from reaching anything else.
func Apply(base models.VLLMConfig, rows []Row, ticked map[string]bool) (models.VLLMConfig, error) {
	if ticked == nil {
		ticked = DefaultTicks(rows)
	}
	c := base
	for _, r := range rows {
		if !ticked[r.Key] {
			continue
		}
		switch r.Kind {
		case RowField:
			if err := setField(&c, r.Field, r.Value); err != nil {
				return base, err
			}
		case RowFlag:
			c.ExtraFlags = process.SetFlag(c.ExtraFlags, r.Flag)
		case RowEnv:
			c.Env = setEnvLine(c.Env, r.Env)
		default:
			return base, fmt.Errorf("unknown row kind %q", r.Kind)
		}
	}
	return c, nil
}

func setField(c *models.VLLMConfig, field, value string) error {
	boolean := func(dst *bool) error {
		v, err := strconv.ParseBool(value)
		if err != nil {
			return fmt.Errorf("%s: %q is not true or false", field, value)
		}
		*dst = v
		return nil
	}
	switch field {
	case "dtype":
		c.Dtype = value
	case "enforce_eager":
		return boolean(&c.EnforceEager)
	case "trust_remote_code":
		return boolean(&c.TrustRemoteCode)
	case "quantization":
		c.Quantization = value
	case "load_format":
		c.LoadFormat = value
	case "enable_prefix_caching":
		return boolean(&c.EnablePrefixCaching)
	case "enable_chunked_prefill":
		return boolean(&c.EnableChunkedPrefill)
	case "max_num_batched_tokens":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_num_batched_tokens: %q is not a positive number", value)
		}
		c.MaxNumBatchedTokens = n
	case "enable_auto_tool_choice":
		return boolean(&c.EnableAutoToolChoice)
	case "tool_call_parser":
		c.ToolCallParser = value
	case "reasoning_parser":
		c.ReasoningParser = value
	case "attention_backend":
		c.AttentionBackend = value
	case "mamba_cache_mode":
		c.MambaCacheMode = value
	case "speculative_config":
		c.SpeculativeConfig = value
	case "compilation_config":
		c.CompilationConfig = value
	case "disable_async_scheduling":
		return boolean(&c.DisableAsyncScheduling)
	case "language_model_only":
		return boolean(&c.LanguageModelOnly)
	case "tokenizer":
		c.Tokenizer = value
	default:
		return fmt.Errorf("autoconfigure cannot set %q", field)
	}
	return nil
}

// setEnvLine puts KEY=VALUE into an env block, replacing the line with the
// same key or appending.
func setEnvLine(block, line string) string {
	key, _, _ := strings.Cut(line, "=")
	var out []string
	replaced := false
	for _, l := range strings.Split(block, "\n") {
		t := strings.TrimSpace(l)
		if t == "" {
			continue
		}
		if k, _, _ := strings.Cut(t, "="); k == key {
			if !replaced {
				out = append(out, line)
				replaced = true
			}
			continue
		}
		out = append(out, t)
	}
	if !replaced {
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

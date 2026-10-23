package models

import (
	"encoding/json"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// MaxContext is the longest --max-model-len the engine will accept for this
// model as configured: the trained max_position_embeddings, stretched by a
// YaRN factor when the extra flags pass one through --hf-overrides. vLLM
// derives its own ceiling the same way, so a card that says to double the
// context with YaRN (and sets the override) can be served at that length.
// Zero when the trained length is unknown.
func MaxContext(m *Model) int {
	trained := m.HFConfig.MaxPositionEmbeddings
	if trained <= 0 {
		return 0
	}
	factor := yarnFactor(m.VLLMConfig.ExtraFlags)
	if factor <= 1 {
		return trained
	}
	// The factor scales the original length when the override names one;
	// that is what vLLM multiplies.
	base := trained
	if f := yarnOriginal(m.VLLMConfig.ExtraFlags); f > 0 {
		base = f
	}
	if n := int(float64(base) * factor); n > trained {
		return n
	}
	return trained
}

// ropeOverride is the rope block of an --hf-overrides value, under any of the
// names vLLM and transformers have used for it, at the top level or under
// text_config for a multimodal model.
func ropeOverride(extraFlags string) map[string]any {
	args := process.SplitFlags(extraFlags)
	for i := 0; i < len(args); i++ {
		name, val, hasVal := strings.Cut(args[i], "=")
		if name != "--hf-overrides" {
			continue
		}
		if !hasVal && i+1 < len(args) {
			val = args[i+1]
			i++
		}
		var o map[string]any
		if json.Unmarshal([]byte(val), &o) != nil {
			continue
		}
		scopes := []map[string]any{o}
		if tc, ok := o["text_config"].(map[string]any); ok {
			scopes = append(scopes, tc)
		}
		for _, scope := range scopes {
			for _, key := range []string{"rope_parameters", "rope_scaling"} {
				if r, ok := scope[key].(map[string]any); ok {
					return r
				}
			}
		}
	}
	return nil
}

// yarnFactor is the YaRN scaling factor in the --hf-overrides flag, or 0
// when there is none.
func yarnFactor(extraFlags string) float64 {
	r := ropeOverride(extraFlags)
	if r == nil {
		return 0
	}
	kind, _ := r["rope_type"].(string)
	if kind == "" {
		kind, _ = r["type"].(string)
	}
	if kind != "yarn" {
		return 0
	}
	f, _ := r["factor"].(float64)
	return f
}

// yarnOriginal is original_max_position_embeddings in the override, or 0.
func yarnOriginal(extraFlags string) int {
	r := ropeOverride(extraFlags)
	if r == nil {
		return 0
	}
	f, _ := r["original_max_position_embeddings"].(float64)
	return int(f)
}

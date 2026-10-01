package autoconfig

import (
	"slices"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// A variant is another of the card's commands that is the chosen command plus
// more: the same model, every flag and variable of the chosen one, and some
// of its own. Cards often present a base recipe and then one variant per
// feature. Qwen3.5-35B-A3B-FP8's has four -- the base, one adding tool calling,
// one adding its MTP head, one adding --language-model-only -- and taking the
// base alone lost the two features anyone serving it would want.
type variant struct {
	cmd  Command
	adds [][]string // the flag groups it adds, in order
	env  []string   // the variables it adds
}

// variantsOf returns the variants of chosen among the card's full commands.
func variantsOf(chosen *Command, full []Command) []variant {
	if chosen == nil {
		return nil
	}
	base := groupKeys(groupFlags(chosen.Args))
	var out []variant
	for _, c := range full {
		if !strings.EqualFold(c.Model, chosen.Model) || slices.Equal(c.Args, chosen.Args) && slices.Equal(c.Env, chosen.Env) {
			continue
		}
		groups := groupFlags(c.Args)
		keys := groupKeys(groups)
		if !subset(base, keys) || !subset(chosen.Env, c.Env) {
			continue
		}
		vr := variant{cmd: c}
		for i, g := range groups {
			if !slices.Contains(base, keys[i]) {
				vr.adds = append(vr.adds, g)
			}
		}
		for _, e := range c.Env {
			if !slices.Contains(chosen.Env, e) {
				vr.env = append(vr.env, e)
			}
		}
		if len(vr.adds) > 0 || len(vr.env) > 0 {
			out = append(out, vr)
		}
	}
	return out
}

func groupKeys(groups [][]string) []string {
	keys := make([]string, len(groups))
	for i, g := range groups {
		keys[i] = strings.Join(g, " ")
	}
	return keys
}

func subset(small, big []string) bool {
	for _, s := range small {
		if !slices.Contains(big, s) {
			return false
		}
	}
	return true
}

// takesAway are the fields a variant can add that remove something rather than
// add a feature: they start unticked. --language-model-only turns vision off;
// the others trade speed for a workaround.
var takesAway = map[string]string{
	"language_model_only":      "One of the card's variants adds it. It turns off the model's vision input, so it starts unticked.",
	"enforce_eager":            "One of the card's variants adds it. It turns off CUDA graphs, which slows generation, so it starts unticked.",
	"disable_async_scheduling": "One of the card's variants adds it. It turns off async scheduling, which slows generation, so it starts unticked.",
}

const (
	variantReason        = "From one of the card's variants of its command, which adds it to the one chosen."
	variantUnknownReason = "One of the card's variants adds it. This tool cannot tell what it does, so it starts unticked."
)

// fromVariants proposes what the variants add: features ticked, anything that
// takes something away unticked. A speculative config is returned rather than
// proposed, for speculative to pair or propose with the rest of that logic.
// The fields proposed are set in start, so the prose readings that follow do
// not propose them again.
func (v *validator) fromVariants(chosen *Command, start *process.VLLMStartConfig) (spec, specQuote string) {
	seen := map[string]bool{}
	for _, vr := range variantsOf(chosen, fullCommands(v.in.Commands)) {
		for _, g := range vr.adds {
			if seen[g[0]] {
				continue
			}
			seen[g[0]] = true
			quote := snippet(vr.cmd.Raw, g[0])
			if field, ok := hardwareFields[g[0]]; ok {
				v.note(field, originMachine, "One of the card's variants uses "+strings.Join(g, " ")+". This is chosen for this machine instead.")
				continue
			}
			st, rest := process.ParseArgs(g)
			if len(rest) > 0 {
				v.flagRow(rest, quote, true)
				if r := v.lastRow("flag:" + rest[0]); r != nil && r.Ticked {
					r.Ticked, r.Reason = false, variantUnknownReason
				}
				continue
			}
			for _, f := range knownFieldValues(st) {
				switch f.field {
				case "speculative_config":
					if spec == "" {
						spec, specQuote = f.value, quote
					}
					continue
				case "chat_template", "tokenizer":
					continue // a path on the author's machine, most likely
				}
				v.fieldRow(f.field, f.value, quote, variantReason, true)
				if why, ok := takesAway[f.field]; ok {
					if r := v.lastRow("field:" + f.field); r != nil {
						r.Ticked, r.Reason = false, why
					}
				}
			}
			merged, _ := process.ParseArgs(append(chosen.Args, g...))
			mergeFeatures(start, merged)
		}
		for _, e := range vr.env {
			name, _, _ := strings.Cut(e, "=")
			if !seen[name] {
				seen[name] = true
				v.envRow(e, snippet(vr.cmd.Raw, e))
			}
		}
	}
	return spec, specQuote
}

// mergeFeatures copies into start the parser settings a variant added, which
// is all the prose readings look at.
func mergeFeatures(start *process.VLLMStartConfig, from process.VLLMStartConfig) {
	if start.ReasoningParser == "" {
		start.ReasoningParser = from.ReasoningParser
	}
	if start.ToolCallParser == "" {
		start.ToolCallParser = from.ToolCallParser
	}
}

// lastRow is the row just added, if it has key; nil when that row was not
// added.
func (v *validator) lastRow(key string) *Row {
	if n := len(v.out.Rows); n > 0 && v.out.Rows[n-1].Key == key {
		return &v.out.Rows[n-1]
	}
	return nil
}

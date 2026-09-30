package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Reasoning toggle mechanisms: how a client turns a model's thinking on or
// off. These strings are part of the discovery contract shared with
// llama-toolchest, and clients match on them exactly.
const (
	// ReasoningToggleChatTemplateKwargs: the chat template branches on a
	// boolean the client passes through chat_template_kwargs, as Qwen's does
	// on enable_thinking.
	ReasoningToggleChatTemplateKwargs = "chat_template_kwargs"
	// ReasoningToggleReasoningEffort: the template takes a reasoning_effort
	// level and has no separate on/off switch.
	ReasoningToggleReasoningEffort = "reasoning_effort"
	// ReasoningToggleNone: nothing for a client to set. The model either has
	// no reasoning mode or cannot have it turned off.
	ReasoningToggleNone = "none"
)

// ReasoningCapability says whether and how a model exposes a thinking mode,
// so a client can drive it without guessing from the model's name.
type ReasoningCapability struct {
	Supported      bool   `json:"supported"`
	DefaultEnabled bool   `json:"default_enabled"`
	Toggle         string `json:"toggle"`
	// Kwarg is the chat_template_kwargs key when Toggle says that is the
	// mechanism, and "" otherwise.
	Kwarg string `json:"kwarg"`
	// EffortLevels is the reasoning_effort values the template accepts, or
	// nil when it takes none or does not say. It is never guessed: a level a
	// template does not know is a raised exception and a failed request, not
	// a degraded one.
	EffortLevels []string `json:"effort_levels"`
	// DefaultEffort is what the template uses when no level is sent.
	DefaultEffort *string `json:"default_effort"`
}

// DetectReasoning reads a model's chat template for its reasoning controls.
// templateOverride is the launch config's --chat-template path, which wins
// over the template in the model directory because it is the one served.
func DetectReasoning(modelDir, templateOverride string) ReasoningCapability {
	return detectReasoning(loadChatTemplate(modelDir, templateOverride))
}

// loadChatTemplate finds the template a model is served with: an explicit
// override, then the standalone file newer repos ship, then the copy inside
// tokenizer_config.json.
func loadChatTemplate(modelDir, templateOverride string) string {
	for _, path := range []string{templateOverride, filepath.Join(modelDir, "chat_template.jinja")} {
		if path == "" {
			continue
		}
		if data, err := os.ReadFile(path); err == nil && len(data) > 0 {
			return string(data)
		}
	}

	data, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json"))
	if err != nil {
		return ""
	}
	var tc struct {
		ChatTemplate any `json:"chat_template"`
	}
	if json.Unmarshal(data, &tc) != nil {
		return ""
	}
	switch v := tc.ChatTemplate.(type) {
	case string:
		return v
	case []any:
		// A list of named templates. They are concatenated: what is wanted
		// is whether any of them reads a reasoning variable.
		var all strings.Builder
		for _, item := range v {
			if m, ok := item.(map[string]any); ok {
				if s, ok := m["template"].(string); ok {
					all.WriteString(s)
					all.WriteByte('\n')
				}
			}
		}
		return all.String()
	}
	return ""
}

var (
	// reasoning_effort|default('xhigh')
	effortDefaultRE = regexp.MustCompile(`reasoning_effort\s*\|\s*default\(\s*['"]([\w-]+)['"]`)
	// resolved_reasoning_effort not in ('xhigh', 'medium', 'low')
	effortSetRE = regexp.MustCompile(`reasoning_effort\s+(?:not\s+)?in\s*[\(\[]([^\)\]]*)[\)\]]`)
	// reasoning_effort == 'low'
	effortEqualsRE = regexp.MustCompile(`reasoning_effort\s*==\s*['"]([\w-]+)['"]`)
	quotedRE       = regexp.MustCompile(`['"]([\w-]+)['"]`)
)

// effortOrder is the known levels, cheapest first, so a client's selector
// reads in the order a person expects whatever order the template lists them.
var effortOrder = map[string]int{"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}

func detectReasoning(template string) ReasoningCapability {
	r := ReasoningCapability{Toggle: ReasoningToggleNone}
	hasThinking := strings.Contains(template, "enable_thinking")
	hasEffort := strings.Contains(template, "reasoning_effort")

	switch {
	case hasThinking:
		// Qwen's shape: thinking is on unless enable_thinking is passed as
		// false. Where the template also reads reasoning_effort it does so
		// inside that branch, so effort is an addition to the switch, not a
		// replacement for it.
		r.Supported = true
		r.DefaultEnabled = true
		r.Toggle = ReasoningToggleChatTemplateKwargs
		r.Kwarg = "enable_thinking"
	case hasEffort:
		r.Supported = true
		r.DefaultEnabled = true
		r.Toggle = ReasoningToggleReasoningEffort
	default:
		return r
	}

	if hasEffort {
		r.EffortLevels = effortLevels(template)
		if m := effortDefaultRE.FindStringSubmatch(template); m != nil {
			r.DefaultEffort = &m[1]
		}
	}
	return r
}

// effortLevels is the levels a template names, or nil when it names none.
//
// A template that validates its input lists them in one place, which is the
// reliable source. One that does not is read for the values it compares
// against -- which can miss a level handled by a final else, so that list is
// only used when it has at least two entries to show for itself.
func effortLevels(template string) []string {
	seen := map[string]bool{}
	var levels []string
	add := func(level string) {
		if !seen[level] {
			seen[level] = true
			levels = append(levels, level)
		}
	}

	if m := effortSetRE.FindStringSubmatch(template); m != nil {
		for _, q := range quotedRE.FindAllStringSubmatch(m[1], -1) {
			add(q[1])
		}
	}
	if len(levels) == 0 {
		for _, m := range effortEqualsRE.FindAllStringSubmatch(template, -1) {
			add(m[1])
		}
		if len(levels) < 2 {
			return nil
		}
	}

	sort.SliceStable(levels, func(i, j int) bool {
		oi, iKnown := effortOrder[levels[i]]
		oj, jKnown := effortOrder[levels[j]]
		if iKnown && jKnown {
			return oi < oj
		}
		return iKnown && !jKnown
	})
	return levels
}

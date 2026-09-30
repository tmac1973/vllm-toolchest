package models

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// The reasoning block of the Qwen 3.8 chat template, as shipped in
// tcclaviger/ThinkingCap-3.8-27B-PARO5.
const qwen38Template = `{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low.' %}
    {%- endif %}
{%- endif %}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}`

func TestQwen38ReasoningIsReadFromItsTemplate(t *testing.T) {
	r := detectReasoning(qwen38Template)

	if !r.Supported || !r.DefaultEnabled || r.Toggle != ReasoningToggleChatTemplateKwargs || r.Kwarg != "enable_thinking" {
		t.Errorf("reasoning = %+v, want it on by default and switched by enable_thinking", r)
	}
	// From the template's own validation list, which names "medium" though
	// no branch compares against it, and in the order a selector wants.
	if want := []string{"low", "medium", "xhigh"}; !reflect.DeepEqual(r.EffortLevels, want) {
		t.Errorf("effort levels = %v, want %v", r.EffortLevels, want)
	}
	if r.DefaultEffort == nil || *r.DefaultEffort != "xhigh" {
		t.Errorf("default effort = %v, want xhigh", r.DefaultEffort)
	}
}

func TestReasoningShapes(t *testing.T) {
	for name, tc := range map[string]struct {
		template string
		toggle   string
		kwarg    string
		levels   []string
	}{
		// Qwen 3.5 and 3.6: a switch and no effort variable at all.
		"a switch only": {
			`{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}`,
			ReasoningToggleChatTemplateKwargs, "enable_thinking", nil,
		},
		"effort with no switch": {
			`{%- if reasoning_effort == 'low' %}a{%- elif reasoning_effort == 'high' %}b{%- endif %}`,
			ReasoningToggleReasoningEffort, "", []string{"low", "high"},
		},
		// Named, and nothing to say what it accepts. A client sending a
		// guessed level gets a template exception, so none are offered.
		"effort named but not enumerated": {
			`Reasoning: {{ reasoning_effort }}`,
			ReasoningToggleReasoningEffort, "", nil,
		},
		"no reasoning": {`{{ messages[0].content }}`, ReasoningToggleNone, "", nil},
		"no template":  {``, ReasoningToggleNone, "", nil},
	} {
		t.Run(name, func(t *testing.T) {
			r := detectReasoning(tc.template)
			if r.Toggle != tc.toggle || r.Kwarg != tc.kwarg || !reflect.DeepEqual(r.EffortLevels, tc.levels) {
				t.Errorf("reasoning = %+v, want toggle %q kwarg %q levels %v", r, tc.toggle, tc.kwarg, tc.levels)
			}
			if r.Supported != (tc.toggle != ReasoningToggleNone) {
				t.Errorf("supported = %v with toggle %q", r.Supported, r.Toggle)
			}
		})
	}
}

// The template that is served is the one that counts: the launch config's
// override, then the standalone file, then the copy in tokenizer_config.json.
func TestTheServedTemplateIsTheOneRead(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}

	write("tokenizer_config.json", `{"chat_template": "{% if enable_thinking %}x{% endif %}"}`)
	if r := DetectReasoning(dir, ""); r.Toggle != ReasoningToggleChatTemplateKwargs {
		t.Errorf("tokenizer_config.json template was not read: %+v", r)
	}

	write("chat_template.jinja", `{{ reasoning_effort }}`)
	if r := DetectReasoning(dir, ""); r.Toggle != ReasoningToggleReasoningEffort {
		t.Errorf("chat_template.jinja did not take precedence: %+v", r)
	}

	plain := write("plain.jinja", `{{ messages }}`)
	if r := DetectReasoning(dir, plain); r.Supported {
		t.Errorf("the --chat-template override was not the one read: %+v", r)
	}
}

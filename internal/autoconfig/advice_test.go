package autoconfig

import (
	"context"
	"encoding/json"
	"sort"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

func TestAdviceSchemaMatchesTheStruct(t *testing.T) {
	s := adviceSchema()
	props := s["properties"].(map[string]any)
	req := s["required"].([]string)
	if !sort.StringsAreSorted(req) || len(req) != len(props) {
		t.Error("every property must be required, in a stable order")
	}
	// Every JSON field of Advice is in the schema, and nothing else.
	var zero Advice
	data, _ := json.Marshal(zero)
	var fields map[string]any
	json.Unmarshal(data, &fields)
	for k := range fields {
		if _, ok := props[k]; !ok {
			t.Errorf("%s is in Advice but not in the schema", k)
		}
	}
	if len(fields) != len(props) {
		t.Errorf("Advice has %d fields and the schema %d", len(fields), len(props))
	}
	if s["additionalProperties"] != false {
		t.Error("the schema admits extra properties")
	}
}

func TestAskSendsTheCommandsAndTheCard(t *testing.T) {
	raw := readCard(t, "flash-next-mxfp4.md")
	card := Card{Raw: raw, Text: TrimCard(raw)}
	var user string
	call := func(_ context.Context, name string, _ map[string]any, system, u string, out any) error {
		user = u
		return json.Unmarshal([]byte(`{"command_index": 1, "temperature": 0.8}`), out)
	}
	adv, err := Ask(context.Background(), call, &models.Model{ID: "org/m"}, card, ExtractCommands(raw), 0)
	if err != nil || adv.CommandIndex == nil || *adv.CommandIndex != 1 || *adv.Temperature != 0.8 {
		t.Fatalf("adv=%+v err=%v", adv, err)
	}
	if !strings.Contains(user, "1. docker run") || !strings.Contains(user, "2. docker run") {
		t.Error("the commands were not numbered for the helper")
	}
	if strings.Contains(user, "3. ") && strings.Contains(user, "3. VLLM_ALLOW") {
		t.Error("a partial command was numbered")
	}
	if !strings.Contains(user, "<<<") || !strings.Contains(user, "Model repository: org/m") {
		t.Error("the card or the repository is missing")
	}

	called := false
	empty := func(context.Context, string, map[string]any, string, string, any) error { called = true; return nil }
	if _, err := Ask(context.Background(), empty, &models.Model{ID: "org/m"}, Card{}, nil, 0); err != nil || called {
		t.Error("an empty card was sent to the helper")
	}
}

// The budget covers the whole prompt, not the card alone. On compute a card
// of the full budget plus eight commands came to 14,337 tokens against a
// 16,384-token helper with 2,048 kept for the answer, and was refused.
func TestAskKeepsTheWholePromptInBudget(t *testing.T) {
	var cmds []Command
	for i := range 8 {
		cmds = append(cmds, ExtractCommands("```\nvllm serve org/m --tensor-parallel-size "+strings.Repeat("8", i+1)+" --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}'\n```\n")...)
	}
	card := Card{Text: strings.Repeat("A paragraph of the model card.\n\n", 3000)}
	budget := CardCharsForContext(models.HelperContext)
	var system, user string
	call := func(_ context.Context, _ string, _ map[string]any, s, u string, _ any) error {
		system, user = s, u
		return nil
	}
	if _, err := Ask(context.Background(), call, &models.Model{ID: "org/m"}, card, cmds, budget); err != nil {
		t.Fatal(err)
	}
	if n := len(system) + len(user); n > budget {
		t.Errorf("the prompt is %d characters, over the budget of %d", n, budget)
	}
	if !strings.Contains(user, "8. vllm serve") || !strings.Contains(user, "left out for length") {
		t.Error("the commands were not all kept, or the card was not cut")
	}
	// At the worst rate the budget assumes, the prompt and the answer fit.
	if tokens := float64(budget)/2.5 + 2048; tokens > float64(models.HelperContext) {
		t.Errorf("the budget comes to %.0f tokens with the answer", tokens)
	}
}

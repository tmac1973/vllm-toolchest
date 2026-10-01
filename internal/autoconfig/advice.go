package autoconfig

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Advice is what the helper model reads out of a model card. Every value is
// nullable: null means the card does not say. Each group carries the card's
// own words, and a value whose words are not actually in the card is not
// used -- the helper's answer is checked, never trusted.
//
// The form is flat and short on purpose. A small model fills a flat form
// reliably; nested optional objects it fills less reliably.
type Advice struct {
	// CommandIndex is 1-based, into the numbered commands the helper was
	// shown; null when none of them is a complete command for this model.
	CommandIndex *int `json:"command_index"`

	Temperature       *float64 `json:"temperature"`
	TopP              *float64 `json:"top_p"`
	TopK              *int     `json:"top_k"`
	MinP              *float64 `json:"min_p"`
	PresencePenalty   *float64 `json:"presence_penalty"`
	RepetitionPenalty *float64 `json:"repetition_penalty"`
	SamplingQuote     string   `json:"sampling_quote"`

	ReasoningParser *string `json:"reasoning_parser"`
	ToolCallParser  *string `json:"tool_call_parser"`
	ParserQuote     string  `json:"parser_quote"`

	// DraftMethod is "dflash", "mtp", "eagle3", "draft_model" or "none".
	DraftMethod      *string `json:"draft_method"`
	DraftRepo        *string `json:"draft_repo"`
	SpeculativeQuote string  `json:"speculative_quote"`

	RecommendedContext *int   `json:"recommended_context"`
	ContextQuote       string `json:"context_quote"`

	// OtherNotes are anything else the card says about running the model.
	// They cannot be checked against the card and never become settings.
	OtherNotes []string `json:"other_notes"`
}

// draftMethods are the values DraftMethod may take.
var draftMethods = []string{"dflash", "mtp", "eagle3", "draft_model", "none"}

// adviceSchema is the JSON Schema the helper's answer is constrained to.
func adviceSchema() map[string]any {
	num := map[string]any{"type": []string{"number", "null"}}
	integer := map[string]any{"type": []string{"integer", "null"}}
	nullableStr := map[string]any{"type": []string{"string", "null"}}
	str := map[string]any{"type": "string"}
	methods := make([]any, 0, len(draftMethods)+1)
	for _, m := range draftMethods {
		methods = append(methods, m)
	}
	methods = append(methods, nil)

	props := map[string]any{
		"command_index":       integer,
		"temperature":         num,
		"top_p":               num,
		"top_k":               integer,
		"min_p":               num,
		"presence_penalty":    num,
		"repetition_penalty":  num,
		"sampling_quote":      str,
		"reasoning_parser":    nullableStr,
		"tool_call_parser":    nullableStr,
		"parser_quote":        str,
		"draft_method":        map[string]any{"enum": methods},
		"draft_repo":          nullableStr,
		"speculative_quote":   str,
		"recommended_context": integer,
		"context_quote":       str,
		"other_notes":         map[string]any{"type": "array", "items": str, "maxItems": 5},
	}
	required := make([]string, 0, len(props))
	for k := range props {
		required = append(required, k)
	}
	sort.Strings(required) // the same request every time
	return map[string]any{
		"type":                 "object",
		"properties":           props,
		"required":             required,
		"additionalProperties": false,
	}
}

const adviceInstructions = `You read the documentation of a language model (its model card) and fill in a form about how to serve it with vLLM.

Rules:
- Use only what the model card says. Do not use what you know about other models, and do not guess.
- When the card does not state a value, answer null for it, and an empty string for its quote.
- For every value you fill in, copy into the matching quote field the sentence from the card that states it, exactly as written.
- command_index: the number of the command, from the numbered list, that the card presents as the normal way to serve THIS repository on all of its GPUs. Prefer a command for this repository over one for the model it was made from. Answer null when none of them is a complete command.
- Sampling values: when the card gives more than one set (for example one for thinking and one for non-thinking use), use the general or default set.
- reasoning_parser and tool_call_parser: only a parser name the card gives for vLLM.
- draft_method: "mtp" when the card recommends the model's own multi-token-prediction head for speculative decoding; "dflash" or "eagle3" when it names that kind of draft; "draft_model" when it recommends a separate smaller model; otherwise null.
- draft_repo: the Hugging Face repository (owner/name) the card names for that draft, or null.
- recommended_context: a context length in tokens the card recommends serving with, or null.
- other_notes: at most five short sentences about anything else the card says about serving this model with vLLM. Leave out other servers, such as SGLang or llama.cpp, and anything about training or evaluation. Leave it empty when there is nothing.`

// CallFunc asks the helper for an answer constrained to schema, decoded into
// out. The caller supplies it, so this package never starts an engine.
type CallFunc func(ctx context.Context, schemaName string, schema map[string]any, system, user string, out any) error

// Ask has the helper read the card and fill in the form. An empty card is
// not sent: there is nothing to read.
//
// maxPrompt bounds the whole prompt, instructions included, in characters;
// the card is cut to whatever the commands leave. Zero means no bound.
func Ask(ctx context.Context, call CallFunc, m *models.Model, card Card, cmds []Command, maxPrompt int) (Advice, error) {
	var adv Advice
	if strings.TrimSpace(card.Text) == "" || call == nil {
		return adv, nil
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Model repository: %s\n\n", m.ID)
	full := fullCommands(cmds)
	if len(full) == 0 {
		b.WriteString("Commands found in the card: none.\n\n")
	} else {
		b.WriteString("Commands found in the card:\n")
		for i, c := range full {
			fmt.Fprintf(&b, "%d. %s\n", i+1, c.Raw)
		}
		b.WriteString("\n")
	}
	text := card.Text
	if maxPrompt > 0 {
		// The commands come first and are kept whole: they are what the
		// helper is choosing between. The card gets what is left.
		// 128 covers the card's wrapper and capText's note that it was cut.
		room := maxPrompt - len(adviceInstructions) - b.Len() - 128
		text = capText(text, max(2000, room))
	}
	fmt.Fprintf(&b, "Model card:\n<<<\n%s\n>>>", text)
	err := call(ctx, "model_card_advice", adviceSchema(), adviceInstructions, b.String(), &adv)
	return adv, err
}

// fullCommands are the commands that are not fragments, in order: what the
// helper numbers and CommandIndex points into.
func fullCommands(cmds []Command) []Command {
	var out []Command
	for _, c := range cmds {
		if !c.Partial {
			out = append(out, c)
		}
	}
	return out
}

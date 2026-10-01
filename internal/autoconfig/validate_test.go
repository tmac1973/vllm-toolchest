package autoconfig

import (
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// thinkingCap is the 27B as registered on compute, with the default config a
// fresh download gets.
func thinkingCap() *models.Model {
	return &models.Model{
		ID: "tcclaviger/ThinkingCap-3.8-27B-PARO5",
		HFConfig: models.HFConfig{
			NumHiddenLayers: 64, HiddenSize: 5120, VocabSize: 248320,
			NumAttentionHeads: 24, NumKeyValueHeads: 4, HeadDim: 256, AttentionLayers: 16,
			MaxPositionEmbeddings: 262144,
		},
		VLLMConfig: models.VLLMConfig{
			MaxModelLen: 8192, TensorParallelSize: 1, GPUMemoryUtilization: 0.90, MaxNumSeqs: 16,
			KVCacheDtype: "auto", EnableAutoToolChoice: true, ToolCallParser: "qwen3_xml",
		},
	}
}

// dflashDraft is the DFlash drafter installed beside it.
func dflashDraft() *models.Model {
	return &models.Model{
		ID: "tcclaviger/Qwen3.8-27B-DFlash2-FP8", DisplayName: "Qwen3.8-27B-DFlash2-FP8",
		LocalPath: "/data/models/tcclaviger/Qwen3.8-27B-DFlash2-FP8",
		HFConfig: models.HFConfig{
			NumHiddenLayers: 5, HiddenSize: 5120, VocabSize: 248320,
			Draft: &models.DraftMeta{Method: "dflash", BlockSize: 8, TargetLayers: 64},
		},
	}
}

// machineEnv is what every model on compute already gets.
var machineEnv = []string{"VLLM_WORKER_MULTIPROC_METHOD=spawn"}

func cardInputs(t *testing.T, file string, m *models.Model) Inputs {
	raw := readCard(t, file)
	return Inputs{
		Model: m, Base: m.VLLMConfig,
		Card:     Card{Raw: raw, Text: TrimCard(raw)},
		Commands: ExtractCommands(raw), Inline: InlineFlags(raw),
		MachineEnv: machineEnv,
	}
}

func rowByKey(rows []Row, key string) *Row {
	for i := range rows {
		if rows[i].Key == key {
			return &rows[i]
		}
	}
	return nil
}

func noteFor(notes []models.ProfileNote, field, contains string) bool {
	for _, n := range notes {
		if n.Field == field && strings.Contains(n.Reason, contains) {
			return true
		}
	}
	return false
}

// The model that drove the design, with no helper: the card's command alone
// reproduces what was configured on compute by hand.
func TestTheTwentySevenBCardReproducesTheHandBuiltConfig(t *testing.T) {
	in := cardInputs(t, "thinkingcap-27b-paro5.md", thinkingCap())
	in.Drafts = []*models.Model{dflashDraft()}
	c := Validate(in)

	for key, want := range map[string]string{
		"field:enable_auto_tool_choice": "true",
		"field:tool_call_parser":        "qwen3_coder",
		"field:reasoning_parser":        "qwen3",
		"field:enable_chunked_prefill":  "true",
		"field:max_num_batched_tokens":  "8192",
	} {
		r := rowByKey(c.Rows, key)
		if r == nil || r.Value != want || !r.Ticked {
			t.Errorf("%s: %+v, want %q ticked", key, r, want)
		}
	}
	if r := rowByKey(c.Rows, "field:compilation_config"); r == nil || !r.Ticked {
		t.Errorf("compilation config: %+v", r)
	}
	if r := rowByKey(c.Rows, "flag:--override-generation-config"); r == nil || !r.Ticked {
		t.Errorf("sampling override: %+v", r)
	}
	spec := rowByKey(c.Rows, "field:speculative_config")
	if spec == nil || !spec.Ticked || !strings.Contains(spec.Value, `"num_speculative_tokens": 7`) ||
		!strings.Contains(spec.Value, dflashDraft().LocalPath) {
		t.Fatalf("speculative row: %+v", spec)
	}

	for _, name := range []string{"OMP_NUM_THREADS", "VLLM_ROCM_USE_AITER", "GPU_MAX_HW_QUEUES", "HSA_ENABLE_INTERRUPT", "HSA_ENABLE_MWAITX"} {
		if r := rowByKey(c.Rows, "env:"+name); r == nil || !r.Ticked {
			t.Errorf("env %s: %+v", name, r)
		}
	}
	for _, name := range []string{"TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "ROCR_VISIBLE_DEVICES"} {
		if r := rowByKey(c.Rows, "env:"+name); r == nil || r.Ticked {
			t.Errorf("env %s should be proposed unticked: %+v", name, r)
		}
	}

	for field, used := range map[string]string{
		"tensor_parallel_size": "4", "max_model_len": "262144", "gpu_memory_utilization": "0.92", "max_num_seqs": "8",
	} {
		if !noteFor(c.Notes, field, "used "+used) {
			t.Errorf("no note that the card used %s=%s", field, used)
		}
	}
	if c.CardKVDtype != "fp8" {
		t.Errorf("CardKVDtype = %q", c.CardKVDtype)
	}
	for _, r := range c.Rows {
		if r.Quote == "" || r.Origin != "model card" {
			t.Errorf("row %s without the card's words or its origin", r.Key)
		}
	}

	// The configuration that serves on compute, less the hardware fields the
	// planner decides.
	got, err := Apply(in.Base, c.Rows, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.ToolCallParser != "qwen3_coder" || got.ReasoningParser != "qwen3" || !got.EnableAutoToolChoice ||
		!got.EnableChunkedPrefill || got.MaxNumBatchedTokens != 8192 {
		t.Errorf("parsers and batching: %+v", got)
	}
	if got.SpeculativeConfig != `{"method": "dflash", "model": "/data/models/tcclaviger/Qwen3.8-27B-DFlash2-FP8", "num_speculative_tokens": 7}` {
		t.Errorf("speculative config = %s", got.SpeculativeConfig)
	}
	wantEnv := "OMP_NUM_THREADS=8\nVLLM_ROCM_USE_AITER=0\nGPU_MAX_HW_QUEUES=2\nHSA_ENABLE_INTERRUPT=1\nHSA_ENABLE_MWAITX=1"
	if got.Env != wantEnv {
		t.Errorf("env block:\n%s\nwant\n%s", got.Env, wantEnv)
	}
	args := process.SplitFlags(got.ExtraFlags)
	if len(args) != 2 || args[0] != "--override-generation-config" ||
		args[1] != `{"max_tokens": 65536, "temperature": 0.6, "top_p": 0.95, "top_k": 20}` {
		t.Errorf("extra flags = %q", args)
	}
}

func TestAMissingDraftIsWanted(t *testing.T) {
	c := Validate(cardInputs(t, "thinkingcap-27b-paro5.md", thinkingCap()))
	if rowByKey(c.Rows, "field:speculative_config") != nil {
		t.Error("a speculative config was proposed with no draft installed")
	}
	if c.WantDraft != "dflash" || c.DraftRepo != "" {
		t.Errorf("want=%q repo=%q; /app/draft is not a repository", c.WantDraft, c.DraftRepo)
	}
	if !noteFor(c.Notes, "speculative_config", "not installed") {
		t.Error("no note that the draft is missing")
	}

	// A draft of another shape does not pair.
	in := cardInputs(t, "thinkingcap-27b-paro5.md", thinkingCap())
	wrong := dflashDraft()
	wrong.HFConfig.Draft.TargetLayers = 48
	in.Drafts = []*models.Model{wrong}
	if c := Validate(in); rowByKey(c.Rows, "field:speculative_config") != nil {
		t.Error("a draft made for a 48-layer model was paired with a 64-layer one")
	}
}

func TestTheMoECard(t *testing.T) {
	m := &models.Model{ID: "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ", HFConfig: models.HFConfig{MaxPositionEmbeddings: 262144}}
	c := Validate(cardInputs(t, "flash-next-mxfp4.md", m))
	if c.Command == nil || c.Command.Settings().Start.TensorParallelSize != 4 {
		t.Fatalf("chosen command: %+v", c.Command)
	}
	if !noteFor(c.Notes, "", "2 complete commands") {
		t.Error("no note that there were two commands")
	}
	if !noteFor(c.Notes, "", "special case") {
		t.Error("the partial command was not noted")
	}
	// Its own MTP head, written in the command: proposed as written, ticked.
	if r := rowByKey(c.Rows, "field:speculative_config"); r == nil || !r.Ticked || !strings.Contains(r.Value, `"mtp"`) {
		t.Errorf("mtp row: %+v", r)
	}
	// Rope scaling for a context past the model's own is not taken.
	if r := rowByKey(c.Rows, "flag:--hf-overrides"); r == nil || r.Ticked {
		t.Errorf("hf-overrides: %+v", r)
	}
}

func TestAFlagTheImageLacksStartsUnticked(t *testing.T) {
	m := &models.Model{ID: "Qwen/Qwen3-14B-AWQ"}
	in := cardInputs(t, "qwen3-14b-awq.md", m)
	in.Flags = fakeFlags{"--reasoning-parser": true}
	c := Validate(in)
	if r := rowByKey(c.Rows, "flag:--enable-reasoning"); r == nil || r.Ticked || !strings.Contains(r.Reason, "does not list") {
		t.Errorf("--enable-reasoning: %+v", r)
	}
	if r := rowByKey(c.Rows, "field:reasoning_parser"); r == nil || !r.Ticked || r.Value != "deepseek_r1" {
		t.Errorf("reasoning parser: %+v", r)
	}
}

type fakeFlags map[string]bool

func (fakeFlags) Known() bool            { return true }
func (f fakeFlags) Has(flag string) bool { return f[flag] }

func f64(v float64) *float64 { return &v }
func str(s string) *string   { return &s }
func intp(n int) *int        { return &n }

func TestTheHelpersReadingIsChecked(t *testing.T) {
	card := "# Model\n\n## Best practice\n\nWe recommend a temperature of 0.7 and top_p of 0.8 for general use.\n\nUse the qwen3 reasoning parser with vLLM for thinking output.\n"
	m := &models.Model{ID: "org/m"}
	base := Inputs{Model: m, Card: Card{Raw: card, Text: card}, HasAdvice: true}

	in := base
	in.Advice = Advice{Temperature: f64(0.7), TopP: f64(0.8), SamplingQuote: "We recommend a temperature of 0.7 and top_p of 0.8"}
	c := Validate(in)
	if r := rowByKey(c.Rows, "flag:--override-generation-config"); r == nil || !strings.Contains(r.Flag[1], `"temperature":0.7`) {
		t.Errorf("verified prose sampling: %+v", r)
	}

	// The same values, a sentence not in the card: dropped, with a note.
	in.Advice.SamplingQuote = "The authors suggest a temperature of 0.7."
	c = Validate(in)
	if rowByKey(c.Rows, "flag:--override-generation-config") != nil || !noteFor(c.Notes, "", "not in the card") {
		t.Errorf("an unverified quote was used: rows=%+v", c.Rows)
	}

	// Out of range.
	in.Advice = Advice{Temperature: f64(7), SamplingQuote: "We recommend a temperature of 0.7"}
	c = Validate(in)
	if rowByKey(c.Rows, "flag:--override-generation-config") != nil || !noteFor(c.Notes, "extra_flags", "outside") {
		t.Error("a temperature of 7 was used")
	}

	// A generation_config.json says otherwise: the file wins, and the
	// difference is noted.
	withDefaults := &models.Model{ID: "org/m", GenDefaults: models.GenDefaults{Temperature: f64(0.6)}}
	in = base
	in.Model = withDefaults
	in.Advice = Advice{Temperature: f64(0.7), SamplingQuote: "We recommend a temperature of 0.7"}
	c = Validate(in)
	if rowByKey(c.Rows, "flag:--override-generation-config") != nil || !noteFor(c.Notes, "extra_flags", "generation_config.json") {
		t.Error("prose sampling overrode the model's generation_config.json")
	}

	// What the file does not set, the card fills: Qwen3.5's file sets the
	// temperature, and its card adds a presence penalty.
	presence := "# Model\n\nWe recommend temperature=0.6, presence_penalty=1.5 for general use.\n"
	in = Inputs{Model: withDefaults, Card: Card{Raw: presence, Text: presence}, HasAdvice: true,
		Advice: Advice{Temperature: f64(0.6), PresencePenalty: f64(1.5), SamplingQuote: "temperature=0.6, presence_penalty=1.5"}}
	c = Validate(in)
	if r := rowByKey(c.Rows, "flag:--override-generation-config"); r == nil || !r.Ticked || r.Flag[1] != `{"presence_penalty":1.5}` {
		t.Errorf("the presence penalty the file leaves out: %+v", r)
	}

	// Parsers from prose, verified.
	in = base
	in.Advice = Advice{ReasoningParser: str("qwen3"), ParserQuote: "Use the qwen3 reasoning parser with vLLM"}
	if r := rowByKey(Validate(in).Rows, "field:reasoning_parser"); r == nil || r.Value != "qwen3" {
		t.Errorf("prose parser: %+v", r)
	}

	// Summaries are marked as unchecked.
	in = base
	in.Advice = Advice{OtherNotes: []string{"Serve with a recent vLLM."}}
	notes := Validate(in).Notes
	if len(notes) != 1 || notes[0].Origin != "helper summary" {
		t.Errorf("notes = %+v", notes)
	}
}

func TestTheHelpersCommandChoice(t *testing.T) {
	m := &models.Model{ID: "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"}
	in := cardInputs(t, "flash-next-mxfp4.md", m)
	in.HasAdvice = true
	in.Advice = Advice{CommandIndex: intp(2)}
	if c := Validate(in); c.Command.Settings().Start.TensorParallelSize != 2 {
		t.Error("the helper's choice of the second command was ignored")
	}
	in.Advice = Advice{CommandIndex: intp(9)}
	c := Validate(in)
	if c.Command.Settings().Start.TensorParallelSize != 4 || !noteFor(c.Notes, "", "does not exist") {
		t.Error("an out-of-range choice did not fall back to the first command")
	}
}

func TestTrustRemoteCodeIsFlagged(t *testing.T) {
	card := "```\nvllm serve org/m --trust-remote-code --max-model-len 4096\n```\nAlso try `--enforce-eager`."
	m := &models.Model{ID: "org/m"}
	in := Inputs{Model: m, Card: Card{Raw: card}, Commands: ExtractCommands(card), Inline: InlineFlags(card)}
	c := Validate(in)
	r := rowByKey(c.Rows, "field:trust_remote_code")
	if r == nil || !r.Ticked || r.Warning == "" {
		t.Errorf("trust_remote_code: %+v", r)
	}
	if r := rowByKey(c.Rows, "field:enforce_eager"); r == nil || r.Ticked {
		t.Errorf("an inline mention should start unticked: %+v", r)
	}
}

func TestApply(t *testing.T) {
	base := models.VLLMConfig{
		ExtraFlags: `--keep '{"a": 1}' --override-generation-config '{"temperature": 1}'`,
		Env:        "OMP_NUM_THREADS=4\nKEEP=1",
	}
	rows := []Row{
		{Key: "flag:--override-generation-config", Kind: RowFlag, Flag: []string{"--override-generation-config", `{"temperature": 0.6}`}, Ticked: true},
		{Key: "env:OMP_NUM_THREADS", Kind: RowEnv, Env: "OMP_NUM_THREADS=8", Ticked: true},
		{Key: "field:reasoning_parser", Kind: RowField, Field: "reasoning_parser", Value: "qwen3", Ticked: false},
	}
	got, err := Apply(base, rows, nil)
	if err != nil {
		t.Fatal(err)
	}
	args := process.SplitFlags(got.ExtraFlags)
	if !slices.Equal(args, []string{"--keep", `{"a": 1}`, "--override-generation-config", `{"temperature": 0.6}`}) {
		t.Errorf("extra flags = %q", args)
	}
	if got.Env != "OMP_NUM_THREADS=8\nKEEP=1" {
		t.Errorf("env = %q", got.Env)
	}
	if got.ReasoningParser != "" {
		t.Error("an unticked row was applied")
	}
	got, _ = Apply(base, rows, map[string]bool{"field:reasoning_parser": true})
	if got.ReasoningParser != "qwen3" || got.Env != base.Env {
		t.Error("explicit ticks were not followed")
	}
	if _, err := Apply(base, []Row{{Key: "k", Kind: RowField, Field: "api_key", Value: "x", Ticked: true}}, nil); err == nil {
		t.Error("a field autoconfigure may not set was applied")
	}
	if _, err := Apply(base, []Row{{Key: "k", Kind: RowField, Field: "enforce_eager", Value: "maybe", Ticked: true}}, nil); err == nil {
		t.Error("an unparseable value was applied")
	}
}

// Found on compute: the helper cited a real sentence from the card for a
// recommended context, but the sentence said nothing about context. A quote
// has to state the value it is cited for.
func TestAQuoteMustStateItsValue(t *testing.T) {
	card := "# Model\n\nNOTICE: This checkpoint runs only on the tcclaviger/vllm image.\n\nWe serve it with a 256K context and a temperature of 0.6, top_p 0.95.\n\nUse the qwen3 reasoning parser.\n"
	base := Inputs{Model: &models.Model{ID: "org/m"}, Card: Card{Raw: card, Text: card}, HasAdvice: true}

	in := base
	in.Advice = Advice{RecommendedContext: intp(262144), ContextQuote: "NOTICE: This checkpoint runs only on the tcclaviger/vllm image."}
	if noteFor(Validate(in).Notes, "max_model_len", "recommends a context") {
		t.Error("a context was attributed to a sentence that does not state it")
	}
	in.Advice.ContextQuote = "We serve it with a 256K context"
	if !noteFor(Validate(in).Notes, "max_model_len", "recommends a context of 262144") {
		t.Error("256K was not recognised as 262,144 tokens")
	}

	in = base
	in.Advice = Advice{Temperature: f64(0.6), TopK: intp(20), SamplingQuote: "a temperature of 0.6, top_p 0.95"}
	c := Validate(in)
	r := rowByKey(c.Rows, "flag:--override-generation-config")
	if r == nil || !strings.Contains(r.Flag[1], `"temperature":0.6`) || strings.Contains(r.Flag[1], "top_k") {
		t.Errorf("sampling row: %+v", r)
	}
	if !noteFor(c.Notes, "extra_flags", "top_k of 20") {
		t.Error("a value the quote does not give was not reported")
	}

	in = base
	in.Advice = Advice{ReasoningParser: str("deepseek_r1"), ParserQuote: "Use the qwen3 reasoning parser."}
	if rowByKey(Validate(in).Rows, "field:reasoning_parser") != nil {
		t.Error("a parser the quote does not name was proposed")
	}
}

func TestAQuoteWithoutTheCardsMarkdownStillMatches(t *testing.T) {
	card := "> - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, **recommended**\n"
	in := Inputs{Model: &models.Model{ID: "org/m"}, Card: Card{Raw: card, Text: card}, HasAdvice: true,
		Advice: Advice{Temperature: f64(0.7), SamplingQuote: "Instruct (or non-thinking) mode: temperature=0.7, top_p=0.80, top_k=20, recommended"}}
	if rowByKey(Validate(in).Rows, "flag:--override-generation-config") == nil {
		t.Error("a quote that differs from the card only in markdown was rejected")
	}
}

// Found on compute: the MoE's base card lists its sampling sets under a
// numbered item, with a blank line and indentation between them. A quote of
// those lines without the blank line, or with a line of its own invention,
// has to be told apart.
func TestAMultiLineQuote(t *testing.T) {
	card := normalise("1. **Sampling Parameters**: We suggest using the following sets of sampling parameters:\n    \n    - Thinking Mode: `temperature=1.0`, `top_p=0.95`\n    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`\n\n2. **Adequate Output Length**: use 32,768 tokens.")
	if !quoteInCard(card, "We suggest using the following sets of sampling parameters:\n- Thinking Mode: temperature=1.0, top_p=0.95\n\n2. Adequate Output Length: use 32,768 tokens.") {
		t.Error("a quote of real lines with the list's spacing changed was rejected")
	}
	if quoteInCard(card, "We suggest using the following sets of sampling parameters:\n- Coding Mode: temperature=0.2, top_p=0.9") {
		t.Error("a quote with an invented line was accepted")
	}
}

// A reading that could not have been used is not worth a note: the command
// already sets the sampling defaults and both parsers.
func TestNoNoteForAReadingThatCouldNotMatter(t *testing.T) {
	card := "```\nvllm serve org/m --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --override-generation-config '{\"temperature\": 0.8}'\n```\n"
	in := Inputs{Model: &models.Model{ID: "org/m"}, Card: Card{Raw: card}, Commands: ExtractCommands(card), HasAdvice: true,
		Advice: Advice{Temperature: f64(0.7), SamplingQuote: "a sentence the card does not contain",
			ToolCallParser: str("hermes"), ParserQuote: "another sentence the card does not contain"}}
	if noteFor(Validate(in).Notes, "", "not in the card") {
		t.Error("a rejected reading the command overrides anyway was reported")
	}
}

// Without the helper's reading, the command that sets the most for the model
// is used, not the first: cards open with the plainest recipe. A richer one
// for the base model does not win over this repository's own.
func TestWithoutTheHelperTheMostCompleteCommandIsUsed(t *testing.T) {
	raw := "```\nvllm serve org/quant --tensor-parallel-size 4\n```\n\n" +
		"```\nvllm serve org/quant --tensor-parallel-size 4 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder\n```\n\n" +
		"```\nvllm serve org/quant --tensor-parallel-size 4 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'\n```\n\n" +
		"```\nvllm serve org/base --tensor-parallel-size 8 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --speculative-config '{\"method\":\"mtp\"}' --enable-prefix-caching --trust-remote-code\n```\n"
	m := &models.Model{ID: "org/quant"}
	in := Inputs{Model: m, Card: Card{Raw: raw, Text: raw}, Commands: ExtractCommands(raw), MachineEnv: machineEnv}
	c := Validate(in)
	if c.Command == nil || c.Command.Model != "org/quant" {
		t.Fatalf("chosen command: %+v", c.Command)
	}
	if rowByKey(c.Rows, "field:speculative_config") == nil || rowByKey(c.Rows, "field:tool_call_parser") == nil {
		t.Errorf("the fullest recipe was not proposed: %+v", c.Rows)
	}
	if rowByKey(c.Rows, "field:trust_remote_code") != nil {
		t.Error("the base model's command was used")
	}
	if !noteFor(c.Notes, "", "the most settings") {
		t.Error("no note of how the command was chosen")
	}

	// The helper's choice still stands when it makes one.
	one := 1
	in.Advice, in.HasAdvice = Advice{CommandIndex: &one}, true
	if c := Validate(in); c.Command == nil || c.Command.Settings().Start.ToolCallParser != "" {
		t.Errorf("the helper chose the first command, and got %+v", c.Command)
	}
}

// Qwen3.5-35B-A3B-FP8's card: a base command, then one variant per feature.
// What the variants add is proposed with the base: tool calling and the MTP
// head ticked, --language-model-only (vision off) unticked.
func TestTheVariantsOfTheChosenCommand(t *testing.T) {
	cmd := "vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3"
	raw := "```\n" + cmd + "\n```\n\n```\n" + cmd + " --enable-auto-tool-choice --tool-call-parser qwen3_coder\n```\n\n" +
		"```\n" + cmd + ` --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` + "\n```\n\n" +
		"```\n" + cmd + " --language-model-only\n```\n\n" +
		"```\nvllm serve Qwen/Qwen3.5-35B-A3B-FP8 --tensor-parallel-size 8 --enforce-eager\n```\n"
	m := &models.Model{ID: "Qwen/Qwen3.5-35B-A3B-FP8"}
	one := 1
	in := Inputs{Model: m, Card: Card{Raw: raw, Text: raw}, Commands: ExtractCommands(raw), MachineEnv: machineEnv,
		HasAdvice: true, Advice: Advice{CommandIndex: &one, ToolCallParser: str("qwen3_coder"), ParserQuote: "--tool-call-parser qwen3_coder"}}
	c := Validate(in)

	for key, ticked := range map[string]bool{
		"field:reasoning_parser": true, "field:enable_auto_tool_choice": true, "field:tool_call_parser": true,
		"field:speculative_config": true, "field:language_model_only": false,
	} {
		r := rowByKey(c.Rows, key)
		if r == nil || r.Ticked != ticked {
			t.Errorf("%s: %+v, want ticked=%v", key, r, ticked)
		}
	}
	if r := rowByKey(c.Rows, "field:speculative_config"); r == nil || !strings.Contains(r.Value, "qwen3_next_mtp") {
		t.Errorf("speculative config: %+v", r)
	}
	// The last command does not contain the chosen one: not a variant.
	if rowByKey(c.Rows, "field:enforce_eager") != nil {
		t.Error("a command that is not a variant added a row")
	}
	// Without the helper, the base is chosen: its variants bring everything,
	// where the richest single variant would bring only its own.
	in.HasAdvice, in.Advice = false, Advice{}
	if c := Validate(in); c.Command == nil || c.Command.Settings().Start.ToolCallParser != "" || rowByKey(c.Rows, "field:speculative_config") == nil {
		t.Errorf("without the helper: %+v", c.Command)
	}

	n := 0
	for _, r := range Validate(in).Rows {
		if r.Key == "field:tool_call_parser" {
			n++
		}
	}
	if n != 1 {
		t.Errorf("%d tool-call parser rows", n)
	}
}

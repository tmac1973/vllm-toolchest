package autoconfig

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// RowKind says what a row changes.
type RowKind string

const (
	RowField RowKind = "field" // one VLLMConfig field
	RowFlag  RowKind = "flag"  // one flag in extra flags
	RowEnv   RowKind = "env"   // one line of the env block
)

// Row is one change the card proposes: a setting, a flag or a variable, with
// where it came from and the card's own words for it. The operator can untick
// any row before saving.
type Row struct {
	// Key identifies the row in the review's form.
	Key   string
	Kind  RowKind
	Field string   // VLLMConfig JSON key, for a field row
	Value string   // the field's value as text
	Flag  []string // the flag and its value, for a flag row
	Env   string   // KEY=VALUE, for an env row

	Origin  string // always "model card": this machine's settings are not rows
	Reason  string
	Quote   string // the card's words for it
	Ticked  bool
	Warning string
}

// FlagChecker says whether the installed vLLM accepts a flag.
type FlagChecker interface {
	Known() bool
	Has(flag string) bool
}

// allFlags accepts everything: what a flag list that could not be read
// amounts to.
type allFlags struct{}

func (allFlags) Known() bool     { return false }
func (allFlags) Has(string) bool { return true }

// Inputs is what Validate checks the card and the helper's answer against.
type Inputs struct {
	Model    *models.Model
	Base     models.VLLMConfig
	Card     Card
	Commands []Command
	Inline   [][]string

	Advice    Advice
	HasAdvice bool

	Flags FlagChecker
	// Backends are the attention backends this image offers; nil when not
	// known, in which case any backend passes.
	Backends []string
	// MachineEnv is the environment every model already gets.
	MachineEnv []string
	// Drafts are the draft models on this disk.
	Drafts []*models.Model
}

// Checked is the card after checking: the rows to propose, notes for the
// review, and what the planner and the draft search need.
type Checked struct {
	Rows  []Row
	Notes []models.ProfileNote
	// CardKVDtype is the KV cache dtype the chosen command names, "" when it
	// names none: the one hardware value the card decides.
	CardKVDtype string
	// WantDraft is the speculative method the card recommends when the draft
	// it needs is not installed; DraftRepo the repository it names, if any.
	WantDraft string
	DraftRepo string
	// Command is the command the rows came from, nil when there was none.
	Command *Command
}

// fieldFlags is the flag each field is written as, for asking whether the
// image has it.
var fieldFlags = map[string]string{
	"dtype": "--dtype", "enforce_eager": "--enforce-eager", "quantization": "--quantization",
	"load_format": "--load-format", "enable_prefix_caching": "--enable-prefix-caching",
	"enable_chunked_prefill": "--enable-chunked-prefill", "max_num_batched_tokens": "--max-num-batched-tokens",
	"enable_auto_tool_choice": "--enable-auto-tool-choice", "tool_call_parser": "--tool-call-parser",
	"reasoning_parser": "--reasoning-parser", "mamba_cache_mode": "--mamba-cache-mode",
	"compilation_config": "--compilation-config", "disable_async_scheduling": "--no-async-scheduling",
	"language_model_only": "--language-model-only", "trust_remote_code": "--trust-remote-code",
	"attention_backend": "--attention-backend", "tokenizer": "--tokenizer",
	"speculative_config": "--speculative-config",
}

// hardwareFields are decided for this machine, never taken from a card.
var hardwareFields = map[string]string{
	"--tensor-parallel-size": "tensor_parallel_size", "-tp": "tensor_parallel_size",
	"--max-model-len": "max_model_len", "--gpu-memory-utilization": "gpu_memory_utilization",
	"--max-num-seqs": "max_num_seqs", "--kv-cache-memory": "kv_cache_memory",
}

const (
	overrideFlag         = "--override-generation-config"
	trustWarning         = "Lets the engine run Python code shipped in the model repository. Untick it if you do not trust the publisher."
	notListedReason      = "This image's vLLM does not list this flag, so it starts unticked: an unrecognised flag stops the engine starting."
	contextExtendReason  = "It extends the context past the model's native length, which the context chosen here does not exceed."
	containerPathReason  = "Its value is a path inside the author's container, which does not exist here."
	mistralWeightsReason = "It is for Mistral's consolidated weights, which were not downloaded: the same weights are here as Hugging Face shards, which load without it."
	mentionedReason      = "Mentioned in the card's text rather than in its command, so it starts unticked."
	originCard           = "model card"
	originMachine        = "this machine"
	originHelperSummary  = "helper summary"
	maxQuote             = 240
	minVerifiableQuote   = 12
)

type validator struct {
	in    Inputs
	out   Checked
	flags FlagChecker
	plain string // the card, normalised for quote checks
}

// Validate turns a card's commands and the helper's reading into rows and
// notes, keeping only values that are usable here and whose words are in the
// card. Hardware values never become rows: this machine's fit decides those.
func Validate(in Inputs) Checked {
	v := &validator{in: in, flags: in.Flags, plain: normalise(in.Card.Plain())}
	if v.flags == nil {
		v.flags = allFlags{}
	}

	chosen := v.chooseCommand()
	var start process.VLLMStartConfig
	var variantSpec, variantSpecQuote string
	if chosen != nil {
		v.out.Command = chosen
		s := chosen.Settings()
		start = s.Start
		v.fromCommand(chosen, s)
		variantSpec, variantSpecQuote = v.fromVariants(chosen, &start)
	}
	for _, c := range in.Commands {
		if c.Partial {
			v.note("", originCard, "The card also shows, for a special case: "+clip(c.Raw, maxQuote))
		}
	}
	v.fromInline(chosen)
	v.speculative(chosen, start, variantSpec, variantSpecQuote)
	v.samplingFromProse(chosen)
	v.parsersFromProse(start)
	v.otherAdvice()
	return v.out
}

func (v *validator) note(field, origin, reason string) {
	v.out.Notes = append(v.out.Notes, models.ProfileNote{Field: field, Reason: reason, Origin: origin})
}

func (v *validator) row(r Row) {
	r.Origin = originCard
	if r.Quote == "" {
		return // no words from the card, no row
	}
	if slices.ContainsFunc(v.out.Rows, func(x Row) bool { return x.Key == r.Key }) {
		return // the first source to propose a setting decides it
	}
	v.out.Rows = append(v.out.Rows, r)
}

// chooseCommand picks the command the rows come from.
func (v *validator) chooseCommand() *Command {
	full := fullCommands(v.in.Commands)
	if len(full) == 0 {
		return nil
	}
	a := v.in.Advice
	if v.in.HasAdvice && a.CommandIndex != nil {
		if i := *a.CommandIndex; i >= 1 && i <= len(full) {
			c := full[i-1]
			return &c
		}
		v.note("", originCard, fmt.Sprintf("The helper chose command %d of %d, which does not exist; the one with the most settings for the model is used.", *a.CommandIndex, len(full)))
	}
	c := mostComplete(full, v.in.Model)
	if len(full) > 1 && !(v.in.HasAdvice && a.CommandIndex != nil) {
		v.note("", originCard, fmt.Sprintf("The card shows %d complete commands; the one with the most settings for the model, from this repository's own card where it has one, is used.", len(full)))
	}
	return &c
}

// mostComplete picks, without the helper's reading, the command that sets the
// most model settings -- parsers, tool calling, a speculative config -- ties
// going to the first. Cards tend to open with the plainest recipe and build
// up: Qwen3.5-35B-A3B-FP8's first command had neither tool calling nor its MTP
// head, which the later ones added. Commands for this repository are preferred
// over the base model's when there are any.
func mostComplete(full []Command, m *models.Model) Command {
	pool := full
	if m != nil {
		var own []Command
		for _, c := range full {
			if strings.EqualFold(c.Model, m.ID) {
				own = append(own, c)
			}
		}
		if len(own) > 0 {
			pool = own
		}
	}
	// A command counts what its variants add too: a base recipe with one
	// variant per feature brings them all, where the richest single variant
	// would bring only its own.
	best, bestN := pool[0], -1
	for _, c := range pool {
		n := len(knownFieldValues(c.Settings().Start))
		for _, vr := range variantsOf(&c, full) {
			st, _ := process.ParseArgs(slices.Concat(vr.adds...))
			n += len(knownFieldValues(st))
		}
		if n > bestN {
			best, bestN = c, n
		}
	}
	return best
}

// fromCommand turns the chosen command's flags and variables into rows and
// notes.
func (v *validator) fromCommand(c *Command, s CommandSettings) {
	st := s.Start
	quote := func(needle string) string { return snippet(c.Raw, needle) }

	// Hardware values: notes, never rows.
	for _, h := range []struct {
		field string
		set   bool
		value string
	}{
		{"tensor_parallel_size", st.TensorParallelSize > 0, strconv.Itoa(st.TensorParallelSize)},
		{"max_model_len", st.MaxModelLen > 0, strconv.Itoa(st.MaxModelLen)},
		{"gpu_memory_utilization", st.GPUMemoryUtilization > 0, strconv.FormatFloat(st.GPUMemoryUtilization, 'g', -1, 64)},
		{"max_num_seqs", st.MaxNumSeqs > 0, strconv.Itoa(st.MaxNumSeqs)},
		{"kv_cache_memory", st.KVCacheMemory > 0, strconv.FormatInt(st.KVCacheMemory, 10)},
	} {
		if h.set {
			v.note(h.field, originMachine, fmt.Sprintf("The card's command used %s. This is chosen for this machine instead.", h.value))
		}
	}

	if st.KVCacheDtype != "" {
		switch st.KVCacheDtype {
		case "auto", "fp8", "fp8_e5m2", "fp8_e4m3":
			v.out.CardKVDtype = st.KVCacheDtype
		default:
			v.note("kv_cache_dtype", originCard, fmt.Sprintf("The card's command uses a KV cache dtype of %q, which this tool does not offer.", st.KVCacheDtype))
		}
	}

	for _, f := range knownFieldValues(st) {
		switch f.field {
		case "speculative_config":
			continue // rule 12
		case "chat_template":
			v.note("chat_template", originCard, "The card's command names a chat template file, "+f.value+", which is on the author's machine.")
			continue
		case "tokenizer":
			if filepath.IsAbs(f.value) {
				v.note("tokenizer", originCard, "The card's command names a tokenizer at "+f.value+", a path on the author's machine.")
				continue
			}
		}
		v.fieldRow(f.field, f.value, quote(fieldFlags[f.field]), "From the card's command.", true)
	}

	for _, g := range s.Rest {
		v.flagRow(g, quote(g[0]), true)
	}

	for _, e := range s.Env {
		v.envRow(e, quote(e))
	}
}

type fieldValue struct{ field, value string }

// knownFieldValues lists the fields a command set, as text. Hardware fields
// and the KV dtype are handled apart.
func knownFieldValues(st process.VLLMStartConfig) []fieldValue {
	var out []fieldValue
	add := func(field, value string, set bool) {
		if set {
			out = append(out, fieldValue{field, value})
		}
	}
	add("dtype", st.Dtype, st.Dtype != "" && st.Dtype != "auto")
	add("enforce_eager", "true", st.EnforceEager)
	add("trust_remote_code", "true", st.TrustRemoteCode)
	add("quantization", st.Quantization, st.Quantization != "")
	add("load_format", st.LoadFormat, st.LoadFormat != "" && st.LoadFormat != "auto")
	add("enable_prefix_caching", "true", st.EnablePrefixCaching)
	add("enable_chunked_prefill", "true", st.EnableChunkedPrefill)
	add("max_num_batched_tokens", strconv.Itoa(st.MaxNumBatchedTokens), st.MaxNumBatchedTokens > 0)
	add("enable_auto_tool_choice", "true", st.EnableAutoToolChoice)
	add("tool_call_parser", st.ToolCallParser, st.ToolCallParser != "")
	add("reasoning_parser", st.ReasoningParser, st.ReasoningParser != "")
	add("attention_backend", st.AttentionBackend, st.AttentionBackend != "")
	add("mamba_cache_mode", st.MambaCacheMode, st.MambaCacheMode != "")
	add("speculative_config", st.SpeculativeConfig, st.SpeculativeConfig != "")
	add("compilation_config", st.CompilationConfig, st.CompilationConfig != "")
	add("disable_async_scheduling", "true", st.DisableAsyncScheduling)
	add("language_model_only", "true", st.LanguageModelOnly)
	add("tokenizer", st.Tokenizer, st.Tokenizer != "")
	add("chat_template", st.ChatTemplate, st.ChatTemplate != "")
	return out
}

// fieldRow proposes one field. prescribed is false for a value only
// mentioned in the card's text, which starts unticked.
func (v *validator) fieldRow(field, value, quote, reason string, prescribed bool) {
	r := Row{Key: "field:" + field, Kind: RowField, Field: field, Value: value, Quote: quote, Reason: reason, Ticked: prescribed}
	if !prescribed {
		r.Reason = mentionedReason
	}
	if current, ok := fieldText(v.in.Base, field); ok && current == value {
		r.Reason = "Already set. " + r.Reason
	}
	switch {
	case field == "trust_remote_code":
		r.Warning = trustWarning
	case field == "load_format" && value == "mistral" && !hasConsolidatedWeights(v.in.Model):
		r.Ticked = false
		r.Reason = mistralWeightsReason
	case field == "attention_backend" && v.in.Backends != nil && !slices.Contains(v.in.Backends, value):
		r.Ticked = false
		r.Reason = "This image does not offer the " + value + " attention backend, so it starts unticked."
	case !v.flags.Has(fieldFlags[field]):
		r.Ticked = false
		r.Reason = notListedReason
	}
	v.row(r)
}

// hasConsolidatedWeights reports Mistral's single-file weights in the model's
// directory, which --load_format mistral reads. A download leaves them out
// when the repo has the same weights as Hugging Face shards.
func hasConsolidatedWeights(m *models.Model) bool {
	if m == nil || m.LocalPath == "" {
		return false
	}
	entries, err := os.ReadDir(m.LocalPath)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if huggingface.IsConsolidatedWeights(e.Name()) {
			return true
		}
	}
	return false
}

// contextExtending reports a flag or variable whose job is to serve past the
// model's native context, which a planned context never does.
func contextExtending(name, value string) bool {
	switch name {
	case "--rope-scaling", "--rope-theta", "VLLM_ALLOW_LONG_MAX_MODEL_LEN":
		return true
	case "--hf-overrides":
		return strings.Contains(strings.ToLower(value), "rope")
	}
	return false
}

// flagRow proposes one flag this tool has no field for.
func (v *validator) flagRow(g []string, quote string, prescribed bool) {
	name := g[0]
	value := ""
	if len(g) > 1 {
		value = g[1]
	}
	r := Row{Key: "flag:" + name, Kind: RowFlag, Flag: g, Quote: quote, Ticked: prescribed, Reason: "From the card's command."}
	if name == overrideFlag {
		r.Reason = "Sampling settings from the card's command."
	}
	switch {
	case !prescribed:
		r.Reason = mentionedReason
	case name == "--config-format" && value == "mistral" && !hasConsolidatedWeights(v.in.Model):
		// It goes with --load_format mistral: the model it builds expects
		// the consolidated weights' names, not the shards'.
		r.Ticked = false
		r.Reason = mistralWeightsReason
	case contextExtending(name, value):
		r.Ticked = false
		r.Reason = contextExtendReason
	case !v.flags.Has(name):
		r.Ticked = false
		r.Reason = notListedReason
	}
	v.row(r)
}

// envRow proposes one variable from the command.
func (v *validator) envRow(line, quote string) {
	name, value, _ := strings.Cut(line, "=")
	if slices.Contains(v.in.MachineEnv, line) {
		return // every model already gets it
	}
	set := config.EnvSet{Extra: line}
	if err := set.Validate(); err != nil {
		v.note("env", originCard, fmt.Sprintf("The card sets %s, which is not a valid variable here: %v.", name, err))
		return
	}
	r := Row{Key: "env:" + name, Kind: RowEnv, Env: line, Quote: quote, Ticked: true, Reason: "From the card's command."}
	if strings.Contains(v.in.Base.Env, line) {
		r.Reason = "Already set. " + r.Reason
	}
	switch w := set.Warnings(); {
	case len(w) > 0:
		r.Ticked = false
		r.Reason = w[0] + "."
	case strings.HasPrefix(value, "/"):
		r.Ticked = false
		r.Reason = containerPathReason
	case contextExtending(name, value):
		r.Ticked = false
		r.Reason = contextExtendReason
	}
	v.row(r)
}

// fromInline proposes the known flags the card mentions outside its command.
func (v *validator) fromInline(chosen *Command) {
	for _, g := range v.in.Inline {
		name := g[0]
		if chosen != nil && slices.ContainsFunc(chosen.Args, func(a string) bool {
			n, _, _ := strings.Cut(a, "=")
			return n == name
		}) {
			continue
		}
		span := strings.Join(g, " ")
		if field, ok := hardwareFields[name]; ok {
			v.note(field, originMachine, fmt.Sprintf("The card's text mentions %s. This is chosen for this machine instead.", span))
			continue
		}
		switch name {
		case "--kv-cache-dtype":
			v.note("kv_cache_dtype", originCard, "The card's text mentions "+span+"; its command does not set it.")
			continue
		case "--speculative-config":
			continue // rule 12
		}
		st, rest := process.ParseArgs(g)
		if len(rest) > 0 {
			continue // an owned flag, or a value that did not parse
		}
		for _, f := range knownFieldValues(st) {
			v.fieldRow(f.field, f.value, clip(span, maxQuote), "", false)
		}
	}
}

// speculative proposes speculative decoding, pairing an installed draft
// when the card's recommendation needs one.
//
// variantSpec is a speculative config one of the chosen command's variants
// adds, with its quote: used, ticked, when the command itself has none.
func (v *validator) speculative(chosen *Command, start process.VLLMStartConfig, variantSpec, variantQuote string) {
	type source struct {
		raw, quote string
		prescribed bool
	}
	var src *source
	if start.SpeculativeConfig != "" {
		src = &source{start.SpeculativeConfig, snippet(chosen.Raw, "--speculative-config"), true}
	} else if variantSpec != "" {
		src = &source{variantSpec, variantQuote, true}
	} else {
		for _, g := range v.in.Inline {
			if g[0] == "--speculative-config" && len(g) > 1 {
				src = &source{g[1], clip(strings.Join(g, " "), maxQuote), false}
				break
			}
		}
	}

	if src != nil {
		ref, ok := models.ParseSpeculative(src.raw)
		if !ok || ref.Method == "" {
			v.note("speculative_config", originCard, "The card's speculative config could not be read: "+clip(src.raw, 120))
			return
		}
		if ref.Model == "" {
			v.fieldRow("speculative_config", src.raw, src.quote,
				"Speculative decoding with the model's own draft head, as the card gives it. It speeds up generation and does not change the answers.", src.prescribed)
			return
		}
		repo := ""
		if isRepoID(ref.Model) {
			repo = ref.Model
		}
		v.pairDraft(ref.Method, ref.Tokens, repo, src.quote)
		return
	}

	a := v.in.Advice
	if !v.in.HasAdvice || a.DraftMethod == nil || *a.DraftMethod == "none" {
		return
	}
	if !v.verified(a.SpeculativeQuote) {
		v.unverified("draft method", *a.DraftMethod, a.SpeculativeQuote)
		return
	}
	switch *a.DraftMethod {
	case "mtp":
		v.note("speculative_config", originCard,
			`The card mentions the model's own MTP head. Add a speculative config with {"method": "mtp", "num_speculative_tokens": N} to use it; the card does not say how many tokens.`)
	default:
		repo := ""
		if a.DraftRepo != nil && isRepoID(*a.DraftRepo) {
			repo = *a.DraftRepo
		}
		v.pairDraft(*a.DraftMethod, 0, repo, clip(a.SpeculativeQuote, maxQuote))
	}
}

// pairDraft proposes the first installed draft that matches the method and
// the model, or records that the draft is wanted.
func (v *validator) pairDraft(method string, tokens int, repo, quote string) {
	var fits []*models.Model
	for _, d := range v.in.Drafts {
		if d.IsDraft() && !d.Orphaned && d.HFConfig.Draft.Method == method && models.DraftMismatch(v.in.Model, d) == "" {
			fits = append(fits, d)
		}
	}
	sort.Slice(fits, func(i, j int) bool { return fits[i].ID < fits[j].ID })
	if len(fits) == 0 {
		v.out.WantDraft, v.out.DraftRepo = method, repo
		v.note("speculative_config", originCard, fmt.Sprintf("The card recommends speculative decoding with a %s draft model, which is not installed.", method))
		return
	}
	d := fits[0]
	t := 0
	if tokens >= 1 && tokens <= d.MaxDraftTokens() {
		t = tokens
	}
	// A matching draft does not change the answers, so this is the one
	// mention that starts ticked even when it is only in the card's text.
	v.fieldRow("speculative_config", d.SpeculativeConfigFor(t), quote,
		fmt.Sprintf("Speculative decoding with the installed draft %s, as the card recommends. It speeds up generation and does not change the answers.", draftName(d)), true)
}

func draftName(d *models.Model) string {
	if d.DisplayName != "" {
		return d.DisplayName
	}
	return d.ID
}

var repoID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)

func isRepoID(s string) bool { return repoID.MatchString(s) }

// samplingFromProse proposes sampling values the card states in prose, when
// nothing more structured already sets them.
func (v *validator) samplingFromProse(chosen *Command) {
	a := v.in.Advice
	if !v.in.HasAdvice {
		return
	}
	if noSampling(a) && v.verified(a.SamplingQuote) {
		a = fillSampling(a)
	}
	values := map[string]float64{}
	var rejected []string
	check := func(key string, val *float64, lo, hi float64) {
		if val == nil {
			return
		}
		if *val < lo || *val > hi {
			rejected = append(rejected, fmt.Sprintf("%s of %g is outside %g to %g", key, *val, lo, hi))
			return
		}
		values[key] = *val
	}
	check("temperature", a.Temperature, 0, 2)
	check("top_p", a.TopP, 0, 1)
	check("min_p", a.MinP, 0, 1)
	check("presence_penalty", a.PresencePenalty, 0, 2)
	check("repetition_penalty", a.RepetitionPenalty, 0, 2)
	if a.TopK != nil {
		k := float64(*a.TopK)
		check("top_k", &k, 0, 1000)
	}
	for _, r := range rejected {
		v.note("extra_flags", originCard, "The helper read a "+r+", so it was not used.")
	}
	if len(values) == 0 {
		return
	}
	if !v.verified(a.SamplingQuote) {
		// Only worth saying when the reading could have been used: with the
		// command's own override, or a generation_config.json setting every
		// value read, it could not.
		if !commandOverrides(chosen) && !fileSetsAll(v.in.Model, values) {
			v.unverified("sampling values", "", a.SamplingQuote)
		}
		return
	}
	// The sentence has to state each value, not merely exist: a value the
	// quote does not contain was not read from it.
	for k, val := range values {
		if !numberInQuote(a.SamplingQuote, val) {
			delete(values, k)
			v.note("extra_flags", originCard, fmt.Sprintf("The helper read a %s of %g, but the sentence it cited does not give that value, so it was not used.", k, val))
		}
	}
	if len(values) == 0 {
		return
	}

	var override map[string]any
	if chosen != nil {
		for _, g := range chosen.Settings().Rest {
			if g[0] == overrideFlag && len(g) > 1 {
				json.Unmarshal([]byte(g[1]), &override)
			}
		}
	}
	switch {
	case override != nil:
		if differs(values, override) {
			v.note("extra_flags", originCard, "The card's text gives different sampling values from its command; the command's are used.")
		}
	case genDefaultsSet(v.in.Model):
		file := genDefaultsMap(v.in.Model.GenDefaults)
		if differs(values, file) {
			v.note("extra_flags", originCard, "The card's text gives different sampling values from the model's generation_config.json; that file's are used.")
		}
		// What the file does not set, the card's value fills: vLLM merges the
		// override into the file's values. Found on compute: Qwen3.5's file
		// sets temperature, top_p and top_k, and its card adds a presence
		// penalty of 1.5 that was dropped without a word.
		missing := map[string]float64{}
		for k, val := range values {
			if _, set := file[k]; !set && !neutralSampling(k, val) {
				missing[k] = val
			}
		}
		if len(missing) > 0 {
			data, _ := json.Marshal(missing)
			v.row(Row{Key: "flag:" + overrideFlag, Kind: RowFlag, Flag: []string{overrideFlag, string(data)},
				Quote: clip(a.SamplingQuote, maxQuote), Ticked: true,
				Reason: "Sampling settings the card recommends that the model's generation_config.json does not set. vLLM adds them to that file's."})
		}
	default:
		data, _ := json.Marshal(values)
		r := Row{Key: "flag:" + overrideFlag, Kind: RowFlag, Flag: []string{overrideFlag, string(data)},
			Quote: clip(a.SamplingQuote, maxQuote), Ticked: true,
			Reason: "Sampling settings the card recommends. The model publishes no generation_config.json, so nothing else sets them."}
		v.row(r)
	}
}

// fileSetsAll reports a generation_config.json that sets every value read.
func fileSetsAll(m *models.Model, values map[string]float64) bool {
	if !genDefaultsSet(m) {
		return false
	}
	file := genDefaultsMap(m.GenDefaults)
	for k := range values {
		if _, ok := file[k]; !ok {
			return false
		}
	}
	return true
}

// neutralSampling reports a value that is vLLM's own default, which an
// override need not repeat: a min_p of 0, a presence penalty of 0, a
// repetition penalty of 1.
func neutralSampling(key string, val float64) bool {
	switch key {
	case "min_p", "presence_penalty":
		return val == 0
	case "repetition_penalty":
		return val == 1
	}
	return false
}

func genDefaultsSet(m *models.Model) bool {
	if m == nil {
		return false
	}
	g := m.GenDefaults
	return g.Temperature != nil || g.TopP != nil || g.TopK != nil || g.RepetitionPenalty != nil
}

func genDefaultsMap(g models.GenDefaults) map[string]any {
	out := map[string]any{}
	if g.Temperature != nil {
		out["temperature"] = *g.Temperature
	}
	if g.TopP != nil {
		out["top_p"] = *g.TopP
	}
	if g.TopK != nil {
		out["top_k"] = float64(*g.TopK)
	}
	if g.RepetitionPenalty != nil {
		out["repetition_penalty"] = *g.RepetitionPenalty
	}
	return out
}

// differs reports a prose value that the used source gives differently.
func differs(prose map[string]float64, used map[string]any) bool {
	for k, pv := range prose {
		if uv, ok := used[k].(float64); ok && fmt.Sprintf("%.3f", uv) != fmt.Sprintf("%.3f", pv) {
			return true
		}
	}
	return false
}

// parsersFromProse proposes parsers the card names in prose, when its
// command set none.
func (v *validator) parsersFromProse(start process.VLLMStartConfig) {
	a := v.in.Advice
	if !v.in.HasAdvice || (a.ReasoningParser == nil && a.ToolCallParser == nil) {
		return
	}
	if start.ReasoningParser != "" && start.ToolCallParser != "" {
		return // the command names both; prose cannot add anything
	}
	if !v.verified(a.ParserQuote) {
		v.unverified("a parser", "", a.ParserQuote)
		return
	}
	quote := clip(a.ParserQuote, maxQuote)
	named := func(p *string, kind string) bool {
		if p == nil || *p == "" {
			return false
		}
		if !namesParser(a.ParserQuote, *p, kind) {
			v.note("", originCard, "The helper read a "+kind+" parser named "+*p+", but the sentence it cited does not name it as one, so it was not used.")
			return false
		}
		return true
	}
	if !named(a.ReasoningParser, "reasoning") {
		a.ReasoningParser = nil
	}
	if !named(a.ToolCallParser, "tool") {
		a.ToolCallParser = nil
	}
	if a.ReasoningParser != nil && start.ReasoningParser == "" && *a.ReasoningParser != "" {
		v.fieldRow("reasoning_parser", *a.ReasoningParser, quote, "The reasoning parser the card names.", true)
	}
	if a.ToolCallParser != nil && start.ToolCallParser == "" && *a.ToolCallParser != "" {
		v.fieldRow("enable_auto_tool_choice", "true", quote, "Tool calling, which the card describes.", true)
		v.fieldRow("tool_call_parser", *a.ToolCallParser, quote, "The tool-call parser the card names.", true)
	}
}

// namesParser reports a quote that names parser as a parser of kind
// ("reasoning" or "tool"): the whole name, with the kind in the words beside
// it -- "--reasoning-parser qwen3", "the qwen3 reasoning parser". The name
// alone is not enough. On Mistral Small 3.2, which does not reason, the
// helper read a reasoning parser named mistral and cited "--tool-call-parser
// mistral": the name was in the quote, as the tool parser's.
func namesParser(quote, parser, kind string) bool {
	q, name := strings.ToLower(quote), strings.ToLower(parser)
	for from := 0; ; {
		i := strings.Index(q[from:], name)
		if i < 0 {
			return false
		}
		i += from
		end := i + len(name)
		from = end
		if i > 0 && isNameChar(q[i-1]) || end < len(q) && isNameChar(q[end]) {
			continue // part of a longer name: qwen3 in qwen3_coder
		}
		// A flag between the name and the words beside it ends them:
		// "--reasoning-parser qwen3 --tool-call-parser qwen3_coder" names
		// qwen3_coder as the tool parser only.
		before := q[max(0, i-40):i]
		if j := strings.LastIndex(before, "--"); j >= 0 {
			before = before[j:]
		}
		after := q[end:min(len(q), end+30)]
		if j := strings.Index(after, "--"); j >= 0 {
			after = after[:j]
		}
		if strings.Contains(before, kind) || strings.Contains(after, kind) {
			return true
		}
	}
}

func isNameChar(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9'
}

// otherAdvice passes on what the helper read that is not a setting.
func (v *validator) otherAdvice() {
	a := v.in.Advice
	if !v.in.HasAdvice {
		return
	}
	if a.RecommendedContext != nil {
		if v.verified(a.ContextQuote) && contextInQuote(a.ContextQuote, *a.RecommendedContext) {
			v.note("max_model_len", originCard, fmt.Sprintf("The card recommends a context of %d tokens. The context is chosen for this machine from the size you asked for. The card says: “%s”",
				*a.RecommendedContext, clip(a.ContextQuote, maxQuote)))
		}
	}
	n := 0
	for _, s := range a.OtherNotes {
		if s = strings.TrimSpace(s); s != "" && n < 5 {
			v.note("", originHelperSummary, clip(s, 300))
			n++
		}
	}
}

func (v *validator) verified(quote string) bool {
	return quoteInCard(v.plain, quote)
}

// unverified reports a reading whose cited sentence is not in the card, and
// shows the sentence, so the operator can see what the helper claimed and
// judge for themselves -- and so a check that is too strict can be noticed.
func (v *validator) unverified(what, value, quote string) {
	msg := "The helper reported " + what
	if value != "" {
		msg += " " + value
	}
	msg += ", but the sentence it cited is not in the card, so it was not used."
	if q := strings.TrimSpace(quote); q != "" {
		msg += " It cited: “" + clip(q, 400) + "”"
	} else {
		msg += " It cited nothing."
	}
	v.note("", originCard, msg)
}

var spaceRuns = regexp.MustCompile(`\s+`)

// markdownMarks are the formatting characters a reader does not see and a
// helper does not copy: code ticks, emphasis, blockquote markers.
var markdownMarks = strings.NewReplacer("`", "", "**", "", "*", "", "> ", " ")

// normalise is text as compared for a quote: lower case, markdown marks
// dropped, whitespace collapsed. Found on compute: the MoE's card lists its
// sampling values as `temperature=0.7`, `top_p=0.80`, and a helper quoting
// that line without the ticks was told its sentence was not in the card.
func normalise(s string) string {
	s = markdownMarks.Replace(strings.ToLower(s))
	return strings.TrimSpace(spaceRuns.ReplaceAllString(s, " "))
}

// quoteInCard reports whether a quote is in the card, allowing for case and
// whitespace. A quote shorter than a few words proves nothing and fails.
func quoteInCard(normalisedCard, quote string) bool {
	q := normalise(quote)
	if len(q) < minVerifiableQuote {
		return false
	}
	if strings.Contains(normalisedCard, q) {
		return true
	}
	// A quote of several lines -- a list, say -- may skip a blank line or a
	// marker the card has between them. Each line must still be in the card.
	found := 0
	for _, line := range strings.Split(quote, "\n") {
		l := normalise(listMarker.ReplaceAllString(strings.TrimSpace(line), ""))
		if len(l) < minVerifiableQuote {
			continue
		}
		if !strings.Contains(normalisedCard, l) {
			return false
		}
		found++
	}
	return found > 1
}

// listMarker is a list bullet or number at the start of a line.
var listMarker = regexp.MustCompile(`^([-*+]|\d+[.)])\s+`)

// snippet is up to maxQuote characters of text with needle in view.
func snippet(text, needle string) string {
	text = strings.TrimSpace(spaceRuns.ReplaceAllString(text, " "))
	if len(text) <= maxQuote {
		return text
	}
	i := strings.Index(text, needle)
	if i < 0 {
		return text[:maxQuote] + "…"
	}
	start := max(0, i-maxQuote/3)
	end := min(len(text), start+maxQuote)
	out := text[start:end]
	if start > 0 {
		out = "…" + out
	}
	if end < len(text) {
		out += "…"
	}
	return out
}

func clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// fieldText is a config field as the text a row carries, for the fields a
// row can set.
func fieldText(c models.VLLMConfig, field string) (string, bool) {
	b := strconv.FormatBool
	switch field {
	case "dtype":
		return c.Dtype, true
	case "enforce_eager":
		return b(c.EnforceEager), true
	case "trust_remote_code":
		return b(c.TrustRemoteCode), true
	case "quantization":
		return c.Quantization, true
	case "load_format":
		return c.LoadFormat, true
	case "enable_prefix_caching":
		return b(c.EnablePrefixCaching), true
	case "enable_chunked_prefill":
		return b(c.EnableChunkedPrefill), true
	case "max_num_batched_tokens":
		return strconv.Itoa(c.MaxNumBatchedTokens), true
	case "enable_auto_tool_choice":
		return b(c.EnableAutoToolChoice), true
	case "tool_call_parser":
		return c.ToolCallParser, true
	case "reasoning_parser":
		return c.ReasoningParser, true
	case "attention_backend":
		return c.AttentionBackend, true
	case "mamba_cache_mode":
		return c.MambaCacheMode, true
	case "speculative_config":
		return c.SpeculativeConfig, true
	case "compilation_config":
		return c.CompilationConfig, true
	case "disable_async_scheduling":
		return b(c.DisableAsyncScheduling), true
	case "language_model_only":
		return b(c.LanguageModelOnly), true
	case "tokenizer":
		return c.Tokenizer, true
	}
	return "", false
}

// numberInQuote reports whether a quote states a value: 0.6 as "0.6" or
// ".6", 20 as "20".
func numberInQuote(quote string, val float64) bool {
	q := strings.ToLower(quote)
	for _, form := range []string{strconv.FormatFloat(val, 'f', -1, 64), strconv.FormatFloat(val, 'g', -1, 64)} {
		if strings.Contains(q, form) || strings.Contains(q, strings.TrimPrefix(form, "0")) && strings.HasPrefix(form, "0.") {
			return true
		}
	}
	return false
}

// contextInQuote reports whether a quote states a context length, in any of
// the ways cards write one: 262144, 262,144, 256K or 256k.
func contextInQuote(quote string, tokens int) bool {
	q := strings.ToLower(strings.ReplaceAll(quote, "\u00a0", " "))
	forms := []string{strconv.Itoa(tokens), groupDigits(tokens)}
	if tokens%1024 == 0 {
		forms = append(forms, strconv.Itoa(tokens/1024)+"k", strconv.Itoa(tokens/1024)+" k")
	}
	if tokens%1000 == 0 {
		forms = append(forms, strconv.Itoa(tokens/1000)+"k")
	}
	for _, f := range forms {
		if strings.Contains(q, f) {
			return true
		}
	}
	return false
}

func groupDigits(n int) string {
	s := strconv.Itoa(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// commandOverrides reports a chosen command that sets the sampling defaults.
func commandOverrides(chosen *Command) bool {
	if chosen == nil {
		return false
	}
	for _, g := range chosen.Settings().Rest {
		if g[0] == overrideFlag {
			return true
		}
	}
	return false
}

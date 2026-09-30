# Phase 08 — From a card to checked proposals

**Depends on:** phase 01 (`ProfileNote`), phase 05 (`Card`, `Command`,
`InlineFlags`, `ParseArgs`) · **Enables:** the run (phase 09) and everything
the review shows (phase 10)

## Goal

Decide what a card is allowed to change. The inputs are the commands phase 05
extracted and, when a helper was available, the form it filled in from the
card's prose. The output is a list of rows -- one per setting, flag or
variable -- each with its source, its reason, the card's own words where there
are any, and whether it starts ticked. Hardware values in the card never
become rows: they become notes, because this machine's fit decides those. The
helper's answer is untrusted throughout: a value whose quoted sentence is not
actually in the card is thrown away.

This phase is pure functions over data. It starts no engine and calls no
network; the helper is reached through a function the caller supplies.

## Files touched

- `internal/autoconfig/advice.go` — new. `Advice`, `adviceSchema`,
  `adviceInstructions`, `Ask`.
- `internal/autoconfig/validate.go` — new. `Row`, `Inputs`, `Checked`,
  `Validate`, `quoteInCard`.
- `internal/autoconfig/apply.go` — new. `DefaultTicks`, `Apply`.
- `internal/process/flags.go` — new. `JoinFlags`, `SetFlag`, `RemoveFlag`.
- `internal/autoconfig/advice_test.go`, `validate_test.go`, `apply_test.go`,
  `internal/process/flags_test.go` — new.

## Steps

1. `process/flags.go`, the write side of `SplitFlags`:
   - `func JoinFlags(args []string) string` quotes an element with single
     quotes when it contains whitespace, a quote or a brace, so that
     `SplitFlags(JoinFlags(a))` returns `a`.
   - `func SetFlag(extra string, flag []string) string` replaces an existing
     occurrence of `flag[0]` together with its value, or appends.
   - `func RemoveFlag(extra, name string) string` removes it with its value.

2. `Advice`, flat and nullable as llama-toolchest's is, because a small model
   fills a short flat form more reliably than nested optional objects:

   ```go
   type Advice struct {
       CommandIndex *int `json:"command_index"` // 1-based, into the numbered commands; null = none fits

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

       DraftMethod      *string `json:"draft_method"` // "dflash", "mtp", "eagle3", "draft_model", "none"
       DraftRepo        *string `json:"draft_repo"`
       SpeculativeQuote string  `json:"speculative_quote"`

       RecommendedContext *int   `json:"recommended_context"`
       ContextQuote       string `json:"context_quote"`

       OtherNotes []string `json:"other_notes"` // at most five
   }
   ```

   `adviceSchema()` makes every property required, nullable where the Go type
   is a pointer, `additionalProperties: false`, `other_notes` with
   `maxItems: 5`, and the required list sorted so the request is identical
   every time.

3. `adviceInstructions` tells the helper: use only what the card says; answer
   null and an empty quote when the card does not state a value; copy into
   each quote field the sentence that states the value; for sampling, when the
   card gives several sets, use the general or default one; `command_index` is
   the numbered command the card presents as the normal way to serve **this
   repository** on all of its GPUs, preferring one for this repository over
   one for the model it was made from, and null when none is a complete
   command; `draft_method` and `draft_repo` only when the card recommends
   speculative decoding; `other_notes` are at most five short sentences about
   running the model with vLLM, leaving out other servers and anything about
   training.

4. `func Ask(ctx, call CallFunc, m *models.Model, card Card, cmds []Command) (Advice, error)`
   where `type CallFunc func(ctx context.Context, schemaName string, schema map[string]any, system, user string, out any) error`.
   The user message is the repository ID, then the non-partial commands
   numbered from 1 with their raw text, then `card.Text` between `<<<` and
   `>>>`. An empty card is not sent; `Ask` returns a zero `Advice`.

5. `Row`:

   ```go
   type RowKind string // "field", "flag", "env"

   type Row struct {
       Key     string   // "field:reasoning_parser", "flag:--enable-expert-offload", "env:OMP_NUM_THREADS"
       Kind    RowKind
       Field   string   // VLLMConfig JSON key, for a field row
       Value   string   // the field's value as text
       Flag    []string // the flag and its value, for a flag row
       Env     string   // KEY=VALUE, for an env row
       Origin  string   // always "model card"; the machine's settings are not rows
       Reason  string
       Quote   string   // the card's words: see step 7's rule 0
       Ticked  bool
       Warning string
   }
   ```

6. `Inputs` is what `Validate` checks against: the `Model`, the `Base` config,
   the `Card`, the `Commands`, the `Inline` flags, `Advice` and `HasAdvice`,
   `Flags` (an interface with `Known() bool` and `Has(string) bool`, which
   phase 04's `flagSupport` satisfies), `Backends []string` (the attention
   backends this image offers, nil when unknown), `MachineEnv []string`, and
   `Drafts []*models.Model` (installed drafts).
   `Checked` is `Rows`, `Notes []models.ProfileNote`, `CardKVDtype string`,
   `WantDraft string` and `DraftRepo string`.

7. `Validate`, rule by rule. Every note it emits has origin "model card",
   except the hardware notes of rule 2, which are "this machine".
   0. **Every row carries the card's words.** A row from the chosen command
      has the command's raw text as its quote, cut to 240 characters with the
      flag or variable in question kept in view; a row from an inline mention
      has that code span; a row from prose has the helper's verified quote.
      A row with no quote is not produced.
   1. **Which command.** Partial commands are never chosen. If
      `Advice.CommandIndex` names a valid non-partial command, use it.
      Otherwise use the first non-partial command -- the model's own card is
      first in `Card.Raw`, so that is the quantizer's -- and add a note when
      there was more than one to choose from. No command at all is not an
      error.
   2. **Hardware values in the command** -- `tensor_parallel_size`,
      `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`,
      `kv_cache_memory` -- become notes with that field name: "The card's
      command used N. This is chosen for this machine instead."
   3. **`kv_cache_dtype`** is returned as `CardKVDtype` when it is one of
      `auto`, `fp8`, `fp8_e5m2`, `fp8_e4m3`, and is otherwise a note.
   4. **Known fields** become field rows with origin "model card" and the
      command as the quote, ticked when `Flags.Has` lists the flag the field is
      written as and unticked otherwise, with the same reason as rule 8: `dtype`, `enforce_eager`, `quantization`,
      `load_format`, `enable_prefix_caching`, `enable_chunked_prefill`,
      `max_num_batched_tokens`, `enable_auto_tool_choice`, `tool_call_parser`,
      `reasoning_parser`, `mamba_cache_mode`, `compilation_config`,
      `disable_async_scheduling`, `language_model_only`. A row whose value
      equals `Base`'s is kept and marked in its reason as already set, so
      "already right" does not read as "overlooked".
   5. **`trust_remote_code`** is a ticked row with the warning "Lets the
      engine run Python code shipped in the model repository. Untick it if you
      do not trust the publisher."
   6. **`attention_backend`** is ticked only when `Backends` is nil or
      contains the value; otherwise unticked, with the reason that this image
      does not offer it.
   7. **`tokenizer`** is a ticked row when the value is a repository ID and a
      note when it is an absolute path. **`chat_template`** is always a note:
      it is a file on the author's machine.
   8. **Other flags** in the command become flag rows, ticked when
      `Flags.Has(flag)`. An unticked one says "this image's vLLM does not list
      this flag". `--override-generation-config` is such a row, with the
      reason "sampling settings from the card's command".
   9. **Environment** becomes env rows. Unticked, with the reason, when the
      name ends in `_VISIBLE_DEVICES` (it hides cards from the engine; the
      wording already in `internal/config/runtime_env.go` is reused), or the
      value begins with `/` (a path inside the author's container). Dropped
      with a note when `config.EnvSet{Extra: line}.Validate()` rejects it.
      Skipped silently when `MachineEnv` already has the same name and value.
   10. **Partial commands** contribute one note each: "The card also shows,
       for a special case:" followed by the fragment.
   11. **Inline flags** not already in the chosen command -- which phase 05
       limits to flags this tool knows -- follow rules 2 and 4-7, with the reason "mentioned in the card's text", and the rows are
       **unticked**: the card mentions them, it does not prescribe them. An
       inline `--kv-cache-dtype` is a note, never `CardKVDtype`, for the same
       reason. An inline `--trust-remote-code` is unticked like the rest and
       carries rule 5's warning. `--speculative-config` is the exception and
       goes to rule 12.
   12. **Speculative decoding.** The wanted method comes from the chosen
       command's speculative config, else an inline one, else
       `Advice.DraftMethod` when its quote verifies. `"none"` means nothing is
       wanted. A written config that names no model (MTP) becomes a
       `speculative_config` row as written: ticked when it is in the chosen
       command, unticked when it is only mentioned inline. A method known only from prose has
       no config to copy: for `mtp` that is a note ("The card mentions the
       model's own MTP head. Add a speculative config with
       `{"method": "mtp", "num_speculative_tokens": N}` to use it."), because
       the number of tokens is the card's to state and it did not; for
       `dflash`, `eagle3` and `draft_model` the installed-draft search below
       applies, since `SpeculativeConfigFor` supplies the whole config. One
       that names a model needs a draft on this disk: take the installed
       drafts whose `HFConfig.Draft.Method` is the wanted method and for which
       `models.DraftMismatch(target, draft)` is empty, in ID order; with one
       or more, the row is ticked -- speculative decoding with a matching
       draft does not change the answers, which is why it is the one inline
       mention that starts ticked -- and its value is the first's
       `SpeculativeConfigFor(tokens)`, where `tokens` is the card's
       `num_speculative_tokens` when it is between 1 and the draft's
       `MaxDraftTokens()` and 0 (the maximum) otherwise. With none, set
       `WantDraft` to the method and `DraftRepo` to the card's model value
       when it is `owner/name`, else to `Advice.DraftRepo` when its quote
       verifies, and add a note that the draft is not installed.
   13. **Sampling from prose** is used only when the command carries no
       `--override-generation-config` **and** the model's `GenDefaults` is
       empty -- a `generation_config.json` is the publisher's structured
       statement and the engine applies it already. Each value is
       range-checked (temperature 0-2, top_p 0-1, min_p 0-1, presence 0-2,
       repetition 0-2, top_k 0-1000) and the quote must verify; what survives
       becomes one ticked flag row, `--override-generation-config` with the
       values as JSON, origin "model card". When the prose differs from the
       values that are used instead, that is a note: "The card's text gives
       different sampling values from its command; the command's are used." or
       "…from the model's generation_config.json; that file's are used."
   14. **Parsers from prose** are used only when the chosen command set none
       and the quote verifies: a reasoning parser as one row, a tool parser as
       two (`enable_auto_tool_choice` and `tool_call_parser`).
   15. **Recommended context** is always a note, and only when its quote
       verifies. **Other notes** cannot be checked against the card, since
       they are the helper's summary rather than a quotation: they become
       notes with origin "helper summary", trimmed, at most five, which the
       review shows under the heading "The helper's summary of the card, not
       checked", and they never become a setting.

8. `quoteInCard(raw, quote string) bool`: both sides lower-cased with runs of
   whitespace collapsed to one space; true when the quote is at least 12
   characters and is a substring. A value with an empty or unverified quote is
   dropped, with a note: "The helper reported a … of … but the sentence it
   cited is not in the card, so it was not used."

9. `func DefaultTicks(rows []Row) map[string]bool` returns each row's own
   `Ticked` by key. `func Apply(base models.VLLMConfig, rows []Row, ticked map[string]bool) (models.VLLMConfig, error)`;
   a nil `ticked` means `DefaultTicks(rows)`.
   Field rows set their field, parsing the text by the field's type and
   returning an error for a value that does not parse. Flag rows go into
   `ExtraFlags` with `process.SetFlag`, so a flag the base already had is
   replaced and the rest of the base's extra flags are kept. Env rows go into
   the `Env` block by name, replacing a line with the same key and appending
   otherwise. Only rows whose key is ticked are applied. Unknown field names
   are an error, which is what keeps a helper's answer from reaching anything
   this function does not name.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

## Test plan

Against phase 05's three excerpts, with no helper (`HasAdvice` false):

- 27B card with a compatible DFlash draft installed: rows for
  `enable_auto_tool_choice`, `tool_call_parser qwen3_coder`,
  `reasoning_parser qwen3`, `enable_chunked_prefill`,
  `max_num_batched_tokens 8192`, `compilation_config`, the
  `--override-generation-config` flag and a `speculative_config` naming the
  local draft with 7 tokens, all ticked; env rows for `OMP_NUM_THREADS`,
  `VLLM_ROCM_USE_AITER`, `GPU_MAX_HW_QUEUES`, `HSA_ENABLE_INTERRUPT` and
  `HSA_ENABLE_MWAITX` ticked, and for the three cache directories and
  `ROCR_VISIBLE_DEVICES` unticked; notes for TP 4, 262,144 context, 0.92
  utilization and 8 sequences; `CardKVDtype == "fp8"`. `Apply` with the
  default ticks reproduces the hand-built config's parsers, flags, draft and
  env exactly.
- The same with no draft installed: no speculative row, `WantDraft ==
  "dflash"`, a note.
- MoE card: the first of the two docker commands is chosen and a note says
  there were two; the partial `--hf-overrides` line is a note.
- Qwen3-14B-AWQ with a flag list lacking `--enable-reasoning`: that row is
  unticked with the reason; `reasoning_parser deepseek_r1` is ticked.

With a helper's `Advice`:

- `CommandIndex: 2` selects the second command; an out-of-range index falls
  back to the first.
- A temperature whose quote is in the card becomes the override row only when
  the model has no `GenDefaults` and the command has no override. With the
  command's override present and a different prose temperature, there is no
  prose row and one note saying the command's values are used; with
  `GenDefaults` present and a different prose temperature, one note saying the
  file's are used; with either present and the same value, no row and no
  note.
- A temperature of 7, and a value whose quote is not in the card, are each
  dropped with a note.
- `trust_remote_code` carries the warning; an unoffered attention backend is
  unticked.

`Apply`: replaces an existing `--override-generation-config` in the base's
extra flags and keeps an unrelated base flag; replaces `OMP_NUM_THREADS=4` with
the card's line; rejects a row naming a field it does not know.
`SplitFlags(JoinFlags(a)) == a` for arguments containing JSON with spaces and
quotes.

## Commit

```
feat(autoconfig): turn a card's command and a helper's reading into checked proposals
```

## Rollback

Revert the commit. No caller exists until phase 09.

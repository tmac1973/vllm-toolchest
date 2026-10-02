# Phase 09 — One run, end to end, without a screen

**Depends on:** phase 01 (profile record), phase 02 (planner), phase 04 (flag
support), phase 05 (card), phase 06 (engine loan), phase 07 (helper,
`llmcall`), phase 08 (advice, validation) · **Enables:** the button and the
review (phase 10)

## Goal

Put the pieces in order and produce a result: read the card, extract its
commands, borrow the engine for the helper when there is one, check the
answer, plan the hardware, and hold the outcome until someone saves or
discards it. The run is a function with its collaborators passed in, so it is
tested with fakes, and a small piece of server state allows one run at a time.
No route or template is added here; phase 10 puts a face on it.

## Files touched

- `internal/autoconfig/run.go` — new. `Deps`, `Result`, `Run`,
  `(*Result).Config`, `(*Result).Profile`.
- `internal/autoconfig/run_test.go` — new.
- `internal/api/autoconfig_run.go` — new. `autoconfigState`, `autoconfigRun`,
  `(*Server).startAutoconfig`, `autoconfigSnapshot`, `clearAutoconfigRun`,
  `autoconfigDeps`, `askHelper`.
- `internal/api/autoconfig_run_test.go` — new.
- `internal/api/server.go` — `autoconf autoconfigState` and
  `llm *llmcall.Client` fields.

## Steps

1. `Deps`:

   ```go
   type Deps struct {
       Model       *models.Model
       Base        models.VLLMConfig          // the live config when the run began
       Previous    func(cardHash string) (json.RawMessage, bool) // an earlier reading of this exact card
       Reread      bool                       // ignore Previous and ask the helper again
       Fetcher     Fetcher
       Flags       interface{ Known() bool; Has(string) bool }
       Backends    []string
       MachineEnv  []string
       Drafts      []*models.Model
       Plan        func(base models.VLLMConfig, cardKVDtype string) models.FitPlan
       Helper      CallFunc                   // nil when there is no helper to ask
       NoHelperWhy string                     // shown when Helper is nil
       CardChars   int
       Progress    func(string)
   }
   ```

2. `Result`: `ModelID`, `Class`, `Base`, `Plan models.FitPlan`, `Rows []Row`,
   `Notes []models.ProfileNote`, `CardSources []string`, `CardHash string`,
   `Advice json.RawMessage`, `CardKVDtype string` (kept so a re-plan at save
   still honours the card's choice), `AdviceFrom string` -- `"helper"` when the helper
   read the card in this run, `"previous"` when the last run's reading was
   reused, `""` when the text was not read -- `WantDraft`, `DraftRepo`, and an
   unexported `plan` holding `Deps.Plan`, used again at save (step 4).

3. `func Run(ctx context.Context, d Deps, class models.ContextClass) (*Result, error)`:
   1. `Progress("Reading the model card")`. `FetchCard`, then
      `ExtractCommands(card.Raw)` and `InlineFlags(card.Raw)`. An empty card is
      a note -- "No model card was found, so these settings come from this
      machine alone." -- and the run continues.
   2. Advice. In order of preference:
      - `Reread` is false and `d.Previous(card.Hash)` returns advice: decode
        it, `Progress("Using the card advice from the last run")`. Nothing is
        loaded.
      - `d.Helper != nil` and the card is not empty: `Ask`. The progress lines
        for this step come from the helper call itself (step 9), in the order
        the engine work happens. If it
        returns an error and `ctx` is done, return the context's error;
        otherwise continue without advice and add the note "The helper model
        could not read the card (…), so the card's text was not used. Its
        command still was."
      - Otherwise no advice, and the note is `NoHelperWhy`.
   3. `Progress("Checking the answer")`. `Validate`. When advice was used but
      it produced no row and no note beyond the command's, add the note "The
      helper found nothing in the card's text beyond its command."
   4. `Progress("Checking what fits on this machine")`. Build the config the
      card produces with `Apply(d.Base, rows, nil)` -- every row at its own
      default -- and call
      `d.Plan(thatConfig, checked.CardKVDtype)`. Planning on the card-applied
      config matters: a speculative config adds a drafter's weights and KV
      cache to the estimate, and an offload variable removes tens of gigabytes
      from it. If `Apply` fails, plan on `d.Base` and note the failure.
   Notes the run itself writes -- no card, no helper, the helper failed, the
   helper found nothing -- have origin "default".
   5. `r.Notes` holds the notes from validation and from the run itself, and
      never the plan's: those stay on `r.Plan`, so that a re-plan at save
      replaces them rather than leaving stale ones behind. Phase 08's rule-2
      notes state only what the card used; the value chosen for this machine
      is what the review's hardware table shows beside them.
   6. Order the notes by field, general notes last, and return.

4. `func (r *Result) Config(width string, ticked map[string]bool) (cfg models.VLLMConfig, plan models.FitPlan, err error)`:
   `Apply(r.Base, r.Rows, ticked)`. When `ticked` equals the defaults the plan
   is `r.Plan`; otherwise the plan is computed again with `r.plan` on the
   config just built and `r.CardKVDtype`, because unticking a speculative
   config or an offload variable changes what the model needs. Then copy the five hardware fields
   -- `TensorParallelSize`, `MaxModelLen`, `GPUMemoryUtilization`,
   `MaxNumSeqs`, `KVCacheDtype` -- from `plan.All.Config`, or from
   `plan.Narrow.Config` when `width == "narrow"` and it exists (falling back
   to `All` when the recomputed plan has no narrow option), and set
   `KVCacheMemory = 0`. When `plan.Known` is false the hardware fields are left
   as `Apply` returned them. The plan used is returned so the caller can say
   when it differs from the one the review showed.

5. `func (r *Result) Profile(width string, ticked map[string]bool, meta models.ProfileMeta) (models.ConfigProfile, models.FitPlan, error)`
   returns the config and plan from step 4 with `Source: ProfileSourceAutoconfig`,
   `Variant` and `VariantVersion` from `meta`, the notes -- one per ticked row,
   its `Field` being the row's field for a field row, `"extra_flags"` for a
   flag row and `"env"` for an env row, its reason naming the flag or variable
   for the latter two; then the chosen width's notes and the general notes
   of the plan actually used; then `r.Notes` -- and an
   `AutoconfigRecord` holding the class, the width, `FirstGuess` from the plan,
   the card sources and hash, and the raw advice.

6. Server state, `internal/api/autoconfig_run.go`, following
   llama-toolchest's `internal/api/autoconfig.go`:

   ```go
   type autoconfigState struct {
       mu         sync.Mutex
       run        *autoconfigRun
       lastAdvice map[string]storedAdvice // by model ID; in memory only
   }
   type storedAdvice struct {
       CardHash string
       Advice   json.RawMessage
   }
   type autoconfigRun struct {
       modelID  string
       class    models.ContextClass
       progress string
       restoreNote string
       done     bool
       result   *autoconfig.Result
       err      error
   }
   ```

   One run exists at a time. A finished run is kept until it is saved or
   discarded, or until another run starts for any model.

7. `func (s *Server) startAutoconfig(id string, class models.ContextClass, reread bool) error`
   refuses when the model is missing, orphaned, a draft or the helper, and
   when a run is in progress. It does not refuse on a busy engine: whether the
   helper is needed is only known once the card has been fetched and compared
   with the last run's, and if it is needed while the engine is busy,
   `borrowEngine` returns the reason, which `Run` treats as any other helper
   failure -- the run completes on the command and the machine, and the note
   says why the text was not read. It records the run and starts a goroutine with
   a 25-minute timeout that calls `Run` and stores the result. A deadline is
   reported as "Autoconfigure took longer than 25 minutes and was stopped".

8. `func (s *Server) autoconfigDeps(m *models.Model, run *autoconfigRun, reread bool) autoconfig.Deps`:
   - `Base` is `m.VLLMConfig`. `Previous(hash)` returns the first reading
     whose card hash equals `hash`, looking first in `lastAdvice[m.ID]` --
     written by every completed run that obtained advice, persisted nowhere --
     and then in the record on the model's Autoconfig profile. Checking both
     for a match, rather than taking whichever exists, means an older saved
     profile does not hide a newer discarded run's reading of the same card,
     and a stale entry in either is simply not a match.
   - `Fetcher` is `s.hfClient`, wrapped in a small in-memory cache keyed by
     repository with a 15-minute life, so a second run does not refetch.
   - `Flags` is `s.serveFlags()`; `Backends` is the descriptor's attention
     backends when the variant is known and declares any; `MachineEnv` is
     `s.configuredEnvPairs(nil)`; `Drafts` is every non-orphaned registry
     model for which `IsDraft()` is true.
   - `Plan` closes over `s.planInput(m, base, class, kv)` and
     `models.PlanFit`.
   - `CardChars` is `autoconfig.CardCharsForContext(models.HelperContext)`.
   - `Helper` is a closure over `run` --
     `func(ctx, name, schema, system, user, out) error { return s.askHelper(ctx, run, name, schema, system, user, out) }`
     -- when `s.helperModel()` is non-nil and
     `models.HelperFits(inventory, models.HelperUtil(cfg.GPUMemoryUtil))`
     says yes. `NoHelperWhy` is, respectively, "No helper
     model is installed, so the card's text was not read. Its command still
     was. Install the helper in Settings to include the rest." or the
     sentence from `HelperFits`.

9. `func (s *Server) askHelper(ctx context.Context, run *autoconfigRun, schemaName string, schema map[string]any, system, user string, out any) error`
   is what the `CallFunc` closure calls; `run` is how its progress lines reach
   the right run, through a setter that takes `s.autoconf.mu`. It builds the
   loan from the helper model:
   `ModelPath: process.ResolveModelPath(helper.LocalPath)`,
   `Args: process.BuildArgs(start)` where `start` is
   `models.HelperConfig(models.HelperUtil(cfg.GPUMemoryUtil)).StartConfig()` with
   `ServedModelName = models.HelperServedName`, `Env: s.launchEnv(helper)`,
   `ModelID: helper.ID`, `StartWait: 10 * time.Minute`. `borrowEngine`'s
   `restore` report, when non-empty, is stored on `run.restoreNote`; the
   goroutine of step 7 appends it to the finished result's notes with origin
   "default", so an answer the helper gave survives a model that could not be
   brought back, and the operator is still told. It calls
   `s.borrowEngine(ctx, "autoconfigure", loan, progress, work)`, and `work`
   calls `progress("Asking the helper model to read the card")` and then
   `s.llm.JSON(ctx, baseURL, models.HelperServedName, schemaName, schema,
   messages, out)`. Progress lines from the loan flow into the run's
   progress. Because the loan is taken inside the call, a run that reuses
   earlier advice never touches the engine.

10. `autoconfigSnapshot()` returns a copy of the current run;
    `clearAutoconfigRun(id)` drops a finished run for that model.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

## Test plan

`run_test.go`, with a fake `Fetcher` serving phase 05's excerpts, a fake
`CallFunc`, and a `Plan` stub:

- No helper: the result has the command's rows, the hardware plan, and the
  no-helper note; `AdviceFrom` is `""`.
- Helper answering: `CommandIndex` is honoured, `Advice` is stored raw,
  `AdviceFrom` is `"helper"`.
- Helper answering with every value null: the "found nothing" note.
- Helper failing: the run completes with the command's rows and the failure
  note. Helper failing because the context was cancelled: `Run` returns the
  context error.
- Previous record with the same card hash: the helper is not called and the
  stored advice is applied. A discarded run followed by a new run on the same
  model also reuses its advice, from `lastAdvice`. With `Reread`, or with a changed hash, it is
  called.
- Empty card: hardware plan only, with the note.
- The plan is called with the card-applied config: a fake `Plan` that records
  its argument sees the speculative config and the env block.
- `Config("all", nil)` and `Config("narrow", nil)` differ only in the hardware
  fields and do not call `Plan` again; `Config` with a row unticked omits it
  and calls `Plan` once more with the reduced config; with `Plan.Known` false
  the base's hardware fields survive.
- `AdviceFrom` is `"helper"`, `"previous"` or `""` in the three advice
  cases.
- `Profile` carries the source, the record and one note per ticked row.

`autoconfig_run_test.go`, on a test server with `fakevllm` and an `httptest`
engine for the helper:

- A run with a helper and a model serving: the progress lines are, in order,
  reading the card, stopping the model, starting the helper, asking it,
  stopping the helper, restarting the model, checking the answer, and checking
  what fits; afterwards the served model is
  the process manager's model again and the result is ready.
- A second `startAutoconfig` during a run is refused; a run for a draft and
  for the helper itself are refused.
- A run while a benchmark is active completes without touching the engine:
  with no earlier advice the result carries the busy reason as a note, and
  with reusable advice it is a full result.

## Commit

```
feat(autoconfig): run one autoconfigure pass and hold its result
```

## Rollback

Revert the commit. There is still no route that starts a run, so reverting
changes nothing an operator can see.

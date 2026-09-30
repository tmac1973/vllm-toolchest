# Phase 02 — The hardware planner

**Depends on:** phase 01 (`ContextClass`, `ProfileNote`) · **Enables:** the
feed's seeding (phase 03), the run (phase 09), the review's width choice
(phase 10) and refinement (phase 13)

## Goal

One function that answers, for a model on this machine: how many cards, how
much context, which KV dtype, and what memory utilization and sequence cap.
It is pure arithmetic over `Fit`, takes no helper and no network, and returns
both widths the operator chooses between -- all cards, and the narrowest that
holds the context asked for. Everything that later sets a hardware value calls
it, so there is one set of rules.

## Files touched

- `internal/models/plan.go` — new. `PlanInput`, `WidthPlan`, `FitPlan`,
  `PlanFit`.
- `internal/models/plan_test.go` — new.
- `internal/api/plan_input.go` — new. `(*Server).planInput`, which supplies
  the inventory, the defaults and the estimate closure.
- `internal/api/plan_input_test.go` — new.

## Steps

1. Define the types in `plan.go`:

   ```go
   type PlanDefaults struct {
       GPUMemoryUtilization float64 // cfg.GPUMemoryUtil, 0.90 when unset
       MaxNumSeqs           int     // cfg.MaxNumSeqs, 16 when unset
   }

   type PlanInput struct {
       Model     *Model
       Base      VLLMConfig                          // the config to build on
       Inventory GPUInventory
       Class     ContextClass
       Defaults  PlanDefaults
       CardKVDtype string                            // "" when the card names none
       FixedTP   int                                 // >0 plans that width only
       Estimate  func(c VLLMConfig) VRAMEstimate     // measured when it applies, else projected
   }

   type WidthPlan struct {
       TP                  int
       Config              VLLMConfig
       ContextTokens       int
       ContextReduced      bool    // less than the class asked for
       FullContextRequests int
       RequiredGB, RequiredHighGB, AvailableGB float64
       Measured            bool
       Notes               []ProfileNote // why each field has this width's value
   }

   type FitPlan struct {
       Known      bool
       Why        string      // when !Known
       All        WidthPlan
       Narrow     *WidthPlan  // nil when it would be the same width as All, or none reaches the target
       FirstGuess bool        // no applicable measurement behind the figures
       Notes      []ProfileNote // general: true whichever width is chosen
   }

   func PlanFit(in PlanInput) FitPlan
   ```

2. Target context: `in.Class.Tokens()`, capped at
   `Model.HFConfig.MaxPositionEmbeddings` when that is known. `ContextMax` is
   that maximum; when the maximum is unknown, `ContextMax` falls back to the
   long class's 131,072.

3. Candidate widths: `FixedTP` alone when set, otherwise every power of two up
   to `Inventory.Count`, as `Fit` enumerates them. "All cards" is the largest.

4. For one width and one KV dtype, build the candidate config from `Base` with
   `TensorParallelSize = tp`, `GPUMemoryUtilization = Defaults.GPUMemoryUtilization`,
   `MaxNumSeqs = Defaults.MaxNumSeqs`, `KVCacheDtype = dtype`,
   `MaxModelLen = target` and `KVCacheMemory = 0` (a pinned pool would defeat
   the engine's own sizing). When `FixedTP` is set the planner is refining a
   configuration that has run, and only the context may move: the candidate
   keeps `Base`'s utilization, sequence cap, KV dtype and `KVCacheMemory`,
   step 6 is skipped,
   and so the candidate differs from the live config in `MaxModelLen` alone --
   which is outside the measurement fingerprint, so a measurement taken on the
   live config still applies to it. Call `in.Estimate(candidate)`, then
   `Fit(est, candidate, in.Inventory)` and take the option for `tp`.

5. Affordable context at that width, on the **expected** figure, not the
   pessimistic one -- the pessimistic end is beyond the host for most large
   models and would never choose anything:

   ```
   nonKV  = option.RequiredGB - option.KVGB
   spare  = option.AvailableGB - nonKV
   tokens = spare * 2^30 / est.KVCachePerTokenB
   tokens = tokens * margin          // 0.85 projected, 0.97 measured
   tokens = floor(tokens / 1024) * 1024
   context = min(target, tokens)
   ```

   A width whose `context` comes out under 2,048 does not hold the model and is
   skipped. `FullContextRequests` is `max(1, floor(tokens / context))`, where
   `tokens` is the affordable figure after the margin and before it is limited
   to the target. `est.KVCachePerTokenB == 0` (shape unknown) sets the context
   to `min(target, Base.MaxModelLen)` when the base has one and 8,192
   otherwise, sets `FullContextRequests` to 0 -- which the review shows as
   "unknown" -- and adds a note that the context could not be sized. In that
   case a width holds the model when `option.RequiredGB - option.KVGB` is at
   most `option.AvailableGB`, the weights and overhead without any cache.

6. KV dtype. When `CardKVDtype` is set, use it at every width. Otherwise plan
   with `auto`; if the all-cards context comes out below the target, plan
   again with `fp8` and keep whichever gives more context, adding a note when
   fp8 was chosen ("the context asked for does not fit without it"). The dtype
   chosen for the all-cards width is then used for every width, including the
   narrow one, so the two options differ only in how many cards they use.

7. `All` is the plan at the largest width that holds the model. `Narrow` is the
   smallest width whose context reaches the target; nil when that is the same
   width as `All`, or when no width reaches the target, or when `FixedTP` is
   set. If no width holds the model at all, return `Known: false` with
   `Why: "does not fit on this machine at any width"`.

8. `Known` is also false, with the reason, when the inventory is not known or
   `est.Unknown` is set. In that case the caller proposes no hardware change.

9. `FirstGuess` is true when the estimate behind `All` has
   `Source != SourceMeasured`.

10. Notes, each with `Origin: "this machine"`. On each `WidthPlan`: one per
    field set (`tensor_parallel_size`, `max_model_len`,
    `gpu_memory_utilization`, `max_num_seqs`, `kv_cache_dtype`), saying in one
    sentence what it is and why for that width, and one when that width's
    context was reduced, naming the target and what fits. On `FitPlan`: the
    note that fp8 was chosen (it applies to both widths) and, when
    `FirstGuess` is true, that the figures are an estimate a real start will
    refine.

11. `internal/api/plan_input.go`: `func (s *Server) planInput(m *models.Model, base models.VLLMConfig, class models.ContextClass, cardKVDtype string) models.PlanInput`.
    `PlanInput.Base` is `base`, which every caller states: autoconfigure
    passes the config the card produced, refinement the live config, and
    seeding the model's default config.
    The inventory is `s.gpuInventory()` with `FreePerCardGB` cleared, because
    the question is what the model needs when it is the thing running. The
    estimate closure copies the model, sets the candidate config, and returns
    `MeasuredEstimate(copy, s.engineIdentity())` when that applies and
    `EstimateVRAM(copy, s.configuredEnvPairs(copy))` with
    `Source = SourceProjected` otherwise -- the same rule `effectiveVRAM`
    uses.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

## Test plan

Table tests in `plan_test.go`, with the fixtures already in the package
(`measuredModel()`, the 27B shape in `vram_test.go`):

- 27B projected, four 31.86 GiB cards, `ContextMax`: `All.TP == 4`, context is
  the model's 262,144, `Narrow` is non-nil at TP=2, `FirstGuess` is true.
- The same with a measurement that applies at TP=4: `All.Measured`, margin
  0.97, `FirstGuess` false, and the context equals what the measured pool
  holds to within one 1,024-token step.
- A 23 GB checkpoint on one 24 GiB card, `ContextLong`: `All.TP == 1`,
  `Narrow == nil`, context below 131,072 with `ContextReduced`.
- A class target the machine cannot reach with `auto` but can with `fp8`: the
  plan chooses fp8 and says so; with `CardKVDtype: "fp8_e5m2"` it uses that and
  adds no note.
- `FixedTP: 2` on a four-card host returns only that width and no `Narrow`.
- Unknown inventory, and an estimate with `Unknown` set: `Known` is false and
  `Why` is non-empty.
- A checkpoint larger than every width: `Known` false, "does not fit".
- Without `FixedTP`, every returned config has `KVCacheMemory == 0` and the
  default `MaxNumSeqs`. With `FixedTP`, the returned config equals `Base` in
  every field but `MaxModelLen`.
- `plan_input_test.go`: with the golden server's two-card inventory, the
  closure returns a projected estimate for a model with no measurement and a
  measured one for a model whose fingerprint matches.

## Commit

```
feat(models): plan a model's width, context and KV dtype for this machine
```

## Rollback

Revert the commit. Nothing calls the planner until phase 09, and phase 03 only
changes documents, so it is safe to leave applied or to remove.

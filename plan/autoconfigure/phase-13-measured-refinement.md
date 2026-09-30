# Phase 13 — Refine from a real start

**Depends on:** phase 01 (the record on the profile), phase 02 (`PlanFit` with
`FixedTP`), phase 10 (the `autoconfig_hardware_table` template), phase 12
(`startFixFor`, and the notice position it shares) · **Enables:** acceptance (phase 14)

## Goal

Before a model has run, its hardware settings are a first guess made from a
wide estimate, with a margin taken off. After one successful start the engine
has reported what the model actually consumed, and the project already stores
that. This phase uses it: when the measured figures would give a different
context than the one configured, the model's config panel says "Measured on a
real start: review refined settings". The button opens the same hardware table
the autoconfigure review shows, and one more click applies it. The settings
that came from the card are left exactly as they are, so no card is read and no
engine is touched. Of the hardware settings only the context can move: the
width, the KV dtype and the batch size are part of what the measurement
describes, and changing them would retire it.

This is the step that reverses the old rule that a measurement never leads to
a config change. The rule that survives is that nothing changes without an
apply.

## Files touched

- `internal/api/refine.go` — new. `refinement`, `(*Server).refinementFor`,
  `handleRefinementReview`, `handleApplyRefinement`.
- `web/templates/partials/autoconfig_refine.html` — new. The refinement
  review.
- `internal/api/refine_test.go` — new.
- `internal/api/models_config_view.go` — `modelConfigView.Refinement`.
- `internal/api/server.go` — the route.
- `web/templates/partials/model_config.html` — the notice, in the same place
  as phase 12's.
- Golden recordings for the config panel with a refinement.

## Steps

1. When a refinement exists. `func (s *Server) refinementFor(m *models.Model) *refinement`
   returns nil unless all of these hold:
   - the model's active profile is Autoconfig (modified or not) and that
     profile has an `AutoconfigRecord` -- the record is where the context
     class the operator asked for is kept, and without it there is no target
     to refine towards;
   - `models.MeasuredEstimate(m, s.engineIdentity())` applies to the live
     config, which means the measurement was taken at this width, KV dtype,
     batch size, flags and environment, on this engine;
   - phase 12's `startFixFor(m)` is nil -- a failed start takes precedence.

2. What it proposes. Call `models.PlanFit` with
   `s.planInput(m, m.VLLMConfig, record.Class, m.VLLMConfig.KVCacheDtype)` and
   `FixedTP: m.VLLMConfig.TensorParallelSize`. With `FixedTP` set the planner
   keeps the live config's utilization, sequence cap and KV dtype (phase 02,
   step 4), so the only candidate differs from the live config in context
   alone and the measurement applies to it. The estimate closure returns
   the measured estimate for this configuration, so the plan uses the 0.97
   margin and the engine's own bytes per token and consumed memory. The width
   is fixed because the measurement says nothing reliable about another one,
   and changing it would retire the measurement anyway. Only the context is
   compared:

   ```go
   type refinement struct {
       MeasuredWhen        string // "30 Sep 15:12"
       CurrentContext      int
       ProposedContext     int
       ClassLabel          string // "Maximum", "Long", …
       FullContextRequests int
       Direction           string // "more" or "less"
   }
   ```

   Return nil when `plan.Known` is false, or when the proposed context is
   within 1,024 tokens of the current one. A proposal never exceeds the class
   target: a model configured for Medium is not offered 262,144 because it
   happens to fit.

3. Both directions are offered. More context when the first guess was
   cautious. Less when the measured pool, with the margin taken off, holds
   fewer tokens than are configured: the start succeeded, but only just, and a
   request near the full length would be refused.

4. The notice in `model_config.html`, rendered when `.Refinement` is set and
   phase 12's `.StartFix` is not: "Measured on a real start, 30 Sep 15:12.
   Context can be N tokens (now M)." and a button **Review refined settings**,
   which `hx-get`s `/api/models/autoconfig/refine?id=` into a slot below the
   notice.

5. `GET /api/models/autoconfig/refine?id=` renders `autoconfig_refine`: the
   line "The first setting was an estimate made before this model had run.
   These figures come from its start on <date>.", then
   `autoconfig_hardware_table` with the live config as current and the live
   config with the proposed context as proposed, then "Room for R full-context
   requests.", then **Apply** and **Close**. When `refinementFor` is nil it
   renders "Nothing to refine: the configured context already matches the
   measurement."

6. `POST /api/models/autoconfig/refine?id=`. It recomputes `refinementFor(m)`
   -- the form carries no values -- and refuses when it is nil or the registry
   is read-only. It sets `max_model_len` through `applyToConfig`, then
   `UpdateConfig`, the estimate recompute and `Register`. Context length is
   deliberately outside the measurement fingerprint, so the measurement still
   applies afterwards and the KV figure re-scales exactly.

7. Update the profile by phase 12's rule, step 6. When the Autoconfig profile
   is active and unmodified: read it, set its `Config` to the new live config,
   clear `Autoconfig.FirstGuess`, append the note "Context set to N from a real
   start measured on <date>." with origin "this machine", and write it back
   with `SaveProfileFrom`. When it is active but modified by hand, only the
   live config changes and the profile is left as it was saved, for the same
   reason phase 12 gives: copying the change into it would also copy hand
   edits the operator has not saved there. Step 1 still offers the refinement
   in that case, because the context it corrects is the live one.

8. The response re-renders the config panel with the banner "Applied. It takes
   effect the next time this model starts." After the change the plan and the
   live config agree, so `refinementFor` returns nil and the notice is gone
   without any dismissed-state being stored.

9. If the operator would rather keep the current context, they do nothing: the
   notice stays, which is accurate. There is no dismiss button and no stored
   "ignored" flag; a notice that can be silenced is one more piece of state
   that outlives the machine it was written for.

10. As with a failed start in phase 12, nothing is added to the model's card.
    A refinement is an improvement on offer and is found in Configure, where
    the measured figures are already shown.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
go test ./internal/api -run 'TestGoldenFragments|TestGoldenPartials' -update
```

## Test plan

On a test server with the `measuredModel()` fixture, an Autoconfig profile
with a record, and a four-card inventory:

- Live context 131,072, class Maximum, measurement showing room for far more:
  the refinement proposes the model's 262,144 and says "more". Applying sets
  the live config and the profile, clears `FirstGuess`, leaves the measurement
  in place, and the next render has no notice.
- Live context 262,144 with a measurement whose pool holds 200,000 tokens:
  the refinement proposes a lower context and says "less".
- Class Medium, live context 32,768, room for ten times that: no refinement.
- Proposed within 1,024 tokens of current: no refinement.
- Active profile is not Autoconfig, or the Autoconfig profile has no record:
  no refinement.
- The measurement does not apply (the width was changed since): no
  refinement.
- A failed start and a measurement both present: the start fix is shown and
  the refinement is not.
- Applying on a read-only registry refuses and changes nothing.
- A stale page posting when nothing is on offer gets "nothing to apply".
- Goldens for the config panel with a refinement notice, and for
  `autoconfig_refine` with a proposal and with nothing to refine.

Manual, on compute, as part of phase 14: after the first autoconfigured start
of the 27B, the notice appears in Configure with a context no larger than the
engine's reported `GPU KV cache size` in the log.

## Commit

```
feat(autoconfig): offer refined settings on the model after a measured start
```

## Rollback

Revert the commit. Measurements continue to be recorded and to correct the
VRAM estimate shown; they simply stop producing a proposal.

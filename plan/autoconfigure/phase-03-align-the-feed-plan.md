# Phase 03 — Bring the recommend-feed plan into line

**Depends on:** phase 01 (`ContextMax`), phase 02 (the planner's names, which
the rewritten steps cite)
· **Enables:** building phases 15-19 later without inheriting a contradiction

## Goal

Two planning documents say the opposite of what this project now intends.
`plan/recommend-models-overview.md` lists "no launch-config optimizer" and "no
automatic re-configuration after a run" among its non-goals, and
`plan/phase-19-fit-handoff.md` has a section arguing that measurement must
never touch a config. Neither has been built, so this phase is documents only:
it amends them so that whoever builds the feed seeds a model by calling the
planner from phase 02, and reads a plan that agrees with autoconfigure.

## Files touched

- `plan/recommend-models-overview.md` — amend Goals, Non-goals and the primary
  flow's step 7.
- `plan/phase-19-fit-handoff.md` — rewrite the Goal, the "What seeding is not"
  section, the files list and the steps that compute the seed.
- `plan/README.md` — add the autoconfigure folder to the Active list.
- `plan/todo.md` — update "Direction, decided 2026-09-30".

## Steps

1. `recommend-models-overview.md`, Non-goals:
   - Replace the **No launch-config optimizer** bullet with: seeding on
     download is the whole of the *feed's* handoff; configuring a model
     properly is autoconfigure's job, defined in
     `plan/autoconfigure/overview.md`.
   - Replace the **No automatic re-configuration after a run** bullet with:
     nothing rewrites `VLLMConfig` without an explicit apply. A measurement
     may produce a *proposal*, which autoconfigure's refinement shows on a
     model that has been autoconfigured, and the operator applies. A model
     that was only seeded has no Autoconfig profile, so it is offered
     refinement once it has been through Autoconfigure. Keep the sentence that measurement
     corrects the VRAM estimate.
   - Leave **No config profile is created on download** as it is: seeding
     writes the live config of a model that has none yet, and the Autoconfig
     profile is created only by an autoconfigure run.

2. Same file, Goals: change the "Fit output survives the download" bullet to
   say the seeded values are whatever `models.PlanFit` returns for the model on
   this machine with `ContextMax` and the all-cards width, recorded as seeded.
   Primary flow step 7: replace "nothing later rewrites them" with "they stay
   as written until the operator changes them or applies an autoconfigure
   proposal".

3. `phase-19-fit-handoff.md`:
   - Header: add `plan/autoconfigure` phase 02 to **Depends on**.
   - Goal: the seed is the five hardware fields of
     `models.PlanFit(...).All.Config` -- `TensorParallelSize`, `MaxModelLen`,
     `KVCacheDtype`, `GPUMemoryUtilization`, `MaxNumSeqs` -- copied onto the new
     model's default config, with a pinned `KVCacheMemory` cleared, exactly as
     autoconfigure writes them. State why: seeding and autoconfigure's hardware
     half follow one set of rules because one function computes both. Their
     inputs differ -- seeding reads no card and always asks for the maximum
     context -- so their results agree only when the inputs do: an
     autoconfigure run with Maximum on a card that contributes no rows gives
     the same values.
   - Replace "What seeding is not" with "What seeding is, and what comes
     after": seeding is the hardware half of autoconfigure with no card read
     and no review, applied once to a model that has never been configured.
     `ConfigSource` remains a provenance marker. Seeding creates no profile,
     so refinement from a measured start (autoconfigure's phase 13) does not
     apply to a seeded model until the operator runs Autoconfigure on it;
     after that, a correction is proposed and applied by the operator, never
     silently.
   - Files touched: `internal/recommend/seed.go` keeps only `SeedConfig`,
     which copies the five fields from a `models.VLLMConfig` it is given onto
     the new model's config and sets `ConfigSource`; remove any text about it
     computing a width or context itself. The computation happens in
     `internal/api/hf_download.go`, which phase 19 already lists: at download
     completion it calls `s.planInput(m, m.VLLMConfig, models.ContextMax, "")` -- a method
     on the server, which `internal/recommend` cannot call -- then
     `models.PlanFit`, and passes `plan.All.Config` to `SeedConfig`.
   - Steps: wherever a step takes the width and context "from the ranking",
     replace with the call above, made against the downloaded model's own
     `config.json`, now on disk. If `FitPlan.Known` is false the model keeps
     the default config and `ConfigSource` stays empty.
   - Test plan: add "the seed equals `PlanFit` called directly with the same
     model, `ContextMax`, no card KV dtype and the default config, in all five
     fields".

4. `plan/README.md`, Active list: add an entry for
   [`autoconfigure/`](autoconfigure/overview.md) describing it in two lines --
   one action that configures a model from its card and this machine, with a
   helper model, a review, and refinement from a measured start -- and noting
   it has its own phase numbering. Amend the phase-19 entry's last sentence,
   which still says measurement does not supersede seeding.

5. `plan/todo.md`, "Direction, decided 2026-09-30": replace the sentence saying
   the two documents "still say otherwise and need amending" with a statement
   that they were amended in this phase, and replace "Next is the autoconfigure
   phase document" with a pointer to `plan/autoconfigure/`.

## Build gate

Documents only, so the gate is that they agree with each other:

```
grep -ni "no automatic re-configuration\|no launch-config optimizer\|what seeding is not" plan/recommend-models-overview.md plan/phase-19-fit-handoff.md
```

must print nothing, and `go test ./...` still passes (nothing in it reads
these files, which confirms no code was touched).

## Test plan

- Read `recommend-models-overview.md` and `phase-19-fit-handoff.md` top to
  bottom after the edit: no sentence says a measurement cannot lead to a config
  change, and none implies a change happens without an apply.
- Every identifier the amended phase 19 cites (`models.PlanFit`, `FitPlan`,
  `ContextMax`, `planInput`) exists in the code after phase 02.
- `phase-15` to `phase-18` are unchanged: `git diff --stat` lists only the
  four files above.
- The links added to `plan/README.md` resolve.

## Commit

```
docs(plan): have the recommend feed seed through the autoconfigure planner
```

## Rollback

Revert the commit. No code depends on these documents.

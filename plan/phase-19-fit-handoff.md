# Phase 19 — Carry the fit into the downloaded model

**Depends on:** phase 17 (the candidate pool), phase 18 (the Download button
on a card), `plan/autoconfigure` phase 02 (`models.PlanFit` and
`(*Server).planInput`) · **Enables:** nothing further. This closes the
feature.

## Goal

When a model is downloaded from the feed, seed its `models.json` entry with
hardware settings for this machine, and record that they were seeded rather
than chosen. The user would otherwise go to the Models page and set by hand
numbers this tool can work out.

The seed is the five hardware fields of `models.PlanFit(...).All.Config` --
`TensorParallelSize`, `MaxModelLen`, `KVCacheDtype`, `GPUMemoryUtilization`,
`MaxNumSeqs` -- copied onto the new model's default config, with a pinned
`KVCacheMemory` cleared, exactly as autoconfigure writes them. Seeding and
autoconfigure's hardware half follow one set of rules because one function
computes both. Their inputs differ -- seeding reads no card and always asks
for the maximum context -- so their results agree only when the inputs do: an
autoconfigure run with Maximum on a card that contributes no rows gives the
same values.

*Amended 2026-09-30.* This phase first took the width and context from the
ranking and rounded the context to a power of two. It now calls the planner,
so that there is one set of hardware rules in the project rather than two.

**What seeding is, and what comes after.** Seeding is the hardware half of
autoconfigure with no card read and no review, applied once to a model that
has never been configured. `ConfigSource` remains a provenance marker, not a
precedence rule: the seeded values are ordinary configuration that persists
until a person changes it. Seeding creates no profile, so refinement from a
measured start (autoconfigure's phase 13) does not apply to a seeded model
until the operator runs Autoconfigure on it; after that, a correction is
proposed and applied by the operator, never silently.

## Files touched

- `internal/models/registry.go` — add `ConfigSource string` to the model
  entry. The schema version is deliberately **not** bumped; see step 1.
- `internal/recommend/seed.go` — new. `SeedConfig`, which copies the five
  hardware fields from a `models.VLLMConfig` it is given onto the new model's
  config, clears a pinned pool and sets `ConfigSource`. It computes nothing
  itself.
- `internal/recommend/seed_test.go` — new.
- `internal/recommend/engine.go` — implement `SeedFor`, declared in phase 17
  step 2, as a lookup of whether a repository id is a candidate in the current
  pool.
- `internal/api/hf_download.go` — at download completion, for a model newly
  registered from the feed, call `s.planInput(m, m.VLLMConfig,
  models.ContextMax, "")` -- a method on the server, which `internal/recommend`
  cannot call -- then `models.PlanFit`, and pass `plan.All.Config` to
  `SeedConfig`.
- `web/templates/partials/model_config.html` — show the provenance marker
  beside a seeded config.
- `internal/api/recommend_seed_test.go` — new. End-to-end through
  registration.

No template in phase 18 changes: the Download button already posts the
repository id, which is the only key this phase needs.

## Steps

1. Add to the registry entry:

   ```go
   ConfigSource string `json:"config_source,omitempty"`
   ```

   Two values are written: empty, meaning the user set it or the image
   defaulted it, and `"seeded"`, meaning this feature chose it. Empty is the
   zero value and the existing behaviour, so no data migration is needed.

   **Leave `schemaVersion` (`internal/models/registry.go:233`) at 3.** The
   gate refuses a file *newer* than the build while accepting an older one, so
   bumping to 4 would make every `models.json` written by this phase
   unreadable the moment the phase was reverted — turning an additive,
   `omitempty` field into a one-way door. An optional field that older builds
   ignore and newer builds populate needs no version change; the version is
   for changes that would break an older reader, and this is not one.

   `"measured"` is deliberately **not** a value. Nothing measures a
   configuration; measurement produces an estimate, and that lives in
   `Model.Measured` where it already does.

2. Write `SeedConfig(m *models.Model, planned models.VLLMConfig)` in
   `internal/recommend`. It sets exactly the five hardware fields from
   `planned`, sets `KVCacheMemory` to 0, and sets `ConfigSource = "seeded"`.
   It does not work out a width, a context or a dtype: that is
   `models.PlanFit`'s, called by the completion path in step 6 against the
   downloaded model's own `config.json`, now on disk.

3. Every other field of the model's config stays as registration defaulted
   it. This matters: `variants/radiance.conf` is explicit that knobs default
   to unset so values track upstream rather than being pinned here, and
   seeding a field would pin it.

4. Key the seed by repository id against the engine's current pool, via
   `SeedFor(modelID)`. There is no seed token and nothing rides on the
   request: a download is asynchronous and can outlive the request that
   started it, so the seed must be looked up at completion time from a pool
   that is still addressable by the id the download already carries.

5. If the pool has been refreshed or has changed profile by completion time
   and the candidate is gone, register the model with no seed, leave
   `ConfigSource` empty, and log at `Info`. The same when `FitPlan.Known` is
   false: the model keeps its default config. A missing seed is a lesser
   outcome, not an error.

6. Apply the seed in the completion path where the downloader already
   registers the model, setting `ConfigSource = "seeded"`. Apply it **only**
   when registering a model not already in the registry. A re-download of an
   existing model must never overwrite a config the user has since tuned.

7. Show the provenance marker on the Models page beside a seeded config: a
   short muted note saying these values were suggested when the model was
   downloaded and can be changed freely. Do not imply anything will overwrite
   them: the only thing that may change them is an autoconfigure proposal the
   operator applies.

## Build gate

```
gofmt -l ./internal ./cmd
go build ./...
go vet ./internal/...
go test ./...
```

## Test plan

- **Unit, seed equals the planner.** The seed equals `PlanFit` called
  directly with the same model, `ContextMax`, no card KV dtype and the default
  config, in all five fields, and `KVCacheMemory` is 0.
- **Unit, nothing else pinned.** Assert every other field of the model's
  config is what registration defaulted. This is the test that stops the seed
  quietly growing into an optimizer; configuring the rest is autoconfigure's.
- **Unit, unknown fit.** When `PlanFit` returns `Known: false` the model keeps
  its default config and an empty `ConfigSource`.
- **Unit, re-download.** Registering a model already present leaves its config
  and its `ConfigSource` untouched.
- **Unit, seed expired.** A completion whose candidate is no longer in the pool
  registers the model with a zero config, an empty `ConfigSource`, and no
  error.
- **Unit, schema.** A `models.json` containing `config_source` loads under a
  build without the field (simulating a revert) and is not refused, and
  entries without `config_source` read as empty under this build.
- **Integration.** Download from the feed against a stub Hub and assert the
  resulting entry has the five planned fields and `ConfigSource: "seeded"`.
- **Manual, the overview's criterion.** Download a model from the feed, note
  the seeded tensor-parallel width, start it, and confirm two things: the
  width the engine reports running at matches the width the feed stated, and
  `VLLMConfig` is unchanged by the run while the VRAM estimate shown against
  the model switches to the measured source. Nothing is proposed for it until
  it has been autoconfigured, since it has no Autoconfig profile. The first half is the check that
  the fit arithmetic was right; the second is the check that this phase
  changed nothing it should not.

## Commit

```
feat(recommend): seed a downloaded model's config from its fit
```

## Rollback

Revert the commit. `ConfigSource` disappears from newly written entries and is
ignored on existing ones, so a `models.json` written while this phase was
applied still loads — the field is additive, `omitempty`, and simply unread
after the revert. This is the reason step 1 leaves the schema version alone:
had it been bumped, every file written under this phase would be refused by
the reverted build as newer than itself. Models already seeded keep their
values, which are valid configurations regardless of where they came from;
they simply lose the note saying they were suggested. Safe to leave partially
applied up to step 6, since nothing reads `ConfigSource` until step 7 renders
it.

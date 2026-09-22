# Phase 19 — Carry the fit into the downloaded model

**Depends on:** phase 17 (the fit output), phase 18 (the Download button on a
card) · **Enables:** nothing further. This closes the feature.

## Goal

When a model is downloaded from the feed, seed its `models.json` entry with
the tensor-parallel width and context length that ranked it, and record that
those values were seeded rather than chosen. Ranking already computes them —
`Fit` returns `TPOption`s with `Fits`, `SpareGB` and `AvailableGB`, and phase
17 stores the recommended width and the affordable context — and today that
work would be discarded at the exact moment it is most useful. The user would
otherwise go to the Models page and set by hand a number this tool already
knows.

**What seeding is not.** An earlier draft of this plan assumed the seeded
values would be superseded by measurement on the first real start. They are
not, and the code is clear about it: `recordMeasurement`
(`internal/api/service.go:275`) *reads* `m.VLLMConfig.TensorParallelSize` to
record what a run was configured with, and `MeasuredEstimate`
(`internal/models/measured.go:88`) uses the stored `RunMeasurement` to produce
a better **VRAM estimate**. Nothing anywhere writes `VLLMConfig` from a
measurement, and nothing should: the configuration is the user's, and silently
rewriting it would make the Models page lie about what will be launched.

So the seeded values are ordinary configuration that persists until a person
changes it. `ConfigSource` exists to say where they came from, so the Models
page can show it and the user can tell a value the tool picked from one they
picked themselves. That is a provenance marker, not a precedence rule.

## Files touched

- `internal/models/registry.go` — add `ConfigSource string` to the model
  entry. The schema version is deliberately **not** bumped; see step 1.
- `internal/recommend/seed.go` — new. `SeedConfig`.
- `internal/recommend/seed_test.go` — new.
- `internal/recommend/engine.go` — implement `SeedFor`, declared in phase 17
  step 2, against the current pool.
- `internal/api/hf_download.go` — apply the seed when a download completes and
  a new model is registered.
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

2. Write `SeedConfig(c Candidate, p Profile) (models.VLLMConfig, bool)` in
   `internal/recommend`. The `Profile` is passed rather than reached for,
   because the KV dtype depends on the hardware and a `Candidate` carries only
   what the Hub and the fit produced. `SeedFor` supplies the pool's own
   profile, so the seed is always derived from the hardware the candidate was
   ranked against. It returns `false` for anything that is not Verified —
   never seed from a fit that was not computed. Otherwise it sets exactly
   three fields:

   - `TensorParallelSize` = the recommended width phase 17 chose.
   - `MaxModelLen` = the `affordableTokens` phase 17 already stored on the
     candidate, rounded **down** to the nearest power of two. Recomputing it
     here would be a second copy of the arithmetic; read the stored field.
   - `KVCacheDtype` = `"fp8"` only when `p.Accelerated` contains `fp8`,
     otherwise left empty.

3. Every other field of the returned `VLLMConfig` stays at its zero value so
   the image default stands. This matters: `variants/radiance.conf` is
   explicit that knobs default to unset so values track upstream rather than
   being pinned here, and seeding a field would pin it.

4. Key the seed by repository id against the engine's current pool, via
   `SeedFor(modelID)`. There is no seed token and nothing rides on the
   request: a download is asynchronous and can outlive the request that
   started it, so the seed must be looked up at completion time from a pool
   that is still addressable by the id the download already carries.

5. If the pool has been refreshed or has changed profile by completion time
   and the candidate is gone, register the model with no seed, leave
   `ConfigSource` empty, and log at `Info`. A missing seed is a lesser
   outcome, not an error.

6. Apply the seed in the completion path where the downloader already
   registers the model, setting `ConfigSource = "seeded"`. Apply it **only**
   when registering a model not already in the registry. A re-download of an
   existing model must never overwrite a config the user has since tuned.

7. Show the provenance marker on the Models page beside a seeded config: a
   short muted note saying these values were suggested when the model was
   downloaded and can be changed freely. Do not imply anything will overwrite
   them, because nothing will.

## Build gate

```
gofmt -l ./internal ./cmd
go build ./...
go vet ./internal/...
go test ./...
```

## Test plan

- **Unit, seed derivation.** A verified candidate with recommended width 2 and
  `affordableTokens` 140000 produces `TensorParallelSize: 2` and
  `MaxModelLen: 131072`.
- **Unit, power-of-two rounding.** 140000 → 131072; 131072 → 131072 (already a
  power of two, not halved); 4000 → 2048.
- **Unit, kv dtype.** `fp8` is set when the profile lists it as accelerated and
  left empty when it does not.
- **Unit, unverified.** `SeedConfig` on an unverified candidate returns `false`
  and a zero `VLLMConfig`.
- **Unit, nothing else pinned.** Assert every other field of the returned
  `VLLMConfig` is its zero value. This is the test that stops the seed quietly
  growing into an optimizer, which the overview lists as a non-goal.
- **Unit, re-download.** Registering a model already present leaves its config
  and its `ConfigSource` untouched.
- **Unit, seed expired.** A completion whose candidate is no longer in the pool
  registers the model with a zero config, an empty `ConfigSource`, and no
  error.
- **Unit, schema.** A `models.json` containing `config_source` loads under a
  build without the field (simulating a revert) and is not refused, and
  entries without `config_source` read as empty under this build.
- **Integration.** Download from the feed against a stub Hub and assert the
  resulting entry has the three expected fields and `ConfigSource: "seeded"`.
- **Manual, the overview's criterion.** Download a model from the feed, note
  the seeded tensor-parallel width, start it, and confirm two things: the
  width the engine reports running at matches the width the feed stated, and
  `VLLMConfig` is unchanged by the run while the VRAM estimate shown against
  the model switches to the measured source. The first half is the check that
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

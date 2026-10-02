# Phase 01 — Profiles that carry where they came from

**Depends on:** nothing · **Enables:** saving an autoconfigure result (phase
10), explaining it afterwards, and the two notices that revise it (phases 12
and 13)

## Goal

A config profile today is a name, a `VLLMConfig`, the time it was saved and
the image it was saved on (`Variant`, `VariantVersion`, set through the
existing `ProfileMeta`), and the only way to make one is to snapshot the
model's live config. Autoconfigure needs to save a
config that is *not* live yet, and to keep with it what it was built from: why
each setting was chosen, which context size was asked for, which width was
picked, and what the card said. This phase adds that to the profile record and
adds the one registry method that saves an arbitrary config. Nothing visible
changes.

## Files touched

- `internal/models/profiles.go` — add `Source`, `Notes` and `Autoconfig` to
  `ConfigProfile`; add `ProfileNote`, `AutoconfigRecord`, the source constants
  and `Registry.SaveProfileFrom`.
- `internal/models/context_class.go` — new. `ContextClass` and its four
  values, here rather than with the planner because the profile record stores
  one.
- `internal/models/profiles_test.go` — cover the new method and fields.
- `internal/models/schema_test.go` — a file written without the new fields
  still loads; one written with them round-trips.

## Steps

1. Create `internal/models/context_class.go`:

   ```go
   type ContextClass string

   const (
       ContextShort  ContextClass = "short"  // 8,192 tokens
       ContextMedium ContextClass = "medium" // 32,768
       ContextLong   ContextClass = "long"   // 131,072
       ContextMax    ContextClass = "max"    // the model's own maximum
   )

   // Tokens is the target for a class, or 0 for ContextMax.
   func (c ContextClass) Tokens() int
   // ParseContextClass returns ContextMedium for anything unrecognised.
   func ParseContextClass(s string) ContextClass
   ```

2. In `profiles.go` add:

   ```go
   const (
       ProfileSourceManual     = ""           // saved from the panel
       ProfileSourceAutoconfig = "autoconfig"
   )

   // AutoconfigProfileName is the one profile autoconfigure writes.
   const AutoconfigProfileName = "Autoconfig"

   type ProfileNote struct {
       Field  string `json:"field,omitempty"` // VLLMConfig JSON key, "" for a general note
       Reason string `json:"reason"`
       Origin string `json:"origin"` // "model card", "this machine", "default" or "helper summary"
   }

   type AutoconfigRecord struct {
       At          time.Time       `json:"at"`
       Class       ContextClass    `json:"class"`
       Width       string          `json:"width"` // "all" or "narrow"
       FirstGuess  bool            `json:"first_guess,omitempty"`
       CardSources []string        `json:"card_sources,omitempty"`
       CardHash    string          `json:"card_hash,omitempty"`
       Advice      json.RawMessage `json:"advice,omitempty"`
   }
   ```

   and to `ConfigProfile`: `Source string` (`json:"source,omitempty"`),
   `Notes []ProfileNote` (`json:"notes,omitempty"`) and
   `Autoconfig *AutoconfigRecord` (`json:"autoconfig,omitempty"`).

3. Confirm `Registry.ActiveProfile` still compiles and still compares only
   `profiles[i].Config != m.VLLMConfig`. The new fields hold a slice and a
   pointer, which is fine on `ConfigProfile`; `VLLMConfig` itself must stay
   all-scalar and is not touched.

4. Add `func (r *Registry) SaveProfileFrom(modelID, name string, p ConfigProfile) (replaced bool, err error)`.
   It normalises and validates the name exactly as `SaveProfile` does, calls
   `writableLocked()` first, requires the model to exist, overwrites
   `p.ModelID`, `p.Name` and `p.SavedAt` with the real values, replaces an
   existing profile of that name or inserts in sorted position, and saves. It
   does **not** set `m.ActiveProfile` and does not touch `m.VLLMConfig`:
   saving a proposal must not change what will be launched. `ApplyProfile` is
   unchanged and is what makes it live.

5. `schemaVersion` stays at 3. The precedent is `VLLMConfig.Env` and
   `Model.Measured`, both added as omitempty fields after version 3 without a
   bump; a bump is for a new envelope key. State the consequence in the doc
   comment on `AutoconfigRecord`: a build older than this one that rewrites
   `models.json` drops these fields, so an Autoconfig profile survives as a
   plain profile with its config intact and its explanation gone.

6. Leave `SaveProfile` untouched: a profile saved from the panel keeps an empty
   `Source`, no notes and no record. If the name saved over is "Autoconfig",
   the record is dropped with it, which is correct -- the config is no longer
   what autoconfigure proposed.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

## Test plan

- `SaveProfileFrom` on a model whose live config differs: the profile holds
  the given config, the live config is unchanged, `ActiveProfile` still
  reports the previous name.
- `SaveProfileFrom` then `ApplyProfile`: the live config equals the profile's
  and `ActiveProfile` reports it unmodified.
- Saving twice under "Autoconfig" reports `replaced` and leaves one profile.
- A read-only registry refuses with the same error `SaveProfile` gives, and
  nothing in memory changes.
- An unknown model and an empty name are refused.
- Round trip through `models.json`: notes, record and raw advice come back
  byte-identical; a version-3 file with no such fields loads with them zero.
- `ParseContextClass("")` and `ParseContextClass("bogus")` are medium;
  `ContextMax.Tokens()` is 0.

## Commit

```
feat(profiles): let a profile carry where it came from, and save one that is not live
```

## Rollback

Revert the commit. Profiles written meanwhile keep loading: the extra JSON
fields are ignored by the older code and dropped the next time it rewrites
the file. Safe to leave applied on its own, since nothing calls the new method
until phase 10.

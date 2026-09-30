# Phase 10 — The button, the dialog and the review

**Depends on:** phase 01 (`SaveProfileFrom`), phase 02 (`FitPlan`), phase 06
(`engineBusy`), phase 07 (`helperModel`, `HelperFits`, the helper download
route), phase 08 (`Row`), phase 09 (the run and its state) · **Enables:** draft downloads from the review (phase 11), the two
notices (phases 12 and 13), acceptance (phase 14)

## Goal

Make autoconfigure something an operator can use: a button on each model
card, a start panel that asks for the context size and says what will be
interrupted, a progress line while it runs, and a review that shows every
proposed setting with its source and reason before anything is saved. Saving
writes the **Autoconfig** profile; *Save and apply* also makes it the live
config. This is the first phase with a visible result, and after it the
feature works end to end on a model that has never run.

## Files touched

- `internal/api/autoconfig.go` — new. The five handlers, the view structs.
- `internal/api/autoconfig_test.go` — new.
- `internal/api/server.go` — routes under `/api/models/autoconfig`.
- `internal/api/models.go` — `modelRow.Autoconfigurable`.
- `web/templates/partials/model_list.html` — the button and the
  `#autoconfig-{SafeID}` slot.
- `web/templates/partials/autoconfig_dialog.html`, `autoconfig_progress.html`,
  `autoconfig_review.html`, `autoconfig_hardware_table.html`,
  `autoconfig_message.html` — new.
- `web/templates/layout.html` — styles for the review table and the warning
  row.
- `internal/api/golden_test.go`, `partials_test.go` and
  `internal/api/testdata/golden/` — new recordings.

## Steps

1. Routes, all taking `?id=` as the other model routes do:
   - `GET  /api/models/autoconfig` — the start panel, or the progress or
     review of a run already held for this model.
   - `POST /api/models/autoconfig/start` — form: `context_class`, `reread`.
   - `GET  /api/models/autoconfig/status` — polled while running.
   - `POST /api/models/autoconfig/save` — form: `width`, `apply`, and one
     `row` value per ticked row key.
   - `POST /api/models/autoconfig/discard`.
   Every handler answers 200 with a partial, including refusals: htmx does not
   swap a non-2xx response, and a refusal nobody sees looks like a button that
   does nothing.

2. The button. `modelRow.Autoconfigurable` is true for a model that is neither
   a draft nor orphaned (the helper is already absent from the list). In
   `model_list.html`, inside `.model-card-buttons` after Configure:

   ```html
   <button class="secondary outline" hx-get="/api/models/autoconfig?id={{.ID}}"
           hx-target="#autoconfig-{{.SafeID}}" hx-swap="innerHTML">Autoconfigure</button>
   ```

   with the same before-request toggle Configure uses, and a third slot,
   `<div id="autoconfig-{{.SafeID}}" class="model-card-config"></div>`, beside
   `#config-…` and `#update-…`. The panel is inline under the card, like
   Configure and Update, not a modal: the review is a table that needs the
   width, and a run outlives any dialog the operator closes. The button grid is
   already two wide and sized for this label.

3. The start panel, `autoconfig_dialog`, shows:
   - four radios for the context size, medium checked, with the labels
     llama-toolchest uses -- "Short — about 8,000 tokens", "Medium — about
     32,000 tokens", "Long — about 128,000 tokens", and "Maximum — up to N
     tokens, as much as fits" with the model's own maximum;
   - what reading the card will cost. Every sentence that applies is shown,
     in this order:
     1. `engineBusy()`'s reason, with "The helper cannot be started now. An
        earlier reading of this card will be used if there is one; otherwise
        the card's command is used without its text.";
     2. the helper does not fit: `HelperFits`'s sentence, and "The card's
        command is still used.";
     3. the helper is not installed: "The helper model reads the card's text.
        It is about 8 GB." with a **Download the helper** button posting to
        phase 07's `/api/settings/helper/download` and targeting a slot in the
        dialog, into which that handler's `helper_model_panel` renders -- the
        same panel as in Settings, showing the download's progress -- and "Or
        start now: the card's command is still used.";
     4. a model is serving: "If the card needs reading, **X** will be stopped
        while it is read and restarted afterwards. A card read before and
        unchanged since is not read again.";
     5. when none of 1-4 applies: "If the card needs reading, the helper model
        will be started, used and stopped.";
   - when an earlier reading exists for this model -- in `lastAdvice` or on
     its Autoconfig profile -- a checkbox "Read the card again", unchecked:
     left unchecked, an unchanged card is not re-read and nothing is
     interrupted. Whether the card has changed is only known once it is
     fetched, so the checkbox is offered whenever there is a reading to reuse;
   - Start and Cancel. Start posts to `/start` and swaps the panel.

4. `/start` calls `startAutoconfig` and renders the progress partial; a refusal
   renders `autoconfig_message` with the reason.
   `autoconfig_progress` shows the model name and the current progress line
   and polls `/status` with `hx-trigger="every 2s"` and `hx-swap="outerHTML"`,
   which is how benchmark runs already poll. When the run is done `/status`
   returns the review, which has no trigger, so polling stops.

5. The review, `autoconfig_review`, from `Result`:
   - a heading line naming the sources: the repositories whose cards were
     read, and from `AdviceFrom` one of "the helper read the text", "the
     text's reading from the last run was reused", or "the text was not read";
   - when `Plan.FirstGuess`: "This model has not run here yet, so the hardware
     settings are a first guess. A real start will refine them.";
   - the width choice, as radios named `width`: "All N cards — room for R
     full-context requests", checked, and, when `Plan.Narrow` exists, "M cards
     — room for R requests, leaves N−M free". With no `Narrow` there is one
     line and no choice. With `Plan.Known` false, the reason replaces the
     block and the hardware settings are left as they are;
   - a **This machine** table: tensor-parallel size, context length, KV cache
     dtype, GPU memory utilization, max sequences -- current, proposed, why.
     It is its own template, `autoconfig_hardware_table`, taking the current
     config, the proposed config, the chosen width's notes, and the card's
     hardware notes (the result's notes whose field is one of the five), so
     that each line shows what the card used beside what was chosen; the
     refinement in phase 13 shows the same table. The narrow width's values
     and notes are both rendered into `data-` attributes and swapped together; When the current config pins a KV pool
     (`kv_cache_memory` non-zero) the table has a sixth line showing it
     cleared, since every proposal sets it to 0;
     The proposed column shows `Plan.All`; a small script swaps in the narrow
     plan's values when that radio is chosen, both sets being rendered into
     `data-` attributes;
   - a **From the model card** table, one line per row: a checkbox named `row`
     with the row's key, checked per `Ticked`; the setting, flag or variable;
     current; proposed; the source (the row's origin, "model card"); the
     reason; and the quote, which every row has, in a `<details>`. A row with a `Warning` gets the warning class and its text above
     the reason. An unticked row shows why it is unticked;
   - the notes, grouped as llama-toolchest's review groups them: notes about a
     field under that field, general notes in a list at the end, and notes
     with origin "helper summary" last, under "The helper's summary of the
     card, not checked";
   - three buttons: **Save and apply**, **Save as Autoconfig profile**, **Discard**.
   An error result renders `autoconfig_message` with the error and a Close
   button that discards it.

6. `/save`:
   1. Refuse when there is no finished result for this model ("Run it
      again."), and when `s.registry.ReadOnly()` is non-empty -- checked
      before anything is changed, as the profile handlers do.
   2. Build `ticked` from the posted `row` values. Rows absent from the form
      are unticked: an unchecked checkbox posts nothing.
   3. `profile, plan, err := result.Profile(width, ticked, models.ProfileMeta{…})`
      with the variant and version from `s.vllmEnv`. `Profile` builds its
      config with `Result.Config`, so an unticked row that changes the
      estimate re-plans the hardware; when the saved context or width differs
      from what the review showed, the message in step 6.7 adds "The fit was
      recalculated for the settings you kept: context N, M cards."
   4. Run `validateNamedBackends` on the profile's config, exactly as
      `handleUpdateModelConfig` does, and refuse with its message. Keep
      `envBlockWarning(cfg.Env)`; when it is non-empty, `autoconfig_message`
      shows it as a warning line beneath the saved message.
   5. `s.registry.SaveProfileFrom(id, models.AutoconfigProfileName, profile)`.
   6. With `apply=1`: `s.registry.ApplyProfile(id, AutoconfigProfileName)`,
      then the two steps every config save takes -- recompute
      `m.VRAMEstimate` with `s.configuredEnvPairs(m)` and `Register`.
   7. `clearAutoconfigRun(id)`. Respond with `autoconfig_message` and the
      header `HX-Trigger: modelsChanged`, so the card's VRAM figure and an open
      Configure panel are refreshed. The message is "Saved as the Autoconfig
      profile and now in use. It takes effect the next time this model
      starts." or "Saved as the Autoconfig profile. Restore it from Configure
      when you want to use it.", with "replacing the earlier one" when it did.

7. `/discard` clears the run and returns an empty body, which empties the
   slot.

8. Styles in `layout.html`: `.autoconfig-table` (the five columns, the reason
   wrapping), `.autoconfig-warning` (the warning row's border and
   background, using the existing warning colour variables), and
   `.autoconfig-width` for the radios.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
go test ./internal/api -run 'TestGoldenFragments|TestGoldenPartials' -update
git diff --stat internal/api/testdata/golden/   # only the model list and the new recordings change
```

## Test plan

Automated:

- Goldens: `models_list` (now with the button and slot);
  `partial_autoconfig_hardware_table` with and without a pinned KV pool;
  `partial_autoconfig_dialog` in the nothing-serving, model-serving,
  no-helper, helper-too-large and engine-busy states;
  `partial_autoconfig_progress`; `partial_autoconfig_review` for a two-width
  plan with a warning row, an unticked row and a quote, for a single-width
  plan, and for an unknown plan; `partial_autoconfig_message`.
- `/start` for a draft is refused with a visible message.
- `/save` with `apply=0`: the profile exists with source `autoconfig`, the
  live config is unchanged, `ActiveProfile` is unchanged.
- `/save` with `apply=1`: the live config equals the profile's, the active
  profile is Autoconfig and unmodified, and the stored VRAM estimate was
  recomputed.
- `/save` with a row key omitted: that row's setting is absent from the saved
  config. With `width=narrow`: the narrow plan's five hardware fields.
- `/save` on a read-only registry refuses and changes nothing; with an
  attention backend the image does not offer it refuses with the existing
  message.
- `/save` twice in a row: the second reports that there is no result.
- `/status` for another model's run returns nothing.

Manual, on the workstation:

1. Press Autoconfigure on a model, choose Medium, Start. Watch the progress
   lines; the review opens by itself.
2. Untick one row, choose *Save and apply*, open Configure: the profile picker
   shows Autoconfig as active, and the unticked setting is not in the config.
3. Press Autoconfigure again and Start without "Read the card again": the
   review appears in seconds and the engine was never touched.

## Commit

```
feat(autoconfig): the Autoconfigure button, its start panel and its review
```

## Rollback

Revert the commit. Autoconfig profiles already saved stay in `models.json` and
remain restorable from the Configure panel's profile picker, since they are
ordinary profiles with extra fields.

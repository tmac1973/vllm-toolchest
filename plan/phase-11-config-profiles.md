# Phase 11: Per-model config profiles

**Status:** implemented on the `variant-manifests` branch, in the four commits
§7 lists, with unit, handler and golden tests. Not yet exercised in a browser
against a live server.

## About this document

Written 2026-09-11 from a planning session that was cut off before any code was
written. The decisions below were made with Tim in that session; the design was
produced alongside them and then checked against the code before starting. Where
the two disagreed, §6 says so.

## 1. What this is

A named snapshot of one model's launch config, taken when the operator asks and
restored when the operator asks.

- **Save as profile** copies the model's live `vllm_config` under a name the
  operator types. Saving under an existing name replaces it and says so.
- **Restore** picks a profile from a dropdown and copies it over the live config.
- The config panel keeps autosaving exactly as it does now. There is no
  explicit save of the live config and no automatic snapshot: overwriting a
  config that was never saved is the operator's lookout.

## 2. Decisions

| Question | Decision | Why |
|---|---|---|
| Scope | Per-model only. Settings → vLLM Defaults stays as it is. | A global template cannot carry `speculative_config`, which is shaped by the model and is the field most worth switching. It would be a weaker object with twice the UI. |
| Safety net | None beyond the profiles themselves. | Explicit save, explicit restore. |
| Storage | `models.json`, with `schema_version` actually read. | Today `load()` never reads the version, drops unknown fields, and `save()` rewrites the whole file — so an older build would silently delete every profile. |
| Benchmarks | Widen `ConfigSnapshot` and record the profile name. | It captures 7 fields and none of attention backend, speculative config or prefix caching: the knobs a profile exists to A/B. |

## 3. Storage

Profiles live on the registry envelope, `registryFile.Profiles`
(`config_profiles`), not on `Model`. `Register` replaces the whole `*Model`, and
`RegisterFromDownload` builds a fresh one, so anything hung off the old record
is destroyed by a re-download — the case profiles are most useful for. The
envelope is also where `pending_configs` already lives, for the same reason.

`Model.ActiveProfile` records the profile the live config was last saved as or
restored from. It is a label, not a link. Drift is detected by comparing
`VLLMConfig` with `==`, which works because every field is a scalar; a test pins
that.

Profiles are not pruned when a model is deleted: a model removed to free disk
and pulled again is the main reason to have had one.

### The schema gate

`load()` used to log a parse error and return with an empty map, after which
the next `save()` wrote that empty map over the file. A corrupt file, or one
from a newer build, destroyed the registry.

Now `load()` sets a read-only reason on a read error, a parse error, or a
`schema_version` newer than this build writes. Every mutator refuses up front,
and `save()` refuses as the backstop. The models that did parse are still
listed, so the operator sees their models and the reason rather than an empty
page. The write is now write-then-rename.

Versions: 2 is `pending_configs` and the first gated build; 3 adds
`config_profiles`. Version 0 (absent) is read as current.

`benchmarks.json` has the same bug and gets the same gate in commit 4. It
matters more there: benchmark history cannot be rebuilt by rescanning.

## 4. UI

A profile bar sits between the VRAM banner and the config `<form>`, and must
stay **outside** it. The form autosaves on `change` with
`hx-include="closest form"`; a `<select>` inside it would fire a save each time
the operator browsed the list, and the name box would be posted with every
save. A handler test posts the form's own fields and asserts no profile changed.

- A dropdown of the model's profiles, labelled with the date and image they
  were saved on, with Restore and Delete. Delete confirms; Restore does not.
- A name box pre-filled with the active profile, and **Save as profile**.
- "From profile *x* — edited since" under it when the live config has drifted.

Routes are POST under `/api/models/profiles`, `/profiles/apply` and
`/profiles/delete`. Delete is a POST because htmx sends included parameters in
the body for DELETE, where `ParseForm` does not look.

Every handler re-renders the panel through one function with a banner
(OK/Warning/Error), rather than swapping an error over the form.

### Restore validates

A profile is a config written at another time, possibly on another image. The
restore handler runs the same check the config PUT does
(`validateNamedBackends`) on `speculative_config` and `extra_flags`, and
**refuses** on failure: those are free text nothing renders as a choice, and a
bad value aborts the engine minutes into a load.

A picker `attention_backend` the image does not offer is **restored with a
warning** instead. The picker already keeps such a value as a labelled option
so it is not lost; refusing would make the profile permanently unrestorable on
this image.

## 5. Benchmarks

`ConfigSnapshot` gains `ProfileName`, `ProfileModified`, and the fields that
move the numbers: speculative config, attention backend, compilation config,
prefix caching with mamba cache mode, chunked prefill with its token budget,
KV cache memory, async scheduling, forced quantization, and extra flags.
Parsers, tokenizer, chat template and the like stay out: they do not change
throughput, and every field added is a comparison dimension.

The two builders of the snapshot (`configSnapshotFromModel` in `bench.go` and
the literal in `jobs_env.go`) have already drifted — neither sets `MaxNumSeqs` —
so they collapse into one method. A sweep job clears the profile name on its
cells, since only one cell of a sweep can match the profile.

The comparison view gains columns that only appear when runs disagree, and CSV
export gains the matching columns.

## 6. Where the design was corrected against the code

- `unofferedBackendWarning`, which the restore handler calls, does not exist.
  It is written in commit 3 beside `backendOptionsFor`, from the same rule.
- The design refused writes only in `save()`. That is too late: `UpdateConfig`
  and friends change memory first, so a refused write would leave this process
  launching a config the panel had reported as not saved, and `Delete` removes
  the model's files before it reaches `save()` at all. Every mutator checks the
  gate first.
- The version bump to 3 moves to commit 2, where `config_profiles` actually
  appears. Commit 1 starts reading the version and keeps writing 2.

## 7. Commits

1. The `models.json` schema gate and atomic write. No feature; a data-loss fix
   that should be revertable on its own.
2. The profile type and registry methods, with tests. Schema version 3.
3. Routes, handlers, the template, and regenerated golden files.
4. The wider benchmark snapshot, one snapshot builder, comparison and CSV
   columns, and the `benchmarks.json` gate.

## 8. Found along the way

**A `max_num_seqs` sweep did not change what was launched.** It is offered as
a sweep parameter (`internal/benchmark/sweep.go`), and `applyOverrides` writes
the swept value into the cell's snapshot, but `vllmStartConfigFor`
(`internal/api/jobs_env.go`) built the launch from six snapshot fields and
`MaxNumSeqs` was not one of them. Every cell of such a sweep ran at the model's
saved value while its run recorded the swept one, so the comparison showed a
difference that was never measured. It predated this phase, and is fixed in its
own commit with a test. Sweep results for `max_num_seqs` recorded before that
fix measured the saved value in every cell and should not be trusted.

## 9. Out of scope

- Converting the autosave's rejection to the banner, so a refused save no
  longer replaces the panel with a bare error line. An obvious follow-up; it
  would change the golden files again.
- Preserving unknown envelope keys across a write. With the version gate, the
  only file that could carry them is one from a sibling branch at the same
  version.
- Validating tool-call and reasoning parser names, which vary by vLLM version.
  The better source is `vllmctl` reading the installed parser registry at boot.
- Showing a read-only `benchmarks.json` on the Benchmarks page. The store
  refuses to write and logs why, but nothing in the UI says so yet; a run
  recorded meanwhile is visible until the next restart and then gone. The
  config panel does show a read-only `models.json`.
- A test for the sweep clearing the profile label on its cells. It is one line
  in `runCell`, and driving `runCell` needs a job-runner harness.

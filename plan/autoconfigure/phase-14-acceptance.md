# Phase 14 — Prove it on real models, and document it

**Depends on:** phases 01-13 · **Enables:** nothing further. This closes the
feature.

## Goal

The overview's success criteria are about real hardware: an autoconfigured
model must start on the first attempt and, after refinement, match the config
built by hand. This phase runs that test on the reference host and on a
single-card host, records what happened, fixes what it finds, and writes the
help text and README entry. It is also where the helper model is confirmed to
load on the images available, since "universal" has so far been reasoned about
and not measured.

## Files touched

- `plan/autoconfigure/acceptance.md` — new. The record of each run.
- `web/templates/help.html` — an Autoconfigure section.
- `README.md` — the feature, in the list the README already keeps.
- `plan/README.md` — mark the feature built.
- `plan/todo.md` — remove the items this closes; add what acceptance leaves
  open.
- Whatever source files the runs show to be wrong, each as its own commit with
  its own test.

## Steps

1. **Deploy** the build to compute and to the workstation with
   `./setup.sh quick`.

2. **Preserve the hand-built configs first.** On compute, for
   `tcclaviger/ThinkingCap-3.8-27B-PARO5` and
   `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`, save the current config as a
   profile named `Hand-tuned` from the Configure panel. On the workstation do
   the same for `cyankiwi/Qwen3.5-4B-AWQ-4bit`, the model whose figures are
   already in `plan/todo.md` from its single RX 7900 XTX. These are what the results are compared
   against and what the machines are restored to.

3. **Helper on each image.** On compute (`rdna4-clav`) and on the workstation's
   image: download the helper from Settings, then run Autoconfigure on any
   model with nothing serving. Record the helper's start time, the engine's
   reported consumed memory and KV pool from its log, and whether the form
   came back on the first request or needed the `guided_json` retry. The
   images not available to test (`cuda`, `xpu`, `gfx906`, `rocm-cdna`,
   `strix-halo`, `gb10`, `radiance`) are listed in `acceptance.md` as
   unverified, by name.

4. **The reference models.** For each of the two models on compute:
   1. Reset the live config to defaults by deleting and re-scanning the model
      record with files kept (`DELETE /api/models/delete?id=…&keep_files=true`,
      then a scan), which also clears its measurement. The `Hand-tuned` profile
      survives, because profiles outlive the record.
   2. With the other model serving, run Autoconfigure with **Maximum** and
      default choices. Record the downtime of the serving model, from its stop
      to its being healthy again, and confirm it is serving afterwards.
   3. *Save and apply*, then Start. **Criterion: it starts on the first
      attempt.** If it does not, record the engine's error, confirm phase 12's
      notice proposes a fix, apply it, and record how many attempts it took.
   4. Open Configure. If phase 13's notice is present, apply it and restart.
   5. Compare with `Hand-tuned`, field by field, using
      `GET /api/models/get?id=…` for the live config and the profile list for
      the other. **Criterion: the same** tool-call parser, reasoning parser,
      speculative config, extra flags other than sampling overrides, and
      environment lines; **at least** the hand-tuned `max_model_len` and
      `max_num_seqs`. Record every difference, including the ones allowed.

5. **The single-card host.** On the workstation, the same sequence for
   `cyankiwi/Qwen3.5-4B-AWQ-4bit`, with the criterion unchanged. Record whether the helper fits beside
   nothing on that card, and the review's statement of the width (there is
   only one).

6. **The edges the automated tests cannot reach**, each recorded as pass or
   fail with what was seen:
   - a run with the helper removed: a proposal from this machine and the
     card's command, with the note that the text was not read;
   - a second run on an unchanged card: no engine activity
     (`GET /api/service/status` shows the same PID throughout);
   - a run started while a benchmark job is running: passes when the run
     completes, the engine's PID is unchanged throughout, the benchmark job
     finishes unaffected, and the result either reuses earlier advice or
     carries the busy reason as a note;
   - the helper failing to start (rename its weights directory for one run):
     the serving model is restored and the result is from this machine and
     the card's command, with the failure noted;
   - a model whose card carries a flag this image lacks
     (`Qwen/Qwen3-14B-AWQ` and `--enable-reasoning`): the row is unticked.

7. **Write `acceptance.md`**: a table per host with a row per model -- attempts
   to first start, downtime of the interrupted model, differences from
   `Hand-tuned`, context before and after refinement against the engine's
   reported pool -- and the list of unverified images. State plainly which
   success criteria were met and which were not.

8. **Fix what failed.** The scope is the success criteria: a defect that
   makes a criterion fail is fixed here, in the phase's code it belongs to,
   with a test that reproduces it, as a separate commit. Anything else found
   along the way goes to `plan/todo.md` and is not fixed in this phase. A
   criterion that cannot be met is not quietly dropped: it is written into
   `acceptance.md` and `plan/todo.md` with the reason.

9. **Restore the machines.** Apply `Hand-tuned` on any model where the
   autoconfigured result was worse, and leave compute serving the model it was
   serving before step 3.

10. **Help text.** A section in `help.html` after the VRAM estimate entry:
    what the button does; that it may stop the serving model for a few minutes
    and restarts it; what the helper model is, how large, and where to install
    or remove it; what "first guess" means and why a notice appears after the
    first start; that every flag and variable from a card can be unticked, and
    that `trust_remote_code` is called out when a card asks for it; that the
    result is a profile and nothing changes until it is applied.

11. **README and plan index.** Add Autoconfigure to the README's feature list
    in two sentences. In `plan/README.md` change the autoconfigure entry from
    planned to built, with the date. In `plan/todo.md`, under "Direction",
    record that autoconfigure is built, and list anything acceptance left
    open.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

and `acceptance.md` exists with a result for every criterion in the overview,
each marked met or not met.

## Test plan

This phase is the test plan for the feature. It is done when:

- both reference models and the single-card model have a recorded run;
- each run's attempts, downtime and differences are written down, not
  summarised from memory;
- every image variant is listed as verified or unverified for the helper;
- the machines are back on the configs and the serving model they started
  with, confirmed with `GET /api/service/status` and `GET /api/models/get`.

## Commit

```
docs(autoconfig): record acceptance on real models, and document the feature
```

with each code fix before it as its own `fix(autoconfig): …` commit.

## Rollback

The documents revert cleanly. Code fixes made here revert individually. On the
machines, applying the `Hand-tuned` profile to a model restores its config
exactly; that profile is never modified by this phase.

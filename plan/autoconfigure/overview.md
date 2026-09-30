# Autoconfigure — Project Overview

## Problem

Configuring a model for vLLM by hand takes knowledge the operator should not
need. The right tool-call and reasoning parsers, the sampling settings the
publisher recommends, whether a draft model exists for it, which offload flags
a specialised checkpoint requires -- all of that is in the model card, usually
as a `vllm serve` command written for the author's machine. The rest depends on
this machine: how many cards to split over, how much context fits, how many
requests can run at once. Today the operator reads the card, translates the
command by hand, guesses the hardware settings, and finds out whether the guess
was right several minutes into a load.

llama-toolchest already solves the same problem for llama.cpp: an
Autoconfigure button has a small helper model read the card, checks the answer,
fits the model to the machine, and offers the result as a profile to review.
vllm-toolchest has the pieces it would need -- config profiles, a VRAM estimate
and fit, measurements captured from every successful start, engine advice
parsed from the log, draft-model detection -- but nothing that puts them
together, and its plans so far stop at seeding three values on download.

## Goals

- An **Autoconfigure** button on every servable model card that produces a
  complete proposed `VLLMConfig` in one action.
- A **card half**: the model card is turned into settings. The whole of the
  card's serve command is taken -- flags that map to a known field go to that
  field, other flags go to extra flags, environment variables go to the
  model's env block. The command is extracted by code, so it is used even
  where no helper can run; a helper model reads the rest -- which of several
  commands applies, and advice written as prose: sampling, parsers, a draft
  model.
- A **hardware half**: tensor-parallel width, context length, KV cache dtype,
  GPU memory utilization and the sequence cap, computed for this machine. It
  works without the helper, and its values win over hardware values named in
  the card, which are shown as notes -- with one exception: a KV cache dtype
  the card's command names is used, because it is the author's tested setting
  for that checkpoint. The sequence cap is the machine-wide default, and the
  review reports how many full-context requests fit beside it.
- The width is offered as a choice between **all cards** (preselected) and the
  **narrowest that fits**, each shown with the full-context requests it buys.
- A **review** before anything changes: current versus proposed for each
  setting, with its source (the model card, or this machine) and the reason, quoting the card where the card is the source. Then
  *Save as Autoconfig profile*, *Save and apply*, or *Discard*.
- **Refinement from a real start.** When a start records a measurement that
  would change the context, the model's config panel says so and opens a
  review of the hardware settings with the refined value. The settings that
  came from the card are left as they are, so nothing is read, loaded or
  interrupted to produce it.
- **Recovery from a failed start.** When a start fails, the model shows a
  corrected proposal built from what the engine itself said: the context it
  reported room for, the memory fraction it reported free, or the flag it did
  not recognise. A bare out-of-memory, where the engine names no figure, gets
  the context halved, and says so.
- **Draft models.** A compatible draft already installed is paired in the
  proposal. A draft the card names that is not installed is listed in the
  review with a Download button.
- **One hardware planner** for the whole project. The recommend feed's download
  step (phase 19) seeds a new model by calling it, so seeding and
  autoconfigure follow one set of rules and agree whenever their inputs do.

## Non-goals

- **No benchmarking.** Autoconfigure measures nothing about speed and compares
  no alternatives. llama-toolchest's Autotune, which does, is not ported here.
- **No configuration without a click.** Autoconfigure never runs on
  download, nothing retries a failed start by itself, and a measurement never
  rewrites `VLLMConfig`. Every change autoconfigure makes to a live config
  follows an explicit apply. The one exception is outside this feature: the
  recommend feed's own download step (phase 19, not yet built) seeds a model
  that has no configuration yet, as that plan already describes.
- **No choice of helper model.** It is one fixed model, the same on every
  host, downloaded when first needed and removable from Settings. It is not
  selectable, and an external endpoint is not supported.
- **No helper on small cards.** A host whose smallest card cannot hold the
  helper gets the hardware half and the structured sources only, with a note
  saying why.
- **No use of a serving model as the helper.** Whatever is serving is stopped
  for the helper and restored afterwards; its answers are never relied on.
- **No reading of cards from anywhere but the Hub.** A model with no Hub
  repository is configured from the hardware half alone.
- **No change to the recommend feed's ranking, pool or UI** (phases 15-18).
  Only phase 19's seeding is rewritten to call the shared planner.
- **No further estimator calibration.** The projection is used as it stands
  after 2026-09-30; its known gaps are handled by the refinement step rather
  than by more modelling.

## Users & primary flow

The user is the operator of one machine running vLLM through this tool -- four
R9700s on `rdna4-clav` in the case driving the design, a single card in the
other case that must work.

1. They download a model and open **Models**. Its card has an
   **Autoconfigure** button.
2. The start dialog asks one thing: the context size wanted -- short, medium,
   long, or the model's maximum. When the card was read before, it also
   offers to read it again, unchecked; left unchecked, an unchanged card is
   not read a second time. If a model is serving, the dialog names it and
   says it will be stopped while the card is read and restarted afterwards.
   If the helper model is not installed, the dialog offers to download it, or
   to continue without it, using the card's command and this machine.
3. On confirm, the run proceeds with a progress line for each step: reading
   the card; stopping the serving model; starting the helper; asking it to read
   the card; stopping the helper; restarting the model that was serving;
   checking the answer; fitting the model to the machine. The engine steps are
   skipped when the card is unchanged since the last run -- saved, or run
   earlier in this server's lifetime -- whose reading is reused.
4. The review opens. It shows the width choice -- all cards, preselected, and
   the narrowest that fits, each with its full-context request count -- and a
   table of every setting that would change or was deliberately kept, with its
   source and reason. Hardware values the card named appear as notes beside the
   values chosen for this machine. A model that has never run says its hardware
   values are a first guess.
5. They choose *Save and apply*. The proposal is saved as the model's
   **Autoconfig** profile and becomes the live config.
6. They start the model. If the start fails on memory, the model's config panel
   shows a corrected proposal from the engine's own error; they apply it and
   start again.
7. Once a start succeeds, the engine's figures are recorded as they are today.
   If they would change the context -- more fits than was guessed, or less --
   the config panel shows *Measured on a real start: review refined settings*.
   The review shows the hardware table with the refined context; they apply
   it, and it takes effect on the next start.

## Constraints

- **vLLM serves one model at a time and claims its cards at startup.** The
  helper cannot load beside a serving model, so a run that reads the card
  costs a stop, a helper start, and a restart of what was serving. On the
  reference host a large model's reload alone is five to ten minutes.
- **The helper must load on every image variant.** Those span vLLM 0.23 to
  0.29 and NVIDIA, five AMD stacks and Intel. Quantized formats are not
  portable across them (block FP8 cannot run on RDNA3 at all), so the helper is
  unquantized safetensors, and a plain dense architecture old enough for the
  oldest pinned engine: `Qwen/Qwen3-4B-Instruct-2507`. It is 4.02B parameters,
  roughly 8 GB of weights, and
  is launched with a short context, eager mode and one sequence to keep its
  footprint near its weights. Hosts whose smallest card is under about 12 GB
  cannot run it.
- **The helper's answer is untrusted.** It is constrained to a JSON schema,
  every value that can become a setting must carry the card's own words, and
  code checks that those words are in the card before it becomes a proposal.
  The helper's free-text summary cannot be checked that way; it is shown
  labelled as unchecked and never becomes a setting. Flags and variables taken from the card's command are
  passed through the checks a hand-typed config already gets: the named
  attention-backend validation, the env block's validation and warnings, and
  the variant's flag support where it is known.
- **Taking the whole command can propose a flag this image does not have.**
  That aborts a load minutes in. The installed vLLM is asked once per image
  for the flags it accepts, and a card flag it does not list starts unticked.
  The review is the second control: every flag and variable from the card is
  a visible row that can be unticked before saving, and `--trust-remote-code`
  is called out in a warning row of its own.
- **The projection is a range, and wide.** For a large model at four ranks the
  pessimistic end is often beyond the host, so a guaranteed fit is rare before
  a model has run. The planner therefore chooses on the expected figure and
  labels the result a first guess; refinement and failed-start recovery exist
  to close that gap.
- **A model holds one measurement**, tied to its configuration by fingerprint
  and to the engine by image variant and version. Refinement can only use a
  measurement that still applies; changing the width retires it.
- **The registry can be read-only** (a `models.json` newer than the build).
  Saving a profile or applying one must refuse cleanly in that state, as the
  profile and advice handlers already do.
- **Existing mechanisms are reused, not duplicated**: config profiles
  (`internal/models/profiles.go`), `Fit` and `RequiredAt`
  (`internal/models/vram_fit.go`, `vram.go`), `RunMeasurement`
  (`internal/models/measured.go`), engine advice (`internal/advice`), draft
  detection and pairing (`internal/models/draft.go`), and the download queue.
- **Two planning documents contradict this work** and are amended by it:
  `plan/recommend-models-overview.md` lists "no launch-config optimizer" and
  "no automatic re-configuration after a run" as non-goals, and
  `plan/phase-19-fit-handoff.md` says measurement must never touch a config.

## Success criteria

- On the reference host, with the 27B hybrid and the 125B MoE each reset to
  default config, Autoconfigure followed by *Save and apply* produces a config
  that **starts on the first attempt**.
- After the measured refinement is applied, each of those two configs has the
  **same parsers, draft pairing, offload flags and environment** as the config
  built by hand, and **at least its context length and sequence cap**.
- The same holds for one model on a **single-card host**: first-attempt start,
  and settings matching the hand-built config after refinement.
- Every row in a review names its source, and every row sourced from the card
  quotes the card. No setting reaches a proposal that the review does not show.
- A run with no helper installed, and a run on a card whose text yields no
  usable advice, both complete with a proposal built from this machine and the
  card's command, and a note saying the text was not used.
- A run started while a model is serving ends with that model serving again,
  including when the helper fails to start or the run times out.
- A failed start on memory and a measured start that changes the hardware
  settings each produce a notice on the model's config panel; applying either
  changes only the settings shown.
- The recommend feed's plan (phase 19) seeds a downloaded model by calling the
  same planner autoconfigure uses, with the same inputs for a model with no
  card. The feed itself is built later; this criterion is met by the amended
  plan.
- `go test ./...` passes, with the helper and the Hub faked in tests.

## Decisions

- **Trigger** → A button on each model card, run on demand. Nothing runs
  automatically on download.
- **Plan layout** → A folder of its own, `plan/autoconfigure/`, with its own
  phase numbering.
- **First start on a model that has never run** → Detect the narrowest width
  that should fit and surface the choice between narrow and all cards, because
  only one model runs at a time and extra cards buy context and concurrency.
- **Default width** → All cards is preselected; narrowest-that-fits is the
  alternative.
- **Helper model** → A dedicated helper, universal across GPUs and images,
  rather than the serving model, an external endpoint, or rules alone.
- **Helper size** → About 4B parameters, unquantized, plain dense
  architecture. Hosts with cards too small for it get the hardware half only.
- **When a model is serving** → Warn, stop it, run the helper, then restore the
  model that was serving.
- **The card's `vllm serve` command** → Take the whole command: known flags to
  their fields, unknown flags to extra flags, variables to the env block, all
  visible in the review.
- **Card versus this machine on hardware settings** → This machine's fit wins;
  the card's value is shown as a note.
- **Dialog input** → Context size only; concurrency is whatever full-length
  requests the remaining memory holds.
- **Delivering the result** → Review, then save as the Autoconfig profile or
  save and apply; the live config changes only on apply.
- **Refinement after a measured start** → A notice on the model that opens a
  review of the hardware settings. Only the context can change, since the
  width and KV dtype are what the measurement describes; the card's settings
  are left as they are, so no card is read.
- **Failed start** → Propose a fix from the engine's own error, applied by the
  operator before starting again. No automatic retry.
- **Draft models** → Pair installed compatible drafts; suggest a download for
  one the card names that is missing.
- **Relation to the recommend feed's seeding** → One planner; phase 19's
  seeding calls it.
- **Success** → Starts first time and matches the hand-tuned configs after
  refinement, on the reference host and on a single-card host.
- **Helper repository** → `Qwen/Qwen3-4B-Instruct-2507`: Apache-2.0,
  ungated, unquantized BF16, dense, and with no thinking mode to switch off.
- **max_num_seqs** → The machine-wide default, with the full-context request
  count reported in the review. This corrects the dialog-input answer above:
  the sequence cap is a batch limit, not a memory reservation, and setting it
  to the full-context count would have cut throughput below the hand-built
  configs.
- **Checking the card's flags against the image** → Probe the installed
  vLLM's flag list once per image; a flag it does not list starts unticked.
- **KV cache dtype** → The card's choice when its command names one;
  otherwise the engine default, switching to fp8 only when the context asked
  for does not fit without it.
- **`--trust-remote-code` from a card** → Proposed and ticked, in a warning
  row of its own, when the card's command uses it. Mentioned only in the
  card's text, it is proposed unticked with the same warning, like every
  other flag the card mentions without prescribing.
- **Decided earlier, 2026-09-30** → Measurement may correct a config by
  proposal the operator applies, never silently; estimator work is bounded and
  has stopped.

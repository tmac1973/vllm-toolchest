# Autoconfigure — acceptance record

Phase 14's record. Phases 01-13 are built on the `autoconfigure` branch with
`go test ./...` passing. What follows is what has been checked against real
cards and real hardware, and what has not yet, criterion by criterion. Nothing
here is summarised from memory: each line says how it was checked.

## Seventh run on compute: Gemma 4, 2026-10-01

`tcclaviger/gemma-4-31B-it-MXFP416-MTP`: a hybrid of sliding-window and
global attention, a custom MXFP4 format only the rdna4-clav image loads, and
an EAGLE-3 drafter bundled in its own folder. The first review planned 91,136
tokens with an fp8 cache and offered to download the drafter it already had.

- **#66.** Sixty layers were costed as full attention: 983,040 bytes a token.
  Fifty are sliding (1,024-token window), ten global at their own shape; the
  estimate became 85,120 at 262,144.
- **#67.** The bundled drafter is found and paired from the model's folder;
  "256K" is read as 262,144, not 256. **#68** checks the card's required
  image against the running one; **#69** names the drafter on the card.
- **#72, #73.** The drafter cannot start on today's image: its TP-padding
  code (`vllm/config/tp_padding.py`, 2026-09-27) reads the target's
  `num_key_value_heads`, which transformers 5.17 refuses for Gemma 4's
  per-layer config with a RuntimeError that `getattr`'s default does not
  catch. Read over SSH in the image; not yet reported upstream. The failed-start
  notice now offers to drop a speculative config the engine cannot build,
  and unticking a setting already applied now removes it.

Without the drafter, Save and apply, then Start:

- **Started on the first attempt**, 4 cards at 262,144 with the default
  cache; about 11 minutes, most of it engine init.
- KV cache 747,098 tokens: 2.85 full-length requests, against 2 planned.
  25,195 bytes a token per card against 21,280 estimated -- the global
  layers' 20,480 exactly, the sliding layers' share under by about 4,700.
  Window plus one scheduled chunk (2,496) would give 23,229; the rest is not
  explained by one start, and the margin covered it.
- Consumed 11.14 GiB a rank: weights 9.21 plus 1.93, in the wide band.
- A tool call came back correct; decode about 24.5 tokens a second.

## Sixth run on compute: a third-party quant, 2026-10-01

`stelterlab/Mistral-Small-3.2-24B-Instruct-2506-FP8`: someone else's quant,
whose card is the original's with its own command added above it, and whose
repo is Mistral's layout only. Looked at ahead of the download, which found
#63: its `config.json` declares fp8, which a new model's config takes, and
its `params.json` compressed-tensors, which `--config-format mistral` has
the engine read. The review now proposes unsetting the quantization.

The review chose the repo's own command over the original's pasted below it;
ticked Mistral's three flags, tool calling, the unset quantization, and
`{"temperature": 0.15}` -- read from the helper's quote, there being no
`generation_config.json` -- and rejected, with a note, the helper's
`mistral` reasoning parser again. Save and apply, then Start:

- **Started on the first attempt**, about 2.5 minutes. The engine selected
  its compressed-tensors FP8 kernel.
- KV cache 525,040 tokens: 4.01 full-length requests at 131,072, against 2
  projected. Consumed less weights, 2.11 GiB a rank.
- A tool-calling request made a correct `get_weather` call.

Three models of two families, a first-party quant, a first-party BF16 and a
third-party quant, have now gone from download to serving on their first
start with the review's defaults.

## Fifth run on compute: another family, 2026-10-01

`mistralai/Mistral-Small-3.2-24B-Instruct-2506`, chosen for a card unlike
Qwen's: underscored flags, Mistral's own tokenizer, config and weight
formats, a `mistral` tool parser, one sampling value in prose, vision. Run
through the UI by the operator. What it found, in order:

- **#57.** The download tab could not find it: the search asked for the
  `transformers` library, and the repo is tagged `vllm` only.
- **#58, then #60.** The repo holds its weights twice, 48 GB each. #58 kept
  the Hugging Face shards, and the first start failed, "Can't load image
  processor": those shards come with no tokenizer or processor config. #60
  keeps the layout that is complete -- here Mistral's consolidated copy.
  #58 also read `--load_format` and the like as the dashed flags they are.
- **#59.** The helper read a `mistral` reasoning parser for a model that does
  not reason, citing `--tool-call-parser mistral`; a parser's name must now
  appear in the quote as its kind.
- **#61.** A finished download and a delete now show in the model list
  without a manual refresh.

The final review proposed 4 cards at its full 131,072 tokens; tool calling
with `mistral`, `--tokenizer-mode`, `--config-format` and `--load-format
mistral`, and `--limit-mm-per-prompt {"image":10}`, ticked; no sampling
row, since the model's `generation_config.json` already sets the card's
`temperature=0.15`. Save and apply, then Start:

- **Started on the first attempt**, about 4 minutes.
- KV cache 430,672 tokens: 3.29 full-length requests, against 1 projected.
  Consumed less weights was 0.78 GiB a rank (12.24 less 11.46), under even
  the 2.3 the Qwen3.5 run set. The context is the model's maximum, so the
  refinement had nothing to change.
- A tool-calling request made a correct `get_weather` call, and an image
  request read the image's colours.

## Fourth run on compute: a new model, 2026-10-01

A model never configured here, run through the UI by the operator:
`Qwen/Qwen3.5-35B-A3B-FP8`, from a fresh download with the default config
(one card, 8,192 tokens). Four rounds, each fixing what the last one showed:

- **#50.** The helper refused the card: 14,337 prompt tokens and 2,048 for
  the answer, over its 16,384. The budget now covers the whole prompt at
  2.5 characters a token. The base model's card repeated the quant card's
  four commands; a recipe is now listed once. Without the helper, the
  fullest command is used.
- **#51.** The card gives a base command and one variant per feature (tool
  calling, MTP, `--language-model-only`); the helper rightly chose the base,
  and the features were lost. What a chosen command's variants add is now
  proposed: features ticked, vision-off unticked. The card's presence
  penalty was dropped silently because a `generation_config.json` existed;
  values the file does not set are now proposed.
- **#52.** Still no sampling, and no note. The review now shows the helper's
  answer as it gave it.
- **#53.** That answer showed the helper quoted all four sampling sets and
  gave null for every value. The general set is now read from its quote.

The final review proposed 4 cards and 262,144 tokens, with room for 7
full-length requests; reasoning `qwen3`, tool calling with `qwen3_coder`,
MTP (`qwen3_next_mtp`, 2 tokens) and `{"presence_penalty": 1.5}`, ticked;
`--language-model-only`, unticked. Save and apply, then Start:

- **Started on the first attempt.** About 7 minutes, 4m19s of it engine init
  (88 s compiling). vLLM maps `qwen3_next_mtp` to `mtp` with a deprecation
  warning.
- KV cache 1,666,259 tokens: 6.36 full-length requests at once, against the
  7 planned -- the projection was about 10% high, but the context fit, and
  the refinement found nothing to change.
- A tool-calling request answered with reasoning separated and a correct
  `get_weather` call.

This is the first model taken from download to serving by Autoconfigure
alone, with nothing hand-tuned to match.

## Third run on compute, 2026-10-01

Deployed after PR #48.

- **Failed start, context too long: met on hardware.** At two cards, 0.76 and
  262,144 tokens, vLLM 0.29 reported room for 196,112 tokens; Configure
  proposed 195,584, and Apply set it.
- **The MoE's rejected readings, explained** by the cited sentence the note
  now shows. Sampling: the helper quoted the base card's sampling list
  accurately but without the blank line and indentation between its lines --
  the check was too strict, and now accepts a quote whose every line is in
  the card. Parser: the helper cited its own summary sentence, which is not
  in the card -- the rejection was right. Neither could have changed the
  proposal, since the command sets both, and such rejections are no longer
  reported.
- Compute left with both models on `Hand-tuned`; the engine shows the error of the deliberate failure until the next Start, since Stop refuses from that state.

Every success criterion is now met on compute except the single-card host,
which needs the workstation deployed.

## Second run on compute, 2026-10-01

Deployed after PR #47 (the PLE default, value-in-quote and markdown fixes).

- **MoE, re-run from defaults: met.** Planned four cards, 262,144 tokens,
  fp8, room for 2.6 full-context requests; saved, applied, and **started on
  the first attempt** (4 min 11 s). The earlier reading of its card was
  reused from the saved Autoconfig profile, so the helper was not loaded and
  nothing was interrupted.
- **Refinement: met on hardware.** With the autoconfigured 27B's context
  lowered by hand to 65,536, Configure said "Measured on a real start ...
  Context can be 262,144 tokens (now 65,536)"; the review showed the hardware
  table from the measurement; Apply restored 262,144 and the notice cleared.
- **Failed start, unknown flag: met on hardware.** Started with
  `--definitely-not-a-flag`; Configure showed the engine's line and offered
  to remove the flag; Apply removed only that flag and the notice cleared.
- **Failed start, context too long: not met; fixed.** Started at two cards,
  0.76 of each, and 262,144 tokens. vLLM 0.29 refused with "the estimated
  maximum model length is 214240", a wording the advice rules did not know,
  so the notice showed the line with nothing to apply. A rule for it now
  proposes 214,016. Not re-run.
- The MoE's sampling and parser readings were again rejected as "not in the
  card". The likely quotes all match the card as normalised, so the helper
  probably paraphrased; the rejection note now shows the cited sentence, so
  the next run will say which.
- Afterwards both models are on `Hand-tuned` again. The engine is left in
  the error state of the last deliberate failure: Stop refuses from that
  state, and the next Start clears it.

| criterion (overview) | status |
|---|---|
| 27B starts first time after Save and apply | met (first run) |
| MoE starts first time after Save and apply | **met** |
| Same parsers, draft, offload flags, env; at least context and cap | met for the card half of both; the MoE's hand config also has prefix caching and three tunable-op variables the card does not name |
| Single-card host | not yet run |
| Interrupting a serving model and restoring it | met (first run) |
| Failed-start and refinement notices | **met on hardware**, bar the 0.29 wording, now fixed |

## Run on compute, 2026-10-01

Deployed build `816cfe2` (PR #45), image `rdna4-clav`, vLLM 0.29.0.dev0. Both
reference models' configs were saved as `Hand-tuned` first and restored
afterwards; the engine was left stopped, as it was found.

### Helper model

Downloaded from Settings in about 90 s. Loaded on `rdna4-clav` in 3 min 01 s
the first time and 1 min 10 s the second; answered in 18 s and 15 s.
`response_format` with `json_schema` was accepted -- no fallback needed.

### 27B (`tcclaviger/ThinkingCap-3.8-27B-PARO5`)

Reset to defaults, then Autoconfigure with **Maximum** and nothing serving:
3 min 22 s end to end. Saved and applied with the default ticks.

- **First start: succeeded**, ready in 4 min 21 s, serving requests with
  reasoning separated. **Met.**
- Plan: four cards, 262,144 tokens, fp8 from the card's command, room for 4
  full-context requests (5.74 measured); two cards offered as the narrow
  option.
- **Against `Hand-tuned`:** tool and reasoning parsers, chunked prefill,
  8,192 batched tokens, the sampling override, compilation config, the
  DFlash draft at 7 tokens and all five environment lines **identical** --
  the measurement fingerprint is the same as the hand-built config's. Context
  equal (262,144); sequence cap 16 against 8. **Met.** Differences allowed by
  the criterion: memory fraction 0.90 against 0.92, and `trust_remote_code`
  on (the registration default; the card does not set it).
- No refinement was offered, correctly: the context was already the class's
  full 262,144.
- The container cache paths and `ROCR_VISIBLE_DEVICES` were proposed
  unticked, as intended.

### MoE (`tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`)

Reset to defaults, then Autoconfigure with **Maximum** while the 27B served.

- **Interrupting and restoring:** the 27B was stopped at 22:49:03 and was
  serving again at 22:53:33 -- **4 min 30 s of downtime**, most of it its own
  reload. **Met.**
- Card half: parsers, chunked prefill, batching, sampling override,
  compilation config, the MTP speculative config and the five environment
  lines matched `Hand-tuned`. The `--hf-overrides` rope scaling for 524k
  context was proposed unticked, as intended.
- **Hardware half: failed.** The planner said "does not fit on this machine at
  any width" and left one card at 8,192 tokens, which cannot start. Cause:
  the card's command sets no offload variable, so the estimate counted the
  whole checkpoint on the cards. Started by hand at four cards and 262,144
  tokens *without* the variable, the model logged
  `EngramConfig(cpu_offload=True)`, pinned its 38.74 GiB n-gram table in host
  RAM and served -- the image offloads it by default. **Fixed** on
  `autoconfigure-acceptance-fixes` (rdna4-clav declares the default; a
  checkpoint with a PLE table is estimated as offloading). Not yet re-run on
  compute with the fix.
- Against `Hand-tuned`, beyond the hardware: the hand config also enables
  prefix caching and three `PYTORCH_TUNABLEOP_*` variables, none of which the
  card names; the memory fraction is 0.97 there.
- The helper's sampling and parser readings were rejected as "not in the
  card": the card writes them as `` `temperature=0.7` `` and the helper quoted
  them without the ticks. **Fixed** (quotes are compared without markdown).
  Harmless here, since the command already carried both.

### Also found

- The helper cited a real sentence for a "recommended context" that did not
  mention one. **Fixed:** a quote must now state the value it is cited for.
- The MoE's measurement was recorded for the autoconfigured config, which
  differs from `Hand-tuned`; it applies again after the next hand-tuned start.

## Criteria

| criterion (overview) | status | how |
|---|---|---|
| 27B starts on the first attempt after Save and apply | **met** | compute, above |
| MoE starts on the first attempt after Save and apply | **not met; fixed, not re-run** | the PLE default, above |
| Same parsers, draft, offload flags, env; at least context and sequence cap | **met for the 27B**; MoE met but for the hardware | compute, above |
| The same on a single-card host | **not yet run** | needs the workstation deployed |
| Every row names its source and quotes the card | met | tests; both reviews on compute |
| No helper, or nothing usable: machine and command, with a note | met in tests | |
| A run while a model serves ends with it serving again | **met** | compute: 4 min 30 s |
| Failed-start and refinement notices | met in tests; not exercised on hardware | |
| The feed's phase 19 seeds through the same planner | met | phase 03 |
| `go test ./...` passes | met | |

## Images

Helper loaded and answered on `rdna4-clav` (vLLM 0.29). Unverified:
`radiance`, `rocm`, `rocm-source`, `rocm-cdna`, `gfx906`, `strix-halo`,
`cuda`, `cuda-source`, `gb10`, `xpu`; the two 0.23 images are where the
`guided_json` fallback would matter.

## Earlier status, 2026-09-30

The hardware runs of phase 14 steps 1-9 have **not been done**. They need the
branch deployed to compute and to the workstation (`./setup.sh quick`), the
helper downloaded on each, and real starts of the two reference models, which
takes both machines out of service for a while. That is the operator's call,
not something to do unasked, so it is left for a session where it is agreed.

### Checked so far

- **The card half against the real 27B card.** A local build (no vLLM, no
  helper) fetched `tcclaviger/ThinkingCap-3.8-27B-PARO5`'s card from the Hub
  and rendered the review. Proposed and ticked: `enable_chunked_prefill`,
  `max_num_batched_tokens 8192`, `enable_auto_tool_choice`,
  `tool_call_parser qwen3_coder`, `reasoning_parser qwen3`, the compilation
  config, the `--override-generation-config` from the command,
  `OMP_NUM_THREADS=8`, `VLLM_ROCM_USE_AITER=0`, `GPU_MAX_HW_QUEUES=2`,
  `HSA_ENABLE_INTERRUPT=1`, `HSA_ENABLE_MWAITX=1`, and the installed DFlash
  draft at 7 tokens. Unticked with their reasons: the three container cache
  paths and `ROCR_VISIBLE_DEVICES`.
- **That set equals the hand-built config.** `TestTheTwentySevenBCardReproducesTheHandBuiltConfig`
  applies the rows to a default config and compares the parsers, batching,
  sampling override, draft and environment with what serves on compute:
  identical.
- **The planner on the measured 27B** (`TestPlanFixedWidthUsesTheMeasurement`,
  `TestPlanAllCardsForAModelThatHasNotRun`): all four cards, 262,144 tokens,
  default KV dtype when the card names none and fp8 when it does; two cards
  offered as the narrow option under fp8.
- **The serve-flag probe** on a real listing: `vllm serve --help=all` from the
  local 0.27 radiance image, 370 flags, parsed and recognised.

### Found while checking

- **The workstation's integrated GPU was counted as a card.** The monitor
  listed the Ryzen 9800X3D's iGPU (2 GB) beside the RX 9070 XT (16 GB), and
  the inventory took the smallest card, so every fit on the machine was judged
  against 2 GB. This predated autoconfigure -- the Configure panel's fit table
  had it too. **Fixed the same day:** the monitor now records each AMD GPU's
  gfx architecture from KFD and marks integrated ones, and the inventory
  leaves them out whenever a discrete card is present (a Strix Halo, with
  nothing else, keeps its own). Checked on the workstation: the iGPU is
  reported as gfx1036 and integrated, and autoconfiguring
  `Qwen/Qwen3-4B-Instruct-2507` there plans one card at 58,368 tokens with
  fp8, where before it could plan nothing.
- **The workstation's card is an RX 9070 XT (gfx1201, 16 GB)**, not the RX
  7900 XTX the earlier estimator notes describe. The helper fits it.
- **The MoE card's first command carries `--hf-overrides` with rope scaling**
  for a 524,288-token context. Autoconfigure proposes it unticked, because a
  planned context never exceeds the model's native 262,144; the hand-built
  config does not use it either.

## Criteria

| criterion (overview) | status | how |
|---|---|---|
| 27B and MoE start on the first attempt after Save and apply | **not yet run** | needs compute |
| After refinement: same parsers, draft, offload flags, env; at least the context and sequence cap | **met for the 27B's card half in a test; not yet run on hardware** | `TestTheTwentySevenBCardReproducesTheHandBuiltConfig`; the MoE needs compute |
| The same on a single-card host | **not yet run** | the iGPU that blocked it is fixed; needs the workstation deployed |
| Every row names its source and quotes the card | met | `TestTheTwentySevenBCardReproducesTheHandBuiltConfig` asserts it for every row; the review renders it |
| No helper, or a card with nothing usable: a proposal from the machine and the command, with a note | met in tests | `TestRunWithoutAHelper`, `TestRunWithAnAnswerThatAddsNothing`, `TestAutoconfigureWithoutAHelper` |
| A run while a model serves ends with it serving again, including on helper failure or timeout | met in tests | `TestBorrow*`, `TestAutoconfigureInterruptsAndRestores`; not yet on hardware |
| A failed start and a measured start each produce a notice that changes only what it shows | met in tests | `TestAStartThatRanOutOfContext` and siblings, `TestRefineTo*` |
| The feed's phase 19 seeds through the same planner | met | phase 03 amended `plan/phase-19-fit-handoff.md` |
| `go test ./...` passes with the helper and the Hub faked | met | 16 packages |

## Images

The helper has not yet been loaded on any image. Unverified: `rdna4-clav`,
`radiance`, `rocm`, `rocm-source`, `rocm-cdna`, `gfx906`, `strix-halo`,
`cuda`, `cuda-source`, `gb10`, `xpu`. The first two are on compute and the
workstation and are the ones to try first; whether the 0.23 images accept
`response_format`'s `json_schema` or need the `guided_json` fallback is part
of the same check.

## What the hardware run needs, in order

1. Deploy the branch to compute and the workstation.
2. Save each reference model's current config as a `Hand-tuned` profile.
3. Download the helper on each; run Autoconfigure with nothing serving and
   record the helper's start time and whether the form came back first time.
4. Phase 14 steps 4-9 as written.

# Autoconfigure — acceptance record

Phase 14's record. Phases 01-13 are built on the `autoconfigure` branch with
`go test ./...` passing. What follows is what has been checked against real
cards and real hardware, and what has not yet, criterion by criterion. Nothing
here is summarised from memory: each line says how it was checked.

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

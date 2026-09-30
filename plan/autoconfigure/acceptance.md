# Autoconfigure — acceptance record

Phase 14's record. Phases 01-13 are built on the `autoconfigure` branch with
`go test ./...` passing. What follows is what has been checked against real
cards and real hardware, and what has not yet, criterion by criterion. Nothing
here is summarised from memory: each line says how it was checked.

## Status on 2026-09-30

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

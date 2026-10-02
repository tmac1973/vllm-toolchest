# Remaining Work

## Where things stand, 2026-10-02

Built, merged and deployed; acceptance runs in
`plan/archive/autoconfigure/acceptance.md`:

- **Autoconfigure** (`plan/archive/autoconfigure/`, phases 01-15). A helper model
  reads the card, the hardware half plans width, context, KV dtype and
  concurrency, the review proposes, and the first start refines. Four
  families have gone from download to serving on their first start with its
  settings: Qwen3.5-35B-A3B-FP8, Mistral Small 3.2 (BF16 and a third-party
  FP8), Gemma 4 31B (tcclaviger MXFP416, RedHat FP8-dynamic), and
  Qwen3.8-Flash-Next.
- **Expert offload** on the rdna4-clav image (phase 15 of autoconfigure):
  planned when a 4-bit MoE does not fit on the cards, with the n-gram table
  on NVMe when RAM cannot hold it, always with an fp8 KV cache. Flash-Next
  started on two R9700s from the planner's own config, 84 GB of RAM in use.
- **The recommendation feed** (phases 15-19): exact sizes from the file tree,
  the image's architecture registry, the engine ranking Hub models through
  autoconfigure's planner, the feed behind *Find recommended models* on
  Download Models with a *Made for this image* section, and seeding a model
  downloaded from it.

### Still open

1. **Single-card acceptance.** The last criterion of autoconfigure not yet
   run: the workstation (RX 9070 XT, 16 GB) deployed, the helper on one
   16 GB card, and a model autoconfigured and started there. Waiting on the
   workstation's GPU being free.
2. **Two bugs in the tcclaviger image, to report upstream.** Not yet sent.
   - Expert offload with MTP does not start: `expert_plan.py` sizes the KV
     reserve from the registry's figure without the MTP layer (Flash-Next at
     TP=2: 9,655 B/token reserved, 11,597 measured, 0.09 GiB short).
     Autoconfigure leaves MTP out of offload plans until it is fixed.
   - Any draft model fails on Gemma 4: `vllm/config/tp_padding.py`'s
     `_int_attr` reads `num_key_value_heads` with `getattr(cfg, key, None)`,
     and transformers 5.17 raises `AmbiguousGlobalPerLayerAttributeError` (a
     RuntimeError) for Gemma 4's per-layer config. Fix: catch `Exception`.
     Also, that model card's "see docker-compose.example.yml" names a file
     the repository never had.
3. **Estimates resting on one or two starts**, to firm up as more models are
   autoconfigured, not by calibration runs for their own sake:
   - Sliding-window KV (Gemma 4): global layers exact, sliding layers about
     4,700 B/token per card under; the margin covered it.
   - The 8-bit-and-wider overhead band (0-2.1 GiB a rank) is drawn from three
     models; the one dense model sat at its bottom.
   - The offload planner's minimum expert cache is an assumption no start has
     tested below.

## Direction, decided 2026-09-30

*History: everything below was carried out. Kept for the reasoning.*

The end goal is **autoconfigure**: one action that sets a model up properly,
as llama-toolchest does by having a helper model read the card. Nothing in
phases 15-19 is that -- the recommend feed stops at seeding three values on
download and lists a config optimizer among its non-goals -- so it needs a
phase of its own, not yet written.

What the estimator work is for, in that light: autoconfigure has a card half
(sampling, parsers, template, drafter) that needs no estimate at all, and a
hardware half (width, context, KV dtype, concurrency) that has two sources. A
measurement is exact and exists after one start. A projection exists before
it and only has to be good enough that the first start succeeds.

Decisions taken:

- **Estimator first, but bounded.** Two more fixes to the projection -- the
  drafter's KV, and a band for what a rank consumes beyond its weights -- and
  then it stops. Both are done as of 2026-09-30; see below. No further
  calibration runs are planned for their own sake.
- **Autoconfigure is built, bar its hardware acceptance.** Phases 01-13 of
  `plan/archive/autoconfigure/` are on the `autoconfigure` branch. What remains --
  deploying, loading the helper on real images, and the reference models'
  first starts -- is listed in `plan/archive/autoconfigure/acceptance.md`.
- **The autoconfigure plan is `plan/archive/autoconfigure/`.** It answers the "no
  firm fit" question by planning on the estimate's expected figure, labelling
  the result a first guess, and refining it from the first real start.
- **Measurement may correct a config, by proposal.** After a start,
  autoconfigure offers a corrected profile built from measured figures and
  the operator applies it. This reverses "no automatic re-configuration after
  a run" in `recommend-models-overview.md` and the "what seeding is not"
  section of phase 19; both were amended by autoconfigure's phase 03. Nothing
  rewrites `VLLMConfig` unasked.

## GPU inventory counted an integrated GPU as a card -- fixed 2026-09-30

Found while checking autoconfigure on the workstation: the Ryzen 9800X3D's
iGPU (2 GB) was counted beside the RX 9070 XT, and as the smallest card it set
every fit on the machine. The monitor now records each AMD GPU's gfx
architecture from KFD and marks integrated ones (the list llama-toolchest
keeps), and `gpuInventoryFrom` leaves them out when a discrete card is
present; an APU-only host such as a Strix Halo keeps its own.

Still open: the engine itself sees the iGPU. With every device passed into
the container, HIP enumerates it as a GPU, and a tensor-parallel split counts
ranks from device 0. It has not caused a problem on this machine, whose
discrete card enumerates first, but `setup.sh`'s device selection
(`GPU_DEVICES`) is the way to hide it where it does.

## Estimator: two defects fixed 2026-09-30, and what they leave open

Both fixes are on `todo-handover-estimator-fixes`. The evidence table stays
here because everything below still leans on it.

### The evidence base: four measured starts

Every figure below came from `RunMeasurement`, captured automatically on a
successful start. `act/rank` and `graphs/rank` are what the engine reported per
rank; `act total` is `act/rank x TP`, which is what `MeasuredEstimate` stores.

| model | host | shape | TP | batch | act/rank | act total | projected act | graphs/rank |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-14B-AWQ` | local | dense, 40/40 attn, h5120 i17408 | 1 | 2048 | 0.71 | 0.71 | 0.126 | 2.47 |
| `cyankiwi/Qwen3.5-4B-AWQ-4bit` | local | hybrid + vision, 8/32 attn, h2560 i9216 | 1 | 2048 | 5.4 | 5.40 | 0.073 | 3.74 |
| `tcclaviger/ThinkingCap-3.8-27B-PARO5` | compute | hybrid, 16/64 attn, h5120 i17408 | 4 | 8192 | 1.27 | 5.08 | 0.476 | 1.21 |
| `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` | compute | MoE hybrid, 12/48 attn, h2560 moe640 | 4 | 8192 | 1.46 | 5.84 | 0.249 | 0.49 |

KV per token, measured against projected: 163,872 vs 163,840 (+0.02%), 17,304
vs 16,384 (-5%), 47,836 vs 32,768 (**-46%**), 32,126 vs 12,288 (**-2.6x**).

### One model at two widths: the 27B, 2026-09-30

Same engine (0.29.0.dev0), same config, only the width changed. TP=4 was taken
twice that day, before and after, and repeated to the last digit bar 0.01 on
the graph pool -- so run-to-run noise is not a factor in what follows.

| per rank | TP=2 | TP=4 | total at TP=2 | total at TP=4 |
|---|---|---|---|---|
| weights (`Model loading took`) | 13.40 | 7.05 | 26.8 | 28.2 |
| consumed (weights + non-torch) | 19.89 | 11.67 | 39.8 | 46.7 |
| of which non-torch | 2.65 | 3.08 | 5.3 | 12.3 |
| peak activation | 1.90 | 1.27 | 3.80 | 5.08 |
| graph pool | 0.93 | 1.21 | 1.86 | 4.84 |
| KV bytes per token | 47,820 | 47,836 | | |

Everything under this heading that cites a two-width figure cites this table.

### Fixed: activation is carried across widths one way

It was read three ways. `MeasuredEstimate` stored a total across the ranks,
`RequiredAt` charged the projected figure once at every width, and
`evaluateTP` carried a *measured* total flat onto other widths -- the same
assumption as `RequiredAt`, on the other source, which the handover had not
noticed. The field's comment described a fourth behaviour nothing implemented.

`activationAt` (`internal/models/vram.go`) is now the only reader.
`ActivationBaseGB` is a total at the width it describes -- `MeasuredTP` for a
measurement, one rank for a projection -- and at any other width it is a band:
the total unchanged if everything shards, scaled with the width if everything
is replicated. At the described width the ends meet, so a measurement is still
exact where it was taken. `Fits` takes the high end, and rows projected from a
measurement onto another width are now bands too, with `Uncertain` when they
straddle.

What this does **not** do is make the projected activation right. The band at
TP=4 is `1x..4x` a formula that is 5.6x to 74x low, so it still does not reach
any measured figure. It settles the units, which is what was blocking the
table; the magnitude is the next item.

### Fixed: the graph pool is a band

`graphPoolLowPerRankGB` / `graphPoolHighPerRankGB` = 0.05 / 4.0, per rank, in
`internal/models/vram_fit.go`. A test pins that every measured figure sits
inside it. The headline is the midpoint and `Fits` takes the high end.

The low end started at 0.45 and moved the same day: the MoE, restarted with
`cudagraph_capture_sizes: [4]`, reported 0.07. And the idea that wide runs sit
low because the pool shards is dead -- on the 27B a rank's pool *grew* with
the width, 0.93 at TP=2 to 1.21 at TP=4. What separates the measurements is
more likely what was captured than how the model was split:
`compilation_config` is not read by the estimator, and the smallest figure is
the config that captures one size. **Reading the capture sizes is the way to
narrow this band**, not a width term.

Two consequences, both visible and both intended:

- **Every projected figure is now a range**, so the panel shows "(low-high)"
  beside every unmeasured model, not only those with offload.
- **Tight fits become "uncertain".** One rank's band is 3.95 GB wide, so a
  single-card model within that of the budget no longer gets a verdict or a
  full-context request count until it has been started once. In the golden
  fixtures a model that was starred at TP=1 (25.0 of 27.2 GB) now reads
  24.2-28.1 and the star moves to TP=2. At TP=4 the high end is 16 GB against
  0.28, 1.96 and 4.84 measured.

### Settled by the two-width run: how activation splits

Neither reading alone. All-sharded predicts 5.08 GB at TP=2 and all-replicated
2.54; the engine reported 3.80, the midpoint of the band `activationAt` draws
between them. As a line through the two points it is 0.64 GB held by every
rank plus 2.52 GB shared out among them -- about half replicated at TP=4. One
model, so the band stays a band, but a test now pins that it holds the
measurement in both directions.

### Open: the projected activation's magnitude

Still wrong, still understating, and deliberately not refitted -- see the next
section. For the 27B the projection is 0.48 GB against 3.80 and 5.08 measured,
so the band's `1x..TPx` does not reach either. The choice is between a band
wide enough to hold the measurements (per rank: 0.71 to 5.4 GB, which would
put most single-card fits into "uncertain") and the present formula with its
known bias. Not decided.

### Fixed: what a rank consumes beyond its weights

The two-width run found the projection's largest miss, and it was neither term
the handover named. For the 27B the band was 36.4-44.8 GB at TP=2 and
36.5-53.7 at TP=4 against 57.1 and 68.3 needed: the high end 12 to 15 GB short.

What a rank consumes beyond its weights was not modelled at all -- non-torch
memory (2.6 to 3.1 GB a rank) and what the allocator reserves over what it has
allocated. Measured per rank: 4.62 and 6.49 on the 27B, 5.32 and 5.50 on the
MoE. `rankOverheadLowGB` / `rankOverheadHighGB` = 4.5 / 6.5 in
`internal/models/vram_fit.go`, charged from two ranks up.

With that and the drafter's KV the 27B projects 47.9-60.3 and 57.0-82.2, which
hold both measurements, and the MoE's headline is 110.0 against 113.9 measured
where it used to be 83.3. Two tests pin these.

What it costs, and what is still weak:

- **No large model gets a firm verdict at TP=4 before it has run.** Every band
  at its pessimistic end at once adds 42 GB at four ranks (16 of graph pool, 26
  of overhead), and both checkpoints that serve on compute now read
  "uncertain" there. Three tests that asserted a firm fit now assert only that
  the model is not refused. The rule that `Fits` means every worst case at
  once was written when offload was the only band; with four it is strict.
  Whether to keep it is a question for the autoconfigure phase, which has to
  pick a width from these rows.
- **A single rank is charged nothing**, on the strength of one start on
  another host that consumed barely more than its weights. Unmeasured on
  compute. If it is wrong, single-card verdicts are optimistic by a few GB.
- **The band is two models on one host and one engine build family.** It is
  exactly as wide as what was seen.
- **`projectWeights` still carries the old assumption** on the measured path:
  it splits *consumed* memory with the surcharge calibrated on weights, 1.06 GB
  a rank where the two-width line says 3.45. Projecting TP=4 onto TP=2 gives
  44.6 against 39.8 measured, which errs safe; TP=2 onto TP=4 gives 43.4
  against 46.7, which does not. Not fixed: it only affects the rows beside a
  measurement, never the measured row.

### What not to do: refitting activation from a formula

`activationBaseGB` (`internal/models/vram.go`) models
`tokens x hidden x 2 bytes x 6 buffers` plus logits. It is wrong by 5.6x, 10.7x,
23.4x and 74x across the four models, and the shape is wrong rather than the
coefficients:

- The 14B and 27B share `hidden_size` **and** `intermediate_size` exactly, and
  differ 1.8x in per-token activation. No function of those two fields can fit
  both.
- Adding an `intermediate_size` term was tried on paper and rejected: closing
  the 14B's gap alone needs ~8.8 intermediate-width buffers stacked on the
  existing 6 hidden-width ones, which is a fudge factor wearing a structural
  costume.
- The engine's "peak activation" includes the encoder cache, Mamba conv and
  state workspaces, and Triton scratch. The 4B's 74x outlier is the only
  multimodal model in the set, and `VisionMeta` (`registry.go`) records only
  `IsVisionModel bool` -- no vision dimensions are parsed, so a vision term
  cannot be written today even if the base shape were right.

Four points across dense, hybrid, MoE and multimodal, fitted with free
parameters, would be indistinguishable from coincidence. Prefer a band and let
measurements narrow it.

### KV on hybrids: one hypothesis refuted, a better one found

The 27B projects 32,768 B/token and measures 47,836; the 4B, also hybrid, is
within 5%. The question was why the 4B escapes.

**Refuted: state replicated on every rank.** Dividing the excess by
linear-attention layers, ranks and `hidden_size` gave the same constant for
the 4B and the 27B to 2.4%, which predicted ~40,100 B/token for the 27B at
TP=2. It measured 47,820 there against 47,836 at TP=4. KV per token does not
move with the width, so the agreement was a coincidence of exactly the kind
the section above warns about -- and it cost one start to find out.

**What it mostly is: the drafter.** The two models that miss badly are the two
running speculative decoding, not the two on compute. The 27B's DFlash drafter
is 5 attention layers of 8 KV heads at head dim 128, which at fp8 is
`2 x 5 x 8 x 128` = 10,240 B/token in the same pool. 32,768 + 10,240 = 43,008,
which is 10% under the measurement where attention-only was 46% under. The
4B and the 14B are assumed to have run without one -- their configs were not
re-read for this -- and are 5% under and exact.

So the remainder on a hybrid is 5-10%, and the log says what it is:
`Setting attention block size to 1648 tokens to ensure that attention page
size is >= mamba page size`, under `Mamba cache mode 'align'`. The recurrent
layers keep state in the pool at matched page sizes. Not sized here.

**Done 2026-09-30:** `draftKVPerToken` (`internal/models/draft.go`) adds the
drafter's own KV when `SpeculativeConfig` names a draft on this disk, read
from the draft's `config.json`. The MoE's MTP drafter has no separate
checkpoint to read, so its 2.6x stays open, as does the 5-10% of recurrent
state on any hybrid. A draft named by Hub repo id is not on disk and adds
nothing.

## VRAM estimator: what compute measured, 2026-09-16

Checked against the running instance after the rework landed. Three fixes came
out of it, and one judgement was reversed on better evidence.

- **The PLE residual works on the real checkpoints.** On the GPTQ checkpoint it
  puts the table at 43.6 GiB against 47.68 measured, and TP=4 at 17.9 GiB per
  rank against the engine's own 19.07. It had looked inert locally only because
  no checkpoint on the workstation carries such a table. It is now the estimate,
  banded rather than trusted flat, and a verdict is offered again. The residual
  is not all table, though — see the 0.90 share below.
- **`--expert-offload-mem` is a ceiling, not an amount.** A run configured with
  46 was measured moving 18.72 GiB. It bounds the optimistic end and never sets
  a figure. `--expert-cache-gb` is a different quantity again — a cache staged
  *on* the card, so it adds to what a rank holds. Parsing both into one field
  had the second silently overwrite the first.
- **`--expert-cache-gb` is assumed to be GPU-resident.** That reading fits the
  flag names but has not been confirmed against vLLM's source or its memory
  accounting. If it is actually a host-side cache, it is being added to the
  wrong side of the ledger — worth 5.5 GiB per card on the MXFP4 checkpoint.
- **Block-quantized FP8 had no structural size.** `bits` is unset in those
  configs, so bytes-per-param came back "unknown" and the structural figure was
  zero — which would make the offload residual the entire checkpoint. FP8 names
  its own width, so it is now read as 8 bits.

## VRAM estimator: the figure is a total, not a per-card share

Reframed 2026-09-16 after the panel reported 18.5 GB for a checkpoint that is
108.5 GB on disk. The arithmetic was right and the question was wrong.

It had been reporting **weights on one card**, which excludes the KV cache —
the largest consumer in vLLM, and the one that scales with the settings a user
is actually editing. Worse, it invited comparison with `rocm-smi`, which shows
the whole allocation: vLLM claims `gpu_memory_utilization` of every card at
startup regardless, so the two numbers could never agree and the estimate
looked broken whether or not it was.

The estimate is now **the total GPU memory to load the model as configured**,
summed across every card it is split over: weights after offload, KV cache at
the configured context, graph pools, on-card caches and activations. It is
computed from the configuration alone, so it means the same thing on any host
and it moves as the model is configured — which is the entire purpose.

Comparing it to a particular machine is a separate step, and lives in the
config panel beside the settings that move it. The card column shows one
number and no verdict: fit commentary belonged next to the controls, not in a
column whose job is to report a quantity.

Worth remembering if this is ever revisited: "which is more useful for
comparing models" was the wrong axis to optimise. Nobody was comparing models.

## VRAM estimator: calibration still owed on compute

What is still owed:

- ~~**The estimate is compared against free memory, not card size.**~~ It
  confused someone, and the someone was foreseeable. Written as a warning here
  before the change shipped, then shipped unheeded: with the engine loaded, the
  panel subtracted vLLM's own 28.68 GiB per card and reported "of 12.3 GB
  available" beside a model that had been serving for twenty-four minutes.
  Fixed 2026-09-18 — while the engine is up the cards count as empty, because
  the question the panel asks is whether the model *could* be started and the
  engine is not its own competition. With the engine down, resident memory
  still counts, which is the leaked-worker case that motivated the change.
  The lesson worth keeping is about the note, not the code: a risk recorded in
  todo.md and not acted on is indistinguishable from one nobody noticed.

- ~~**Watch a real startup.**~~ Done, four times over -- see the evidence table
  under the estimator section at the top. Weights land within 3% and KV within 5% on dense and on
  one hybrid; KV is 46% low on another hybrid and 2.6x low on the MoE, which is
  now its own open question. The full-context request count still has not been
  checked against anything.
- **The KV figure counts one sequence at `max_model_len`.** That is the
  minimum to serve the configured context at all. The panel reports separately
  how many full-length requests the leftover buys, which is the number to tune
  `max_num_seqs` against. Whether the headline should instead assume full
  concurrency is a judgement that can be revisited once the startup figures
  have been compared.
- **The CUDA-graph pool is a flat 0.9 GiB** and activation assumes vLLM's
  2048-token default chunk when `--max-num-batched-tokens` is unset. Both were
  stand-ins for a measurement nobody had taken; the measurements exist now, and
  both stand-ins are wrong by more than the band they imply. The pool is a
  band as of 2026-09-30; the activation stand-in remains, see "the projected
  activation's magnitude".
- **Vision-tower parameters are not counted.** Draft weights now are: PR #33
  added `VRAMEstimate.DraftGB`, counted once at its size on disk, and
  `RequiredAt` says why it does not scale with the width. The vision tower is
  still absorbed into the residual and attributed to the PLE table, and
  `VisionMeta` records only `IsVisionModel bool` -- no dimensions -- so
  counting it means parsing `vision_config` first.
- **The expert share is fitted to one measurement.** 18.72 GiB moved against a
  46 GiB ceiling, which is 30% of that checkpoint's idle expert weight. That
  30% is now the centre of the estimate with a ±50% band around it, because one
  point cannot support anything tighter. A second offload configuration —
  different ceiling, different model — would either corroborate it or show it
  for the coincidence it might be. Until then this is the weakest number in the
  estimator, and the band is wide on purpose.
- **The projected path is wrong in two terms, both in the same direction.**
  Re-measured 2026-09-22 on the 4B AWQ model on one RX 7900 XTX, against what
  the formula projects for the configuration it is running under:

  | | projected | measured | error |
  |---|---|---|---|
  | weights | 3.78 GB | 3.9 GB | −3% |
  | KV per token | 16,384 | 17,304 | −5% |
  | activation | 0.073 GB | 5.4 GB | **74x low** |
  | graph pool | 0.9 GB | 3.74 GB | **4.2x low** |
  | total | 6.75 GB | ~16.2 GB | **2.4x low** |

  Weights and KV are within 5%. The estimate is not broadly unreliable: it is
  wrong in two specific terms, and both make the model look smaller than it
  is, so a fit verdict built on it is optimistic rather than merely noisy.

  The graph pool is a flat constant and the engine reports the real figure on
  every start, so it is correctable. Activation is not a calibration problem:
  `activationBaseGB` models `tokens x hidden x 2 bytes x 6 buffers` plus
  logits, which is 63 MB here, while the engine's 5.4 GB "peak activation"
  covers the encoder cache (16,384 tokens with image items on this model),
  Mamba conv and state workspaces, and Triton scratch -- none of which scale
  with `hidden_size`. The formula is measuring a different quantity, so no
  coefficient fixes it.

  **Correcting the record.** An earlier version of this entry gave the
  projection as 13.9 GB total with 73,728 B/token, and concluded the errors
  had no consistent direction. `internal/models` has not changed since, and
  the same code now yields 16,384 B/token -- exactly `2 x 8 x 4 x 256 x 1`
  from this model's own config -- and 6.75 GB total. The other three figures
  in that table all followed from the KV one (73,728 B/token over a 131,072
  window is 9.0 GB of KV, which is the whole of the 13.9). Reproducing 73,728
  needs 36 KV layers, 18 KV heads or a head dim of 1152, and this model has 8,
  4 and 256; the likeliest cause is an `HFConfig` that had not yet been
  re-read, but it has not been reconstructed and is not asserted. The figure
  overstated the error at the one term the formula gets right, and the
  conclusion drawn from it -- leave the constants alone -- was wrong.
- **A measurement cannot be captured on vLLM 0.23 at all.** Found on a real
  start 2026-09-21, after this box moved from `rocm-source` (0.27.2.dev0) to
  AMD's prebuilt `rocm` image (0.23.1.dev1). `RunMeasurement.Complete()`
  requires `ConsumedGB`, and 0.23 never prints the line it comes from:

  | line | 0.27 | 0.23 |
  |---|---|---|
  | `Model loading took N GiB` | yes | yes |
  | `Available KV cache memory` | yes | yes |
  | `GPU KV cache size: N tokens` | yes | yes |
  | `Actual usage is … for consumed memory` | yes | **no** |
  | `CUDA graph pool memory:` | yes | **no** — says `Graph capturing finished in N secs, took X GiB` |
  | `--kv-cache-memory=` | yes | **no** |

  So `SetMeasurement` refuses, the previous engine's measurement stays on
  screen looking authoritative, and the only sign is one debug line. The
  panel read 10.85 GiB of KV while the live engine held 15.72. The `rocm`
  variant now points at AMD's unified ROCm 10 / vLLM 0.27.0 image, which fixes
  it there — but `rocm-cdna` pins v0.23.0 and `gfx906` pins v0.23.1rc0, so
  neither can be measured. Teaching `internal/advice` the 0.23 spellings is
  the outstanding half, and the graph-pool line is the easy one.
- **KV bytes per token survives an engine change.** The same model and dtype
  measured 17,301 B/token on 0.27.2 and 17,302 on 0.23.1 — across four minor
  versions and a whole different image. It really is a property of the
  checkpoint and the cache dtype, not of the engine, which is what makes it
  safe to re-scale to another context length. The pool *size* moved a lot
  (10.85 → 15.72 GiB) and the graph pool moved (3.58 → 3.21); those are
  engine behaviour and must not be carried across.
- **The PLE share is corroborated but only twice.** 0.890 and 0.913 of the
  residual across two checkpoints, hence 0.90 ±5%. The gap between residual and
  table is presumably the MTP draft weights and the vision tower; if those were
  counted structurally the residual would *be* the table and this correction
  could go away entirely. That is the better fix and it is not done.

What is pinned by tests: the MoE parameter count against four checkpoints
(Qwen3-30B-A3B → 30.53B, Qwen3-Next-80B → 79.04B, Mixtral-8x7B → 46.7B with
12.9B active, and the Qwen3-Next structural figure reproducing its 45.9 GiB of
safetensors to within 0.1 GiB), and the compute checkpoint's TP=4 verdict with
the per-rank band containing the engine's measured 19.07 GiB.

## Recommend feed (phases 15-19): defects found before building

**Resolved 2026-10-01, when the phases were built** (each phase file has an
*As built* section). The feed does not wrap a candidate for `Fit` at all: it
describes it as registration would (`models.Describe`) with its weight bytes
from the file tree in `TotalSizeBytes` before the estimate, and plans it with
`models.PlanFit` at `ContextMax` -- autoconfigure's planner -- so the width,
context and request count stated are what autoconfigure would configure, and
what seeding writes. The circularity below goes with it: the context is the
planner's answer, not an input. Figures are still projections, and the help
text says they are estimates the first start refines.

Read while reviewing `plan/archive/phase-15..19`, 2026-09-22. None of the five phases
had any code then; these were in the documents:

- **Phase 17 step 12 does not typecheck.** It says call
  `models.ParseHFConfig`, then `models.EstimateVRAM`, then `models.Fit`. But
  `EstimateVRAM(m *Model, envPairs []string)` takes a `*Model`, not an
  `HFConfig`, and `Fit(est, c VLLMConfig, inv)` takes a `VLLMConfig`. The plan
  never says what synthetic `Model` and `VLLMConfig` a Hub candidate is wrapped
  in, and that is not a detail: `MaxModelLen` and `KVCacheDtype` drive
  `KVAtContextGB` and therefore `SpareGB`, which every objective, the headroom
  figure and the phase-19 seed all read.
- **Affordable context is circular unless `MaxModelLen` is 0.** `Spare =
  Available - (W + O + perToken x ctx)`, so `affordableTokens` falls 1:1 with
  whatever context the synthetic config names. Ranking with `MaxModelLen = 0`
  makes `affordableTokens` mean "what this host can hold", which is the honest
  figure phase 17 says it wants -- at the cost that `Fits` then means "will
  load", not "will serve the configured context". The feed has to say which.
  Whatever is chosen, phase 19 *seeds* `KVCacheDtype`, so ranking and seeding
  must read the same source or the seed contradicts the ranking behind it.
- **Step 12 sets `CheckpointGB` too late.** It overrides the field after
  `EstimateVRAM` returns, but `EstimateVRAM` returns early with `Unknown =
  true` when `TotalSizeBytes` is 0 and there is no structural fallback, and
  `Fit` short-circuits on `est.Unknown`. The Hub's weight bytes belong in the
  synthetic `Model.TotalSizeBytes` *before* the call, which also makes the
  override unnecessary.
- **Every feed figure is projected, never measured.** `evaluateTP`'s measured
  branch requires `est.Source == SourceMeasured`, which a model that has never
  run cannot have. So the feed ranks on the projected path. The graph pool
  there is a band since 2026-09-30 and `Fits` takes its high end, so that term
  no longer flatters; activation still understates by the factors in the
  evidence table and KV still understates on an MTP drafter, but the
  per-rank overhead is counted since the same day, and on the two models
  measured the band now holds what the engine needed. The bias has largely
  gone and the price is width: the feed has to decide what an `Uncertain`
  candidate is, because tight single-card fits and nearly every large model
  at TP=4 now are. A Hub candidate also has no draft on disk, so the drafter's
  KV is never in a feed figure. Phase 18 step 11's help text acknowledges
  the figures are estimates; it should also say they are ranges.

## Second VRAM estimator in the HuggingFace client

**Removed 2026-10-01** (recommend phase 15): the detail panel shows the exact
weight size from the file tree, and the formula estimators are gone. What
follows was the state before.

`internal/huggingface/client.go:564` estimated VRAM for models not yet
downloaded, from the repo's advertised parameter count and file sizes. It is a
different question with worse data — there is no local config to parse — and it
was deliberately left alone. It does not know about MoE, offload, or tensor
parallelism. The download page now at least compares its figure against the
smallest card and the whole host rather than against GPU 0.

## ROCm Dockerfile validation
- Rebuild `rocm-source` against `scripts/rocm_vllm_patches.py`. It replaced a
  script vendored from someone else's repo, and carries five edits where that
  one had ten: forcing ROCm detection, pinning `device_type`, and the no-GPU
  GCN-arch fallback are gone as unnecessary, and the `mwaitxintrin.h`, INT8 and
  `HIP_FOUND` patches match nothing upstream any more. `--lenient` gets past an
  edit whose anchor has moved.
  First build, 2026-09-14 (gfx1100): vLLM compiled and installed
  (0.27.2.dev0+g6e448d0ea.rocm723), so the five dropped edits cost nothing at
  build time. One casualty: the final `import vllm` smoke test, which resolves
  the platform and died in `_GCN_ARCH`. Measured in the cached build layer:
  amdsmi enumerates the host's cards with no /dev/kfd, so ROCm is selected and
  rocm.py is imported; `VLLM_TARGET_DEVICE=cpu` does not avoid it (this build
  has no CPU short-circuit) and nor does hiding the GPUs. The check now
  tolerates exactly that error. Consequence to keep in mind: `import vllm`
  cannot work in this image without a usable GPU, which the dropped
  `_get_gcn_arch` patch used to provide.
  The script header now records what was measured rather than the pre-build
  reasoning. Note that editing that file invalidates the cached vLLM build,
  since the Dockerfile COPYs it: the next `rocm-source` build recompiles.
  Served on RDNA3 (RX 7900 XTX) 2026-09-14 with an AWQ 4-bit model. Still
  unproven on RDNA4. Not a defect, but worth knowing: an FP8 checkpoint fails
  on RDNA3 in `torch._scaled_mm`, which needs MI300+ or Ada — the card has no
  FP8 matmul, so no image can serve it.
- **Start-up cost is per image, not per start.** Measured on gfx1100
  2026-09-21, same checkpoint and flags throughout (4B AWQ,
  `max_model_len=131072`, fp8 KV):

  | image | vLLM | init engine | of which compilation |
  |---|---|---|---|
  | `rocm-source` | 0.27.2.dev0 | 177.46 s | 140.19 s |
  | `rocm`, 0.23 base | 0.23.1.dev1 | 56.66 s | 44.74 s |
  | `rocm`, ROCm 10 base, **first** start | 0.27.1.dev5 | 200.27 s | 53.23 s |
  | `rocm`, ROCm 10 base, **restart** | 0.27.1.dev5 | **31.66 s** | 0.44 s |

  A restart is 6.3x faster, and the log says why outright:
  `Directly load AOT compilation from path
  /data/cache/vllm/torch_compile_cache/torch_aot_compile/ab8999fd…`. That
  directory is 612 MB, sits on the `vllmctl-data` volume, and therefore
  survives a rebuild -- but it is keyed to the engine build and model config,
  so switching base image invalidates it and the next start pays in full, once.

  Where the warm 31.66 s goes, from the timestamps:

  | phase | cost |
  |---|---|
  | loading weights | 3.0 s |
  | reconstructing AOT artifacts (13 artifacts, 33 submods) | **~21 s** |
  | `torch.compile` | 0.44 s |
  | profiling / warmup | 0.6 s |
  | graph-memory profiling and KV sizing | ~3 s |
  | CUDA graph capture (3.74 GiB) | 6 s |

  So even a warm start is dominated by the compile cache -- the *read* side of
  it. Deserializing 13 artifacts takes two thirds of the start, which is the
  thing to attack if 30 s is still too slow; profiling and graph capture are
  not where the time is.

  Two corrections to what this entry said an hour earlier, both mine. It
  claimed ~147 s was unattributed and blamed profiling, graph capture or
  warmup: wrong, it was cold-cache AOT compilation, and the phase table above
  is what attributing it actually looks like. And it said a warm start "has not
  been measured" while asserting startup was simply slow -- stated as a caveat,
  but load-bearing, and it took one restart to settle. Measure before
  characterising.
- End-to-end test on RDNA4 hardware (9070 XT)
- Validate flash-attention triton backend on gfx1201
- `plan/archive/` still refers to `scripts/patch_vllm.py`, which is gone. Left
  alone deliberately: the archive records what was decided at the time.

## Benchmarking phase (Phase 6)
- Implement benchmark API endpoints (currently stubbed)
- Token throughput measurement (tokens/sec, TTFT, ITL)
- Store and display benchmark history per model
- Compare results across quantization methods

## Chat UI
- vLLM does not ship a chat interface
- Build or embed a lightweight chat UI
- Support streaming responses via SSE
- Tool use / function calling display in chat

## Model served name mismatch
- vLLM uses full filesystem paths as model identifiers
- Clients need to know the exact served name to send requests
- /v1/models proxy now forwards to vLLM when running (shows actual paths)
- Consider adding --served-model-name flag support in per-model config

## Polish
- Show friendly error on service page when vLLM crashes mid-request
- Model delete confirmation should check if the model is currently running
- Improve error messages when model files are corrupted or incomplete
- Loading states for long operations (model scan, large downloads)
- ~~Wire `startup_timeout_s` config through to `internal/process/manager.go`~~
  Done 2026-09-15. It was worse than an unread setting: the poll returned at
  the deadline, so a model that became healthy a moment later stayed marked
  failed until someone restarted it, with the proxy refusing requests the
  engine was answering on its own port. The watch now continues while the
  process is alive, the health check has its own timeout, and the default is
  30 minutes — a 125B MoE cold start measured 9m26s on four R9700s.
- Relocate Triton/Inductor kernel caches onto the `/data` volume (e.g. `TRITON_CACHE_DIR=/data/cache/triton`, `TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor`) so first-boot kernel compilation only happens once per image rather than every container recreate

## Host prerequisites

- **Locked memory.** A model that offloads experts or the n-gram table to
  system RAM pins it, and rootless podman cannot raise `memlock` above the
  invoking user's hard limit — a compose file asking for `-1` is clamped in
  silence. Fedora's default is 8 MiB, which produced
  `PLE offload: locked 0.0 GiB, FAILED to lock 47.7 GiB` on compute while the
  identical compose file worked on a host carrying
  `* hard memlock unlimited` in `/etc/security/limits.conf`. `setup.sh` now
  warns at install time, and the Quadlet unit carries `LimitMEMLOCK=infinity`
  so the systemd path is not capped either — but neither can substitute for
  the host setting. Verified 2026-09-15: compute `ulimit -H -l` = 8192.

## Setup script enhancements
- Add `enable` / `disable` commands for auto-start (from llama-toolchest)
- Support `setup.sh update` to pull latest image — and note that the staleness
  it should report is now only "the author published a newer tag", since a
  manifest bump itself takes effect on the next install (below).
- Validate GPU driver availability during install

### Fixed 2026-09-14: a manifest bump could not take effect

`variant_base_ref` resolved the base image from `VLLMCTL_BASE_IMAGE` in the
environment, then from `.env`, then from the manifest. `.env` is written by
`setup.sh` itself, so the first install recorded the manifest's image there and
every later install preferred that copy: bumping `rdna4-clav` from 28.02.2 to
28.04.9 was read, ignored, and then overwritten in `.env` with the value it had
just declined to use.

It failed silently. `ensure_base_image` exports what it resolves and an export
beats `.env` for compose, so the build ran on the stale base while `.env`
claimed the new one — and `Dockerfile.prebuilt`'s version assertion is blind to
it whenever two tags share a vLLM build, which 28.02.2 and 28.04.9 do
(`0.27.0.dev0+g55c98e370a`, four releases apart). It surfaced only as
`vllm: error: unrecognized arguments: --enable-expert-offload …` on a flag the
older tag never had.

`.env` is no longer an input. The environment variable remains the one-off
override, and a `.env` pin that disagrees with the manifest now warns.

## Multi-GPU testing
- Test tensor parallelism with TP=2 and TP=4
- Validate GPU memory utilization settings across multiple GPUs
- Document multi-GPU configuration

## Agent CLI tool
- Port llama-toolchest's cmd/agent to work with vLLM
- CLI tool for piping prompts to the local vLLM instance
- Support tool use / function calling from the command line

## Carried over from the UI parity plan

Its six phases are all done and it is archived; these three were listed there
as "cross-cutting, unscheduled" and never landed. Verified still outstanding
2026-09-10.

- `internal/broadcast` — llama-toolchest has a mutex-protected fan-out with
  replayed history, shared by the log and download streams. Here the subscriber
  handling is ad-hoc in `downloader.go` and `sse.go`; consolidating removes the
  duplication and gives a new subscriber the recent backlog rather than
  whatever happens next.
- Capabilities endpoint — `/api/models/{id}/info`, plus a `meta` extension on
  `/v1/models`, so a client can self-configure in one round-trip.
- `/api/ps` — process listing.

## Live Performance: capture streaming requests

The panel only measures non-streaming requests routed through vllmctl's port.
Most chat clients stream by default, so for many users it never accumulates
anything — the panel now says so, but saying so is not the same as working.

Capturing streaming would mean reading the SSE tail for the final `usage`
object rather than buffering a whole response body. It would also give a real
time-to-first-token, which the non-streaming path cannot measure at all:
`AvgPromptTPS` is currently always zero because the proxy sees one response,
not a first token and then the rest.

See `internal/api/proxy.go` — streaming requests are forwarded untouched at the
top of the capture path.

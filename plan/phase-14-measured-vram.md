# Phase 14: Measure, then stop estimating

**Status:** planned 2026-09-16, not built.

## About this document

Written after a successful start was read against the estimator for the first
time. It is a reversal, not a refinement: the previous approach was not wrong
by a coefficient, it was wrong about where numbers should come from.

## 1. The evidence

One transcript, 2026-09-16, `Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` at TP=4 on four
R9700s. Everything below is the engine's own report against what the estimator
predicted.

| term | measured (total) | estimated | |
|---|---|---|---|
| weights | 76.28 | 76.3 | exact |
| PLE table in host RAM | 38.8 | 39.2 | 1% |
| non-torch / allocator | 22.0 | **not modelled** | — |
| peak activation | 5.84 | 0.25 | **23× low** |
| CUDA-graph pool | 1.96 | 3.6 | 1.8× high |
| KV cache | 19.48 | 3.0 | **6.5× low** |
| **total** | **125.6** | **83.3** | **34% low** |

Sort those by how each figure was arrived at and the pattern is not subtle:

- **Calibrated against a measurement** — weights (structural count checked
  against safetensors and against `model loading took`), the PLE residual
  (fitted to two checkpoints). Both right.
- **Derived from first principles** — activation from batch shape, KV from
  attention-layer counting, graph pool from a constant. All wrong, two of them
  by more than a factor of two.

Three days of this produced the same lesson three times: rules and constants
written from reasoning fail, and figures taken from the engine survive. The
parser told the same story on the same day, matching one line in eight because
its patterns were written from what vLLM was assumed to print.

## 2. The reframe

**Stop deriving what the engine reports.**

For a model that has been run, the estimate should *be* the measurement.
Estimation is for models that have never been started, and should say so.

Two consequences worth stating plainly, because both are reversals:

- The KV figure is not to be fixed by a better formula. Attention-layer
  counting is *correct about attention* and wrong about the pool — this hybrid
  allocates recurrent state in the same pool at matched page sizes (`6 KV cache
  group(s)`, `attention block size to 832 tokens to ensure that attention page
  size is >= mamba page size`), plus the MTP draft's own cache. Modelling that
  properly means reimplementing vLLM's allocator. The engine already prints the
  answer.
- The activation term is not to be re-tuned. It was a constant ladder, then a
  shape formula, and it has been wrong both times. Third guess is not the
  answer.

## 3. What one successful start teaches

```
Model loading took 19.07 GiB memory and 58.2 seconds
memory after profile: torch reserved 22.7 GiB, ... non-torch 2.51 GiB
Actual usage is 24.58 GiB for consumed memory (weights + non-torch),
  1.46 GiB for peak activation, and 0.49 GiB for CUDAGraph memory
Available KV cache memory: 4.87 GiB
GPU KV cache size: 651,081 tokens, Maximum concurrency for 262,144 tokens: 2.48x
PLE offload: locked 38.8 GiB of PLE weights in RAM
```

That fourth line alone carries all three terms the estimator gets wrong.

And one derived quantity matters more than any of them:

```
KV bytes/token = KVCacheGB x TP x 2^30 / KVCacheTokens
               = 4.87 x 4 x 2^30 / 651,081 = 32,126
```

which is a property of the model and `kv_cache_dtype` — near enough independent
of context length and tensor-parallel width. **From one run, the "what if I
change max_model_len" question becomes exact arithmetic** rather than the thing
this estimator has been worst at.

## 4. Design

`VRAMEstimate` gains a source, and the panel says which it is:

| source | when | how |
|---|---|---|
| **measured** | this model has been run at a comparable config | engine figures, with KV scaled to the configured context |
| **projected** | never run, or the config moved in a way that invalidates | today's structural estimate, labelled a projection |

**What holds and what moves.** A measurement is valid for the configuration it
was taken under. Recorded alongside it is a fingerprint of the fields that
change the answer:

- `TensorParallelSize` — changes per-rank everything
- `KVCacheDtype` — halves or doubles KV per token
- `Env` + `ExtraFlags` — offload, and therefore weights
- quantization

`MaxModelLen` is deliberately **not** in the fingerprint: varying it is the
whole point, and the KV term scales with it exactly. `MaxNumBatchedTokens`
changes activation and is a judgement call left for §7.

When the fingerprint differs, the measurement is shown but marked stale rather
than silently reused or silently discarded.

**Storage.** A field on the `Model` record, beside `VRAMEstimate`. Not on
`VLLMConfig`, which must stay comparable with `==`. Absent means never
measured; no schema bump, since an old record simply has none.

**Capture.** `process.Manager` already accumulates these as the log streams.
The fiddly part is persistence: the manager owns the measurements, the API
layer owns the registry, and the write wants to happen once on the transition
to running.

*Built 2026-09-17.* Two things it turned out to hinge on, neither obvious from
here:

- **There are three launch paths, not one.** `startModel` covers the Start
  button and auto-start; `handleServiceRestart` calls `process.Restart`
  directly and bypasses it; `jobs_env.go` is the benchmark sweep. A watch hung
  on `startModel` alone would have missed every restart -- which is the more
  interesting case to capture, being what follows a config change.
- **The sweep is excluded deliberately.** It serves one model at several
  context lengths and batch sizes in succession, so recording from there would
  attribute figures to a configuration nobody chose and let the last step win.
  That is why the watch is an explicit call at two sites rather than something
  buried in the process manager where it would catch all three.

The write is narrow (`Registry.SetMeasurement`, in the idiom of `UpdateConfig`)
because it happens from a background goroutine: a whole-record upsert would
race the config panel's autosave and quietly undo an edit made while the engine
was still coming up. Incomplete runs are refused rather than stored half-filled.

## 5. Parser corrections, first

Prerequisites, all found by reading the real transcript:

- Numbers carry thousands separators: `262,144`, `651,081`. `Atoi` rejects
  them, so `ConcurrencyTokens` came back 0.
- `GPU blocks: N, CPU blocks: M` **does not exist**. It is
  `GPU KV cache size: N tokens`. That rule was invented whole.
- `Actual usage is ... consumed ... peak activation ... CUDAGraph memory` —
  the line that matters most, and there is no rule for it at all.
- `Free memory on device (31.23/31.86 GiB)` on a *successful* start has no
  `cuda:N`, so the rule written against the failing form does not match it.
- `CUDA graph pool memory: 0.49 GiB (actual)` — the measured graph pool.
- **A false positive shipped in `main`.** The CUDA-graph profiling note
  contains the word "increase", so every healthy start now raises an error
  claiming the engine ran out of room. Yesterday's fix for a backwards rule
  introduced it.

## 6. What this is worth

The config panel gets what it was always supposed to have: **estimated beside
measured**, and a context-length control whose effect on the KV cache is read
off a real allocation rather than guessed.

It also ends a loop. Every round so far has adjusted a coefficient from one
reading and been found wrong by the next log. Calibrating from starts the user
actually performs replaces that with something that improves on its own.

## 7. Out of scope

- **Acting on any of it.** Still no auto-apply. Unchanged from phase 13, and
  the evidence for that caution has only grown.
- **Re-tuning the projected path's constants.** Deliberately frozen. If a model
  has been run, use the measurement; if not, the figure is labelled a
  projection and its weaknesses are documented rather than papered over.
- **Re-projecting a measurement across a changed tensor-parallel width.** The
  arithmetic is plausible and unvalidated, which is how this went wrong before.
  Mark stale, and wait for a run at the new width.
- **`MaxNumBatchedTokens` scaling of activation.** Same reasoning. One
  measurement does not establish a slope.

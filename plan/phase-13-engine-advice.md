# Phase 13: Engine advice

**Status:** §1–§5 built 2026-09-16 on the `engine-advice` branch. **§6 (display)
is not built** — nothing surfaces any of this in the UI yet, so the capture
runs on every start and is visible only through the API layer. That is the next
piece of work, and the reason to do it is §6's second paragraph: the
estimated-against-measured column is the whole point of having captured it.

## About this document

The first slice of the "optimizer" Tim asked for: a feature that intelligently
populates a model's launch config. This document covers only the parser and
what it captures. Acting on what it captures is a later phase, deliberately —
see §7.

## 1. The case that produced it

vLLM tells you what it measured and what it wants, every single start, and all
of it scrolls past in the log panel and dies there:

```
Available KV cache memory: 10.34 GiB
model loading took 19.07 GiB and 172.5 seconds
Maximum concurrency for 262144 tokens per request: 2.48x
GPU blocks: 42301, CPU blocks: 0
PLE offload: locked 38.8 GiB
```

Two separate losses.

**The measurements are thrown away.** This session spent three pull requests
calibrating an estimator against figures like these, read by hand out of a
terminal and typed into constants: `pleShareOfResidual = 0.90` rests on two
checkpoints somebody eyeballed. The engine reports the true value on every
start. An estimator that could read its own past predictions against measured
outcomes would not need a fudge factor at all.

**The advice is thrown away.** When a start fails, vLLM usually says exactly
what to change — "Try increasing `gpu_memory_utilization`", "decrease
`max_model_len`" — and the user gets a red box with a stack trace in it.

`benchmark.DetectOOM` already reads one of these, for the context probe. It is
the right idea in the wrong place: `internal/benchmark`, scanning a whole
buffer as one string, knowing one kind of advice.

## 2. Decisions

| Question | Decision | Why |
|---|---|---|
| Package | New `internal/advice`, no internal imports | `internal/process` imports only `internal/ansi` and should stay that thin; `benchmark` and `api` both need this too |
| `DetectOOM` | Moves here, becomes one rule among several | Two matchers over the same log text will drift; it already happened once with the env layers |
| Matching | Per line, as it streams | Advice on a live start is worth more than a post-mortem buffer scan. A whole-buffer entry point stays, for the probe |
| Storage | On `process.Manager`, beside the log ring buffer | Same lifetime as the run that produced it; `LogBuffer()` is the precedent to copy |
| Scope of this phase | Capture and display only | See §7 |

**Open, for Tim.** Whether captured measurements should be *persisted* per
model — surviving a restart, so the estimator can be scored against them over
time — or kept only for the current run. This plan builds the run-scoped
version, because that is what displaying it needs, and persistence is a
storage-schema question that should not be settled in passing. §6 is where it
would attach.

## 3. What is captured

Two kinds, and the distinction is load-bearing.

**Measurements** — what the engine reports about what actually happened. These
are facts, and they are the interesting half:

| Line | Captured as | Why it matters |
|---|---|---|
| `Available KV cache memory: X GiB` | `KVCacheGB` | The half of the VRAM estimate never validated against anything |
| `model loading took X GiB` | `WeightsPerRankGB` | The estimator's weights figure, measured |
| `Maximum concurrency for N tokens per request: X.XXx` | `MaxConcurrency` | What the KV pool actually buys |
| `GPU blocks: N, CPU blocks: M` | `GPUBlocks`, `CPUBlocks` | KV pool in the engine's own units |
| `PLE offload: locked X GiB` | `PLEOffloadGB` | Whether offload happened at all — currently *assumed* |
| `... FAILED to lock X GiB` | `PLEOffloadFailed` | The memlock failure this project has hit before, currently invisible |

**Advice** — what the engine suggests changing. Each carries the config field
it implicates and a suggested value where one can be extracted:

| Pattern | Field | Severity |
|---|---|---|
| OOM patterns (the existing eight) | `max_model_len` | error |
| `Free memory on device cuda:N (X/Y GiB) … less than desired` | `gpu_memory_utilization`, suggest ⌊X/Y⌋ to a 0.05 step | error |
| `Decrease` / `increase` **GPU memory utilization** | `gpu_memory_utilization` | error |
| `max seq len (N) is larger than the maximum number of tokens that can be stored in KV cache (M)` | `max_model_len`, suggest M | error |
| `torch._scaled_mm is only supported on …` | — | error |
| `Engine core initialization failed` | — | error |
| `WorkerProc failed to start` | — | error |
| `num_speculative_tokens > 1 … lower acceptance rate` | `speculative_config` | warning |
| `Using fp8 data type to store kv cache … accuracy drop` | `kv_cache_dtype` | info |
| `CUDA_VISIBLE_DEVICES on ROCm is deprecated` | `env` | warning |
| `Chunked prefill is enabled with max_num_batched_tokens=N` | `max_num_batched_tokens` | info |
| unrecognized arguments: `--flag` | `extra_flags` | error |

That last one is from this project's own history: a manifest bump that silently
did nothing surfaced only as `unrecognized arguments: --enable-expert-offload`.

### What a real transcript changed, 2026-09-16

The table above is the second version. The first was written from what vLLM was
assumed to say, and against a real failing start it **matched one line in eight
and missed the failure itself**. Three lessons, all of them cheap to have
avoided by reading a log first:

- **Spelling.** vLLM writes `gpu_memory_utilization` in argument dumps and
  "GPU memory utilization" in prose. A guard on either spelling alone misses
  half the lines that matter — which happened twice here, once in each
  direction, the second caught only because the corpus is now the transcript.
- **Direction.** The original rule always advised *raising*
  `gpu_memory_utilization`. The engine asks for the opposite at least as often,
  and this log says "Decrease". Advice that confidently inverts the engine's
  own instruction is worse than silence.
- **Cheap guards must still be specific.** A secondary check on the substring
  `try` also matches "en**try**.py", "regis**try**", "coun**try**". Only a
  narrow hint was keeping it honest.

The failure in that log was not an OOM at all but a **pre-flight refusal**:
vLLM compares the configured fraction against memory that is *free*, not the
card's size, and something else was already holding ~4.6 GiB per card. Two
consequences beyond this package:

- `gpuInventoryFrom` reported card *size* and ignored `VRAMUsedMB`, which the
  monitor already collects — so the VRAM estimate could promise a comfortable
  fit for a configuration that cannot start. It now takes free memory.
- The memory was held by workers orphaned from a **cancelled tuning job**.
  `internal/tuning` launched its subprocess without a process group, so
  cancelling killed `python` and left the ROCm workers holding their contexts.
  `internal/process` had solved this long ago; the tuner never had. Fixed, with
  the same shell-tree test `stop_test.go` uses.

## 4. Shape

```go
package advice

type Severity int // Info, Warning, Error

type Item struct {
    Severity   Severity
    Message    string // what the engine said, trimmed
    Field      string // the VLLMConfig field implicated, or ""
    Suggested  string // extracted value, or ""
    Line       string // the source line, verbatim
}

type Measurements struct {
    KVCacheGB        float64
    WeightsPerRankGB float64
    MaxConcurrency   float64
    GPUBlocks        int
    CPUBlocks        int
    PLEOffloadGB     float64
    PLEOffloadFailed bool
}

// Scan matches one line. Returns nil when nothing matches, which is the
// overwhelmingly common case and must stay cheap.
func Scan(line string) *Item

// Observe folds a line's measurements into m. Separate from Scan because a
// line can be both, and because measurements accumulate while advice is a list.
func Observe(m *Measurements, line string)

// ScanAll is the whole-buffer entry point, for callers holding a transcript
// rather than a stream — the context probe, and post-mortem of a crashed start.
func ScanAll(logs string) []Item
```

`Scan` runs on every log line of every start, so the rule set is compiled once
at package init and the common path is a handful of `strings.Contains` guards
before any regexp is touched.

## 5. Wiring

- `process.Manager` gains `advice []advice.Item` and `measured advice.Measurements`,
  cleared on start, appended in `streamOutput` beside `appendLog`, exposed as
  `Advice()` and `Measured()` following `LogBuffer()`.
- The readiness check in `streamOutput` becomes one more rule, rather than the
  ad-hoc `strings.Contains` pair sitting there now.
- `benchmark.DetectOOM` delegates to `advice`, keeping its signature and tests.

## 6. Display — half built

The **measurements** half shipped in phase 14: the config panel shows measured
figures beside the estimate and says which it is showing.

The **advice** half is still not built, and the cost of that is now visible.
The process manager's log buffer caps at 5000 lines, and on a host serving
steadily that is a few hours -- by 2026-09-21 a start from the 18th had been
pushed entirely out of the log, startup lines and all. The advice items
themselves survive in `Manager.advice` the whole time, capped at 64 and
unreachable, because nothing reads them back.

So the panel must read `Advice()` rather than re-parse the log. Re-parsing
looks equivalent and stops working within an afternoon.

The service page gets an advice panel beneath the log: severity, message, and
the field it implicates. Errors persist after a failed start — that is when
they are most wanted, and currently when the log panel is least readable.

Measurements go beside the VRAM estimate in the config panel, as a second
column: **estimated** against **measured**, once the model has been run. That
comparison is the whole point. It is also where the persistence question in §2
attaches, and the reason to answer it.

## 7. Out of scope

**Acting on any of it.** No auto-apply, no "fix it for me" button, not in this
phase. The parser has to be trusted before anything is allowed to rewrite a
config from it, and trust here means a few weeks of watching it read real
starts correctly — including the ones where vLLM's own advice is wrong, which
it sometimes is: "increase gpu_memory_utilization" is unhelpful at 0.97.

### Revisited 2026-09-21: a narrow apply

Acting on it is now in, for three suggestions out of six, behind an explicit
click with the engine's own line visible beside it. Nothing happens on its own.

The distinction that makes it safe is that **applicability is declared per rule
rather than inferred from an item having a value**. Half the suggestions are
not settings at all, and applying them would be worse than ignoring them:

| suggestion | apply? | why |
|---|---|---|
| `max_model_len` from seq-len-vs-KV | yes | the engine states the ceiling it measured |
| `kv_cache_memory` from `--kv-cache-memory=` | yes | the engine's own pool byte count |
| `gpu_memory_utilization` from the free-memory refusal | yes, marked *ours* | the shortfall is the engine's; the fraction that clears it is arithmetic done here |
| `--gpu-memory-utilization to 0.9826` | **no** | an equivalence figure; applying it chases an artefact of how memory is counted |
| `extra_flags` from unrecognized arguments | **no** | the flag named is the problem, not the fix |
| `max_num_batched_tokens` from chunked prefill | **no** | an echo of what is already configured |

Two things the panel has to say out loud. A suggestion computed here is
labelled as ours, because a derived number and a reported one do not deserve
equal confidence — that is the whole lesson of phase 14. And pinning
`kv_cache_memory` stops the pool being measured again, which is the one
suggestion that costs something to take; the button says so.

The caution has earned itself twice over: this parser has produced advice that
was confidently backwards (telling the operator to raise a setting the engine
had asked them to lower) and advice that fired on a healthy start. Both would
have been propagated by a button that acted without being asked.

The estimator scoring itself against captured measurements is the phase after
that, and it is the one worth wanting. It is also why §3 lists measurements
first and advice second, though the feature is named for the advice.

# Remaining Work

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

- **Watch a real startup.** Compare the panel against the engine's own
  `Available KV cache memory` and `model loading took` lines. The weights
  figure has been checked against 19.07 GiB per rank; the KV figure and the
  full-context request count have not been checked against anything.
- **The KV figure counts one sequence at `max_model_len`.** That is the
  minimum to serve the configured context at all. The panel reports separately
  how many full-length requests the leftover buys, which is the number to tune
  `max_num_seqs` against. Whether the headline should instead assume full
  concurrency is a judgement that can be revisited once the startup figures
  have been compared.
- **The CUDA-graph pool is a flat 0.9 GiB** (measured 0.49 and 1.32 on two
  checkpoints) and activation assumes vLLM's 2048-token default chunk when
  `--max-num-batched-tokens` is unset. Both are stand-ins for a measurement
  nobody has taken.
- **Speculative/MTP draft weights and vision-tower parameters are not counted.**
  Both checkpoints on compute run MTP with 3 draft tokens; whatever that costs
  is currently absorbed into the residual and attributed to the PLE table.
- **The expert share is fitted to one measurement.** 18.72 GiB moved against a
  46 GiB ceiling, which is 30% of that checkpoint's idle expert weight. That
  30% is now the centre of the estimate with a ±50% band around it, because one
  point cannot support anything tighter. A second offload configuration —
  different ceiling, different model — would either corroborate it or show it
  for the coincidence it might be. Until then this is the weakest number in the
  estimator, and the band is wide on purpose.
- **The projected path is not wrong in a fixable direction.** Measured on a
  second machine 2026-09-21 -- a 4B AWQ model on one RX 7900 XTX -- against
  what the formula projected for the same configuration:

  | | projected | measured |
  |---|---|---|
  | total | 13.9 GB | 16.9 GB |
  | KV per token | 73,728 | 17,301 |
  | activation | 0.20 GB | 6.3 GB |
  | graph pool | 0.9 GB | 3.58 GB |

  On the 124B MXFP4 the KV figure was 2.6x too *low*; here it is 4.3x too
  *high*. Activation is 31x low on this model and 23x low on the other. So the
  errors do not share a direction and no coefficient corrects them -- which is
  the case for measuring rather than tuning, stated in numbers rather than as
  an opinion. Leave the projected constants alone.
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

## Second VRAM estimator in the HuggingFace client

`internal/huggingface/client.go:564` estimates VRAM for models not yet
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

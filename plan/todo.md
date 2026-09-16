# Remaining Work

## VRAM estimator: what compute measured, 2026-09-16

Checked against the running instance after the rework landed. Three fixes came
out of it, and one judgement was reversed on better evidence.

- **The PLE residual works on the real checkpoints.** On the GPTQ checkpoint it
  puts the table at 43.6 GiB against 47.68 measured, and TP=4 at 17.9 GiB per
  rank against the engine's own 19.07. It had looked inert locally only because
  no checkpoint on the workstation carries such a table. It is now the estimate,
  banded at ±20% rather than trusted flat, and a verdict is offered again.
  Tighten the band once more checkpoints with a PLE table have been measured —
  two points is what the ±20% is resting on.
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

## VRAM estimator: calibration still owed on compute

The estimator was reworked 2026-09-16: parameter counting is MoE-aware, the
stored estimate no longer carries a fit verdict, and fit is judged against the
host's real cards at every tensor-parallel width rather than against an
invented single 32 GiB card. What is still owed:

- **Watch a real startup.** Compare the panel against the engine's own
  `Available KV cache memory` and `model loading took` lines. The per-rank
  weight figure has been checked against 19.07 GiB; the KV headroom and the
  max-context column have not been checked against anything.
- **The ±20% band on the PLE residual is a prior, not a measurement.** It rests
  on two checkpoints. A third would justify tightening or widening it, and
  until then the band width is the least evidenced number in the estimator.
- **The CUDA-graph pool is a flat 0.9 GiB** (measured 0.49 and 1.32 on two
  checkpoints) and activation assumes vLLM's 2048-token default chunk when
  `--max-num-batched-tokens` is unset. Both are stand-ins for a measurement
  nobody has taken.
- **Speculative/MTP draft weights and vision-tower parameters are not counted.**
  Both checkpoints on compute run MTP with 3 draft tokens; whatever that costs
  is currently absorbed into the residual and attributed to the PLE table.
- **Expert offload has no pessimistic floor, and the band shows it.** The
  ceiling moves only the optimistic end, so the MXFP4 checkpoint predicts
  8.5–75.4 GiB of device weights — a ninefold spread that still lands a verdict
  at TP=4 only because the budget is generous, and reads "depends on offload"
  at TP=1 and TP=2 where the GPTQ checkpoint gives a plain refusal. An enabled
  offload plainly moves *something*, so a floor would tighten this a great
  deal. Not added: there is one data point for it (18.72 GiB moved against a
  46 GiB ceiling), and inventing a floor from a single measurement is the same
  mistake the PLE residual made when it was calibrated on two. Measure a second
  configuration before closing this end of the band.

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

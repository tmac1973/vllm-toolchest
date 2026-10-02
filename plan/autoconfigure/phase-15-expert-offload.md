# Phase 15 — Expert offload in the planner

**Depends on:** phase 02 (the planner), phase 10 (the review) · **Enables:**
the recommend feed counting offload (`plan/phase-17-recommend-engine.md`,
amended in step 9)

## Goal

On an image that can keep MoE experts in system RAM, a model that does not
fit in VRAM is planned with expert offload instead of "does not fit", and a
model whose context would be cut is offered offload as an alternative. Models
that fit are untouched: offload is proposed only where it is the difference.

## Decisions

Asked 2026-10-01, each with its effect on autoconfigure:

- **When** → only when needed: no width fits, or the context would be cut
  below what was asked for.
- **Default** → ticked when nothing fits without it; otherwise an unticked
  alternative beside the plan that fits.
- **Validation** → measured first, on compute: Flash-Next at two cards, where
  it fits only with offload. What follows is from that.

## What the engine does (rdna4-clav, vLLM 0.29, measured 2026-10-01)

`--enable-expert-offload` makes the image plan each card before loading. For
Flash-Next (`tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`) at TP=2, 65,536
tokens, utilization 0.90:

```
budget (total x util)                          28.67 GiB
non-expert weights                              4.47     PLE 38.74 excluded
KV reserve (one max-model-len request +8%)      0.64     registry 9190 B/token
CUDA graphs                                     0.40
runtime overhead                                8.48     measured 6.38
expert cache space (the rest)                  14.68     of it resident layers [0,1,2] 1.87
host-backed experts (per rank)                 25.1      host usable 112.6 of MemAvailable 179.3
```

- The KV pool is **capped at the reserve**: one request at the planned
  context, 79,953 tokens = 1.22 requests at 65,536. Everything else goes to
  the expert cache.
- **MTP and offload do not start together.** The reserve is sized from the
  engine's registry figure, which leaves out the MTP layer: with MTP the KV
  measured 11,597 B/token per rank against 9,655 reserved, and the start
  failed 0.09 GiB short. Without MTP it measured 8,437 and served. vLLM's
  suggested shorter context would fail the same way, as the reserve shrinks
  with it.
- Loading took 7.7 minutes, most of it pinning experts.
- Decode at TP=2 with offload: about 54 tokens/s (512 tokens, batch 1),
  against 153 for the hand-tuned config at TP=4 without offload (with MTP).
  Not one change apart, but the trade the review offers.
- The image's docs name the supported models: MoE with MXFP4 or W4A16 expert
  weights (Qwen3.8-Flash-Next, DeepSeek-V4, GLM-5.3-Flash).

## Design

1. **Capability.** `variants/rdna4-clav.conf` declares `expert_offload`.
   `PlanInput` carries whether the running image has it, and the host's RAM.

2. **Eligible models.** An MoE (`isMoE`) whose weights are under 8 bits --
   the image's MXFP4 and W4A16 -- on an image with the capability.

3. **Expert size.** `MoE layers × experts × 3 × hidden × moe_intermediate`
   parameters at 4.25 bits (MXFP4's 4-bit values plus an 8-bit scale per 32):
   59.7 GiB for Flash-Next against the engine's 59.9. Non-expert weights are
   the device weights less that, with the replication surcharge per rank.

4. **Feasibility at a width**, per card, against `util × card`:
   non-expert weights + one request's KV at the target context × 1.08 +
   graphs 0.4 + runtime overhead 8.5 (the engine's default; measured 6.4 to
   8.4) + a minimum expert cache. The minimum is two layers' worth for every
   resident layer the engine keeps -- 3 of the model's MoE layers, doubled for
   a swap region -- a stated assumption, since no start with a smaller cache
   has been tried. And in RAM: all the experts, plus any PLE table, within
   90% of the host's RAM less 7 GiB a rank and 5 for the engine, as the
   engine's own sum does.

5. **The plan.** Offload is planned at the all-cards width (more cards, more
   cache, faster decode) at the target context, one full-length request.
   - No width fits without it → the plan *is* the offload plan, and the
     review's offload row starts ticked.
   - The plan fits but cuts the context → the offload plan is a further
     width choice in the review, "All 4 cards, experts in RAM — full context,
     one request at a time, slower", not preselected.
   - Neither feasible → "does not fit", as today.

6. **The review.** The offload option sets `--enable-expert-offload` in the
   extra flags. Choosing it leaves out an MTP speculative config, with a note
   saying why (above). The reason on the option says generation is slower,
   with the measured figure.

7. **Refinement** needs nothing new: the engine's KV pool under offload is
   the reserve, and the measured start says so.

8. **Estimator.** Left as it is. Its "30% of idle experts move" band still
   describes a config with offload set by hand; the planner does not use it,
   working out feasibility from the engine's own sum above, and a start's
   measurement replaces both.

9. **The feed** (`plan/phase-17-recommend-engine.md`, step 14): a finalist
   that does not fit is not Dropped when this planner finds an offload plan
   for it; it is Verified, labelled "experts in system RAM — slower
   generation", scored for context at its offload context, and penalised on
   **Fastest** as weight-only formats are.

## Added 2026-10-02: the PLE table on NVMe

Flash-Next's card gives a two-R9700 recipe the first version could not
reach: expert offload *and* the n-gram table served from NVMe
(`--ple-nvme-offload --ple-cache-gb 8 --ple-cache-reuse true`), 82 GiB of
RAM at runtime and about 100 tokens/s. The planner held the 41.6 GB table in
RAM beside the experts, so a two-card host with 128 GB was told it did not
fit.

- The image declares `ple_nvme_offload`. When the experts and the table do
  not fit in RAM together, the plan serves the table from NVMe and keeps an
  8 GB row cache, and carries the three flags (`models.CarryOffloadFlags`,
  also used by the review and by seeding). The reason says the table file is
  written beside the checkpoint, so that much disk is needed.
- RAM is charged for the experts the cards do not keep: the resident layers
  stay on them (on compute, 50.2 GB of Flash-Next's 59.9 lived in RAM).

Live, two R9700s: 188 GB of RAM, offload with the table in RAM; 128 and
96 GB, offload with the table on NVMe; 64 GB, not recommended.

**Started on compute, 2026-10-02**, from the planner's own two-card, 128 GB
config: the engine wrote the 38.7 GiB table to NVMe and kept an 8 GiB row
cache; 84 GB of RAM in use against the card's 82 GiB; two cards at about
26 GB each. The first start failed 0.84 GiB short of KV cache: the image
reserves the offload's cache from an fp8 figure (9,302 B/token) and the plan
had the default cache, which measured 13,427. With fp8 it measured 7,298 and
served: 360,637 tokens, 1.38 full-length requests at 262,144, about 39
tokens a second. Offload plans now always use an fp8 cache.

## Tests

- The Flash-Next record at two cards: offload planned at 65,536, one
  request; four cards: no offload (it fits).
- A dense model, an 8-bit MoE, or an image without the capability: never.
- Not enough RAM: "does not fit".
- The review: ticked when it is the only plan; an unticked option when the
  context would be cut; choosing it drops an MTP config with a note.

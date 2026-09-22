# Recommend Models — Project Overview

Covers phases 15–19. This document is the shared definition; each phase
document carries its own steps, gates and rollback.

## Problem

The Download Models page can only answer a question the user already knows how
to ask. It searches HuggingFace by name, which is useful when you know what you
want and useless otherwise — and most people do not know what is out there.
New quantizations of good models appear weekly, and nothing on this machine
tells you about them.

Worse, the page is confidently wrong about size. `estimateParamCount`
(`internal/huggingface/client.go:524`) derives parameter counts from a dense
transformer formula: `4*h*h` for attention, `3*h*inter` for the MLP, no term
for experts. On a mixture-of-experts checkpoint, where `intermediate_size` is
per-expert, that understates the model by roughly the expert count. The
machine this is being built for runs a 125B MoE. The existing estimate is off
by an order of magnitude for exactly the models that matter most, and the page
presents it without qualification.

Meanwhile the project already knows almost everything needed to answer "will
this run here": `GPUInventory` has the cards, `Fit()` and `evaluateTP`
(`internal/models/vram_fit.go`) compute tensor-parallel options against a VRAM
estimate, `variants/*.conf` declare what each image can do, and the search
already requests `config=true` for every result and then discards all but the
`quantization_config`. The inputs exist and are thrown away.

## Goals

- A "Recommended for this machine" feed above the search box on Download
  Models, listing models that will actually run on this hardware and runtime.
- Four intent chips — **Best quality**, **Fastest**, **Longest context**,
  **Newest** — each re-ranking the same candidate pool against a different
  objective. All four are arithmetic over data in hand; none requires guessing
  what a model is "for".
- Recommendations generated dynamically from the Hub on every refresh. No
  model names ship in the repository, so nothing goes stale as models are
  published.
- Every card states *why*: the tensor-parallel width it fits at, the VRAM it
  needs against what is present, and whether the format is hardware-accelerated
  here. It also states whether the architecture is in this image's model
  registry, except when the registry has not been read — in which case the
  card says nothing about architecture rather than implying a check that did
  not happen.
- Exact weight sizes from the Hub's `safetensors` metadata rather than a
  formula, replacing the MoE-blind estimate for candidate models.
- Fit output survives the download: the tensor-parallel width and context
  length computed during ranking, plus a KV cache dtype taken from the same
  hardware profile the ranking was judged against, become the new model's
  starting `VLLMConfig`, recorded as having been seeded rather than chosen, so
  the Models page can say where they came from.
- The existing search keeps its place on the page and its behaviour. The
  size figure in the panel that expands under a search result is corrected by
  phase 15, because it is wrong today and the correct one is a prerequisite
  for the feed; the result list itself shows no size and does not change.

## Non-goals

- **No launch-config optimizer.** Seeding `tensor_parallel_size`,
  `max_model_len` and `kv_cache_dtype` at download time is the whole of the
  handoff. Tuning a running model remains phases 13/14's territory.
- **No use-case classification.** No Chat / Code / Tool-use chips. Those
  require inferring a model's purpose from names and tags, which is guesswork,
  and a misclassification is invisible to the user. The chat template that
  would make tool-use detectable is returned by the Hub, so this stays
  possible later — it is deliberately not attempted now.
- **No quality ranking of our own.** The ranking inputs are parameter count,
  fit, the Hub's own popularity and recency figures, and a boost for a shipped
  list of publisher names. Nothing else: no benchmark scores, no leaderboard
  scraping, and no judgement of a model's output.
- **No config profile** is created on download. Phase 11's profile mechanism
  stays uncoupled from this feature.
- **No automatic re-configuration after a run.** The seeded values are
  ordinary configuration and stay as written until a person changes them.
  Measurement corrects the VRAM *estimate* shown against a model; it does not
  and will not rewrite `VLLMConfig`.
- **No GGUF.** Unchanged from today: `isGGUFOnly` keeps filtering it out.
- **No background polling or notifications.** The Hub is queried when a page
  load finds no pool or finds one built against different hardware, and
  whenever the user presses Refresh. A pool older than its TTL is served as-is
  and marked stale rather than rebuilt behind the user's back. Nothing watches
  the Hub on a timer.
- **No change to how search finds or groups models.** Its query gains one
  expansion parameter, and the size shown in the expanded detail panel becomes
  exact; the quant filter, the grouping, the result ordering and the result
  list's own markup are untouched.

## Users & primary flow

The user is the operator of a single machine running vLLM through this tool —
in the case driving the design, four AMD R9700s (gfx1201, 32 GB each) on the
`rdna4-clav` image.

1. They open **Download Models**.
2. Above the search box, a header states what the recommendation is computed
   against: `4× Radeon AI PRO R9700 · 128 GB · gfx1201 · rdna4-clav`. The card
   name comes from the monitor's GPU reading, the same source the dashboard's
   GPU card already uses. This is the profile, and it is visible so a wrong
   recommendation can be traced to a wrong input.
3. The feed loads under the **Best quality** chip by default, showing models
   that will run, best first.
4. Each card gives the repository, its format badge, its exact size, and the
   fit sentence: *"fits at TP=2 · 41 GB of 64 · accelerated on gfx1201 ·
   Qwen3ForCausalLM supported by this image"*. The architecture is named by
   its class, because that is the name vLLM's registry is keyed by and
   therefore the name that was actually checked.
5. Below the verified models, an **Unverified** section lists candidates that
   could not be fully checked, each saying which input was missing — *"size
   unknown — the repository publishes no safetensors metadata"*. Models proven
   not to run are dropped and never shown.
6. The user switches chips. The press is a local request to this server and
   nothing more: it reaches no external service and sorts nothing, because all
   four orders were computed when the pool was built and the chip reads one of
   them back.
7. They press **Download** on a card. The download follows the existing path.
   When it completes, the new `models.json` entry carries the tensor-parallel
   width and context length that ranked it, recorded as seeded rather than
   chosen by a person. They remain ordinary configuration: nothing later
   rewrites them.
8. Below all of this, the HuggingFace search box works exactly as it does
   today.

## Constraints

**The Hub's list endpoint returns two tiers of data, and this shapes
everything.** Verified against the live API while planning:

- `expand[]=safetensors` returns exact per-dtype parameter counts —
  `{"parameters":{"BF16":1558406144,"F8_E4M3":31205621760},"total":32764027904}`.
  This gives an exact weight size with no formula and no MoE problem.
- `expand[]=config` returns **only** `architectures`, `model_type`,
  `quantization_config` and `tokenizer_config`. It does **not** return
  `hidden_size`, `num_hidden_layers`, `num_key_value_heads` or
  `max_position_embeddings`.

So weight size and architecture are cheap and available for the whole pool,
but everything the KV-cache arithmetic needs requires a separate fetch of the
repository's raw `config.json`. Ranking must therefore be staged: coarse rank
on list data, then fetch `config.json` for the finalists only.

- A 50-result query with those expansions is ~854 KB, dominated by chat
  templates inside `tokenizer_config`. A refresh issues one query per sort
  order for each Hub tag the buckets expand to, plus one untagged query for
  unquantized models — five queries per sort on the reference hardware, so ten
  in total, up to ~500 results before deduping, and ~8.5 MB. The pool must be
  cached rather than re-fetched per chip.
- Some repositories return `"total": null` — no safetensors metadata at all.
  Observed on a live query. These are the Unverified bucket, and they are not
  rare.
- The Hub cannot filter on size, architecture or fit. Candidate generation is
  necessarily broad and scoring is necessarily local.
- Architecture support is version-specific and not introspectable from
  outside the container. It must be read from the running image's
  `ModelRegistry`, by the same background-probe-and-cache pattern
  `cmd/vllmctl/main.go` already uses for the vLLM device name.
- Stack is unchanged: Go, `chi`, `html/template`, htmx, Pico. No new runtime
  dependency, no JavaScript framework, no database.
- The tool runs on machines with no outbound network access at times. The feed
  must degrade to an explanatory empty state, never an error page, and never
  block the search box below it.
- HuggingFace API calls must respect the configured token when present, via
  the existing `Client.setAuth`.

## Success criteria

- On the reference machine, the feed's top **Best quality** result fits and
  starts, and its stated TP matches what the engine reports once running.
- A mixture-of-experts checkpoint is sized correctly. Specifically: a 125B MoE
  is reported at its true size, not the ~10× understatement today's formula
  produces.
- A model too large for the machine at every valid tensor-parallel width never
  appears in the feed.
- A model whose architecture is absent from the running image's registry never
  appears as verified; if shown at all, it is in Unverified with that reason.
- Every verified card's fit sentence names a real number for VRAM required and
  VRAM present, and those numbers are reproducible from `Fit()` given the same
  inputs.
- Switching between the four chips issues no request to HuggingFace and
  triggers no re-ranking: the four orders are computed once per refresh and
  a chip press only reads one back.
- With the network unreachable, Download Models still renders, the search box
  still works, and the feed shows a stated reason rather than an error.
- A model downloaded from the feed arrives in `models.json` with
  `tensor_parallel_size` and `max_model_len` set from the fit estimate and
  recorded as seeded, and the Models page says so. Starting the model leaves
  those values alone — the run produces a measurement, and the measurement
  changes the estimate shown, not the configuration.
- No model or repository name appears in any shipped ranking input. The only
  curated data that ships is a list of publisher names used as a ranking
  boost, and emptying that file degrades the ordering without breaking the
  feed. Recorded Hub payloads under `testdata/` are fixtures, not ranking
  inputs, and are exempt.

## Decisions

- **Plan document layout** → Follow the existing `plan/` convention: globally
  numbered `phase-NN-slug.md` written directly into `plan/`, added to the
  Active list in `plan/README.md`. This overview sits beside them as
  `recommend-models-overview.md`.
- **Feature shape** → A "Recommended for this machine" feed above the search
  box on Download Models, with a row of intent chips. Search stays unchanged
  below it.
- **Candidate generation** → Dynamic Hub queries per executable quant bucket,
  blending `sort=downloads` with `sort=lastModified`, ranked locally by fit,
  acceleration, architecture support, freshness and popularity, with a boost
  for a shipped list of trusted quant **publishers**. The shipped list names
  authors, never models.
- **Unverifiable candidates** → Three outcomes. Proven-bad is dropped
  silently; proven-good is ranked normally; anything with a missing input
  appears in a separate Unverified section stating which input was missing.
- **Intent chips** → Computable axes only: Best quality, Fastest, Longest
  context, Newest. No use-case chips, because Code and Tool-use would require
  guessing.
- **Download handoff** → Seed the new model's `VLLMConfig` with the
  tensor-parallel width and context length computed during ranking, plus a KV
  cache dtype read from the hardware profile, recorded as seeded so the Models
  page can say where they came from. Nothing later rewrites them: measurement
  refines the VRAM estimate, not the configuration. No config profile is
  created.

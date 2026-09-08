# UI Parity Plan — vllm-toolchest → llama-toolchest

Bring the vllm-toolchest web UI up to the level llama-toolchest reached at
v2.28.0. Reference implementation lives at `/home/tim/projects/llama-toolchest`;
the two running instances compared for this plan were `compute:3000` (vllm) and
`compute2:3000` (llama).

This is a UI-parity plan, not a feature-copy plan. Three llama-toolchest areas
are deliberately **out of scope** because the engines differ:

| Not ported | Why |
|---|---|
| Builds tab | vLLM ships pre-built in the container; there is nothing to compile. |
| Router / multi-model "enabled" semantics | `vllm serve` is one model per process. Replaced by an *active model* concept (see Phase 4). |
| Evaluation data (perplexity, HellaSwag, Winogrande, KL-divergence) | Depends on `llama-perplexity` and llama.cpp logit dumps. No vLLM equivalent. |

Three vllm-toolchest features have **no llama counterpart and must not regress**:
the Tuning tab (Triton FP8 autotuner), context-length probing, and the Radiance
(RDNA4) settings section — including the all-reduce token ceiling surfaced in
the model config panel.

---

## Decisions taken

1. **Models page** uses a *radio-style active model* column. One model is
   designated active; that is what Start/Restart launches. A restart glyph
   appears when the active model differs from the one currently running.
2. **Benchmarks** get everything portable, including the Plotly visualize page.
   Only the eval datasets are dropped.
3. **All HTML rendering moves into `html/template` partials up front**, in a
   dedicated no-behaviour-change phase, before any parity work lands.
4. **Phase 1 is wide**: shell/theming, the dashboard/service merge, the Models
   page rebuild, and the ROCm GPU-index fix all land together.

Phases 0–1 are the first wave. Phases 2–5 follow in the order listed but are
independent of each other.

---

## Phase 0 — Move HTML rendering into templates — **done**

No visible change. This is a precondition: the Models card system, the shared
grid CSS and the OOB-swap patterns in later phases are impractical to maintain
inside `fmt.Fprintf` strings, and `models.go` alone is 713 lines of them.

### Outcome

Every HTML-emitting handler now renders a partial. `internal/api/htmlout.go`
(`htmlPrinter`, `esc`, `safeHTML`) is gone — `html/template` does the escaping,
and does it per context rather than uniformly.

New partials: `model_list`, `model_config`, `dashboard_cards`,
`service_status`, `hf_results`, `benchmark_runs`, `benchmark_jobs`,
`bench_about`, `probe`, `messages`.

Verification, in three layers:

- `internal/api/golden_test.go` records every fragment a local, deterministic
  Server can produce, byte for byte, and classifies a mismatch as
  whitespace-only or substantive. Run with `-update` to re-record.
- `internal/api/partials_test.go` renders the partials the handler-level
  recordings cannot reach — a live download, a finished run, a populated job,
  a running service — straight from view data. This caught a real bug during
  the conversion: `$` inside `hf_results`' nested range points at the whole
  payload, not the group, so every variant badge targeted an empty id.
- The three config panels were rendered before and after into the same page
  shell and screenshotted: pixel-identical PNGs. The whitespace changes a
  readable template introduces do not reach the rendered page.

### Fixed along the way

Each of these was found by the conversion and is a deliberate change, not a
side effect:

- **`Registry.List()` returned models in map order.** The models table
  reshuffled its rows on every htmx refresh, and the benchmark and probe model
  pickers reordered between openings. Now sorted by ID.
- **A custom context length was silently reset on the next save.** The context
  picker rendered `name="max_model_len"` unconditionally, so on a model whose
  context was already custom both it and the custom number box submitted that
  field; the picker's `"custom"` came first, parsed to 0, and the operator's
  value was replaced by the model default. The picker now carries the name only
  when the custom box does not.
- **Probe results came out in map order too** — the utilization and
  concurrency tables reordered on every 3s poll. Now sorted.
- **`onclick`/`onchange` handlers were HTML-escaped, not JS-escaped.** An id
  containing a quote produced a broken handler. `html/template` escapes by
  context, so this is now correct by construction. Only reachable with a model
  id no HuggingFace repo can have, but it is the same class of bug as the one
  commit `8e22446` fixed.

### Noted, not fixed

- `hx-get="/api/models/config-panel?id=…"` interpolates the model ID into a
  query string without URL-escaping it. Harmless for real HF repo names (no
  spaces, quotes or ampersands are valid), and Phase 1d moves these to
  `/api/models/{id}/…` anyway.
- `/api/service/logs` returns raw log lines under `text/html`. The client
  assigns them with `textContent`, so nothing is interpreted; Phase 1c replaces
  this endpoint with the SSE stream.
- Several files carry pre-existing `gofmt` drift (struct-tag alignment in
  `settings.go`, `config.go`, `vram.go` and others). Left alone rather than
  mixed into this diff.

### What moves

| From | To |
|---|---|
| `internal/api/models.go` — list table, config panel | `partials/model_card.html`, `partials/model_config.html` |
| `internal/api/hf.go` — search results, model detail, download progress | `partials/hf_results.html`, `partials/hf_files.html`, `partials/download_progress.html` |
| `internal/api/server.go` — `handleDashboard` | `partials/dashboard_cards.html` |
| `internal/api/service.go` — status | `partials/service_status.html` |
| `internal/api/bench.go`, `bench_jobs.go`, `bench_probe.go` | `partials/job_list.html`, `partials/job_detail.html`, `partials/benchmark_form.html`, `partials/job_form.html`, `partials/probe_form.html` |
| `internal/api/tuning.go`, `allreduce.go` | `partials/tuning_status.html`, `partials/allreduce.html` |

Partial names mirror llama-toolchest's wherever a counterpart exists, so the two
trees stay diff-able.

### Template funcs to add

Port from `llama-toolchest/internal/api/server.go:templateFuncs`. Keep the
existing `divf`, `pctOf`, `formatBytes`.

```
divGB, cssID, add, tern, deref, vramFit, hfModelURL, version
```

`hfModelURL` replaces llama's `sourceModelURL`/`sourceName` pair — vllm-toolchest
searches HuggingFace only (ModelScope is noted as a possible follow-up in
Phase 5, not committed here).

### Escaping

`html/template` auto-escapes, which makes most of `internal/api/htmlout.go`
(`esc`, `htmlPrinter`) redundant. Retire it as call sites move. `safeHTML` stays
only for the badge helpers (`quantBadgeHTML`, `vramLabelHTML`) and those should
instead become template blocks — no `template.HTML` values crossing the boundary
if it can be avoided. Commit e5188c3's ancestor `8e22446` ("Escape values
interpolated into HTML") fixed this class of bug once; the conversion must not
reintroduce it.

### Acceptance

- `internal/api/pages_test.go`, `models_config_test.go`, `htmlout_test.go` pass
  unchanged, or are updated only where they assert on printf-specific whitespace.
- `go test ./...` green.
- Rendered output for `/models`, `/`, `/benchmarks` byte-compared against a
  pre-conversion capture and reviewed for intentional differences only.

---

## Phase 1 — First wave — **done**

Four independent workstreams that ship together.

### Outcome

All four landed. Deviations from the plan, and what the work turned up:

- **1a** keeps the theme picker as a `<select>` rather than llama's button
  grid: it is a field of the settings form, which is what persists the choice
  server-side, and llama's buttons are browser-local only. Found that the
  stored theme was written on save and then never applied — the layout
  hardcoded dark and read only `localStorage`, so choosing a theme on one
  machine did nothing on any other.
- **1b** is ported but **not yet verified on the R9700 box**. No SSH access
  from here. After deploying, `/api/monitor` should report indices 0–3 with
  four distinct VRAM readings; today it reports index 0 four times.
- **1c** turned up that the settings handler read five checkboxes
  unconditionally, so the server page's two-field save would have cleared tool
  use, Marlin, auto-restart, eager mode and prefix caching. The settings form
  now marks itself and only it drives those.
- **1d** keeps `?id=` query parameters instead of moving to `/{id}`: vLLM
  registry IDs are HuggingFace repo ids and contain a slash, which a chi path
  parameter will not match. llama's IDs are slash-free, which is why it can.
  The GPU map shows the engine against everything else on each card rather
  than model-vs-model, since one vLLM engine spreads evenly across its ranks.
- The downloads panel required fixing the downloader: cancelling or failing
  called `RemoveAll` on the model directory, so the Range-resume support in
  `downloadFile` was unreachable and a network blip discarded the whole
  transfer. Pause/Resume/Discard work on real downloads now (verified against
  a live HuggingFace transfer: paused at 61.8 MB, resumed from 61.8 MB).
  `Discard` also had a path guard that never fired — an id that was not
  `owner/name` resolved to the models root, so `Discard("")` would have
  deleted every model on the box.

Not carried over from llama's Models page: the Embedding Models section and
its preset downloader. vLLM can serve embedding models with `--task embed`, so
it is a plausible follow-up, but it is an addition rather than parity.

### 1a. Shell: theme, typography, sidebar

Port from `llama-toolchest/web/templates/layout.html`.

- **Graphite theme.** The full `[data-custom-theme="graphite"]` block plus its
  two follow-ups (`article` 12px radius, form controls 8px) and the muted
  off-state switch knob rule. This is llama's signature look and the single
  biggest reason vllm "looks unthemed" today.
- **Fonts.** Copy the four woff2 files from
  `llama-toolchest/web/static/fonts/` into `web/static/fonts/`, add the four
  `@font-face` blocks and the `:root` font-family variables. Self-hosted, no CDN.
- **Type scale.** `body { font-size: 16px }`, `h1 { 21px/600/-0.01em }`,
  `small { 0.95em }`, mono stack on `code, kbd, pre, .mono`.
- **Sidebar.** Add the `.sidebar-top` wrapper — its absence is why vllm's nav
  items currently float in the vertical centre of the sidebar instead of sitting
  under the brand. Add the `.brand` block with the version string. Match
  llama's metrics: 224px width, 16px padding, 3px item gap, 9px radius, 17px
  links, `--sidebar-bg` / `--accent-soft` hooks.
- **Shared component CSS.** `.card-h`, `button.action-icon` +
  `.action-icon-placeholder`, `.model-list-controls`, `.available-models-scroll`
  + `.available-model-row`, and the whole `.model-card*` grid system (Phase 1c
  depends on it).
- **Theme picker** in `settings.html` becomes llama's button grid, with Graphite
  first. Keep the `vllmctl-theme` localStorage key.

**Version wiring** (currently absent entirely):

- `Makefile`: `-ldflags "-X main.version=$(VERSION)"`, `VERSION` from
  `git describe --tags --always --dirty`.
- `cmd/vllmctl/main.go`: `var version = "dev"`, passed into `api.NewServer`.
- `api.Server` gains a `version` field and the `version` template func from
  Phase 0.

### 1b. ROCm GPU index fix

**This is a live bug.** `compute:3000/api/monitor` currently returns four GPUs
all reporting `"index": 0` with identical VRAM figures — the sidebar reads
"GPU0" four times, and every per-GPU figure is really GPU 0's.

Cause: `internal/monitor/rocm.go` trusts rocm-smi's `device` column
(`card0`, `card1`, …) as the GPU index. Depending on rocm-smi version and
machine that column is either the DRM card number (driver probe order, shifted
by BMC/iGPU display devices) or rocm-smi's own row number sorted by PCI bus —
sometimes the reverse of KFD order.

Fix, ported from `llama-toolchest/internal/monitor/rocm.go`:

- Add `--showbus` to the rocm-smi invocation and require a `PCI Bus` column.
- Add `listAMDGPUDirs()` and `kfdIndexByBDF()`; match each rocm-smi row to its
  KFD position by bus address.
- Reject the whole collection (falling through to `collectSysfs`, which is
  consistent by construction) when the column is missing, a device is not in the
  KFD topology, or two rows claim one index.
- Preserve vllm-only fields `ROCmVersion` / `DriverVersion`, which llama's
  version does not carry.

Verify on `compute` that `/api/monitor` returns indices 0–3 with four distinct
VRAM readings, and that the sidebar labels GPU0–GPU3.

### 1c. Merge Service into the dashboard

Replace `index.html` + `service.html` with a single `server.html` modelled on
`llama-toolchest/web/templates/server.html`.

- Routes: `/` redirects to `/server`. Nav loses the separate "Service" entry;
  order becomes Server · Models · Download Models · Benchmarks · Tuning ·
  Settings · Help.
- **Server card** — active-model select, Start / Stop / Restart, a status badge
  polling `/api/service/status` every 5s that shows state *and uptime*, an
  inline `#server-action-error` box, and settings auto-save with a transient
  "saved" line. Port `onServerAction()` and the save-then-restart
  `restartServer()` flow; today a failed start is silent.
- **Dashboard cards partial** — Available Models (registry list with a
  copy-name icon per row and a `serving` tag on the running one), Inventory,
  API Endpoint (copy button, `Open Chat UI →` link, tool-use indicator).
- **Live Performance card** — port `partials/timings_summary.html`. Needs
  `MinPromptTokens`/`MaxPromptTokens` added to `RunningAverage` in
  `internal/benchmark/timing.go`; `TimingSample.PromptTokens` is already
  captured, so this is a min/max fold in `AddTiming`, not new instrumentation.
- **Server logs** — drop the bespoke 2s polling loop in `service.html` and use
  the existing SSE endpoint `/api/service/log-stream` with `log-panel.js`.
  Port llama's `data-clear-url` support into `web/static/log-panel.js` (the only
  difference between the two copies) so Clear also empties the server-side
  buffer. Live-tail switch, Copy, Clear in the panel header.
- Log section flexes to fill the viewport (`.server-page` / `.log-section`
  rules from llama's `server.html`).

### 1d. Models page rebuild

The largest single gap. Build on the `.model-card*` CSS from 1a.

- **Card list** replaces the table. Port `partials/model_card.html`: shared grid
  template so columns line up across cards, a `.model-card-header` row labelling
  them, `.model-card-orphan` / `.model-card-incomplete` states.
- **Active-model radio** in the toggle column, `PUT /api/models/{id}/activate`.
  Persist `active_model` in config. Show `↻` when the active model differs from
  the running one. Keep the existing `Enabled` flag as-is — it already gates the
  benchmark and probe model pickers (`bench_jobs.go:231`,
  `bench_probe.go:347`) and is orthogonal to which model is served.
- **Badges and links** — quant `<kbd>`, VRAM estimate (with the existing fit
  label), on-disk size, `tools` mark, `↗` link to the HF model page, `missing`
  for orphans.
- **Configure** button opens the config panel *inside* the card
  (`.model-card-config`), with the owning card's border turning accent. Today
  the model name doubles as the config link and the panel renders in a separate
  table row.
- **Remove** replaces Delete: a dialog offering *Keep Files* vs *Delete Files*.
  `Registry.Delete(id, deleteFiles bool)` already takes the flag; the handler
  needs a `?keep_files=true` query parameter. Also move the route from
  `DELETE /api/models/delete?id=` to `DELETE /api/models/{id}` to match the rest.
- **Filter box** — `filterModels()` over `data-search` attributes.
- **GPU allocation map** — port `partials/gpu_map.html` and `/api/gpu-map`.
  For vLLM the segments come from the running model's tensor-parallel shards
  (`tensor_parallel_size` × per-shard VRAM) rather than llama's per-model GPU
  assignment, but the bar/legend/OVER rendering is unchanged.
- **Downloads panel** — port `partials/downloads_panel.html` and
  `/api/hf/downloads-panel`. The downloader already writes `.part` files and
  resumes with an HTTP `Range` header (`downloader.go:299`) and has `Cancel`,
  so Pause / Resume / Discard need a scan for incomplete downloads
  (`GET`/`DELETE /api/hf/incomplete`) and the panel UI, not new transfer logic.
- **Config panel polish** — `.model-config-header` naming the owning model plus
  the "Restart required" note, `hx-on::response-error` surfacing a rejected
  save, and scroll preservation across the auto-save swap. Add an Aliases field
  mapping to vLLM's `--served-model-name`.

Not ported: the Embedding Models section and its preset downloader. vLLM can
serve embedding models with `--task embed`, so this is a plausible follow-up,
but it is not parity work.

---

## Phase 2 — Download / Browse parity — **done**

Kept vllm's quant-format filter and variant grouping — llama has no equivalent
— and fixed both of them along the way (see below).

- **Disk-space budget.** Done. `internal/huggingface/disk.go` uses
  `syscall.Statfs` rather than llama's gopsutil dependency: this project has
  two dependencies and reads system stats from `/proc` directly, so a library
  for one call would be out of character. Renders the "X available for new
  downloads (free · reserved)" line and refuses an oversized model as
  *Won't Fit* with the numbers in the tooltip.
- **Already-downloaded state.** Done. The panel offered "Download (28.8 GB)"
  for a model already in the registry.
- **Resume from the file list.** Done — the panel's button becomes
  "Resume (12.1 GB of 28.8 GB done)" when a stopped transfer left bytes behind.
- **Deferred VRAM estimates.** **Not needed — measured and dropped.** The file
  list renders in 0.04–0.33s. llama defers because it reads GGUF headers over
  the network; vllm computes from the `config.json` it has already fetched.

Two things this phase turned up that were not on the list:

- **The format filter only ever sieved one page.** It narrowed whatever 50
  repos a query returned by download count, so searching "qwen" for MXFP4 found
  nothing — not because there are none, but because none are popular enough to
  reach that page. The bucket's tags now go into the Hub query. Repeated
  `filter=` parameters AND rather than OR, so a bucket spanning several tags
  issues one request per tag and merges.
- **Formats outside a short list were mislabelled as FP16.** Detection matched
  three tags and six name suffixes; compressed-tensors, MXFP4, quark,
  auto-round, modelopt and fbgemm_fp8 all came back empty, and empty was
  rewritten to FP16. A `w4a16` search returned 36 repos badged FP16, every one
  4-bit. The search API returns `quantization_config` given `&config=true`, so
  detection asks that first, tags second, name last — the name is wrong often
  enough to matter, with 18 of 450 sampled repos carrying "AWQ" in the name
  while being compressed-tensors.

ModelScope as a second source remains uncommitted — a large surface
(`internal/modelscope`, `internal/modelsource`, per-source URL builders) for a
capability vllm-toolchest has never had.

---

## Phase 3 — Benchmarks rebuild — **done**

Structure, sweeps, export, comparison and visualization. Evaluation data
deliberately skipped: it depends on llama.cpp tooling with no vLLM equivalent.

### Outcome

Landed in six commits: job list and detail, sweeps, export, compare,
visualize, lifecycle. Notable departures from the plan:

- **Sweeps are not a port.** llama's `sweep.go` is mostly speculative-decoding
  modes and their parameter grammar, which vLLM has no counterpart for. The
  vLLM axes are the six engine-launch parameters, and because every one of
  them costs a reload, the runner now groups cells by (model, sweep values)
  rather than by model — and `ExpandCells` varies presets *inside* a sweep
  value, so eight cells over four configurations is four loads rather than
  eight. A test asserts that by counting the configuration changes in the
  emitted order.
- **Compare hides shared columns**, which was llama's #169. On the seeded
  five-run example that is four columns instead of fourteen. An absent value
  counts as variation, and runs that differ in nothing are called out — that
  usually means the wrong runs were selected.
- **Export is vLLM-shaped**, without llama's build, eval and memory columns.
  Runs now record the sweep values they were measured at; older runs take them
  from their cell on export.
- **Visualize** sends a description of the data rather than of a chart. Only
  varying dimensions become axes. The chart logic lives in
  `web/static/viz.js` so it can be unit-tested under node; `make test` runs
  those and skips where node is absent.

### Fixed along the way

`TestSubmitJobRejectsWhenAdHocRunActive` failed perhaps half the time —
verified on `e5188c3`, before any of this work — and not on its assertion. It
cancelled the run and returned, and `t.TempDir`'s cleanup then raced the run's
goroutines still writing the store: "directory not empty". It now waits.

### Not done

Batch delete across jobs (the selection is per-open-job, which is the only
scope the checkboxes have), and the evaluation datasets, which stay out of
scope.

---

## Phase 4 — Settings parity — **done**

Keep the Radiance section and the read-only Environment card as they are.

- **Backup & Restore.** Port `internal/backup/` (`backup.go`, `restore.go`),
  `internal/api/backup.go` and `partials/restore_report.html`. Export as JSON
  with an opt-in "include secrets" checkbox; restore with the client-side
  preview that parses the file, counts each section, warns when it contains
  secrets, and lets the user untick sections. Backed-up scope for vllm:
  settings, per-model vLLM configs, Radiance switches. Model configs keyed by HF
  identity so a backup restores onto another host, with unmatched configs held
  pending — port `internal/models/pending.go`.
- **Runtime environment editor.** Port the curated-variable table pattern from
  `internal/config/runtime_env.go` + `internal/api/runtime_env_view.go`, with a
  vLLM-relevant variable set (`VLLM_ATTENTION_BACKEND`, `VLLM_USE_V1`,
  `HIP_VISIBLE_DEVICES`, `PYTORCH_HIP_ALLOC_CONF`, `NCCL_*` / `RCCL_*`,
  `TORCH_BLAS_PREFER_HIPBLASLT`), a free-text extra-env box, and the effective-
  environment preview that shows which values the container environment
  overrides. Note the overlap with the Radiance section — Radiance switches are
  themselves env vars, so this phase should either subsume that section or
  clearly delimit it.
- **Auto-start on container startup.**
- **Storage** — read-only data directory plus an editable models-directory
  override with the "restart the service after changing this" warning.

### Outcome

All four parts landed, in two commits: settings/env/storage/auto-start, then
backup & restore.

**Runtime environment.** The curated set was built by reading the vLLM and
PyTorch installed in the image, not from documentation, and that changed the
list this plan called for:

- `VLLM_ATTENTION_BACKEND` and `VLLM_USE_V1` **do not exist** in this vLLM.
  The attention backend is a launch flag (`--attention-backend`, already set
  per model in the model config panel) and V0 is gone. Both would have been
  controls that do nothing.
- `VLLM_ROCM_USE_AITER` took their place. It defaults to off, so the AITER
  kernels the image spends most of its build time compiling are unused until
  someone turns them on — arguably the single highest-value knob on the page.
- `VLLM_DISABLE_COMPILE_CACHE` likewise: the image ships it at `1`, so every
  start recompiles from scratch.

**Precedence runs the opposite way to llama-toolchest.** vLLM is launched with
`append(os.Environ(), env...)` and os/exec resolves a duplicate name to its
last occurrence, so a value set in Settings *replaces* the image's rather than
being overridden by it. The effective-environment preview says "replaces X=Y"
accordingly. Porting llama's wording unchanged would have told operators their
change had no effect when it is the only thing taking effect.

**The Radiance overlap** was resolved by delimiting rather than subsuming.
Radiance keeps its section and is applied *after* the runtime environment, so
the visible tri-state control wins over a free-text box; a `RADIANCE_*` name
typed into the extra-environment box saves with a warning saying so. All four
launch paths (start, restart, benchmark jobs, context probe) now go through one
`launchEnv`, so a model benchmarked under one environment cannot be served
under another.

**Storage** needed real plumbing, not just a form field: `ModelDir` existed in
the config and was read by nothing, with `dataDir/models` hardcoded in five
places across two packages. A real `modelsDir` now threads through the registry
and the downloader, and the path is validated (absolute, exists, is a
directory) before it is stored.

**Backup & restore** is simpler than llama's: vLLM serves a repository rather
than a file within one, so a model config's identity is the HuggingFace repo ID
and nothing else — no quant, no filename, and no path relativization, since a
vLLM config holds no machine-local paths. What llama's GPU-assignment
normalization does, tensor parallel size does here: a config asking for four
GPUs on a two-GPU box does not fail at restore, it fails minutes into a model
load, so it is clamped with a warning and clamped *before* being held pending.

### Deviations

- No backend/platform dropdown over the environment table. llama has one
  because it has four build backends; here every variable is either universal
  or ROCm, so the platform is shown as a tag and there is nothing to filter.
- `VLLM_USE_TRITON_AWQ` is deliberately not curated — the image already exports
  it as `1` for every model. It stays reachable through the extra box.
- The restore report offers "Find" (deep-linking to the Download page,
  which now accepts `?q=`) rather than llama's direct download button: vLLM
  downloads a whole repository, so there is no single filename to fetch.

---

## Phase 5 — Help page

Port `web/templates/help.html` — the in-page TOC, section layout and styling —
and rewrite the prose for vLLM. Sections:

Quickstart · Core concepts · Server tab · Models · Model config ·
Download Models · Benchmarks · Tuning · Settings · Using the API ·
Troubleshooting

Core concepts differs most from llama's. Where llama explains *enabled vs
loaded* and *Max Loaded*, vllm needs: one process per model and what a restart
costs; `gpu_memory_utilization` and why the KV cache is pre-allocated;
`max_model_len` and how the context probe finds its ceiling;
tensor parallelism and the all-reduce token ceiling; served model names and
aliases; and quantization formats (FP8, AWQ, GPTQ, compressed-tensors) with
which ones RDNA4 actually accelerates.

---

## Cross-cutting, unscheduled

Small items with no natural home; fold into whichever phase touches them first.

- **`internal/broadcast`** — llama's mutex-protected fan-out with replayed
  history, shared by log and download streams. vllm has ad-hoc subscriber
  handling in `downloader.go` and `sse.go`; consolidating removes duplication
  and gives new subscribers the recent backlog.
- **Capabilities endpoint** — `/api/models/{id}/info` and the `meta` extension
  on `/v1/models`, so a client can self-configure in one round-trip.
- **`/api/ps`** — process listing.

---

## Sequencing

```
Phase 0  templates                  ── precondition, no visible change
Phase 1  1a shell/theme  1b rocm fix  1c server merge  1d models page
Phase 2  download/browse
Phase 3  benchmarks
Phase 4  settings
Phase 5  help
```

1b is independent of everything and can land first if a fix is wanted sooner —
it is the only item in this plan that repairs incorrect information rather than
adding missing UI.

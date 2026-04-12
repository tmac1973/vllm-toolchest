# vllm-toolchest Implementation Plan

A containerized vLLM inference server with a web UI for downloading models, configuring vLLM, running inference, and benchmarking. Modeled after [llama-toolchest](https://github.com/tmac1973/llama-toolchest), targeting AMD RDNA4 (R9700) first with NVIDIA support later.

---

## Architecture Overview

Same stack as llama-toolchest:
- **Backend:** Go (single static binary, chi router, embedded assets)
- **Frontend:** Server-side rendered HTML templates + htmx + Pico CSS
- **Real-time:** SSE for logs, progress, metrics
- **Container:** Fedora 43 base, ROCm "TheRock" nightly SDK
- **Dual-mode API:** Every endpoint returns HTML (htmx) or JSON based on `HX-Request` header

### Key Differences from llama-toolchest

| Aspect | llama-toolchest | vllm-toolchest |
|---|---|---|
| Backend engine | llama.cpp (C++, built from source at runtime) | vLLM (Python/PyTorch, pre-built in container) |
| Model format | GGUF files | HuggingFace transformers (safetensors) |
| Build management | Users compile llama.cpp via UI | No runtime compilation -- vLLM ships pre-built in container image |
| Server process | `llama-server` binary (router mode) | `vllm serve` Python process |
| OpenAI API | Reverse proxy to llama-server on :8080 | vLLM has native OpenAI-compatible API -- proxy/augment it |
| Model metadata | Custom GGUF header parser | HuggingFace `config.json` + `tokenizer_config.json` |
| Quantization | GGUF quant types (Q4_K_M, etc.) | AWQ, GPTQ, FP16, BF16, FP8 |
| VRAM estimation | Custom formula from GGUF metadata | Model config + dtype + context length calculation |
| Multi-model | llama-server router mode (hot swap) | One vLLM process per model (or use LoRA adapters) |
| GPU support | ROCm, CUDA, CPU | ROCm (RDNA4 first), CUDA later |

---

## Directory Structure

```
vllm-toolchest/
  cmd/
    vllmctl/main.go              # Main server entry point
  internal/
    api/                         # HTTP handlers
      server.go                  # Router setup, middleware
      proxy.go                   # Reverse proxy to vLLM API
      models.go                  # Model registry endpoints
      hf.go                      # HuggingFace search/download
      bench.go                   # Benchmark endpoints
      monitor.go                 # System metrics endpoints
      service.go                 # vLLM process control endpoints
      settings.go                # Settings endpoints
      sse.go                     # SSE fan-out helper
      middleware.go              # API key auth, logging
      respond.go                 # JSON/HTML response helper
      gpu_map.go                 # GPU allocation visualization
      dashboard.go               # Dashboard summary endpoint
    config/
      config.go                  # YAML config loading
    huggingface/
      client.go                  # HF API client (search, model info)
      downloader.go              # Resumable multi-file downloader
    models/
      registry.go                # Model registry (models.json)
      hfconfig.go                # HF config.json parser (architecture, layers, heads, etc.)
      vram.go                    # VRAM estimation from model config + dtype
      preset.go                  # Curated model presets
      gpu_assign.go              # Multi-GPU assignment logic
    benchmark/
      benchmark.go               # Benchmark orchestration
      runner.go                  # HTTP-based benchmark execution
      stats.go                   # Statistics computation
      context_probe.go           # Max context length probing (inspired by kyuz0)
    monitor/
      monitor.go                 # Unified metrics interface
      cpu.go                     # CPU metrics
      rocm.go                    # ROCm GPU metrics (rocm-smi)
      nvidia.go                  # NVIDIA GPU metrics (nvidia-smi) -- later
    process/
      manager.go                 # vLLM process lifecycle management
  web/
    embed.go                     # go:embed directives
    static/
      htmx.min.js
      htmx-sse.js
      pico.min.css
      log-panel.js
    templates/
      layout.html                # Shared layout with sidebar nav
      index.html                 # Dashboard
      models.html                # Model inventory + config
      models_browse.html         # HuggingFace search
      benchmarks.html            # Benchmark runs + comparison
      server.html                # vLLM process control
      settings.html              # Configuration
      partials/                  # htmx fragment templates
  scripts/
    test-*.sh                    # API test scripts
  plan/                          # This directory
  Dockerfile                     # ROCm RDNA4 (primary)
  Dockerfile.cuda                # NVIDIA CUDA (later)
  docker-compose.yml             # Default (ROCm)
  docker-compose.cuda.yml        # NVIDIA (later)
  docker-compose.models.yml      # Host model dir bind mount
  Makefile                       # Build/dev/docker commands
  setup.sh                       # Distro-agnostic installer
  .env.example
  go.mod
```

---

## Phases

### Phase 1: Project Scaffold & Container Foundation

**Goal:** Bootable container with vLLM working on RDNA4, Go binary serving a hello-world page.

**Container (Dockerfile):**
- Base: `fedora:43` (matches kyuz0 approach for Toolbx compatibility)
- ROCm: "TheRock" nightly SDK tarballs for gfx120X (from kyuz0's approach)
- Python 3.13 venv at `/opt/venv`
- PyTorch nightly from AMD staging index
- Flash Attention from `ROCm/flash-attention` (main_perf branch)
- vLLM built from source with RDNA4 patches:
  - amdsmi bypass (hard-code ROCm platform detection)
  - Mock amdsmi + force gfx1201 device identity
  - Compiler alignment (ROCm Clang for ABI compatibility with PyTorch)
  - tcmalloc LD_PRELOAD for double-free fix
  - bitsandbytes from ROCm fork
- Environment variables from kyuz0's profile.d scripts
- Go binary as entrypoint

**Go scaffold:**
- `go mod init github.com/tmac1973/vllm-toolchest`
- `cmd/vllmctl/main.go` -- chi router, serves on :3000
- Basic layout template with sidebar (copy from llama-toolchest)
- Embedded static assets (htmx, Pico CSS)
- Health check endpoint

**Docker Compose:**
- GPU passthrough: `/dev/kfd`, `/dev/dri`, `ipc: host`, `seccomp=unconfined`, video/render groups
- Volume: `/data` for config + models
- Ports: 3000 (UI), 8000 (vLLM API)

**Data layout inside container:**
```
/data/
  config/vllmctl.yaml       # Main config
  config/models.json         # Model registry
  config/benchmarks.json     # Benchmark results
  models/                    # HF model cache (or bind-mount ~/.cache/huggingface)
```

**Deliverable:** Container boots, Go UI loads at :3000, `vllm serve` can be run manually inside container.

---

### Phase 2: System Monitor & Dashboard

**Goal:** Real-time GPU/CPU/RAM monitoring and dashboard overview.

**Monitor subsystem** (copy + adapt from llama-toolchest):
- `monitor/rocm.go` -- Parse `rocm-smi` output for GPU utilization, VRAM usage, temperature, power
- `monitor/cpu.go` -- `/proc/stat` and `/proc/meminfo` parsing
- `GET /api/monitor` -- Current snapshot (JSON/HTML)
- `GET /api/monitor/stream` -- SSE stream (3s interval)
- Sidebar metrics bar (GPU util, VRAM, CPU, RAM) -- copy from llama-toolchest

**Dashboard** (`index.html`):
- Service status card (vLLM running/stopped)
- Active model card
- Model inventory count
- API endpoint display
- GPU info card (name, VRAM, driver version)
- Auto-refresh via htmx polling (5s)

**Deliverable:** Dashboard shows system state, sidebar shows live GPU metrics.

---

### Phase 3: HuggingFace Model Search & Download

**Goal:** Browse, search, and download models from HuggingFace.

**HuggingFace client** (adapt from llama-toolchest):
- `GET /api/hf/search?q=<query>` -- Search HF Hub (filter for text-generation models)
  - Unlike llama-toolchest (GGUF filter), search for transformer models with safetensors
  - Show model size, architecture, quantization availability (AWQ/GPTQ variants)
- `GET /api/hf/model?repo=<id>` -- Model details (files, sizes, config, README excerpt)
- `POST /api/hf/download` -- Start download
- `GET /api/hf/download/{id}/progress` -- SSE progress stream
- `DELETE /api/hf/download/{id}` -- Cancel download

**Downloader** (adapt from llama-toolchest):
- Resumable downloads (HTTP Range headers)
- Multi-file download (safetensors shards, config.json, tokenizer files, etc.)
- Progress tracking via SSE with fan-out
- Download to `/data/models/<org>/<repo>/` or leverage HF cache structure
- On completion: parse `config.json` to extract model metadata, register in model registry

**Browse page** (`models_browse.html`):
- Search bar with results list
- Model detail view with file listing and sizes
- Download button with real-time progress
- Show total download size and estimated VRAM requirement

**Deliverable:** Users can search HF, see model details, and download models with live progress.

---

### Phase 4: Model Registry & Configuration

**Goal:** Manage downloaded models and their vLLM configurations.

**Model registry** (`models/registry.go`):
- JSON store at `/data/config/models.json`
- Per-model metadata (from HF `config.json`):
  - Architecture, num_layers, hidden_size, num_attention_heads, num_kv_heads
  - Max position embeddings (context length)
  - Quantization type (FP16, BF16, AWQ, GPTQ, FP8)
  - Vocab size, model type
  - Vision capabilities (if applicable)
  - Tool/function calling support (from tokenizer_config.json chat_template)
- Per-model vLLM config:
  - `--dtype` (auto, float16, bfloat16)
  - `--max-model-len` (context length override)
  - `--tensor-parallel-size` (TP)
  - `--gpu-memory-utilization` (0.0-1.0)
  - `--enforce-eager` (disable CUDA graphs)
  - `--trust-remote-code`
  - `--max-num-seqs` (concurrency limit)
  - `--quantization` (awq, gptq, None)
  - `--enable-prefix-caching`
  - `--kv-cache-dtype` (auto, fp8)
  - Extra flags passthrough
- Startup maintenance:
  - Scan `/data/models/` for unregistered models
  - Verify registered models still exist on disk
  - Backfill metadata from config.json

**VRAM estimation** (`models/vram.go`):
- Formula based on: parameter count + KV cache (layers, heads, context) + activation memory
- Account for dtype (FP16=2 bytes, BF16=2, FP8=1, INT4=0.5)
- Show base (weights only) vs peak (full context KV cache) range
- "Fits" / "Needs TP=2" / "Too large" labels per GPU config

**HF config parser** (`models/hfconfig.go`):
- Parse `config.json` for architecture details
- Parse `tokenizer_config.json` for chat template (tool calling detection)
- Handle different model architectures (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, etc.)

**Models page** (`models.html`):
- List all registered models with status indicators
- Per-model expandable config panel:
  - Dtype selector
  - Context length slider (with VRAM impact display)
  - Tensor parallelism selector (1, 2, based on GPU count)
  - GPU memory utilization slider
  - Eager mode toggle
  - Concurrency limit
  - Quantization display
  - Prefix caching toggle
  - KV cache dtype selector
  - Extra flags text input
- Enable/disable toggle per model
- Delete model (files + registry entry)
- VRAM estimate display (updates live as config changes)

**Deliverable:** Full model inventory with per-model configuration UI and VRAM estimates.

---

### Phase 5: vLLM Process Management & Service Control

**Goal:** Start, stop, and manage the vLLM inference server.

**Process manager** (`process/manager.go`):
- Start vLLM: construct `vllm serve <model> [flags]` command from model config
- Key difference from llama-toolchest: **one vLLM process = one model** (no router mode)
  - Possible future: vLLM's model routing / multi-model support as it matures
- Process lifecycle: start, stop (SIGTERM -> SIGKILL), restart
- Stdout/stderr capture with ring buffer for log viewing
- Health check polling (`/health` endpoint on vLLM)
- Auto-restart on crash (configurable)
- Environment variable injection (ROCm vars, attention backend, etc.)
- **Model switching:** Stop current model, start new one
  - Future: support running multiple vLLM instances on different GPUs

**Service API:**
- `GET /api/service/status` -- Running/stopped, PID, uptime, active model, loaded model info
- `POST /api/service/start` -- Start vLLM with selected model
- `POST /api/service/stop` -- Graceful shutdown
- `POST /api/service/restart` -- Stop + start
- `GET /api/service/logs` -- Recent log lines
- `GET /api/service/log-stream` -- SSE log stream
- `GET /api/service/health` -- Proxied health check to vLLM

**Server page** (`server.html`):
- Status display (running/stopped, model name, uptime)
- Model selector dropdown (from enabled models in registry)
- Start/Stop/Restart buttons
- Live log viewer (SSE-powered, auto-scroll)
- Active configuration summary
- Quick actions: switch model, clear KV cache

**OpenAI-compatible proxy** (`api/proxy.go`):
- Reverse proxy from `:3000/v1/*` to vLLM on `:8000/v1/*`
- Purposes:
  - Single entry point (UI + API on same port)
  - API key authentication
  - Request/response logging
  - Timing capture for passive benchmarking
  - Sampling parameter injection (per-model defaults)
- `GET /v1/models` -- Return models from registry (not just the loaded one)
- Pass-through: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`

**Deliverable:** Start/stop vLLM from UI, view logs, proxy API requests through management port.

---

### Phase 6: Benchmarking

**Goal:** Benchmark inference performance with structured results and comparison.

**Benchmark engine** (adapt from llama-toolchest + inspired by kyuz0):
- Presets:
  - Quick (~30s): single prompt length, 1 rep
  - Standard (~3min): multiple prompt lengths, 3 reps
  - Thorough (~15min): full matrix, 5 reps, + vllm bench integration
- Test execution:
  1. Record hardware snapshot (GPU names, VRAM, driver version)
  2. Record model config snapshot
  3. Warmup request
  4. Run matrix: prompt_lengths x repetitions via `/v1/chat/completions`
  5. Optionally run `vllm bench throughput` and `vllm bench serve`
- Metrics captured per test:
  - Prompt tokens/sec (prefill speed)
  - Generation tokens/sec (decode speed)
  - Time to first token (TTFT) in ms
  - Total time in ms
  - Tokens generated
- Summary statistics: mean, min, max, p50, p95, p99

**Context probing** (inspired by kyuz0's `find_max_context.py`):
- `POST /api/benchmarks/probe-context` -- Find max safe context for current model+GPU config
- Iteratively test context lengths, detect OOM from vLLM logs
- Save verified configs for future reference

**Benchmark API:**
- `GET /api/benchmarks` -- List all benchmark runs
- `POST /api/benchmarks` -- Start new benchmark
- `GET /api/benchmarks/{id}` -- Get results
- `GET /api/benchmarks/{id}/progress` -- SSE progress stream
- `DELETE /api/benchmarks/{id}` -- Delete run
- `GET /api/benchmarks/compare` -- Side-by-side comparison
- `GET /api/benchmarks/export` -- CSV export

**Benchmarks page** (`benchmarks.html`):
- Preset selector (quick/standard/thorough)
- Model selector
- Run button with live progress
- Results table (per test point)
- Comparison view with bar charts
- CSV export button
- Passive timing display (from proxy captures)

**Deliverable:** Run structured benchmarks, view results, compare runs, export data.

---

### Phase 7: Settings & Configuration

**Goal:** Persistent settings with UI management.

**Settings:**
- API key (for `/v1/*` authentication)
- HuggingFace token (for gated model access)
- External URL (for displaying connection info)
- Default vLLM flags (applied to all models)
- Attention backend preference (Triton vs ROCm native)
- Auto-restart on crash (on/off)
- Log retention
- Theme selection (dark, light, terminal-green, terminal-amber, cyberpunk)

**Config file** (`/data/config/vllmctl.yaml`):
- Loaded at startup, saved on change
- Environment variable overrides for container configuration

**Settings page** (`settings.html`):
- Form fields for all settings
- Connection test button
- HF token test (validate against API)
- Save button

**Deliverable:** Persistent configuration with UI management.

---

### Phase 8: Setup Script & Polish

**Goal:** One-command setup for end users.

**setup.sh** (adapt from llama-toolchest):
- Detect Linux distro (Fedora, Ubuntu, Arch, etc.)
- Detect GPU (AMD RDNA4 specifically, with fallback to generic AMD/NVIDIA/CPU)
- Detect container runtime (Docker/Podman)
- Install prerequisites if missing
- Configure CDI for GPU passthrough
- Handle AMD-specific setup:
  - video/render group membership
  - HSA_OVERRIDE_GFX_VERSION if needed
  - ROCm driver verification
- Generate `.env` from detected configuration
- Pull/build container image
- Launch via docker-compose

**Makefile:**
- `make build` -- Build Go binary
- `make dev` -- Run locally for development
- `make docker` -- Build container image
- `make up` / `make down` -- Docker compose up/down
- `make logs` -- Tail container logs

**Deliverable:** Users can run `./setup.sh` and get a working instance.

---

### Phase 9: NVIDIA CUDA Support

**Goal:** Add NVIDIA GPU support as a second target.

- `Dockerfile.cuda` -- Based on `nvidia/cuda:12.x-devel-ubuntu24.04`
  - Standard vLLM pip install (no RDNA4 patches needed)
  - PyTorch with CUDA
  - Flash Attention via pip
- `docker-compose.cuda.yml` -- NVIDIA GPU passthrough (CDI or deploy.resources)
- `monitor/nvidia.go` -- Parse `nvidia-smi` for metrics
- Update setup.sh for NVIDIA detection
- Test matrix: single GPU, multi-GPU (TP)

**Deliverable:** Same UI/experience on NVIDIA hardware.

---

## What Carries Over Directly from llama-toolchest

These components can be copied with minimal changes:
- **Web layout & styling** -- `layout.html`, Pico CSS theme system, sidebar structure
- **htmx patterns** -- SSE setup, partial swapping, polling patterns
- **Static assets** -- htmx.min.js, htmx-sse.js, pico.min.css, log-panel.js
- **SSE infrastructure** -- `sse.go` fan-out writer
- **Monitor subsystem** -- `monitor/cpu.go`, `monitor/rocm.go` (nearly identical)
- **API patterns** -- `middleware.go`, `respond.go` (JSON/HTML dual-mode)
- **HuggingFace client** -- `huggingface/client.go` (search API is the same)
- **HuggingFace downloader** -- Core download logic (resume, progress, SSE)
- **Config loading** -- `config/config.go`
- **Benchmark statistics** -- `benchmark/stats.go`
- **GPU map visualization** -- `gpu_map.go`
- **Settings page** -- Most of the settings UI
- **Dashboard structure** -- Card layout, status display
- **Makefile structure** -- Build targets, GPU detection
- **Docker compose patterns** -- GPU passthrough configs
- **Setup script skeleton** -- Distro detection, runtime detection, group management

## What Needs New Implementation

- **vLLM process management** -- Different from llama-server (Python process, different CLI flags, different health check)
- **HF config parser** -- Replaces GGUF parser (read config.json instead of GGUF headers)
- **VRAM estimation** -- Different formula (transformer parameter counting vs GGUF metadata)
- **Model config UI** -- Different knobs (TP, gpu-memory-utilization, eager mode vs GPU layers, flash attention)
- **Dockerfile** -- Entirely different (Python/ROCm/vLLM build vs C++ toolchain)
- **Proxy adjustments** -- vLLM's native API is more complete than llama.cpp's, less augmentation needed
- **No build management** -- vLLM ships pre-built; the Builds page is replaced by a simpler version info/update mechanism
- **Context probing** -- New feature inspired by kyuz0 project
- **Multi-model strategy** -- vLLM doesn't have llama-server's router mode; need different approach (process-per-model or wait for vLLM multi-model support)

---

## Decisions

1. **Model storage strategy:** Custom directory (`/data/models/{org}/{repo}/`) as primary storage with human-readable file names. Additionally support optional bind-mount of host's HF cache (`~/.cache/huggingface/`) as a secondary read-only source so users can leverage existing downloads. Registry scanner checks both locations.
2. **Multi-model support:** Start with single vLLM instance (one model at a time). Design process manager interface to support future expansion to multiple concurrent instances on different GPUs/ports. Keep model switching fast (stop -> start).
3. **Container update strategy:** Support in-container `pip upgrade` for vLLM updates via a UI action (Settings or dedicated update panel). Track installed vs latest version. Full image rebuild remains an option for major updates or ROCm SDK changes.
4. **LoRA adapter support:** Deferred to a future phase. Start simple with base model loading only. Design model config schema to accommodate LoRA fields later.
5. **Speculative decoding:** Yes, include in model config UI. vLLM flags: `--speculative-model`, `--num-speculative-tokens`, `--speculative-draft-tensor-parallel-size`. Show in per-model config panel with draft model selector (from registered models) and token count setting.

## Additional Requirements

6. **Tool use / function calling:** vLLM supports OpenAI-compatible tool use. Per-model config includes `--enable-auto-tool-choice` and `--tool-call-parser` (hermes, llama3_json, granite, mistral, internlm, jamba, pythonic). UI shows tool-capable badge on models (detected from chat template). Proxy passes through tools/tool_choice/tool_calls transparently.
7. **Quantization format support:** Full support for AWQ, GPTQ, FP8, GGUF, BitsAndBytes (4-bit/8-bit), Marlin, SqueezeLLM, and compressed-tensors. Auto-detect quant method from model config files. Show quant type prominently in model list. VRAM estimation accounts for per-format bytes-per-parameter. Marlin auto-upgrade for compatible GPTQ/AWQ models.

# Phase 1: Project Scaffold & Container Foundation

## Goal

Stand up the full project skeleton: Go module with chi router, embedded htmx + Pico CSS frontend, HTML template system, and a multi-stage Dockerfile that produces a working container with vLLM built from source for AMD RDNA4 (gfx1201). At the end of this phase the container boots, the Go web UI loads on `:3000`, and `python -m vllm.entrypoints.openai.api_server` can be invoked manually inside the container to confirm the vLLM build works.

---

## 1. Go Module & Directory Structure

### Module path

```
module github.com/tmac1973/vllm-toolchest
```

Go version: `go 1.25` (matches llama-toolchest). Single dependency beyond stdlib: `github.com/go-chi/chi/v5` and `gopkg.in/yaml.v3`.

### Directory layout

```
vllm-toolchest/
  cmd/
    vllmctl/
      main.go              # entry point — mirrors llama-toolchest/cmd/llamactl/main.go
  internal/
    api/
      server.go            # Server struct, NewServer(), buildRouter(), render()
      sse.go               # SSEWriter (copy from llama-toolchest verbatim)
      middleware.go         # apiKeyAuth middleware (copy from llama-toolchest)
      respond.go           # respondJSON(), respondHTML(), isHTMX() helpers (copy)
      monitor.go           # handleMonitorStatus, handleMonitorStream
    config/
      config.go            # Config struct + Load()
    monitor/
      monitor.go           # Monitor struct, polling loop, subscribe/unsubscribe
      rocm.go              # ROCm GPU metrics (copy from llama-toolchest)
      cpu.go               # CPU + memory metrics (copy from llama-toolchest verbatim)
    vllm/
      (placeholder — Phase 2+)
    huggingface/
      (placeholder — Phase 3)
    models/
      (placeholder — Phase 3+)
  web/
    embed.go               # embed.FS declarations (copy pattern from llama-toolchest)
    static/
      htmx.min.js          # copy from llama-toolchest
      htmx-sse.js          # copy from llama-toolchest
      pico.min.css          # copy from llama-toolchest
    templates/
      layout.html          # adapted from llama-toolchest (rename branding, adjust nav)
      index.html           # dashboard page (skeleton with placeholder cards)
      partials/
        monitor_bar.html   # copy from llama-toolchest
        placeholder.html   # generic "coming soon" partial
  Dockerfile               # THE Dockerfile (TheRock + vLLM from source)
  docker-compose.yml       # ROCm GPU passthrough compose
  .env.example             # all configurable environment variables
  .gitignore
  Makefile
  README.md
  plan/                    # these plan documents
```

### What to copy verbatim from llama-toolchest

These files are generic infrastructure with no llama.cpp-specific logic:

| File | Notes |
|------|-------|
| `internal/api/sse.go` | SSEWriter, StreamLines — no changes needed |
| `internal/api/respond.go` | respondJSON/respondHTML/isHTMX — rename package import path only |
| `internal/api/middleware.go` | apiKeyAuth — rename import path only |
| `internal/monitor/monitor.go` | Monitor struct, polling, subscribe/fan-out — change import path |
| `internal/monitor/rocm.go` | ROCm sysfs + rocm-smi parsing — copy as-is |
| `internal/monitor/cpu.go` | /proc/stat, /proc/meminfo — copy as-is |
| `web/embed.go` | embed.FS declarations — copy as-is |
| `web/static/*` | htmx.min.js, htmx-sse.js, pico.min.css — copy as-is |

### What to adapt from llama-toolchest

| File | Changes |
|------|---------|
| `cmd/vllmctl/main.go` | Rename binary, remove builder/benchmark init, adjust initDataDir to create `/data/config`, `/data/models` only (no `/data/builds`) |
| `internal/config/config.go` | Remove `ActiveBuild`, `ModelsMax`, `LlamaPort`. Add `VLLMPort` (default 8000), `VLLMHost` (default "127.0.0.1"), `ToolUseEnabled` (bool, default true), `DefaultQuantFormat` (string), `MaxModelLen` (int, optional), `TensorParallelSize` (int, default 1) |
| `internal/api/server.go` | Remove builder, benchmark, process references. Add placeholder route groups. Simplify template function map (no GGUF-specific funcs). New nav items in sidebar. |
| `web/templates/layout.html` | Change branding to "vLLM Toolchest", update nav links (remove Builds/Benchmarks/Server, add Chat, Models, Browse HF, Service, Settings), keep theme system and monitor bar |

### What to build new

| File | Purpose |
|------|---------|
| `internal/vllm/` package | Will hold vLLM process manager, API client — placeholder for now |
| `internal/models/` package | Model registry without GGUF-specific logic — placeholder for now |

---

## 2. Entry Point (`cmd/vllmctl/main.go`)

Mirrors llama-toolchest exactly in structure:

```
func main()
  - flag.String("config", "/data/config/vllmctl.yaml", "config file path")
  - config.Load()
  - initDataDir() — creates /data/config, /data/models
  - api.NewServer(cfg)
  - http.Server on cfg.ListenAddr
  - signal.NotifyContext for graceful shutdown
```

The `initDataDir` function creates:
- `/data/config` — YAML config, model registry JSON
- `/data/models` — downloaded model files (mirroring HF structure)

No `/data/builds` directory (vLLM is pre-built in the container image).

---

## 3. Config Struct

```go
type Config struct {
    ListenAddr         string `yaml:"listen_addr"`          // default ":3000"
    DataDir            string `yaml:"data_dir"`             // default "/data"
    VLLMPort           int    `yaml:"vllm_port"`            // default 8000
    VLLMHost           string `yaml:"vllm_host"`            // default "127.0.0.1"
    ExternalURL        string `yaml:"external_url"`         // default "http://localhost:3000"
    HFToken            string `yaml:"hf_token"`             // optional
    APIKey             string `yaml:"api_key"`              // optional, gates /v1/*
    LogLevel           string `yaml:"log_level"`            // default "info"
    ToolUseEnabled     bool   `yaml:"tool_use_enabled"`     // default true
    DefaultQuantFormat string `yaml:"default_quant_format"` // e.g. "awq", "gptq", "fp8", ""
    MaxModelLen        int    `yaml:"max_model_len"`        // 0 = use model default
    TensorParallelSize int    `yaml:"tensor_parallel_size"` // default 1
    GPUMemoryUtil      float64 `yaml:"gpu_memory_util"`     // default 0.90
    EnforceEager       bool   `yaml:"enforce_eager"`        // default false (use CUDA graphs)
}
```

Key differences from llama-toolchest Config:
- No `ActiveBuild` (vLLM is always the same binary)
- No `ModelsMax` (vLLM handles model loading internally)
- `VLLMPort` replaces `LlamaPort` (default 8000 vs 8080 — vLLM's default)
- New fields for vLLM-specific launch parameters
- `ToolUseEnabled` controls whether `--enable-auto-tool-choice` and `--tool-call-parser` flags are passed to vLLM

---

## 4. Chi Router Skeleton

### Middleware stack (same as llama-toolchest)

```go
r.Use(middleware.Logger)
r.Use(middleware.Recoverer)
r.Use(middleware.Compress(5))
```

### Route groups for Phase 1

```go
// Static assets
r.Handle("/static/*", ...)

// Pages (full HTML renders)
r.Get("/", s.handleIndex)           // Dashboard — functional in Phase 1
// These render page skeletons with "coming soon" content:
r.Get("/models", s.handleModelsPage)
r.Get("/models/browse", s.handleModelsBrowsePage)
r.Get("/service", s.handleServicePage)
r.Get("/chat", s.handleChatPage)
r.Get("/settings", s.handleSettingsPage)

// Health check
r.Get("/healthz", s.handleHealthCheck)

// API routes
r.Route("/api", func(r chi.Router) {
    r.Get("/monitor", s.handleMonitorStatus)
    r.Get("/monitor/stream", s.handleMonitorStream)
    r.Get("/dashboard", s.handleDashboard)
    // Placeholder route groups for future phases:
    r.Route("/models", func(r chi.Router) { /* Phase 2-3 */ })
    r.Route("/hf", func(r chi.Router) { /* Phase 3 */ })
    r.Route("/service", func(r chi.Router) { /* Phase 2 */ })
    r.Route("/settings", func(r chi.Router) { /* Phase 2+ */ })
})

// OpenAI-compatible proxy (Phase 2)
r.Route("/v1", func(r chi.Router) {
    r.Use(s.apiKeyAuth)
    // Will proxy to vLLM's /v1 endpoints
})
```

### Health check endpoint

```
GET /healthz
Response: 200 {"status": "ok", "version": "0.1.0"}
```

Simple liveness probe. Does NOT check vLLM status (that's Phase 2). Just confirms the Go process is alive and serving.

---

## 5. Template System

### Copy from llama-toolchest

The template parsing pattern (`parseTemplates()`) is copied directly. It uses a base template (layout.html + all partials) that gets cloned per page, so each page's `{{define "content"}}` block doesn't collide.

### layout.html adaptations

Changes from llama-toolchest's layout.html:

1. **Branding**: "vLLM Toolchest" instead of "Llama Toolchest", subtitle "vLLM Inference Manager"
2. **Nav links**:
   - Dashboard (`/`) 
   - Models (`/models`) — registered models management
   - Search HF (`/models/browse`) — HuggingFace browser
   - Service (`/service`) — vLLM process control
   - Chat (`/chat`) — embedded chat interface (or link to vLLM's)
   - Settings (`/settings`)
3. **Remove**: Builds, Benchmarks, Server links (llama.cpp-specific concepts)
4. **Keep**: Theme system (dark/light/terminal-green/terminal-amber/cyberpunk), monitor bar div with `hx-get="/api/monitor"`, all CSS custom properties
5. **Keep**: localStorage theme persistence JavaScript

### index.html (Dashboard skeleton)

Phase 1 dashboard shows:
- Service status card: "vLLM: Stopped" (hardcoded for now, wired up in Phase 2)
- GPU info card: populated from monitor metrics (GPU name, VRAM)
- Model inventory card: "0 models registered" (wired up in Phase 3)
- API endpoint card: shows `{ExternalURL}/v1` 

Dashboard cards use `hx-get="/api/dashboard"` with `hx-trigger="load, every 5s"` (same pattern as llama-toolchest).

### Placeholder pages

All non-dashboard pages render the layout with a simple "This feature is coming in Phase N" message inside the content block. They're listed in the nav so the UI feels complete from day one.

---

## 6. Dockerfile — The Core Build

This is the most complex and critical part of Phase 1. The Dockerfile builds vLLM from source against TheRock nightly ROCm SDK with all patches needed for gfx1201 (RDNA4 R9700 XT).

### Multi-stage structure

```
Stage 1: go-builder    — golang:1.25-bookworm, builds static Go binary
Stage 2: vllm-builder  — Fedora 43, installs TheRock ROCm SDK, builds vLLM + deps
Stage 3: runtime        — Fedora 43 minimal, copies Go binary + Python venv
```

**Decision**: Use a 2-stage approach instead of 3-stage for the initial implementation. The vllm-builder stage IS the runtime stage because:
- vLLM's ROCm dependencies are extensive (ROCm libs, HIP runtime, etc.)
- Separating builder from runtime gains minimal image size savings
- Simplifies debugging during development

Revisit this if image size becomes a concern (it will be large — 15-20GB is normal for ROCm + vLLM).

### Stage 1: Go Builder

```dockerfile
FROM golang:1.25-bookworm AS go-builder
WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o vllmctl ./cmd/vllmctl
```

Identical pattern to llama-toolchest. Static binary, no CGO.

### Stage 2: vLLM Builder / Runtime (Fedora 43)

#### Base and system packages

```dockerfile
FROM registry.fedoraproject.org/fedora:43

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
    python3.13 python3.13-devel python3-pip \
    aria2 wget curl git-core \
    gcc gcc-c++ make cmake ninja-build lld \
    clang clang-devel compiler-rt libomp-devel \
    openssl-devel zlib-devel \
    gperftools-libs \
    rocminfo \
    && dnf clean all
```

Key packages:
- `python3.13` + `python3.13-devel` — Fedora 43 ships Python 3.13
- `aria2` — parallel chunk downloads for TheRock SDK tarball
- `gcc gcc-c++ make cmake ninja-build lld clang clang-devel compiler-rt libomp-devel` — build toolchain for flash-attention, vLLM C++ extensions
- `gperftools-libs` — provides `libtcmalloc.so` for LD_PRELOAD (vLLM perf optimization)
- `rocminfo` — GPU detection utility

#### TheRock ROCm SDK Installation

TheRock is AMD's open-source ROCm SDK built from source. The nightly tarballs are hosted on GitHub releases.

```dockerfile
# TheRock nightly ROCm SDK — pinned to a known-good nightly
ARG THEROCK_DATE=2026-04-10
ARG THEROCK_URL=https://github.com/ROCm/TheRock/releases/download/nightly%2F${THEROCK_DATE}/therock-dist-nightly_${THEROCK_DATE}_linux-x86_64-gfx1201.tar.xz

RUN mkdir -p /opt/rocm && \
    aria2c -x16 -s16 -k1M -d /tmp -o therock.tar.xz "${THEROCK_URL}" && \
    tar xf /tmp/therock.tar.xz -C /opt/rocm --strip-components=1 && \
    rm /tmp/therock.tar.xz
```

Notes on TheRock installation:
- `aria2c -x16 -s16` — 16 connections, 16 splits for fast parallel download (the tarball is ~2-3GB)
- `-k1M` — 1MB minimum split size
- The tarball extracts to a directory structure matching `/opt/rocm/` layout
- `--strip-components=1` if the tarball has a top-level directory wrapper (verify with actual nightly)
- **Pin to a specific nightly date** via build arg — TheRock nightlies can break, so we need reproducibility
- The tarball is arch-specific: `gfx1201` for RDNA4

#### ROCm Environment Variables

```dockerfile
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
    PYTORCH_ROCM_ARCH=gfx1201 \
    HIP_ARCHITECTURES=gfx1201 \
    AMDGPU_TARGETS=gfx1201 \
    HSA_OVERRIDE_GFX_VERSION=12.0.1 \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH \
    CMAKE_PREFIX_PATH=/opt/rocm
```

#### Python Virtual Environment

```dockerfile
RUN python3.13 -m venv /opt/vllm-venv
ENV PATH=/opt/vllm-venv/bin:$PATH \
    VIRTUAL_ENV=/opt/vllm-venv
```

Using a venv inside the container keeps system Python clean and makes dependency management explicit.

#### PyTorch Nightly (AMD ROCm)

```dockerfile
# PyTorch nightly from AMD staging index (TheRock-compatible builds)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

**Decision point**: The exact index URL depends on which ROCm version TheRock nightly targets. TheRock nightlies as of early 2026 target ROCm 6.3-6.4 APIs. The PyTorch nightly `rocm6.4` wheel should be compatible. If TheRock moves to ROCm 7.x APIs, this URL changes.

Alternative: AMD sometimes hosts staging wheels at `https://download.pytorch.org/whl/nightly/rocm-staging/`. Check the kyuz0/amd-r9700-vllm-toolboxes project for the exact URL they use.

#### Flash Attention (ROCm fork, main_perf branch)

```dockerfile
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_INTERNAL_USE_RTN=1

RUN pip install --no-cache-dir packaging ninja setuptools wheel && \
    git clone --depth 1 -b main_perf https://github.com/ROCm/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    python setup.py install && \
    rm -rf /tmp/flash-attention
```

The `main_perf` branch of ROCm/flash-attention contains AMD-optimized Triton-based flash attention kernels. The `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` env var activates the Triton codepath instead of the CK (Composable Kernel) codepath — this is critical for gfx1201 which may not have CK support yet.

#### bitsandbytes (ROCm fork)

```dockerfile
# bitsandbytes from ROCm fork — enables BnB 4-bit and 8-bit quantization on AMD GPUs
RUN git clone --depth 1 -b rocm_enabled_multi_backend \
        https://github.com/ROCm/bitsandbytes.git /tmp/bitsandbytes && \
    cd /tmp/bitsandbytes && \
    pip install --no-cache-dir -e . && \
    rm -rf /tmp/bitsandbytes/.git
```

This branch has multi-backend support that works with TheRock's ROCm. Without this, BitsAndBytes quantization (4-bit NF4, 8-bit LLM.int8) is unavailable on AMD.

#### vLLM Build From Source

This is the most patch-heavy step. vLLM's ROCm support for gfx1201 requires several workarounds from the kyuz0/amd-r9700-vllm-toolboxes project.

```dockerfile
# Clone vLLM
ARG VLLM_VERSION=main
RUN git clone --depth 1 -b ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /tmp/vllm

WORKDIR /tmp/vllm

# --- Patch 1: amdsmi bypass ---
# vLLM tries to import amdsmi (AMD System Management Interface) at build time
# and at runtime. TheRock SDK doesn't include amdsmi. Create a mock module.
RUN mkdir -p /opt/vllm-venv/lib/python3.13/site-packages/amdsmi && \
    cat > /opt/vllm-venv/lib/python3.13/site-packages/amdsmi/__init__.py << 'PYEOF'
"""Mock amdsmi module for gfx1201 builds where amdsmi is unavailable."""

class AmdSmiException(Exception):
    pass

def amdsmi_init():
    pass

def amdsmi_shut_down():
    pass

def amdsmi_get_processor_handles():
    return []

def amdsmi_get_gpu_device_id(handle):
    return 0

def amdsmi_get_gpu_asic_info(handle):
    class Info:
        market_name = "AMD Radeon RX 9700 XT"
        device_id = 0x7480
    return Info()

def amdsmi_get_gpu_vram_usage(handle):
    class Usage:
        vram_used = 0
        vram_total = 16 * 1024 * 1024 * 1024
    return Usage()

AMDSMI_INIT_ALL_SOCKETS = 0
AMDSMI_PROCESSOR_TYPE_AMD_GPU = 1
PYEOF

# --- Patch 2: Force gfx1201 target in vLLM's CMake ---
# vLLM's setup.py / CMakeLists.txt may not recognize gfx1201. Force it.
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=gfx1201 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    VLLM_USE_TRITON_AWQ=1

# --- Patch 3: Compiler ABI alignment ---
# TheRock uses ROCm Clang with a specific C++ ABI. If vLLM's C++ extensions
# are compiled with the system GCC, there can be ABI mismatches.
# Force the ROCm Clang compiler for C++ extension builds.
ENV CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++ \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin

# --- Build vLLM ---
RUN pip install --no-cache-dir -r requirements-rocm.txt && \
    python setup.py develop 2>&1 | tee /tmp/vllm-build.log

# Verify the build
RUN python -c "import vllm; print(f'vLLM {vllm.__version__} built successfully')"
```

#### Patch details — what each one solves

**Patch 1: Mock amdsmi**
- `amdsmi` is AMD's system management library (GPU enumeration, telemetry)
- It's part of the full ROCm SDK but NOT part of TheRock nightly
- vLLM imports it at startup for GPU detection
- The mock provides enough surface area for vLLM to initialize without crashing
- GPU metrics in our UI come from `rocm-smi` / sysfs (the monitor subsystem), not amdsmi

**Patch 2: Force gfx1201**
- vLLM's build system queries PyTorch for the ROCm architecture target
- If PyTorch doesn't know about gfx1201, it falls back to a default (gfx90a, etc.)
- `PYTORCH_ROCM_ARCH=gfx1201` forces the correct ISA target
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables experimental AOTriton kernels that may include gfx1201 support
- `VLLM_USE_TRITON_AWQ=1` forces Triton-based AWQ dequantization (needed because the CUDA AWQ kernels don't exist for ROCm, and the HIP port may not have gfx1201 support)

**Patch 3: Compiler ABI alignment**
- TheRock's PyTorch is built with ROCm Clang (`/opt/rocm/llvm/bin/clang++`)
- If vLLM's C++ extensions are built with system GCC, the C++ ABI may differ (libstdc++ vs libc++, or different `_GLIBCXX_USE_CXX11_ABI` settings)
- Setting `CC`/`CXX` to ROCm Clang ensures ABI compatibility
- This is the most common cause of "undefined symbol" errors at runtime

#### tcmalloc LD_PRELOAD

```dockerfile
# tcmalloc improves vLLM's memory allocation performance significantly
ENV LD_PRELOAD=/usr/lib64/libtcmalloc.so.4
```

This is a well-known vLLM optimization. The glibc malloc has lock contention issues under vLLM's allocation pattern. tcmalloc (from gperftools) resolves this. The `LD_PRELOAD` is set globally in the container environment.

#### All environment variables (consolidated)

```dockerfile
# --- Consolidated environment for runtime ---
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
    PYTORCH_ROCM_ARCH=gfx1201 \
    HIP_ARCHITECTURES=gfx1201 \
    AMDGPU_TARGETS=gfx1201 \
    HSA_OVERRIDE_GFX_VERSION=12.0.1 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_INTERNAL_USE_RTN=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    VLLM_USE_TRITON_AWQ=1 \
    VLLM_TARGET_DEVICE=rocm \
    LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
    PATH=/opt/vllm-venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    VIRTUAL_ENV=/opt/vllm-venv
```

#### Final assembly

```dockerfile
# Copy Go binary from builder stage
COPY --from=go-builder /app/vllmctl /usr/local/bin/vllmctl

# Create data directories
RUN mkdir -p /data/config /data/models

# Default config
RUN cat > /data/config/vllmctl.yaml << 'EOF'
listen_addr: ":3000"
data_dir: "/data"
vllm_port: 8000
log_level: "info"
tool_use_enabled: true
gpu_memory_util: 0.90
EOF

VOLUME ["/data"]

# vllmctl web UI
EXPOSE 3000
# vLLM inference API
EXPOSE 8000

ENTRYPOINT ["vllmctl", "--config", "/data/config/vllmctl.yaml"]
```

---

## 7. Docker Compose

### `docker-compose.yml`

```yaml
services:
  vllm-toolchest:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        THEROCK_DATE: "2026-04-10"
        VLLM_VERSION: "main"
    container_name: vllm-toolchest
    ports:
      - "${VLLMCTL_PORT:-3000}:3000"
      - "${VLLMCTL_INFERENCE_PORT:-8000}:8000"
    volumes:
      - vllmctl-data:/data:z
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - "${HOST_VIDEO_GID:-video}"
      - "${HOST_RENDER_GID:-render}"
    ipc: host
    security_opt:
      - seccomp=unconfined
    ulimits:
      memlock:
        soft: -1
        hard: -1
    env_file:
      - path: .env
        required: false
    restart: unless-stopped

volumes:
  vllmctl-data:
    name: vllmctl-data
    driver: local
```

Key differences from llama-toolchest's compose:
- Default inference port is 8000 (vLLM) not 8080 (llama-server)
- Build args for TheRock date and vLLM version pinning
- No separate `docker-compose.rocm.yml` — this project is ROCm-only (RDNA4 target)
- Could add a `docker-compose.models.yml` override for bind-mounting host model directory (same pattern as llama-toolchest)

### GPU passthrough details

- `/dev/kfd` — AMD's Kernel Fusion Driver, required for all ROCm GPU compute
- `/dev/dri` — DRM render nodes (`/dev/dri/renderD128` etc.), needed for GPU memory mapping
- `ipc: host` — ROCm uses shared memory segments for GPU communication between processes
- `seccomp=unconfined` — ROCm's HSA runtime uses syscalls that Docker's default seccomp profile blocks
- `ulimits.memlock: -1` — ROCm pins GPU memory, which counts against memlock limits; unlimited prevents OOM
- `group_add: video, render` — ensures the container process can access GPU device files

---

## 8. Data Volume Layout

```
/data/
  config/
    vllmctl.yaml         # main configuration
    models.json          # model registry (Phase 3)
  models/
    {org}/
      {repo}/
        config.json
        tokenizer.json
        tokenizer_config.json
        *.safetensors      # or *.gguf, *.bin
        ...
```

Model storage mirrors HuggingFace directory structure: `/data/models/{org}/{repo}/`. This is different from llama-toolchest which uses `/data/models/{org}--{repo}/{filename}.gguf` because vLLM loads models by directory path, not individual files. vLLM expects the standard HF model layout with `config.json`, tokenizer files, and weight shards in the same directory.

---

## 9. `.env.example`

```ini
# vllm-toolchest environment configuration
#
# Copy this file to .env and edit as needed:
#   cp .env.example .env

# ─── Port configuration ─────────────────────────────────────────────
VLLMCTL_PORT=3000
VLLMCTL_INFERENCE_PORT=8000

# ─── Model storage ──────────────────────────────────────────────────
# Bind-mount a host directory for model persistence across container rebuilds.
# Leave commented to use Docker volume.
#VLLMCTL_MODELS_DIR=/home/user/llm-models

# ─── HuggingFace ────────────────────────────────────────────────────
# Required for gated models (Llama, Mistral, etc.)
#HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ─── API Key ────────────────────────────────────────────────────────
# Protect the /v1 proxy endpoint with a Bearer token
#VLLMCTL_API_KEY=

# ─── vLLM defaults ──────────────────────────────────────────────────
# These can also be set in the web UI (Settings page)
#VLLM_GPU_MEMORY_UTIL=0.90
#VLLM_MAX_MODEL_LEN=0
#VLLM_TENSOR_PARALLEL_SIZE=1
#VLLM_TOOL_USE_ENABLED=true

# ─── AMD ROCm settings ──────────────────────────────────────────────
# Auto-detected. Only override if needed.
#HSA_OVERRIDE_GFX_VERSION=12.0.1

# Host group IDs for GPU device access inside the container.
#HOST_VIDEO_GID=
#HOST_RENDER_GID=

# ─── TheRock / Build pinning ────────────────────────────────────────
# Override at build time: docker compose build --build-arg THEROCK_DATE=2026-04-10
#THEROCK_DATE=2026-04-10
#VLLM_VERSION=main
```

---

## 10. Makefile

```makefile
.PHONY: build run dev docker docker-rebuild up down logs clean

# Local development
build:
	go build -o bin/vllmctl ./cmd/vllmctl

run: build
	./bin/vllmctl --config config.yaml

dev:
	go run ./cmd/vllmctl --config config.yaml

# Container
docker:
	docker compose build

docker-rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	rm -rf bin/
```

Simpler than llama-toolchest's Makefile — no GPU vendor auto-detection (always ROCm), no PID file management, no agent binary.

---

## 11. `.gitignore`

```
bin/
*.exe
.env
/data/
*.log
```

---

## 12. Edge Cases & Decisions

### TheRock nightly availability
- TheRock nightlies are not guaranteed to be available or stable
- The `THEROCK_DATE` build arg pins to a specific date
- If the nightly URL structure changes, the `aria2c` download will fail with a clear HTTP error
- **Mitigation**: Document the last known-good nightly date in README. Consider caching the tarball in a local registry for CI.

### gfx1201 not in PyTorch's target list
- PyTorch nightly may not officially list gfx1201 as a supported target
- The `PYTORCH_ROCM_ARCH` env var forces it, but compilation may produce warnings
- If PyTorch's HIP kernels don't have gfx1201 codegen, they fall back to LLVM JIT compilation (slower first-run, then cached)
- **Mitigation**: The `HSA_OVERRIDE_GFX_VERSION=12.0.1` ensures the HSA runtime recognizes the GPU

### Build time
- Full vLLM from-source build with flash-attention and bitsandbytes: expect 30-60 minutes
- Use Docker build cache aggressively (`--mount=type=cache`)
- Layer ordering matters: TheRock SDK (rarely changes) before Python deps (change more often) before vLLM source (changes most often)

### Container image size
- Expected: 15-25GB. This is normal for ROCm + ML framework images.
- The TheRock SDK alone is 2-3GB, PyTorch is 2-3GB, vLLM with compiled kernels is 1-2GB, plus system packages
- Multi-stage separation of Go builder saves ~1GB (Go toolchain not in final image)

### vLLM version pinning
- `VLLM_VERSION=main` tracks head — good for development, bad for reproducibility
- For production, pin to a specific commit SHA or release tag
- The build arg makes this switchable without editing the Dockerfile

### Tool use / function calling readiness
- Phase 1 only sets `tool_use_enabled` in the config
- The actual `--enable-auto-tool-choice --tool-call-parser hermes` flags are passed when starting vLLM (Phase 2)
- This phase ensures the config infrastructure is in place

### Quantization format support readiness
- AWQ: Enabled via `VLLM_USE_TRITON_AWQ=1` env var (Triton-based AWQ dequant)
- GPTQ: vLLM uses AutoGPTQ or Marlin kernels; works if vLLM compiles successfully
- FP8: Requires ROCm 6.2+ with FP8 support in the GPU ISA; gfx1201 has native FP8
- GGUF: vLLM added GGUF support; works out of the box for single-file models
- BitsAndBytes: Enabled by the bitsandbytes-rocm fork installation
- Marlin: ROCm Marlin kernels may need explicit enabling; check at vLLM build time

---

## 13. Validation Criteria

Phase 1 is complete when:

1. `make docker` builds the container image without errors
2. `make up` starts the container
3. `curl http://localhost:3000/healthz` returns `{"status":"ok"}`
4. Browser navigating to `http://localhost:3000` shows the dashboard with:
   - Sidebar navigation with all pages listed
   - Monitor bar showing GPU metrics (VRAM, utilization)
   - Dashboard cards (service status, GPU info, model count, API endpoint)
5. All nav links render their placeholder pages without errors
6. Inside the container (`docker exec -it vllm-toolchest bash`):
   - `python -c "import vllm; print(vllm.__version__)"` succeeds
   - `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
   - `python -c "import bitsandbytes"` succeeds
   - `rocminfo` shows the gfx1201 GPU agent
   - Manual test: `python -m vllm.entrypoints.openai.api_server --model /path/to/small/model --dtype float16` starts and serves requests (requires a pre-downloaded model)

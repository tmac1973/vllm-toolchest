# Phase 9: NVIDIA CUDA Support

**Goal:** Add NVIDIA GPU support as a second target, providing the same UI and API experience as the ROCm build. NVIDIA is the "easy path" -- standard vLLM pip install, well-supported CUDA kernels, no compiler hacks.

---

## 1. Dockerfile.cuda

File: `Dockerfile.cuda`

### Base Image

```
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS vllm-builder
```

Use the `-devel` variant (not `-runtime`) because:
- vLLM's pip install may compile custom CUDA kernels (flash-attn, vllm C++ extensions).
- Auto-GPTQ and AutoAWQ build CUDA extensions during pip install.
- After building, the final stage can use a slimmer base.

Pin to a specific CUDA 12.8.x version for reproducibility. CUDA 12.8 supports all current NVIDIA GPUs (Ampere, Ada Lovelace, Hopper, Blackwell). Update to 12.9+ as available.

### Multi-Stage Build

**Stage 1: vLLM + Python dependencies**

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS vllm-builder

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git wget curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_NO_CACHE_DIR=1

# PyTorch with CUDA (standard index)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# vLLM -- standard pip install (no source build needed for NVIDIA)
RUN pip install vllm

# Flash Attention -- standard pip install (NVIDIA CUDA kernels)
RUN pip install flash-attn --no-build-isolation

# Quantization libraries
RUN pip install auto-gptq autoawq
# bitsandbytes -- standard pip install has CUDA support by default
RUN pip install bitsandbytes
# SqueezeLLM support (optional, small package)
RUN pip install squeezellm

# No compiler hacks needed
# No tcmalloc workaround needed
# No amdsmi mock needed
# No HSA_OVERRIDE_GFX_VERSION needed
# Standard GCC works fine (no ROCm Clang ABI issues)
```

**Key differences from ROCm Dockerfile:**

| Aspect | ROCm (Dockerfile) | CUDA (Dockerfile.cuda) |
|--------|-------------------|------------------------|
| Base image | `fedora:43` + TheRock nightly SDK | `nvidia/cuda:12.8-devel-ubuntu24.04` |
| PyTorch | AMD staging nightly index | Standard PyTorch CUDA index |
| vLLM | Built from source with RDNA4 patches | Standard `pip install vllm` |
| Flash Attention | `ROCm/flash-attention` fork (main_perf) | Standard `pip install flash-attn` |
| bitsandbytes | ROCm fork (`bitsandbytes-rocm`) | Standard `pip install bitsandbytes` |
| AWQ kernels | `VLLM_USE_TRITON_AWQ=1` (Triton fallback) | Native CUDA AWQ kernels (default) |
| Compiler | ROCm Clang (ABI match with PyTorch) | System GCC (standard) |
| tcmalloc | `LD_PRELOAD` workaround for double-free | Not needed |
| amdsmi | Mock library for device detection | Not applicable |
| Device nodes | `/dev/kfd`, `/dev/dri/renderD*` | NVIDIA driver handles device access |

**Stage 2: Go builder (identical to ROCm)**

```dockerfile
FROM golang:1.24-bookworm AS go-builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w -X main.version=${VERSION}" \
    -o /build/vllmctl ./cmd/vllmctl
```

**Stage 3: Final runtime image**

```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Runtime dependencies only (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python venv from builder
COPY --from=vllm-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Go binary
COPY --from=go-builder /build/vllmctl /usr/local/bin/vllmctl

# Copy web assets
COPY --from=go-builder /build/web /opt/vllm-toolchest/web

# Data directory
RUN mkdir -p /data/config /data/models
VOLUME /data

# Ports
EXPOSE 3000 8000

ENTRYPOINT ["/usr/local/bin/vllmctl"]
```

Note the final stage uses `-runtime` instead of `-devel` to reduce image size (runtime has CUDA libraries but not compilers/headers). The compiled Python extensions from Stage 1 are self-contained in the venv.

---

## 2. docker-compose.cuda.yml

File: `docker-compose.cuda.yml`

### Docker GPU Passthrough

```yaml
version: "3.8"
services:
  vllm-toolchest:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    image: vllm-toolchest:cuda
    container_name: vllm-toolchest
    ports:
      - "${UI_PORT:-3000}:3000"
    volumes:
      - ${DATA_DIR:-./data}:/data
      - ${MODEL_DIR:-./models}:/data/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - VLLMCTL_API_KEY=${VLLMCTL_API_KEY:-}
      - VLLMCTL_HF_TOKEN=${VLLMCTL_HF_TOKEN:-}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: ${GPU_COUNT:-all}
              capabilities: [gpu]
    restart: unless-stopped
    ipc: host
```

**Key differences from ROCm compose:**

| Aspect | ROCm (docker-compose.yml) | CUDA (docker-compose.cuda.yml) |
|--------|---------------------------|-------------------------------|
| GPU passthrough | `/dev/kfd`, `/dev/dri`, group GIDs | `deploy.resources.reservations.devices` |
| Device access | Explicit device nodes + ipc + seccomp | NVIDIA runtime handles it |
| Environment | `HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, `VLLM_USE_TRITON_AWQ` | `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES` |
| Security | `security_opt: seccomp=unconfined` | Not needed |
| IPC | `ipc: host` (shared memory for ROCm) | `ipc: host` (shared memory for NCCL multi-GPU) |

### Podman GPU Passthrough (NVIDIA)

For Podman, NVIDIA Container Toolkit provides CDI (Container Device Interface) support:

```yaml
# Podman variant (detected at runtime, not a separate file)
# Instead of deploy.resources, use:
services:
  vllm-toolchest:
    devices:
      - nvidia.com/gpu=all
```

The CDI spec is managed by `nvidia-ctk cdi generate` and stored at `/etc/cdi/nvidia.yaml`.

### Multi-GPU Configuration

For multi-GPU setups, the compose file supports:

```yaml
environment:
  # Limit to specific GPUs by UUID or index
  - NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
  # For 2-GPU TP=2 setup: both GPUs visible
  # For single-GPU: NVIDIA_VISIBLE_DEVICES=0
```

No changes to compose file needed for multi-GPU -- `count: all` exposes all GPUs. The vLLM `--tensor-parallel-size` flag (set per-model in the registry) controls how many GPUs vLLM actually uses.

---

## 3. monitor/nvidia.go

File: `internal/monitor/nvidia.go`

### nvidia-smi Parsing

Use `nvidia-smi --query-gpu=...` with CSV output for reliable parsing.

**Query command:**
```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,fan.speed,clocks.current.graphics,clocks.current.memory,pci.bus_id,compute_cap --format=csv,noheader,nounits
```

**Parsed fields per GPU:**

```go
type NvidiaGPUMetrics struct {
    Index            int     `json:"index"`
    Name             string  `json:"name"`              // e.g., "NVIDIA GeForce RTX 4090"
    UUID             string  `json:"uuid"`              // GPU UUID for stable identification
    VRAMTotalMB      int     `json:"vram_total_mb"`
    VRAMUsedMB       int     `json:"vram_used_mb"`
    VRAMFreeMB       int     `json:"vram_free_mb"`
    GPUUtilPercent   int     `json:"gpu_util_percent"`  // 0-100
    MemUtilPercent   int     `json:"mem_util_percent"`  // 0-100
    TemperatureC     int     `json:"temperature_c"`
    PowerDrawW       float64 `json:"power_draw_w"`
    PowerLimitW      float64 `json:"power_limit_w"`
    FanSpeedPercent  int     `json:"fan_speed_percent"` // 0-100, "[N/A]" for blower-less
    GraphicsClockMHz int     `json:"graphics_clock_mhz"`
    MemoryClockMHz   int     `json:"memory_clock_mhz"`
    PCIBusID         string  `json:"pci_bus_id"`        // e.g., "00000000:01:00.0"
    ComputeCap       string  `json:"compute_cap"`       // e.g., "8.9" for Ada Lovelace
}
```

**Parsing implementation:**

```go
func (n *NvidiaMonitor) Collect() ([]NvidiaGPUMetrics, error) {
    cmd := exec.Command("nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,"+
            "utilization.gpu,utilization.memory,temperature.gpu,power.draw,"+
            "power.limit,fan.speed,clocks.current.graphics,clocks.current.memory,"+
            "pci.bus_id,compute_cap",
        "--format=csv,noheader,nounits",
    )
    // Parse CSV output, one row per GPU
    // Handle "[N/A]" values (fan speed on passively cooled GPUs, etc.)
    // Handle "[Not Supported]" for fields not available on all GPUs
}
```

**Multi-GPU support:**

The CSV output contains one row per GPU. Parse all rows into a `[]NvidiaGPUMetrics` slice. The monitor endpoint returns the full slice. The sidebar metrics bar shows an aggregate (sum VRAM, avg/max util) or per-GPU breakdown depending on UI space.

**Fallback when nvidia-smi is not available:**

If `nvidia-smi` binary is not in PATH or returns an error:
- Return an error on first call, log warning.
- Subsequent calls return cached empty result without re-executing.
- The monitor endpoint returns `{"gpus": [], "error": "nvidia-smi not available"}`.
- The UI shows "GPU monitoring unavailable" in the sidebar.

**Process metrics (optional enhancement):**

For vLLM-specific GPU metrics, also query process-level GPU usage:
```bash
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits
```
This shows per-process VRAM usage, so we can distinguish vLLM's usage from system overhead.

---

## 4. GPU-Agnostic Abstraction

### Monitor Interface

File: `internal/monitor/monitor.go`

Define a common interface that both ROCm and NVIDIA monitors implement:

```go
// GPUInfo represents static GPU properties (detected at startup)
type GPUInfo struct {
    Index        int    `json:"index"`
    Name         string `json:"name"`
    Vendor       string `json:"vendor"`       // "amd" | "nvidia"
    VRAMTotalMB  int    `json:"vram_total_mb"`
    PCIBusID     string `json:"pci_bus_id"`
    Architecture string `json:"architecture"` // "gfx1201" | "sm_89" (compute cap)
    DriverVersion string `json:"driver_version"`
}

// GPUMetrics represents dynamic GPU metrics (sampled periodically)
type GPUMetrics struct {
    Index           int     `json:"index"`
    VRAMUsedMB      int     `json:"vram_used_mb"`
    VRAMFreeMB      int     `json:"vram_free_mb"`
    GPUUtilPercent  int     `json:"gpu_util_percent"`
    MemUtilPercent  int     `json:"mem_util_percent"`
    TemperatureC    int     `json:"temperature_c"`
    PowerDrawW      float64 `json:"power_draw_w"`
    FanSpeedPercent int     `json:"fan_speed_percent"`   // -1 if unavailable
    GraphicsClockMHz int    `json:"graphics_clock_mhz"`  // 0 if unavailable
    MemoryClockMHz   int    `json:"memory_clock_mhz"`    // 0 if unavailable
}

// GPUMonitor is the interface both backends implement
type GPUMonitor interface {
    // Detect checks if this monitor's GPU vendor is present
    Detect() bool

    // Info returns static GPU properties (called once at startup)
    Info() ([]GPUInfo, error)

    // Metrics returns current GPU metrics (called on poll interval)
    Metrics() ([]GPUMetrics, error)

    // Vendor returns "amd" or "nvidia"
    Vendor() string
}
```

### Auto-Detection at Startup

File: `internal/monitor/detect.go`

```go
func NewGPUMonitor() (GPUMonitor, error) {
    // Try NVIDIA first (nvidia-smi is more commonly available on host)
    nvidia := &NvidiaMonitor{}
    if nvidia.Detect() {
        return nvidia, nil
    }

    // Try AMD
    rocm := &ROCmMonitor{}
    if rocm.Detect() {
        return rocm, nil
    }

    // No GPU monitoring available
    return &NoopMonitor{}, nil
}
```

Detection logic:
- `NvidiaMonitor.Detect()`: Run `nvidia-smi --query-gpu=count --format=csv,noheader` and check exit code 0.
- `ROCmMonitor.Detect()`: Run `rocm-smi --showid` and check exit code 0. Also check `/dev/kfd` exists.
- `NoopMonitor`: Returns empty slices for all methods, never errors. Used when no GPU is detected (CPU-only mode).

### VRAM Estimation Independence

The VRAM estimation logic in `internal/models/vram.go` is already GPU-agnostic. It calculates memory requirements based on model parameters and dtype, not GPU vendor. The only GPU-specific input is `VRAMTotalMB` per GPU, which comes from the `GPUInfo` struct.

```go
func EstimateVRAM(model *ModelConfig, gpus []GPUInfo) VRAMEstimate {
    // Parameter memory = param_count * bytes_per_param(dtype)
    // KV cache memory = layers * kv_heads * head_dim * 2 * context_len * kv_dtype_bytes
    // Activation memory = estimate based on batch_size * hidden_size
    // Total per GPU = (parameter_memory / tp_size) + kv_cache + activations + overhead

    // "Fits" calculation uses gpus[0].VRAMTotalMB (assumes homogeneous GPUs)
    // Works identically for AMD and NVIDIA
}
```

---

## 5. Quantization Differences on NVIDIA

### Marlin Kernels

Marlin was originally developed for NVIDIA GPUs and has the best support there.

**Behavior:**
- On NVIDIA with `prefer_marlin=true` (from settings): Marlin auto-upgrade works the same as ROCm.
- Marlin kernels are faster on NVIDIA due to mature CUDA kernel implementations.
- No environment variable overrides needed (unlike `VLLM_USE_TRITON_AWQ` on ROCm).

**Compatibility matrix for Marlin on NVIDIA:**

| Model Type | Marlin Compatible | Notes |
|------------|------------------|-------|
| GPTQ 4-bit, desc_act=false | Yes | Best case, significant speedup |
| GPTQ 4-bit, desc_act=true | No | Falls back to standard GPTQ |
| GPTQ 8-bit | No | Marlin only supports 4-bit |
| GPTQ 3-bit | No | Marlin only supports 4-bit |
| AWQ 4-bit | Yes | Via AWQ-Marlin bridge |
| AWQ (other bits) | No | Falls back to standard AWQ |

### FP8 Quantization

FP8 is natively supported on NVIDIA Ada Lovelace (compute capability 8.9+) and Hopper (9.0+).

**Compute capability check:**

```go
func SupportsFP8Native(computeCap string) bool {
    // Parse "8.9" -> major=8, minor=9
    // FP8 native: major >= 9 (Hopper) or (major == 8 && minor >= 9) (Ada)
    // Older GPUs (Ampere 8.0, Turing 7.5): FP8 via software emulation (slower)
}
```

**Impact on settings:**
- On Ada Lovelace+ GPUs: FP8 models and FP8 KV cache work at full speed.
- On Ampere GPUs (RTX 3090, A100): FP8 works but may not be faster than FP16. Log a note: "FP8 is emulated on this GPU. Consider FP16 or AWQ 4-bit instead."
- On Turing or older: FP8 is not recommended. Display warning in model config UI.

The `GET /api/settings/gpu-info` response includes `compute_cap`, and the UI can use this to show/hide FP8 recommendations.

### AWQ / GPTQ

**NVIDIA advantages:**
- ExLlama v2 kernels available for GPTQ: faster than default kernels. vLLM auto-selects these when available.
- AWQ uses native CUDA kernels (no `VLLM_USE_TRITON_AWQ` fallback needed).
- Both AWQ and GPTQ have been extensively optimized for NVIDIA.

**Config differences:**
- Remove `VLLM_USE_TRITON_AWQ=1` from environment (only needed on ROCm).
- No `--enforce-eager` needed for AWQ/GPTQ on NVIDIA (CUDA graphs work reliably).

### BitsAndBytes

- Standard `pip install bitsandbytes` supports CUDA out of the box.
- No ROCm fork needed.
- 4-bit and 8-bit quantization work on all CUDA-capable GPUs.
- NF4 (Normal Float 4) quantization type supported.

### Quantization Recommendation Logic

In the model config UI, show quantization recommendations based on detected GPU:

```go
func QuantRecommendations(gpuInfo GPUInfo, modelParams int64) []Recommendation {
    if gpuInfo.Vendor == "nvidia" {
        // Ada Lovelace+ (RTX 4090, etc.)
        if computeCapMajor >= 9 || (computeCapMajor == 8 && computeCapMinor >= 9) {
            // FP8 is fast and high quality
            // Marlin for 4-bit GPTQ/AWQ
            // AWQ with native CUDA kernels
        }
        // Ampere (RTX 3090, A100)
        if computeCapMajor == 8 && computeCapMinor < 9 {
            // AWQ 4-bit or GPTQ 4-bit with Marlin
            // BitsAndBytes 4-bit for on-the-fly quantization
            // FP8 possible but not optimal
        }
    }
    if gpuInfo.Vendor == "amd" {
        // RDNA4: Triton AWQ, GPTQ, FP8 (when supported)
        // Marlin may have limited ROCm support
    }
}
```

---

## 6. Tool Use on NVIDIA

Tool use / function calling works identically on NVIDIA and AMD. There are no GPU-specific concerns.

**Verification:**
- The `--enable-auto-tool-choice` and `--tool-call-parser` flags are vLLM process flags, not GPU-dependent.
- The tool calling logic is in vLLM's Python layer, not in GPU kernels.
- The same model + parser combinations work on both platforms.

**Testing:**
- The tool use API test (`scripts/test-proxy-tools.sh`) should pass identically on both platforms.
- Include tool use in the NVIDIA integration test matrix.

---

## 7. Setup.sh Updates for NVIDIA

### NVIDIA Driver Detection

```bash
detect_nvidia() {
    # Check for NVIDIA GPU via lspci
    if ! lspci -nn | grep -qi nvidia; then
        return 1
    fi

    # Check nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        NVIDIA_GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        NVIDIA_GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        NVIDIA_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        NVIDIA_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    else
        echo "WARNING: nvidia-smi not found. NVIDIA driver may not be installed."
        return 1
    fi

    # Check minimum driver version
    # CUDA 12.8 requires driver >= 570.x
    REQUIRED_DRIVER_MAJOR=570
    ACTUAL_DRIVER_MAJOR=$(echo "$NVIDIA_DRIVER_VERSION" | cut -d. -f1)
    if [ "$ACTUAL_DRIVER_MAJOR" -lt "$REQUIRED_DRIVER_MAJOR" ]; then
        echo "WARNING: NVIDIA driver $NVIDIA_DRIVER_VERSION is too old."
        echo "CUDA 12.8 requires driver >= ${REQUIRED_DRIVER_MAJOR}.x"
        echo "Update your driver: https://www.nvidia.com/drivers"
        return 1
    fi

    GPU_TYPE="nvidia"
    GPU_COUNT="$NVIDIA_GPU_COUNT"
    GPU_NAME="$NVIDIA_GPU_NAME"
    return 0
}
```

### NVIDIA Container Toolkit Detection and Installation

```bash
detect_nvidia_container_toolkit() {
    # Check for nvidia-ctk
    if command -v nvidia-ctk >/dev/null 2>&1; then
        NVIDIA_CTK_VERSION=$(nvidia-ctk --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
        echo "NVIDIA Container Toolkit $NVIDIA_CTK_VERSION detected."
        return 0
    fi

    # Check for nvidia-container-runtime (older name)
    if command -v nvidia-container-runtime >/dev/null 2>&1; then
        echo "NVIDIA Container Runtime detected (older toolkit)."
        return 0
    fi

    return 1
}

install_nvidia_container_toolkit() {
    echo "Installing NVIDIA Container Toolkit..."
    case "$DISTRO_ID" in
        fedora)
            # NVIDIA's Fedora repo
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            sudo dnf install -y nvidia-container-toolkit
            ;;
        ubuntu|debian)
            # NVIDIA's apt repo
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
                sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
            ;;
        arch)
            # AUR or NVIDIA's repo
            echo "Install nvidia-container-toolkit from AUR:"
            echo "  yay -S nvidia-container-toolkit"
            echo "  OR paru -S nvidia-container-toolkit"
            return 1
            ;;
        opensuse*)
            sudo zypper install -y nvidia-container-toolkit
            ;;
    esac
}
```

### CDI Configuration for Podman (NVIDIA)

```bash
configure_nvidia_cdi() {
    if [ "$CONTAINER_RUNTIME" != "podman" ]; then
        return 0  # Docker uses deploy.resources, not CDI
    fi

    echo "Generating NVIDIA CDI spec for Podman..."
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

    # Verify CDI spec
    if [ -f /etc/cdi/nvidia.yaml ]; then
        echo "NVIDIA CDI spec generated at /etc/cdi/nvidia.yaml"
        # List available CDI devices
        nvidia-ctk cdi list
    else
        echo "ERROR: Failed to generate CDI spec."
        return 1
    fi
}
```

### Docker Runtime Configuration (NVIDIA)

For Docker, configure the NVIDIA runtime:

```bash
configure_nvidia_docker() {
    if [ "$CONTAINER_RUNTIME" != "docker" ]; then
        return 0
    fi

    # Configure NVIDIA runtime for Docker
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    # Verify
    docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
    if [ $? -eq 0 ]; then
        echo "Docker NVIDIA runtime configured successfully."
    else
        echo "ERROR: Docker NVIDIA GPU access failed."
        return 1
    fi
}
```

### Compose File Selection

```bash
if [ "$GPU_TYPE" = "nvidia" ]; then
    COMPOSE_FILE="docker-compose.cuda.yml"
    # No HSA_OVERRIDE_GFX_VERSION needed
    # No VLLM_USE_TRITON_AWQ needed
    # No render/video group GID needed
else
    COMPOSE_FILE="docker-compose.yml"
fi
```

---

## 8. Testing Matrix

### Single GPU Targets

| GPU | Arch | Compute Cap | VRAM | Key Tests |
|-----|------|-------------|------|-----------|
| RTX 3090 | Ampere | 8.6 | 24GB | AWQ/GPTQ, BitsAndBytes, no native FP8 |
| RTX 3090 Ti | Ampere | 8.6 | 24GB | Same as 3090 |
| RTX 4090 | Ada Lovelace | 8.9 | 24GB | Full suite: FP8 native, Marlin, AWQ, GPTQ |
| RTX 4080 | Ada Lovelace | 8.9 | 16GB | VRAM-constrained tests |
| A100 40GB | Ampere | 8.0 | 40GB | Data center GPU, large models |
| A100 80GB | Ampere | 8.0 | 80GB | Very large models, FP16 70B |
| H100 | Hopper | 9.0 | 80GB | FP8 native, fastest inference |

### Multi-GPU Targets (TP=2)

| Configuration | Total VRAM | Key Tests |
|---------------|------------|-----------|
| 2x RTX 3090 | 48GB | 70B AWQ 4-bit with TP=2 |
| 2x RTX 4090 | 48GB | 70B AWQ 4-bit with TP=2, Marlin |
| 2x A100 80GB | 160GB | 70B FP16, 405B AWQ 4-bit |

### Test Cases Per GPU

**Basic functionality:**
1. Container starts and UI is accessible.
2. GPU detected correctly: name, VRAM, driver version, compute capability.
3. `nvidia-smi` monitoring returns valid metrics.
4. Download a small model (Qwen2.5-0.5B).
5. Start vLLM, health check passes.
6. Inference via `/v1/chat/completions` returns valid response.
7. Stop vLLM cleanly.

**Quantization-specific:**
8. Load AWQ 4-bit model, verify `--quantization awq` flag.
9. Load GPTQ 4-bit model with `prefer_marlin=true`, verify `--quantization marlin` flag.
10. Load GPTQ 4-bit model with `desc_act=true`, verify Marlin NOT used (fallback to gptq).
11. Load FP8 model on Ada Lovelace+, verify inference works.
12. Load FP8 model on Ampere, verify works but log performance note.
13. Load BitsAndBytes 4-bit model, verify `--quantization bitsandbytes` flag.
14. FP8 KV cache: set `kv_cache_dtype=fp8`, verify it works.

**Tool use:**
15. Start model with `--enable-auto-tool-choice --tool-call-parser hermes`.
16. Send function calling request, verify `tool_calls` in response.
17. Send follow-up with tool result, verify incorporation.

**Multi-GPU (if available):**
18. Configure model with `tensor_parallel_size=2`.
19. Start vLLM, verify both GPUs utilized (check nvidia-smi).
20. Inference works correctly with TP=2.
21. VRAM usage distributed across GPUs.

**Performance baseline:**
22. Run "quick" benchmark preset.
23. Verify tokens/sec is reasonable for the GPU/model combination.
24. Compare with ROCm results for the same model (informal, not automated).

### Verify Same UI/API Behavior

The critical verification: the UI and API must behave identically regardless of GPU vendor. Specifically:

- All API endpoints return the same response shapes.
- Model config UI shows the same options (except GPU-specific notes).
- Benchmark results use the same format and statistics.
- Settings page works the same (different GPU info panel content, same structure).
- SSE streams work identically.
- Theme system works identically.
- Log viewer works identically.

The API test scripts (`scripts/test-*.sh`) should pass without modification on both ROCm and CUDA builds. Any GPU-specific behavior must be behind the `GPUMonitor` interface.

---

## 9. Future Considerations

### Unified Dockerfile with Build Args

Instead of maintaining two separate Dockerfiles, consider a unified approach:

```dockerfile
ARG GPU_TARGET=rocm  # rocm | cuda

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS cuda-base
FROM fedora:43 AS rocm-base

FROM ${GPU_TARGET}-base AS base
# ... rest of build
```

**Pros:** Single Dockerfile to maintain, less drift between builds.
**Cons:** Docker multi-stage FROM selection is limited; complex conditional logic is harder to read than two separate files. ARG-based FROM selection has edge cases with build caching.

**Recommendation:** Keep separate Dockerfiles for now. The ROCm build is significantly more complex (TheRock SDK, patches, compiler hacks) and mixing it with the simple CUDA build would make both harder to maintain. Revisit when ROCm RDNA4 support stabilizes and the ROCm Dockerfile simplifies.

### CI/CD: Build and Push Both Images

**GitHub Actions workflow:**

```yaml
# .github/workflows/build.yml
jobs:
  build-cuda:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build CUDA image
        run: docker build -f Dockerfile.cuda -t ghcr.io/tmac1973/vllm-toolchest:latest-cuda .
      - name: Push
        run: docker push ghcr.io/tmac1973/vllm-toolchest:latest-cuda

  build-rocm:
    runs-on: ubuntu-latest  # Note: ROCm build may need self-hosted runner with more resources
    steps:
      - uses: actions/checkout@v4
      - name: Build ROCm image
        run: docker build -f Dockerfile -t ghcr.io/tmac1973/vllm-toolchest:latest-rocm .
      - name: Push
        run: docker push ghcr.io/tmac1973/vllm-toolchest:latest-rocm
```

**Image tagging strategy:**
- `latest-rocm`, `latest-cuda` -- latest builds
- `v1.0.0-rocm`, `v1.0.0-cuda` -- versioned releases
- `main-rocm`, `main-cuda` -- nightly from main branch

**Build time considerations:**
- CUDA build: ~15-20 minutes (pip installs are straightforward).
- ROCm build: ~45-90 minutes (TheRock SDK download, vLLM source build, flash-attention compile).
- Consider building ROCm image on a self-hosted runner with build caching.

### ARM Support

**NVIDIA Jetson (AGX Orin, etc.):**
- JetPack SDK includes CUDA and container runtime.
- vLLM has experimental Jetson support.
- Would need `Dockerfile.jetson` based on `nvcr.io/nvidia/l4t-pytorch` base image.
- Limited VRAM (32-64GB unified memory) constrains model sizes.
- ARM64 Go cross-compilation: `GOARCH=arm64 go build ...`

**NVIDIA Grace Hopper (GH200):**
- 96GB+ HBM3 unified memory.
- Standard CUDA build should work (x86_64 or arm64 depending on host CPU).
- Would need testing but no expected code changes.

**Apple Silicon (M-series):**
- No CUDA or ROCm.
- vLLM has experimental Metal/MPS support (very early).
- Not a near-term target. Users on macOS should use llama.cpp/llama-toolchest.

**Priority:** Jetson and Grace Hopper are niche. ARM support is a Phase 10+ consideration, only if there's user demand. The architecture is already GPU-agnostic (via the `GPUMonitor` interface), so adding a new backend is straightforward.

### Intel Arc / oneAPI

- Intel Arc GPUs (A770, B580) are gaining support in PyTorch via Intel Extension for PyTorch (IPEX).
- vLLM has early Intel GPU support via `--device xpu`.
- Would need `Dockerfile.intel` based on Intel's oneAPI base images.
- Another Phase 10+ item, but the abstraction layers are already in place.

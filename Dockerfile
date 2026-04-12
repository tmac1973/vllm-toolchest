# ── Stage 1: Go builder ─────────────────────────────────────────────
FROM golang:1.26-bookworm AS go-builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o vllmctl ./cmd/vllmctl

# ── Stage 2: vLLM runtime (Fedora 43 + TheRock ROCm) ───────────────
FROM registry.fedoraproject.org/fedora:43

# ── System packages ─────────────────────────────────────────────────
RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
    python3.13 python3.13-devel python3-pip \
    aria2 wget curl git-core \
    gcc gcc-c++ make cmake ninja-build lld \
    clang clang-devel compiler-rt libomp-devel \
    openssl-devel zlib-devel \
    gperftools-libs \
    rocminfo \
    && dnf clean all

# ── TheRock nightly ROCm SDK ────────────────────────────────────────
ARG THEROCK_DATE=2026-04-10
ARG THEROCK_URL=https://github.com/ROCm/TheRock/releases/download/nightly%2F${THEROCK_DATE}/therock-dist-nightly_${THEROCK_DATE}_linux-x86_64-gfx1201.tar.xz

RUN mkdir -p /opt/rocm && \
    aria2c -x16 -s16 -k1M -d /tmp -o therock.tar.xz "${THEROCK_URL}" && \
    tar xf /tmp/therock.tar.xz -C /opt/rocm --strip-components=1 && \
    rm /tmp/therock.tar.xz

# ── ROCm environment ───────────────────────────────────────────────
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
    PYTORCH_ROCM_ARCH=gfx1201 \
    HIP_ARCHITECTURES=gfx1201 \
    AMDGPU_TARGETS=gfx1201 \
    HSA_OVERRIDE_GFX_VERSION=12.0.1 \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    CMAKE_PREFIX_PATH=/opt/rocm

# ── Python venv ────────────────────────────────────────────────────
RUN python3.13 -m venv /opt/vllm-venv
ENV PATH=/opt/vllm-venv/bin:$PATH \
    VIRTUAL_ENV=/opt/vllm-venv

RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# ── PyTorch nightly (AMD ROCm) ─────────────────────────────────────
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm6.4

# ── Flash Attention (ROCm fork) ───────────────────────────────────
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_INTERNAL_USE_RTN=1

RUN git clone --depth 1 -b main_perf \
        https://github.com/ROCm/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    python setup.py install && \
    rm -rf /tmp/flash-attention

# ── bitsandbytes (ROCm fork) ──────────────────────────────────────
ENV BNB_ROCM_ARCH=gfx1201
RUN git clone --depth 1 -b rocm_enabled_multi_backend \
        https://github.com/ROCm/bitsandbytes.git /tmp/bitsandbytes && \
    cd /tmp/bitsandbytes && \
    cmake -S . -B build \
        -DCOMPUTE_BACKEND=hip \
        -DBNB_ROCM_ARCH=gfx1201 \
        -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ && \
    cmake --build build -j4 && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/bitsandbytes

# ── Mock amdsmi (required for vLLM on TheRock) ────────────────────
RUN mkdir -p /opt/vllm-venv/lib/python3.13/site-packages/amdsmi && \
    cat > /opt/vllm-venv/lib/python3.13/site-packages/amdsmi/__init__.py << 'PYEOF'
"""Mock amdsmi module for gfx1201 builds where amdsmi is unavailable."""

class AmdSmiException(Exception):
    pass

def amdsmi_init(*args, **kwargs):
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
        vram_total = 32 * 1024 * 1024 * 1024
    return Usage()

AMDSMI_INIT_ALL_SOCKETS = 0
AMDSMI_PROCESSOR_TYPE_AMD_GPU = 1
PYEOF

# ── vLLM from source ──────────────────────────────────────────────
ARG VLLM_VERSION=main

ENV VLLM_TARGET_DEVICE=rocm \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    VLLM_USE_TRITON_AWQ=1 \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++ \
    MAX_JOBS=4

RUN git clone --depth 1 -b ${VLLM_VERSION} \
        https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    pip install --no-cache-dir -r requirements-rocm.txt && \
    pip wheel --no-build-isolation --no-deps -w /tmp/vllm-wheel . && \
    pip install --no-cache-dir /tmp/vllm-wheel/*.whl && \
    rm -rf /tmp/vllm /tmp/vllm-wheel

# Verify vLLM installed
RUN python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"

# ── Runtime environment ────────────────────────────────────────────
ENV LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
    HIP_FORCE_DEV_KERNARG=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    ROCBLAS_USE_HIPBLASLT=1

# ── Go management binary ──────────────────────────────────────────
COPY --from=go-builder /app/vllmctl /usr/local/bin/vllmctl

# ── Data directories ──────────────────────────────────────────────
RUN mkdir -p /data/config /data/models

RUN cat > /data/config/vllmctl.yaml << 'EOF'
listen_addr: ":3000"
data_dir: "/data"
vllm_port: 8000
log_level: "info"
tool_use_enabled: true
gpu_memory_util: 0.90
EOF

VOLUME ["/data"]

EXPOSE 3000
EXPOSE 8000

ENTRYPOINT ["vllmctl", "--config", "/data/config/vllmctl.yaml"]

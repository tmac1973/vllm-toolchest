# vllm-toolchest

A containerized vLLM serving platform with a web UI for model management, monitoring, and configuration.

## What it does

vllm-toolchest packages vLLM into a Docker/Podman container with a full web interface for managing local LLM inference. Download models from HuggingFace, configure serving parameters per-model, start/stop vLLM, and monitor GPU usage -- all from a browser.

## Features

- **Model management** -- register, configure, and delete local models
- **HuggingFace integration** -- search and download models directly from HF Hub
- **Per-model vLLM config** -- dtype, context length, tensor parallelism, quantization, memory utilization
- **Process control** -- start, stop, restart vLLM with live log streaming
- **GPU monitoring** -- real-time VRAM usage and GPU utilization (NVIDIA and AMD)
- **Tool use support** -- auto tool choice with configurable tool call parsers
- **Multiple quantization formats** -- GPTQ, AWQ, GGUF, Marlin, bitsandbytes
- **OpenAI-compatible API** -- proxies /v1/* endpoints to vLLM

## Quick start

```bash
./setup.sh install
```

This builds the container image and creates a systemd (or podman) service. Once running, open `http://localhost:8080` in your browser.

On an RDNA4 card the installer offers a choice of two images -- see [Image variants](#image-variants).

## Requirements

- **Container runtime**: Docker or Podman
- **GPU**: NVIDIA or AMD GPU with sufficient VRAM for your model
- **OS**: Linux (tested on Arch/CachyOS, should work on any distro with container support)

## GPU support

| Backend | Status |
|---------|--------|
| NVIDIA CUDA | Tested and working |
| AMD ROCm (RDNA3) | Supported |
| AMD ROCm (RDNA4) | Supported -- generic, or the tuned `radiance` variant |

## Image variants

vllm-toolchest is the same Go binary and web UI either way; what differs is the
vLLM stack underneath it.

| | `generic` | `radiance` |
|---|---|---|
| Base | Fedora + ROCm (or CUDA), vLLM built from source | [vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance), a stack hand-tuned for gfx1201 |
| GPUs | Any supported NVIDIA or AMD card | AMD RDNA4 only (gfx1201: R9700, RX 9070/XT) |
| vLLM | Tracks `main` | Pinned to the release radiance was built against |
| Model support | Newest | Frozen at that vLLM/transformers release |
| Install | Long -- compiles the whole stack | Fast -- pulls a prebuilt base |
| Performance on RDNA4 | Correctness patches only | Custom attention/GEMM/all-reduce kernels, tuned FP8 + MoE configs, MTP drafting |

Pick a variant at install time, or force one at any point:

```bash
./setup.sh install                     # offers the choice on an RDNA4 card
VARIANT=radiance ./setup.sh install    # skip the prompt
VARIANT=generic  ./setup.sh rebuild    # switch back later
```

The choice is stored in `.env` as `VLLMCTL_VARIANT` and reused by every later
command. `./setup.sh detect` prints the current backend and variant.

### What the radiance variant adds

Its tuned paths are switchable from the **Settings** page (and via `RADIANCE_*`
variables in `.env`), each defaulting to whatever the image ships so they track
upstream rather than being pinned here:

- **Hand-written gfx1201 kernels** (`libr4d`): paged attention, the fused
  gated-delta-net prefill scan, a TP=2 P2P all-reduce, a skinny bf16 GEMM, and
  a head_dim-72 ViT flash kernel.
- **Tuned GEMM and attention configs** for FP8 block-scale and fine-grained MoE.
- **Lossless MTP speculative drafting** with a dynamic per-request draft depth.
- **RDNA4 correctness patches** -- GPU enumeration, AITER enablement for gfx12x,
  and an attention LDS fit without which CUDA-graph capture aborts outright.

Not everything applies to every model. The correctness patches, block-FP8 GEMM
routing, attention tuning and all-reduce are model-agnostic; the gated-delta-net
paths need a hybrid linear-attention model, the `R4D` attention backend refuses
any shape other than head_dim 256 / paged block 16 / GQA 6, and the drafting
work needs a speculative config. Radiance's measurements are all on FP8 models
across two R9700s.

Because radiance pins vLLM and transformers, **a model newer than that release
will not load**. If you need the newest architectures, use `generic`.

## Development

```bash
# Run the Go server with live reload (requires air)
make dev

# Rebuild and restart the container
make reload
```

## Screenshots

*Coming soon.*

## License

TBD

## Credits

Based on [llama-toolchest](https://github.com/tmac1973/llama-toolchest).

The `radiance` image variant is built on
**[vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance)** by
StillDeadcode -- a from-source ROCm + PyTorch + Triton + AITER + vLLM stack for
the AMD Radeon AI PRO R9700 (gfx1201 / RDNA4), together with
**[libr4d](https://codeberg.org/StillDeadcode/libr4d)**, its hand-written HIP
kernel library. All of the RDNA4 patches, custom kernels and tuning in that
variant are their work; vllm-toolchest only layers this management UI on top of
the published image. If you use it, go read their
[DOCKERHUB.md](https://codeberg.org/StillDeadcode/vllm-radiance/src/branch/main/DOCKERHUB.md)
-- it documents every knob in far more detail than we reproduce here.

The generic ROCm image's RDNA4 build patches come from
[kyuz0/amd-r9700-vllm-toolboxes](https://github.com/kyuz0/amd-r9700-vllm-toolboxes).

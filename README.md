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

`setup.sh` handles everything: it detects your GPUs and container runtime,
asks what it needs to know, writes the config, and builds. On an RDNA4 card it
offers a second, hand-tuned image -- see [Image variants](#image-variants).

## Requirements

- **Container runtime**: Docker or Podman
- **GPU**: NVIDIA or AMD GPU with sufficient VRAM for your model
- **OS**: Linux (tested on Arch/CachyOS, should work on any distro with container support)

## GPU support

| Backend | Status |
|---------|--------|
| NVIDIA CUDA | Tested and working |
| AMD ROCm (RDNA3) | Supported |
| AMD ROCm (RDNA4) | Supported -- from source, or the tuned `radiance` variant |

## Image variants

vllm-toolchest is the same Go binary and web UI whichever you pick; what
differs is the vLLM stack underneath it. Each variant is described by one file
in [`variants/`](variants/) -- its base image, the hardware it runs on, and the
feature switches its Settings page offers.

| Variant | Runs on | Stack |
|---|---|---|
| `rocm-source` | Any supported AMD card | Fedora + ROCm, vLLM built from source, tracking `main` |
| `cuda-source` | Any supported NVIDIA card | CUDA devel image + vLLM from PyPI |
| `radiance` | AMD RDNA4 (gfx1201: R9700, RX 9070/XT) | [vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance), hand-tuned for gfx1201 |

Run `./setup.sh variants` to see the list on your machine, with the ones that
fit your GPU marked.

Each variant carries a **support tier**: `tested` means someone on this project
ran it on that hardware, `community` means it is published upstream but not
validated here, and `experimental` means little upstream support.

The trade-off between the two kinds: a from-source build takes a long time and
tracks vLLM `main`, so model support is as new as it gets. A prebuilt variant
installs fast but pins vLLM to the release its base was built against, so **a
model needing a newer vLLM will not load**. The install prints the pin, and so
does `./setup.sh status`.

`./setup.sh install` detects the GPU, offers the variants that fit it with the
best match preselected, then asks the remaining questions -- which GPUs to use,
which ports, where to keep models -- and builds. You do not need to write a
config file first.

To skip the prompt, or to change your mind later:

```bash
VLLMCTL_VARIANT=radiance ./setup.sh install     # build a specific variant
VLLMCTL_VARIANT=rocm-source ./setup.sh rebuild  # switch to the from-source image
```

A variant states what it needs of the host -- a GPU architecture, a minimum
driver version -- and the install checks those before building rather than
after. If a check is wrong for your machine, `VLLMCTL_SKIP_HOSTCHECK=1` lifts
all of them.

The answers are stored in `.env` and reused by every later command, so `up`,
`down`, `logs` and `rebuild` all act on the variant you installed.
`./setup.sh detect` prints the current backend and variant. setup.sh only
rewrites the keys it owns, so anything you add to `.env` yourself survives.

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
will not load**. If you need the newest architectures, use `rocm-source`.

### Docker and Podman

Both are supported, and `setup.sh` adapts to whichever you have.

There is one difference worth knowing about. The published vllm-radiance image
is an OCI manifest whose layers carry *Docker* media types. Docker tolerates
the mix and builds on it normally (verified against Docker 29.8). But
`containers/image` -- the library behind Podman, Buildah and Skopeo alike --
refuses to rewrite such a manifest, so under Podman the build fails at the
first instruction:

```
unsupported MIME type for compression:
"application/vnd.docker.image.rootfs.diff.tar.gzip"
```

Podman can *run* the image fine; only using it as a build base is affected, and
no manifest-level repair works -- `push --format`, `save` and skopeo all hit
the same wall.

`setup.sh` handles this for you. It probes whether your runtime can build on
the image and only if it cannot does it flatten the image into a single-layer
local one first, which costs roughly 10 GB and a few minutes, once per radiance
version:

| Runtime | What happens |
|---|---|
| Docker | Probe passes, builds directly from the published image |
| Podman | Probe fails, image is flattened once, then builds |

Since it is a probe and not a runtime check, this also disappears by itself the
day a conformant image is published upstream.

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

The `rocm-source` image's RDNA4 build patches come from
[kyuz0/amd-r9700-vllm-toolboxes](https://github.com/kyuz0/amd-r9700-vllm-toolboxes).

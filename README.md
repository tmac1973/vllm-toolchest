# vllm-toolchest

A containerized vLLM serving platform with a web UI for model management, monitoring, and configuration.

## What it does

vllm-toolchest packages vLLM into a Docker/Podman container with a full web interface for managing local LLM inference. Download models from HuggingFace, configure serving parameters per-model, start/stop vLLM, and monitor GPU usage -- all from a browser.

## Features

- **Model management** -- register, configure, and delete local models
- **HuggingFace integration** -- search and download models directly from HF Hub
- **Per-model vLLM config** -- dtype, context length, tensor parallelism, quantization, memory utilization
- **Autoconfigure** -- one button proposes a whole config from the model card's own serve command and this machine's GPUs, read by a small helper model and reviewed setting by setting before anything is applied; a real start then refines it
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

> **Locked memory, if you plan to offload.** Models that keep experts or a
> large embedding table in system RAM pin that memory, and a rootless
> container cannot lock more than your user's hard limit — the compose file
> asks for unlimited and is clamped in silence. Most distributions default to
> 8 MiB. Check with `ulimit -H -l`; if it is not `unlimited`, add
>
> ```
> *  soft  memlock  unlimited
> *  hard  memlock  unlimited
> ```
>
> to `/etc/security/limits.conf` and log out and back in. `setup.sh` warns
> about this at install time. Without it the engine loads unpinned or fails,
> reporting something like `PLE offload: locked 0.0 GiB, FAILED to lock
> 47.7 GiB` deep in its startup log.


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
| `cuda` | Any supported NVIDIA card | [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai) -- the vLLM project's own image |
| `cuda-source` | Any supported NVIDIA card | CUDA devel image + vLLM from PyPI |
| `rocm` | AMD RDNA3 and RDNA4 (gfx1100-gfx1201) | [rocm/vllm](https://hub.docker.com/r/rocm/vllm) -- AMD's own build |
| `rocm-cdna` | AMD Instinct (MI200, MI300, MI350) | [rocm/vllm](https://hub.docker.com/r/rocm/vllm) -- AMD's CDNA build |
| `rocm-source` | Any supported AMD card | Fedora + ROCm, vLLM built from source, tracking `main` |
| `radiance` | AMD RDNA4 (gfx1201: R9700, RX 9070/XT) | [vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) -- hand-tuned for gfx1201 |
| `rdna4-clav` | AMD RDNA4 (gfx1201 and gfx1200) | [tcclaviger/vllm](https://blog.robai.net/vllmdocs/) -- hand-written HIP kernels |
| `strix-halo` | Ryzen AI MAX (gfx1151) | [kyuz0/vllm-therock-gfx1151](https://hub.docker.com/r/kyuz0/vllm-therock-gfx1151) |
| `gfx906` | MI50, MI60, Radeon VII | [vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick) -- serve with `--dtype float16` |
| `xpu` | Intel Arc Pro B60/B70 (Battlemage) | [intel/llm-scaler-vllm](https://hub.docker.com/r/intel/llm-scaler-vllm) |
| `gb10` | NVIDIA DGX Spark (arm64, sm_121a) | [vllm-gb10](https://github.com/timothystewart6/vllm-gb10) |

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

### Variants that follow a moving image

Most prebuilt variants name one release of the image they build on, and moving
to a newer one is a change to the manifest. `rdna4-clav` does not: its image and
the checkpoints published for it move together and often, so it follows
`tcclaviger/vllm:latest`. What that means in practice:

```bash
./setup.sh pull      # fetch the newest base image and rebuild on it
./setup.sh quick     # rebuild after a code change -- same engine as before
./setup.sh status    # which vLLM and which image release is installed
```

- `install`, `rebuild` and `pull` ask the registry where the tag points now.
  `quick` never does, so the engine changes when you ask and not as a side
  effect of rebuilding the UI.
- The vLLM version and the ref the FP8 tuner script is fetched from are read
  out of the image that was pulled, not stated in the manifest.
- The image that was in use before a move is kept, and the move prints the
  command that goes back to it:
  `VLLMCTL_BASE_IMAGE=<previous> ./setup.sh pull`. Going back sticks through
  later `quick` rebuilds, until the next `pull` without the override. Only the
  current base and the one before it are kept; older ones are removed.

What was resolved is recorded in `.env` as `VLLMCTL_BASE_IMAGE` (the local name
the build uses), `VLLMCTL_BASE_DIGEST`, `VLLMCTL_BASE_PREVIOUS`,
`VLLMCTL_BASE_VLLM` and `VLLMCTL_BASE_RELEASE`.

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

### SELinux and io_uring

On a distribution with SELinux enforcing -- Fedora, RHEL and their derivatives
-- starting a model logs an alert that looks alarming and is not:

```
SELinux is preventing vllm from create access on the anon_inode
labeled io_uring_t
```

vLLM's engine asks the kernel for an io_uring instance, SELinux refuses, and
vLLM falls back to ordinary syscalls. Nothing is lost: the model loads and
serves, and vLLM does not consider the refusal worth logging. You will see a
handful of alerts as the engine starts and then none, because it stops asking.

**Do not run the `audit2allow` command the alert suggests.** `container_t` is
the domain every container on the machine runs in, so that module would grant
io_uring to all of them, permanently, to fix something that is not broken.
io_uring is a large and historically CVE-prone kernel surface, and the
container policy denies it deliberately -- note that Fedora ships no
`container_use_io_uring` boolean, where it does ship one for the device access
this tool actually needs (`container_use_devices`, which `setup.sh` enables for
you).

If you ever do want it -- and you should want a measurement first, not a
notification -- scope it to this container rather than granting it globally.

## Client auto-discovery

A client can ask the server what it is serving and how to drive it, instead of
being configured by hand. The contract is the same one
[llama-toolchest](https://github.com/tmac1973/llama-toolchest) exposes, so a
client written against either works with both. Haruspex probes it this way:

1. `GET /api/service/status` — answers 200 when this is a toolchest.
2. `GET /api/service/loaded-models` — everything else, in one request.

```jsonc
{
  "schema_version": 1,
  "server": "vllm-toolchest",
  "running": true,
  "models": [{
    "id": "owner/repo",              // what to send as "model"
    "status": "loaded",              // or "loading" while the engine starts
    "public_name": "owner/repo",
    "registry_id": "owner/repo",
    "capabilities": {
      "schema_version": 1,
      "context_size": 131072,        // served --max-model-len: compact on this
      "context_length": 262144,      // trained maximum, for information
      "parallel": 8,                 // --max-num-seqs
      "context_per_request": 131072, // same as context_size; see below
      "vision": true,
      "tools": true,                 // tool calls are parsed on this serve
      "embedding": false,
      "reasoning": {
        "supported": true,
        "default_enabled": true,
        "toggle": "chat_template_kwargs",  // | "reasoning_effort" | "none"
        "kwarg": "enable_thinking",
        "effort_levels": ["low", "medium", "xhigh"],  // null when not enumerated
        "default_effort": "xhigh"
      },
      "sampling": {
        "source": "override-generation-config",  // | "generation_config.json" | null
        "source_url": null,
        "default": { "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "min_p": null, "presence_penalty": null, "repeat_penalty": null },
        "presets": []
      },
      "max_output_tokens": 65536     // null when the engine imposes none
    }
  }]
}
```

Things worth knowing as a client:

- **One model or none.** vLLM serves one model per process, so the list is what
  `/v1` will actually answer for, not the registry.
- **What is reported is what is running.** Values come from the flags the engine
  was started with, so a config edited since the last start does not show up
  until the next one.
- **`context_per_request` is not divided by `parallel`.** llama.cpp gives each
  slot a fixed share of the context; vLLM draws every request from one KV pool,
  so each may use the whole window and one that does not fit yet waits.
- **`effort_levels` is never guessed.** It is read from the chat template, and
  is `null` when the template takes no effort level or does not say which.
- **Every key is always present**, with `null` or `false` for "known to be
  absent", so a missing key can only mean an older server.
- `sampling.default` is what the engine applies when a request names no
  sampling of its own: the launch's `--override-generation-config` where it sets
  a value, the checkpoint's `generation_config.json` where it does not.

## Security

vllm-toolchest has no login. Put the UI port (`VLLMCTL_PORT`, 3000) on a
network you trust: anyone who can reach it can start and stop models, change
settings, and read or replace the HF token and API key through backup and
restore.

- **API key.** With `VLLMCTL_API_KEY` (or the key in Settings) set, every `/v1`
  request needs `Authorization: Bearer <key>`, both through the proxy on
  `VLLMCTL_PORT` and on vLLM's own port (`VLLMCTL_INFERENCE_PORT`, 8000), which
  is started requiring the same key. A key changed in Settings reaches vLLM's
  port at its next start; the proxy accepts the new key straight away.
- **Cross-site requests.** A state-changing `/api` request that a browser marks
  as coming from another site is refused, so a web page you visit cannot drive
  the UI or post a restore. Scripts and `curl` are unaffected, and so is `/v1`,
  which browser chat front-ends call cross-origin on purpose.
- **Secrets on disk.** The config file holding the token and key is written
  owner-only (`0600`). A backup exported with secrets holds them in plain text.

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

Every other prebuilt variant is somebody else's work too, and the whole reason
this tool can serve on hardware upstream vLLM does not reach. vllm-toolchest
adds a management UI to a published image and changes nothing about the stack:

- **[tcclaviger/vllm](https://blog.robai.net/vllmdocs/)** (`rdna4-clav`) --
  RDNA4 with hand-written AOT HIP kernels across attention, the gated-delta-net
  path, the quantized and block-FP8 GEMMs, the KV-cache write and the
  all-reduce.
- **[kyuz0/vllm-therock-gfx1151](https://hub.docker.com/r/kyuz0/vllm-therock-gfx1151)**
  (`strix-halo`) -- built on TheRock nightlies, for a long time the only
  working vLLM path on Ryzen AI MAX.
- **[vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick)**
  (`gfx906`) -- keeps Vega 20 alive after upstream dropped it, and successor to
  the archived [nlzy/vllm-gfx906](https://github.com/nlzy/vllm-gfx906).
- **[intel/llm-scaler-vllm](https://hub.docker.com/r/intel/llm-scaler-vllm)**
  (`xpu`) -- Intel's own Arc Pro build, with oneAPI and oneCCL pinned to each
  release.
- **[vllm-gb10](https://github.com/timothystewart6/vllm-gb10)** (`gb10`) --
  DGX Spark, pinning every input by commit or digest.
- **[rocm/vllm](https://hub.docker.com/r/rocm/vllm)** (`rocm`, `rocm-cdna`) and
  **[vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai)** (`cuda`) --
  the vendor and upstream images.

## Adding a variant

One file. `variants/<id>.conf` states what the image is, what it runs on and
what it needs; nothing else has to change. The format is a strict subset of
bash so `setup.sh` can source it with no `jq`, `yq` or Python on the host, and
the Go binary embeds and parses the same bytes.

```sh
VARIANT_ID='my-variant'
VARIANT_LABEL='Something descriptive'
VARIANT_SUMMARY='One line, shown in the install menu.'
VARIANT_VENDOR='amd'                  # picks docker-compose.<vendor>.yml
VARIANT_TIER='community'              # tested | community | experimental
VARIANT_BASE_IMAGE='docker.io/someone/their-vllm:1.2.3'
VARIANT_DOCKERFILE='Dockerfile.prebuilt'
VARIANT_VLLM_PIN='v0.28.0'            # surfaced as the model-support ceiling
VARIANT_GFX_TARGETS='gfx1201'         # empty = not tied to one architecture
VARIANT_VENV_ROOT='/opt/venv'         # where vLLM lives inside that image
KNOBS=''
```

Two things are worth reading off the published image rather than guessing, because
the build asserts them: `VARIANT_VENV_ROOT` (from `VIRTUAL_ENV` or `PATH` in the
image config) and `VARIANT_VLLM_PIN`. Get the pin wrong and the build stops with
a message telling you the right value.

A base that names a tag which moves (`latest`, `nightly`) has to say so, or it
is pulled once and never again:

```sh
VARIANT_BASE_IMAGE='docker.io/someone/their-vllm:latest'
VARIANT_BASE_PULL='always'            # re-resolve on install, rebuild and pull
VARIANT_VLLM_PIN='image'              # read the vLLM version out of the base
```

`VARIANT_VLLM_PIN='image'` is required alongside `VARIANT_BASE_PULL`: a pin
written down in advance is right until the image's next vLLM bump and then
fails every build. It can also be used on its own, with a fixed tag, to save
looking the version up. `VARIANT_BASE_RELEASE_CMD`, optional, is a shell command
run inside the image that prints its author's release number, for `status` to
show.

Feature switches are declared here too, once, and the config layer, the Settings
page and `.env.example` all generate from that declaration:

```sh
KNOBS='FAST_PATH'
KNOB_FAST_PATH_ENV='THEIR_FAST_PATH'
KNOB_FAST_PATH_TYPE='select'          # select | text
KNOB_FAST_PATH_LABEL='Fast path'
KNOB_FAST_PATH_HELP='What it does, and what it costs when it is wrong.'
KNOB_FAST_PATH_VALUES='- 1 0'         # "-" is "leave the image default alone"
KNOB_FAST_PATH_RECOMMENDED='-'
```

Attention backends are declared the same way, and only the ones the image
actually has:

```sh
VARIANT_ATTENTION_BACKENDS='ROCM_ATTN TRITON_ATTN'
BACKEND_ROCM_ATTN_LABEL='ROCM_ATTN — what vLLM picks here by default'
```

Leave it out if you do not know, and the picker offers "auto" alone. That is
the right default: vLLM's own selection is usually correct, and naming a
backend the stack does not have aborts the engine minutes into a load rather
than falling back. To find the real list, start a model and read the engine log
line `out of potential backends: [...]`.

After adding a switch, run `make env-example` to regenerate its documentation,
and add its variable to the vendor's compose file as a bare `- THEIR_FAST_PATH`
entry so it reaches the container. Tests fail if you forget either.

`go test ./variants/` checks that the manifest parses, that bash and Go read it
identically, that the Settings page can draw every control it declares, and
that its compose file and Dockerfile exist.

# Phase 10: Image variant expansion

**Goal:** extend the build-time variant mechanism from two hardcoded choices
(`generic`, `radiance`) to a manifest-driven set covering NVIDIA, AMD and Intel
accelerators, with hardware detection that recommends the right one.

**Status:** implemented. See the `variant-manifests` branch.

---

## About this document

It began as a scoping note written against upstream sources in September 2026,
which correctly warned that "all version numbers drift; re-check tags at
implementation time." Re-checking them changed several conclusions, so this is
the corrected version: what was verified, what it changed, and what is still
taken on trust.

**Verification method.** Every image below was queried against its live
registry on **2026-09-10** — Docker Hub's v2 API for tags, and the registry
manifest/config blob for the environment the image actually declares. Where
this document states an architecture list, a venv path or an entrypoint, that
came out of the published image config, not from a README.

Three conventions are used throughout:

- **Verified** — read from the registry on the date above.
- **Unverified** — carried over from the original note, plausible but not
  independently checked. Treat as a claim, not a fact.
- **Corrected** — the original note said something else, and this is what the
  registry says.

---

## 1. The variant set as shipped

Eleven variants, one manifest each in `variants/<id>.conf`.

| id | vendor | base image | tier |
|---|---|---|---|
| `cuda` | nvidia | `vllm/vllm-openai:v0.29.0` | community |
| `cuda-source` | nvidia | `Dockerfile.cuda` (PyPI onto CUDA devel) | tested |
| `rocm` | amd | `rocm/vllm:rocm7.14.1_rdna_…_vllm_0.23.0` | **tested** |
| `rocm-cdna` | amd | `rocm/vllm:rocm7.14.1_cdna_…_vllm_0.23.0` | community |
| `rocm-source` | amd | `Dockerfile.rocm` (Fedora + ROCm, from source) | tested |
| `radiance` | amd | `stilldeadcode/vllm-radiance:0.9.3` | tested |
| `rdna4-clav` | amd | `tcclaviger/vllm:28.02.2` | community |
| `strix-halo` | amd | `kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.28.0` | community |
| `gfx906` | amd | `aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0` | community |
| `xpu` | intel | `intel/llm-scaler-vllm:0.26.0-b2` | community |
| `gb10` | nvidia | `ghcr.io/timothystewart6/vllm-gb10:v0.28.0-gb10.2` | experimental |

Eleven rather than the eight originally proposed, for two reasons given in §2
and §3.

`tested` means someone on this project ran it on that hardware. Available here:
RDNA4 (gfx1201), RDNA3 (RX 7900 XTX, gfx1100) and an NVIDIA card.

Only `rocm` has been earned by this work: built and served on the RX 7900 XTX
on 2026-09-11. The other three `tested` tiers are inherited from the
pre-existing paths. Everything else is a registry config and an inference until
somebody builds it.

---

## 2. Corrections that changed the plan

### 2.1 The RDNA4 MoE image does not exist under that name

**Corrected.** The note recommended `tcclaviger/vllm-rocm-rdna4-mxfp4` as a
second RDNA4 variant, for MXFP4 MoE work that `radiance` leaves on the table.
That repository returns 404. The `tcclaviger` namespace holds two repositories:
`vllm22` (deprecated, redirecting to the other) and `vllm`.

`tcclaviger/vllm` is the live successor and has moved past the MXFP4 framing.
Its own description leads on native AOT HIP kernels across the stack —
attention (CLAV_ATTN), the gated-delta-net path, RFI/RFA quantized GEMMs at
2/4/6/8-bit, the block-scaled FP8 GEMM, the KV-cache write, the GDN causal-conv
pair, spec-decode state copies and the tensor-parallel all-reduce. The headline
consequence is startup rather than throughput: removing the Triton compile
takes a fully cold 27B TP4 start from 10–15 minutes to about 2:39, and removes
the need for a Triton cache volume.

Shipped as `rdna4-clav`, described as what it is. **Verified:**
`PYTORCH_ROCM_ARCH=gfx1201;gfx1200`, `VIRTUAL_ENV=/opt/vllm`,
`VLLM_TARGET_DEVICE=rocm`, entrypoint `/app/tools/image_entrypoint.sh`, newest
tag `28.02.2` (2026-09-08), built on the vLLM v24 tree with torch 2.11 and
ROCm 7.2.3.

Two consequences worth noting. It covers **gfx1200** as well as gfx1201, which
`radiance` does not — so RX 9060 XT (Navi 44) owners get a tuned option, which
the original note listed as an open question under `patcarter883`. And its
entrypoint runs `vllm serve "$@"` when given no tool flag, so it works as a
launcher and keeps whatever it sets up in our log stream.

### 2.2 AMD publishes RDNA and CDNA as separate images

**Corrected.** The note treated `rocm` as one variant covering "gfx1100/gfx1101
(RDNA3), CDNA". `rocm/vllm` ships those as different images, compiled for
different targets. One variant claiming both would be built for neither, so
there are two.

**Verified**, from the image configs:

- RDNA: `PYTORCH_ROCM_ARCH=gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201`
- CDNA: `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950`

Both put Python at `/opt/python`, not in a venv, and declare no entrypoint.

The RDNA list is wider than the note assumed — it reaches RDNA4 and Strix Halo
— which is why `rocm` acts as the general AMD fallback that every
card-specific variant is ranked against. On gfx1201 it is outranked by
`radiance` and `rdna4-clav`; on gfx1151 by `strix-halo`.

AMD also publishes older per-architecture tags (`gfx110X-all`, `gfx1150`,
`gfx1151`, `gfx1152`, `gfx120X-all`, `gfx94X-dcgpu`, `gfx950-dcgpu`) at vLLM
0.19.1 from 2026-05-19, and a `rocm10.0.0_…_vllm_0.27.0` tag from 2026-08-27
carrying a newer vLLM but no architecture in its name. The consolidated
rdna/cdna pair is the current line and is what the manifests use.

### 2.3 The upstream ROCm image has no releases

**Corrected.** The note offered `vllm/vllm-openai-rocm` as an alternative base
for the `rocm` variant. It has 91 tags and every one is a nightly or a
`base-nightly`. There is no release to pin, so AMD's own `rocm/vllm` — the
note's first option — is the only viable choice.

---

## 3. Per-variant detail

Venv roots are **verified** from each image's config, because
`Dockerfile.prebuilt` asserts them and a wrong guess fails every build. They
are not uniform.

### cuda

The reference platform, and the reason no tuned community fork exists for
mainstream NVIDIA silicon: this is the hardware vLLM is developed against.

**Verified:** `TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0` (Ampere
through Blackwell), entrypoint `['vllm','serve']`, workdir `/vllm-workspace`,
no `VIRTUAL_ENV` — vLLM is installed system-wide, so the venv root is
`/usr/local`. Newest stable tags `v0.29.0` and `v0.28.0` among 598 total, most
of which are nightlies and per-model builds.

The manifest deliberately declares **no** architecture targets. Detection
reports one compute capability; matching it against seven strings would reject
a card the image supports through PTX. Compute capability is only used where it
discriminates, which is `gb10`.

### rocm / rocm-cdna

See §2.2. `VARIANT_VENV_ROOT=/opt/python`, confirmed by building.

The tag reads `vllm_0.23.0` and the image reports
`0.23.1.dev1+g9ddef7117.d20260901.rocm714` — a build from a commit between
releases. The pin stays at the nearest real release and `VARIANT_TUNER_REF`
names the commit, because there is no `v0.23.1` tag to fetch the tuner script
from. `9ddef7117` is a real vllm-project commit (2026-07-14), so the script
comes from the exact tree AMD built against rather than a nearby tag.

`rocm-cdna` keeps the tag and no tuner ref: the minor matches either way, and
the commit behind that image is unknown.

### rocm-source / cuda-source

The pre-existing from-source paths, unchanged, renamed from the single
`generic` variant they used to share. That name covered two different builds
with the GPU found at install time deciding which you got; the split makes the
vendor, Dockerfile and compose file properties of the manifest rather than of
detection. Existing installs migrate by vendor on any command.

### radiance

Already implemented before this work; now described by a manifest like the
rest. **Verified:** `0.9.3` is still the newest of 11 tags.

Its twelve `RADIANCE_*` switches are the only populated knob set. Two items the
original note flagged as possibly unwired are now expressible but **not
populated**: `RADIANCE_NUMA_BIND` is declared as a knob (and the compose file
grants `SYS_NICE` for it), and `RADIANCE_MXFP4_W4A8_MIN_M` is not declared at
all — it was not among the twelve the previous implementation carried, and
nothing here verified its behaviour.

### rdna4-clav

See §2.1. `VARIANT_VENV_ROOT=/opt/vllm`, launcher
`/app/tools/image_entrypoint.sh`.

### strix-halo

**Verified:** `PYTORCH_ROCM_ARCH=gfx1151` and nothing else,
`VIRTUAL_ENV=/opt/venv`, 128 tags with `rocm10.0.0-torch2.11.0-vllm0.28.0`
among the newest. Built on TheRock nightlies.

**Unverified:** the note's claims about a custom ROCm/RCCL with native
RDMA/RoCE v2 enabling TP=2 across two nodes as a single 256 GB pool, and about
which attention backends it exposes (`TRITON_ATTN`, `ROCM_ATTN`,
`ROCM_AITER_UNIFIED_ATTN`, deliberately omitting `ROCM_AITER_FA` because its
paged-attention decode kernel has no Navi implementation). None of that was
checked. If true, the backend list belongs in the manifest's
`VARIANT_ATTENTION_BACKENDS`, which is currently empty for this variant.

**Unverified but carried into docs:** the rootless-Podman caveat, that
`--group-add keep-groups` is needed rather than `--group-add video --group-add
render`. The compose file uses the GID form for all vendors.

### gfx906

**Verified:** `PYTORCH_ROCM_ARCH=gfx906`, 4 tags, newest
`v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0`, workdir
`/workspace/vllm-gfx906-mobydick`. No `VIRTUAL_ENV` and no venv on `PATH`, so
vLLM is installed against the system interpreter; the manifest guesses
`/usr/local`. A wrong guess here costs the tuner and the bitsandbytes check,
not the ability to serve, because vllmctl probes every declared root plus both
historical defaults.

**Verified by absence:** the note said `nlzy/vllm-gfx906` is archived and
`ttdxq/gfx906-vllm` publishes no image. Neither has a Docker Hub repository,
consistent with both claims.

**Unverified:** that bf16 is not native on gfx906 and falls back to fp32, hence
`--dtype float16`. Widely documented, not checked here. It is in the manifest
summary because getting it wrong is slow rather than loud.

### xpu

The heaviest host integration of the set.

**Verified:** `VIRTUAL_ENV=/opt/venv`, workdir `/llm-scaler/vllm`, 32 tags with
`0.26.0-b2` newest, and an entrypoint of

```
bash -c 'source /opt/intel/oneapi/setvars.sh --force && \
         source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force && \
         exec vllm serve "$@"' --
```

That entrypoint is the reason the manifest format needed a numbered
`VARIANT_LAUNCHER_n` form: without the oneAPI setup vLLM cannot find its
libraries, and a space-separated launcher string cannot carry an argv element
containing spaces. The manifest carries Intel's line verbatim rather than a
reconstruction.

**Verified:** `intel/vllm` exists with `0.21.0-xpu` tags, which is the image the
note warns against — users following Intel's own docs onto it hit
`RuntimeError: PyTorch was compiled without CUDA support` on a B60.

**Verified:** `intelanalytics/multi-arc-serving` (the Alchemist/A770 path) last
published on **2025-08-19**, over a year ago. The note's "poorly maintained" is
if anything generous, which is why the Battlemage/Alchemist split matters.

**Unverified and the weakest part of this work:** the Battlemage PCI device id
list (`e202 e20b e20c e20d e210 e211 e212 e215 e216`). Assembled from general
knowledge, not from Intel documentation, and not testable here. It is a
**warn**, not a block, deliberately: the list will grow as Intel ships more of
the family, and refusing to install over a part we have not heard of is the
worse failure.

**Unverified:** the note's account of Intel's install flow (Ubuntu 24.04 →
offline installer for kernel, firmware, driver and tools → image) and that each
release pins host OS, vLLM, PyTorch, oneAPI, oneCCL, UMD, KMD, GuC firmware and
XPU Manager together, failing at runtime rather than at build on a mismatch.
Carried into the manifest note because it sets expectations correctly even if
the details have moved.

### gb10

**Verified:** `TORCH_CUDA_ARCH_LIST=12.1a`, no `VIRTUAL_ENV`, workdir
`/workspace`, newest release-style tag `v0.28.0-gb10.2` on GHCR.

Matching is on host architecture as much as GPU: nothing else here is arm64, so
a machine that is both `aarch64` and `sm_121` is a DGX Spark. The manifest
blocks on `aarch64` and warns on compute capability.

**Verified:** the alternatives the note listed are real but stale —
`scitrera/dgx-spark-vllm` last published 2026-03-08,
`hellohal2064/vllm-dgx-spark-gb10` has a single tag from 2026-03-17.
`ghcr.io/bjk110/vllm-spark` was not checked (GHCR, needs a per-repo token).

---

## 4. What was built

The original note proposed collapsing to `Dockerfile.prebuilt` plus a
per-variant manifest, and listed candidate manifest fields. That is roughly
what happened, with these differences worth recording.

**The manifest is bash-sourceable, not YAML.** `setup.sh` has never needed
`jq`, `yq` or Python on the host, and a manifest format that broke that would
have been a real regression for a script people curl. The grammar is
`NAME='value'`, one per line, always single-quoted — bash single quotes are
fully literal, which is what lets a tooltip carry commas, double quotes,
em-dashes and `$` unescaped. The one character it cannot carry is the ASCII
apostrophe, and the Go parser rejects it rather than letting the two readers
disagree about where a value ends.

**Go reads the same bytes via `go:embed`, with no generate step.** The `.conf`
files have to ship for bash anyway, so generating Go source would buy a second
artifact to keep in sync plus a `go generate` a contributor forgets.

**Compose collapsed by vendor, not by variant.** Device wiring is a property of
the vendor — AMD's `/dev/kfd` + `/dev/dri` + group_add, NVIDIA's
`deploy.resources`, Intel's render node — so there are three compose files, not
eleven. Per-variant build inputs reach them through `.env`, which `setup.sh`
derives from the manifest and rewrites on every install.

**Knobs are declared once and everything generates from that.** The twelve
`RADIANCE_*` names previously lived in six places: a Go struct with yaml tags, a
field→env table, the reverse table, twelve form-handler branches, 107 lines of
template markup and the `.env.example` prose. Now: the manifest, and the config
layer, the Settings renderer and `.env.example` all generate from it.

**`VARIANT_IMAGE_ENV` was not in the original field list and turned out to be
necessary.** A Dockerfile cannot set `ENV` from a variable-length build arg,
and radiance bakes an AITER routing matrix that is variant-specific. It moves
to the manifest and is applied when vllmctl spawns the server — which covers
`podman run` and the Quadlet unit as well as compose, a stronger guarantee than
baking it.

**Host requirements carry their own severity.** `HOSTREQ_n='kind|block|warn|value|message'`.
Severity is data rather than code because most of these images run on hardware
nobody here can test, so a check that turns out to be wrong is a one-line
manifest edit rather than a release. `VLLMCTL_SKIP_HOSTCHECK=1` lifts them all.

**The pinned-vLLM warning is generic**, driven by `VARIANT_VLLM_PIN`, and shown
in the install menu and `./setup.sh status`.

**The flatten workaround generalized.** The OCI-manifest-with-Docker-layers bug
that made `FROM stilldeadcode/vllm-radiance` fail under podman is now handled
for any variant whose manifest sets `VARIANT_NEEDS_FLATTEN`, as the note
predicted would be needed.

### Two bugs the cross-checks caught

Worth recording because neither would have surfaced in normal use for a long
time.

- **bash and Go disagreed about sort order.** A glob expands in locale
  collation order, which ignores punctuation and puts `rocm-cdna` before
  `rocm`; Go sorts by byte value and puts `rocm` first. Fixed with `LC_ALL=C`.
  Found by the parity test that runs `setup.sh`'s own reader functions against
  every manifest and compares to the Go parser.

- **A knob's placeholder is not an example value.** It is grey hint text in the
  UI: `1:8,2:7,4:6` is a value, `off — try: auto` is prose. The
  `.env.example` generator emitted `#RADIANCE_NUMA_BIND=off — try: auto` until
  it learned to use a placeholder only when it contains no whitespace.

---

## 5. Deliberately excluded

Unchanged from the original note, and still right:

- **vLLM hardware plugins** (Ascend, Gaudi, TPU, Spyre, Tenstorrent, MetaX,
  Rebellions). These are `vllm.platform_plugins` pip packages, not base images.
  Different integration shape, and none of that hardware appears in homelabs.
- **TensorRT-LLM / Dynamo runtime images.** A different engine, not a vLLM
  base. Would need a parallel process-control path.
- **`patcarter883/rdna4-vllm`.** Fat gfx1200+gfx1201 wheels rather than an
  image, so a build input and not a `FROM` target. **Verified:** no Docker Hub
  repository exists, consistent with that. Its motivating case — RX 9060 XT
  (gfx1200) coverage — is now served by `rdna4-clav` instead.

---

## 6. Alternates considered and not shipped

Both are real, active and could be added as one manifest each. Neither is, for
the reasons given.

- **`magiccodingman/vllm-radiance`** — **verified**: 28 tags, newest
  2026-09-09, versioned `1.0` / `1.0.16`. The note describes it as the same
  `RADIANCE_*` surface and libr4d kernels pinned to vLLM 0.28.0 with native
  gfx1201 MXFP4/W4A8, marked experimental with speculative modes opt-in because
  a cross-mode output-equivalence gate has not passed. If that holds it is a
  drop-in for `radiance` by base image alone, and would partly answer
  radiance's "a model newer than the pin will not load" problem. Not shipped
  because the equivalence caveat is exactly the kind of thing that should be
  understood before it is offered in a menu.

- **`capicua25x/vllm-rocm-rdna4`** — **verified**: 13 tags, newest 2026-09-03,
  `0.28.0-rdna4`. Described upstream as vLLM 0.28.0 for R9700/9070XT with MTP-3
  and DFlash2-FP8 speculative decoding, tuned per-shape GEMM configs
  regenerated with vLLM's own tuner, geared to concurrent serving at 32
  sequences. Overlaps `radiance` heavily. Worth adding if concurrency-tuned
  serving becomes a stated goal.

---

## 7. What is still unproven

Listed plainly, because the manifests read with more confidence than the
evidence supports.

1. **One variant has been built and served; ten have not.** `rocm` was built
   and run on an RX 7900 XTX (gfx1100) on 2026-09-11 — a 4B AWQ model loaded
   and answered — which proves `Dockerfile.prebuilt` works against a base it
   was not written for, and that the `/opt/python` venv root read off the image
   config was right. Everything else is still a registry config and an
   inference.

   That build found a real defect, which is worth recording because the
   prediction here was only half right. The assertion did fire on a wrong pin,
   and the value it suggested would have broken the build one step earlier:
   `VLLM_VERSION` was both the git ref the tuner script is fetched from and the
   version compared against, and AMD's image reports
   `0.23.1.dev1+g9ddef7117.d20260901.rocm714` — not a ref, and `v0.23.1` does
   not exist as a tag. Fixed by splitting `VARIANT_TUNER_REF` out of
   `VARIANT_VLLM_PIN` and comparing on the minor. So: a good failure mode, with
   bad advice attached, found within minutes of the first real build.

2. **`xpu` and `gb10` cannot be validated by anyone here.** `gb10` is gated on
   `aarch64` so it cannot be offered by mistake. `xpu` carries the PCI id list
   described in §3, and its host-stack pinning is unverified.

3. **`strix-halo`'s attention-backend opinions are not encoded**, because they
   were not checked. Its `VARIANT_ATTENTION_BACKENDS` is empty, so it gets the
   AMD vendor list.

4. **The support tiers are mostly inherited, not earned.** `rocm` was promoted
   to `tested` on the strength of the build above; it is the only one this work
   moved. `rdna4-clav` and `radiance` on the RDNA4 box are the next cheapest,
   and `cuda` on the NVIDIA card after that.

5. **Serving under SELinux logs an io_uring denial**, on Fedora and its
   relatives. vLLM's engine asks for an io_uring instance, the container policy
   refuses, and vLLM falls back silently — the model loads and serves, and the
   engine log contains nothing about it. Observed on the `rocm` build. It is
   documented in the README because the alert `setroubleshoot` raises suggests
   an `audit2allow` module that would grant io_uring to every container on the
   machine, which is a bad trade for a fallback that already works.

6. **`rdna4-clav` is a name this project chose.** The upstream repository is
   `tcclaviger/vllm`; "clav" comes from the CLAV kernels the image itself
   names. If it acquires a name of its own, the manifest should follow it.

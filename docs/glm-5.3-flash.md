# GLM-5.3-Flash on rdna4-clav

How `Intel/GLM-5.3-Flash-W4A16-AutoRound` was brought up on four Radeon AI PRO
R9700s (the `rdna4-clav` variant), following the
[GLM 5.3 Flash recipe](https://blog.robai.net/vllmdocs/#recipes-glm-5-3-flash),
and what had to change for it to start. Working as of 2026-10-06.

## The checkpoint

The recipe describes the model as "W4A16 with Intel AutoRound (GPTQ-packed int4
experts, everything else bf16)" but never names a repository. The one that
matches is **`Intel/GLM-5.3-Flash-W4A16-AutoRound`**:

- `quant_method: auto-round`, `packing_format: auto_round:auto_gptq`, 4-bit,
  group size 128, symmetric. Attention, shared experts, gates and embeddings
  are held at 16 bits; only the routed experts are int4.
- `num_nextn_predict_layers: 1`, so the recipe's MTP speculative config works.
- About 181.5 GB on disk.

The other W4A16 and AWQ uploads use compressed-tensors or AWQ packing rather
than AutoRound's GPTQ packing, and the NVFP4 and GGUF builds target other
hardware or runtimes.

## Model settings

The recipe is a compose file. This is how it maps onto the model's
`vllm_config` (set from the Models page, or with
`PUT /api/models/config?id=Intel/GLM-5.3-Flash-W4A16-AutoRound`):

| Field | Value |
|---|---|
| `tensor_parallel_size` | `4` |
| `max_model_len` | `262144` |
| `gpu_memory_utilization` | `0.95` |
| `max_num_seqs` | `8` |
| `enable_chunked_prefill` | `true` |
| `max_num_batched_tokens` | `2048` |
| `kv_cache_dtype` | `auto` |
| `trust_remote_code` | `false` |
| `enable_auto_tool_choice` | `true` |
| `tool_call_parser` | `glm47` |
| `reasoning_parser` | `glm45` |
| `speculative_config` | `{"method": "mtp", "num_speculative_tokens": 3}` |
| `compilation_config` | `{"cudagraph_capture_sizes": [4], "max_cudagraph_capture_size": 4}` |
| `extra_flags` | `--enable-expert-offload --expert-resident-layers 3 --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 2 --mm-processor-cache-gb 0.2` |
| `env` | `OMP_NUM_THREADS=8`, `CLAV_TUNABLEOP_SWEEP=0`, `CLAV_ATTN_AUTOTUNE=0`, `VLLM_ROCM_USE_AITER=0`, `GPU_MAX_HW_QUEUES=2`, `HSA_ENABLE_INTERRUPT=1`, `HSA_ENABLE_MWAITX=1` (one per line) |

Left out of the recipe on purpose:

- Port, host, `--served-model-name` and the model path: vllmctl sets these, so
  the model serves on `:8000` under its repository ID rather than on `:8080`.
- The volumes, `shm_size` and devices: the container already keeps its Triton,
  vLLM and Inductor caches under `/data/cache` and has a 95 GB `/dev/shm`.
- `trust_remote_code`: registration turned it on automatically. The recipe
  does not use it.

The parsers are the GLM 4.x ones; 5.3 Flash speaks them unchanged.

### `--expert-resident-layers` takes indices

The recipe describes `--expert-resident-layers 3` as pinning "the first three
layers". In this fork the flag is a comma-separated list of layer **indices**
(default `0,1,2`), so `3` keeps every expert of layer 3 in VRAM and nothing
else. GLM's layers 0-2 are dense, so layer 3 is its first MoE layer and the
flag is valid as written. To pin three MoE layers, use `3,4,5`.

### Host memory

The experts that do not fit in the VRAM cache live in pinned host memory. On a
188 GB host the load peaked at about 156 GB used, with roughly 34 GB pinned per
rank. Starting with less than about 160 GB available is likely to fail.

## What went wrong, and the fix

### Base 29.07.1: KV cache short at any context

On the `tcclaviger/vllm:latest` base (release 29.07.1), the first start ran for
about ten minutes and then failed:

```
ValueError: To serve at least one request with the model's max seq len (262144),
(3.32 GiB KV cache is needed, which is larger than the available KV cache memory (2.9 GiB).
```

The fork's expert planner (`vllm/model_executor/offloader/expert_plan.py`)
splits each card's budget like this:

```
budget (VRAM x gpu_memory_utilization)
  - non-expert weights
  - KV reserve for one max-model-len request (+2% for GLM)
  - CUDA graphs
  - runtime overhead (per-architecture figure from the registry)
  = expert cache
```

The expert cache takes everything that is left, so the real KV space is the
reserve minus however far the overhead figure undershoots. GLM's figure in
`vllm/model_executor/offloader/expert_activation_registry.py` is 6.25 GiB
(`_glm53_overhead_default`). The plan check logged the real value:

```
expert offload plan check (this rank): runtime overhead planned 6.25 GiB, measured 6.8 GiB (+0.55)
```

That 0.55 GiB comes straight out of the KV space. Lowering `max_model_len` or
`gpu_memory_utilization` does not help, because the planner shrinks the KV
reserve or the budget to match and the cache absorbs the difference.

### Base 29.08.1 (`:dev`) rejects GLM

The recipe names `tcclaviger/vllm:dev`, which at the time was release 29.08.1.
Moving to it:

```bash
VLLMCTL_BASE_IMAGE=docker.io/tcclaviger/vllm:dev ./setup.sh pull
```

did not help. Its GLM overhead figure is still 6.25 GiB, and its rewritten
("v2") planner refuses the architecture before anything loads:

```
ExpertPlanError: Glm5NextForConditionalGeneration has no get_kv_cache_layers_from_config;
the v2 plan sizes KV from the model's config-time specs and does not guess
```

The install was rolled back with the command `setup.sh` prints for that:

```bash
VLLMCTL_BASE_IMAGE=localhost/vllmctl-rdna4-clav-base:a7b08ce3268e ./setup.sh pull
```

### The fix: raise the overhead figure

On 29.07.1, GLM's default overhead was raised from 6.25 to 7.0 GiB inside the
running container, with the original kept beside it:

```bash
podman exec vllm-toolchest sh -c '
  f=/opt/vllm/lib/python3.14/site-packages/vllm/model_executor/offloader/expert_activation_registry.py
  cp "$f" "$f.orig"
  sed -i "/def _glm53_overhead_default/,/return/ s/int(6.25 \* GIB/int(7.0 * GIB/" "$f"
'
```

The next start reached `running` in about four minutes:

```
runtime overhead planned 7.0 GiB, measured 6.76 GiB (-0.24)
GPU KV cache size: 269,664 tokens, Maximum concurrency for 262,144 tokens per request: 1.03x
```

The expert cache shrank slightly (13.9 GiB per card, against 14.65 before), and
a test chat request answered with reasoning split out correctly.

**The patch does not survive a rebuild.** `setup.sh pull`, `rebuild` and
`quick` all recreate the container from the image, so the edit has to be
reapplied afterwards. Without it, GLM fails at the KV check about ten minutes
into loading.

## Upstream

Both problems are in the fork rather than in vllm-toolchest. They are worth
reporting to its author:

- On 29.07.1, GLM's default overhead (6.25 GiB) is about 0.55 GiB below what
  four R9700s measure at TP4, MTP-3, 2048 batched tokens.
- On 29.08.1, the v2 expert planner needs `get_kv_cache_layers_from_config`,
  which `Glm5NextForConditionalGeneration` does not implement.

Once a fixed base is published, a plain `./setup.sh pull` (or a pull with the
`:dev` override) should run the recipe without the patch.

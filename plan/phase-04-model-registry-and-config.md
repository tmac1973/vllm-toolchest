# Phase 4: Model Registry & Configuration

Manage downloaded models and their vLLM configurations. Detect model capabilities (quantization, tool use, vision), estimate VRAM requirements, and provide a detailed per-model configuration UI.

---

## Model Registry Data Model

### models.json Schema

The registry lives at `/data/config/models.json`. It is a JSON object keyed by model ID (the HuggingFace repo slug, e.g. `Qwen/Qwen2.5-72B-Instruct-AWQ`). The Go struct backing this is `internal/models/registry.go:Model`.

```jsonc
{
  "models": {
    "Qwen/Qwen2.5-72B-Instruct-AWQ": {
      // --- Identity ---
      "id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
      "display_name": "Qwen 2.5 72B Instruct AWQ",
      "local_path": "/data/models/Qwen/Qwen2.5-72B-Instruct-AWQ",
      "enabled": true,
      "download_date": "2026-03-15T10:30:00Z",
      "total_size_bytes": 42949672960,

      // --- HuggingFace Config Metadata ---
      "hf_config": {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "vocab_size": 152064,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
        "rope_scaling": {
          "type": "yarn",
          "factor": 4.0
        },
        "tie_word_embeddings": false
      },

      // --- Quantization Metadata ---
      "quantization": {
        "method": "awq",          // awq | gptq | fp8 | gguf | bitsandbytes | marlin | squeezellm | compressed_tensors | none
        "bits": 4,
        "group_size": 128,
        "desc_act": false,         // GPTQ only: descending activation order
        "sym": true,               // Symmetric quantization (relevant for Marlin upgrade)
        "awq_version": "gemm",     // AWQ only: gemm | gemv | marlin
        "quant_config_source": "quantize_config.json",  // Where quant info was found
        "gguf_quant_type": "",     // GGUF only: Q4_K_M, Q5_K_S, etc.
        "bytes_per_param": 0.5625  // Calculated: for GPTQ-4bit w/ group scales
      },

      // --- Tool Use Metadata ---
      "tool_use": {
        "has_tool_support": true,
        "tool_call_parser": "hermes",    // hermes | llama3_json | granite | mistral | internlm | jamba | pythonic | ""
        "chat_template_source": "tokenizer_config.json",
        "detection_method": "chat_template_regex",  // how we detected: chat_template_regex | known_model_family | manual
        "tool_template_patterns_found": ["<tool_call>", "functions"]  // Which patterns matched
      },

      // --- Vision Metadata ---
      "vision": {
        "is_vision_model": false,
        "has_processor_config": false,
        "has_image_processor": false,
        "has_preprocessor_config": false,
        "image_size": 0
      },

      // --- Generation Defaults (from generation_config.json) ---
      "generation_defaults": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "max_new_tokens": 8192,
        "do_sample": true
      },

      // --- VRAM Estimate ---
      "vram_estimate": {
        "param_count_billion": 72.7,
        "weight_memory_gb": 38.5,
        "kv_cache_per_token_bytes": 2621440,
        "kv_cache_at_max_ctx_gb": 20.48,
        "activation_overhead_gb": 1.5,
        "total_single_gpu_gb": 60.48,
        "total_per_gpu_tp2_gb": 30.24,
        "fits_single_gpu": false,       // Based on 32GB VRAM
        "needs_tp2": true,
        "too_large": false,
        "recommended_tp": 2,
        "recommended_max_ctx_single": 0,   // 0 = does not fit single GPU
        "recommended_max_ctx_tp2": 16384
      },

      // --- Per-Model vLLM Configuration ---
      "vllm_config": {
        // ... see detailed section below
      }
    }
  },
  "schema_version": 2,
  "last_scan": "2026-03-15T12:00:00Z"
}
```

### Notes on the Schema

- `id` is always the HuggingFace repo slug. For locally-placed models (not downloaded via HF), use the directory name relative to `/data/models/`.
- `total_size_bytes` is sum of all files on disk (safetensors, configs, tokenizer, etc.).
- `hf_config` is a denormalized snapshot of key fields from `config.json`. We do NOT store the entire config.json -- only the fields we need for VRAM estimation, display, and decision-making.
- `quantization.method` is `"none"` for FP16/BF16 models. We still track `torch_dtype` in `hf_config` to know whether it is FP16 or BF16.
- `vram_estimate` is recomputed whenever `hf_config` or `vllm_config` changes.

---

## Per-Model vLLM Configuration Schema

Stored as `vllm_config` within each model entry. Every field maps to a `vllm serve` CLI flag.

```jsonc
{
  "vllm_config": {
    // --- Core ---
    "dtype": "auto",                    // auto | float16 | bfloat16 | float32
                                        // Flag: --dtype
                                        // "auto" uses model's torch_dtype

    "max_model_len": 8192,             // Override context window length
                                        // Flag: --max-model-len
                                        // 0 = use model default (max_position_embeddings)
                                        // Must be <= max_position_embeddings (or rope_scaling adjusted)

    "tensor_parallel_size": 1,          // 1 or 2 (number of GPUs)
                                        // Flag: --tensor-parallel-size

    "gpu_memory_utilization": 0.90,     // 0.10 to 0.99
                                        // Flag: --gpu-memory-utilization
                                        // Lower = more headroom for KV cache spikes
                                        // Higher = can fit larger context or model

    // --- Performance ---
    "enforce_eager": false,             // Disable CUDA/HIP graph compilation
                                        // Flag: --enforce-eager
                                        // Set true if graph compilation crashes (common on new ROCm)
                                        // Set true for debugging memory issues
                                        // Slower steady-state but faster startup

    "enable_prefix_caching": true,      // AKA automatic prefix caching (APC)
                                        // Flag: --enable-prefix-caching
                                        // Reuses KV cache for shared prefixes (system prompts)
                                        // Generally safe to enable

    "enable_chunked_prefill": false,    // Chunk long prompts during prefill
                                        // Flag: --enable-chunked-prefill
                                        // Helps with long-context latency spikes
                                        // May slightly reduce throughput

    "max_num_batched_tokens": 0,        // Max tokens in a single batch step
                                        // Flag: --max-num-batched-tokens
                                        // 0 = vLLM default (auto-calculated)
                                        // Lower = less memory, higher latency
                                        // Usually paired with chunked prefill

    "max_num_seqs": 16,                 // Max concurrent sequences
                                        // Flag: --max-num-seqs
                                        // Allowed values: 1, 4, 8, 16, 32, 64, 128, 256
                                        // Lower = less memory, more predictable latency
                                        // Single-user: 1-4. Multi-user: 16-64.

    // --- Quantization ---
    "quantization": "awq",              // awq | gptq | fp8 | gguf | bitsandbytes | marlin |
                                        // squeezellm | compressed_tensors | ""
                                        // Flag: --quantization
                                        // "" = auto-detect from model config
                                        // Usually matches quantization.method from metadata
                                        // Can override: e.g. set "marlin" for a GPTQ model

    "load_format": "auto",              // auto | gguf | safetensors | pt | ...
                                        // Flag: --load-format
                                        // "gguf" must be set for GGUF files
                                        // "auto" works for safetensors/pytorch

    "kv_cache_dtype": "auto",           // auto | fp8 | fp8_e5m2 | fp8_e4m3
                                        // Flag: --kv-cache-dtype
                                        // fp8 halves KV cache memory, slight quality loss
                                        // Useful for fitting larger contexts

    // --- Trust & Code ---
    "trust_remote_code": false,         // Flag: --trust-remote-code
                                        // Required for some models (Yi, InternLM, etc.)
                                        // Security risk: executes arbitrary Python from model repo

    // --- Tool Use ---
    "enable_auto_tool_choice": false,   // Flag: --enable-auto-tool-choice
                                        // Allows model to decide when to use tools
                                        // Requires tool_call_parser to also be set

    "tool_call_parser": "",             // hermes | llama3_json | granite | mistral |
                                        // internlm | jamba | pythonic | ""
                                        // Flag: --tool-call-parser
                                        // Must match model's chat template format
                                        // "" = disabled

    // --- Overrides ---
    "tokenizer": "",                    // Override tokenizer path
                                        // Flag: --tokenizer
                                        // Useful for models with broken tokenizer configs

    "chat_template": "",                // Override chat template file path
                                        // Flag: --chat-template
                                        // Path to a Jinja2 template file
                                        // Use for fixing broken templates or adding tool support

    "extra_flags": ""                   // Raw string appended to vllm serve command
                                        // Allows: --disable-log-requests, --swap-space, etc.
                                        // User's escape hatch for any flag we don't expose
  }
}
```

### Configuration Constraints & Validation

- `tensor_parallel_size`: Must not exceed number of detected GPUs. Show warning and disable TP=2 if only 1 GPU found.
- `gpu_memory_utilization`: Clamp to [0.10, 0.99]. Show warning below 0.50 ("unusually low") and above 0.95 ("may OOM on KV cache allocation").
- `max_model_len`: Must be > 0 and <= model's `max_position_embeddings` (unless rope_scaling extends it). Show yellow warning when > recommended context for VRAM.
- `max_num_seqs`: Must be one of the allowed values. Higher values require more memory for scheduling overhead.
- `quantization` + model mismatch: Warn if user sets `quantization` to a value that doesn't match `quantization.method`. E.g. setting `"awq"` on an FP16 model will crash vLLM.
- `enable_auto_tool_choice` without `tool_call_parser`: Warn that tool choice will be ignored without a parser.
- `tool_call_parser` without `enable_auto_tool_choice`: Warn that parser is set but auto tool choice is disabled.
- `marlin` quantization: Only valid if original quant is GPTQ or AWQ with symmetric quantization and supported group sizes (128 or channel-wise). Show compatibility check result.
- `load_format` must be `"gguf"` when model contains `.gguf` files.
- `trust_remote_code`: Show security warning icon. Default false.
- `extra_flags`: Validate no conflicting flags (e.g. don't allow `--model` or `--port` in extra flags since we control those).

### Default Values by Quantization Type

When a model is first registered, set intelligent defaults based on its quantization:

| Quant Method | dtype | quantization flag | kv_cache_dtype | enforce_eager | Notes |
|---|---|---|---|---|---|
| none (FP16) | auto | "" | auto | false | Largest VRAM footprint |
| none (BF16) | auto | "" | auto | false | Same size as FP16, better training dtype |
| awq | auto | awq | auto | false | 4-bit weights, ~4x compression |
| gptq | auto | gptq | auto | false | 4-bit weights, slightly larger than AWQ due to group scales |
| fp8 | auto | fp8 | fp8 | false | 8-bit, half the FP16 size |
| gguf | auto | gguf | auto | false | load_format must be "gguf" |
| bitsandbytes | float16 | bitsandbytes | auto | true | Requires eager mode; dynamic quantization |
| marlin | auto | marlin | auto | false | Optimized kernel for GPTQ/AWQ; check compatibility |
| squeezellm | auto | squeezellm | auto | false | Rare; older quantization method |
| compressed_tensors | auto | compressed_tensors | auto | false | vLLM's native compressed format |

---

## HF config.json Parser (`internal/models/hfconfig.go`)

### Files to Parse

For each model directory, parse these files in order:

1. **`config.json`** -- Primary model architecture config
2. **`quantize_config.json`** -- GPTQ/AWQ quantization details (may not exist)
3. **`tokenizer_config.json`** -- Chat template, tokenizer class, tool use detection
4. **`generation_config.json`** -- Default sampling parameters (may not exist)
5. **`processor_config.json`** -- Vision model indicator (may not exist)
6. **`preprocessor_config.json`** -- Alternative vision model indicator
7. **`*.gguf`** -- GGUF file presence (glob, don't parse headers)

### config.json Parser

Go struct for the subset of fields we need:

```
HFConfig struct:
  Architectures        []string     // e.g. ["LlamaForCausalLM"]
  ModelType            string       // e.g. "llama", "qwen2", "mistral"
  NumHiddenLayers      int          // Transformer layer count
  HiddenSize           int          // Embedding dimension
  IntermediateSize     int          // MLP intermediate dimension (FFN)
  NumAttentionHeads    int          // Query heads
  NumKeyValueHeads     int          // KV heads (for GQA; == NumAttentionHeads for MHA)
  HeadDim              int          // Explicit head dim (some models); else HiddenSize / NumAttentionHeads
  MaxPositionEmbeddings int         // Base context length
  VocabSize            int
  TorchDtype           string       // "float16", "bfloat16", "float32"
  RopeTheta            float64      // RoPE base frequency
  RopeScaling          map          // type, factor (for extended context)
  TieWordEmbeddings    bool         // If true, input/output embeddings are shared (affects param count)
  SlidingWindow        *int         // nil = no sliding window, else window size
  QuantizationConfig   map          // Some models embed quant config here instead of separate file
```

### Architecture-Specific Field Name Handling

Different model families use different field names for the same concepts. The parser must handle aliases:

| Canonical Field | Variants |
|---|---|
| `num_key_value_heads` | `num_kv_heads`, absent (implies == num_attention_heads for MHA) |
| `head_dim` | absent (compute from hidden_size / num_attention_heads) |
| `intermediate_size` | `ffn_dim` (some architectures) |
| `max_position_embeddings` | `max_sequence_length`, `seq_length`, `n_positions` |
| `num_hidden_layers` | `n_layer`, `num_layers`, `n_layers` |
| `hidden_size` | `d_model`, `n_embd` |
| `num_attention_heads` | `n_head`, `num_heads`, `n_heads` |

Implementation: try canonical name first, then fall back to variants. Log a warning if a critical field is missing entirely.

### quantize_config.json Parser

Present for GPTQ and AWQ models:

```
QuantizeConfig struct:
  QuantMethod    string  // "gptq" or "awq"
  Bits           int     // 2, 3, 4, 8
  GroupSize      int     // 32, 64, 128, -1 (channel-wise)
  DescAct        bool    // GPTQ only: descending activation order
  Sym            bool    // Symmetric quantization
  TrueSequential bool    // GPTQ sequential quantization
  AWQVersion     string  // AWQ: "gemm", "gemv", "marlin"
```

Also check `config.json` -> `quantization_config` field, which some models use instead of a separate file. Same schema but nested under the config key.

### tokenizer_config.json Parser -- Tool Use Detection

Parse `chat_template` field (a Jinja2 template string). Detect tool use support via regex pattern matching:

**Patterns indicating tool/function calling support:**

| Pattern | Parser Type | Model Families |
|---|---|---|
| `<tool_call>` or `<\|tool_call\|>` | hermes | Hermes-2, NousResearch models |
| `tool_calls` in template | varies | Generic tool call support |
| `<\|python_tag\|>` | pythonic | Some code-oriented models |
| `"type": "function"` in template | llama3_json | Llama 3.1+, Llama 3.2+ |
| `[TOOL_CALLS]` or `[AVAILABLE_TOOLS]` | mistral | Mistral, Mixtral |
| `<\|plugin\|>` | internlm | InternLM models |
| `<function=` | granite | IBM Granite models |
| `<tool_response>` or `tool_response` | varies | Generic tool response handling |

**Detection algorithm:**

1. Read `chat_template` string from `tokenizer_config.json`.
2. Test each regex pattern against the template.
3. If multiple patterns match, prefer the most specific one (model-family-specific over generic).
4. Also check model name/architecture against known tool-capable families as fallback:
   - `Hermes` in model name -> hermes parser
   - `Llama-3.1`, `Llama-3.2`, `Llama-3.3` -> llama3_json parser
   - `Mistral`, `Mixtral` -> mistral parser
   - `Granite` -> granite parser
   - `InternLM` -> internlm parser
   - `Qwen2.5` -> hermes parser (Qwen 2.5 uses Hermes-style tool calling)
   - `Jamba` -> jamba parser
5. Store detection method: `"chat_template_regex"` or `"known_model_family"`.
6. Allow manual override via `vllm_config.tool_call_parser`.

**Edge cases:**
- Some models have a `chat_template` that is a list of dicts (multiple templates for different scenarios). In this case, check the one with `"name": "tool_use"` if present, otherwise check all templates.
- Some models have `chat_template` as a path to a file rather than inline. Resolve and read the file.
- Tool detection false positives: a template mentioning "tool" in a comment or documentation string. The regex should look for structural patterns, not just the word "tool".

### generation_config.json Parser

Optional file with default sampling parameters:

```
GenerationConfig struct:
  Temperature       *float64
  TopP              *float64
  TopK              *int
  RepetitionPenalty *float64
  MaxNewTokens      *int
  DoSample          *bool
  Eos_token_id      interface{}  // Can be int or []int
```

Use pointer types to distinguish "not set" from zero values. Only populate `generation_defaults` for fields that are present.

### Vision Model Detection

A model is a vision model if ANY of the following are true:
- `processor_config.json` exists in model directory
- `preprocessor_config.json` exists in model directory
- `image_processor` field in `config.json` is non-null
- Architecture name contains "Vision", "VL", "Pix" (e.g. `LlavaForConditionalGeneration`, `Qwen2VLForConditionalGeneration`)
- `config.json` contains `vision_config` key

Store all detection signals in the `vision` metadata block.

---

## VRAM Estimation (`internal/models/vram.go`)

### Parameter Count Estimation

Estimate parameter count from `config.json` fields. This is an approximation -- actual param count requires reading safetensors metadata, which is expensive.

**Transformer parameter formula:**

```
Embedding:
  vocab_embedding = vocab_size * hidden_size
  output_embedding = vocab_size * hidden_size (if not tie_word_embeddings, else 0)

Per-layer:
  // Self-attention
  q_proj = hidden_size * num_attention_heads * head_dim
  k_proj = hidden_size * num_key_value_heads * head_dim
  v_proj = hidden_size * num_key_value_heads * head_dim
  o_proj = num_attention_heads * head_dim * hidden_size
  attention_total = q_proj + k_proj + v_proj + o_proj

  // MLP (gate/up/down for LLaMA-style; dense_h_to_4h/dense_4h_to_h for GPT-style)
  mlp_total = 3 * hidden_size * intermediate_size  // gate + up + down for LLaMA
  // Some architectures: 2 * hidden_size * intermediate_size (no gate proj)

  // LayerNorm (small)
  layernorm = 2 * hidden_size  // input_layernorm + post_attention_layernorm

  layer_total = attention_total + mlp_total + layernorm

Total:
  total_params = vocab_embedding + output_embedding + (num_hidden_layers * layer_total) + final_layernorm
```

**Architecture-specific MLP factor:**
- LLaMA, Mistral, Qwen2, Gemma: 3x (gate_proj + up_proj + down_proj) -> `3 * hidden_size * intermediate_size`
- GPT-2, GPT-Neo, OPT: 2x (up + down) -> `2 * hidden_size * intermediate_size`
- Default to 3x if unknown architecture

### Bytes Per Parameter by Quantization

| Format | Bytes/Param | Notes |
|---|---|---|
| FP32 | 4.0 | Almost never used for inference |
| FP16 / BF16 | 2.0 | Standard unquantized |
| FP8 (E4M3/E5M2) | 1.0 | |
| INT8 (bitsandbytes 8-bit) | 1.0 | Plus ~10% overhead for absmax scales |
| INT4 (bitsandbytes 4-bit) | 0.5 | Plus ~12% overhead for scales/zeros |
| GPTQ 4-bit | ~0.5625 | 0.5 base + group scales (128 group_size: 2 bytes per 128 params / 2) |
| GPTQ 3-bit | ~0.4375 | |
| GPTQ 8-bit | ~1.0625 | |
| AWQ 4-bit | ~0.5625 | Similar overhead to GPTQ |
| Marlin (4-bit) | ~0.5625 | Same storage, different kernel |
| SqueezeLLM | ~0.5 | Varies; approximate as 4-bit |
| GGUF Q4_K_M | ~0.5625 | Mixed quantization; approximate |
| GGUF Q5_K_M | ~0.6875 | |
| GGUF Q6_K | ~0.8125 | |
| GGUF Q8_0 | ~1.0625 | |

**GPTQ/AWQ overhead calculation:**
```
overhead_per_param = (dtype_bytes_for_scales / group_size) * 2  // scales + zeros
bytes_per_param = (bits / 8) + overhead_per_param
```

For GPTQ 4-bit, group_size 128: `(4/8) + (2/128)*2 = 0.5 + 0.03125 = 0.53125`. In practice closer to 0.5625 due to additional metadata per group.

### KV Cache Size Estimation

```
kv_cache_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes

// dtype_bytes for KV cache:
//   kv_cache_dtype == "auto"    -> 2 (FP16)
//   kv_cache_dtype == "fp8"     -> 1
//   kv_cache_dtype == "fp8_e5m2"-> 1
//   kv_cache_dtype == "fp8_e4m3"-> 1

kv_cache_total = kv_cache_per_token * max_model_len * max_num_seqs
```

**Important:** This is the MAXIMUM KV cache. vLLM uses paged attention and allocates KV cache blocks dynamically. The `gpu_memory_utilization` controls how much total GPU memory vLLM will use, and the remaining after model weights is used for KV cache.

More useful metric: **max tokens that fit in KV cache given VRAM budget**:

```
available_for_kv = (total_vram * gpu_memory_utilization) - weight_memory - activation_overhead
max_tokens_in_kv = available_for_kv / kv_cache_per_token
// max_tokens_in_kv is across all concurrent sequences
effective_max_ctx = max_tokens_in_kv / max_num_seqs
```

### Activation Memory Overhead

Rough estimate of transient memory during forward pass:

```
activation_memory ≈ batch_size * seq_len * hidden_size * 2 bytes * multiplier
// multiplier ≈ 4-6x for intermediate activations during attention + MLP
// For inference (not training), this is relatively small compared to weights + KV cache
// Approximate as 0.5-2.0 GB depending on model size
```

Simplified tiers for estimation:
- Models < 3B params: ~0.3 GB activation overhead
- Models 3B-13B: ~0.5 GB
- Models 13B-34B: ~1.0 GB
- Models 34B-72B: ~1.5 GB
- Models > 72B: ~2.0 GB

### Per-GPU Calculation

```
single_gpu_total = weight_memory + max_kv_cache_at_configured_ctx + activation_overhead

// For TP=2:
per_gpu_weights = weight_memory / 2  // Weights are sharded across GPUs
per_gpu_kv = kv_cache_total / 2       // KV cache is also sharded
per_gpu_activation = activation_overhead  // Activation memory is NOT evenly split; each GPU needs full activation for its shard
per_gpu_total_tp2 = per_gpu_weights + per_gpu_kv + per_gpu_activation
```

### Fit Labels

Given 32GB VRAM per GPU (R9700 XT):

| Condition | Label | Color |
|---|---|---|
| `single_gpu_total <= 32 * gpu_memory_utilization` | "Fits single GPU" | Green |
| `per_gpu_total_tp2 <= 32 * gpu_memory_utilization` | "Needs TP=2" | Yellow |
| `per_gpu_total_tp2 > 32 * gpu_memory_utilization` | "Too large" | Red |

Also compute `recommended_max_ctx_single` and `recommended_max_ctx_tp2`: the maximum context length that fits within the VRAM budget at the configured `gpu_memory_utilization` and `max_num_seqs`.

### VRAM Estimate Refresh Triggers

Recompute whenever any of these change:
- `max_model_len`
- `tensor_parallel_size`
- `gpu_memory_utilization`
- `kv_cache_dtype`
- `max_num_seqs`
- `quantization` flag override
- `dtype` override

---

## Quantization-Specific Logic

### Auto-Detection of Quantization Method

Run at model registration time and during metadata backfill. Check in priority order:

1. **`quantize_config.json` exists?** -- Parse it. Field `quant_method` gives "gptq" or "awq".
2. **`config.json` has `quantization_config`?** -- Parse nested config. Same fields as `quantize_config.json`.
3. **`.gguf` files present?** -- Set method to "gguf". Extract quant type from filename patterns:
   - Pattern: `*-Q4_K_M.gguf`, `*-Q5_K_S.gguf`, `*.q4_0.gguf`, etc.
   - Regex: `[.-](Q\d+_K?_?[A-Z]?|q\d+_\d+|[fF]\d+)\.gguf$`
   - Store extracted quant type in `quantization.gguf_quant_type`.
4. **`config.json` has `quantization_config.quant_method` == "compressed-tensors"?** -- Set method to "compressed_tensors".
5. **Check for FP8 indicators:** `config.json` has `quantization_config.quant_method` == "fp8" or model name contains "FP8".
6. **None of the above?** -- Set method to "none". Use `torch_dtype` from config.json to determine FP16 vs BF16.

### GGUF-Specific Handling

- vLLM can serve GGUF files directly but requires `--load-format gguf` flag.
- GGUF models may have a single `.gguf` file or be split into parts (`model-00001-of-00003.gguf`).
- The model path for vLLM should point to the directory (for split) or the specific file (for single).
- vLLM's GGUF support has limitations: not all quant types are supported. Generally Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 are well-supported. K-quants (Q4_K_M, Q5_K_S, etc.) have varying support depending on vLLM version.
- If a GGUF model has a `config.json` alongside it (from the original HF repo), parse it for architecture info. If not, architecture info must come from the GGUF filename or be manually specified.

### Marlin Auto-Upgrade Detection

Marlin is an optimized 4-bit kernel that can be used as a drop-in replacement for GPTQ and AWQ inference. It provides significant speedup but has compatibility requirements:

**Requirements for Marlin upgrade:**
- Original quantization is GPTQ or AWQ
- Symmetric quantization (`sym: true`)
- 4-bit quantization (`bits: 4`)
- Group size is 128 or -1 (channel-wise)
- `desc_act` is false (for GPTQ)
- AWQ version is "gemm" (not "gemv")

**Detection in UI:**
- If a model meets all Marlin requirements, show a badge: "Marlin-compatible"
- Offer a one-click option to switch quantization flag from `gptq`/`awq` to `marlin`
- Note: this doesn't change the model files, just the inference kernel

**Edge cases:**
- Marlin on ROCm: check vLLM ROCm Marlin kernel availability. Marlin kernels are CUDA-first; ROCm support may be partial or via Triton. If ROCm Marlin is not available, hide the Marlin option and show a note.
- AWQ version "marlin" in `quantize_config.json`: some AWQ models are pre-packaged for Marlin. These should use `marlin` quantization flag directly.

### BitsAndBytes Handling

BitsAndBytes (bnb) performs dynamic quantization at load time -- the model files are stored in FP16/BF16, and bnb quantizes on the fly.

**Specifics:**
- `--quantization bitsandbytes` flag
- Must also set `--enforce-eager true` (bnb is incompatible with CUDA/HIP graphs)
- vLLM config for bnb uses the model's native dtype (float16) as the compute dtype
- ROCm support: requires the ROCm fork of bitsandbytes (`bitsandbytes-rocm`)
- 4-bit mode: ~4x compression, NF4 or FP4 data type, double quantization optional
- 8-bit mode: ~2x compression, INT8 with outlier handling
- In the UI, when user selects `bitsandbytes` quantization:
  - Auto-enable `enforce_eager`
  - Show sub-options: 4-bit vs 8-bit (store as metadata; vLLM infers from model config)
  - Show warning: "Dynamic quantization -- slower model loading vs pre-quantized models"

### Compressed Tensors

vLLM's native quantization format. Models with `quant_method: "compressed-tensors"` in their config:
- No special flags beyond `--quantization compressed_tensors`
- Supports mixed precision (different layers at different bit widths)
- The `quantization_config` in `config.json` contains full details about per-layer quantization schemes

---

## Model Registry Startup Maintenance

Run at application startup (`internal/models/registry.go:Maintenance()`):

### 1. Scan for Unregistered Models

```
Walk /data/models/ directory:
  For each subdirectory (1-2 levels deep, matching org/repo pattern):
    If contains config.json OR *.gguf files:
      If not in models.json registry:
        Parse config files
        Create new registry entry with defaults
        Log: "Discovered unregistered model: <path>"
```

**Directory patterns to recognize:**
- `/data/models/Qwen/Qwen2.5-72B-Instruct-AWQ/` -- org/repo
- `/data/models/TheBloke/some-model-GGUF/` -- org/repo (GGUF)
- `/data/models/local-model/` -- no org, just repo name

### 2. Verify Registered Models Exist

```
For each entry in models.json:
  If local_path does not exist OR is empty:
    Mark as orphaned: set "orphaned": true, "orphaned_at": timestamp
    Log: "Model path missing: <path>"
```

Do NOT auto-delete orphaned entries. The user may have temporarily unmounted a volume.

### 3. Backfill Metadata

```
For each registered model:
  If hf_config is empty/incomplete:
    Re-parse config files from local_path
    Update hf_config, quantization, tool_use, vision fields
    Recompute vram_estimate
    Log: "Backfilled metadata for: <id>"
```

This handles the case where the schema was upgraded (new fields added) and existing models need the new fields populated.

### 4. Detect Config Schema Migration

Store `schema_version` in models.json. When the application's expected version > stored version:
- Run migration logic (add new fields with defaults, rename fields, etc.)
- Update `schema_version`
- Log: "Migrated models.json from schema v1 to v2"

### 5. Update last_scan Timestamp

Set `last_scan` to current time after maintenance completes.

---

## Models Page UI (`web/templates/models.html`)

### Model List View

Table/card layout with columns:

| Column | Content | Notes |
|---|---|---|
| Name | Display name + repo ID | Clickable to expand config panel |
| Architecture | Model type (e.g. "Llama", "Qwen2", "Mistral") | From `hf_config.model_type` |
| Quant | Quantization badge (e.g. "AWQ 4-bit", "GPTQ 4-bit g128", "FP16", "GGUF Q4_K_M") | Color-coded by type |
| Size | Total disk size | e.g. "40.2 GB" |
| Params | Estimated parameter count | e.g. "72.7B" |
| VRAM | Estimated VRAM with fit label | e.g. "38.5 GB (TP=2)" with green/yellow/red label |
| Tool Use | Tool support badge | Green checkmark if `has_tool_support`, gray X if not. Hover shows parser type. |
| Vision | Vision badge | Eye icon if vision model |
| Status | Enabled/disabled toggle | htmx PATCH to `/api/models/{id}/toggle` |
| Actions | Expand config / Delete | |

**Sorting:** Default by name. Allow sort by size, VRAM estimate, quant type.

**Filtering:** Quick filter buttons: All, Quantized, Full Precision, Tool-capable, Vision.

### Expandable Per-Model Config Panel

When a model row is clicked/expanded, load the config panel via htmx (`GET /api/models/{id}/config-panel` returning HTML partial).

**Config sections grouped by category:**

#### Core Settings
- **dtype** -- Dropdown: auto, float16, bfloat16, float32
- **max_model_len** -- Number input with slider. Show model's `max_position_embeddings` as reference. Show VRAM impact in real-time.
- **tensor_parallel_size** -- Radio: 1, 2. Disabled if only 1 GPU. Shows per-GPU VRAM split.
- **gpu_memory_utilization** -- Slider 0.10-0.99 with number display. Default 0.90.

#### Performance
- **enforce_eager** -- Toggle. Show note: "Disable HIP graph compilation. Slower but more compatible."
- **enable_prefix_caching** -- Toggle. Show note: "Cache shared prefixes (system prompts)."
- **enable_chunked_prefill** -- Toggle. Show note: "Chunk long prompts to reduce latency spikes."
- **max_num_batched_tokens** -- Number input. 0 = auto. Only shown when chunked prefill enabled.
- **max_num_seqs** -- Dropdown: 1, 4, 8, 16, 32, 64, 128, 256. Show note about memory vs throughput tradeoff.

#### Quantization
- **Detected quant method** -- Read-only display of auto-detected method with details.
- **quantization override** -- Dropdown with all options. Default matches detected method. Show Marlin compatibility badge if applicable.
- **load_format** -- Dropdown: auto, gguf, safetensors, pt. Auto-set to "gguf" for GGUF models.
- **kv_cache_dtype** -- Dropdown: auto, fp8, fp8_e5m2, fp8_e4m3. Show note: "FP8 KV cache halves cache memory at slight quality cost."

#### Tool Use
- **Tool support detected** -- Read-only badge showing detection result and method.
- **enable_auto_tool_choice** -- Toggle. Gray out if no tool support detected (but still allow override).
- **tool_call_parser** -- Dropdown: (none), hermes, llama3_json, granite, mistral, internlm, jamba, pythonic. Pre-populated from auto-detection. Show note about which models use which parser.
- **chat_template override** -- File path input. For fixing broken tool templates or using custom ones.

#### Trust & Overrides
- **trust_remote_code** -- Toggle with security warning icon.
- **tokenizer override** -- Text input for path.
- **extra_flags** -- Text area. Show note: "Raw flags appended to vllm serve command. One flag per line or space-separated."

#### VRAM Estimate Display
- Located at the top of the config panel, always visible.
- Shows: weight memory, KV cache at configured context, activation overhead, total.
- Shows per-GPU breakdown for current TP setting.
- Shows fit label with color.
- **Updates in real-time** via htmx as config values change:
  - Each input that affects VRAM has `hx-trigger="change"` pointing to `POST /api/models/{id}/estimate-vram` which returns the updated VRAM display partial.
  - Debounce slider changes to avoid excessive requests (use htmx `hx-trigger="change delay:300ms"`).

### Delete Model

- Button at bottom of expanded config panel.
- Click shows confirmation dialog (htmx modal or `hx-confirm`): "Delete <model name>? This will remove model files from disk and the registry entry."
- `DELETE /api/models/{id}` -- removes files from `/data/models/<path>` and removes registry entry.
- If model is currently loaded in vLLM, show additional warning: "This model is currently running. Stop the server first."

### API Endpoints for Models Page

- `GET /api/models` -- List all models (JSON array or HTML table partial)
- `GET /api/models/{id}` -- Single model details (JSON or HTML card)
- `GET /api/models/{id}/config-panel` -- HTML partial for the config editor
- `PUT /api/models/{id}/config` -- Update vLLM config for a model (JSON body with config fields)
- `PATCH /api/models/{id}/toggle` -- Toggle enabled/disabled
- `POST /api/models/{id}/estimate-vram` -- Recalculate VRAM estimate with provided config overrides (JSON body), returns VRAM display partial
- `DELETE /api/models/{id}` -- Delete model files and registry entry
- `POST /api/models/scan` -- Trigger manual rescan of `/data/models/`
- `POST /api/models/{id}/backfill` -- Re-parse config files and update metadata

### What to Copy from llama-toolchest

- **Model list HTML structure** -- Table layout with expandable rows. Adapt columns (replace GGUF-specific columns with quant type, tool use, etc.)
- **Config panel pattern** -- Expandable config form with htmx partial swaps. Replace llama.cpp flags with vLLM flags.
- **Enable/disable toggle** -- Same htmx PATCH pattern.
- **Delete confirmation** -- Same pattern.
- **Real-time VRAM estimate** -- Same htmx partial swap on config change. Replace GGUF VRAM formula with transformer formula.
- **Respond helper** -- JSON/HTML dual-mode response pattern from `respond.go`.

### What is New vs llama-toolchest

- **Tool use detection and configuration** -- Entirely new. llama-toolchest has no tool use UI.
- **HF config parsing** -- Replaces GGUF header parsing. Different fields, different Go structs.
- **Quantization variety** -- llama-toolchest only has GGUF quant types. Here we have AWQ, GPTQ, FP8, bnb, Marlin, compressed_tensors, GGUF, each with different config needs.
- **VRAM estimation formula** -- Completely different. Transformer param counting vs GGUF metadata.
- **Vision model detection** -- New.
- **Marlin compatibility detection** -- New.
- **Generation defaults from generation_config.json** -- New.
- **Config schema migration** -- New (llama-toolchest had simpler schema).

---

## File Layout Summary

```
internal/models/
  registry.go        -- Model struct, Load/Save models.json, CRUD operations, maintenance
  hfconfig.go        -- Parse config.json, quantize_config.json, tokenizer_config.json,
                        generation_config.json. Architecture-specific field handling.
  vram.go            -- VRAM estimation: param counting, bytes/param tables, KV cache,
                        fit labels, per-GPU calculations
  preset.go          -- (from llama-toolchest) Curated model presets / recommendations
  gpu_assign.go      -- (from llama-toolchest) Multi-GPU assignment logic

internal/api/
  models.go          -- HTTP handlers for all /api/models/* endpoints

web/templates/
  models.html        -- Full models page
  partials/
    model_row.html         -- Single model table row
    model_config_panel.html -- Expandable config editor
    vram_estimate.html      -- VRAM display fragment (for htmx swap)
```

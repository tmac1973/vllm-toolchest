# Phase 7: Settings & Configuration

**Goal:** Comprehensive persistent configuration system with a full settings UI, environment variable overrides, validation, and connection testing.

---

## 1. Configuration File Structure (`/data/config/vllmctl.yaml`)

The primary config file lives inside the container's persistent volume. On first boot, if absent, the Go binary writes a default config. All fields have sensible defaults so the application runs out of the box.

### Full YAML Schema

```yaml
# /data/config/vllmctl.yaml

server:
  listen_address: "0.0.0.0"       # Bind address for the Go HTTP server
  port: 3000                       # Port for UI + API
  log_level: "info"                # debug | info | warn | error
  base_path: ""                    # URL prefix if reverse-proxied (e.g. "/vllm")

api:
  api_key: ""                      # If set, required as Bearer token for /v1/* endpoints
  external_url: ""                 # Public-facing URL for display (e.g. "https://mybox:3000")

huggingface:
  hf_token: ""                     # HuggingFace API token for gated model access
  default_search_filters:
    pipeline_tag: "text-generation"
    library: "transformers"
    sort: "downloads"
    direction: -1
    limit: 25

vllm_defaults:
  dtype: "auto"                    # auto | float16 | bfloat16 | float32
  gpu_memory_utilization: 0.90     # 0.0-1.0, fraction of GPU VRAM vLLM may use
  max_num_seqs: 256                # Maximum concurrent sequences
  attention_backend: "TRITON_FLASH_ATTN"  # TRITON_FLASH_ATTN | ROCM_FLASH | FLASH_ATTN | XFORMERS
  enforce_eager: false             # Disable CUDA/HIP graphs
  enable_prefix_caching: true      # KV cache prefix sharing
  max_model_len: 0                 # 0 = use model's default from config.json
  extra_flags: []                  # Arbitrary extra CLI flags passed to vllm serve

tool_use:
  enable_auto_tool_choice: false   # --enable-auto-tool-choice flag
  tool_call_parser: ""             # --tool-call-parser value: hermes | mistral | llama3_json | internlm | jamba | pythonic
  # Parser selection guide (stored as comments in default config):
  #   hermes   - Hermes-2-Pro, Hermes-3, NousHermes models
  #   mistral  - Mistral/Mixtral models with tool support
  #   llama3_json - Llama 3.1+ models
  #   internlm - InternLM2+ models
  #   jamba    - Jamba models
  #   pythonic - Models using Python-style function calls

quantization:
  prefer_marlin: true              # Auto-upgrade compatible GPTQ/AWQ to Marlin kernels
  default_kv_cache_dtype: "auto"   # auto | fp8 | fp8_e4m3 | fp8_e5m2
  # Marlin auto-upgrade logic:
  #   If prefer_marlin=true AND model is GPTQ 4-bit with desc_act=false,
  #   use --quantization marlin instead of --quantization gptq.
  #   If prefer_marlin=true AND model is AWQ 4-bit,
  #   use --quantization marlin instead of --quantization awq.
  #   Marlin provides significantly faster inference for compatible models.

process:
  auto_restart: true               # Restart vLLM on unexpected exit
  restart_backoff: "5s"            # Duration string, wait before restart
  startup_timeout: "300s"          # Max time to wait for vLLM /health to respond
  shutdown_timeout: "30s"          # SIGTERM -> SIGKILL grace period
  health_check_interval: "5s"     # How often to poll vLLM /health when running

monitor:
  poll_interval: "3s"              # GPU/CPU metrics sampling interval
  metrics_retention: "1h"          # How long to keep metric history in memory

benchmark:
  default_preset: "standard"       # quick | standard | thorough
  shareGPT_dataset_path: ""        # Path to ShareGPT dataset for vllm bench throughput
  warmup_requests: 1               # Number of warmup requests before measurement
  default_prompt_lengths: [128, 512, 1024, 2048]
  default_output_length: 256

theme:
  selected: "dark"                 # dark | light | terminal-green | terminal-amber | cyberpunk

model_storage:
  model_dir: "/data/models"        # Root directory for downloaded models
  use_hf_cache: false              # If true, use HF_HOME/hub cache structure instead
  hf_home: ""                      # Override HF_HOME (default: /data/hf_cache)
```

### Go Struct Mapping

```
internal/config/config.go

type Config struct {
    Server       ServerConfig       `yaml:"server"`
    API          APIConfig          `yaml:"api"`
    HuggingFace  HFConfig           `yaml:"huggingface"`
    VLLMDefaults VLLMDefaultsConfig `yaml:"vllm_defaults"`
    ToolUse      ToolUseConfig      `yaml:"tool_use"`
    Quantization QuantConfig        `yaml:"quantization"`
    Process      ProcessConfig      `yaml:"process"`
    Monitor      MonitorConfig      `yaml:"monitor"`
    Benchmark    BenchmarkConfig    `yaml:"benchmark"`
    Theme        ThemeConfig        `yaml:"theme"`
    ModelStorage ModelStorageConfig `yaml:"model_storage"`
}
```

Each sub-struct maps directly to the YAML block. Duration fields (`restart_backoff`, `startup_timeout`, etc.) are stored as `time.Duration` with a custom YAML unmarshaller that accepts Go duration strings ("5s", "1m30s", etc.).

---

## 2. Environment Variable Overrides

Every YAML field can be overridden via an environment variable. The naming convention is:

```
VLLMCTL_<SECTION>_<FIELD>
```

All uppercase, dots and hyphens replaced with underscores.

### Mapping Examples

| YAML Path | Environment Variable |
|---|---|
| `server.port` | `VLLMCTL_SERVER_PORT` |
| `server.log_level` | `VLLMCTL_SERVER_LOG_LEVEL` |
| `api.api_key` | `VLLMCTL_API_KEY` |
| `api.external_url` | `VLLMCTL_API_EXTERNAL_URL` |
| `huggingface.hf_token` | `VLLMCTL_HF_TOKEN` (also respects `HF_TOKEN` for compatibility) |
| `vllm_defaults.dtype` | `VLLMCTL_VLLM_DEFAULTS_DTYPE` |
| `vllm_defaults.gpu_memory_utilization` | `VLLMCTL_VLLM_DEFAULTS_GPU_MEMORY_UTILIZATION` |
| `vllm_defaults.attention_backend` | `VLLMCTL_VLLM_DEFAULTS_ATTENTION_BACKEND` |
| `tool_use.enable_auto_tool_choice` | `VLLMCTL_TOOL_USE_ENABLE_AUTO_TOOL_CHOICE` |
| `tool_use.tool_call_parser` | `VLLMCTL_TOOL_USE_TOOL_CALL_PARSER` |
| `quantization.prefer_marlin` | `VLLMCTL_QUANTIZATION_PREFER_MARLIN` |
| `quantization.default_kv_cache_dtype` | `VLLMCTL_QUANTIZATION_DEFAULT_KV_CACHE_DTYPE` |
| `process.auto_restart` | `VLLMCTL_PROCESS_AUTO_RESTART` |
| `model_storage.model_dir` | `VLLMCTL_MODEL_STORAGE_MODEL_DIR` |
| `theme.selected` | `VLLMCTL_THEME_SELECTED` |

### Implementation

In `internal/config/config.go`:

1. Define a `fieldMapping` table that maps struct field paths to env var names.
2. After loading YAML, iterate over the mapping table and apply any set env vars.
3. Type coercion: string env vars are converted to the target Go type (int, float64, bool, time.Duration, []string).
4. For slice fields (like `extra_flags`, `default_prompt_lengths`), env vars accept comma-separated values.
5. Special case: `HF_TOKEN` (no prefix) is also checked as a fallback for `VLLMCTL_HF_TOKEN` since HuggingFace libraries natively use `HF_TOKEN`.

---

## 3. Config Loading Order

The loading order determines precedence (later wins):

```
1. Compiled defaults (hardcoded in Go struct tags / DefaultConfig() function)
2. /data/config/vllmctl.yaml (if exists)
3. Environment variables (VLLMCTL_* prefix)
```

### Loading Logic (`config.Load()`)

```
func Load(configPath string) (*Config, error):
    1. cfg := DefaultConfig()                      // Hardcoded defaults
    2. if fileExists(configPath):
           yamlBytes := readFile(configPath)
           yaml.Unmarshal(yamlBytes, &cfg)         // YAML overlays defaults
    3. applyEnvOverrides(&cfg)                      // Env vars overlay everything
    4. validate(&cfg)                               // Reject invalid combinations
    5. return &cfg, nil
```

### Validation Rules

- `server.port`: 1-65535
- `vllm_defaults.gpu_memory_utilization`: 0.01-1.0
- `vllm_defaults.max_num_seqs`: 1-4096
- `vllm_defaults.dtype`: must be one of [auto, float16, bfloat16, float32]
- `vllm_defaults.attention_backend`: must be one of known backends
- `tool_use.tool_call_parser`: if set, must be one of [hermes, mistral, llama3_json, internlm, jamba, pythonic]
- `tool_use`: if `tool_call_parser` is set, `enable_auto_tool_choice` must be true (auto-enable it with a warning log)
- `quantization.default_kv_cache_dtype`: must be one of [auto, fp8, fp8_e4m3, fp8_e5m2]
- `process.startup_timeout`: minimum 30s (vLLM model loading can be slow)
- `process.shutdown_timeout`: minimum 5s
- `theme.selected`: must be one of known theme names
- `model_storage.model_dir`: path must exist or be creatable

### Config Persistence

When settings are updated via the API:

1. Merge incoming changes into the current in-memory config.
2. Validate the merged config.
3. Write the entire config to `/data/config/vllmctl.yaml` (atomic write: write to `.tmp`, then rename).
4. If writing fails, return error without updating in-memory config.
5. The config object is protected by a `sync.RWMutex` -- reads acquire RLock, writes acquire Lock.

---

## 4. Settings API

All settings endpoints live in `internal/api/settings.go`.

### `GET /api/settings`

Returns the current configuration with sensitive values sanitized.

**Response (JSON mode):**
```json
{
  "server": {
    "listen_address": "0.0.0.0",
    "port": 3000,
    "log_level": "info",
    "base_path": ""
  },
  "api": {
    "api_key": "****",
    "api_key_set": true,
    "external_url": "https://mybox:3000"
  },
  "huggingface": {
    "hf_token": "****",
    "hf_token_set": true,
    "default_search_filters": { ... }
  },
  "vllm_defaults": { ... },
  "tool_use": {
    "enable_auto_tool_choice": true,
    "tool_call_parser": "hermes"
  },
  "quantization": {
    "prefer_marlin": true,
    "default_kv_cache_dtype": "auto"
  },
  "process": { ... },
  "monitor": { ... },
  "benchmark": { ... },
  "theme": { "selected": "dark" },
  "model_storage": { ... }
}
```

**Sanitization rules:**
- `api.api_key` -> masked as `"****"`, separate `api_key_set` boolean field
- `huggingface.hf_token` -> masked as `"****"`, separate `hf_token_set` boolean field
- All other fields returned as-is

**HTML mode (HX-Request header present):** Returns the `settings.html` template rendered with the config struct.

### `PUT /api/settings`

Accepts a partial or full config update. Only provided fields are updated (JSON merge semantics).

**Request body:**
```json
{
  "vllm_defaults": {
    "gpu_memory_utilization": 0.95
  },
  "tool_use": {
    "enable_auto_tool_choice": true,
    "tool_call_parser": "hermes"
  }
}
```

**Behavior:**
1. Parse request body into a sparse config struct (using pointers for optional fields or a map-based approach).
2. Deep-merge into current config.
3. Validate the merged result.
4. If valid: save to YAML, update in-memory config, return 200 with updated config.
5. If invalid: return 400 with validation errors.
6. If vLLM is currently running and the changed settings affect it (e.g. `vllm_defaults`, `tool_use`), include a warning in the response: `"restart_required": true`.

**Validation error response:**
```json
{
  "errors": {
    "vllm_defaults.gpu_memory_utilization": "must be between 0.01 and 1.0",
    "tool_use.tool_call_parser": "invalid parser, must be one of: hermes, mistral, llama3_json, internlm, jamba, pythonic"
  }
}
```

### `POST /api/settings/test-connection`

Tests connectivity to the vLLM process.

**Behavior:**
1. Send GET to `http://localhost:8000/health` (the vLLM health endpoint).
2. If 200: return `{"status": "ok", "vllm_version": "...", "model": "..."}`.
3. If connection refused: return `{"status": "not_running", "message": "vLLM process is not running"}`.
4. If timeout: return `{"status": "timeout", "message": "vLLM did not respond within 5s"}`.
5. If error: return `{"status": "error", "message": "<error details>"}`.

Also test the OpenAI-compatible endpoints:
- `GET http://localhost:8000/v1/models` to verify model is loaded.

### `POST /api/settings/test-hf-token`

Validates the HuggingFace token.

**Request body:**
```json
{
  "token": "hf_xxxxxxxxxxxx"
}
```
If `token` is empty, use the currently configured token.

**Behavior:**
1. Call `GET https://huggingface.co/api/whoami-v2` with `Authorization: Bearer <token>`.
2. If 200: return `{"status": "valid", "username": "...", "can_access_gated": true}`.
3. If 401: return `{"status": "invalid", "message": "Token is invalid or expired"}`.
4. Check specific permissions by also testing access to a known gated repo (e.g., `meta-llama/Llama-3.1-8B`).

### `GET /api/settings/gpu-info`

Returns detected GPU information.

**Response:**
```json
{
  "gpus": [
    {
      "index": 0,
      "name": "AMD Radeon RX 9700 XT",
      "gfx_version": "gfx1201",
      "vram_total_mb": 32768,
      "vram_used_mb": 1024,
      "temperature_c": 45,
      "driver_version": "6.8.0",
      "pci_bus": "0000:03:00.0"
    }
  ],
  "gpu_count": 1,
  "vendor": "amd",
  "multi_gpu": false
}
```

**Implementation:**
- AMD: Parse `rocm-smi --showid --showproductname --showmeminfo vram --showtemp --showbus` output.
- NVIDIA: Parse `nvidia-smi --query-gpu=...` (Phase 9).
- Fallback: Parse `/sys/class/drm/card*/device/vendor` and `lspci` output.

### `GET /api/settings/rocm-info`

Returns ROCm-specific installation details. AMD only -- returns 404 on non-ROCm systems.

**Response:**
```json
{
  "rocm_version": "6.4.0",
  "rocm_path": "/opt/rocm",
  "hip_version": "6.4.0",
  "hsa_override_gfx_version": "12.0.1",
  "kfd_available": true,
  "dri_render_nodes": ["/dev/dri/renderD128"],
  "gpu_topology": {
    "0": { "gfx_version": "gfx1201", "numa_node": 0 }
  },
  "therock_build": true,
  "torch_hip_arch": "gfx1201",
  "flash_attention": "ROCm/flash-attention main_perf",
  "triton_available": true
}
```

**Implementation:**
- `rocm-smi --showversion` for ROCm version.
- Check `/opt/rocm/include/rocm-core/rocm_version.h` for version parsing.
- `ls /dev/kfd` and `ls /dev/dri/renderD*` for device nodes.
- Environment variables: `HSA_OVERRIDE_GFX_VERSION`, `HIP_VISIBLE_DEVICES`, `PYTORCH_ROCM_ARCH`.
- `python3 -c "import torch; print(torch.version.hip)"` for PyTorch HIP version.

---

## 5. Settings Page UI

File: `web/templates/settings.html`

The settings page is organized into tabbed or accordion sections. Each section is an independent htmx form that can be saved individually. All sections also support a global "Save All" action.

### Section Layout

#### 5.1 General Section

- **Listen Address** -- text input, default "0.0.0.0" (usually not changed in container mode)
- **Port** -- number input, default 3000
- **Log Level** -- select dropdown: debug, info, warn, error
- **Base Path** -- text input, for reverse proxy setups

Display note: "Changes to listen address, port, or base path require a container restart."

#### 5.2 API Section

- **API Key** -- password input with show/hide toggle, generate random button
  - Generate button calls `crypto/rand` to produce a 32-char alphanumeric key
  - Show current masked value, "Change" button to reveal edit field
- **External URL** -- text input with auto-detect button
  - Auto-detect: use request's `Host` header + scheme

#### 5.3 HuggingFace Section

- **HF Token** -- password input with show/hide toggle
- **Test Token** button
  - htmx: `hx-post="/api/settings/test-hf-token"` `hx-target="#hf-token-status"`
  - Status indicator: green checkmark (valid), red X (invalid), spinner (testing)
  - On success, show username and access level
- **Default Search Filters** -- read-only display or advanced toggle to edit
  - Pipeline tag, library, sort order, result limit

#### 5.4 vLLM Defaults Section

- **Default dtype** -- select: auto, float16, bfloat16, float32
  - Help text: "auto selects based on model config. bfloat16 recommended for RDNA4."
- **GPU Memory Utilization** -- range slider (0.50 - 1.00, step 0.01) with numeric display
  - Help text: "Fraction of GPU VRAM vLLM may use. Lower values leave room for KV cache growth."
- **Max Concurrent Sequences** -- number input, default 256
- **Attention Backend** -- select: TRITON_FLASH_ATTN, ROCM_FLASH, FLASH_ATTN, XFORMERS
  - Help text explains which backends work on ROCm vs CUDA
  - On ROCm: highlight TRITON_FLASH_ATTN as recommended
- **Enforce Eager Mode** -- checkbox
  - Help text: "Disables HIP/CUDA graphs. Slower but uses less memory. Try this if you get OOM errors."
- **Enable Prefix Caching** -- checkbox
  - Help text: "Shares KV cache across requests with matching prefixes. Reduces TTFT for repeated system prompts."
- **Default Max Model Length** -- number input (0 = use model default)
- **Extra Flags** -- textarea, one flag per line
  - Help text: "Additional flags passed to vllm serve. One per line. Example: --disable-log-requests"

Display a warning banner if vLLM is currently running: "vLLM is running. Changes to vLLM defaults will take effect on next restart."

#### 5.5 Tool Use Section

- **Enable Auto Tool Choice** -- checkbox
  - Help text: "Pass --enable-auto-tool-choice to vLLM. Required for OpenAI-compatible function calling."
- **Tool Call Parser** -- select: (none), hermes, mistral, llama3_json, internlm, jamba, pythonic
  - Disabled/grayed out when enable_auto_tool_choice is false
  - Help text for each parser option explaining which models it works with
  - When user changes the active model, if the model's tokenizer_config.json chat_template contains tool-related tokens, auto-suggest the appropriate parser

**Tool Use Compatibility Info Panel:**
- Display a table of known model families and their recommended parser
- Indicate whether the currently loaded model supports tool use based on its chat_template
- Link to vLLM tool calling documentation

#### 5.6 Quantization Section

- **Prefer Marlin Kernels** -- checkbox
  - Help text: "When enabled, automatically use Marlin kernels for compatible GPTQ (4-bit, no desc_act) and AWQ (4-bit) models. Marlin provides 3-4x faster inference for these models."
  - Show compatibility note: "Marlin requires: 4-bit quantization, no activation ordering (desc_act=false), groupsize 128 or -1."
- **Default KV Cache Dtype** -- select: auto, fp8, fp8_e4m3, fp8_e5m2
  - Help text: "fp8 KV cache reduces memory usage by ~50% with minimal quality loss. Recommended for large context lengths."
  - Note for RDNA4: "FP8 KV cache support on RDNA4 depends on ROCm/vLLM version. Verify with a test inference."

**Quantization Reference Panel** (read-only informational):
- Table of supported quantization formats:

| Format | Bits | Library | Notes |
|--------|------|---------|-------|
| AWQ | 4 | AutoAWQ / Marlin | Fast on both ROCm and CUDA. Marlin upgrade available. |
| GPTQ | 2,3,4,8 | AutoGPTQ / Marlin | Widely available. 4-bit can use Marlin. |
| FP8 | 8 | native vLLM | Near-FP16 quality, 50% size reduction. |
| GGUF | varies | vLLM GGUF loader | Limited support in vLLM. Not recommended -- use AWQ/GPTQ instead. |
| BitsAndBytes | 4,8 | bitsandbytes | On-the-fly quantization. ROCm requires ROCm fork of bitsandbytes. |
| Marlin | 4 | Marlin kernels | Not a format -- optimized kernels for compatible GPTQ/AWQ models. |
| SqueezeLLM | 4 | SqueezeLLM | Experimental. Limited model availability. |
| compressed-tensors | varies | nm-vllm | Neural Magic sparse+quantized models. |

#### 5.7 Process Section

- **Auto Restart** -- checkbox
  - Help text: "Automatically restart vLLM if it crashes unexpectedly."
- **Restart Backoff** -- duration input (text, validated as Go duration), default "5s"
- **Startup Timeout** -- duration input, default "300s"
  - Help text: "Maximum time to wait for vLLM to load a model and become healthy. Large models may need 5+ minutes."
- **Shutdown Timeout** -- duration input, default "30s"
- **Health Check Interval** -- duration input, default "5s"

#### 5.8 Theme Section

- **Theme Selector** -- visual cards showing preview swatches for each theme
  - dark: Dark background, light text (default)
  - light: White background, dark text
  - terminal-green: Black background, green text (monospace feel)
  - terminal-amber: Black background, amber/orange text
  - cyberpunk: Dark purple background, neon accents

**Implementation:**
- Each theme is a CSS file (`web/static/themes/<name>.css`) that overrides Pico CSS custom properties.
- Theme selection is applied immediately via htmx swap of the `<link>` tag (no page reload).
- `hx-get="/api/settings/theme-preview?name=cyberpunk"` returns a `<link>` tag swap.
- Selected theme is persisted and loaded in `layout.html` from config.

#### 5.9 GPU / ROCm Info Panel

Read-only informational panel at the bottom of settings page.

- **GPU Info:** Name, VRAM, driver version, PCI bus, temperature (auto-refreshes every 10s)
  - `hx-get="/api/settings/gpu-info"` `hx-trigger="load, every 10s"`
- **ROCm Info:** Version, path, HIP version, device topology
  - `hx-get="/api/settings/rocm-info"` `hx-trigger="load"`
- **Connection Test:**
  - Button: "Test vLLM Connection"
  - `hx-post="/api/settings/test-connection"` `hx-target="#connection-status"`
  - Shows: connected (green) / disconnected (red) / testing (spinner)
  - On success: display vLLM version and loaded model name

#### 5.10 Danger Zone

Visually separated section at the very bottom with a red border (Pico CSS `contrast` styling).

- **Clear All Model Configs** -- Button to reset all per-model vLLM configurations in `models.json` to defaults. Does NOT delete model files.
  - Confirmation modal: "This will reset all per-model configurations (dtype, context length, TP, etc.) to defaults. Model files will not be deleted. Continue?"
  - `hx-delete="/api/settings/model-configs"` `hx-confirm="..."`

- **Reset Settings to Defaults** -- Button to reset `vllmctl.yaml` to default values.
  - Confirmation modal: "This will reset all settings to defaults. Your API key and HF token will be cleared. Continue?"
  - `hx-post="/api/settings/reset"` `hx-confirm="..."`

- **Delete All Benchmark Data** -- Button to clear `benchmarks.json`.
  - `hx-delete="/api/settings/benchmark-data"` `hx-confirm="..."`

### Inline Validation and Save Feedback

Each section form uses htmx for submission:

```html
<form hx-put="/api/settings"
      hx-target="#vllm-defaults-feedback"
      hx-swap="innerHTML"
      hx-indicator="#vllm-defaults-spinner">
  <!-- fields -->
  <button type="submit">Save vLLM Defaults</button>
  <span id="vllm-defaults-spinner" class="htmx-indicator">Saving...</span>
  <div id="vllm-defaults-feedback"></div>
</form>
```

**Success response (HTML partial):**
```html
<div class="notice success" role="alert">Settings saved successfully.</div>
```

**Validation error response (HTML partial):**
```html
<div class="notice error" role="alert">
  <ul>
    <li>GPU Memory Utilization: must be between 0.01 and 1.0</li>
  </ul>
</div>
```

**Client-side validation (before submit):**
- Range inputs enforce min/max via HTML attributes.
- Duration fields: regex pattern `^\d+[smh]$` on input.
- The tool_call_parser select is disabled when enable_auto_tool_choice is unchecked (JavaScript toggle).
- GPU memory utilization slider updates a live VRAM estimate display.

---

## 6. What to Copy from llama-toolchest vs Adapt

### Direct Copy (minimal changes)

| Component | llama-toolchest Source | Changes Needed |
|---|---|---|
| Config loading skeleton | `internal/config/config.go` | New struct fields, same loading pattern |
| Settings page HTML structure | `web/templates/settings.html` | New sections, same form/htmx patterns |
| Theme system | `web/static/themes/` | Copy all theme CSS files as-is |
| Theme selector UI | Settings page theme section | Same card-based selector |
| API key generation | Settings page JS | Copy the random key generator |
| respond.go dual-mode helper | `internal/api/respond.go` | Copy as-is |
| middleware.go API key auth | `internal/api/middleware.go` | Copy as-is, same Bearer token check |

### Adapt (same pattern, different content)

| Component | What Changes |
|---|---|
| Config struct | Replace llama.cpp fields (gpu_layers, flash_attention, mmap) with vLLM fields (dtype, gpu_memory_utilization, tensor_parallel_size, attention_backend) |
| Config defaults | Different default values for vLLM-specific fields |
| Connection test | Test vLLM on :8000/health instead of llama-server on :8080/health |
| GPU info endpoint | Same rocm-smi parsing, add vLLM-specific VRAM info |
| HF token test | Same endpoint, same logic -- direct copy |
| Env var override logic | Same pattern, different variable names |
| Validation rules | Different field constraints |

### New (not in llama-toolchest)

| Component | Why New |
|---|---|
| Tool Use settings section | llama-toolchest doesn't have tool calling support |
| Quantization settings section | llama-toolchest uses GGUF quant types, not AWQ/GPTQ/Marlin |
| ROCm info endpoint | New endpoint for detailed ROCm installation info |
| Marlin auto-upgrade logic | Quantization-specific optimization not applicable to llama.cpp |
| Tool call parser auto-detection | Inspect model's chat_template for tool tokens |

---

## 7. Edge Cases and Error Handling

### Config File Corruption
- If `vllmctl.yaml` fails to parse: log error, fall back to defaults, display warning banner on settings page ("Config file could not be parsed. Using defaults. Save settings to fix.").
- Keep a backup: before writing, copy current file to `vllmctl.yaml.bak`.

### Environment Variable Type Mismatches
- If an env var cannot be parsed to the target type (e.g., `VLLMCTL_SERVER_PORT=abc`): log a warning, skip the override, use the YAML/default value.

### Concurrent Config Access
- The config is accessed from multiple goroutines (HTTP handlers, process manager, monitor).
- Use `sync.RWMutex`: readers (GET /api/settings, process start) acquire RLock; writers (PUT /api/settings) acquire full Lock.
- Config updates are atomic from the perspective of readers.

### Sensitive Value Handling
- API key and HF token are never logged.
- In debug log mode, config dump redacts these fields.
- PUT /api/settings accepts empty string to clear a sensitive field, or the literal `"****"` (the masked value) to mean "keep current value unchanged."

### Model Dir Permissions
- On startup, verify `model_dir` is writable. If not, log error and display warning on dashboard.
- If `model_dir` is changed via settings, verify the new path before saving.

### Tool Use Config Consistency
- If `tool_call_parser` is set but `enable_auto_tool_choice` is false: auto-enable it and return a warning in the response.
- If `enable_auto_tool_choice` is true but `tool_call_parser` is empty: return a warning that tool calling will not work without a parser.

### Theme Missing
- If `theme.selected` references a theme file that doesn't exist on disk: fall back to "dark" and log a warning.

### First-Run Experience
- On first boot (no config file exists): write defaults, display a "Welcome" banner on the dashboard suggesting the user visit Settings to configure their API key and HF token.
- The settings page highlights unconfigured fields (HF token, API key) with an info badge.

# Phase 5: vLLM Process Management & Service Control

Start, stop, monitor, and proxy the vLLM inference server. Build the full command line from per-model config, manage process lifecycle, capture logs, handle model switching, and proxy the OpenAI-compatible API through the Go management server.

---

## Process Manager (`internal/process/manager.go`)

### Core Responsibilities

The process manager owns the vLLM child process. It is a singleton -- one manager instance, one vLLM process at a time. The manager struct holds:

```
ProcessManager struct:
  mu              sync.RWMutex
  cmd             *exec.Cmd
  state           ProcessState       // Stopped, Starting, Running, Stopping, Error
  modelID         string             // Currently loaded model ID (registry key)
  modelConfig     ModelConfigSnapshot // Frozen copy of config at start time
  pid             int
  startTime       time.Time
  logBuffer       *RingBuffer        // Recent log lines (stdout + stderr merged)
  logSubscribers  *SSEFanOut         // Live log SSE subscribers
  healthTicker    *time.Ticker
  healthStatus    HealthStatus       // Healthy, Unhealthy, Unknown
  lastHealthCheck time.Time
  restartCount    int
  lastError       string
  cancelFunc      context.CancelFunc // For killing the process
  vllmPort        int                // Default 8000
  readyCh         chan struct{}       // Closed when vLLM reports ready
}
```

### ProcessState Enum

```
Stopped   -- No vLLM process running
Starting  -- Process spawned, waiting for ready signal
Running   -- Health check passing, serving requests
Stopping  -- SIGTERM sent, waiting for exit
Error     -- Process exited unexpectedly or health check failing
Switching -- Stop + start in progress (model change)
```

### Command Construction

Build the full `vllm serve` command from the model's `vllm_config` and registry metadata.

**Base command:**
```
vllm serve <model_path> --host 0.0.0.0 --port 8000
```

**Flag mapping from vllm_config:**

| Config Field | CLI Flag | Condition |
|---|---|---|
| `dtype` | `--dtype <value>` | Always (default "auto") |
| `max_model_len` | `--max-model-len <value>` | When > 0 |
| `tensor_parallel_size` | `--tensor-parallel-size <value>` | Always (default 1) |
| `gpu_memory_utilization` | `--gpu-memory-utilization <value>` | Always (default 0.90) |
| `enforce_eager` | `--enforce-eager` | When true (no value, just flag) |
| `trust_remote_code` | `--trust-remote-code` | When true |
| `max_num_seqs` | `--max-num-seqs <value>` | Always (default 16) |
| `quantization` | `--quantization <value>` | When non-empty |
| `load_format` | `--load-format <value>` | When non-empty and != "auto" |
| `enable_prefix_caching` | `--enable-prefix-caching` | When true |
| `kv_cache_dtype` | `--kv-cache-dtype <value>` | When non-empty and != "auto" |
| `enable_chunked_prefill` | `--enable-chunked-prefill` | When true |
| `max_num_batched_tokens` | `--max-num-batched-tokens <value>` | When > 0 |
| `tokenizer` | `--tokenizer <value>` | When non-empty |
| `chat_template` | `--chat-template <value>` | When non-empty |
| `enable_auto_tool_choice` | `--enable-auto-tool-choice` | When true |
| `tool_call_parser` | `--tool-call-parser <value>` | When non-empty |
| `extra_flags` | (appended raw) | When non-empty |

**Quantization-specific flag logic:**

| Quant Method | Flags | Notes |
|---|---|---|
| awq | `--quantization awq` | Standard AWQ inference |
| gptq | `--quantization gptq` | Standard GPTQ inference |
| fp8 | `--quantization fp8` | May also set `--kv-cache-dtype fp8` |
| gguf | `--quantization gguf --load-format gguf` | Model path points to .gguf file(s) |
| bitsandbytes | `--quantization bitsandbytes --enforce-eager` | Always enforce eager with bnb |
| marlin | `--quantization marlin` | Optimized GPTQ/AWQ kernel |
| squeezellm | `--quantization squeezellm` | |
| compressed_tensors | `--quantization compressed_tensors` | |
| none/"" | (no --quantization flag) | |

**GGUF model path resolution:**
- If model directory contains a single `.gguf` file: use that file as the model path
- If model directory contains multiple `.gguf` files (split model): use the directory path
- If model directory contains `.gguf` and `config.json`: vLLM may need the directory (it reads config.json too)

**Tool use flag logic:**
```
if config.enable_auto_tool_choice && config.tool_call_parser != "":
    append("--enable-auto-tool-choice")
    append("--tool-call-parser", config.tool_call_parser)
elif config.enable_auto_tool_choice && config.tool_call_parser == "":
    log warning: "enable_auto_tool_choice is set but no tool_call_parser specified"
    // Still pass the flag -- vLLM will warn but won't crash
    append("--enable-auto-tool-choice")
```

**Full command example:**
```bash
vllm serve /data/models/NousResearch/Hermes-3-Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --disable-log-requests
```

### Environment Variable Injection

Set these environment variables on the child process:

**Always set:**
```
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
```

**ROCm-specific (always set on AMD GPU hosts):**
```
HSA_OVERRIDE_GFX_VERSION=12.0.1          # Force gfx1201 identity
PYTORCH_ROCM_ARCH=gfx1201                # Target architecture
HIP_VISIBLE_DEVICES=0                     # Or "0,1" for TP=2
GPU_MAX_HW_QUEUES=2                       # Optimal for RDNA4
VLLM_USE_TRITON_FLASH_ATTN=1             # Triton flash attention (often needed for ROCm)
TOKENIZERS_PARALLELISM=false              # Avoid deadlocks in tokenizer
```

**Conditional:**
```
# AWQ with Triton kernel (if Marlin not available on ROCm)
VLLM_USE_TRITON_AWQ=1                    # When quant == awq and ROCm

# tcmalloc preload (fixes double-free crashes in some vLLM versions)
LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so  # If tcmalloc installed

# Attention backend override (from global settings)
VLLM_ATTENTION_BACKEND=FLASH_ATTN        # or TRITON_FLASH_ATTN, ROCM_FLASH, XFORMERS

# For BitsAndBytes on ROCm
BNB_CUDA_VERSION=                          # Unset to force ROCm path

# Disable custom all-reduce for TP (if buggy on RDNA4)
VLLM_DISABLED_CUSTOM_ALL_REDUCE=1         # Conditional, based on known issues
```

**TP=2 GPU assignment:**
```
# Tensor parallelism with 2 GPUs
HIP_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,1                  # Set both for compatibility
# vLLM handles the TP sharding internally via --tensor-parallel-size 2
# Ray is used under the hood for multi-GPU; may need:
RAY_EXPERIMENTAL_NOSET_ROCM_VISIBLE_DEVICES=1  # Prevent Ray from overriding device selection
```

### Process Lifecycle

#### Spawn

```
func (pm *ProcessManager) Start(modelID string) error:
  1. Acquire write lock
  2. If state is Running or Starting, return error "already running"
  3. Load model config from registry
  4. Validate config (check model path exists, check GPU count for TP, etc.)
  5. Build command + args
  6. Build environment
  7. Set state = Starting
  8. Create context with cancel
  9. cmd = exec.CommandContext(ctx, "vllm", args...)
  10. Set cmd.Env, cmd.Dir
  11. Create pipes for stdout, stderr
  12. cmd.Start()
  13. Store PID, start time
  14. Release write lock
  15. Launch goroutine: monitorOutput(stdout, stderr)
  16. Launch goroutine: waitForReady()
  17. Launch goroutine: cmd.Wait() -> handleExit()
```

#### Monitor Output

```
func (pm *ProcessManager) monitorOutput(stdout, stderr io.Reader):
  // Merge stdout and stderr into a single stream
  // Use bufio.Scanner for line-by-line processing
  For each line:
    1. Append to ring buffer (with timestamp, source tag stdout/stderr)
    2. Broadcast to SSE subscribers
    3. Check for ready indicator (see below)
    4. Check for error patterns (OOM, CUDA error, etc.)
```

#### Detect Ready State

vLLM prints specific messages when it is ready to serve:

**Primary ready indicators (check stdout):**
- `"Uvicorn running on http://"`
- `"Application startup complete"`

**Secondary confirmation:**
- Poll `GET http://localhost:8000/health` -- returns 200 when ready

**Ready detection flow:**
```
func (pm *ProcessManager) waitForReady():
  timeout := 10 * time.Minute  // Large models take several minutes to load
  ticker := time.NewTicker(2 * time.Second)

  select {
  case <-pm.readyCh:   // Set by stdout monitor when ready line detected
    // Confirm with health check
    if healthCheck() == 200:
      pm.setState(Running)
    else:
      // Ready line seen but health check fails -- keep polling
      pollHealthUntilReady(30 * time.Second)

  case <-time.After(timeout):
    pm.setState(Error)
    pm.lastError = "Startup timeout after 10 minutes"
    pm.Stop()  // Kill the stuck process
  }
```

**Startup progress parsing (informational, broadcast via SSE):**
- `"Loading model weights"` -> progress update
- `"Loading safetensors"` with shard numbers -> per-shard progress
- `"Using <X> attention"` -> log attention backend
- `"torch.compile"` or `"Compiling"` -> graph compilation phase (can be slow)
- `"CUDA graphs"` or `"CUDAGraph"` -> graph capture phase
- Memory usage lines -> extract and broadcast

#### Graceful Shutdown

```
func (pm *ProcessManager) Stop() error:
  1. Acquire write lock
  2. If state is Stopped, return nil
  3. Set state = Stopping
  4. Send SIGTERM to process (cmd.Process.Signal(syscall.SIGTERM))
  5. Release write lock
  6. Wait up to 30 seconds for process to exit
  7. If still running after 30s: send SIGKILL
  8. Wait for cmd.Wait() goroutine to complete
  9. Set state = Stopped
  10. Clear PID, health status
  11. Broadcast state change to SSE subscribers
```

**Why 30s for graceful shutdown:** vLLM needs time to finish in-flight requests and clean up GPU memory. Shorter timeouts risk leaving GPU memory allocated (zombie VRAM).

#### Handle Exit

```
func (pm *ProcessManager) handleExit(err error):
  exitCode := cmd.ProcessState.ExitCode()

  if pm.state == Stopping:
    // Expected shutdown
    pm.setState(Stopped)
    return

  // Unexpected exit
  pm.setState(Error)
  pm.lastError = fmt.Sprintf("vLLM exited with code %d: %v", exitCode, err)

  // Check for known error patterns in recent logs
  recentLogs := pm.logBuffer.Last(50)
  if containsOOM(recentLogs):
    pm.lastError += " (GPU out of memory -- reduce max-model-len or gpu-memory-utilization)"
  elif containsGPUError(recentLogs):
    pm.lastError += " (GPU error -- try --enforce-eager or clear vLLM cache)"

  // Auto-restart logic
  if pm.config.AutoRestart && pm.restartCount < pm.config.MaxRestarts:
    backoff := time.Duration(math.Pow(2, float64(pm.restartCount))) * time.Second
    if backoff > 60*time.Second:
      backoff = 60 * time.Second
    time.Sleep(backoff)
    pm.restartCount++
    pm.Start(pm.modelID)
```

### Log Capture

**Ring buffer (`internal/process/ringbuffer.go`):**
```
RingBuffer struct:
  lines    []LogLine
  capacity int       // Default 10000 lines
  head     int
  mu       sync.RWMutex

LogLine struct:
  Timestamp time.Time
  Source    string    // "stdout" | "stderr"
  Text      string
  Level     string    // "info" | "warn" | "error" (parsed from vLLM log format)
}
```

vLLM uses Python's logging module. Log lines typically look like:
```
INFO 01-15 10:30:00 model_runner.py:123] Loading model weights...
WARNING 01-15 10:30:01 config.py:456] Some deprecation warning
ERROR 01-15 10:30:02 worker.py:789] CUDA error: out of memory
```

Parse the level prefix (INFO/WARNING/ERROR) and store as structured `Level` field.

**Log fan-out to SSE subscribers:**
- Each new LogLine is broadcast to all SSE subscribers via the SSEFanOut helper (copied from llama-toolchest's `sse.go`).
- Subscribers connect via `GET /api/service/log-stream`.
- On disconnect, subscriber is removed from fan-out list.

### Health Check Polling

```
func (pm *ProcessManager) healthCheckLoop():
  ticker := time.NewTicker(10 * time.Second)
  for range ticker.C:
    if pm.state != Running:
      continue

    resp, err := http.Get(fmt.Sprintf("http://localhost:%d/health", pm.vllmPort))
    if err != nil || resp.StatusCode != 200:
      pm.healthStatus = Unhealthy
      pm.consecutiveFailures++
      if pm.consecutiveFailures >= 3:
        pm.setState(Error)
        pm.lastError = "Health check failed 3 consecutive times"
        // Don't auto-stop -- the process might recover
    else:
      pm.healthStatus = Healthy
      pm.consecutiveFailures = 0
```

### Startup Timeout Handling

Large models (70B+) can take 3-10 minutes to load, depending on:
- Model size (safetensors loading from disk)
- Quantization (dequantization during load for bnb)
- Graph compilation (if enforce_eager is false)
- TP=2 (Ray initialization + weight distribution)

**Timeout tiers:**
- Models < 3B params: 2 minute timeout
- Models 3B-13B: 4 minute timeout
- Models 13B-34B: 6 minute timeout
- Models 34B-72B: 8 minute timeout
- Models > 72B: 12 minute timeout
- TP=2 adds 2 minutes to any tier
- `enforce_eager: false` adds 2 minutes (graph compilation)

Calculate from `vram_estimate.param_count_billion`. Store as computed field, but allow override via global settings.

### Auto-Restart on Crash

Configurable via global settings:
```yaml
auto_restart:
  enabled: true
  max_restarts: 3           # Max restarts before giving up
  reset_window: 300         # Reset restart counter after this many seconds of stable running
  backoff_base: 2           # Exponential backoff: 2^n seconds
  backoff_max: 60           # Cap backoff at 60 seconds
```

Reset `restartCount` to 0 if the process has been Running for longer than `reset_window` seconds.

---

## Model Switching Strategy

### Stop-Then-Start Flow

```
func (pm *ProcessManager) SwitchModel(newModelID string) error:
  1. Set state = Switching
  2. Broadcast state change via SSE
  3. Stop current process (full graceful shutdown)
  4. Wait for process to fully exit
  5. Optional: explicit GPU memory cleanup (see below)
  6. Start new model
  7. Return when new model is Running or Error
```

### GPU Memory Cleanup

After stopping vLLM, GPU memory should be freed. However, if the Python process exits cleanly, PyTorch/ROCm should release all GPU memory. Edge cases:

- **Zombie VRAM:** Sometimes GPU memory is not fully released after process exit, especially if SIGKILL was used. Detect by checking `rocm-smi --showmeminfo vram` before starting new model. If previous model's memory is still allocated, wait up to 10 seconds for it to clear.
- **HIP context cleanup:** On ROCm, `hipDeviceReset()` is called on process exit. If it hangs, the SIGKILL fallback handles it, but VRAM may leak. Monitor and document as a known issue.
- **Clear vLLM cache:** The `~/.cache/vllm/` directory stores compiled graphs. If switching between models that use different dtypes or architectures, stale graphs can cause issues. Offer a "clear cache on switch" option.

### UI During Switching

The server page shows a "Switching" state with:
- "Stopping <current model>..." with spinner
- "Starting <new model>..." with spinner
- Progress messages from stdout as the new model loads
- Cancel button (which stops the new model load and returns to Stopped state)

---

## Multi-Model Future Considerations

vLLM does not have llama-server's "router mode" where multiple models can be loaded in a single process. Options for future multi-model support:

### Option A: Multiple vLLM Processes (Different Ports)

- Run separate `vllm serve` processes on ports 8000, 8001, etc.
- Each process loads one model.
- GPU assignment via `HIP_VISIBLE_DEVICES`: process 1 on GPU 0, process 2 on GPU 1.
- Proxy routes requests by model name to the correct port.
- **Pros:** True concurrent serving, isolation between models.
- **Cons:** Complex resource management, can't do TP=2 if each GPU has its own model.

### Option B: vLLM's `--served-model-name`

- Single vLLM process loads one model but responds to requests for any `--served-model-name` alias.
- Doesn't actually load multiple models -- just aliases.
- Useful for: making the same physical model respond under multiple names (e.g. "gpt-4" alias).
- Not useful for actual multi-model.

### Option C: LoRA Adapters

- vLLM supports LoRA adapter hot-loading via `--lora-modules`.
- Single base model with multiple LoRA adapters, each exposed as a different model name.
- Adapters are loaded/unloaded dynamically.
- **Future phase consideration.**

### Current Design Decision

For Phase 5: **single model at a time.** Design the `ProcessManager` interface to allow future extension to multiple processes:

```
// Current: singleton
GetManager() *ProcessManager

// Future: keyed by model or port
GetManager(modelID string) *ProcessManager
ListManagers() []*ProcessManager
```

The `ServiceStatus` response already includes `model_id`, so clients can track which model is loaded. The proxy's `/v1/models` endpoint returns all enabled models from the registry (for client discoverability), but only the loaded model will actually serve completions.

---

## Service API Endpoints

### GET /api/service/status

Returns current service state.

**Response (JSON):**
```json
{
  "state": "running",
  "pid": 12345,
  "uptime_seconds": 3600,
  "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
  "model_display_name": "Hermes 3 Llama 3.1 8B",
  "model_config": {
    "dtype": "auto",
    "max_model_len": 8192,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "quantization": "",
    "enforce_eager": false,
    "enable_auto_tool_choice": true,
    "tool_call_parser": "hermes",
    "max_num_seqs": 16
  },
  "health": "healthy",
  "last_health_check": "2026-04-12T10:30:00Z",
  "restart_count": 0,
  "last_error": "",
  "vllm_port": 8000,
  "vllm_version": "0.8.x"
}
```

**HTML response:** Status card partial with color-coded banner (green=running, red=error, gray=stopped, yellow=starting/switching).

### POST /api/service/start

Start vLLM with a specific model.

**Request body:**
```json
{
  "model_id": "NousResearch/Hermes-3-Llama-3.1-8B"
}
```

**Behavior:**
1. Validate model_id exists in registry and is enabled.
2. Validate model path exists on disk.
3. If already running with a different model, return 409 Conflict with message "Stop current model first or use /api/service/switch".
4. If already running with the same model, return 409 Conflict "Already running".
5. Call `pm.Start(modelID)`.
6. Return 202 Accepted immediately (startup is async).

**Response:**
```json
{
  "status": "starting",
  "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
  "message": "vLLM starting, subscribe to /api/service/log-stream for progress"
}
```

**Error responses:**
- 404: Model not found in registry
- 409: Already running
- 422: Model validation failed (e.g. TP=2 requested but only 1 GPU)
- 500: Failed to spawn process

### POST /api/service/stop

Graceful shutdown.

**No request body needed.**

**Behavior:**
1. If not running, return 409 "Not running".
2. Call `pm.Stop()`.
3. Return 202 Accepted (shutdown is async).

**Response:**
```json
{
  "status": "stopping",
  "message": "Sending SIGTERM, waiting for graceful shutdown"
}
```

### POST /api/service/restart

Stop then start with the same model.

**No request body.** Uses the currently loaded model ID.

**Behavior:**
1. If not running, return 409 "Not running".
2. Store current model ID.
3. Stop, wait for exit, start.
4. Return 202 Accepted.

### POST /api/service/switch

Switch to a different model (convenience endpoint combining stop + start).

**Request body:**
```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct"
}
```

**Behavior:**
1. Validate new model.
2. Call `pm.SwitchModel(newModelID)`.
3. Return 202 Accepted.

### GET /api/service/logs

Return recent log lines from the ring buffer.

**Query params:**
- `lines` -- Number of lines to return (default 200, max 10000)
- `level` -- Filter by level: "all", "error", "warn" (default "all")

**Response (JSON):**
```json
{
  "lines": [
    {
      "timestamp": "2026-04-12T10:30:00.123Z",
      "source": "stdout",
      "text": "INFO 04-12 10:30:00 model_runner.py:123] Loading model weights...",
      "level": "info"
    }
  ],
  "total_buffered": 5432
}
```

**HTML response:** Pre-formatted log lines with ANSI color rendering and level-based color coding.

### GET /api/service/log-stream

SSE stream of live log lines.

**Event format:**
```
event: log
data: {"timestamp":"2026-04-12T10:30:00.123Z","source":"stdout","text":"INFO ...","level":"info"}

event: state
data: {"state":"running","health":"healthy"}
```

Two event types:
- `log` -- New log line from vLLM process
- `state` -- State change notification (starting -> running, running -> error, etc.)

### GET /api/service/health

Proxy to vLLM's `/health` endpoint. Returns the same status code vLLM returns.

**Response when vLLM healthy:**
```json
{
  "vllm_healthy": true,
  "manager_state": "running",
  "model_id": "NousResearch/Hermes-3-Llama-3.1-8B"
}
```

**Response when vLLM not running:**
```json
{
  "vllm_healthy": false,
  "manager_state": "stopped",
  "model_id": ""
}
```

### POST /api/service/clear-cache

Delete vLLM's compilation cache. Fixes graph compilation errors and stale cached kernels.

**What it deletes:**
- `~/.cache/vllm/` -- Compiled graph cache
- `/tmp/vllm_*` -- Temporary files from vLLM

**Behavior:**
1. If vLLM is running, return 409 "Stop vLLM before clearing cache".
2. Delete cache directories.
3. Return 200 with summary of bytes freed.

**Response:**
```json
{
  "cleared": true,
  "bytes_freed": 1073741824,
  "paths_cleared": ["~/.cache/vllm/", "/tmp/vllm_*"]
}
```

---

## OpenAI-Compatible Proxy (`internal/api/proxy.go`)

### Architecture

The Go server on `:3000` acts as a reverse proxy for `/v1/*` routes, forwarding to vLLM on `:8000`. This provides:

1. **Single port entry point** -- UI and API on `:3000`
2. **API key authentication** -- vLLM has no built-in auth
3. **Request/response logging** -- Configurable verbosity
4. **Timing capture** -- For passive benchmarking
5. **Sampling parameter injection** -- Per-model defaults
6. **Model list augmentation** -- Return all enabled models, not just loaded

### Reverse Proxy Setup

```
func SetupProxy(r chi.Router, pm *ProcessManager, registry *ModelRegistry, config *Config):
  proxy := httputil.NewSingleHostReverseProxy(
    &url.URL{Scheme: "http", Host: fmt.Sprintf("localhost:%d", pm.vllmPort)},
  )

  // Custom director: modify request before forwarding
  proxy.Director = func(req *http.Request):
    req.URL.Scheme = "http"
    req.URL.Host = fmt.Sprintf("localhost:%d", pm.vllmPort)
    req.Host = req.URL.Host
    // Strip /v1 prefix if needed (vLLM expects /v1/*)
    // Preserve original path

  // Custom response modifier: capture timing, log response
  proxy.ModifyResponse = func(resp *http.Response) error:
    captureResponseTiming(resp)
    logResponse(resp)
    return nil

  // Error handler: when vLLM is down
  proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error):
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusBadGateway)
    json.Encode(w, map[string]string{
      "error": "vLLM server is not running",
      "detail": err.Error(),
    })

  r.Route("/v1", func(r chi.Router):
    r.Use(apiKeyAuthMiddleware)
    r.Get("/models", handleModelsEndpoint)  // Custom handler, not proxied
    r.Handle("/*", proxy)                   // Everything else proxied
  )
```

### API Key Authentication Middleware

```
func apiKeyAuthMiddleware(next http.Handler) http.Handler:
  return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request):
    if config.APIKey == "":
      next.ServeHTTP(w, r)  // No key configured, allow all
      return

    auth := r.Header.Get("Authorization")
    if auth == "":
      respondError(w, 401, "Missing Authorization header")
      return

    if !strings.HasPrefix(auth, "Bearer "):
      respondError(w, 401, "Invalid Authorization format, expected 'Bearer <key>'")
      return

    key := strings.TrimPrefix(auth, "Bearer ")
    if key != config.APIKey:
      respondError(w, 401, "Invalid API key")
      return

    next.ServeHTTP(w, r)
  )
```

### GET /v1/models -- Custom Handler

Override vLLM's `/v1/models` to return all enabled models from the registry, not just the currently loaded model. This lets clients see available models even when they aren't loaded.

**Response format (OpenAI-compatible):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "NousResearch/Hermes-3-Llama-3.1-8B",
      "object": "model",
      "created": 1710000000,
      "owned_by": "NousResearch",
      "loaded": true,
      "quantization": "none",
      "has_tool_support": true,
      "tool_call_parser": "hermes",
      "max_model_len": 8192
    },
    {
      "id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
      "object": "model",
      "created": 1710100000,
      "owned_by": "Qwen",
      "loaded": false,
      "quantization": "awq",
      "has_tool_support": true,
      "tool_call_parser": "hermes",
      "max_model_len": 32768
    }
  ]
}
```

The `loaded` field is an extension beyond the OpenAI spec. Standard OpenAI clients will ignore it. The extra fields (`quantization`, `has_tool_support`, etc.) are also extensions for our UI.

### Tool Use Pass-Through

Tool/function calling requires correct proxying of specific request and response fields.

**Request fields that must pass through unchanged:**
```json
{
  "model": "NousResearch/Hermes-3-Llama-3.1-8B",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"  // "auto" | "none" | "required" | {"type":"function","function":{"name":"..."}}
}
```

**Response fields that must pass through unchanged:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"London\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

**Critical proxy requirements for tool use:**
- Do NOT modify the request body for `/v1/chat/completions` -- the `tools` and `tool_choice` fields must reach vLLM unaltered.
- Do NOT strip or modify `tool_calls` in the response.
- The `finish_reason` of `"tool_calls"` must pass through (some proxies normalize this).
- For streaming responses with tool calls, the `delta.tool_calls` field must pass through in each SSE chunk.
- Content-Type must remain `application/json` for non-streaming, `text/event-stream` for streaming.

**Validation before proxying tool use requests:**
- If request contains `tools` or `tool_choice` but the loaded model does NOT have `enable_auto_tool_choice`:
  - Log a warning (don't block -- vLLM will handle it, possibly ignoring tools)
  - Optionally add `X-Warning: Model loaded without --enable-auto-tool-choice` header

### Response Timing Capture

For passive benchmarking, capture timing data from proxied requests.

```
func captureResponseTiming(req *http.Request, resp *http.Response):
  if req.URL.Path != "/v1/chat/completions" && req.URL.Path != "/v1/completions":
    return

  // Only capture non-streaming responses (streaming has different timing characteristics)
  if resp.Header.Get("Content-Type") != "application/json":
    return

  // Read response body (need to buffer it to both read and forward)
  body, _ := io.ReadAll(resp.Body)
  resp.Body = io.NopCloser(bytes.NewReader(body))

  var result struct {
    Usage struct {
      PromptTokens     int `json:"prompt_tokens"`
      CompletionTokens int `json:"completion_tokens"`
      TotalTokens      int `json:"total_tokens"`
    } `json:"usage"`
  }
  json.Unmarshal(body, &result)

  timing := PassiveTiming{
    Timestamp:        time.Now(),
    ModelID:          pm.modelID,
    PromptTokens:     result.Usage.PromptTokens,
    CompletionTokens: result.Usage.CompletionTokens,
    TotalTimeMs:      time.Since(requestStartTime).Milliseconds(),
    HasToolCalls:     containsToolCalls(body),
  }

  passiveTimingStore.Add(timing)
```

### Sampling Parameter Injection

If the client request does NOT specify certain sampling parameters, inject per-model defaults from `generation_defaults`:

```
func injectDefaults(reqBody map[string]interface{}, model *Model):
  defaults := model.GenerationDefaults
  // Only inject if not already set by client
  if _, ok := reqBody["temperature"]; !ok && defaults.Temperature != nil:
    reqBody["temperature"] = *defaults.Temperature
  if _, ok := reqBody["top_p"]; !ok && defaults.TopP != nil:
    reqBody["top_p"] = *defaults.TopP
  // etc.
```

**Configurable behavior:** Global setting to enable/disable default injection. Default: disabled (pass through as-is, let vLLM use its own defaults).

### Request/Response Logging

Configurable verbosity levels:

| Level | What is logged |
|---|---|
| 0 (off) | Nothing |
| 1 (minimal) | Method, path, status code, latency |
| 2 (standard) | Level 1 + model, prompt/completion token counts |
| 3 (verbose) | Level 2 + request body (truncated), response body (truncated) |

Default: level 1. Stored in Go's standard logger. Also written to the ring buffer so they appear in the log viewer alongside vLLM logs (tagged as "proxy").

### Streaming (SSE) Response Proxying

vLLM supports streaming responses via SSE for `/v1/chat/completions` and `/v1/completions` when `"stream": true`.

**Critical requirements:**
- Set `Transfer-Encoding: chunked` and `Content-Type: text/event-stream` on the proxy response.
- Flush each SSE event immediately (use `http.Flusher`).
- Do NOT buffer the entire response (would defeat the purpose of streaming).
- Handle `data: [DONE]` as the terminal event.
- For streaming tool calls: `delta.tool_calls` arrives incrementally across multiple chunks. Pass each chunk through as-is.

```
proxy.FlushInterval = -1  // Flush immediately (for httputil.ReverseProxy)
// OR use custom transport that flushes per-chunk
```

**Edge case:** If the client disconnects mid-stream, cancel the upstream request to vLLM (via context cancellation) to free resources.

---

## Server Page UI (`web/templates/server.html`)

### Layout

Top-to-bottom layout:

1. **Status banner** (full width)
2. **Active model info + Quick actions** (left 2/3 + right 1/3)
3. **Log viewer** (full width, takes remaining vertical space)

### Status Banner

Full-width colored bar showing current state:

| State | Color | Text | Icon |
|---|---|---|---|
| Running | Green | "vLLM Running -- Hermes 3 Llama 3.1 8B" | Spinning circle/pulse |
| Stopped | Gray | "vLLM Stopped" | Stop icon |
| Starting | Yellow | "vLLM Starting -- Loading Hermes 3 Llama 3.1 8B..." | Spinner |
| Stopping | Yellow | "vLLM Stopping..." | Spinner |
| Switching | Yellow | "Switching model to Qwen 2.5 72B AWQ..." | Spinner |
| Error | Red | "vLLM Error -- GPU out of memory" | Warning triangle |

Updates via htmx polling (`hx-get="/api/service/status" hx-trigger="every 3s"`) or via SSE state events.

### Active Model Display

When running, show:
- Model name and repo ID
- Key config values: dtype, context length, TP, quant type, tool parser
- Uptime
- Health status indicator

### Model Selector and Controls

- **Model dropdown** -- Shows all enabled models from registry, grouped by quantization type:
  ```
  ── Full Precision ──
  Meta-Llama-3.1-8B-Instruct (BF16, 16.1GB)
  ── AWQ ──
  Qwen2.5-72B-Instruct-AWQ (AWQ 4-bit, 38.5GB, TP=2)
  ── GPTQ ──
  Hermes-3-Llama-3.1-8B-GPTQ (GPTQ 4-bit, 4.5GB)
  ── GGUF ──
  Mistral-7B-v0.3-Q4_K_M (GGUF Q4_K_M, 4.1GB)
  ```
  Each option shows: display name, quant type, VRAM estimate, TP requirement.

- **Start button** -- `hx-post="/api/service/start"` with selected model ID. Disabled while running.
- **Stop button** -- `hx-post="/api/service/stop"`. Disabled while stopped.
- **Restart button** -- `hx-post="/api/service/restart"`. Disabled while stopped.
- **Switch button** -- Only shown when running and a different model is selected in dropdown. `hx-post="/api/service/switch"`.

All buttons show loading state (spinner, disabled) during async operations.

### Live Log Viewer

- SSE-driven log display using `htmx-sse.js`.
- Connect to `/api/service/log-stream`.
- Auto-scroll to bottom (with scroll-lock toggle: click to pin/unpin).
- ANSI color rendering:
  - Parse ANSI escape codes from vLLM output (Python's colorama/rich output).
  - Convert to `<span>` elements with appropriate CSS classes.
  - Or use a lightweight JS library like `ansi_up`.
- Level-based coloring:
  - INFO lines: default color
  - WARNING lines: yellow
  - ERROR lines: red
- Monospace font, dark background (regardless of theme -- log viewers should always be dark).
- "Clear" button to clear the displayed log (doesn't clear ring buffer).
- "Download" button to download full ring buffer as text file.
- Line count indicator: "Showing 200 / 5,432 lines"

### Quick Actions Section

- **Clear vLLM cache** -- `hx-post="/api/service/clear-cache"`. Disabled while running. Show freed space in response.
- **Toggle eager mode** -- Quick toggle that updates model config and shows restart prompt.
- **Copy API endpoint** -- Button that copies `http://<host>:3000/v1` to clipboard.

### Connection Info Panel

Collapsible panel showing curl examples:

**Basic chat completion:**
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "NousResearch/Hermes-3-Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Chat completion with tool use:**
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "NousResearch/Hermes-3-Llama-3.1-8B",
    "messages": [{"role": "user", "content": "What is the weather in London?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

**Streaming:**
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "NousResearch/Hermes-3-Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Models list:**
```bash
curl http://localhost:3000/v1/models \
  -H "Authorization: Bearer <your-api-key>"
```

Replace `localhost:3000` with the configured external URL if set. Replace `<your-api-key>` with the actual key if configured (or omit the header if no key set).

### What to Copy from llama-toolchest

- **Server page layout structure** -- Status banner, model selector, start/stop buttons, log viewer. Same htmx patterns.
- **Log viewer component** -- SSE connection, auto-scroll, dark background. Adapt from `log-panel.js`.
- **Button loading states** -- htmx `hx-indicator` pattern with spinner SVGs.
- **Status polling** -- htmx periodic refresh of status banner.

### What is New vs llama-toolchest

- **Model switching** -- llama-toolchest had multi-model router mode. Here we have explicit stop/switch/start.
- **Tool use in connection info** -- New curl examples with tools/tool_choice.
- **Quantization in model selector grouping** -- llama-toolchest grouped by GGUF quant type, here by broader quant categories.
- **ANSI color rendering in log viewer** -- vLLM's Python logging uses rich/colorama more heavily than llama-server.
- **Clear vLLM cache action** -- New. llama-server had no persistent cache to clear.
- **Environment variable injection complexity** -- Much more complex than llama-server (ROCm vars, triton, tcmalloc, etc.).

---

## File Layout Summary

```
internal/process/
  manager.go       -- ProcessManager struct, Start/Stop/Restart/Switch, command construction,
                      environment injection, health checking, auto-restart
  ringbuffer.go    -- Ring buffer for log lines

internal/api/
  proxy.go         -- Reverse proxy setup, API key auth middleware, /v1/models custom handler,
                      tool use pass-through, timing capture, streaming support
  service.go       -- HTTP handlers for /api/service/* endpoints

web/templates/
  server.html      -- Server control page
  partials/
    server_status.html  -- Status banner partial (for htmx refresh)
    log_viewer.html     -- Log viewer component
```

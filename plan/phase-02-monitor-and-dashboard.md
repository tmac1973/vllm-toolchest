# Phase 2: System Monitor & Dashboard

## Goal

Build the real-time monitoring subsystem and a functional dashboard. The monitor polls GPU, CPU, and memory metrics on a configurable interval and pushes updates to the browser via SSE. The dashboard page displays live service status, GPU information, model inventory, and API endpoint info. The sidebar metrics bar updates in real-time.

---

## 1. Monitor Subsystem Architecture

### Copy from llama-toolchest

The entire `internal/monitor/` package is copied from llama-toolchest with minimal changes. The architecture is backend-agnostic via the `GPUBackend` interface.

#### Files to copy verbatim

| File | Rationale |
|------|-----------|
| `internal/monitor/monitor.go` | `Monitor` struct, polling goroutine, `Subscribe()`/`Unsubscribe()` fan-out — completely generic |
| `internal/monitor/rocm.go` | ROCm sysfs + rocm-smi parsing — already handles RDNA GPUs, gfx1201 uses same sysfs paths |
| `internal/monitor/cpu.go` | `/proc/stat` and `/proc/meminfo` parsing — Linux-generic, no changes needed |

#### Files NOT copied

| File | Why |
|------|-----|
| `internal/monitor/nvidia.go` | NVIDIA GPU backend — not needed for RDNA4-only project |

The `detectGPUBackend()` function in `monitor.go` tries NVIDIA first, then ROCm. For vllm-toolchest, simplify to only try ROCm:

```go
func detectGPUBackend() GPUBackend {
    if b := newROCm(); b != nil {
        return b
    }
    return nil
}
```

### Data Structures (unchanged from llama-toolchest)

```go
type Metrics struct {
    Timestamp time.Time  `json:"timestamp"`
    GPU       []GPUInfo  `json:"gpu,omitempty"`
    CPU       CPUInfo    `json:"cpu"`
    Memory    MemoryInfo `json:"memory"`
}

type GPUInfo struct {
    Index       int     `json:"index"`
    Name        string  `json:"name"`
    UtilPercent int     `json:"util_percent"`      // 0-100
    VRAMUsedMB  int     `json:"vram_used_mb"`
    VRAMTotalMB int     `json:"vram_total_mb"`
    TempC       int     `json:"temp_c"`
    PowerW      float64 `json:"power_w,omitempty"`
}

type CPUInfo struct {
    UsagePercent float64 `json:"usage_percent"`
    Cores        int     `json:"cores"`
}

type MemoryInfo struct {
    UsedMB  int `json:"used_mb"`
    TotalMB int `json:"total_mb"`
}
```

### New fields to add to GPUInfo

For RDNA4-specific monitoring, extend `GPUInfo` with:

```go
type GPUInfo struct {
    // ... existing fields ...
    FanPercent   int     `json:"fan_percent,omitempty"`    // 0-100
    ClockMHz     int     `json:"clock_mhz,omitempty"`     // current GPU clock
    DriverVersion string `json:"driver_version,omitempty"` // e.g. "7.2.1"
    ROCmVersion   string `json:"rocm_version,omitempty"`   // from rocm-smi or /opt/rocm/.info/version
}
```

These are populated at startup (driver/ROCm version) and per-poll (fan, clock).

---

## 2. rocm.go — GPU Metrics Collection

### Primary method: rocm-smi CSV

The llama-toolchest `rocm.go` already uses `rocm-smi --showuse --showmemuse --showtemp --showpower --csv`. Extend with additional flags:

```
rocm-smi --showuse --showmemuse --showtemp --showpower --showfan --showclocks --csv
```

#### rocm-smi CSV column names to parse

| Column Header | Maps to |
|---------------|---------|
| `device` | `GPUInfo.Index` |
| `GPU use (%)` | `GPUInfo.UtilPercent` |
| `Temperature (Sensor edge) (C)` | `GPUInfo.TempC` |
| `Average Graphics Package Power (W)` | `GPUInfo.PowerW` |
| `Fan speed (%)` or `Fan Speed (%)` | `GPUInfo.FanPercent` |
| `sclk clock speed (MHz)` or `SCLK` | `GPUInfo.ClockMHz` |

**Note**: rocm-smi CSV header names vary between ROCm versions. The llama-toolchest approach of building a `colIdx` map from the header row handles this robustly. TheRock's rocm-smi may have slightly different column names — parse defensively, log warnings for unrecognized columns instead of failing.

#### VRAM from sysfs (preferred)

The llama-toolchest already reads VRAM from sysfs rather than rocm-smi CSV (more reliable):

```
/sys/class/drm/card{N}/device/mem_info_vram_used    → bytes
/sys/class/drm/card{N}/device/mem_info_vram_total   → bytes
```

This works for gfx1201 — the amdgpu kernel driver exposes these for all RDNA GPUs.

#### GPU name from rocminfo

The llama-toolchest uses `rocminfo` output to get the marketing name. For gfx1201, this should report "AMD Radeon RX 9700 XT" (or similar). The existing parsing logic filters out CPU agents (Ryzen/EPYC) and iterates to the correct GPU index.

**Edge case**: Inside the container, `rocminfo` may report the GPU differently than on bare metal. If TheRock's ROCm doesn't recognize gfx1201's marketing name, it may report "gfx1201" as the name. Handle this gracefully — display whatever `rocminfo` provides, or fall back to "AMD GPU 0" as llama-toolchest already does.

#### Fallback: sysfs-only collection

If `rocm-smi` is not available in the container (TheRock nightlies may or may not include it), the sysfs fallback in llama-toolchest's `collectSysfs()` provides:

- GPU utilization: `/sys/class/drm/card{N}/device/gpu_busy_percent`
- VRAM: `mem_info_vram_used`, `mem_info_vram_total`
- Temperature: `/sys/class/drm/card{N}/device/hwmon/hwmon*/temp1_input` (millidegrees)
- Power: `/sys/class/drm/card{N}/device/hwmon/hwmon*/power1_average` (microwatts)

Extend sysfs fallback to also read:

- Fan: `/sys/class/drm/card{N}/device/hwmon/hwmon*/pwm1` (0-255 scale, convert to percentage)
- Clock: `/sys/class/drm/card{N}/device/pp_dpm_sclk` (parse the active entry marked with `*`)

#### ROCm/driver version detection

Read once at startup, cache the result:

```go
func readROCmVersion() string {
    // Try /opt/rocm/.info/version first
    if data, err := os.ReadFile("/opt/rocm/.info/version"); err == nil {
        return strings.TrimSpace(string(data))
    }
    // Try rocm-smi --showdriverversion
    if out, err := exec.Command("rocm-smi", "--showdriverversion").Output(); err == nil {
        // parse "Driver version: X.Y.Z"
    }
    return "unknown"
}
```

### Polling interval

Default 3 seconds (same as llama-toolchest). Configurable via `monitor_interval` in config if needed, but 3s is a good balance between responsiveness and CPU overhead.

The `collect()` method runs `rocm-smi` as a subprocess each poll. At 3s intervals this is negligible overhead. The sysfs reads are even cheaper (just file reads from procfs/sysfs virtual filesystems).

---

## 3. cpu.go — CPU and Memory Metrics

### Copy verbatim from llama-toolchest

The `collectCPU()` function:
1. Reads `/proc/stat` first line: `cpu user nice system idle iowait irq softirq steal`
2. Computes delta between current and previous total/idle
3. Returns usage percentage and core count via `runtime.NumCPU()`

The `collectMemory()` function:
1. Reads `/proc/meminfo`
2. Extracts `MemTotal` and `MemAvailable`
3. Returns used (total - available) and total in MB

Both use the `init()` function to prime the CPU calculation with an initial reading.

No changes needed for vllm-toolchest.

---

## 4. Monitor Polling Loop & Fan-out

### Copied from llama-toolchest

The `Monitor` struct:
```go
type Monitor struct {
    gpu      GPUBackend
    interval time.Duration
    mu       sync.RWMutex
    current  Metrics
    subs     map[chan Metrics]struct{}
    stop     chan struct{}
}
```

Key behaviors:
- `Start()` collects once immediately, then starts a ticker goroutine
- `collect()` gathers all metrics, stores in `current`, broadcasts to all subscribers
- `Subscribe()` returns a buffered channel (capacity 4) — subscriber gets latest metrics on each poll
- Broadcast is non-blocking: if a subscriber's channel is full, the update is dropped (slow consumer protection)
- `Unsubscribe()` removes the channel from the map
- `Current()` returns the latest snapshot under RLock (for one-shot requests)

This fan-out pattern is the foundation for SSE streaming to multiple browser tabs.

---

## 5. SSE Stream Endpoint

### API endpoints

```
GET /api/monitor        → latest metrics (JSON or HTML partial depending on HX-Request header)
GET /api/monitor/stream → SSE stream of metrics updates
```

### `/api/monitor` (htmx polling)

Copied from llama-toolchest. When the request has `HX-Request` header (htmx):
- Respond with the `monitor_bar` HTML partial
- This is used by the sidebar's `hx-get="/api/monitor" hx-trigger="load, every 3s"`

When it's a plain API request:
- Respond with JSON metrics

### `/api/monitor/stream` (SSE)

Copied from llama-toolchest:
1. Create `SSEWriter` from the `ResponseWriter`
2. Subscribe to the monitor
3. Send current state immediately as an `event: metrics` SSE message
4. Loop: receive from subscription channel, marshal to JSON, send as SSE event
5. Exit when client disconnects (`r.Context().Done()`) or channel closes

### SSE infrastructure (`internal/api/sse.go`)

Copied verbatim from llama-toolchest. The `SSEWriter` type:
- Sets `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`
- `SendEvent(event, data)` — named event
- `SendData(data)` — unnamed event
- `SendLine(data)` — multi-line SSE encoding
- `StreamLines(w, ctx, ch, doneMsg)` — helper for streaming log-like output

---

## 6. Dashboard Page

### Dashboard handler (`handleDashboard`)

`GET /api/dashboard` returns an HTML partial (for htmx) with four cards. This is called by the dashboard page with `hx-get="/api/dashboard" hx-trigger="load, every 5s" hx-swap="innerHTML"`.

### Card 1: Service Status

```html
<article>
    <header>vLLM Service</header>
    <p>{state badge: Running/Stopped/Starting/Failed}</p>
    <p>Model: {active model name or "None"}</p>
    <p>Uptime: {duration since start}</p>
    <p><a href="/service">Manage →</a></p>
</article>
```

**Data source**: In Phase 2, the vLLM process manager tracks state. For now in the dashboard handler:

```go
type vllmStatus struct {
    State       string // "running", "stopped", "starting", "failed"
    ModelName   string // currently loaded model
    StartedAt   time.Time
    ToolsEnabled bool
}
```

The dashboard handler calls `s.vllmManager.Status()` (to be implemented in a future phase, but the interface is defined now). For Phase 2, implement a minimal process manager that:
- Tracks whether a vLLM subprocess is running
- Stores the model name it was started with
- Reports uptime

State badge HTML (same pattern as llama-toolchest):
```go
switch status.State {
case "running":  badge = `<ins>Running</ins>`
case "starting": badge = `<mark>Starting...</mark>`
case "failed":   badge = `<del>Failed</del>`
default:         badge = `Stopped`
}
```

If tools/function-calling is enabled for the running model, show an indicator:
```html
<small>Tool use: <ins>enabled</ins></small>
```

### Card 2: GPU Info

```html
<article>
    <header>GPU</header>
    <p><strong>{GPU name}</strong></p>
    <p>VRAM: {used}/{total} GB</p>
    <p>Driver: {version} · ROCm: {version}</p>
    <p>Architecture: gfx1201 (RDNA 4)</p>
</article>
```

**Data source**: `s.monitor.Current()` — the latest metrics snapshot.

If multiple GPUs are detected (tensor parallel), show each GPU. For the R9700 XT target, there's typically one GPU.

### Card 3: Model Inventory

```html
<article>
    <header>Models</header>
    <p><strong>{count}</strong> models registered</p>
    <p><strong>{enabled_count}</strong> enabled for serving</p>
    <p>Formats: {list of detected quant formats}</p>
    <p><a href="/models">Manage →</a> · <a href="/models/browse">Get New →</a></p>
</article>
```

**Data source**: `s.registry.List()` (model registry, Phase 3). For now, returns 0/0.

The "Formats" line shows which quantization formats are present in the model inventory (e.g., "AWQ, GPTQ, FP16, GGUF"). This is informational — helps the user see what they have at a glance.

### Card 4: API Endpoint

```html
<article>
    <header>API Endpoint</header>
    <pre style="user-select:all; cursor:pointer;">{external_url}/v1</pre>
    <p>Compatible with OpenAI API clients</p>
    <p>Tool use: {enabled/disabled}</p>
    <p><a href="/settings">Settings →</a></p>
</article>
```

**Data source**: `s.cfg.ExternalURL` + "/v1" and `s.cfg.ToolUseEnabled`.

Shows the externally-reachable URL that clients can point their OpenAI SDK at. The "Tool use" indicator tells the user whether the endpoint will support function calling.

### Quick-launch section

Below the four cards, an optional quick-launch area:

```html
<article>
    <header>Quick Launch</header>
    <!-- Only shown if there's a previously-used model and vLLM is stopped -->
    <p>Last used: <strong>{model name}</strong></p>
    <button hx-post="/api/service/start" hx-vals='{"model_id": "{id}"}'>
        Start with {model name}
    </button>
</article>
```

This stores the last-used model ID in the config or a small state file. If vLLM is already running, this section doesn't appear.

---

## 7. Sidebar Metrics Bar

### Template: `partials/monitor_bar.html`

Copied from llama-toolchest with minor adjustments. Rendered by `handleMonitorStatus` when called with `HX-Request` header.

```html
{{define "monitor_bar"}}
{{range .GPUs}}
<div style="margin-bottom: 0.5rem;">
    <small><strong>GPU {{.Index}}</strong></small><br>
    <small>{{.Name}}</small><br>
    <small>Util: {{.UtilPercent}}%</small>
    <progress value="{{.UtilPercent}}" max="100" style="height:6px;margin:2px 0;"></progress>
    <small>VRAM: {{.VRAMGB}}</small>
    <progress value="{{.VRAMPercent}}" max="100" style="height:6px;margin:2px 0;"></progress>
    <small>{{.Details}}</small>
</div>
{{end}}
<div>
    <small>CPU: {{printf "%.0f" .CPUPercent}}%</small>
    <progress value="{{printf "%.0f" .CPUPercent}}" max="100" style="height:6px;margin:2px 0;"></progress>
    <small>RAM: {{.RAMGB}}</small>
    <progress value="{{.RAMPercent}}" max="100" style="height:6px;margin:2px 0;"></progress>
</div>
{{end}}
```

The `monitorBarData()` function (copied from llama-toolchest) pre-computes display values:

```go
type monitorBarGPU struct {
    Index       int
    Name        string
    UtilPercent int
    VRAMPercent int
    VRAMGB      string    // "4.2/16.0GB"
    Details     string    // "65°C · 120W · Fan: 45% · 2400MHz"
}
```

Extended `Details` string for RDNA4: include fan speed and clock speed from the new GPUInfo fields:

```go
details := fmt.Sprintf("%d°C", gpu.TempC)
if gpu.PowerW > 0 {
    details += fmt.Sprintf(" · %.0fW", gpu.PowerW)
}
if gpu.FanPercent > 0 {
    details += fmt.Sprintf(" · Fan: %d%%", gpu.FanPercent)
}
if gpu.ClockMHz > 0 {
    details += fmt.Sprintf(" · %dMHz", gpu.ClockMHz)
}
```

### htmx polling vs SSE

The sidebar uses `hx-get="/api/monitor" hx-trigger="load, every 3s"` — simple polling, same as llama-toolchest. This is simpler than SSE for the sidebar because:
- The sidebar HTML is small (< 1KB)
- 3s polling matches the monitor interval
- No need to maintain a persistent SSE connection just for the sidebar
- SSE is reserved for things that need sub-second updates (download progress, log streaming)

The `/api/monitor/stream` SSE endpoint is available for clients that want real-time metrics (e.g., a custom monitoring dashboard, Grafana integration).

---

## 8. Detailed Data Flow

### On page load

1. Browser requests `GET /` → Go renders `index.html` with layout, empty `#dashboard-content` div
2. htmx fires `hx-get="/api/dashboard"` on load → handler returns HTML cards → swapped into `#dashboard-content`
3. htmx fires `hx-get="/api/monitor"` on load → handler returns monitor bar HTML → swapped into `#monitor-bar`
4. Every 3 seconds, both fire again (htmx `every 3s` trigger)

### Monitor poll cycle (server-side)

1. `Monitor.Start()` creates a `time.Ticker` at 3s interval
2. On tick: `collect()` calls `collectCPU()`, `collectMemory()`, `gpu.Collect()`
3. `gpu.Collect()` runs `rocm-smi --csv` (or falls back to sysfs)
4. Result stored in `Monitor.current` under write lock
5. All subscriber channels receive the new `Metrics` (non-blocking send)

### SSE stream (for `/api/monitor/stream`)

1. Client connects with `EventSource('/api/monitor/stream')`
2. Handler subscribes to monitor, sends current state immediately
3. On each new metrics from subscription channel: JSON-marshal, send as `event: metrics`
4. Client disconnect detected via `r.Context().Done()`
5. Handler unsubscribes and returns

---

## 9. Server Struct Changes

### New fields

```go
type Server struct {
    cfg       *config.Config
    pages     map[string]*template.Template
    router    chi.Router
    monitor   *monitor.Monitor
    // Phase 2 additions:
    // vllmMgr  *vllm.Manager    — placeholder, implemented in service control phase
    // registry *models.Registry — placeholder, implemented in Phase 3
}
```

### NewServer initialization

```go
func NewServer(cfg *config.Config) *Server {
    mon := monitor.New(3 * time.Second)
    mon.Start()

    s := &Server{
        cfg:     cfg,
        monitor: mon,
    }
    s.pages = s.parseTemplates()
    s.router = s.buildRouter()
    return s
}
```

Much simpler than llama-toolchest's `NewServer` — no builder, benchmark, registry, process manager, downloader initialization. Those are added in later phases.

### Template function map

```go
funcMap := template.FuncMap{
    "divGB": func(bytes int64) float64 { return float64(bytes) / (1024 * 1024 * 1024) },
    "pctOf": func(value, max float64) float64 {
        if max == 0 { return 0 }
        return (value / max) * 100
    },
}
```

Simpler than llama-toolchest — no `vramFit` (which depends on GGUF-specific VRAM estimation), no `divf` (unused).

---

## 10. What to Copy vs Build New — Summary

### Copy from llama-toolchest (with import path rename)

- `internal/api/sse.go` — verbatim
- `internal/api/respond.go` — verbatim
- `internal/api/middleware.go` — verbatim
- `internal/api/monitor.go` — adapt `monitorBarData` Details string for new GPU fields
- `internal/monitor/monitor.go` — remove NVIDIA from `detectGPUBackend`
- `internal/monitor/rocm.go` — extend for fan speed, clock, driver version
- `internal/monitor/cpu.go` — verbatim
- `web/static/*` — verbatim
- `web/embed.go` — verbatim
- `web/templates/partials/monitor_bar.html` — extend for new fields
- `web/templates/layout.html` — rebrand, update nav

### Build new

- `internal/api/server.go` — new Server struct, simplified buildRouter, new dashboard handler
- `internal/config/config.go` — new Config struct with vLLM-specific fields
- `cmd/vllmctl/main.go` — simplified entry point
- `web/templates/index.html` — new dashboard template with vLLM-specific cards

---

## 11. Edge Cases

### rocm-smi not in TheRock

TheRock nightlies may not include `rocm-smi`. The sysfs fallback handles this gracefully — it's already implemented in llama-toolchest's `rocm.go`. The `Collect()` method tries `rocm-smi` first, falls back to sysfs on any error.

**Check**: Verify that TheRock installs `rocminfo` (needed for GPU name detection). If not, the GPU name falls back to "AMD GPU 0".

### No GPU in development (local dev without GPU)

When developing on a machine without an AMD GPU:
- `detectGPUBackend()` returns `nil` (no `/dev/kfd`)
- `Monitor.collect()` skips GPU metrics (`m.gpu == nil`)
- Dashboard shows "No GPU detected" instead of GPU card
- All other functionality works (the Go UI, htmx, templates)

### Multiple GPUs

The monitor supports multiple GPUs natively (iterates sysfs cards, collects per-GPU metrics). The dashboard and sidebar show all GPUs. For tensor-parallel vLLM configs, this is important — the user needs to see VRAM usage across all GPUs.

### Monitor memory leak

The subscribe/unsubscribe pattern has a potential leak if a handler crashes without unsubscribing. The `defer s.monitor.Unsubscribe(ch)` in the SSE handler prevents this. The buffered channel (capacity 4) prevents blocking the monitor's collect loop if a subscriber is slow.

### Container restart and state persistence

The monitor is stateless — it re-collects on every poll. No state needs to persist across container restarts. The only persistent state in Phase 2 is the config file (`/data/config/vllmctl.yaml`).

---

## 12. Validation Criteria

Phase 2 is complete when:

1. Dashboard loads at `/` with four functional cards
2. GPU card shows correct GPU name, VRAM total, and live VRAM usage
3. Monitor bar in sidebar updates every 3 seconds with:
   - GPU utilization percentage + progress bar
   - VRAM used/total + progress bar
   - Temperature, power, fan speed, clock in details line
   - CPU usage + progress bar
   - RAM used/total + progress bar
4. `GET /api/monitor` returns JSON metrics with all fields populated
5. `GET /api/monitor/stream` opens an SSE connection and streams metrics every 3 seconds
6. Dashboard `every 5s` polling updates cards without full page reload
7. No GPU machine: dashboard gracefully shows "No GPU detected", sidebar shows CPU/RAM only
8. Multiple browser tabs: each gets independent SSE streams without interference

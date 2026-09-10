# Phase 6: Benchmarking

Adapt the benchmarking subsystem from [llama-toolchest](https://github.com/tmlabonte/llama-toolchest) to vllm-toolchest. The goal is feature parity with llama-toolchest's benchmark UI and data model, plus vLLM-specific additions (context-length probing, vLLM-process-aware job execution).

Two benchmark sources, mirroring llama-toolchest:

- **internal-*** presets — Go HTTP loop that streams `/v1/chat/completions`, measures TTFT from the first SSE chunk, and reports per-test-point progress live.
- **benchy-*** presets — Shell out to [`llama-benchy`](https://github.com/eugr/llama-benchy) via `uvx`. Engine-agnostic, gives apples-to-apples comparison against llama.cpp / Ollama / other OpenAI-compatible engines.

Plus three vLLM-specific features:

- **Jobs / matrices** — Sweep `{models} × {presets}` with model-grouped execution (load each model once, run all its cells, then swap).
- **Context-length probing** — Binary-search the maximum `--max-model-len` that doesn't OOM across multiple `gpu_memory_utilization` and `max_num_seqs` levels. Stores verified values per model.
- **Passive timing capture** — The OpenAI proxy records timing from every real chat completion and maintains per-model running averages, surfaced as badges on the dashboard and model cards.

---

## File Layout

```
internal/benchmark/
  benchmark.go       -- Store, BenchmarkRun, BenchmarkSummary, Preset, GPUSnapshot, ConfigSnapshot
  runner.go          -- Internal HTTP runner (streaming SSE for TTFT), warmup, prompt generation
  benchy.go          -- llama-benchy shell-out via uvx, JSON result parsing
  stats.go           -- ComputeSummary, BuildComparison (port from llama-toolchest)
  job.go             -- BenchmarkJob, JobCell, ConfigOverrides, status constants
  job_runner.go      -- JobQueue (single-job-at-a-time), JobEnv interface, model-grouped execution
  context_probe.go   -- ProbeMaxContext, binary search, tryStartVLLM, OOM log detection
  timing.go          -- TimingSample, per-model ring buffer, running averages (passive capture)

internal/api/
  bench.go           -- Run handlers: list, get, start, cancel, delete, batch-delete, progress SSE
  bench_jobs.go      -- Job handlers: list, get, create, cancel, retry-failed, delete (cascade/orphan)
  bench_about.go     -- "About benchmarks" disclosure modal (presets, prompt text, benchy command)
  bench_export.go    -- CSV / JSON export (single run, multi-run)
  bench_probe.go     -- Context probe handlers (start, progress SSE, results)
  jobs_env.go        -- *Server -> JobEnv adapter

web/templates/
  benchmarks.html               -- Page with job-grouped table, ad-hoc job, multi-select actions
  partials/
    bench_job_list.html         -- Job-grouped tables (one tbody per job, expandable runs)
    bench_run_row.html          -- Single run row (used inside job tbody)
    bench_detail.html           -- Expanded run detail (results table, summary, config snapshot)
    bench_compare.html          -- Side-by-side comparison of selected runs
    bench_progress.html         -- Active run progress panel (SSE-driven)
    bench_about.html            -- "About benchmarks" modal contents
    bench_form.html             -- New-benchmark form (single run)
    bench_job_form.html         -- New-job form (matrix selector)
    bench_probe.html            -- Context probe modal + results display

internal/process/        -- (existing) extended with secondary-port spawn for context probing

Dockerfile.rocm
Dockerfile.cuda          -- Both add: install uv so `uvx llama-benchy` works inside the container
```

---

## Data Model

Closely mirrors llama-toolchest. Differences are flagged with **vLLM** annotations.

### BenchmarkRun

```go
type BenchmarkRun struct {
    ID        string    `json:"id"`             // uuid
    JobID     string    `json:"job_id,omitempty"` // owning job; "adhoc" for single-run path
    CreatedAt time.Time `json:"created_at"`
    Status    string    `json:"status"`         // running | completed | failed
    Error     string    `json:"error,omitempty"`

    ModelID     string  `json:"model_id"`       // HF repo id from registry
    ModelName   string  `json:"model_name"`     // display name
    Quant       string  `json:"quant"`          // "awq" | "gptq" | "fp8" | "bf16" | "none"
    SizeGB      float64 `json:"size_gb"`        // on-disk size

    Config ConfigSnapshot `json:"config"`

    // vLLM: no Build snapshot (vLLM ships pre-built in the container image).
    // Instead, freeze the container image tag and vLLM version reported by the running process.
    VLLMVersion string `json:"vllm_version,omitempty"` // from /v1/models or proc startup log
    ImageTag    string `json:"image_tag,omitempty"`    // from env or /etc/os-release

    GPUs []GPUSnapshot `json:"gpus"`

    Preset       string `json:"preset"`
    PromptTokens []int  `json:"prompt_tokens"`
    GenTokens    int    `json:"gen_tokens"`

    Results    []BenchmarkResult   `json:"results,omitempty"`
    Summary    *BenchmarkSummary   `json:"summary,omitempty"`
    LlamaBenchy []LlamaBenchyResult `json:"llama_benchy,omitempty"` // benchy-* presets

    BenchyCommand string `json:"benchy_command,omitempty"`

    Warnings []string `json:"warnings,omitempty"`

    ProgressDetail string `json:"progress_detail,omitempty"` // transient, polled by HTMX
    DurationMs     int64  `json:"duration_ms,omitempty"`
}
```

### ConfigSnapshot (vLLM-specific)

Per the answered scope: capture only fields that meaningfully affect perf.

```go
type ConfigSnapshot struct {
    MaxModelLen          int     `json:"max_model_len"`
    TensorParallelSize   int     `json:"tensor_parallel_size"`
    GPUMemoryUtilization float64 `json:"gpu_memory_utilization"`
    KVCacheDtype         string  `json:"kv_cache_dtype"`           // "auto" | "fp8" | "fp8_e5m2"
    EnforceEager         bool    `json:"enforce_eager"`
    Dtype                string  `json:"dtype"`                    // "auto" | "bfloat16" | "float16"
    QuantMethod          string  `json:"quant_method,omitempty"`   // "awq" | "gptq" | "fp8" | ""
}
```

### BenchmarkResult (per test point)

```go
type BenchmarkResult struct {
    PromptTokens    int     `json:"prompt_tokens"`    // from usage.prompt_tokens
    GenTokens       int     `json:"gen_tokens"`       // from usage.completion_tokens
    Repetition      int     `json:"repetition"`
    PromptTokPerSec float64 `json:"prompt_tok_per_sec"` // PromptTokens / (TTFT/1000)
    GenTokPerSec    float64 `json:"gen_tok_per_sec"`    // GenTokens / ((TotalMs-TTFT)/1000)
    TTFTMs          float64 `json:"ttft_ms"`            // measured from first SSE chunk
    TotalMs         float64 `json:"total_ms"`           // wall clock for full request
}
```

### BenchmarkSummary

```go
type BenchmarkSummary struct {
    AvgPromptTokPerSec float64 `json:"avg_prompt_tok_per_sec"`
    AvgGenTokPerSec    float64 `json:"avg_gen_tok_per_sec"`
    AvgTTFTMs          float64 `json:"avg_ttft_ms"`
    MinGenTokPerSec    float64 `json:"min_gen_tok_per_sec"`
    MaxGenTokPerSec    float64 `json:"max_gen_tok_per_sec"`
}
```

### GPUSnapshot

```go
type GPUSnapshot struct {
    Index       int    `json:"index"`
    Name        string `json:"name"`          // "AMD Radeon RX 9070 XT"
    VRAMTotalMB int    `json:"vram_total_mb"` // 32768
}
```

Built from `monitor.Metrics` via a `GPUSnapshotsFromMetrics` helper (same pattern as llama-toolchest).

### LlamaBenchyResult

Copy verbatim from `llama-toolchest/internal/benchmark/benchy.go`. Same upstream schema (`pp_throughput`, `tg_throughput`, `ttfr`, `e2e_ttft`, etc.).

---

## Presets

Mirrors llama-toolchest's pattern: an `internal-*` family and a `benchy-*` family. Different token counts because vLLM typically handles larger contexts than CPU-bound llama.cpp builds.

```go
func Presets() []Preset {
    return []Preset{
        {
            Name:         "internal-quick",
            Label:        "internal-quick — 1 rep, 512-token prompt (~15s)",
            Description:  "Single streaming chat completion at 512-token prompt / 128 gen tokens. Sanity check after model load.",
            Source:       PresetSourceInternal,
            PromptTokens: []int{512}, GenTokens: 128, Repetitions: 1,
        },
        {
            Name:         "internal-standard",
            Label:        "internal-standard — 3 reps × 3 prompt sizes (~2 min)",
            Description:  "Three streaming chat completions at 128 / 512 / 2048-token prompts (128 gen tokens). Captures TTFT and gen-TPS scaling across short contexts.",
            Source:       PresetSourceInternal,
            PromptTokens: []int{128, 512, 2048}, GenTokens: 128, Repetitions: 3,
        },
        {
            Name:         "internal-thorough",
            Label:        "internal-thorough — 5 reps × 4 prompt sizes up to 8K (~8 min)",
            Description:  "Five repetitions at 128 / 512 / 2048 / 8192-token prompts with 256 generated tokens. Stresses long-context prefill.",
            Source:       PresetSourceInternal,
            PromptTokens: []int{128, 512, 2048, 8192}, GenTokens: 256, Repetitions: 5,
        },
        {
            Name:         "internal-long-ctx",
            Label:        "internal-long-ctx — 1 rep, 32K prompt / 512 gen",
            Description:  "Single 32768-token prompt with 512 generated tokens. Stresses KV cache, paged attention, and KV-cache dtype on long contexts.",
            Source:       PresetSourceInternal,
            PromptTokens: []int{32768}, GenTokens: 512, Repetitions: 1,
        },
        {
            Name:         "benchy-quick",
            Label:        "benchy-quick — 1 rep, 512 prompt / 32 gen via llama-benchy (~30s)",
            Description:  "Single-shot llama-benchy run. Smoke test for the engine-agnostic comparison path.",
            Source:       PresetSourceBenchy,
            PromptTokens: []int{512}, GenTokens: 32, Repetitions: 1, Concurrency: []int{1},
        },
        {
            Name:         "benchy-standard",
            Label:        "benchy-standard — 3 reps, 2048 prompt / 128 gen via llama-benchy (~2 min)",
            Description:  "Three-run llama-benchy benchmark at 2048-token prompts. Comparable to llama-toolchest's benchy-standard.",
            Source:       PresetSourceBenchy,
            PromptTokens: []int{2048}, GenTokens: 128, Repetitions: 3, Concurrency: []int{1},
        },
        {
            Name:         "benchy-concurrency",
            Label:        "benchy-concurrency — 3 reps × {1,2,4,8} concurrent via llama-benchy (~5 min)",
            Description:  "Stress vLLM's continuous batching by sweeping concurrency levels. Highlights where prefill saturates vs. throughput scales.",
            Source:       PresetSourceBenchy,
            PromptTokens: []int{1024}, GenTokens: 128, Repetitions: 3, Concurrency: []int{1, 2, 4, 8},
        },
    }
}
```

**Aliases** for backward compatibility (none needed — vllm-toolchest is starting fresh — but reserve the pattern for future renames).

Skip rule (same as llama-toolchest): if a preset's largest `PromptTokens + GenTokens` exceeds the loaded model's `max_model_len`, the runner skips that point and adds a warning to `run.Warnings` rather than erroring the whole run.

---

## Storage

Single `/data/config/benchmarks.json` file, loaded whole at startup. v1 envelope (no migration needed; we're starting fresh):

```json
{
  "version": 1,
  "jobs": [
    { /* BenchmarkJob */ }
  ],
  "runs": [
    { /* BenchmarkRun */ }
  ]
}
```

```go
type Store struct {
    mu       sync.RWMutex
    dataDir  string
    runs     []BenchmarkRun
    jobs     []BenchmarkJob

    timingsMu sync.RWMutex
    timings   map[string][]TimingSample  // per-model ring buffer (passive capture)
}

const schemaVersion = 1
const maxTimingSamples = 1000
```

**Methods** (port from llama-toolchest verbatim, removing build-related logic):

- `NewStore(dataDir string) *Store` — loads on construction
- `List() []BenchmarkRun` — newest first
- `Get(id string) (*BenchmarkRun, error)`
- `Save(run BenchmarkRun)` — append or update; rewrites file atomically (write to `.tmp`, rename)
- `Delete(id string) error`
- `BatchDelete(ids []string) (deleted, notFound int)`
- `ListJobs() []BenchmarkJob`
- `GetJob(id string) (*BenchmarkJob, error)`
- `SaveJob(job BenchmarkJob)`
- `DeleteJob(id string, disposition DeleteDisposition) error` — cascade deletes runs, orphan reassigns to `AdhocJobID`
- `RunsForJob(jobID string) []BenchmarkRun`
- `AddTiming(sample TimingSample)` — passive capture; trims to `maxTimingSamples` per model
- `RecentTimings(modelID string, n int) []TimingSample`
- `RunningAverage(modelID string) (avgGenTPS float64, count int, ok bool)`

The `AdhocJobID = "adhoc"` constant is a synthetic catch-all for single-run benchmarks not tied to an explicit job. The store synthesizes an Ad-Hoc Runs pseudo-job for display when there's at least one orphan run.

---

## Internal HTTP Runner (`runner.go`)

Port the shape from llama-toolchest's `runner.go`. Key changes for vLLM:

1. **Streaming SSE for TTFT.** llama.cpp returns a `timings` struct in its non-streaming response; vLLM does not. So every internal-* request is streamed, and we time the first `data:` chunk.
2. **No router unload/load.** vLLM runs one model per process; switching models is a process restart (see Job Runner below). For single-run benchmarks the model is assumed already loaded; the runner just verifies via `GET /v1/models`.
3. **Warmup** stays: send a short streaming request (64 prompt / 16 gen) with retry-with-backoff, in case vLLM is mid-startup.

### Streaming Loop

```go
func (r *Runner) runOneTest(ctx context.Context, vllmURL, model string, promptTokens, genTokens, rep int) (*BenchmarkResult, error) {
    prompt := buildPrompt(promptTokens, rep)
    body, _ := json.Marshal(map[string]any{
        "model":       model,
        "messages":    []map[string]string{{"role": "user", "content": prompt}},
        "max_tokens":  genTokens,
        "temperature": 0.0,
        "stream":      true,
        "stream_options": map[string]any{"include_usage": true},
    })

    req, _ := http.NewRequestWithContext(ctx, "POST", vllmURL+"/v1/chat/completions", bytes.NewReader(body))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "text/event-stream")

    client := &http.Client{Timeout: 10 * time.Minute}
    startTime := time.Now()
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    if resp.StatusCode != 200 {
        b, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, b)
    }

    var (
        ttft        time.Duration
        firstChunk  = true
        usagePrompt int
        usageGen    int
    )

    scanner := bufio.NewScanner(resp.Body)
    scanner.Buffer(make([]byte, 1<<20), 1<<20) // 1 MiB lines for big chunks
    for scanner.Scan() {
        line := scanner.Text()
        if !strings.HasPrefix(line, "data: ") {
            continue
        }
        payload := strings.TrimPrefix(line, "data: ")
        if payload == "[DONE]" {
            break
        }
        if firstChunk {
            ttft = time.Since(startTime)
            firstChunk = false
        }
        // Last chunk (with stream_options.include_usage) carries usage
        var chunk struct {
            Usage *struct {
                PromptTokens     int `json:"prompt_tokens"`
                CompletionTokens int `json:"completion_tokens"`
            } `json:"usage"`
        }
        if json.Unmarshal([]byte(payload), &chunk) == nil && chunk.Usage != nil {
            usagePrompt = chunk.Usage.PromptTokens
            usageGen = chunk.Usage.CompletionTokens
        }
    }
    if err := scanner.Err(); err != nil {
        return nil, err
    }

    total := time.Since(startTime)
    if usageGen == 0 {
        return nil, fmt.Errorf("no usage in response (set stream_options.include_usage)")
    }

    ttftMs := float64(ttft.Milliseconds())
    totalMs := float64(total.Milliseconds())
    genMs := totalMs - ttftMs
    if genMs < 1 {
        genMs = 1 // guard for tiny generations
    }

    return &BenchmarkResult{
        PromptTokens:    usagePrompt,
        GenTokens:       usageGen,
        Repetition:      rep,
        TTFTMs:          ttftMs,
        TotalMs:         totalMs,
        PromptTokPerSec: float64(usagePrompt) / (ttftMs / 1000.0),
        GenTokPerSec:    float64(usageGen) / (genMs / 1000.0),
    }, nil
}
```

**Note on `stream_options.include_usage`:** vLLM supports this OpenAI extension and emits a final SSE chunk with `usage` populated. If a future vLLM version drops this we fall back to counting tokens client-side (approximate).

### Prompt Generation

Copy the exact constants and helper from llama-toolchest so the prompts are byte-for-byte the same and results compare cleanly across engines:

```go
const BenchPromptText = `The history of computing is a story of human ingenuity ...`
const BenchPromptPrefixTemplate = "This is benchmark repetition number %d. Please analyze the following text carefully and provide a detailed response.\n\n"
const BenchPromptCharsPerToken = 4

func buildPrompt(targetTokens, rep int) string { /* unchanged */ }
```

The repetition prefix defeats prompt caching (vLLM's prefix cache, like llama.cpp's, would otherwise inflate prompt TPS on repeated runs).

---

## Benchy Runner (`benchy.go`)

Copy `benchy.go` from llama-toolchest **almost verbatim**. Only behavioral differences:

- The `BaseURL` points at `http://localhost:<vllm-port>/v1` instead of the llama.cpp router.
- `--tokenizer` receives the HF repo id from the model registry (same as llama-toolchest).
- The container has `uv` pre-installed (see Container Changes below) so the `exec.LookPath("uvx")` check should always succeed when running inside the container. When run on the host (development), we surface the error as before.

The `BenchyConfig`, `BuildBenchyArgs`, `FormatBenchyCommand`, `summarizeBenchy`, and `runLlamaBenchy` functions copy directly.

Result mapping: `summarizeBenchy` picks the `Concurrency: 1` row for the BenchmarkSummary so the badge values are comparable to internal-* runs. The full multi-concurrency report is preserved in `run.LlamaBenchy`.

---

## Jobs (`job.go`, `job_runner.go`)

### Job Model

Mirrors llama-toolchest's job.go. **vLLM-specific change:** no `BuildIDs` dimension (vLLM is monolithic). Matrix is `{ModelIDs} × {Presets}`.

```go
type BenchmarkJob struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Description string    `json:"description,omitempty"`
    Kind        string    `json:"kind"`   // "batch" | "ad-hoc"
    Status      string    `json:"status"`

    CreatedAt  time.Time `json:"created_at"`
    StartedAt  time.Time `json:"started_at,omitempty"`
    FinishedAt time.Time `json:"finished_at,omitempty"`

    ModelIDs []string  `json:"model_ids,omitempty"`
    Presets  []string  `json:"presets,omitempty"`
    Overrides *ConfigOverrides `json:"overrides,omitempty"`

    Cells []JobCell `json:"cells,omitempty"`
}

type ConfigOverrides struct {
    MaxModelLen          *int     `json:"max_model_len,omitempty"`
    TensorParallelSize   *int     `json:"tensor_parallel_size,omitempty"`
    GPUMemoryUtilization *float64 `json:"gpu_memory_utilization,omitempty"`
    KVCacheDtype         *string  `json:"kv_cache_dtype,omitempty"`
    EnforceEager         *bool    `json:"enforce_eager,omitempty"`
    Dtype                *string  `json:"dtype,omitempty"`
}

type JobCell struct {
    ModelID        string `json:"model_id"`
    Preset         string `json:"preset"`
    Status         string `json:"status"`
    Attempt        int    `json:"attempt"`
    BenchmarkRunID string `json:"benchmark_run_id,omitempty"`
    Error          string `json:"error,omitempty"`
}
```

Status constants are the same as llama-toolchest: `JobStatus{Pending,Running,Completed,Failed,Canceled}`; `CellStatus{Pending,Running,Completed,Failed,Skipped}`.

### JobEnv Interface (vLLM-adapted)

```go
type JobEnv interface {
    // EnsureModelLoaded stops the current vLLM process if a different model is
    // active, starts vLLM with the target model+config, and blocks until the
    // server reports healthy via /v1/models. May take minutes for large models.
    EnsureModelLoaded(ctx context.Context, modelID string, cfg ConfigSnapshot) error

    // CurrentLoadedModel returns the HF repo id of the model vLLM is currently
    // serving, or "" if vLLM is stopped.
    CurrentLoadedModel() string

    // ResolveModel returns registry data for a model.
    ResolveModel(modelID string) (ModelInfo, error)

    // CurrentMetrics returns the latest GPU metrics for snapshotting.
    CurrentMetrics() monitor.Metrics

    // VLLMURL returns the base URL the runner targets (e.g. http://localhost:8000).
    VLLMURL() string

    // HFToken / HFCacheDir forwarded to llama-benchy (same as llama-toolchest).
    HFToken() string
    HFCacheDir() string

    // VLLMVersion returns the version string of the running vllm (parsed from
    // its startup log or /v1/models -- whichever's accessible).
    VLLMVersion() string
}

type ModelInfo struct {
    HFRepoID    string         // passed to llama-benchy --tokenizer and used as ConfigSnapshot.ModelID
    Quant       string
    SizeGB      float64
    DisplayName string
    ServedName  string         // what /v1/models returns; passed in the chat request "model" field
    Config      ConfigSnapshot // model's saved baseline; overlay ConfigOverrides on this
}
```

The api layer implements `JobEnv` in `internal/api/jobs_env.go`, holding refs to `*process.Manager`, the model registry, the monitor, and config.

### Job Execution: Model-Grouped

Per the answered scope, cells are reordered so each model loads exactly once per job. Within a model, all its presets run back-to-back, then we swap.

```go
// In job_runner.go
func (q *JobQueue) runJob(ctx context.Context, job *BenchmarkJob) {
    // Group cells by model, preserving preset order within each group.
    byModel := map[string][]int{}
    var modelOrder []string
    for i, cell := range job.Cells {
        if _, seen := byModel[cell.ModelID]; !seen {
            modelOrder = append(modelOrder, cell.ModelID)
        }
        byModel[cell.ModelID] = append(byModel[cell.ModelID], i)
    }

    for _, modelID := range modelOrder {
        if ctx.Err() != nil {
            break
        }

        // Load this model once (with the per-model config + job overrides).
        info, err := q.env.ResolveModel(modelID)
        if err != nil {
            q.markCellsFailed(job, byModel[modelID], err.Error())
            continue
        }
        cfg := applyOverrides(info.Config, job.Overrides)

        if err := q.env.EnsureModelLoaded(ctx, modelID, cfg); err != nil {
            q.markCellsFailed(job, byModel[modelID], "model load failed: "+err.Error())
            continue
        }

        // Run all cells for this model.
        for _, idx := range byModel[modelID] {
            if ctx.Err() != nil {
                job.Cells[idx].Status = CellStatusSkipped
                continue
            }
            q.runCell(ctx, job, idx, info, cfg)
            q.store.SaveJob(*job)
        }
    }

    job.Status = jobFinalStatus(job)
    job.FinishedAt = time.Now()
    q.store.SaveJob(*job)
}
```

Single-job-at-a-time semantics (a `sync.Mutex` guard on `JobQueue.current`) — same as llama-toolchest. `ErrJobAlreadyRunning` returned on `Submit` when busy.

### Ad-Hoc Single Runs

The existing single-run "start a benchmark on the current model" path still works: it creates a one-cell `BenchmarkJob` with `Kind: "ad-hoc"`, `ID: "adhoc"` (the synthetic catch-all is reused as the JobID — or a fresh one-shot job is created and the run is reassigned to AdhocJobID on save; choose one and stick with it). The runner skips `EnsureModelLoaded` if the requested model matches `CurrentLoadedModel()`.

---

## Context-Length Probing (`context_probe.go`)

Binary-search the maximum `--max-model-len` that doesn't OOM. Full-version scope: sweep `{utilization} × {concurrency}`.

### Data Model

Stored under the model's registry entry, not in benchmarks.json (probe results describe model capability, not a benchmark).

```go
// In models/registry: extend ModelEntry with
type ContextProbe struct {
    LastProbed time.Time             `json:"last_probed"`
    Results    map[int]TPProbeResult `json:"results"` // key: tensor_parallel_size
}

type TPProbeResult struct {
    Utilization map[string]int `json:"utilization"` // "0.98" -> 32768
    Concurrency map[int]int    `json:"concurrency"` // 4 -> 16384
}
```

### Probe Orchestration

```go
type ProbeConfig struct {
    ModelID           string
    TPSize            int
    UtilizationLevels []float64 // default [0.98, 0.95, 0.90]
    ConcurrencyLevels []int     // default [1, 4, 8, 16]
    TimeoutPerTest    time.Duration // default 3 * time.Minute
    TestPort          int       // default 8001 (must differ from main vLLM port)
}

type ProbeResult struct {
    ModelID            string                  `json:"model_id"`
    TPSize             int                     `json:"tp_size"`
    UtilizationResults map[float64]int         `json:"utilization_results"`
    ConcurrencyResults map[int]int             `json:"concurrency_results"`
    Timestamp          time.Time               `json:"timestamp"`
    Warnings           []string                `json:"warnings,omitempty"`
}

func ProbeMaxContext(ctx context.Context, env ProbeEnv, cfg ProbeConfig, progress chan<- ProbeProgress) (*ProbeResult, error)
```

### Algorithm (per `(util, concurrency)` point)

1. **Exponential probe:** start at 1024, double until OOM or `max_position_embeddings` (capped at 131072). Record the last good value.
2. **Binary search** between `lastGood` and `firstBad`, granularity 256 (vLLM's KV cache block size). Round midpoints to the nearest 256.
3. Each `tryStartVLLM(...)` call starts a temporary vLLM process on `cfg.TestPort` and waits for one of:
   - **Ready:** `/v1/models` returns 200 → success
   - **OOM:** stderr / log matches the OOM regex (see below) → failure
   - **Timeout:** `cfg.TimeoutPerTest` elapsed → failure (conservative)
4. Stop the test process. Wait 5s for VRAM to free. Move on.

### tryStartVLLM (extends `internal/process/manager.go`)

The existing `process.Manager` already knows how to start vLLM. Extend it to support a "secondary" mode on `cfg.TestPort` so the probe can spawn vLLM without disturbing the main instance.

**Precondition:** main vLLM must be stopped before probing (otherwise GPU memory contention skews the probe). The API endpoint returns 409 if main vLLM is running.

### OOM Detection

Stream the test process's combined stdout/stderr and match these patterns:

```
torch.cuda.OutOfMemoryError
torch.OutOfMemoryError
CUDA out of memory
HIP out of memory
RuntimeError: out of memory
ValueError: The model's max seq len .* is larger than the maximum
not enough memory (case-insensitive)
KV cache .* cannot fit
```

When vLLM logs a helpful suggestion ("Try reducing max_model_len to N"), extract `N` and snap the upper bound to it to short-circuit further iterations.

### Progress Reporting

```go
type ProbeProgress struct {
    Phase           string  // "exponential" | "binary_search" | "swept_util" | "swept_concurrency"
    Utilization     float64
    Concurrency     int
    TestingContext  int
    Result          string  // "success" | "oom" | "timeout"
    ElapsedSeconds  float64
}
```

SSE-streamed at `/api/benchmarks/probe-context/{id}/progress`.

### Apply-Results UX

In the probe results display (and the Models page), each row has an "Apply" button that writes `max_model_len` + the matching `gpu_memory_utilization` or `max_num_seqs` back into the model's saved config. Probe values are advisory defaults; the user can override.

---

## Passive Timing Capture (`timing.go`, proxy hook)

### Capture Path

`internal/api/proxy.go` already proxies `/v1/*` requests to vLLM. Extend the proxy to capture timing from **non-streaming** chat completion responses (streaming responses are harder to instrument cheaply; skip them).

```go
// proxy.go (sketch)
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    var reqJSON struct {
        Stream bool   `json:"stream"`
        Model  string `json:"model"`
    }
    json.Unmarshal(body, &reqJSON)

    start := time.Now()
    rec := &timingRecorder{ResponseWriter: w}
    // ... forward the request to vLLM, recording the response body if non-streaming
    s.proxyTo(rec, r, body)

    if !reqJSON.Stream {
        var resp struct {
            Usage struct {
                PromptTokens     int `json:"prompt_tokens"`
                CompletionTokens int `json:"completion_tokens"`
            } `json:"usage"`
        }
        if json.Unmarshal(rec.body, &resp) == nil && resp.Usage.CompletionTokens > 0 {
            s.bench.AddTiming(benchmark.TimingSample{
                Timestamp:       start,
                ModelID:         reqJSON.Model,
                PromptTokens:    resp.Usage.PromptTokens,
                GenTokens:       resp.Usage.CompletionTokens,
                PromptTokPerSec: 0, // unknown without TTFT split
                GenTokPerSec:    float64(resp.Usage.CompletionTokens) / time.Since(start).Seconds(),
            })
        }
    }
}
```

### Store API

```go
type TimingSample struct {
    Timestamp       time.Time `json:"ts"`
    ModelID         string    `json:"model"`
    PromptTokens    int       `json:"prompt_n"`
    GenTokens       int       `json:"gen_n"`
    PromptTokPerSec float64   `json:"prompt_tps"`
    GenTokPerSec    float64   `json:"gen_tps"`
}

// On Store:
func (s *Store) AddTiming(sample TimingSample)
func (s *Store) RecentTimings(modelID string, n int) []TimingSample
func (s *Store) RunningAverage(modelID string) (avgGenTPS float64, count int, ok bool)
```

Per-model ring buffer of `maxTimingSamples=1000`. Running average is an exponential moving average (alpha=0.1) updated on every `AddTiming`, displayed once at least 10 samples have accumulated.

Timing samples are **not persisted** across restarts (in-memory only). That keeps `benchmarks.json` small and avoids file churn on every API request. If persistence becomes valuable later, write a separate `timings.jsonl` append-only log.

### Display

- **Dashboard** — per-model card showing `avg gen TPS`, sample count, last-updated.
- **Model cards (Models page)** — small badge near the "Loaded" indicator when ≥10 samples.
- **Benchmark comparison** — optional "include passive averages" checkbox that synthesizes pseudo-runs from the timing store for comparison plotting.

---

## API Endpoints

Mirrors llama-toolchest's split across `bench.go`, `bench_jobs.go`, `bench_about.go`, `bench_export.go`, plus a new `bench_probe.go`. Dual-mode (HTML/JSON via `HX-Request`) on every endpoint.

### Benchmark Runs

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/benchmarks` | List runs (newest first); query params: `model_id`, `preset`, `job_id`, `sort`, `order` |
| GET | `/api/benchmarks/{id}` | Single run detail |
| POST | `/api/benchmarks` | Start a single (ad-hoc) run. Body: `{model_id, preset, overrides?}` |
| DELETE | `/api/benchmarks/{id}` | Delete one |
| DELETE | `/api/benchmarks/batch-delete?ids=a,b,c` | Delete many |
| POST | `/api/benchmarks/{id}/cancel` | Cancel an in-flight run |
| GET | `/api/benchmarks/{id}/progress` | SSE stream of progress |
| GET | `/api/benchmarks/compare?ids=a,b,c` | Comparison view data (2–10 runs) |
| GET | `/api/benchmarks/export?ids=a,b&format=csv\|json` | Export selected runs |
| GET | `/api/benchmarks/about` | Disclosure modal data (presets, prompt text, benchy command) |
| GET | `/api/benchmarks/form` | New-benchmark form partial (HTML) |

**Validation on POST `/api/benchmarks`:**

- The model must be currently loaded in vLLM (`CurrentLoadedModel() == model_id`). If not, return 409 with `{error: "Start the model first", action: "/api/service/start"}`. Single-run benchmarks do not auto-load.
- Preset's largest prompt + gen must not exceed the model's configured `max_model_len`. If it would, the runner skips those points with a warning, but the run still starts.

### Benchmark Jobs

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/benchmark-jobs` | List jobs |
| GET | `/api/benchmark-jobs/{id}` | Job detail with cells |
| POST | `/api/benchmark-jobs` | Create + start a job. Body: `{name, description?, model_ids, presets, overrides?}` |
| POST | `/api/benchmark-jobs/{id}/cancel` | Cancel an in-flight job |
| POST | `/api/benchmark-jobs/{id}/retry-failed` | Retry failed cells (reuses the same job ID, bumps `Attempt`) |
| DELETE | `/api/benchmark-jobs/{id}?runs=cascade\|orphan` | Delete a job; cascade removes runs, orphan reassigns to AdhocJobID |
| GET | `/api/benchmark-jobs/form` | New-job form partial |

**Validation on POST `/api/benchmark-jobs`:**

- At least one model and one preset.
- All `model_ids` exist in the registry.
- All `presets` exist in `Presets()`.
- Returns 409 if another job is already running (`ErrJobAlreadyRunning`).

### Context Probe

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/benchmarks/probe-context` | Start a probe. Body: `{model_id, tp_size, utilization_levels?, concurrency_levels?}` |
| GET | `/api/benchmarks/probe-context/{id}/progress` | SSE progress |
| GET | `/api/benchmarks/probe-context/{model_id}` | Latest stored probe results for a model |

**Validation:**

- Main vLLM must be stopped (else 409). Probe spawns its own vLLM on `TestPort`.

### Passive Timing

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/benchmarks/timings/{model_id}` | Recent timing samples + running average |
| GET | `/api/benchmarks/timings` | Per-model running averages (all models with ≥10 samples) |

---

## UI

Port the llama-toolchest UI structure: job-grouped tables with expandable run rows, multi-select actions (compare / export / batch-delete), and the "About benchmarks" disclosure modal.

### Page Layout (`benchmarks.html`)

```
┌─ Benchmarks ───────────────────────────────────────┐
│ [+ New Run]  [+ New Job]  [Probe Max Context]  [?] │  ← Top actions; [?] opens About modal
├────────────────────────────────────────────────────┤
│ Active Job Progress (when running)                 │  ← Polled via hx-get + hx-trigger=load,every 2s
│   ▓▓▓▓▓░░░░░  3/8 cells   Job: "Quant compare"     │
│   Cell 3: Hermes-3-8B  internal-standard  rep 2/3  │
├────────────────────────────────────────────────────┤
│ Job-grouped runs list                              │
│                                                    │
│ ▼ "Quant compare" (batch, completed)  [delete]     │
│   ┌─[✓] Hermes-3-8B / internal-standard / 45.2 t/s │
│   ├─[ ] Hermes-3-8B-GPTQ / internal-standard / 62.5 │
│   └─[ ] Hermes-3-8B-AWQ / internal-standard / 58.9 │
│                                                    │
│ ▼ Ad-Hoc Runs                                       │
│   ┌─[ ] Hermes-3-8B / benchy-quick / 44.8 t/s      │
│   └─[ ] Llama-3.1-70B / internal-quick / 12.3 t/s  │
│                                                    │
│ [Compare selected] [Export selected ▾] [Delete selected]
└────────────────────────────────────────────────────┘
```

### Partial Templates

- `bench_job_list.html` — top-level: one `<tbody class="bench-row-group">` per job, header row + run rows
- `bench_run_row.html` — single run row with checkbox, summary stats, expand-on-click
- `bench_detail.html` — expanded view loaded via `hx-get="/api/benchmarks/{id}"` on row click. Shows:
  - Hardware snapshot (GPU name, VRAM, count)
  - Config snapshot (the 7 ConfigSnapshot fields)
  - vLLM version / image tag
  - Per-test-point table (rep, prompt tokens, gen tokens, TTFT, total ms, prompt-tps, gen-tps)
  - Summary stats card
  - For benchy runs: the full multi-concurrency `LlamaBenchy` results table + the disclosed command string
  - Warnings list
- `bench_compare.html` — side-by-side comparison view; horizontal-bar visualization per prompt length (inline SVG, no chart library); summary table with best-value highlighting
- `bench_progress.html` — active progress panel with phase indicator, cell progress, ETA
- `bench_about.html` — modal contents:
  - Preset table with computed durations
  - The `BenchPromptText` and `BenchPromptPrefixTemplate` shown verbatim
  - For each benchy preset: the exact `uvx llama-benchy …` command (via `FormatBenchyCommand`)
  - Link to llama-benchy upstream docs
- `bench_form.html` — new-run form: model selector (only loaded model selectable; others shown disabled with "start first" hint), preset radio buttons with descriptions
- `bench_job_form.html` — new-job form: multi-select for models (checkboxes), multi-select for presets, optional config overrides section
- `bench_probe.html` — probe form (tp_size, utilization levels, concurrency levels) + results display table

### htmx Patterns (copy from llama-toolchest)

- Auto-refresh active progress: `hx-get="/api/benchmark-jobs/{id}"` with `hx-trigger="every 2s"` while status=running
- SSE for per-cell live metrics: `hx-ext="sse" sse-connect="/api/benchmarks/{run-id}/progress" sse-swap="metric"`
- Selection state: vanilla JS scoped to the bench-runs container (port the `benchContainer(btn)` helper)
- Modal open/close: native `<dialog>` element, JS triggers `.showModal()` / `.close()`

---

## Container Changes

Both `Dockerfile.rocm` and `Dockerfile.cuda` need `uv` so `uvx llama-benchy` works in-process.

Add to each Dockerfile (after vLLM is installed):

```dockerfile
# Install uv for llama-benchy
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx
```

`uv` is statically-linked Rust and ~10 MB on disk — negligible image bloat.

The HF tokenizer cache should live on the `/data` volume so it persists across container recreates:

```yaml
# docker-compose.*.yml
environment:
  - HF_HOME=/data/cache/huggingface
```

The benchmark code already forwards `HF_HOME` to the `uvx` subprocess via `BenchyConfig.HFHome`.

---

## What to Copy Verbatim From llama-toolchest

These files / blocks transfer with only the package import path changed:

- `internal/benchmark/stats.go` — `ComputeSummary`, `BuildComparison`, `ComparisonData` (statistics math is identical)
- `internal/benchmark/benchy.go` — `BenchyConfig`, `LlamaBenchyResult`, `LlamaBenchyReport`, `BuildBenchyArgs`, `FormatBenchyCommand`, `summarizeBenchy`, `runLlamaBenchy` (engine-agnostic by design)
- The `BenchPromptText`, `BenchPromptPrefixTemplate`, `BenchPromptCharsPerToken` constants and `buildPrompt()` helper from `runner.go`
- `internal/benchmark/job.go` job/cell status constants, `DeleteDisposition`, `AdhocJobID`, `newAdhocJob` helper
- `internal/api/bench_about.go` — minor changes to remove llama.cpp build references; otherwise identical structure
- `internal/api/bench_export.go` — CSV writer is engine-agnostic; just adjusts column set for vLLM (drop llama-bench columns, add TTFT)
- Most htmx-driven template logic in `web/templates/benchmarks.html` and partials — selection state, modal handling, group toggles, export menu

## What to Adapt

These need real changes — same shape, different details:

- `internal/benchmark/benchmark.go` — Drop `Build*` fields; replace `ConfigSnapshot` with the vLLM-specific seven fields; add `VLLMVersion` / `ImageTag` fields on `BenchmarkRun`
- `internal/benchmark/runner.go` — Replace llama.cpp `timings` parsing with streaming SSE TTFT measurement; drop `unloadAllModels` / `ensureModelLoaded` (vLLM swap is process restart, owned by the Job Runner); keep warmup with retry
- `internal/benchmark/job_runner.go` — Replace `JobEnv.EnsureBuildActive` with `JobEnv.EnsureModelLoaded`; add the model-grouping reorder pass; drop build snapshot logic
- `internal/api/bench.go`, `bench_jobs.go` — Adjust for the new ConfigSnapshot, drop build inputs, validate against current loaded model
- `web/templates/benchmarks.html` and partials — Remove build-related columns / selectors; add vLLM version / image tag display; add probe-context button to the top action bar

## Entirely New

- `internal/benchmark/context_probe.go` — Context-length probing (no llama-toolchest equivalent)
- `internal/benchmark/timing.go` — `TimingSample`, ring buffer, running average (split out of `benchmark.go` for clarity; passive capture pattern partly existed in llama-toolchest but is rewritten for vLLM's response shape)
- `internal/api/bench_probe.go` — Probe API handlers and SSE
- Proxy hook in `internal/api/proxy.go` that feeds `TimingSample`s into the store
- `process.Manager` extension for spawning secondary vLLM on `TestPort`
- `Dockerfile.{rocm,cuda}` lines that install `uv`

---

## Implementation Order

Six steps, each independently mergeable. Each step ends with a working app — partial benchmarking is better than half-finished everything.

1. **Foundation** — `benchmark.go` types and `Store`. Persistence works; list endpoint returns empty list. Wire `/api/benchmarks` GET (returns empty), the benchmarks page renders. Smoke test: store loads cleanly on startup; deleted runs don't reappear.

2. **Internal runner** — `runner.go` streaming SSE loop, prompt generation, single-run ad-hoc path. Run `internal-quick` against a loaded model. Verify TTFT and gen-TPS values are sane against a known-good model. Display run on benchmarks page. Cancel works.

3. **Benchy runner + container `uv`** — Dockerfile adds `uv`. `benchy.go` ported. Run `benchy-quick` against the same model. Verify llama-benchy result parses and renders. Disclose the command in the About modal.

4. **Jobs** — `job.go`, `job_runner.go`, `JobEnv` adapter, job UI (job-grouped list, multi-select, new-job form). Test a 2-model × 2-preset matrix (4 cells) and verify model-grouping reorder: each model loads exactly once. Cancel mid-job leaves partial results.

5. **Passive timing capture** — Proxy hook + dashboard / model-card badges. Send a few real chat completions via the proxy, observe running average converge. Verify streaming requests don't capture (by design).

6. **Context probing** — `context_probe.go`, secondary-port spawn in `process.Manager`, OOM log detection, probe UI, "Apply" button on model config. Test on a model that easily fits at the default max and one that doesn't; verify probe converges to a reasonable value at each utilization level.

Step 1 unblocks the UI work. Steps 2 and 3 can land in either order. Step 4 depends on 2+3 (the runners it orchestrates). Steps 5 and 6 are independent of 4 and can land in parallel.

---

## Open Questions

- Should `internal-thorough`'s 8K prompt be skipped on models with `max_model_len < 8192`, or should we substitute a smaller test point? Current plan: skip + warning. Reconsider if many small-context models exist in the registry.
- For comparison view across runs with different model sizes (e.g. 8B vs 70B), should we normalize tokens/sec by parameter count? Probably no — that's analysis, not measurement; keep the raw numbers and let the user reason about them.
- Should benchy runs auto-start the model if it's not currently loaded? Today: no, return 409. The job path will load models automatically; the ad-hoc path stays explicit. Revisit if friction is high.

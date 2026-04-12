# Phase 6: Benchmarking

Benchmark inference performance with structured results, comparison across runs, context length probing, and passive timing capture from the proxy.

---

## Benchmark Engine Architecture (`internal/benchmark/`)

### File Organization

```
internal/benchmark/
  benchmark.go      -- Orchestration: run lifecycle, storage, preset management
  runner.go         -- HTTP-based test execution against /v1/chat/completions
  stats.go          -- Statistical computation (mean, min, max, median, percentiles, stddev)
  context_probe.go  -- Max context length probing (binary search for OOM boundary)
```

### Core Types

```
BenchmarkRun struct:
  ID              string             // UUID
  Status          RunStatus          // Pending, Running, Completed, Failed, Cancelled
  Preset          string             // "quick", "standard", "thorough", "custom"
  ModelID         string             // Registry model ID
  ModelConfig     ModelConfigSnapshot // Frozen copy of vLLM config at run time
  Hardware        HardwareSnapshot
  Config          RunConfig          // What to test
  Progress        RunProgress        // Current progress (for SSE)
  Results         []TestPoint        // Individual test results
  Summary         RunSummary         // Computed statistics
  VLLMBench       *VLLMBenchResults  // Optional: vllm CLI benchmark results
  StartedAt       time.Time
  CompletedAt     time.Time
  Error           string

RunStatus: Pending | Running | Completed | Failed | Cancelled

HardwareSnapshot struct:
  GPUs           []GPUInfo    // Name, VRAM total, VRAM used at start, device ID
  GPUCount       int
  GPUDriver      string       // ROCm version string
  ROCmVersion    string       // e.g. "6.4.0"
  CPUModel       string
  CPUCores       int
  RAMTotalGB     float64
  Hostname       string

GPUInfo struct:
  Index          int
  Name           string       // e.g. "AMD Radeon RX 9070 XT"
  VRAMTotalMB    int          // 32768
  VRAMUsedMB     int          // At snapshot time
  GFXVersion     string       // "gfx1201"

ModelConfigSnapshot struct:
  // Full copy of the model's vllm_config at run time
  // Plus key model metadata: architecture, param count, quant method
  ModelID              string
  Architecture         string
  ParamCountBillion    float64
  QuantMethod          string
  QuantBits            int
  Dtype                string
  MaxModelLen          int
  TensorParallelSize   int
  GPUMemoryUtilization float64
  EnforceEager         bool
  EnablePrefixCaching  bool
  KVCacheDtype         string
  MaxNumSeqs           int
  EnableAutoToolChoice bool
  ToolCallParser       string
  ExtraFlags           string

RunConfig struct:
  PromptLengths       []int    // Target prompt token counts
  GenerationTokens    int      // Max tokens to generate per test
  Repetitions         int      // Number of repetitions per prompt length
  WarmupRequests      int      // Warmup requests before timing (default 1)
  IncludeVLLMBench    bool     // Run vllm CLI benchmarks after HTTP tests
  VLLMBenchQPS        []float64 // QPS levels for vllm bench serve (e.g. [1.0, 4.0, 8.0])
  ConcurrentRequests  int      // For throughput testing (default 1 for latency tests)

RunProgress struct:
  Phase           string   // "warmup", "testing", "vllm_bench", "computing_stats"
  CurrentTest     int      // 1-indexed
  TotalTests      int      // prompt_lengths * repetitions
  CurrentPromptLen int
  CurrentRep      int
  LastMetrics     *TestPoint // Most recent test result (for live display)
  ElapsedSeconds  float64
  EstRemainingS   float64
```

---

## Benchmark Presets

### Quick (~30 seconds)

```
RunConfig{
  PromptLengths:    [512],
  GenerationTokens: 128,
  Repetitions:      1,
  WarmupRequests:   1,
  IncludeVLLMBench: false,
  ConcurrentRequests: 1,
}
```

Purpose: Fast sanity check. Is the model working? Roughly how fast is it? Use after model switch or config change.

Total test points: 1 prompt length x 1 rep = 1 test. Plus warmup.

### Standard (~3 minutes)

```
RunConfig{
  PromptLengths:    [128, 512, 2048],
  GenerationTokens: 128,
  Repetitions:      3,
  WarmupRequests:   2,
  IncludeVLLMBench: false,
  ConcurrentRequests: 1,
}
```

Purpose: Meaningful performance profile across different prompt lengths. Captures prefill speed scaling and generation stability.

Total test points: 3 prompt lengths x 3 reps = 9 tests. Plus warmup.

### Thorough (~15 minutes)

```
RunConfig{
  PromptLengths:    [128, 512, 2048, 8192],
  GenerationTokens: 256,
  Repetitions:      5,
  WarmupRequests:   3,
  IncludeVLLMBench: true,
  VLLMBenchQPS:     [1.0, 4.0, 8.0],
  ConcurrentRequests: 1,
}
```

Purpose: Comprehensive performance characterization. Includes long-context behavior and vLLM's built-in benchmark suite.

Total test points: 4 prompt lengths x 5 reps = 20 tests. Plus warmup, plus vllm bench throughput, plus vllm bench serve at 3 QPS levels.

Note: If `max_model_len` is < 8192, the 8192 prompt length is skipped (with a note in results). Similarly for 2048 if context is shorter.

### Custom

User specifies all RunConfig fields via the UI. Validation:
- `PromptLengths`: each must be > 0 and < `max_model_len - GenerationTokens`
- `GenerationTokens`: must be > 0 and < `max_model_len`
- `Repetitions`: 1-20
- `ConcurrentRequests`: 1-64 (higher values test throughput under load)

---

## Test Execution Flow

### Step 1: Snapshot Hardware State

Before any test runs:

```
func snapshotHardware() HardwareSnapshot:
  // Parse rocm-smi for GPU info
  //   rocm-smi --showproductname --showmeminfo vram --showdriverversion --json
  // Parse /proc/cpuinfo for CPU model + cores
  // Parse /proc/meminfo for total RAM
  // Get hostname
```

This snapshot is frozen for the entire run. If GPU VRAM changes during the run (it will, as KV cache grows), the snapshot reflects the state BEFORE testing.

### Step 2: Snapshot Model Config

Deep copy the current model's `vllm_config` and relevant metadata from the registry. This ensures the benchmark results are tied to the exact configuration, even if the user changes config later.

### Step 3: Warmup

```
func warmup(config RunConfig) error:
  for i := 0; i < config.WarmupRequests; i++:
    prompt := generatePrompt(256)  // Short prompt for warmup
    resp, err := sendCompletionRequest(prompt, 32)  // Short generation
    if err != nil:
      // Retry with backoff
      for retry := 0; retry < 3; retry++:
        time.Sleep(time.Duration(math.Pow(2, float64(retry))) * time.Second)
        resp, err = sendCompletionRequest(prompt, 32)
        if err == nil: break
      if err != nil:
        return fmt.Errorf("warmup failed after retries: %w", err)
    // Discard result -- just warming up KV cache allocation, graph compilation, etc.
```

Warmup is critical because:
- First request after model load triggers graph compilation (if not eager mode)
- KV cache block allocation happens lazily
- CUDA/HIP kernels are JIT-compiled on first invocation
- Triton kernels are compiled on first use

### Step 4: Test Matrix Execution

```
func runTests(config RunConfig, progress chan<- RunProgress) []TestPoint:
  results := []TestPoint{}
  totalTests := len(config.PromptLengths) * config.Repetitions
  testNum := 0

  for _, promptLen := range config.PromptLengths:
    // Skip if prompt length exceeds model's configured context
    if promptLen + config.GenerationTokens > modelConfig.MaxModelLen:
      // Record skip
      continue

    for rep := 0; rep < config.Repetitions; rep++:
      testNum++

      // Generate prompt of target token count
      prompt := generatePromptTokens(promptLen)

      // Report progress
      progress <- RunProgress{
        Phase:           "testing",
        CurrentTest:     testNum,
        TotalTests:      totalTests,
        CurrentPromptLen: promptLen,
        CurrentRep:      rep + 1,
      }

      // Execute single test
      point := runSingleTest(prompt, config.GenerationTokens)
      results = append(results, point)

      // Report live metrics
      progress <- RunProgress{
        ...
        LastMetrics: &point,
      }

  return results
```

### Single Test Execution

```
func runSingleTest(prompt string, maxTokens int) TestPoint:
  reqBody := map[string]interface{}{
    "model":       pm.modelID,
    "messages":    []map[string]string{{"role": "user", "content": prompt}},
    "max_tokens":  maxTokens,
    "temperature": 0.0,     // Deterministic for benchmarking
    "stream":      false,   // Non-streaming for accurate total timing
  }

  startTime := time.Now()
  resp, err := http.Post(
    fmt.Sprintf("http://localhost:%d/v1/chat/completions", vllmPort),
    "application/json",
    jsonEncode(reqBody),
  )
  totalTime := time.Since(startTime)

  // Parse response
  var result struct {
    Usage struct {
      PromptTokens     int `json:"prompt_tokens"`
      CompletionTokens int `json:"completion_tokens"`
    } `json:"usage"`
    Choices []struct {
      FinishReason string `json:"finish_reason"`
    } `json:"choices"`
  }
  json.Decode(resp.Body, &result)

  promptTokens := result.Usage.PromptTokens
  completionTokens := result.Usage.CompletionTokens
  totalMs := totalTime.Milliseconds()

  // Estimate TTFT (non-streaming doesn't give us true TTFT)
  // For accurate TTFT, we'd need streaming -- but streaming adds SSE overhead
  // Approximate: totalTime * (promptTokens / (promptTokens + completionTokens * genTimeRatio))
  // Better: run a separate TTFT test with streaming and 1 max_token
  // For now: record totalMs and note TTFT is estimated

  // Calculate throughput
  promptTPS := float64(promptTokens) / (float64(totalMs) / 1000.0)  // Rough: includes generation time
  genTPS := float64(completionTokens) / (float64(totalMs) / 1000.0) // Rough: includes prefill time

  // Better decomposition if we assume prefill dominates for large prompts:
  // Estimate prefill time ≈ totalTime - (completionTokens / expected_gen_tps)
  // But we don't know expected_gen_tps yet. Use simpler metrics.

  return TestPoint{
    PromptTokens:     promptTokens,
    CompletionTokens: completionTokens,
    TargetPromptLen:  len(prompt),  // What we asked for (may differ from actual token count)
    TotalTimeMs:      totalMs,
    FinishReason:     result.Choices[0].FinishReason,
    Timestamp:        startTime,
  }
```

**TTFT measurement strategy:**

For accurate TTFT, run a separate streaming request:
```
func measureTTFT(prompt string) time.Duration:
  reqBody := map[string]interface{}{
    "model":      pm.modelID,
    "messages":   []map[string]string{{"role": "user", "content": prompt}},
    "max_tokens": 1,       // Only need first token
    "stream":     true,
    "temperature": 0.0,
  }

  startTime := time.Now()
  resp, _ := http.Post(...)
  scanner := bufio.NewScanner(resp.Body)
  for scanner.Scan():
    line := scanner.Text()
    if strings.HasPrefix(line, "data: ") && line != "data: [DONE]":
      return time.Since(startTime)  // Time to first SSE data event
  return 0  // Should not reach here
```

Include TTFT measurement in Standard and Thorough presets (one TTFT test per prompt length, before the non-streaming repetitions).

### Prompt Generation

Generate prompts that tokenize to approximately the target token count.

```
func generatePromptTokens(targetTokens int) string:
  // Strategy: use a known tokens-per-word ratio for English text
  // Average English: ~1.3 tokens per word (varies by tokenizer)
  // Generate extra, then trim

  // Use a corpus of varied English text (not just repeated words --
  // repeated text may compress differently in some tokenizers)
  // Store a ~100KB text corpus embedded in the binary

  // Approach:
  // 1. Start with targetTokens * 0.75 words (conservative estimate)
  // 2. Tokenize using a simple whitespace heuristic (actual tokenization happens server-side)
  // 3. Adjust if needed (we accept +-10% accuracy; actual token count is recorded from response)

  words := corpus.RandomWords(int(float64(targetTokens) * 0.75))
  return strings.Join(words, " ")
```

The actual token count is recorded from vLLM's response (`usage.prompt_tokens`), so the prompt generation doesn't need to be perfectly accurate -- we just need to be in the right ballpark.

### Step 5: Optional vLLM CLI Benchmarks

When `IncludeVLLMBench` is true, run vLLM's built-in benchmark tools after the HTTP tests.

#### vllm bench throughput

```bash
vllm bench throughput \
  --model /data/models/<path> \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 100 \
  --dtype auto \
  --quantization <quant> \
  --tensor-parallel-size <tp>
```

Parse stdout for:
- Total throughput (tokens/sec)
- Request throughput (requests/sec)
- Average latency

#### vllm bench serve

Benchmarks against a running vLLM server at multiple QPS levels:

```bash
# For each QPS level:
vllm bench serve \
  --model <model_name> \
  --host localhost \
  --port 8000 \
  --dataset-name sharegpt \
  --num-prompts 50 \
  --request-rate <qps>
```

Parse stdout for per-QPS metrics:
- Mean/median/p99 TTFT
- Mean/median/p99 TPOT (time per output token)
- Mean/median/p99 ITL (inter-token latency)
- Throughput (output tokens/sec)

**Edge case:** `vllm bench serve` requires a running vLLM server. Since we're running benchmarks, the server should already be running. But verify health before starting.

**Edge case:** ShareGPT dataset download. `vllm bench` may need to download the ShareGPT dataset on first run. Handle this gracefully (detect download, report progress).

```
VLLMBenchResults struct:
  Throughput *VLLMThroughputResult
  Serve      []VLLMServeResult      // One per QPS level

VLLMThroughputResult struct:
  TotalTokensPerSec   float64
  RequestsPerSec      float64
  AvgLatencyMs        float64
  NumPrompts          int
  InputLen            int
  OutputLen           int

VLLMServeResult struct:
  QPS                float64
  NumPrompts         int
  MeanTTFTMs         float64
  MedianTTFTMs       float64
  P99TTFTMs          float64
  MeanTPOTMs         float64
  MedianTPOTMs       float64
  P99TPOTMs          float64
  MeanITLMs          float64
  MedianITLMs        float64
  P99ITLMs           float64
  OutputTokensPerSec float64
  CompletedRequests  int
  FailedRequests     int
```

### Step 6: Compute Summary Statistics

```
func computeSummary(results []TestPoint) RunSummary:
  // Group by prompt length
  grouped := groupByPromptLen(results)

  perPromptLen := []PromptLenSummary{}
  for promptLen, points := range grouped:
    totalTimes := extractField(points, "TotalTimeMs")
    genTokens := extractField(points, "CompletionTokens")
    promptTokens := extractField(points, "PromptTokens")

    // Compute gen_tps for each point
    genTPS := []float64{}
    for _, p := range points:
      genTPS = append(genTPS, float64(p.CompletionTokens) / (float64(p.TotalTimeMs) / 1000.0))

    perPromptLen = append(perPromptLen, PromptLenSummary{
      PromptTokens: promptLen,
      NumTests:     len(points),
      TotalTime:    computeStats(totalTimes),
      GenTPS:       computeStats(genTPS),
      // TTFT stats if measured separately
    })

  // Overall summary across all prompt lengths
  allGenTPS := []float64{} // All gen_tps values across all tests
  // ...

  return RunSummary{
    PerPromptLen: perPromptLen,
    Overall: OverallSummary{
      TotalTests:     len(results),
      TotalDuration:  ...,
      GenTPS:         computeStats(allGenTPS),
      AvgGenTPS:      mean(allGenTPS),
      BestGenTPS:     max(allGenTPS),
      WorstGenTPS:    min(allGenTPS),
    },
  }
```

### Step 7: Store Results

Append to `/data/config/benchmarks.json`:

```json
{
  "runs": [
    { /* BenchmarkRun */ },
    { /* BenchmarkRun */ }
  ],
  "schema_version": 1
}
```

File is loaded into memory at startup. Runs are appended and the whole file is rewritten. For a typical user this file will stay small (tens to low hundreds of runs). If it grows large, consider per-run files in a `/data/config/benchmarks/` directory (future optimization).

---

## Statistics Computation (`internal/benchmark/stats.go`)

Copy and adapt from llama-toolchest's `stats.go`.

```
Stats struct:
  Mean    float64
  Min     float64
  Max     float64
  Median  float64  // p50
  P95     float64
  P99     float64
  Stddev  float64
  Count   int

func computeStats(values []float64) Stats:
  if len(values) == 0:
    return Stats{}

  sort.Float64s(values)
  n := len(values)

  sum := 0.0
  for _, v := range values:
    sum += v

  mean := sum / float64(n)

  // Variance
  sumSqDiff := 0.0
  for _, v := range values:
    diff := v - mean
    sumSqDiff += diff * diff
  variance := sumSqDiff / float64(n)  // Population variance (not sample)
  stddev := math.Sqrt(variance)

  return Stats{
    Mean:   mean,
    Min:    values[0],
    Max:    values[n-1],
    Median: percentile(values, 50),
    P95:    percentile(values, 95),
    P99:    percentile(values, 99),
    Stddev: stddev,
    Count:  n,
  }

func percentile(sorted []float64, p float64) float64:
  if len(sorted) == 0:
    return 0
  if len(sorted) == 1:
    return sorted[0]

  rank := (p / 100.0) * float64(len(sorted) - 1)
  lower := int(math.Floor(rank))
  upper := int(math.Ceil(rank))
  if lower == upper:
    return sorted[lower]

  // Linear interpolation
  frac := rank - float64(lower)
  return sorted[lower] + frac * (sorted[upper] - sorted[lower])
```

---

## Context Length Probing (`internal/benchmark/context_probe.go`)

Inspired by kyuz0's `find_max_context.py`. Iteratively find the maximum `--max-model-len` that doesn't OOM for a given model + GPU config.

### Probe Algorithm

```
func ProbeMaxContext(modelID string, config ProbeConfig) ProbeResult:
  model := registry.Get(modelID)

  // Starting point: model's max_position_embeddings
  maxPossible := model.HFConfig.MaxPositionEmbeddings
  // Cap at something reasonable to avoid hours of testing
  if maxPossible > 131072:
    maxPossible = 131072

  // Test at multiple gpu-memory-utilization levels
  results := []ProbeUtilResult{}
  for _, util := range config.UtilizationLevels:  // e.g. [0.98, 0.95, 0.90]
    maxCtx := binarySearchMaxContext(modelID, maxPossible, util, config.TPSize)
    results = append(results, ProbeUtilResult{
      GPUMemoryUtilization: util,
      MaxContextLength:     maxCtx,
    })

  // Test at different concurrency levels for the best utilization
  bestUtil := results[0]  // Highest utilization
  concurrencyResults := []ProbeConcurrencyResult{}
  for _, concurrency := range config.ConcurrencyLevels:  // e.g. [1, 4, 8, 16]
    maxCtx := binarySearchMaxContext(
      modelID, bestUtil.MaxContextLength, bestUtil.GPUMemoryUtilization,
      config.TPSize, WithMaxNumSeqs(concurrency),
    )
    concurrencyResults = append(concurrencyResults, ProbeConcurrencyResult{
      MaxNumSeqs:       concurrency,
      MaxContextLength: maxCtx,
    })

  return ProbeResult{
    ModelID:            modelID,
    TPSize:             config.TPSize,
    UtilizationResults: results,
    ConcurrencyResults: concurrencyResults,
    Timestamp:          time.Now(),
  }
```

### Binary Search Implementation

```
func binarySearchMaxContext(modelID string, maxPossible int, util float64, tp int) int:
  low := 256      // Minimum useful context
  high := maxPossible
  lastGood := 0

  // Step 1: Quick exponential probe to find approximate upper bound
  // Start from a known-good baseline and double until OOM
  test := 1024
  for test <= high:
    if tryStartVLLM(modelID, test, util, tp):
      lastGood = test
      test *= 2
    else:
      high = test
      break

  // Step 2: Binary search between lastGood and high
  low = lastGood
  for high - low > 256:  // 256-token granularity
    mid := (low + high) / 2
    // Round to nearest 256 (vLLM allocates KV cache in blocks)
    mid = (mid / 256) * 256

    if tryStartVLLM(modelID, mid, util, tp):
      low = mid
      lastGood = mid
    else:
      high = mid

  return lastGood
```

### tryStartVLLM Implementation

```
func tryStartVLLM(modelID string, maxModelLen int, util float64, tp int) bool:
  // Build config with test parameters
  testConfig := model.VLLMConfig
  testConfig.MaxModelLen = maxModelLen
  testConfig.GPUMemoryUtilization = util
  testConfig.TensorParallelSize = tp

  // Start vLLM with test config
  // Use a temporary process manager (don't disturb the main one)
  tmpPM := NewProcessManager(testPort)  // Use different port (8001)
  err := tmpPM.Start(modelID, testConfig)

  // Wait for either:
  // - Ready state (success)
  // - OOM in logs (failure)
  // - Startup timeout (failure)

  outcome := tmpPM.WaitForOutcome(3 * time.Minute)

  // Stop the test process regardless
  tmpPM.Stop()
  // Wait for GPU memory to be freed
  time.Sleep(5 * time.Second)

  switch outcome:
  case Ready:
    return true
  case OOM:
    return false
  case Timeout:
    return false  // Treat timeout as failure (conservative)
  case OtherError:
    return false
```

### OOM Detection from vLLM Logs

Parse vLLM stdout/stderr for OOM indicators:

```
OOM patterns to match:
  "torch.cuda.OutOfMemoryError"
  "torch.OutOfMemoryError"
  "CUDA out of memory"
  "HIP out of memory"
  "Cannot allocate"
  "RuntimeError: out of memory"
  "ValueError: The model's max seq len" ... "is larger than the maximum"
  "not enough memory" (case insensitive)
```

Also look for vLLM's helpful suggestions:
```
  "Try reducing max_model_len" -- extract suggested value if present
  "gpu_memory_utilization is set to" ... "but" ... "is required"
```

If vLLM suggests a specific max length, use it to narrow the binary search faster.

### Probe Config

```
ProbeConfig struct:
  TPSize              int        // 1 or 2
  UtilizationLevels   []float64  // Default: [0.98, 0.95, 0.90]
  ConcurrencyLevels   []int      // Default: [1, 4, 8, 16]
  TimeoutPerTest      time.Duration  // Default: 3 minutes
  TestPort            int        // Default: 8001 (avoid conflicting with main vLLM)
}
```

### Saving Probe Results

Store in the model's registry entry:

```json
{
  "context_probe": {
    "last_probed": "2026-04-12T10:30:00Z",
    "tp1": {
      "util_098": 32768,
      "util_095": 28672,
      "util_090": 24576,
      "concurrency_1": 32768,
      "concurrency_4": 16384,
      "concurrency_8": 8192,
      "concurrency_16": 4096
    },
    "tp2": {
      "util_098": 65536,
      "util_095": 57344,
      "util_090": 49152,
      "concurrency_1": 65536,
      "concurrency_4": 32768,
      "concurrency_8": 16384,
      "concurrency_16": 8192
    }
  }
}
```

These verified values can be used as smart defaults: when the user sets TP and max_num_seqs, auto-suggest the probed max context length.

---

## Results Data Model

### TestPoint (per individual test)

```
TestPoint struct:
  Index            int       // 1-indexed within run
  PromptTokens     int       // Actual from vLLM response
  CompletionTokens int       // Actual from vLLM response
  TargetPromptLen  int       // What we requested
  TTFTMs           float64   // Time to first token (if measured via streaming)
  TotalTimeMs      int64     // Wall clock time for full request
  PromptTPS        float64   // Prompt tokens / total time (rough prefill speed)
  GenTPS           float64   // Completion tokens / total time (rough gen speed)
  FinishReason     string    // "stop", "length"
  Timestamp        time.Time
  Error            string    // Non-empty if this test failed
}
```

### RunSummary

```
RunSummary struct:
  PerPromptLen []PromptLenSummary
  Overall      OverallSummary
  TTFTSummary  *TTFTSummary  // nil if TTFT not measured

PromptLenSummary struct:
  PromptTokens int
  NumTests     int
  TotalTime    Stats   // ms
  GenTPS       Stats   // tokens/sec
  PromptTPS    Stats   // tokens/sec
  TTFTMs       *Stats  // nil if not measured for this prompt length

OverallSummary struct:
  TotalTests      int
  TotalDurationMs int64
  GenTPS          Stats
  PromptTPS       Stats
  AvgGenTPS       float64   // Convenience: mean of GenTPS across all tests
  BestGenTPS      float64   // Convenience: max
  WorstGenTPS     float64   // Convenience: min

TTFTSummary struct:
  PerPromptLen map[int]Stats  // prompt_length -> TTFT stats
  Overall      Stats
```

### Comparison Data

For cross-run comparison, compute normalized scores:

```
ComparisonEntry struct:
  RunID          string
  ModelID        string
  ModelName      string
  QuantMethod    string
  QuantBits      int
  TPSize         int
  MaxModelLen    int
  AvgGenTPS      float64
  BestGenTPS     float64
  AvgTTFTMs      float64   // 0 if not measured
  // Per-prompt-len breakdowns for chart
  PromptLenData  map[int]PromptLenCompare

PromptLenCompare struct:
  AvgGenTPS   float64
  AvgTotalMs  float64
  AvgTTFTMs   float64
}
```

---

## Passive Timing Capture

The proxy (Phase 5) captures timing data from non-streaming `/v1/chat/completions` responses.

### Storage

```
PassiveTiming struct:
  Timestamp        time.Time
  ModelID          string
  PromptTokens     int
  CompletionTokens int
  TotalTimeMs      int64
  HasToolCalls     bool

PassiveTimingStore struct:
  mu       sync.RWMutex
  entries  []PassiveTiming     // Recent entries (ring buffer, capacity 10000)
  perModel map[string]*RunningAverage

RunningAverage struct:
  ModelID          string
  Count            int
  AvgGenTPS        float64
  AvgPromptTPS     float64
  AvgTotalMs       float64
  LastUpdated      time.Time
}
```

### Updating Running Averages

```
func (s *PassiveTimingStore) Add(t PassiveTiming):
  s.mu.Lock()
  defer s.mu.Unlock()

  // Add to ring buffer
  s.entries.Add(t)

  // Update running average for model
  avg, ok := s.perModel[t.ModelID]
  if !ok:
    avg = &RunningAverage{ModelID: t.ModelID}
    s.perModel[t.ModelID] = avg

  genTPS := float64(t.CompletionTokens) / (float64(t.TotalTimeMs) / 1000.0)

  // Exponential moving average (alpha = 0.1 for smoothing)
  alpha := 0.1
  if avg.Count == 0:
    avg.AvgGenTPS = genTPS
  else:
    avg.AvgGenTPS = alpha * genTPS + (1 - alpha) * avg.AvgGenTPS

  avg.Count++
  avg.LastUpdated = time.Now()
```

### Display

- **Dashboard:** Show per-model passive metrics card with avg gen TPS, request count.
- **Model cards:** Show passive average gen TPS badge if available (> 10 samples).
- **Benchmark comparison:** Allow including passive averages as a baseline in comparisons.

---

## Benchmark API

### GET /api/benchmarks

List all benchmark runs.

**Query params:**
- `model_id` -- Filter by model (optional)
- `preset` -- Filter by preset (optional)
- `sort` -- "date" (default), "gen_tps", "model"
- `order` -- "desc" (default), "asc"

**Response (JSON):**
```json
{
  "runs": [
    {
      "id": "abc-123",
      "status": "completed",
      "preset": "standard",
      "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
      "model_display_name": "Hermes 3 Llama 3.1 8B",
      "quant_method": "none",
      "started_at": "2026-04-12T10:00:00Z",
      "completed_at": "2026-04-12T10:03:15Z",
      "summary": {
        "overall": {
          "avg_gen_tps": 45.2,
          "best_gen_tps": 52.1,
          "worst_gen_tps": 38.7,
          "total_tests": 9
        }
      }
    }
  ],
  "total": 15
}
```

**HTML response:** Summary card list with model name, date, avg gen TPS, preset badge.

### POST /api/benchmarks

Start a new benchmark run.

**Request body:**
```json
{
  "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
  "preset": "standard",
  "custom_config": null
}
```

Or with custom config:
```json
{
  "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
  "preset": "custom",
  "custom_config": {
    "prompt_lengths": [256, 1024, 4096],
    "generation_tokens": 256,
    "repetitions": 3,
    "warmup_requests": 2,
    "include_vllm_bench": true,
    "vllm_bench_qps": [1.0, 4.0],
    "concurrent_requests": 1
  }
}
```

**Validation:**
- Model must be currently loaded in vLLM (check service status).
- If model is not loaded, return 409 with message "Start the model first via /api/service/start".
- Custom config values must be within allowed ranges.

**Response:** 202 Accepted with run ID.
```json
{
  "id": "abc-123",
  "status": "pending",
  "message": "Benchmark started, subscribe to /api/benchmarks/abc-123/progress for updates"
}
```

### GET /api/benchmarks/{id}

Full results for a completed run.

**Response:** The complete `BenchmarkRun` struct serialized as JSON. Includes all test points, summaries, hardware snapshot, model config snapshot.

**HTML response:** Detailed results page with tables and inline charts.

### GET /api/benchmarks/{id}/progress

SSE stream of benchmark progress.

**Events:**
```
event: progress
data: {"phase":"testing","current_test":3,"total_tests":9,"current_prompt_len":512,"current_rep":1,"elapsed_seconds":45.2,"est_remaining_s":90.5}

event: metric
data: {"index":3,"prompt_tokens":515,"completion_tokens":128,"total_time_ms":2850,"gen_tps":44.9}

event: phase
data: {"phase":"vllm_bench","message":"Running vllm bench throughput..."}

event: complete
data: {"id":"abc-123","status":"completed","summary":{"overall":{"avg_gen_tps":45.2}}}

event: error
data: {"id":"abc-123","status":"failed","error":"Connection refused -- is vLLM still running?"}
```

### DELETE /api/benchmarks/{id}

Delete a single benchmark run.

**Response:** 204 No Content on success, 404 if not found.

### DELETE /api/benchmarks/batch-delete

Delete multiple runs at once.

**Request body:**
```json
{
  "ids": ["abc-123", "def-456", "ghi-789"]
}
```

**Response:**
```json
{
  "deleted": 3,
  "not_found": 0
}
```

### GET /api/benchmarks/compare?ids=1,2,3

Comparison view data for multiple runs.

**Query params:**
- `ids` -- Comma-separated run IDs (2-10 runs)

**Response:**
```json
{
  "runs": [
    {
      "id": "abc-123",
      "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
      "model_display_name": "Hermes 3 8B",
      "quant_method": "none",
      "tp_size": 1,
      "max_model_len": 8192,
      "avg_gen_tps": 45.2,
      "best_gen_tps": 52.1,
      "avg_ttft_ms": 120.5,
      "prompt_len_data": {
        "128": {"avg_gen_tps": 52.1, "avg_total_ms": 2450},
        "512": {"avg_gen_tps": 45.2, "avg_total_ms": 2830},
        "2048": {"avg_gen_tps": 38.7, "avg_total_ms": 3310}
      }
    },
    {
      "id": "def-456",
      "model_id": "NousResearch/Hermes-3-Llama-3.1-8B-GPTQ",
      "model_display_name": "Hermes 3 8B GPTQ",
      "quant_method": "gptq",
      "tp_size": 1,
      "max_model_len": 16384,
      "avg_gen_tps": 62.5,
      "best_gen_tps": 71.3,
      "avg_ttft_ms": 95.2,
      "prompt_len_data": {
        "128": {"avg_gen_tps": 71.3, "avg_total_ms": 1790},
        "512": {"avg_gen_tps": 62.5, "avg_total_ms": 2050},
        "2048": {"avg_gen_tps": 53.8, "avg_total_ms": 2380}
      }
    }
  ],
  "common_prompt_lens": [128, 512, 2048]
}
```

`common_prompt_lens` lists prompt lengths that all selected runs have in common (for apples-to-apples chart).

### GET /api/benchmarks/export?id=1

CSV export of a single run's test points.

**Response headers:**
```
Content-Type: text/csv
Content-Disposition: attachment; filename="benchmark-abc123-hermes-3-8b-2026-04-12.csv"
```

**CSV columns:**
```
test_index,prompt_tokens,completion_tokens,target_prompt_len,ttft_ms,total_time_ms,prompt_tps,gen_tps,finish_reason,timestamp
1,130,128,128,,2450,53.1,52.2,stop,2026-04-12T10:00:15Z
2,131,128,128,,2480,52.8,51.6,stop,2026-04-12T10:00:18Z
...
```

### POST /api/benchmarks/probe-context

Start context length probing.

**Request body:**
```json
{
  "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
  "tp_size": 1,
  "utilization_levels": [0.98, 0.95, 0.90],
  "concurrency_levels": [1, 4, 8, 16]
}
```

**Validation:**
- Model does NOT need to be currently loaded (probing starts its own vLLM instances on a different port).
- But main vLLM should be stopped to avoid GPU memory contention. If running, return 409 "Stop main vLLM process before context probing".

**Response:** 202 Accepted with probe ID.

**Progress via SSE:** `GET /api/benchmarks/probe-context/{id}/progress`
```
event: progress
data: {"phase":"binary_search","utilization":0.98,"testing_context":16384,"result":"success"}

event: progress
data: {"phase":"binary_search","utilization":0.98,"testing_context":24576,"result":"oom"}

event: complete
data: {"model_id":"...","results":{...}}
```

---

## Benchmarks Page UI (`web/templates/benchmarks.html`)

### Layout

Three sections, top to bottom:

1. **New Benchmark Panel** (collapsible, expanded by default if no runs exist)
2. **Active Benchmark Progress** (shown only when a benchmark is running)
3. **Results List** (always shown)

### New Benchmark Panel

- **Model selector** -- Dropdown of loaded model only (or all enabled models with "start first" note for unloaded ones). Show current model as default.
- **Preset selector** -- Radio buttons: Quick, Standard, Thorough, Custom
  - Each preset shows estimated duration and test count
  - Selecting Custom reveals additional fields:
    - **Prompt lengths** -- Multi-select chips or comma-separated input: 128, 256, 512, 1024, 2048, 4096, 8192, 16384. Pre-filter to only show values <= model's max_model_len.
    - **Generation tokens** -- Number input, default 128. Range: 1 to max_model_len.
    - **Repetitions** -- Number input, default 3. Range: 1-20.
    - **Include vLLM bench** -- Toggle. Shows sub-options for QPS levels.
    - **QPS levels** -- Multi-select: 1.0, 2.0, 4.0, 8.0, 16.0
- **Run button** -- `hx-post="/api/benchmarks"`. Disabled if no model loaded. Shows confirmation with estimated duration.
- **Context probe button** -- Separate button: "Probe Max Context". Opens probe config modal (TP selector, utilization levels). Requires main vLLM to be stopped.

### Active Benchmark Progress

Shown only when `status == "running"`. Driven by SSE from `/api/benchmarks/{id}/progress`.

- **Progress bar** -- `currentTest / totalTests` as percentage.
- **Phase indicator** -- "Warmup", "Testing (3/9)", "Running vllm bench throughput", "Computing statistics".
- **Live metrics table** -- Updated after each test point:

  | Prompt Tokens | Completion Tokens | Total Time (ms) | Gen TPS |
  |---|---|---|---|
  | 128 | 128 | 2450 | 52.2 |
  | 128 | 128 | 2480 | 51.6 |
  | 512 | 128 | 2830 | 45.2 |

- **Elapsed / Estimated remaining** -- "1:15 elapsed, ~2:00 remaining"
- **Cancel button** -- `hx-post="/api/benchmarks/{id}/cancel"`. Stops the current run.

### Results List

Card-style list of completed benchmark runs. Each card shows:

- **Model name** + quant badge (e.g. "Hermes 3 8B" with "GPTQ 4-bit" badge)
- **Date** -- "April 12, 2026 10:00 AM"
- **Preset** badge -- Quick / Standard / Thorough / Custom
- **Key metrics** -- Avg gen TPS, best gen TPS in large text
- **Expandable details** -- Click to reveal full results (htmx `hx-get="/api/benchmarks/{id}"`)

**Sorting controls:** Date (default), Gen TPS, Model name.

**Filter controls:** By model (dropdown), by preset (checkboxes).

**Selection checkboxes:** For comparison and batch delete.

**Actions row:**
- **Compare selected** -- Button, enabled when 2-10 runs selected. Opens comparison view.
- **Delete selected** -- Button with confirmation.

### Detailed Results View

Expanded within a card or as a separate panel. Shows:

**Configuration snapshot:**
- Model, quant, TP, context length, gpu_memory_utilization, enforce_eager, kv_cache_dtype, tool config
- Hardware: GPU name, VRAM, ROCm version

**Per-test-point table:**

| # | Prompt Tokens | Completion Tokens | TTFT (ms) | Total Time (ms) | Gen TPS | Finish |
|---|---|---|---|---|---|---|
| 1 | 130 | 128 | - | 2450 | 52.2 | stop |
| 2 | 131 | 128 | - | 2480 | 51.6 | stop |
| 3 | 515 | 128 | - | 2830 | 45.2 | stop |

**Summary statistics per prompt length:**

| Prompt Length | Avg Gen TPS | Min | Max | Median | P95 | Stddev |
|---|---|---|---|---|---|---|
| 128 | 51.9 | 51.6 | 52.2 | 51.9 | 52.2 | 0.3 |
| 512 | 45.0 | 44.5 | 45.5 | 45.0 | 45.5 | 0.4 |
| 2048 | 38.5 | 37.8 | 39.2 | 38.5 | 39.1 | 0.6 |

**Mini chart:** Simple horizontal bar chart showing avg gen TPS per prompt length. Built with inline SVG or a lightweight chart helper (no heavy chart library -- keep it Pico CSS compatible). Or use ASCII-style bars:

```
128 tokens:  ████████████████████████████████████████████  52.2 tps
512 tokens:  ██████████████████████████████████████         45.2 tps
2048 tokens: ██████████████████████████████                 38.7 tps
```

**vLLM bench results** (if included):
- Throughput: X tokens/sec, Y requests/sec
- Per-QPS serve results table:
  | QPS | Mean TTFT | P99 TTFT | Mean TPOT | Throughput | Failed |
  |---|---|---|---|---|---|
  | 1.0 | 95ms | 120ms | 22ms | 45.2 tok/s | 0 |
  | 4.0 | 110ms | 180ms | 25ms | 160.5 tok/s | 0 |
  | 8.0 | 250ms | 450ms | 35ms | 280.1 tok/s | 2 |

**CSV export button:** `<a href="/api/benchmarks/{id}/export">Export CSV</a>`

### Comparison View

Side-by-side display of 2-10 selected runs. Driven by `/api/benchmarks/compare?ids=...`.

**Header row:** One column per run, showing model name, quant type, key config.

**Bar chart per prompt length:**
For each common prompt length, horizontal bars showing avg gen TPS:
```
128 tokens:
  Hermes 3 8B FP16:  ████████████████████████████████████  52.2
  Hermes 3 8B GPTQ:  █████████████████████████████████████████████████  71.3
  Hermes 3 8B AWQ:   ███████████████████████████████████████████████  68.9

512 tokens:
  Hermes 3 8B FP16:  ████████████████████████████████  45.2
  Hermes 3 8B GPTQ:  ██████████████████████████████████████████  62.5
  Hermes 3 8B AWQ:   ████████████████████████████████████████  59.1
```

**Summary comparison table:**

| Metric | Hermes 3 FP16 | Hermes 3 GPTQ | Hermes 3 AWQ |
|---|---|---|---|
| Avg Gen TPS | 45.2 | 62.5 | 59.1 |
| Best Gen TPS | 52.2 | 71.3 | 68.9 |
| Avg TTFT | 120ms | 95ms | 100ms |
| Context Length | 8192 | 16384 | 16384 |
| VRAM Used | 16.1 GB | 4.5 GB | 4.3 GB |
| TP Size | 1 | 1 | 1 |

Highlight the best value in each row (green background).

### Context Probe Results Section

Shown on the models page (Phase 4) and linked from benchmarks page.

Display probed results as a table:

**TP=1:**
| GPU Mem Util | Max Context |
|---|---|
| 0.98 | 32768 |
| 0.95 | 28672 |
| 0.90 | 24576 |

| Concurrency (max_num_seqs) | Max Context |
|---|---|
| 1 | 32768 |
| 4 | 16384 |
| 8 | 8192 |
| 16 | 4096 |

"Apply" button next to each row: sets `max_model_len` and `gpu_memory_utilization`/`max_num_seqs` in the model's config to the probed values.

---

## What to Copy from llama-toolchest vs Adapt

### Copy Directly

- **`stats.go`** -- Statistical computation functions. Identical math, same Go code.
- **SSE progress pattern** -- Same fan-out writer, same event format, same htmx-sse.js consumption.
- **Benchmark list UI layout** -- Card-style list with expand/collapse. Same htmx patterns.
- **CSV export endpoint pattern** -- Same Content-Disposition header, same CSV writer.
- **Progress bar component** -- Same HTML/CSS, different SSE source.

### Adapt (Same Pattern, Different Details)

- **`runner.go`** -- llama-toolchest sends to llama.cpp's `/v1/chat/completions`. Here we send to vLLM's. Same endpoint, but:
  - vLLM returns `usage.prompt_tokens` and `usage.completion_tokens` (llama.cpp may use different field names)
  - vLLM's streaming format may have slight differences in chunk structure
  - Temperature 0.0 may behave differently (vLLM might need `temperature: 0.01` to avoid greedy-mode edge cases)
- **`benchmark.go`** -- Orchestration is similar but:
  - Add vLLM CLI bench integration (new)
  - Add TTFT measurement via streaming (new)
  - Hardware snapshot uses rocm-smi instead of (or in addition to) nvidia-smi
- **Benchmark presets** -- Same concept, different default values (vLLM typically handles larger contexts)
- **Results display** -- Same table/card layout, add columns for TTFT and vLLM-specific metrics

### Entirely New

- **`context_probe.go`** -- No equivalent in llama-toolchest. Entirely new feature.
- **vLLM CLI bench integration** -- `vllm bench throughput` and `vllm bench serve` parsing.
- **Passive timing from proxy** -- llama-toolchest had this but it was simpler (no tool call detection, different response format).
- **Comparison view** -- llama-toolchest had basic comparison; enhance with per-prompt-length breakdown charts.

---

## File Layout Summary

```
internal/benchmark/
  benchmark.go      -- BenchmarkRun struct, RunConfig, presets, orchestration, storage
  runner.go         -- sendCompletionRequest, runSingleTest, measureTTFT, generatePrompt,
                       parseVLLMBenchOutput
  stats.go          -- Stats struct, computeStats, percentile (copy from llama-toolchest)
  context_probe.go  -- ProbeMaxContext, binarySearchMaxContext, tryStartVLLM, OOM detection

internal/api/
  bench.go          -- HTTP handlers for /api/benchmarks/* endpoints

web/templates/
  benchmarks.html   -- Full benchmarks page
  partials/
    benchmark_card.html       -- Summary card for results list
    benchmark_detail.html     -- Expanded results view
    benchmark_compare.html    -- Side-by-side comparison
    benchmark_progress.html   -- Active benchmark progress panel
    context_probe.html        -- Probe results display
```

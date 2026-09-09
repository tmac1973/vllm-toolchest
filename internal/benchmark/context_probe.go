package benchmark

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Probe constants. The 256-token granularity matches vLLM's KV-cache block
// size: smaller increments produce identical OOM outcomes.
const (
	defaultProbeMinContext  = 1024
	defaultProbeMaxContext  = 131072 // 128K; long enough for any realistic deployment
	probeContextGranularity = 256
	defaultProbeTestTimeout = 3 * time.Minute
	defaultProbeTestPort    = 8001
)

// Default sweeps when the caller doesn't override.
var (
	DefaultProbeUtilizationLevels = []float64{0.98, 0.95, 0.90}
	DefaultProbeConcurrencyLevels = []int{1, 4, 8, 16}
)

// ProbeAttemptOutcome is the result of one tryStartVLLM call.
type ProbeAttemptOutcome string

const (
	ProbeReady   ProbeAttemptOutcome = "ready"
	ProbeOOM     ProbeAttemptOutcome = "oom"
	ProbeTimeout ProbeAttemptOutcome = "timeout"
	ProbeOther   ProbeAttemptOutcome = "other"
)

// ProbeAttempt is one trial of the probe — start vLLM with these
// parameters and see whether it becomes ready or OOMs.
type ProbeAttempt struct {
	MaxModelLen          int
	GPUMemoryUtilization float64
	TensorParallelSize   int
	MaxNumSeqs           int
}

// ProbeAttemptResult bundles the outcome and any suggested upper bound
// vLLM hinted at (e.g. "Try reducing max_model_len to N").
type ProbeAttemptResult struct {
	Outcome         ProbeAttemptOutcome
	SuggestedMaxLen int    // 0 when vLLM didn't suggest one
	Detail          string // short human-readable
}

// ProbeEnv abstracts everything the probe needs from the outside world.
// The api layer implements this against process.Manager + the model
// registry.
type ProbeEnv interface {
	// TrySpawn starts vLLM on the secondary port with the given attempt,
	// waits for one of {ready, OOM, timeout}, stops the process, and
	// returns the outcome.
	TrySpawn(ctx context.Context, modelID string, attempt ProbeAttempt) ProbeAttemptResult

	// ModelMaxPositionEmbeddings returns the model's hf-config max
	// position embeddings, used as the absolute ceiling. Returns 0 when
	// unknown, in which case the probe uses defaultProbeMaxContext.
	ModelMaxPositionEmbeddings(modelID string) int
}

// ProbeConfig is the input to a probe run.
type ProbeConfig struct {
	ModelID           string
	TPSize            int
	UtilizationLevels []float64
	ConcurrencyLevels []int
	MinContext        int
	MaxContext        int
	TimeoutPerTest    time.Duration
}

// ProbeResult is the final aggregated outcome of a probe.
type ProbeResult struct {
	ModelID            string         `json:"model_id"`
	TPSize             int            `json:"tp_size"`
	UtilizationResults map[string]int `json:"utilization_results"` // util-string → max context
	ConcurrencyResults map[int]int    `json:"concurrency_results"` // max_num_seqs → max context
	Timestamp          time.Time      `json:"timestamp"`
	Warnings           []string       `json:"warnings,omitempty"`
}

// ProbeProgress is one SSE event during a probe run.
type ProbeProgress struct {
	Phase          string  `json:"phase"` // "exponential" | "binary_search" | "phase_done" | "complete"
	Utilization    float64 `json:"utilization,omitempty"`
	Concurrency    int     `json:"concurrency,omitempty"`
	TestingContext int     `json:"testing_context"`
	Outcome        string  `json:"outcome,omitempty"` // ProbeAttemptOutcome values
	Best           int     `json:"best,omitempty"`    // largest known-good context so far
	ElapsedSec     float64 `json:"elapsed_sec,omitempty"`
	Detail         string  `json:"detail,omitempty"`
}

// RunProbe executes a full probe. Sends progress updates to progress
// (closes on exit). Returns the aggregated ProbeResult.
func RunProbe(ctx context.Context, env ProbeEnv, cfg ProbeConfig, progress chan<- ProbeProgress) (*ProbeResult, error) {
	if cfg.ModelID == "" {
		return nil, fmt.Errorf("ProbeConfig.ModelID is required")
	}
	if cfg.TPSize <= 0 {
		cfg.TPSize = 1
	}
	if len(cfg.UtilizationLevels) == 0 {
		cfg.UtilizationLevels = DefaultProbeUtilizationLevels
	}
	if len(cfg.ConcurrencyLevels) == 0 {
		cfg.ConcurrencyLevels = DefaultProbeConcurrencyLevels
	}
	if cfg.MinContext <= 0 {
		cfg.MinContext = defaultProbeMinContext
	}
	if cfg.MaxContext <= 0 {
		cfg.MaxContext = defaultProbeMaxContext
	}
	if cfg.TimeoutPerTest == 0 {
		cfg.TimeoutPerTest = defaultProbeTestTimeout
	}

	// Apply model's hf-config ceiling if known.
	if hfMax := env.ModelMaxPositionEmbeddings(cfg.ModelID); hfMax > 0 && hfMax < cfg.MaxContext {
		cfg.MaxContext = hfMax
	}

	defer func() {
		if progress != nil {
			close(progress)
		}
	}()

	// Sort utilization descending so the first (highest) result becomes
	// the ceiling for the concurrency sweep.
	utilLevels := append([]float64(nil), cfg.UtilizationLevels...)
	sort.Sort(sort.Reverse(sort.Float64Slice(utilLevels)))

	concLevels := append([]int(nil), cfg.ConcurrencyLevels...)
	sort.Ints(concLevels)

	result := &ProbeResult{
		ModelID:            cfg.ModelID,
		TPSize:             cfg.TPSize,
		UtilizationResults: map[string]int{},
		ConcurrencyResults: map[int]int{},
		Timestamp:          time.Now(),
	}

	start := time.Now()
	send := func(p ProbeProgress) {
		if progress == nil {
			return
		}
		p.ElapsedSec = time.Since(start).Seconds()
		select {
		case progress <- p:
		default:
		}
	}

	highestUtilMax := 0

	// Utilization sweep: at each level, find the max context that doesn't OOM.
	for _, util := range utilLevels {
		if ctx.Err() != nil {
			result.Warnings = append(result.Warnings, "probe cancelled before completion")
			return result, ctx.Err()
		}
		maxCtx := bisect(ctx, env, cfg, util, 1, send)
		result.UtilizationResults[utilKey(util)] = maxCtx
		if maxCtx > highestUtilMax {
			highestUtilMax = maxCtx
		}
		send(ProbeProgress{
			Phase:       "phase_done",
			Utilization: util,
			Best:        maxCtx,
		})
	}

	// Concurrency sweep at the highest utilization level: higher
	// max_num_seqs eats KV cache budget, shrinking the max context.
	highestUtil := utilLevels[0]
	for _, conc := range concLevels {
		if ctx.Err() != nil {
			result.Warnings = append(result.Warnings, "probe cancelled before concurrency sweep finished")
			return result, ctx.Err()
		}
		maxCtx := bisect(ctx, env, cfg, highestUtil, conc, send)
		result.ConcurrencyResults[conc] = maxCtx
		send(ProbeProgress{
			Phase:       "phase_done",
			Utilization: highestUtil,
			Concurrency: conc,
			Best:        maxCtx,
		})
	}

	send(ProbeProgress{Phase: "complete", Best: highestUtilMax})
	return result, nil
}

// bisect performs the exponential-then-binary-search to find the
// largest known-good context length at (util, concurrency).
func bisect(ctx context.Context, env ProbeEnv, cfg ProbeConfig, util float64, concurrency int, send func(ProbeProgress)) int {
	tryFn := func(maxLen int) (ProbeAttemptOutcome, int) {
		attemptCtx, cancel := context.WithTimeout(ctx, cfg.TimeoutPerTest)
		defer cancel()
		res := env.TrySpawn(attemptCtx, cfg.ModelID, ProbeAttempt{
			MaxModelLen:          maxLen,
			GPUMemoryUtilization: util,
			TensorParallelSize:   cfg.TPSize,
			MaxNumSeqs:           concurrency,
		})
		send(ProbeProgress{
			Phase:          "binary_search",
			Utilization:    util,
			Concurrency:    concurrency,
			TestingContext: maxLen,
			Outcome:        string(res.Outcome),
			Detail:         res.Detail,
		})
		return res.Outcome, res.SuggestedMaxLen
	}

	low := cfg.MinContext
	high := cfg.MaxContext
	lastGood := 0

	// Step 1: exponential probe upward from MinContext. Stops when we
	// hit OOM (sets high) or reach the ceiling (returns directly).
	test := roundDown(low, probeContextGranularity)
	if test < probeContextGranularity {
		test = probeContextGranularity
	}
expLoop:
	for test <= high {
		if ctx.Err() != nil {
			return lastGood
		}
		outcome, suggested := tryFn(test)
		switch outcome {
		case ProbeReady:
			lastGood = test
			next := test * 2
			if next > high {
				return lastGood
			}
			test = next
		case ProbeOOM:
			high = test - probeContextGranularity
			if suggested > 0 && suggested < high {
				high = roundDown(suggested, probeContextGranularity)
			}
			low = lastGood + probeContextGranularity
			break expLoop
		case ProbeTimeout, ProbeOther:
			high = test - probeContextGranularity
			low = lastGood + probeContextGranularity
			break expLoop
		}
	}

	// Step 2: binary search between lastGood and high.
	for high >= low+probeContextGranularity {
		if ctx.Err() != nil {
			return lastGood
		}
		mid := roundDown((low+high)/2, probeContextGranularity)
		if mid <= lastGood {
			mid = lastGood + probeContextGranularity
		}
		if mid > high {
			break
		}
		outcome, suggested := tryFn(mid)
		switch outcome {
		case ProbeReady:
			lastGood = mid
			low = mid + probeContextGranularity
		case ProbeOOM:
			high = mid - probeContextGranularity
			if suggested > 0 && suggested < high {
				high = roundDown(suggested, probeContextGranularity)
			}
		case ProbeTimeout, ProbeOther:
			high = mid - probeContextGranularity
		}
	}

	return lastGood
}

func roundDown(v, granularity int) int {
	if granularity <= 0 {
		return v
	}
	return (v / granularity) * granularity
}

func utilKey(u float64) string {
	return fmt.Sprintf("%.2f", u)
}

// OOM log patterns. Compiled once at package init so the probe doesn't
// re-compile on every log line.
var oomPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)torch\.cuda\.OutOfMemoryError`),
	regexp.MustCompile(`(?i)torch\.OutOfMemoryError`),
	regexp.MustCompile(`(?i)CUDA out of memory`),
	regexp.MustCompile(`(?i)HIP out of memory`),
	regexp.MustCompile(`(?i)RuntimeError:\s*out of memory`),
	regexp.MustCompile(`(?i)not enough memory`),
	regexp.MustCompile(`(?i)KV cache.*cannot fit`),
	regexp.MustCompile(`(?i)ValueError:\s*The model's max seq len .* is larger than the maximum`),
}

// vLLM occasionally suggests "Try reducing `max_model_len` to N" or
// similar. The probe uses this to short-circuit further iterations.
var suggestedMaxLenPattern = regexp.MustCompile(`(?i)max[_ ]model[_ ]len.*?(\d{3,7})`)

// DetectOOM scans a log buffer for OOM patterns. Returns true and any
// suggested max_model_len value found, or false otherwise.
func DetectOOM(logs string) (oom bool, suggestedMaxLen int) {
	for _, re := range oomPatterns {
		if re.MatchString(logs) {
			oom = true
			break
		}
	}
	if !oom {
		return false, 0
	}
	// Take the smallest suggested value (most conservative).
	matches := suggestedMaxLenPattern.FindAllStringSubmatch(logs, -1)
	for _, m := range matches {
		var n int
		if _, err := fmt.Sscanf(m[1], "%d", &n); err == nil && n > 0 {
			if suggestedMaxLen == 0 || n < suggestedMaxLen {
				suggestedMaxLen = n
			}
		}
	}
	return oom, suggestedMaxLen
}

// ClassifyLogs is a convenience for ProbeEnv implementations: given the
// captured stdout/stderr of a probe attempt that didn't reach ready,
// classify it as OOM, generic error, or "other" so the algorithm can
// react appropriately.
func ClassifyLogs(logs string) ProbeAttemptResult {
	oom, suggested := DetectOOM(logs)
	if oom {
		return ProbeAttemptResult{
			Outcome:         ProbeOOM,
			SuggestedMaxLen: suggested,
			Detail:          "OOM detected in vLLM logs",
		}
	}
	if strings.Contains(strings.ToLower(logs), "error") {
		return ProbeAttemptResult{Outcome: ProbeOther, Detail: "vLLM logged error"}
	}
	return ProbeAttemptResult{Outcome: ProbeOther, Detail: "process exited without ready signal"}
}

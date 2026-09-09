// Package benchmark drives inference performance measurement against the
// local vLLM instance. It exposes two sources: an internal Go HTTP loop
// (streaming SSE for TTFT) and a shell-out to `llama-benchy` via uvx for
// engine-agnostic comparison. Runs are grouped into BenchmarkJobs that
// sweep a {models} × {presets} matrix.
package benchmark

import (
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/monitor"
)

// Run-level status values.
const (
	StatusRunning   = "running"
	StatusCompleted = "completed"
	StatusFailed    = "failed"
)

// BenchmarkRun is one complete benchmark execution.
type BenchmarkRun struct {
	ID        string    `json:"id"`
	JobID     string    `json:"job_id,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	Status    string    `json:"status"`
	Error     string    `json:"error,omitempty"`

	ModelID   string  `json:"model_id"`
	ModelName string  `json:"model_name"`
	Quant     string  `json:"quant"`
	SizeGB    float64 `json:"size_gb"`

	Config ConfigSnapshot `json:"config"`

	VLLMVersion string `json:"vllm_version,omitempty"`
	ImageTag    string `json:"image_tag,omitempty"`

	GPUs []GPUSnapshot `json:"gpus"`

	Preset       string `json:"preset"`
	PromptTokens []int  `json:"prompt_tokens"`
	GenTokens    int    `json:"gen_tokens"`

	// SweepValues are the swept parameters this run was measured at. A run
	// records its own conditions so a number never has to be traced back
	// through the job that produced it.
	SweepValues map[string]string `json:"sweep_values,omitempty"`

	Results     []BenchmarkResult   `json:"results,omitempty"`
	Summary     *BenchmarkSummary   `json:"summary,omitempty"`
	LlamaBenchy []LlamaBenchyResult `json:"llama_benchy,omitempty"`

	BenchyCommand string `json:"benchy_command,omitempty"`

	Warnings []string `json:"warnings,omitempty"`

	// ProgressDetail is transient — set during a run so a polling endpoint
	// can surface stage/cell info without subscribing to SSE.
	ProgressDetail string `json:"progress_detail,omitempty"`

	DurationMs int64 `json:"duration_ms,omitempty"`
}

// ConfigSnapshot freezes the vLLM configuration that produced a run. Only
// fields that meaningfully affect perf are captured; the full vllm_config
// stays on the model registry entry.
type ConfigSnapshot struct {
	MaxModelLen          int     `json:"max_model_len"`
	TensorParallelSize   int     `json:"tensor_parallel_size"`
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization"`
	KVCacheDtype         string  `json:"kv_cache_dtype"`
	EnforceEager         bool    `json:"enforce_eager"`
	Dtype                string  `json:"dtype"`
	QuantMethod          string  `json:"quant_method,omitempty"`
	MaxNumSeqs           int     `json:"max_num_seqs,omitempty"`
}

// GPUSnapshot captures GPU identity at run time. VRAM-used isn't recorded
// here because it changes as the KV cache grows during the run; this is
// the static fingerprint of the device.
type GPUSnapshot struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	VRAMTotalMB int    `json:"vram_total_mb"`
}

// BenchmarkResult is one test point inside a run.
type BenchmarkResult struct {
	PromptTokens    int     `json:"prompt_tokens"`
	GenTokens       int     `json:"gen_tokens"`
	Repetition      int     `json:"repetition"`
	PromptTokPerSec float64 `json:"prompt_tok_per_sec"`
	GenTokPerSec    float64 `json:"gen_tok_per_sec"`
	TTFTMs          float64 `json:"ttft_ms"`
	TotalMs         float64 `json:"total_ms"`
}

// BenchmarkSummary aggregates a run's results into a single row of numbers.
type BenchmarkSummary struct {
	AvgPromptTokPerSec float64 `json:"avg_prompt_tok_per_sec"`
	AvgGenTokPerSec    float64 `json:"avg_gen_tok_per_sec"`
	AvgTTFTMs          float64 `json:"avg_ttft_ms"`
	MinGenTokPerSec    float64 `json:"min_gen_tok_per_sec"`
	MaxGenTokPerSec    float64 `json:"max_gen_tok_per_sec"`
}

// LlamaBenchyMetric is the per-metric shape llama-benchy emits.
type LlamaBenchyMetric struct {
	Mean   float64   `json:"mean"`
	Std    float64   `json:"std"`
	Values []float64 `json:"values,omitempty"`
}

// LlamaBenchyResult is one (concurrency × prompt_size × response_size)
// point from a llama-benchy report.
type LlamaBenchyResult struct {
	Concurrency           int  `json:"concurrency"`
	ContextSize           int  `json:"context_size"`
	PromptSize            int  `json:"prompt_size"`
	ResponseSize          int  `json:"response_size"`
	IsContextPrefillPhase bool `json:"is_context_prefill_phase"`

	PPThroughput      *LlamaBenchyMetric `json:"pp_throughput,omitempty"`
	PPReqThroughput   *LlamaBenchyMetric `json:"pp_req_throughput,omitempty"`
	TGThroughput      *LlamaBenchyMetric `json:"tg_throughput,omitempty"`
	TGReqThroughput   *LlamaBenchyMetric `json:"tg_req_throughput,omitempty"`
	PeakThroughput    *LlamaBenchyMetric `json:"peak_throughput,omitempty"`
	PeakReqThroughput *LlamaBenchyMetric `json:"peak_req_throughput,omitempty"`
	TTFR              *LlamaBenchyMetric `json:"ttfr,omitempty"`
	EstPPT            *LlamaBenchyMetric `json:"est_ppt,omitempty"`
	E2ETTFT           *LlamaBenchyMetric `json:"e2e_ttft,omitempty"`
}

// GPUSnapshotsFromMetrics maps the monitor's GPU info onto the snapshot
// shape the benchmark store persists.
func GPUSnapshotsFromMetrics(m monitor.Metrics) []GPUSnapshot {
	snaps := make([]GPUSnapshot, len(m.GPU))
	for i, g := range m.GPU {
		snaps[i] = GPUSnapshot{
			Index:       g.Index,
			Name:        g.Name,
			VRAMTotalMB: g.VRAMTotalMB,
		}
	}
	return snaps
}

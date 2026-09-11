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
// fields that move the numbers are captured; the full vllm_config stays on the
// model registry entry.
//
// The first block does two jobs: it is also what a job's ConfigOverrides can
// set, where a zero value means "use the model's saved setting". Everything
// after it is recorded only. The launch already carries those settings, since
// a job starts the model from its whole saved config; putting one in the
// overlay is a separate decision, and needs the pointer treatment
// ConfigOverrides uses, because a bool here cannot say "explicitly off".
type ConfigSnapshot struct {
	MaxModelLen          int     `json:"max_model_len"`
	TensorParallelSize   int     `json:"tensor_parallel_size"`
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization"`
	KVCacheDtype         string  `json:"kv_cache_dtype"`
	EnforceEager         bool    `json:"enforce_eager"`
	Dtype                string  `json:"dtype"`
	QuantMethod          string  `json:"quant_method,omitempty"`
	MaxNumSeqs           int     `json:"max_num_seqs,omitempty"`

	// ProfileName is the model's active config profile when the run started,
	// and ProfileModified says the live config had been edited away from it.
	// Both, rather than one string: a run labelled "long-ctx" whose numbers
	// came from something else is worse than an unlabelled run.
	ProfileName     string `json:"profile_name,omitempty"`
	ProfileModified bool   `json:"profile_modified,omitempty"`

	// SpeculativeConfig is the largest factor this used to miss: MTP drafting
	// can move generation throughput by half again or more, and two runs that
	// differed only in it were indistinguishable in the history.
	SpeculativeConfig string `json:"speculative_config,omitempty"`
	// AttentionBackend swaps the decode kernel outright, and is the setting
	// most likely to differ between two machines' runs of the "same" config.
	AttentionBackend string `json:"attention_backend,omitempty"`
	// CompilationConfig trims the CUDA-graph capture ladder. A batch size off
	// the ladder runs eager for that step, so this shapes the latency curve.
	CompilationConfig string `json:"compilation_config,omitempty"`
	// EnablePrefixCaching collapses TTFT on any preset that reuses a prompt
	// prefix, and MambaCacheMode is what lets it work at all on a hybrid, so
	// the pair only means something together.
	EnablePrefixCaching bool   `json:"enable_prefix_caching,omitempty"`
	MambaCacheMode      string `json:"mamba_cache_mode,omitempty"`
	// Chunked prefill and its token budget decide whether a long prompt
	// monopolises a step: prompt throughput and inter-token latency under
	// concurrency both move with them, in opposite directions.
	EnableChunkedPrefill bool `json:"enable_chunked_prefill,omitempty"`
	MaxNumBatchedTokens  int  `json:"max_num_batched_tokens,omitempty"`
	// KVCacheMemory pins the pool, so a pinned run is not comparable with an
	// unpinned one at the same gpu_memory_utilization.
	KVCacheMemory int64 `json:"kv_cache_memory,omitempty"`
	// DisableAsyncScheduling costs decode throughput, and some speculative
	// configs require it, so it moves in step with one.
	DisableAsyncScheduling bool `json:"disable_async_scheduling,omitempty"`
	// Quantization is the method the engine was told to use, as against
	// QuantMethod, which is what the checkpoint is. Forcing marlin over gptq
	// is a kernel swap on identical weights.
	Quantization string `json:"quantization,omitempty"`
	// ExtraFlags can change anything at all, so it is recorded verbatim and
	// never interpreted.
	ExtraFlags string `json:"extra_flags,omitempty"`
}

// ProfileLabel is the profile a run's config came from, as a comparison should
// read it. A config edited away from its profile says so rather than claiming
// the name.
func (c ConfigSnapshot) ProfileLabel() string {
	switch {
	case c.ProfileName == "":
		return ""
	case c.ProfileModified:
		return c.ProfileName + " (edited)"
	default:
		return c.ProfileName
	}
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

package benchmark

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"log/slog"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/monitor"
)

// ErrJobAlreadyActive is returned when a job submit is attempted while
// another run or job is already in flight on this Service.
var ErrJobAlreadyActive = errors.New("a benchmark run or job is already in progress")

// JobEnv is the integration point the job runner needs from outside the
// benchmark package. The api layer implements it against the Server so
// this package stays free of imports from models / process / config.
type JobEnv interface {
	// ResolveModel returns registry data for one model: HF repo id (used
	// as --tokenizer for llama-benchy), display name, the model
	// identifier vLLM responds to, and the saved per-model config that
	// ConfigOverrides overlay on top of.
	ResolveModel(modelID string) (ModelInfo, error)

	// CurrentLoadedModel returns the HF repo id of the model vLLM is
	// currently serving, or "" if no model is loaded.
	CurrentLoadedModel() string

	// EnsureModelLoaded swaps vLLM to the target model+config if it isn't
	// already serving it. Blocks until /health returns 200 or ctx expires.
	// May take minutes for large models.
	EnsureModelLoaded(ctx context.Context, modelID string, cfg ConfigSnapshot) error

	// CurrentMetrics returns the latest GPU metrics for snapshotting onto
	// each run.
	CurrentMetrics() monitor.Metrics

	// VLLMURL returns the base URL the runner should target (e.g.
	// http://localhost:8000).
	VLLMURL() string

	// HFToken / HFCacheDir are forwarded to llama-benchy so the tokenizer
	// download is authenticated and cached across cells.
	HFToken() string
	HFCacheDir() string

	// VLLMVersion returns the running vLLM version string, or "" if
	// unavailable.
	VLLMVersion() string
}

// ModelInfo bundles registry data needed to build a RunnerConfig for one
// cell. Lives in the benchmark package so the JobEnv contract is local.
type ModelInfo struct {
	HFRepoID    string
	Quant       string
	SizeGB      float64
	DisplayName string
	ServedName  string         // the identifier returned by /v1/models
	Config      ConfigSnapshot // saved baseline; ConfigOverrides overlay
}

// applyOverrides returns a ConfigSnapshot with non-nil override values
// substituted in. Pointer fields on ConfigOverrides distinguish "use
// default" (nil) from "use this value" (set).
func applyOverrides(base ConfigSnapshot, ov *ConfigOverrides) ConfigSnapshot {
	if ov == nil {
		return base
	}
	out := base
	if ov.MaxModelLen != nil {
		out.MaxModelLen = *ov.MaxModelLen
	}
	if ov.TensorParallelSize != nil {
		out.TensorParallelSize = *ov.TensorParallelSize
	}
	if ov.GPUMemoryUtilization != nil {
		out.GPUMemoryUtilization = *ov.GPUMemoryUtilization
	}
	if ov.KVCacheDtype != nil {
		out.KVCacheDtype = *ov.KVCacheDtype
	}
	if ov.EnforceEager != nil {
		out.EnforceEager = *ov.EnforceEager
	}
	if ov.Dtype != nil {
		out.Dtype = *ov.Dtype
	}
	if ov.MaxNumSeqs != nil {
		out.MaxNumSeqs = *ov.MaxNumSeqs
	}
	return out
}

// ExpandCells expands a job's {ModelIDs} × {Presets} matrix into Cells in
// model-grouped order: every cell for the first model, then every cell
// for the second, etc. Stable preset order within each group. Called by
// the api layer when persisting a new job; the runner re-checks the
// order at execution time but expects this layout.
func ExpandCells(modelIDs, presets []string, sweeps []SweepAxis) []JobCell {
	combos := SweepCombinations(sweeps)
	cells := make([]JobCell, 0, len(modelIDs)*len(presets)*len(combos))
	for _, m := range modelIDs {
		// Sweep values before presets: a sweep value change costs an engine
		// reload and a preset change does not, so this ordering loads the
		// engine once per (model, combination) instead of once per cell.
		for _, combo := range combos {
			for _, p := range presets {
				cell := JobCell{
					ModelID: m,
					Preset:  p,
					Status:  CellStatusPending,
					Attempt: 0,
				}
				if len(combo) > 0 {
					cell.SweepValues = combo
				}
				cells = append(cells, cell)
			}
		}
	}
	return cells
}

// runJob executes one job. Cells are walked in their stored order; the
// runner relies on ExpandCells (or retry-failed's cell ordering) to have
// grouped cells so each engine configuration loads exactly once per job.
//
// The function blocks until the job finishes or ctx is cancelled. It
// updates the job in the store after every cell so a crash leaves a
// partially-progressed but valid record.
func (s *Service) runJob(ctx context.Context, job *BenchmarkJob) {
	job.Status = JobStatusRunning
	job.StartedAt = time.Now()
	_ = s.store.SaveJob(*job)

	// Group cell indices by (model, sweep values) in stored order.
	//
	// Grouping by model alone was enough when every cell of a model shared one
	// engine configuration. Sweeps break that: each axis is a launch
	// parameter — context length, tensor-parallel size, KV cache dtype — that
	// vLLM cannot change without a reload. Cells sharing a key share a
	// configuration and are measured on one load.
	type group struct {
		modelID string
		sweep   map[string]string
		indices []int
	}
	var groupOrder []string
	groups := map[string]*group{}
	for i, cell := range job.Cells {
		// Skip cells already completed (retry-failed leaves them).
		if cell.Status == CellStatusCompleted {
			continue
		}
		key := cell.ModelID + "\x00" + SweepKey(cell.SweepValues)
		g, seen := groups[key]
		if !seen {
			g = &group{modelID: cell.ModelID, sweep: cell.SweepValues}
			groups[key] = g
			groupOrder = append(groupOrder, key)
		}
		g.indices = append(g.indices, i)
	}

	for _, key := range groupOrder {
		g := groups[key]
		if ctx.Err() != nil {
			s.markPendingCells(job, JobStatusCanceled)
			break
		}

		info, err := s.env.ResolveModel(g.modelID)
		if err != nil {
			s.failCells(job, g.indices, "resolve model: "+err.Error())
			_ = s.store.SaveJob(*job)
			continue
		}

		overrides, err := ApplySweep(job.Overrides, g.sweep)
		if err != nil {
			s.failCells(job, g.indices, "sweep values: "+err.Error())
			_ = s.store.SaveJob(*job)
			continue
		}
		cfg := applyOverrides(info.Config, overrides)

		// Load model (no-op if already serving it with this configuration).
		loadCtx, loadCancel := context.WithTimeout(ctx, 15*time.Minute)
		err = s.env.EnsureModelLoaded(loadCtx, g.modelID, cfg)
		loadCancel()
		if err != nil {
			s.failCells(job, g.indices, "load model: "+err.Error())
			_ = s.store.SaveJob(*job)
			continue
		}

		for _, idx := range g.indices {
			if ctx.Err() != nil {
				job.Cells[idx].Status = CellStatusSkipped
				continue
			}
			s.runCell(ctx, job, idx, info, cfg)
			_ = s.store.SaveJob(*job)
		}
	}

	job.Status = jobFinalStatus(job, ctx.Err() != nil)
	job.FinishedAt = time.Now()
	_ = s.store.SaveJob(*job)
}

// runCell executes one cell as a benchmark run.
func (s *Service) runCell(ctx context.Context, job *BenchmarkJob, idx int, info ModelInfo, cfg ConfigSnapshot) {
	cell := &job.Cells[idx]
	cell.Status = CellStatusRunning
	cell.Attempt++
	_ = s.store.SaveJob(*job)

	preset := GetPreset(cell.Preset)
	if preset.Name != cell.Preset {
		cell.Status = CellStatusFailed
		cell.Error = "unknown preset: " + cell.Preset
		return
	}

	run := BenchmarkRun{
		ID:        newID(),
		JobID:     job.ID,
		CreatedAt: time.Now(),
		Status:    StatusRunning,

		ModelID:   info.HFRepoID,
		ModelName: info.DisplayName,
		Quant:     info.Quant,
		SizeGB:    info.SizeGB,

		Config:      cfg,
		VLLMVersion: s.env.VLLMVersion(),
		GPUs:        GPUSnapshotsFromMetrics(s.env.CurrentMetrics()),

		Preset:       preset.Name,
		PromptTokens: preset.PromptTokens,
		GenTokens:    preset.GenTokens,
	}
	cell.BenchmarkRunID = run.ID
	if err := s.store.Save(run); err != nil {
		cell.Status = CellStatusFailed
		cell.Error = "save run: " + err.Error()
		return
	}

	cfgRC := RunnerConfig{
		Run:         run,
		Preset:      preset,
		VLLMURL:     s.env.VLLMURL(),
		ServedName:  info.ServedName,
		MaxModelLen: cfg.MaxModelLen,
		HFRepoID:    info.HFRepoID,
		HFToken:     s.env.HFToken(),
		HFHome:      s.env.HFCacheDir(),
	}

	// Drain runner progress events; we don't fan them out per-cell here
	// (jobs don't have an SSE subscriber today), but draining the channel
	// is mandatory or the runner blocks.
	progress := make(chan ProgressUpdate, 16)
	go func() {
		for range progress {
		}
	}()
	s.runner.Run(ctx, cfgRC, progress)

	// Reload the run to capture the runner's final status / results.
	final, err := s.store.Get(run.ID)
	if err != nil {
		cell.Status = CellStatusFailed
		cell.Error = "read final run: " + err.Error()
		return
	}
	switch final.Status {
	case StatusCompleted:
		cell.Status = CellStatusCompleted
		cell.Error = ""
	default:
		cell.Status = CellStatusFailed
		cell.Error = final.Error
	}
}

func (s *Service) failCells(job *BenchmarkJob, indices []int, errMsg string) {
	slog.Warn("job: failing cells", "job_id", job.ID, "count", len(indices), "error", errMsg)
	for _, idx := range indices {
		if job.Cells[idx].Status == CellStatusCompleted {
			continue
		}
		job.Cells[idx].Status = CellStatusFailed
		job.Cells[idx].Error = errMsg
	}
}

func (s *Service) markPendingCells(job *BenchmarkJob, _ string) {
	for i := range job.Cells {
		if job.Cells[i].Status == CellStatusPending || job.Cells[i].Status == CellStatusRunning {
			job.Cells[i].Status = CellStatusSkipped
		}
	}
}

// jobFinalStatus aggregates cell outcomes into a single job status. The
// cancelled flag wins outright; otherwise: any-failed → failed,
// else completed.
func jobFinalStatus(job *BenchmarkJob, cancelled bool) string {
	if cancelled {
		return JobStatusCanceled
	}
	anyFailed := false
	allSkipped := true
	for _, c := range job.Cells {
		if c.Status != CellStatusSkipped {
			allSkipped = false
		}
		if c.Status == CellStatusFailed {
			anyFailed = true
		}
	}
	if allSkipped {
		return JobStatusCanceled
	}
	if anyFailed {
		return JobStatusFailed
	}
	return JobStatusCompleted
}

// newID returns a short hex identifier suitable for URLs.
func newID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

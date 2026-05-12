package benchmark

import "time"

// AdhocJobID is the synthetic catch-all job that holds runs not produced
// by an explicit batch (single-run / quick-benchmark path).
const AdhocJobID = "adhoc"

// Job kinds.
const (
	JobKindBatch = "batch"
	JobKindAdhoc = "ad-hoc"
)

// Job-level status values.
const (
	JobStatusPending   = "pending"
	JobStatusRunning   = "running"
	JobStatusCompleted = "completed"
	JobStatusFailed    = "failed"
	JobStatusCanceled  = "canceled"
)

// Cell-level status values inside a job.
const (
	CellStatusPending   = "pending"
	CellStatusRunning   = "running"
	CellStatusCompleted = "completed"
	CellStatusFailed    = "failed"
	CellStatusSkipped   = "skipped"
)

// DeleteDisposition controls what happens to a job's runs when the job
// itself is deleted: cascade removes them, orphan reassigns to AdhocJobID.
type DeleteDisposition string

const (
	DeleteCascade DeleteDisposition = "cascade"
	DeleteOrphan  DeleteDisposition = "orphan"
)

// BenchmarkJob is a named, persistent batch run sweeping a {ModelIDs} ×
// {Presets} matrix. The expanded matrix lives in Cells. Once a job has
// started, the matrix is fixed; retry-failed bumps Attempt on individual
// cells.
type BenchmarkJob struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Kind        string `json:"kind"`
	Status      string `json:"status"`

	CreatedAt  time.Time `json:"created_at"`
	StartedAt  time.Time `json:"started_at,omitempty"`
	FinishedAt time.Time `json:"finished_at,omitempty"`

	ModelIDs  []string         `json:"model_ids,omitempty"`
	Presets   []string         `json:"presets,omitempty"`
	Overrides *ConfigOverrides `json:"overrides,omitempty"`

	Cells []JobCell `json:"cells,omitempty"`
}

// ConfigOverrides applies on top of each model's saved VLLMConfig for
// every cell. Nil pointer = "use the model's saved value".
type ConfigOverrides struct {
	MaxModelLen          *int     `json:"max_model_len,omitempty"`
	TensorParallelSize   *int     `json:"tensor_parallel_size,omitempty"`
	GPUMemoryUtilization *float64 `json:"gpu_memory_utilization,omitempty"`
	KVCacheDtype         *string  `json:"kv_cache_dtype,omitempty"`
	EnforceEager         *bool    `json:"enforce_eager,omitempty"`
	Dtype                *string  `json:"dtype,omitempty"`
}

// JobCell is one (model, preset) point in the matrix. The cell owns at
// most one BenchmarkRun at a time; on retry the run ID is rewritten to
// point at the latest attempt.
type JobCell struct {
	ModelID        string `json:"model_id"`
	Preset         string `json:"preset"`
	Status         string `json:"status"`
	Attempt        int    `json:"attempt"`
	BenchmarkRunID string `json:"benchmark_run_id,omitempty"`
	Error          string `json:"error,omitempty"`
}

// newAdhocJob synthesizes the catch-all "Ad-Hoc Runs" pseudo-job that
// holds runs not produced by an explicit batch. CreatedAt matches the
// oldest orphan run so the entry sorts naturally in the user's history.
func newAdhocJob(createdAt time.Time) BenchmarkJob {
	if createdAt.IsZero() {
		createdAt = time.Now()
	}
	return BenchmarkJob{
		ID:        AdhocJobID,
		Name:      "Ad-Hoc Runs",
		Kind:      JobKindAdhoc,
		Status:    JobStatusCompleted,
		CreatedAt: createdAt,
	}
}

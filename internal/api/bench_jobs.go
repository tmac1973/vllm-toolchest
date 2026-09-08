package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
)

// handleListJobs returns all jobs, newest first. Dual-mode.
func (s *Server) handleListJobs(w http.ResponseWriter, r *http.Request) {
	jobs := s.bench.ListJobs()
	if !isHTMX(r) {
		respondJSON(w, map[string]any{"jobs": jobs, "total": len(jobs)})
		return
	}
	respondHTML(w)
	s.renderJobList(w, jobs)
}

// handleGetJob returns one job with its cells. Dual-mode.
func (s *Server) handleGetJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	job, err := s.bench.GetJob(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if !isHTMX(r) {
		respondJSON(w, job)
		return
	}
	respondHTML(w)
	s.renderJobDetail(w, job)
}

// createJobRequest is the POST body for new jobs.
type createJobRequest struct {
	Name        string                     `json:"name"`
	Description string                     `json:"description,omitempty"`
	ModelIDs    []string                   `json:"model_ids"`
	Presets     []string                   `json:"presets"`
	Overrides   *benchmark.ConfigOverrides `json:"overrides,omitempty"`
}

// handleCreateJob persists a new batch job and dispatches it.
func (s *Server) handleCreateJob(w http.ResponseWriter, r *http.Request) {
	var req createJobRequest
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "json") {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		if err := r.ParseForm(); err != nil {
			http.Error(w, "invalid form", http.StatusBadRequest)
			return
		}
		req.Name = r.FormValue("name")
		req.Description = r.FormValue("description")
		req.ModelIDs = r.Form["model_ids"]
		req.Presets = r.Form["presets"]
	}

	if len(req.ModelIDs) == 0 {
		http.Error(w, "at least one model is required", http.StatusBadRequest)
		return
	}
	if len(req.Presets) == 0 {
		http.Error(w, "at least one preset is required", http.StatusBadRequest)
		return
	}
	if req.Name == "" {
		req.Name = fmt.Sprintf("Batch %s", time.Now().Format("2006-01-02 15:04"))
	}

	for _, m := range req.ModelIDs {
		if _, ok := s.registry.Get(m); !ok {
			http.Error(w, "model not registered: "+m, http.StatusBadRequest)
			return
		}
	}
	presetSet := map[string]bool{}
	for _, p := range benchmark.Presets() {
		presetSet[p.Name] = true
	}
	for _, p := range req.Presets {
		if !presetSet[p] {
			http.Error(w, "unknown preset: "+p, http.StatusBadRequest)
			return
		}
	}

	job := benchmark.BenchmarkJob{
		ID:          newRunID(),
		Name:        req.Name,
		Description: req.Description,
		Kind:        benchmark.JobKindBatch,
		Status:      benchmark.JobStatusPending,
		CreatedAt:   time.Now(),
		ModelIDs:    req.ModelIDs,
		Presets:     req.Presets,
		Overrides:   req.Overrides,
		Cells:       benchmark.ExpandCells(req.ModelIDs, req.Presets),
	}

	if err := s.bench.SaveJob(job); err != nil {
		http.Error(w, "save job: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if err := s.benchSvc.SubmitJob(job); err != nil {
		// Roll the job back to failed so the UI doesn't show it forever-pending.
		job.Status = benchmark.JobStatusFailed
		_ = s.bench.SaveJob(job)
		if errors.Is(err, benchmark.ErrRunAlreadyActive) {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if !isHTMX(r) {
		w.WriteHeader(http.StatusAccepted)
		respondJSON(w, map[string]any{"id": job.ID, "status": job.Status})
		return
	}
	respondHTML(w)
	w.WriteHeader(http.StatusAccepted)
	s.renderPartial(w, "job_started", job.ID)
}

// handleCancelJob cancels the in-flight job with the given id.
func (s *Server) handleCancelJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if !s.benchSvc.CancelJob(id) {
		http.Error(w, "no active job with that id", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

// handleRetryFailedCells resets failed cells to pending and resubmits the
// job. Already-completed cells are preserved.
func (s *Server) handleRetryFailedCells(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	job, err := s.bench.GetJob(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	count := 0
	for i := range job.Cells {
		if job.Cells[i].Status == benchmark.CellStatusFailed || job.Cells[i].Status == benchmark.CellStatusSkipped {
			job.Cells[i].Status = benchmark.CellStatusPending
			job.Cells[i].Error = ""
			count++
		}
	}
	if count == 0 {
		http.Error(w, "no failed or skipped cells to retry", http.StatusBadRequest)
		return
	}

	job.Status = benchmark.JobStatusPending
	job.StartedAt = time.Time{}
	job.FinishedAt = time.Time{}
	if err := s.bench.SaveJob(*job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := s.benchSvc.SubmitJob(*job); err != nil {
		if errors.Is(err, benchmark.ErrRunAlreadyActive) {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

// handleDeleteJob removes a job. ?runs=cascade removes the job's runs;
// ?runs=orphan reassigns them to AdhocJobID (the default).
func (s *Server) handleDeleteJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if active, ok := s.benchSvc.ActiveJobID(); ok && active == id {
		http.Error(w, "cannot delete a running job; cancel first", http.StatusConflict)
		return
	}

	disposition := benchmark.DeleteDisposition(r.URL.Query().Get("runs"))
	if disposition == "" {
		disposition = benchmark.DeleteOrphan
	}

	if err := s.bench.DeleteJob(id, disposition); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleJobForm renders the new-job form partial.
func (s *Server) handleJobForm(w http.ResponseWriter, r *http.Request) {
	type modelChoice struct{ ID, Name string }
	var choices []modelChoice
	for _, m := range s.registry.List() {
		if !m.Enabled || m.Orphaned {
			continue
		}
		choices = append(choices, modelChoice{ID: m.ID, Name: displayNameOf(m)})
	}

	respondHTML(w)
	s.renderPartial(w, "job_form", struct {
		Models  []modelChoice
		Presets []benchmark.Preset
	}{choices, benchmark.Presets()})
}

// jobRow is one job's collapsed summary line.
type jobRow struct {
	ID          string
	Name        string
	Description string
	Status      string
	IsAdhoc     bool
	Done        int
	Total       int
	Failed      int
	RunCount    int
	CreatedAt   string
	FinishedAt  string
}

// jobSummary counts a job's cells. The ad-hoc pseudo-job has no cells — its
// runs arrived one at a time — so it counts runs instead.
func (s *Server) jobSummary(job benchmark.BenchmarkJob) jobRow {
	row := jobRow{
		ID:          job.ID,
		Name:        job.Name,
		Description: job.Description,
		Status:      job.Status,
		IsAdhoc:     job.ID == benchmark.AdhocJobID,
		Total:       len(job.Cells),
	}
	if !job.CreatedAt.IsZero() {
		row.CreatedAt = job.CreatedAt.Format("Jan 2 15:04")
	}
	if !job.FinishedAt.IsZero() {
		row.FinishedAt = job.FinishedAt.Format("Jan 2 15:04")
	}
	for _, c := range job.Cells {
		switch c.Status {
		case benchmark.CellStatusCompleted:
			row.Done++
		case benchmark.CellStatusFailed:
			row.Failed++
		}
	}
	if row.IsAdhoc {
		row.RunCount = len(s.bench.RunsForJob(job.ID))
	}
	return row
}

// renderJobList renders the collapsed job rows. Each one loads its own cells
// on first open.
func (s *Server) renderJobList(w http.ResponseWriter, jobs []benchmark.BenchmarkJob) {
	rows := make([]jobRow, 0, len(jobs))
	for _, job := range jobs {
		rows = append(rows, s.jobSummary(job))
	}
	s.renderPartial(w, "job_list", rows)
}

// jobCellRow is one cell of a job's matrix, joined with whatever its run
// measured.
type jobCellRow struct {
	Idx        int
	ModelName  string
	Quant      string
	Preset     string
	SweepText  string
	Status     string
	Error      string
	ErrorShort string
	TGTPS      string
	PPTPS      string
	TTFT       string
	Attempt    int
	RunID      string
}

// errorSummary is the first line of an error, for a table cell. The full text
// stays in the title attribute — a vLLM traceback is hundreds of lines and
// would otherwise be the whole page.
func errorSummary(err string) string {
	if err == "" {
		return ""
	}
	line := err
	if i := strings.IndexByte(line, '\n'); i >= 0 {
		line = line[:i]
	}
	const max = 60
	if len(line) > max {
		line = line[:max-1] + "\u2026"
	}
	return line
}

func (s *Server) renderJobDetail(w http.ResponseWriter, job *benchmark.BenchmarkJob) {
	// Runs indexed by id, so each cell can show what its attempt measured
	// without a lookup per row.
	runs := make(map[string]benchmark.BenchmarkRun)
	for _, r := range s.bench.RunsForJob(job.ID) {
		runs[r.ID] = r
	}

	summary := s.jobSummary(*job)
	cells := job.Cells
	// The ad-hoc job holds runs rather than cells; synthesize a row per run so
	// it lists like any other job.
	if summary.IsAdhoc && len(cells) == 0 {
		for _, r := range s.bench.RunsForJob(job.ID) {
			cells = append(cells, benchmark.JobCell{
				ModelID:        r.ModelID,
				Preset:         r.Preset,
				Status:         cellStatusForRun(r.Status),
				Attempt:        1,
				BenchmarkRunID: r.ID,
			})
		}
	}

	hasSweeps := false
	rows := make([]jobCellRow, 0, len(cells))
	for i, c := range cells {
		row := jobCellRow{
			Idx:        i,
			ModelName:  c.ModelID,
			Preset:     c.Preset,
			Status:     c.Status,
			Error:      c.Error,
			ErrorShort: errorSummary(c.Error),
			Attempt:    c.Attempt,
			RunID:      c.BenchmarkRunID,
			TGTPS:      "\u2014",
			PPTPS:      "\u2014",
			TTFT:       "\u2014",
		}
		if m, ok := s.registry.Get(c.ModelID); ok {
			row.ModelName = displayNameOf(m)
		}
		if run, ok := runs[c.BenchmarkRunID]; ok {
			row.Quant = run.Quant
			if run.Summary != nil {
				row.TGTPS = fmt.Sprintf("%.1f", run.Summary.AvgGenTokPerSec)
				row.PPTPS = fmt.Sprintf("%.0f", run.Summary.AvgPromptTokPerSec)
				row.TTFT = fmt.Sprintf("%.0f ms", run.Summary.AvgTTFTMs)
			}
		}
		if len(c.SweepValues) > 0 {
			row.SweepText = sweepValuesText(c.SweepValues)
			hasSweeps = true
		}
		rows = append(rows, row)
	}

	// Checkbox, model, quant, preset, status, TG, PP, TTFT, attempt, detail —
	// plus the sweep column when there is one.
	colSpan := 10
	if hasSweeps {
		colSpan++
	}

	s.renderPartial(w, "job_detail", struct {
		jobRow
		Running      bool
		HasSweeps    bool
		ColSpan      int
		OverrideText string
		Rows         []jobCellRow
	}{
		jobRow:       summary,
		Running:      job.Status == benchmark.JobStatusRunning,
		HasSweeps:    hasSweeps,
		ColSpan:      colSpan,
		OverrideText: overridesText(job.Overrides),
		Rows:         rows,
	})
}

// cellStatusForRun maps a run's status onto the cell vocabulary, for the
// synthesized ad-hoc rows.
func cellStatusForRun(status string) string {
	switch status {
	case benchmark.StatusCompleted:
		return benchmark.CellStatusCompleted
	case benchmark.StatusFailed:
		return benchmark.CellStatusFailed
	case benchmark.StatusRunning:
		return benchmark.CellStatusRunning
	}
	return benchmark.CellStatusPending
}

// overridesText renders the config overrides a job applied on top of each
// model's saved settings, so a surprising number has somewhere to come from.
func overridesText(o *benchmark.ConfigOverrides) string {
	if o == nil {
		return ""
	}
	var parts []string
	if o.MaxModelLen != nil {
		parts = append(parts, fmt.Sprintf("max_model_len=%d", *o.MaxModelLen))
	}
	if o.TensorParallelSize != nil {
		parts = append(parts, fmt.Sprintf("tensor_parallel_size=%d", *o.TensorParallelSize))
	}
	if o.GPUMemoryUtilization != nil {
		parts = append(parts, fmt.Sprintf("gpu_memory_utilization=%.2f", *o.GPUMemoryUtilization))
	}
	if o.KVCacheDtype != nil {
		parts = append(parts, "kv_cache_dtype="+*o.KVCacheDtype)
	}
	if o.EnforceEager != nil {
		parts = append(parts, fmt.Sprintf("enforce_eager=%v", *o.EnforceEager))
	}
	if o.Dtype != nil {
		parts = append(parts, "dtype="+*o.Dtype)
	}
	return strings.Join(parts, " · ")
}

// sweepValuesText renders a cell's swept values in a stable order — a map
// would otherwise reorder them on every poll.
func sweepValuesText(values map[string]string) string {
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+values[k])
	}
	return strings.Join(parts, " ")
}

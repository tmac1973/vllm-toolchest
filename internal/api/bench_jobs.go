package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
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

// jobRow is one job's summary line plus the runs it produced.
type jobRow struct {
	ID          string
	Name        string
	Kind        string
	Status      string
	CellSummary string
	CreatedAt   string
	CanCancel   bool
	CanRetry    bool
	CanDelete   bool
	Runs        []runRow
}

// renderJobList renders the job-grouped table for the benchmarks page.
func (s *Server) renderJobList(w http.ResponseWriter, jobs []benchmark.BenchmarkJob) {
	rows := make([]jobRow, 0, len(jobs))
	for _, job := range jobs {
		row := jobRow{
			ID:          job.ID,
			Name:        job.Name,
			Kind:        job.Kind,
			Status:      job.Status,
			CellSummary: fmt.Sprintf("%d cell(s)", len(job.Cells)),
			CreatedAt:   job.CreatedAt.Format("Jan 2 15:04"),
			CanCancel:   job.Status == benchmark.JobStatusRunning,
			CanRetry:    job.Status == benchmark.JobStatusFailed || job.Status == benchmark.JobStatusCanceled,
			// The ad-hoc job is the bucket every unattached run lands in;
			// deleting it would have nowhere to put them.
			CanDelete: job.ID != benchmark.AdhocJobID,
		}
		if len(job.Cells) > 0 {
			done, failed := 0, 0
			for _, c := range job.Cells {
				switch c.Status {
				case benchmark.CellStatusCompleted:
					done++
				case benchmark.CellStatusFailed:
					failed++
				}
			}
			row.CellSummary = fmt.Sprintf("%d/%d done, %d failed", done, len(job.Cells), failed)
		}
		for _, run := range s.bench.RunsForJob(job.ID) {
			r := runRow{
				ID:        run.ID,
				ModelName: run.ModelName,
				Preset:    run.Preset,
				AvgGen:    "\u2014",
				AvgTTFT:   "\u2014",
				Status:    run.Status,
			}
			if run.Summary != nil {
				r.AvgGen = fmt.Sprintf("%.1f t/s", run.Summary.AvgGenTokPerSec)
				r.AvgTTFT = fmt.Sprintf("%.0f ms", run.Summary.AvgTTFTMs)
			}
			row.Runs = append(row.Runs, r)
		}
		rows = append(rows, row)
	}
	s.renderPartial(w, "job_list", rows)
}

// jobCellRow is one cell of a job's model x preset matrix.
type jobCellRow struct {
	ModelID string
	Preset  string
	Status  string
	Error   string
	Attempt int
	RunID   string
}

func (s *Server) renderJobDetail(w http.ResponseWriter, job *benchmark.BenchmarkJob) {
	cells := make([]jobCellRow, 0, len(job.Cells))
	for _, c := range job.Cells {
		cells = append(cells, jobCellRow{
			ModelID: c.ModelID,
			Preset:  c.Preset,
			Status:  c.Status,
			Error:   c.Error,
			Attempt: c.Attempt,
			RunID:   c.BenchmarkRunID,
		})
	}
	s.renderPartial(w, "job_detail", struct {
		Name        string
		Kind        string
		Status      string
		Description string
		Cells       []jobCellRow
	}{job.Name, job.Kind, job.Status, job.Description, cells})
}

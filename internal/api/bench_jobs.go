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
	renderJobList(w, s, jobs)
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
	renderJobDetail(w, s, job)
}

// createJobRequest is the POST body for new jobs.
type createJobRequest struct {
	Name        string                       `json:"name"`
	Description string                       `json:"description,omitempty"`
	ModelIDs    []string                     `json:"model_ids"`
	Presets     []string                     `json:"presets"`
	Overrides   *benchmark.ConfigOverrides   `json:"overrides,omitempty"`
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
	fmt.Fprintf(w, `<p>Started job <code>%s</code>. <a href="#" hx-get="/api/benchmark-jobs/" hx-target="#bench-jobs">Refresh</a></p>`, htmlEscape(job.ID))
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
	respondHTML(w)
	allModels := s.registry.List()

	fmt.Fprint(w, `<article>
  <header><strong>New batch job</strong></header>
  <form hx-post="/api/benchmark-jobs/" hx-encoding="application/x-www-form-urlencoded" hx-target="#bench-job-result" hx-swap="innerHTML">
    <label>Name <input type="text" name="name" placeholder="Quant compare"></label>
    <label>Description <input type="text" name="description" placeholder="optional"></label>

    <label>Models</label>
    <div>`)

	if len(allModels) == 0 {
		fmt.Fprint(w, `<small style="opacity:0.7;">No models in the registry. <a href="/models/browse">Browse HuggingFace</a> first.</small>`)
	}
	for _, m := range allModels {
		if !m.Enabled || m.Orphaned {
			continue
		}
		fmt.Fprintf(w, `<label><input type="checkbox" name="model_ids" value="%s"> %s <small style="opacity:0.6;">(%s)</small></label>`,
			htmlEscape(m.ID), htmlEscape(displayNameOf(m)), htmlEscape(m.ID))
	}

	fmt.Fprint(w, `</div>

    <label>Presets</label>
    <div>`)
	for _, p := range benchmark.Presets() {
		fmt.Fprintf(w, `<label><input type="checkbox" name="presets" value="%s"> <code>%s</code> <small style="opacity:0.7;">— %s</small></label>`,
			htmlEscape(p.Name), htmlEscape(p.Name), htmlEscape(p.Description))
	}
	fmt.Fprint(w, `</div>

    <button type="submit">Submit job</button>
  </form>
  <div id="bench-job-result"></div>
</article>`)
}

// renderJobList renders the job-grouped table for the benchmarks page.
func renderJobList(w http.ResponseWriter, s *Server, jobs []benchmark.BenchmarkJob) {
	if len(jobs) == 0 {
		fmt.Fprint(w, `<p style="opacity:0.7;">No benchmark jobs yet.</p>`)
		return
	}

	for _, job := range jobs {
		runs := s.bench.RunsForJob(job.ID)

		cellSummary := fmt.Sprintf("%d cell(s)", len(job.Cells))
		if len(job.Cells) > 0 {
			done := 0
			failed := 0
			for _, c := range job.Cells {
				switch c.Status {
				case benchmark.CellStatusCompleted:
					done++
				case benchmark.CellStatusFailed:
					failed++
				}
			}
			cellSummary = fmt.Sprintf("%d/%d done, %d failed", done, len(job.Cells), failed)
		}

		actions := ""
		switch job.Status {
		case benchmark.JobStatusRunning:
			actions = fmt.Sprintf(`<a href="#" hx-post="/api/benchmark-jobs/%s/cancel" hx-confirm="Cancel this job?">cancel</a>`, job.ID)
		case benchmark.JobStatusFailed, benchmark.JobStatusCanceled:
			actions = fmt.Sprintf(`<a href="#" hx-post="/api/benchmark-jobs/%s/retry-failed" hx-target="#bench-jobs" hx-swap="none" hx-on::after-request="htmx.ajax('GET','/api/benchmark-jobs/','#bench-jobs')">retry failed</a>`, job.ID)
		}
		if job.ID != benchmark.AdhocJobID {
			if actions != "" {
				actions += " &middot; "
			}
			actions += fmt.Sprintf(
				`<a href="#" hx-delete="/api/benchmark-jobs/%s?runs=orphan" hx-confirm="Delete job (orphan runs to ad-hoc)?" hx-target="#bench-jobs" hx-swap="none" hx-on::after-request="htmx.ajax('GET','/api/benchmark-jobs/','#bench-jobs')">delete</a>`,
				job.ID,
			)
		}

		fmt.Fprintf(w, `<article style="margin-bottom:1rem;">
  <header>
    <strong>%s</strong>
    <small style="opacity:0.7;">&middot; %s &middot; %s &middot; %s &middot; %s</small>
    <span style="float:right;font-size:0.85em;">%s</span>
  </header>`,
			htmlEscape(job.Name),
			job.Kind,
			statusBadgeJob(job.Status),
			cellSummary,
			job.CreatedAt.Format("Jan 2 15:04"),
			actions,
		)

		if len(runs) > 0 {
			fmt.Fprint(w, `<table style="margin-top:0.5rem;"><thead><tr>
  <th>Model</th><th>Preset</th><th>Avg gen TPS</th><th>Avg TTFT</th><th>Status</th><th></th>
</tr></thead><tbody>`)
			for _, run := range runs {
				var avgGen, avgTTFT string
				if run.Summary != nil {
					avgGen = fmt.Sprintf("%.1f t/s", run.Summary.AvgGenTokPerSec)
					avgTTFT = fmt.Sprintf("%.0f ms", run.Summary.AvgTTFTMs)
				} else {
					avgGen = "—"
					avgTTFT = "—"
				}
				fmt.Fprintf(w, `<tr>
  <td><small>%s</small></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>
  <td><small>
    <a href="#" hx-get="/api/benchmarks/%s" hx-target="#bench-detail-%s" hx-swap="innerHTML">details</a>
  </small></td>
</tr>
<tr><td colspan="6"><div id="bench-detail-%s"></div></td></tr>`,
					htmlEscape(run.ModelName), run.Preset, avgGen, avgTTFT, statusBadge(run.Status),
					run.ID, run.ID, run.ID)
			}
			fmt.Fprint(w, `</tbody></table>`)
		}

		fmt.Fprint(w, `</article>`)
	}
}

func renderJobDetail(w http.ResponseWriter, s *Server, job *benchmark.BenchmarkJob) {
	fmt.Fprintf(w, `<article>
  <header><strong>%s</strong> &middot; %s &middot; %s</header>
  <p><small>%s</small></p>
  <table><thead><tr><th>Model</th><th>Preset</th><th>Status</th><th>Attempt</th><th>Run</th></tr></thead><tbody>`,
		htmlEscape(job.Name), htmlEscape(job.Kind), statusBadgeJob(job.Status), htmlEscape(job.Description))
	for _, c := range job.Cells {
		runLink := "—"
		if c.BenchmarkRunID != "" {
			runLink = fmt.Sprintf(`<a href="#" hx-get="/api/benchmarks/%s">view</a>`, c.BenchmarkRunID)
		}
		errText := ""
		if c.Error != "" {
			errText = fmt.Sprintf(`<br><small><del>%s</del></small>`, htmlEscape(c.Error))
		}
		fmt.Fprintf(w, `<tr>
  <td><small>%s</small></td><td>%s</td><td>%s%s</td><td>%d</td><td><small>%s</small></td>
</tr>`,
			htmlEscape(c.ModelID), c.Preset, statusBadgeCell(c.Status), errText, c.Attempt, runLink)
	}
	fmt.Fprint(w, `</tbody></table></article>`)
}

func statusBadgeJob(s string) string {
	switch s {
	case benchmark.JobStatusRunning:
		return `<mark>running</mark>`
	case benchmark.JobStatusCompleted:
		return `<ins>completed</ins>`
	case benchmark.JobStatusFailed:
		return `<del>failed</del>`
	case benchmark.JobStatusCanceled:
		return `<small>canceled</small>`
	case benchmark.JobStatusPending:
		return `<small>pending</small>`
	default:
		return s
	}
}

func statusBadgeCell(s string) string {
	switch s {
	case benchmark.CellStatusRunning:
		return `<mark>running</mark>`
	case benchmark.CellStatusCompleted:
		return `<ins>done</ins>`
	case benchmark.CellStatusFailed:
		return `<del>failed</del>`
	case benchmark.CellStatusSkipped:
		return `<small>skipped</small>`
	default:
		return s
	}
}

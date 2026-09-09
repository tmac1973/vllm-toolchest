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
	// The ad-hoc job is a container for individually-started runs, not a cell
	// matrix, and the questions asked of it are different: which model, when,
	// and how did these compare — not which cell of a sweep failed. So it gets
	// the grouped run list rather than the cell table.
	if job.ID == benchmark.AdhocJobID {
		s.renderRunList(w, s.bench.RunsForJob(benchmark.AdhocJobID))
		return
	}
	s.renderJobDetail(w, job)
}

// createJobRequest is the POST body for new jobs.
type createJobRequest struct {
	Name        string                     `json:"name"`
	Description string                     `json:"description,omitempty"`
	ModelIDs    []string                   `json:"model_ids"`
	Presets     []string                   `json:"presets"`
	Overrides   *benchmark.ConfigOverrides `json:"overrides,omitempty"`
	Sweeps      []benchmark.SweepAxis      `json:"sweeps,omitempty"`
}

// handleCreateJob persists a new batch job and dispatches it.
// jobFail reports a rejected job submission. htmx does not swap a non-2xx
// response, so an htmx caller given http.Error sees nothing at all — the
// button clicks, the form sits there, and the reason is only in the network
// tab. It gets 200 and the error partial instead; everything else keeps real
// status codes.
func (s *Server) jobFail(w http.ResponseWriter, r *http.Request, status int, msg string) {
	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "error_message", msg)
		return
	}
	http.Error(w, msg, status)
}

func (s *Server) handleCreateJob(w http.ResponseWriter, r *http.Request) {
	var req createJobRequest
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "json") {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.jobFail(w, r, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
	} else {
		if err := r.ParseForm(); err != nil {
			s.jobFail(w, r, http.StatusBadRequest, "invalid form")
			return
		}
		req.Name = r.FormValue("name")
		req.Description = r.FormValue("description")
		req.ModelIDs = r.Form["model_ids"]
		req.Presets = r.Form["presets"]

		// One field per sweepable parameter, named sweep_<field>. The form
		// submits it once per ticked checkbox, so the value arrives as a
		// repeated field; a single comma-separated string is still accepted,
		// which is what a hand-rolled request or an older client sends.
		// ParseSweepValues handles both and drops duplicates, so the two
		// shapes cannot disagree.
		for _, f := range benchmark.SweepFields() {
			raw := strings.Join(r.Form["sweep_"+f.Name], ",")
			if strings.TrimSpace(raw) == "" {
				continue
			}
			values, err := benchmark.ParseSweepValues(f, raw)
			if err != nil {
				s.jobFail(w, r, http.StatusBadRequest, err.Error())
				return
			}
			if len(values) > 0 {
				req.Sweeps = append(req.Sweeps, benchmark.SweepAxis{Field: f.Name, Values: values})
			}
		}
	}

	if len(req.ModelIDs) == 0 {
		s.jobFail(w, r, http.StatusBadRequest, "at least one model is required")
		return
	}
	if len(req.Presets) == 0 {
		s.jobFail(w, r, http.StatusBadRequest, "at least one preset is required")
		return
	}
	if req.Name == "" {
		req.Name = fmt.Sprintf("Batch %s", time.Now().Format("2006-01-02 15:04"))
	}

	for _, m := range req.ModelIDs {
		if _, ok := s.registry.Get(m); !ok {
			s.jobFail(w, r, http.StatusBadRequest, "model not registered: "+m)
			return
		}
	}
	if err := benchmark.ValidateSweeps(req.Sweeps); err != nil {
		s.jobFail(w, r, http.StatusBadRequest, err.Error())
		return
	}
	presetSet := map[string]bool{}
	for _, p := range benchmark.Presets() {
		presetSet[p.Name] = true
	}
	for _, p := range req.Presets {
		if !presetSet[p] {
			s.jobFail(w, r, http.StatusBadRequest, "unknown preset: "+p)
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
		Sweeps:      req.Sweeps,
		Cells:       benchmark.ExpandCells(req.ModelIDs, req.Presets, req.Sweeps),
	}

	if err := s.bench.SaveJob(job); err != nil {
		s.jobFail(w, r, http.StatusInternalServerError, "save job: "+err.Error())
		return
	}

	if err := s.benchSvc.SubmitJob(job); err != nil {
		// Roll the job back to failed so the UI doesn't show it forever-pending.
		job.Status = benchmark.JobStatusFailed
		_ = s.bench.SaveJob(job)
		if errors.Is(err, benchmark.ErrRunAlreadyActive) {
			s.jobFail(w, r, http.StatusConflict, err.Error())
			return
		}
		s.jobFail(w, r, http.StatusInternalServerError, err.Error())
		return
	}

	if !isHTMX(r) {
		w.WriteHeader(http.StatusAccepted)
		respondJSON(w, map[string]any{"id": job.ID, "status": job.Status})
		return
	}
	respondHTML(w)
	// Tells the page the submission took, so it can close the form. Without
	// it the editor stays open over a confirmation it is hiding, which reads
	// as nothing having happened.
	w.Header().Set("HX-Trigger", "jobSubmitted")
	w.WriteHeader(http.StatusAccepted)
	s.renderPartial(w, "job_started", struct {
		ID   string
		Name string
	}{job.ID, job.Name})
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
// jobFormChoice is one checkbox in the job form.
type jobFormChoice struct {
	ID          string
	Name        string
	Description string
	Checked     bool
}

// jobFormSweep is one sweep input, pre-filled when re-running a job.
// jobFormSweep is one parameter's row in the job form: the curated choices,
// each marked with whether this job has it ticked.
type jobFormSweep struct {
	Name    string
	Label   string
	Help    string
	Example string
	Options []sweepOption
	// Selected counts the ticked options and Summary is the text the closed
	// menu shows. Both are rendered server-side so a form re-opened from an
	// existing job is already correct: the browser syncs these on load too,
	// but only after a swap, and a summary that says "use the saved value"
	// over three ticked boxes is wrong for as long as it is on screen.
	Selected int
	Summary  string
}

// sweepOption is one value a parameter can take.
type sweepOption struct {
	Value   string
	Checked bool
	// Custom marks a value that is not in the curated list — one a previous
	// job set by hand. It is rendered as a ticked checkbox like any other, so
	// re-running a job cannot silently drop a value the list does not cover.
	Custom bool
}

// sweepSummary is the one-line description of a parameter's selection shown on
// its closed menu. It has to agree exactly with what the browser writes when
// the selection changes — see syncParamRow in benchmarks.html — or re-opening
// a job would show one thing until the first click and another after it.
func sweepSummary(values []string) string {
	switch len(values) {
	case 0:
		return "Use model's saved value"
	case 1:
		return values[0]
	default:
		return strings.Join(values, ", ") + " \u2014 sweep"
	}
}

// handleJobForm renders the batch-job form. With ?from=<id> it comes back
// pre-filled from an existing job, which is what Edit & Re-run opens: a job
// worth repeating is usually worth repeating with one thing changed.
func (s *Server) handleJobForm(w http.ResponseWriter, r *http.Request) {
	var from *benchmark.BenchmarkJob
	if id := r.URL.Query().Get("from"); id != "" {
		if job, err := s.bench.GetJob(id); err == nil {
			from = job
		}
	}

	chosenModels := map[string]bool{}
	chosenPresets := map[string]bool{}
	sweepValues := map[string][]string{}
	name, description, fromName := "", "", ""
	if from != nil {
		fromName = from.Name
		// A re-run is a new job, so the name gets a marker rather than
		// silently colliding with the one it came from.
		name = from.Name + " (re-run)"
		description = from.Description
		for _, m := range from.ModelIDs {
			chosenModels[m] = true
		}
		for _, p := range from.Presets {
			chosenPresets[p] = true
		}
		for _, axis := range from.Sweeps {
			sweepValues[axis.Field] = axis.Values
		}
	}

	var models []jobFormChoice
	for _, m := range s.registry.List() {
		if !m.Enabled || m.Orphaned {
			continue
		}
		models = append(models, jobFormChoice{
			ID: m.ID, Name: displayNameOf(m), Checked: chosenModels[m.ID],
		})
	}

	var presets []jobFormChoice
	for _, p := range benchmark.Presets() {
		presets = append(presets, jobFormChoice{
			Name: p.Name, Description: p.Description, Checked: chosenPresets[p.Name],
		})
	}

	var sweeps []jobFormSweep
	for _, f := range benchmark.SweepFields() {
		chosen := map[string]bool{}
		for _, v := range sweepValues[f.Name] {
			chosen[v] = true
		}
		row := jobFormSweep{Name: f.Name, Label: f.Label, Help: f.Help, Example: f.Example}
		for _, c := range f.Choices {
			row.Options = append(row.Options, sweepOption{Value: c, Checked: chosen[c]})
			delete(chosen, c)
		}
		// Whatever is left was set by hand on the job being re-run. Appending
		// it keeps the value visible and ticked; dropping it would silently
		// change what the re-run measures.
		for _, v := range sweepValues[f.Name] {
			if chosen[v] {
				row.Options = append(row.Options, sweepOption{Value: v, Checked: true, Custom: true})
				delete(chosen, v)
			}
		}
		var picked []string
		for _, o := range row.Options {
			if o.Checked {
				row.Selected++
				picked = append(picked, o.Value)
			}
		}
		row.Summary = sweepSummary(picked)
		sweeps = append(sweeps, row)
	}

	respondHTML(w)
	s.renderPartial(w, "job_form", struct {
		FromJob     string
		Name        string
		Description string
		Models      []jobFormChoice
		Presets     []jobFormChoice
		SweepFields []jobFormSweep
		HasSweeps   bool
		MaxCells    int
	}{
		FromJob:     fromName,
		Name:        name,
		Description: description,
		Models:      models,
		Presets:     presets,
		SweepFields: sweeps,
		HasSweeps:   len(sweepValues) > 0,
		MaxCells:    benchmark.MaxSweepCombinations,
	})
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

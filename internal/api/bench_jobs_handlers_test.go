package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// bhPutJob submits the Edit & re-run form for job id, as htmx when htmx is set.
func bhPutJob(s *Server, id string, form url.Values, htmx bool) *httptest.ResponseRecorder {
	r := bhRequest(http.MethodPut, "/api/benchmark-jobs/"+id,
		strings.NewReader(form.Encode()), map[string]string{"id": id})
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if htmx {
		r.Header.Set("HX-Request", "true")
	}
	return bhServe(s.handleUpdateJob, r)
}

func bhEditForm(preset string) url.Values {
	return url.Values{"name": {"edited"}, "model_ids": {bhModel}, "presets": {preset}}
}

// bhSaveJob stores a job without submitting it.
func bhSaveJob(t *testing.T, s *Server, job benchmark.BenchmarkJob) {
	t.Helper()
	if err := s.bench.SaveJob(job); err != nil {
		t.Fatal(err)
	}
}

func bhGetJob(t *testing.T, s *Server, id string) *benchmark.BenchmarkJob {
	t.Helper()
	job, err := s.bench.GetJob(id)
	if err != nil {
		t.Fatal(err)
	}
	return job
}

// An unknown preset is refused before the stored definition is touched. The
// htmx caller gets the reason in a 200 so the form shows it; a script keeps
// the 400.
func TestEditingAJobWithAnUnknownPresetIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveJob(t, s, bhJob("j1", benchmark.JobStatusCompleted))

	w := bhPutJob(s, "j1", bhEditForm("nope"), true)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "unknown preset") {
		t.Errorf("htmx: status %d %q, want 200 with the reason", w.Code, w.Body)
	}
	w = bhPutJob(s, "j1", bhEditForm("nope"), false)
	if w.Code != http.StatusBadRequest {
		t.Errorf("plain: status %d, want 400", w.Code)
	}
	if got := bhGetJob(t, s, "j1"); got.Name != "job j1" || got.Status != benchmark.JobStatusCompleted {
		t.Errorf("a refused edit changed the job: name %q, status %q", got.Name, got.Status)
	}
}

func TestEditingAnUnknownJobIsNotFound(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := bhPutJob(s, "missing", bhEditForm("internal-quick"), false)
	if w.Code != http.StatusNotFound {
		t.Errorf("plain: status %d, want 404", w.Code)
	}
	w = bhPutJob(s, "missing", bhEditForm("internal-quick"), true)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "job not found") {
		t.Errorf("htmx: status %d %q, want 200 with the reason", w.Code, w.Body)
	}
}

// The Ad-Hoc Runs row is synthesized from loose runs; there is no definition
// behind it to edit.
func TestEditingTheAdhocJobIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := bhPutJob(s, benchmark.AdhocJobID, bhEditForm("internal-quick"), false)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
}

// Editing re-runs the job under the same id, so a successful edit both
// rewrites the definition and hands the job to the runner.
func TestEditingAJobRewritesAndResubmitsIt(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveJob(t, s, bhJob("j1", benchmark.JobStatusCompleted))
	s.benchSvc.SetJobEnv(&stubJobEnv{})
	t.Cleanup(func() { bhStopJob(t, s, "j1") })

	w := bhPutJob(s, "j1", bhEditForm("internal-standard"), true)
	if w.Code != http.StatusAccepted || w.Header().Get("HX-Trigger") != "jobSubmitted" {
		t.Fatalf("status %d, trigger %q: %s", w.Code, w.Header().Get("HX-Trigger"), w.Body)
	}
	if got := bhGetJob(t, s, "j1"); got.Name != "edited" || got.Presets[0] != "internal-standard" {
		t.Errorf("definition not rewritten: %+v", got)
	}
	if id, busy := s.benchSvc.ActiveJobID(); !busy || id != "j1" {
		t.Errorf("active job %q, %v; want j1 resubmitted", id, busy)
	}
}

// A refused edit must leave the job as it was. The handler rewrites the
// stored definition (status reset to pending, cells re-expanded) before it
// asks the service whether it can run, so while anything else is running the
// caller is told 409 yet the job has already changed and now sits "pending"
// with nothing to ever run it.
func TestEditingAJobWhileAnotherRunsLeavesItUntouched(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveJob(t, s, bhJob("idle", benchmark.JobStatusCompleted))
	bhStartActiveJob(t, s, "busy")

	w := bhPutJob(s, "idle", bhEditForm("internal-standard"), false)
	if w.Code != http.StatusConflict {
		t.Errorf("status %d, want 409", w.Code)
	}
	got := bhGetJob(t, s, "idle")
	if got.Name != "job idle" || got.Status != benchmark.JobStatusCompleted {
		t.Errorf("a refused edit changed the job: name %q, status %q", got.Name, got.Status)
	}
}

func TestRetryingAnUnknownJobIsNotFound(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := bhServe(s.handleRetryFailedCells, bhRequest(http.MethodPost,
		"/api/benchmark-jobs/missing/retry-failed", nil, map[string]string{"id": "missing"}))
	if w.Code != http.StatusNotFound {
		t.Errorf("status %d, want 404", w.Code)
	}
}

// With nothing failed there is nothing to retry; resubmitting would reload
// every model just to skip every cell.
func TestRetryingAJobWithNoFailedCellsIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	job := bhJob("j1", benchmark.JobStatusCompleted)
	job.Cells[0].Status = benchmark.CellStatusCompleted
	bhSaveJob(t, s, job)

	w := bhServe(s.handleRetryFailedCells, bhRequest(http.MethodPost,
		"/api/benchmark-jobs/j1/retry-failed", nil, map[string]string{"id": "j1"}))
	if w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
	if _, busy := s.benchSvc.ActiveJobID(); busy {
		t.Error("the job was submitted anyway")
	}
}

// Failed and skipped cells go back to pending with their errors cleared, and
// a completed cell keeps its result: that is the whole point of retrying only
// the failures.
func TestRetryingResetsOnlyFailedAndSkippedCells(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	job := bhJob("j1", benchmark.JobStatusFailed)
	job.Cells = []benchmark.JobCell{
		{ModelID: bhModel, Preset: "internal-quick", Status: benchmark.CellStatusCompleted, BenchmarkRunID: "kept"},
		{ModelID: bhModel, Preset: "internal-standard", Status: benchmark.CellStatusFailed, Error: "oom"},
		{ModelID: bhModel, Preset: "internal-thorough", Status: benchmark.CellStatusSkipped, Error: "too long"},
	}
	bhSaveJob(t, s, job)
	s.benchSvc.SetJobEnv(&stubJobEnv{})
	t.Cleanup(func() { bhStopJob(t, s, "j1") })

	w := bhServe(s.handleRetryFailedCells, bhRequest(http.MethodPost,
		"/api/benchmark-jobs/j1/retry-failed", nil, map[string]string{"id": "j1"}))
	if w.Code != http.StatusAccepted {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	if id, busy := s.benchSvc.ActiveJobID(); !busy || id != "j1" {
		t.Errorf("active job %q, %v; want j1 resubmitted", id, busy)
	}
	// Read the cells as the handler left them, before the runner moves them on.
	// The runner works on its own copy, so the completed cell is the one
	// whose result must still be on it.
	got := bhGetJob(t, s, "j1")
	if c := got.Cells[0]; c.Status != benchmark.CellStatusCompleted || c.BenchmarkRunID != "kept" {
		t.Errorf("completed cell was disturbed: %+v", c)
	}
	for _, c := range got.Cells[1:] {
		if c.Error != "" {
			t.Errorf("cell %s kept its old error %q", c.Preset, c.Error)
		}
	}
}

// Like an edit, a refused retry must not leave the job changed. The handler
// saves the reset cells and a pending status before SubmitJob refuses, so the
// job sits "pending" forever while the caller was told 409.
func TestRetryingWhileAnotherJobRunsLeavesTheJobUntouched(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	job := bhJob("idle", benchmark.JobStatusFailed)
	job.Cells[0].Status = benchmark.CellStatusFailed
	bhSaveJob(t, s, job)
	bhStartActiveJob(t, s, "busy")

	w := bhServe(s.handleRetryFailedCells, bhRequest(http.MethodPost,
		"/api/benchmark-jobs/idle/retry-failed", nil, map[string]string{"id": "idle"}))
	if w.Code != http.StatusConflict {
		t.Errorf("status %d, want 409", w.Code)
	}
	if got := bhGetJob(t, s, "idle"); got.Status != benchmark.JobStatusFailed {
		t.Errorf("status %q, want it left failed", got.Status)
	}
}

// bhDeleteJob deletes job id with the given query string.
func bhDeleteJob(s *Server, id, query string) *httptest.ResponseRecorder {
	return bhServe(s.handleDeleteJob, bhRequest(http.MethodDelete,
		"/api/benchmark-jobs/"+id+query, nil, map[string]string{"id": id}))
}

// The runner saves its copy of the job as it goes, so a deleted running job
// would reappear. It has to be cancelled first.
func TestDeletingARunningJobIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhStartActiveJob(t, s, "busy")

	w := bhDeleteJob(s, "busy", "")
	if w.Code != http.StatusConflict {
		t.Errorf("status %d, want 409", w.Code)
	}
	bhGetJob(t, s, "busy")
}

func TestDeletingAnUnknownJobIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	if w := bhDeleteJob(s, "missing", ""); w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
}

// By default a deleted job's runs are kept and moved to Ad-Hoc Runs: they
// measured something real. Cascade is the explicit opt-in to lose them.
func TestDeletingAJobOrphansOrCascadesItsRuns(t *testing.T) {
	for _, tc := range []struct {
		query     string
		wantAdhoc int
	}{
		{"", 1},
		{"?runs=orphan", 1},
		{"?runs=cascade", 0},
	} {
		t.Run(tc.query, func(t *testing.T) {
			s := newGoldenServer(t, goldenEnvGeneric)
			bhSaveJob(t, s, bhJob("j1", benchmark.JobStatusCompleted))
			if err := s.bench.Save(benchmark.BenchmarkRun{
				ID: "r1", JobID: "j1", Status: benchmark.StatusCompleted, CreatedAt: time.Now(),
			}); err != nil {
				t.Fatal(err)
			}

			if w := bhDeleteJob(s, "j1", tc.query); w.Code != http.StatusNoContent {
				t.Fatalf("status %d: %s", w.Code, w.Body)
			}
			if _, err := s.bench.GetJob("j1"); err == nil {
				t.Error("the job is still there")
			}
			if got := len(s.bench.RunsForJob(benchmark.AdhocJobID)); got != tc.wantAdhoc {
				t.Errorf("%d ad-hoc runs, want %d", got, tc.wantAdhoc)
			}
		})
	}
}

// A misspelt disposition is refused, and a refusal must not delete anything.
// Store.DeleteJob removes the job from memory before it validates the
// disposition, so the 400 comes back with the job already gone (and the
// removal reaches disk on the next unrelated save).
func TestDeletingAJobWithAnUnknownDispositionKeepsTheJob(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveJob(t, s, bhJob("j1", benchmark.JobStatusCompleted))

	if w := bhDeleteJob(s, "j1", "?runs=bogus"); w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
	if _, err := s.bench.GetJob("j1"); err != nil {
		t.Error("a refused delete removed the job")
	}
}

// bhStopJob cancels job id if it is the active one and waits for the runner
// to let go, so the temp dir is not removed under its final save.
func bhStopJob(t *testing.T, s *Server, id string) {
	t.Helper()
	s.benchSvc.CancelJob(id)
	testutil.Eventually(t, 10*time.Second, func() bool {
		_, busy := s.benchSvc.ActiveJobID()
		return !busy
	}, "job %s did not finish after being cancelled", id)
	// The runner's final save lands just after the flag clears; see
	// TestJobSubmissionSignalsSuccessOnlyWhenItSucceeded.
	time.Sleep(20 * time.Millisecond)
}

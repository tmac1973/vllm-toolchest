package benchmark

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func tempStore(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	return NewStore(dir)
}

func TestStoreEmptyList(t *testing.T) {
	s := tempStore(t)
	if got := s.List(); len(got) != 0 {
		t.Fatalf("expected empty list, got %d", len(got))
	}
	if got := s.ListJobs(); len(got) != 0 {
		t.Fatalf("expected empty jobs list, got %d", len(got))
	}
}

func TestStoreSaveAndReload(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	s1 := NewStore(dir)

	run := BenchmarkRun{
		ID:        "run-1",
		CreatedAt: time.Now(),
		Status:    StatusCompleted,
		ModelID:   "test/model",
		ModelName: "Test Model",
		Preset:    "internal-quick",
	}
	if err := s1.Save(run); err != nil {
		t.Fatal(err)
	}

	// Run should be auto-assigned to AdhocJobID since it had no JobID.
	got, err := s1.Get("run-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.JobID != AdhocJobID {
		t.Fatalf("expected JobID=%q, got %q", AdhocJobID, got.JobID)
	}

	// Reload from disk.
	s2 := NewStore(dir)
	runs := s2.List()
	if len(runs) != 1 {
		t.Fatalf("expected 1 run after reload, got %d", len(runs))
	}
	if runs[0].ID != "run-1" {
		t.Fatalf("expected run-1, got %q", runs[0].ID)
	}

	// Synthetic Ad-Hoc job should materialize since there's a run with that JobID.
	jobs := s2.ListJobs()
	if len(jobs) != 1 {
		t.Fatalf("expected 1 (synthetic) job, got %d", len(jobs))
	}
	if jobs[0].ID != AdhocJobID || jobs[0].Kind != JobKindAdhoc {
		t.Fatalf("expected synthetic adhoc job, got %+v", jobs[0])
	}
}

func TestStoreDeleteAndBatchDelete(t *testing.T) {
	s := tempStore(t)
	for i, id := range []string{"a", "b", "c", "d"} {
		s.Save(BenchmarkRun{
			ID:        id,
			CreatedAt: time.Now().Add(time.Duration(i) * time.Second),
			Status:    StatusCompleted,
		})
	}

	if err := s.Delete("a"); err != nil {
		t.Fatal(err)
	}
	if got := s.List(); len(got) != 3 {
		t.Fatalf("expected 3 after delete, got %d", len(got))
	}

	// Idempotent: deleting a missing ID is fine.
	if err := s.Delete("zzz"); err != nil {
		t.Fatalf("expected idempotent delete, got %v", err)
	}

	deleted, notFound, err := s.BatchDelete([]string{"b", "c", "missing"})
	if err != nil {
		t.Fatal(err)
	}
	if deleted != 2 || notFound != 1 {
		t.Fatalf("batch delete: expected deleted=2 notFound=1, got %d/%d", deleted, notFound)
	}
	if got := s.List(); len(got) != 1 || got[0].ID != "d" {
		t.Fatalf("expected only 'd' left, got %+v", got)
	}
}

func TestStoreJobDeleteCascadeAndOrphan(t *testing.T) {
	s := tempStore(t)

	job := BenchmarkJob{
		ID:        "job-1",
		Name:      "Test",
		Kind:      JobKindBatch,
		Status:    JobStatusCompleted,
		CreatedAt: time.Now(),
	}
	if err := s.SaveJob(job); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"r1", "r2"} {
		s.Save(BenchmarkRun{ID: id, JobID: "job-1", CreatedAt: time.Now()})
	}

	// Orphan: runs survive, reassigned to AdhocJobID.
	if err := s.DeleteJob("job-1", DeleteOrphan); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"r1", "r2"} {
		got, err := s.Get(id)
		if err != nil {
			t.Fatalf("expected %s to survive orphan, got %v", id, err)
		}
		if got.JobID != AdhocJobID {
			t.Fatalf("expected %s reassigned to adhoc, got %q", id, got.JobID)
		}
	}

	// Set up a second job + run, this time cascade-delete.
	s.SaveJob(BenchmarkJob{ID: "job-2", Kind: JobKindBatch, CreatedAt: time.Now()})
	s.Save(BenchmarkRun{ID: "r3", JobID: "job-2", CreatedAt: time.Now()})
	if err := s.DeleteJob("job-2", DeleteCascade); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get("r3"); err == nil {
		t.Fatal("expected r3 to be removed by cascade")
	}

	// Synthetic ad-hoc job cannot be saved or deleted.
	if err := s.SaveJob(newAdhocJob(time.Now())); err == nil {
		t.Fatal("expected SaveJob(adhoc) to error")
	}
	if err := s.DeleteJob(AdhocJobID, DeleteCascade); err == nil {
		t.Fatal("expected DeleteJob(adhoc) to error")
	}
}

func TestStoreRunsForJob(t *testing.T) {
	s := tempStore(t)
	s.SaveJob(BenchmarkJob{ID: "job-x", Kind: JobKindBatch, CreatedAt: time.Now()})
	s.Save(BenchmarkRun{ID: "in", JobID: "job-x", CreatedAt: time.Now()})
	s.Save(BenchmarkRun{ID: "out", CreatedAt: time.Now()}) // adhoc

	got := s.RunsForJob("job-x")
	if len(got) != 1 || got[0].ID != "in" {
		t.Fatalf("expected only 'in' for job-x, got %+v", got)
	}
}

// updateStore builds a store holding one two-cell job whose cells both
// completed, each with a run.
func updateStore(t *testing.T) (*Store, BenchmarkJob) {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	s := NewStore(dir)

	sweeps := []SweepAxis{{Field: "max_model_len", Values: []string{"8192", "32768"}}}
	job := BenchmarkJob{
		ID: "j1", Name: "ctx sweep", Kind: JobKindBatch,
		Status: JobStatusCompleted, CreatedAt: time.Now(),
		ModelIDs: []string{"m"}, Presets: []string{"quick"}, Sweeps: sweeps,
		Cells: ExpandCells([]string{"m"}, []string{"quick"}, sweeps),
	}
	for i := range job.Cells {
		job.Cells[i].Status = CellStatusCompleted
		job.Cells[i].BenchmarkRunID = "run" + strconv.Itoa(i)
	}
	if err := s.SaveJob(job); err != nil {
		t.Fatal(err)
	}
	for i := range job.Cells {
		if err := s.Save(BenchmarkRun{
			ID: "run" + strconv.Itoa(i), JobID: "j1", Status: StatusCompleted,
			CreatedAt: time.Now(),
		}); err != nil {
			t.Fatal(err)
		}
	}
	return s, job
}

// Editing keeps the job's identity. Creating a new one each time is what
// produced names like "test run (re-run) (re-run)".
func TestUpdateJobDefinitionKeepsTheJob(t *testing.T) {
	s, job := updateStore(t)

	updated, err := s.UpdateJobDefinition("j1", JobDefinition{
		Name: job.Name, ModelIDs: job.ModelIDs, Presets: job.Presets, Sweeps: job.Sweeps,
	})
	if err != nil {
		t.Fatal(err)
	}
	if updated.ID != "j1" {
		t.Errorf("job id changed to %s", updated.ID)
	}
	if got := len(s.ListJobs()); got != 1 {
		t.Errorf("expected the edit to leave one job, got %d", got)
	}
	if updated.Status != JobStatusPending {
		t.Errorf("status = %s, want pending so it runs again", updated.Status)
	}
	if !updated.FinishedAt.IsZero() {
		t.Error("finished timestamp should be cleared for the new run")
	}
}

// A cell that survives the edit unchanged keeps its result: re-running to
// change one thing should not re-measure everything, and each re-measurement
// is an engine load.
func TestUpdateJobDefinitionKeepsCompletedCells(t *testing.T) {
	s, job := updateStore(t)

	// Add a third sweep point; the first two are unchanged.
	sweeps := []SweepAxis{{Field: "max_model_len", Values: []string{"8192", "32768", "65536"}}}
	updated, err := s.UpdateJobDefinition("j1", JobDefinition{
		Name: job.Name, ModelIDs: job.ModelIDs, Presets: job.Presets, Sweeps: sweeps,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(updated.Cells) != 3 {
		t.Fatalf("expected 3 cells, got %d", len(updated.Cells))
	}
	var completed, pending int
	for _, c := range updated.Cells {
		switch c.Status {
		case CellStatusCompleted:
			completed++
		case CellStatusPending:
			pending++
		}
	}
	if completed != 2 || pending != 1 {
		t.Errorf("got %d completed and %d pending; want the two existing points kept and only the new one to run",
			completed, pending)
	}
	// Their runs stay attached to this job.
	if got := len(s.RunsForJob("j1")); got != 2 {
		t.Errorf("kept cells should keep their runs; job has %d", got)
	}
}

// A result that no longer belongs to any cell measured something real. It
// moves to the ad-hoc list rather than being deleted.
func TestUpdateJobDefinitionOrphansRatherThanDeletes(t *testing.T) {
	s, job := updateStore(t)

	// Drop a sweep point, so one cell no longer exists.
	sweeps := []SweepAxis{{Field: "max_model_len", Values: []string{"8192"}}}
	updated, err := s.UpdateJobDefinition("j1", JobDefinition{
		Name: job.Name, ModelIDs: job.ModelIDs, Presets: job.Presets, Sweeps: sweeps,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(updated.Cells) != 1 {
		t.Fatalf("expected 1 cell, got %d", len(updated.Cells))
	}
	if got := len(s.List()); got != 2 {
		t.Errorf("no run should have been deleted; store holds %d", got)
	}
	if got := len(s.RunsForJob("j1")); got != 1 {
		t.Errorf("job should keep only the surviving cell's run; got %d", got)
	}
	if got := len(s.RunsForJob(AdhocJobID)); got != 1 {
		t.Errorf("the dropped cell's run should be in Ad-Hoc Runs; got %d", got)
	}
}

// Changing what a cell measures makes it a different cell, so its old result
// must not be presented as the new one's.
func TestUpdateJobDefinitionDoesNotKeepAChangedCell(t *testing.T) {
	s, job := updateStore(t)

	sweeps := []SweepAxis{{Field: "max_model_len", Values: []string{"131072"}}}
	updated, err := s.UpdateJobDefinition("j1", JobDefinition{
		Name: job.Name, ModelIDs: job.ModelIDs, Presets: job.Presets, Sweeps: sweeps,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(updated.Cells) != 1 {
		t.Fatalf("expected 1 cell, got %d", len(updated.Cells))
	}
	if updated.Cells[0].Status != CellStatusPending {
		t.Errorf("a new sweep point must be pending, got %s", updated.Cells[0].Status)
	}
	if updated.Cells[0].BenchmarkRunID != "" {
		t.Error("a new sweep point must not inherit another point's run")
	}
}

func TestUpdateJobDefinitionRejects(t *testing.T) {
	s, _ := updateStore(t)
	if _, err := s.UpdateJobDefinition(AdhocJobID, JobDefinition{}); err == nil {
		t.Error("the synthetic ad-hoc job is not editable")
	}
	if _, err := s.UpdateJobDefinition("nope", JobDefinition{}); err == nil {
		t.Error("a missing job should be reported")
	}
}

// Sweep values live in a map, and a map's iteration order is random. Identity
// has to be stable or a cell can fail to match itself between edits.
func TestCellIdentityIsStableAcrossMapOrder(t *testing.T) {
	a := JobCell{ModelID: "m", Preset: "p",
		SweepValues: map[string]string{"max_model_len": "8192", "kv_cache_dtype": "fp8"}}
	b := JobCell{ModelID: "m", Preset: "p",
		SweepValues: map[string]string{"kv_cache_dtype": "fp8", "max_model_len": "8192"}}
	for i := 0; i < 50; i++ {
		if cellIdentity(a) != cellIdentity(b) {
			t.Fatalf("identity differs by map order: %q vs %q", cellIdentity(a), cellIdentity(b))
		}
	}
	c := JobCell{ModelID: "m", Preset: "p", SweepValues: map[string]string{"max_model_len": "32768"}}
	if cellIdentity(a) == cellIdentity(c) {
		t.Error("different sweep points must not share an identity")
	}
}

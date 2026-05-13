package benchmark

import (
	"os"
	"path/filepath"
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

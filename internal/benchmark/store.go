package benchmark

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

var noTime time.Time

// schemaVersion is the on-disk envelope version this build writes.
const schemaVersion = 1

// benchmarksFileName is the on-disk location relative to the data dir's
// config subdirectory.
const benchmarksFileName = "benchmarks.json"

// benchmarkFile is the persistence envelope.
type benchmarkFile struct {
	Version int            `json:"version"`
	Jobs    []BenchmarkJob `json:"jobs"`
	Runs    []BenchmarkRun `json:"runs"`
}

// Store manages benchmark persistence. Runs and jobs share a single file;
// it's loaded whole at startup and rewritten atomically on every save.
//
// Timing samples (passive proxy capture) live in memory only — they're
// added in Step 5 and aren't part of the persisted envelope.
type Store struct {
	mu       sync.RWMutex
	dataDir  string
	filePath string
	runs     []BenchmarkRun
	jobs     []BenchmarkJob
}

// NewStore creates a store and loads persisted benchmarks. A missing
// file is treated as an empty store, not an error.
func NewStore(dataDir string) *Store {
	s := &Store{
		dataDir:  dataDir,
		filePath: filepath.Join(dataDir, "config", benchmarksFileName),
	}
	s.load()
	return s
}

func (s *Store) load() {
	data, err := os.ReadFile(s.filePath)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Error("failed to load benchmarks.json", "error", err)
		}
		return
	}

	var bf benchmarkFile
	if err := json.Unmarshal(data, &bf); err != nil {
		slog.Error("failed to parse benchmarks.json", "error", err)
		return
	}
	s.runs = bf.Runs
	s.jobs = bf.Jobs
}

// save writes the current store state to disk atomically. Callers must
// hold s.mu (read or write — we copy slices before serializing).
func (s *Store) save() error {
	bf := benchmarkFile{
		Version: schemaVersion,
		Jobs:    s.jobs,
		Runs:    s.runs,
	}
	data, err := json.MarshalIndent(bf, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(s.filePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp := s.filePath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.filePath)
}

// List returns all benchmark runs, newest first.
func (s *Store) List() []BenchmarkRun {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]BenchmarkRun, len(s.runs))
	copy(out, s.runs)
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

// Get returns a single run by ID.
func (s *Store) Get(id string) (*BenchmarkRun, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i := range s.runs {
		if s.runs[i].ID == id {
			run := s.runs[i]
			return &run, nil
		}
	}
	return nil, fmt.Errorf("benchmark not found: %s", id)
}

// Save inserts or updates a run by ID. Runs with no JobID get reassigned
// to the synthetic AdhocJobID so every run belongs to a job.
func (s *Store) Save(run BenchmarkRun) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if run.JobID == "" {
		run.JobID = AdhocJobID
	}

	for i := range s.runs {
		if s.runs[i].ID == run.ID {
			s.runs[i] = run
			return s.save()
		}
	}
	s.runs = append(s.runs, run)
	return s.save()
}

// Delete removes a single run. Returns nil even if the run didn't exist
// so callers can treat the operation as idempotent.
func (s *Store) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i := range s.runs {
		if s.runs[i].ID == id {
			s.runs = append(s.runs[:i], s.runs[i+1:]...)
			return s.save()
		}
	}
	return nil
}

// BatchDelete removes multiple runs in one save. Returns the count of
// runs actually removed and the count not found.
func (s *Store) BatchDelete(ids []string) (deleted, notFound int, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	idSet := make(map[string]bool, len(ids))
	for _, id := range ids {
		idSet[id] = true
	}

	out := s.runs[:0]
	for _, r := range s.runs {
		if idSet[r.ID] {
			deleted++
			delete(idSet, r.ID)
			continue
		}
		out = append(out, r)
	}
	notFound = len(idSet)
	s.runs = out
	if err := s.save(); err != nil {
		return deleted, notFound, err
	}
	return deleted, notFound, nil
}

// ListJobs returns all jobs, newest first. The synthetic Ad-Hoc Runs
// pseudo-job is included if there's at least one run with JobID=AdhocJobID
// or no real jobs exist yet.
func (s *Store) ListJobs() []BenchmarkJob {
	s.mu.RLock()
	defer s.mu.RUnlock()

	jobs := make([]BenchmarkJob, len(s.jobs))
	copy(jobs, s.jobs)

	hasAdhocRuns := false
	hasAdhocJob := false
	var oldestAdhocRun = noTime
	for _, r := range s.runs {
		if r.JobID == AdhocJobID {
			hasAdhocRuns = true
			if oldestAdhocRun.IsZero() || r.CreatedAt.Before(oldestAdhocRun) {
				oldestAdhocRun = r.CreatedAt
			}
		}
	}
	for _, j := range jobs {
		if j.ID == AdhocJobID {
			hasAdhocJob = true
			break
		}
	}
	if hasAdhocRuns && !hasAdhocJob {
		jobs = append(jobs, newAdhocJob(oldestAdhocRun))
	}

	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].CreatedAt.After(jobs[j].CreatedAt)
	})
	return jobs
}

// GetJob returns a job by ID. The synthetic Ad-Hoc job is materialized
// on demand if requested.
func (s *Store) GetJob(id string) (*BenchmarkJob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i := range s.jobs {
		if s.jobs[i].ID == id {
			job := s.jobs[i]
			return &job, nil
		}
	}
	if id == AdhocJobID {
		var oldest = noTime
		for _, r := range s.runs {
			if r.JobID == AdhocJobID && (oldest.IsZero() || r.CreatedAt.Before(oldest)) {
				oldest = r.CreatedAt
			}
		}
		job := newAdhocJob(oldest)
		return &job, nil
	}
	return nil, fmt.Errorf("job not found: %s", id)
}

// SaveJob inserts or updates a job by ID.
func (s *Store) SaveJob(job BenchmarkJob) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if job.ID == AdhocJobID {
		// The Ad-Hoc job is synthesized on read; refuse to persist it.
		return errors.New("cannot persist the synthetic ad-hoc job")
	}

	for i := range s.jobs {
		if s.jobs[i].ID == job.ID {
			s.jobs[i] = job
			return s.save()
		}
	}
	s.jobs = append(s.jobs, job)
	return s.save()
}

// DeleteJob removes a job. With DeleteCascade the job's runs are also
// removed; with DeleteOrphan they're reassigned to AdhocJobID.
func (s *Store) DeleteJob(id string, disposition DeleteDisposition) error {
	if id == AdhocJobID {
		return errors.New("cannot delete the synthetic ad-hoc job")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	found := false
	out := s.jobs[:0]
	for _, j := range s.jobs {
		if j.ID == id {
			found = true
			continue
		}
		out = append(out, j)
	}
	if !found {
		return fmt.Errorf("job not found: %s", id)
	}
	s.jobs = out

	switch disposition {
	case DeleteCascade:
		runs := s.runs[:0]
		for _, r := range s.runs {
			if r.JobID == id {
				continue
			}
			runs = append(runs, r)
		}
		s.runs = runs
	case DeleteOrphan, "":
		for i := range s.runs {
			if s.runs[i].JobID == id {
				s.runs[i].JobID = AdhocJobID
			}
		}
	default:
		return fmt.Errorf("unknown delete disposition: %q", disposition)
	}
	return s.save()
}

// RunsForJob returns all runs belonging to a job, newest first.
func (s *Store) RunsForJob(jobID string) []BenchmarkRun {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []BenchmarkRun
	for _, r := range s.runs {
		if r.JobID == jobID {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

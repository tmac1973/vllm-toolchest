package benchmark

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
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
// a live view of recent inference activity, not benchmark history, and
// shouldn't bloat the on-disk envelope.
type Store struct {
	mu       sync.RWMutex
	dataDir  string
	filePath string
	runs     []BenchmarkRun
	jobs     []BenchmarkJob

	timing *timingStore
}

// NewStore creates a store and loads persisted benchmarks. A missing
// file is treated as an empty store, not an error.
func NewStore(dataDir string) *Store {
	s := &Store{
		dataDir:  dataDir,
		filePath: filepath.Join(dataDir, "config", benchmarksFileName),
		timing:   newTimingStore(),
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

// JobDefinition is the part of a job an edit may change: what to measure.
// Everything else — the id, the cells, the results — is a consequence.
type JobDefinition struct {
	Name        string
	Description string
	ModelIDs    []string
	Presets     []string
	Overrides   *ConfigOverrides
	Sweeps      []SweepAxis
}

// UpdateJobDefinition rewrites a job in place and returns it ready to submit
// again. This is what "Edit & re-run" does: the job keeps its identity, so
// re-running after a tweak does not leave a trail of near-duplicate jobs with
// names ending in "(re-run) (re-run)".
//
// Two things are deliberately preserved:
//
//   - A cell that survives the edit unchanged and had completed keeps its
//     result. Re-running a job to fix one failed cell, or to add a value to a
//     sweep, should not re-measure everything that already worked — those are
//     the expensive engine loads.
//   - A run whose cell did not survive is reassigned to the ad-hoc job rather
//     than deleted. It measured something real; the job it belonged to has
//     simply stopped claiming it.
func (s *Store) UpdateJobDefinition(id string, def JobDefinition) (*BenchmarkJob, error) {
	if id == AdhocJobID {
		return nil, errors.New("cannot edit the synthetic ad-hoc job")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	idx := -1
	for i := range s.jobs {
		if s.jobs[i].ID == id {
			idx = i
			break
		}
	}
	if idx == -1 {
		return nil, fmt.Errorf("job not found: %s", id)
	}
	job := s.jobs[idx]

	prev := make(map[string]JobCell, len(job.Cells))
	for _, c := range job.Cells {
		prev[cellIdentity(c)] = c
	}

	cells := ExpandCells(def.ModelIDs, def.Presets, def.Sweeps)
	kept := map[string]bool{}
	for i := range cells {
		old, ok := prev[cellIdentity(cells[i])]
		if !ok || old.Status != CellStatusCompleted {
			continue
		}
		cells[i] = old
		if old.BenchmarkRunID != "" {
			kept[old.BenchmarkRunID] = true
		}
	}

	for i := range s.runs {
		if s.runs[i].JobID == id && !kept[s.runs[i].ID] {
			s.runs[i].JobID = AdhocJobID
		}
	}

	job.Name = def.Name
	job.Description = def.Description
	job.ModelIDs = def.ModelIDs
	job.Presets = def.Presets
	job.Overrides = def.Overrides
	job.Sweeps = def.Sweeps
	job.Cells = cells
	job.Status = JobStatusPending
	job.StartedAt = time.Time{}
	job.FinishedAt = time.Time{}
	s.jobs[idx] = job

	if err := s.save(); err != nil {
		return nil, err
	}
	out := job
	return &out, nil
}

// cellIdentity is what makes two cells the same measurement: the model, the
// preset, and the sweep point. Sweep values are key-sorted so a map's
// iteration order cannot make a cell fail to match itself.
func cellIdentity(c JobCell) string {
	keys := make([]string, 0, len(c.SweepValues))
	for k := range c.SweepValues {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+c.SweepValues[k])
	}
	return c.ModelID + "\x00" + c.Preset + "\x00" + strings.Join(parts, ",")
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

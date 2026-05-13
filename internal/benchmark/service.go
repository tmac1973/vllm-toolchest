package benchmark

import (
	"context"
	"errors"
	"sync"
)

// ErrRunAlreadyActive is returned by StartRun when another run or job
// is in flight. vLLM is single-process and concurrent benchmarks would
// cross-contaminate timing.
var ErrRunAlreadyActive = errors.New("a benchmark run or job is already in progress")

// Service coordinates active benchmark work — ad-hoc runs and jobs share
// one queue, since both contend for the underlying vLLM process. At most
// one is active at a time.
type Service struct {
	store  *Store
	runner *Runner
	env    JobEnv // optional; required for jobs

	mu        sync.Mutex
	activeRun *activeRun
	activeJob *activeJob
}

type activeRun struct {
	id     string
	cancel context.CancelFunc
	done   chan struct{}

	subMu sync.Mutex
	subs  map[chan ProgressUpdate]struct{}
	last  *ProgressUpdate
}

type activeJob struct {
	id     string
	cancel context.CancelFunc
	done   chan struct{}
}

// NewService wires a Service over a Store. JobEnv may be set later via
// SetJobEnv; ad-hoc runs work without it.
func NewService(store *Store) *Service {
	return &Service{
		store:  store,
		runner: NewRunner(store),
	}
}

// SetJobEnv injects the JobEnv adapter from the api layer. Must be called
// before SubmitJob is used.
func (s *Service) SetJobEnv(env JobEnv) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.env = env
}

// ActiveRunID returns the ID of the in-flight ad-hoc run, if any.
func (s *Service) ActiveRunID() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.activeRun == nil {
		return "", false
	}
	return s.activeRun.id, true
}

// ActiveJobID returns the ID of the in-flight job, if any.
func (s *Service) ActiveJobID() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.activeJob == nil {
		return "", false
	}
	return s.activeJob.id, true
}

// StartRun begins an ad-hoc benchmark in a background goroutine. The run
// must already exist in the store with StatusRunning. Returns
// ErrRunAlreadyActive when either a run or a job is in flight.
func (s *Service) StartRun(cfg RunnerConfig) error {
	s.mu.Lock()
	if s.activeRun != nil || s.activeJob != nil {
		s.mu.Unlock()
		return ErrRunAlreadyActive
	}

	ctx, cancel := context.WithCancel(context.Background())
	ar := &activeRun{
		id:     cfg.Run.ID,
		cancel: cancel,
		done:   make(chan struct{}),
		subs:   make(map[chan ProgressUpdate]struct{}),
	}
	s.activeRun = ar
	s.mu.Unlock()

	progress := make(chan ProgressUpdate, 16)

	go func() {
		for update := range progress {
			update := update
			ar.subMu.Lock()
			ar.last = &update
			for sub := range ar.subs {
				select {
				case sub <- update:
				default:
				}
			}
			ar.subMu.Unlock()
		}
		ar.subMu.Lock()
		for sub := range ar.subs {
			close(sub)
			delete(ar.subs, sub)
		}
		ar.subMu.Unlock()

		s.mu.Lock()
		s.activeRun = nil
		s.mu.Unlock()
		close(ar.done)
	}()

	go s.runner.Run(ctx, cfg, progress)
	return nil
}

// CancelRun cancels the in-flight ad-hoc run, if any.
func (s *Service) CancelRun(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.activeRun == nil || s.activeRun.id != id {
		return false
	}
	s.activeRun.cancel()
	return true
}

// Subscribe registers a channel to receive progress updates for the
// in-flight ad-hoc run. Returns the subscription channel, the most
// recent update (for immediate replay), and an unsubscribe function.
// Returns (nil, nil, nil) when no run matches.
func (s *Service) Subscribe(id string) (<-chan ProgressUpdate, *ProgressUpdate, func()) {
	s.mu.Lock()
	if s.activeRun == nil || s.activeRun.id != id {
		s.mu.Unlock()
		return nil, nil, nil
	}
	ar := s.activeRun
	s.mu.Unlock()

	sub := make(chan ProgressUpdate, 16)
	ar.subMu.Lock()
	ar.subs[sub] = struct{}{}
	last := ar.last
	ar.subMu.Unlock()

	unsub := func() {
		ar.subMu.Lock()
		if _, ok := ar.subs[sub]; ok {
			delete(ar.subs, sub)
			close(sub)
		}
		ar.subMu.Unlock()
	}

	return sub, last, unsub
}

// SubmitJob persists the job (with StatusPending → JobStatusRunning at
// start) and dispatches a background goroutine to execute its cells.
// Returns ErrRunAlreadyActive when busy.
func (s *Service) SubmitJob(job BenchmarkJob) error {
	s.mu.Lock()
	if s.activeRun != nil || s.activeJob != nil {
		s.mu.Unlock()
		return ErrRunAlreadyActive
	}
	if s.env == nil {
		s.mu.Unlock()
		return errors.New("benchmark service has no JobEnv; jobs are unavailable")
	}

	ctx, cancel := context.WithCancel(context.Background())
	aj := &activeJob{
		id:     job.ID,
		cancel: cancel,
		done:   make(chan struct{}),
	}
	s.activeJob = aj
	s.mu.Unlock()

	go func() {
		s.runJob(ctx, &job)

		s.mu.Lock()
		s.activeJob = nil
		s.mu.Unlock()
		close(aj.done)
	}()
	return nil
}

// CancelJob cancels the in-flight job.
func (s *Service) CancelJob(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.activeJob == nil || s.activeJob.id != id {
		return false
	}
	s.activeJob.cancel()
	return true
}

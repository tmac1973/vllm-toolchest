package benchmark

import (
	"context"
	"errors"
	"sync"
)

// ErrRunAlreadyActive is returned by StartRun when another run is in
// flight. The benchmark service serializes runs: vLLM is single-process
// and concurrent benchmarks would cross-contaminate timing.
var ErrRunAlreadyActive = errors.New("a benchmark run is already in progress")

// Service coordinates active runs: tracks the in-flight run, fans
// progress updates out to multiple SSE subscribers, and provides
// cancellation.
type Service struct {
	store  *Store
	runner *Runner

	mu     sync.Mutex
	active *activeRun
}

type activeRun struct {
	id      string
	cancel  context.CancelFunc
	done    chan struct{}

	subMu sync.Mutex
	subs  map[chan ProgressUpdate]struct{}
	last  *ProgressUpdate
}

// NewService wires a Service over a Store. The Runner is created here so
// the api package never needs to hold one directly.
func NewService(store *Store) *Service {
	return &Service{
		store:  store,
		runner: NewRunner(store),
	}
}

// ActiveRunID returns the ID of the in-flight run, if any.
func (s *Service) ActiveRunID() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active == nil {
		return "", false
	}
	return s.active.id, true
}

// StartRun begins a benchmark in a background goroutine. The run must
// already exist in the store with StatusRunning. Returns ErrRunAlreadyActive
// when the service is busy.
func (s *Service) StartRun(cfg RunnerConfig) error {
	s.mu.Lock()
	if s.active != nil {
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
	s.active = ar
	s.mu.Unlock()

	progress := make(chan ProgressUpdate, 16)

	// Fan-out goroutine: consumes the runner's progress channel, stamps
	// the latest update onto activeRun.last (for late-joining SSE clients),
	// and broadcasts to every subscriber.
	go func() {
		for update := range progress {
			update := update
			ar.subMu.Lock()
			ar.last = &update
			for sub := range ar.subs {
				select {
				case sub <- update:
				default:
					// Slow subscriber — drop the update for them. They'll
					// pick up the next one (or close).
				}
			}
			ar.subMu.Unlock()
		}
		// Channel closed by Runner.Run — close subscriber channels too
		// so SSE handlers exit cleanly.
		ar.subMu.Lock()
		for sub := range ar.subs {
			close(sub)
			delete(ar.subs, sub)
		}
		ar.subMu.Unlock()

		s.mu.Lock()
		s.active = nil
		s.mu.Unlock()
		close(ar.done)
	}()

	go s.runner.Run(ctx, cfg, progress)
	return nil
}

// CancelRun cancels the in-flight run, if any. Returns false when no run
// is active or when the id doesn't match the active run.
func (s *Service) CancelRun(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active == nil || s.active.id != id {
		return false
	}
	s.active.cancel()
	return true
}

// Subscribe registers a channel to receive progress updates for the
// in-flight run. Returns the subscription channel, the most recent
// update (for immediate replay), and an unsubscribe function. Returns
// (nil, nil, nil) when no run matches.
//
// The returned channel is closed when the run completes.
func (s *Service) Subscribe(id string) (<-chan ProgressUpdate, *ProgressUpdate, func()) {
	s.mu.Lock()
	if s.active == nil || s.active.id != id {
		s.mu.Unlock()
		return nil, nil, nil
	}
	ar := s.active
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

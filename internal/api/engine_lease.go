package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// engineLease marks the engine as borrowed: stopped from what it was serving
// and running something else for a while. vLLM serves one model and holds
// the cards while it does, so anything that needs another model loaded --
// autoconfigure's helper -- has to take the engine away from whatever is
// serving, and give it back.
type engineLease struct {
	mu     sync.Mutex
	holder string
}

func (l *engineLease) take(holder string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.holder != "" {
		return false
	}
	l.holder = holder
	return true
}

func (l *engineLease) release() {
	l.mu.Lock()
	l.holder = ""
	l.mu.Unlock()
}

func (l *engineLease) heldBy() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.holder
}

// leasePoll is how often a loan checks the engine's state. A variable so
// tests need not wait whole seconds.
var leasePoll = time.Second

// leaseStopWait bounds each wait for the engine to stop. The manager itself
// escalates to SIGKILL at 30 seconds.
var leaseStopWait = 60 * time.Second

// leaseRestoreWait bounds the whole give-back, which runs on a context of its
// own: when the caller's context has expired is exactly when it matters.
const leaseRestoreWait = 90 * time.Second

// engineBusy is why the engine cannot be borrowed now, or "".
func (s *Server) engineBusy() string {
	if s.benchSvc != nil {
		if _, ok := s.benchSvc.ActiveRunID(); ok {
			return "A benchmark is running and using the GPUs."
		}
		if _, ok := s.benchSvc.ActiveJobID(); ok {
			return "A benchmark job is running and using the GPUs."
		}
	}
	if s.probe != nil {
		s.probe.mu.Lock()
		active := s.probe.active != nil
		s.probe.mu.Unlock()
		if active {
			return "A context probe is running."
		}
	}
	if s.tuner != nil && s.tuner.ActiveJob() != nil {
		return "Kernel tuning is running."
	}
	if s.lease.heldBy() != "" {
		return "Autoconfigure is reading a model card."
	}
	switch s.process.GetStatus().State {
	case process.StateStarting:
		return "The engine is still starting."
	case process.StateStopping:
		return "The engine is still stopping."
	}
	return ""
}

// refuseWhileBorrowed answers a request that would change what the engine is
// doing while it is on loan, and reports whether it did. A refusal renders as
// a message for htmx, which does not swap a non-2xx response.
func (s *Server) refuseWhileBorrowed(w http.ResponseWriter, r *http.Request) bool {
	if s.lease.heldBy() == "" {
		return false
	}
	const msg = "Autoconfigure is using the engine to read a model card. Try again when it has finished; whatever was serving is restarted then."
	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "error_message", msg)
		return true
	}
	http.Error(w, msg, http.StatusConflict)
	return true
}

// engineLoan is what to run while the engine is borrowed.
type engineLoan struct {
	// ModelID is what the process manager records as running.
	ModelID   string
	ModelPath string
	Args      []string
	Env       []string
	// StartWait bounds the wait for it to come up.
	StartWait time.Duration
}

// borrowEngine stops whatever is serving, runs the loan, calls work once it
// is healthy, stops it, and restarts what was serving.
//
// err is the loan's own failure: the engine would not stop, the loaned model
// would not start, or work failed. restore reports anything that went wrong
// giving the engine back, "" when nothing did. They are kept apart because
// the caller's result depends on the first and not on the second: a helper
// that answered is still an answer if the model it displaced could not be
// restarted.
//
// Manager.Restart is not used anywhere here: it calls Stop first, and Stop
// fails from the error state, which is exactly the state a failed loan leaves.
func (s *Server) borrowEngine(ctx context.Context, holder string, loan engineLoan,
	progress func(string), work func(ctx context.Context, baseURL string) error) (err error, restore string) {
	if progress == nil {
		progress = func(string) {}
	}
	if busy := s.engineBusy(); busy != "" {
		return errors.New(busy), ""
	}
	if !s.lease.take(holder) {
		return errors.New("the engine is already borrowed"), ""
	}
	defer s.lease.release()

	prev := ""
	if st := s.process.GetStatus(); st.State == process.StateRunning {
		prev = st.ModelID
	}
	if prev != "" {
		progress("Stopping " + s.displayName(prev))
		if err := s.stopEngine(); err != nil {
			return err, ""
		}
	}

	// From here on the engine is ours to give back.
	defer func() {
		restore = s.giveBack(prev, progress)
	}()

	progress("Starting the helper model")
	if err := s.process.Start(loan.ModelID, loan.ModelPath, loan.Args, loan.Env); err != nil {
		return fmt.Errorf("the helper model could not be started: %w", err), ""
	}
	if err := s.waitRunning(ctx, loan.StartWait); err != nil {
		return err, ""
	}
	return work(ctx, fmt.Sprintf("http://%s:%d", s.cfg.VLLMHost, s.cfg.VLLMPort)), ""
}

// giveBack stops the loan and restarts prev, and says what went wrong, if
// anything did.
func (s *Server) giveBack(prev string, progress func(string)) string {
	ctx, cancel := context.WithTimeout(context.Background(), leaseRestoreWait)
	defer cancel()

	switch s.process.GetStatus().State {
	case process.StateRunning, process.StateStarting:
		progress("Stopping the helper model")
		if err := s.stopEngine(); err != nil {
			if prev != "" {
				return fmt.Sprintf("the helper model did not stop, so %s was not restarted", s.displayName(prev))
			}
			return "the helper model did not stop"
		}
	}
	if prev == "" || ctx.Err() != nil {
		return ""
	}

	m, ok := s.registry.Get(prev)
	if !ok || m.Orphaned {
		return fmt.Sprintf("%s was not restarted: it is no longer in the registry", prev)
	}
	progress("Restarting " + s.displayName(prev))
	// Not waited for: a large model takes minutes to load, and nothing the
	// caller does depends on it.
	if err := s.startModel(m); err != nil {
		slog.Warn("could not restart the model autoconfigure displaced", "model", prev, "error", err)
		return fmt.Sprintf("%s was not restarted: %v", s.displayName(prev), err)
	}
	return ""
}

// stopEngine stops the engine and waits for it to be stopped.
func (s *Server) stopEngine() error {
	if st := s.process.GetStatus().State; st == process.StateStopped || st == process.StateError {
		return nil
	}
	if err := s.process.Stop(); err != nil {
		return fmt.Errorf("the engine could not be stopped: %w", err)
	}
	deadline := time.Now().Add(leaseStopWait)
	for time.Now().Before(deadline) {
		switch s.process.GetStatus().State {
		case process.StateStopped, process.StateError:
			return nil
		}
		time.Sleep(leasePoll)
	}
	return errors.New("the engine did not stop within a minute")
}

// waitRunning waits for a started engine to be serving.
func (s *Server) waitRunning(ctx context.Context, limit time.Duration) error {
	if limit <= 0 {
		limit = 10 * time.Minute
	}
	deadline := time.Now().Add(limit)
	for {
		st := s.process.GetStatus()
		switch {
		case st.State == process.StateRunning:
			return nil
		case st.State == process.StateError, st.State == process.StateStopped:
			return fmt.Errorf("the helper model did not start: %s", orUnknown(st.Error))
		case st.StartFailed:
			return errors.New("the helper model did not start: the engine reported it could not initialise")
		case st.Overdue:
			return errors.New("the helper model is taking longer than the startup timeout")
		case time.Now().After(deadline):
			return fmt.Errorf("the helper model did not start within %s", limit)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(leasePoll):
		}
	}
}

func orUnknown(s string) string {
	if s == "" {
		return "no reason given"
	}
	return s
}

// displayName is a model's name for a progress line.
func (s *Server) displayName(id string) string {
	if m, ok := s.registry.Get(id); ok && m.DisplayName != "" {
		return m.DisplayName
	}
	return id
}

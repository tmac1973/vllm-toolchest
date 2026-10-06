package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// leaseServer is a server whose engine is a script: a model path ending in
// "fail" exits at once, anything else announces it is up and waits.
func leaseServer(t *testing.T) *Server {
	t.Helper()
	oldPoll, oldStop := leasePoll, leaseStopWait
	leasePoll, leaseStopWait = 20*time.Millisecond, 5*time.Second
	t.Cleanup(func() { leasePoll, leaseStopWait = oldPoll, oldStop })

	fake := testutil.WriteScript(t, "case \"$1\" in *fail) echo 'boom' >&2; exit 1;; esac\n"+
		"echo 'INFO Application startup complete.'\nexec sleep 60\n")
	s := newTestServer(t, "http://127.0.0.1:1")
	s.process = process.NewManager("127.0.0.1", 0, 0)
	s.process.SetLauncher(process.Launcher{Bin: fake})
	t.Cleanup(func() { _ = s.process.Stop() })

	if err := s.registry.Register(&models.Model{
		ID: "org/served", DisplayName: "Served", LocalPath: "/models/org/served",
		VLLMConfig: models.VLLMConfig{MaxModelLen: 4096},
	}); err != nil {
		t.Fatal(err)
	}
	return s
}

func serve(t *testing.T, s *Server, id string) {
	t.Helper()
	m, _ := s.registry.Get(id)
	if err := s.startModel(m); err != nil {
		t.Fatal(err)
	}
	testutil.Eventually(t, 5*time.Second, func() bool { return s.process.GetStatus().State == process.StateRunning },
		"%s never came up", id)
}

var helperLoan = engineLoan{ModelID: "helper", ModelPath: "/models/helper", StartWait: 5 * time.Second}

type progressLog struct {
	mu    sync.Mutex
	lines []string
}

func (p *progressLog) add(s string) { p.mu.Lock(); p.lines = append(p.lines, s); p.mu.Unlock() }

func TestBorrowWithNothingServing(t *testing.T) {
	s := leaseServer(t)
	var during process.Status
	err, restore := s.borrowEngine(context.Background(), "test", helperLoan, nil,
		func(ctx context.Context, baseURL string) error {
			during = s.process.GetStatus()
			return nil
		})
	if err != nil || restore != "" {
		t.Fatalf("err=%v restore=%q", err, restore)
	}
	if during.ModelID != "helper" || during.State != process.StateRunning {
		t.Errorf("during the loan: %+v", during)
	}
	if st := s.process.GetStatus(); st.State != process.StateStopped {
		t.Errorf("after the loan the engine is %s, want stopped: nothing was serving", st.State)
	}
}

func TestBorrowGivesBackWhatWasServing(t *testing.T) {
	s := leaseServer(t)
	serve(t, s, "org/served")
	servedArgs := s.process.GetStatus().Args

	var log progressLog
	err, restore := s.borrowEngine(context.Background(), "test", helperLoan, log.add,
		func(ctx context.Context, baseURL string) error {
			if s.lease.heldBy() == "" {
				t.Error("the lease is not held during the loan")
			}
			return nil
		})
	if err != nil || restore != "" {
		t.Fatalf("err=%v restore=%q", err, restore)
	}
	st := s.process.GetStatus()
	if st.ModelID != "org/served" || !slices.Equal(st.Args, servedArgs) {
		t.Errorf("after the loan: %+v; want org/served back with its own args", st)
	}
	want := []string{"Stopping Served", "Starting the helper model", "Stopping the helper model", "Restarting Served"}
	if !slices.Equal(log.lines, want) {
		t.Errorf("progress = %q\nwant       %q", log.lines, want)
	}
	if s.lease.heldBy() != "" {
		t.Error("the lease was not released")
	}
}

func TestBorrowRestoresAfterAFailure(t *testing.T) {
	s := leaseServer(t)
	serve(t, s, "org/served")

	boom := errors.New("boom")
	err, restore := s.borrowEngine(context.Background(), "test", helperLoan, nil,
		func(context.Context, string) error { return boom })
	if !errors.Is(err, boom) || restore != "" {
		t.Errorf("err=%v restore=%q; want work's error and a clean restore", err, restore)
	}
	if s.process.GetStatus().ModelID != "org/served" {
		t.Error("the served model was not restored after work failed")
	}

	// The loan itself fails to start: work never runs, the model comes back.
	failing := helperLoan
	failing.ModelPath = "/models/fail"
	called := false
	err, _ = s.borrowEngine(context.Background(), "test", failing, nil,
		func(context.Context, string) error { called = true; return nil })
	if err == nil || called {
		t.Errorf("err=%v called=%v; a failed start must not reach work", err, called)
	}
	testutil.Eventually(t, 5*time.Second, func() bool { return s.process.GetStatus().ModelID == "org/served" },
		"the served model was never restored after the failed start")
}

func TestBorrowRestoresWhenTheCallerGivesUp(t *testing.T) {
	s := leaseServer(t)
	serve(t, s, "org/served")
	ctx, cancel := context.WithCancel(context.Background())
	err, _ := s.borrowEngine(ctx, "test", helperLoan, nil,
		func(ctx context.Context, _ string) error { cancel(); return ctx.Err() })
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v", err)
	}
	if s.process.GetStatus().ModelID != "org/served" {
		t.Error("a cancelled caller left the served model stopped")
	}
}

// The model is deleted while the engine is on loan: the helper's answer
// stands, and the report says why nothing came back.
func TestBorrowReportsAModelThatCannotComeBack(t *testing.T) {
	s := leaseServer(t)
	serve(t, s, "org/served")
	err, restore := s.borrowEngine(context.Background(), "test", helperLoan, nil,
		func(context.Context, string) error { return s.registry.Delete("org/served", false) })
	if err != nil {
		t.Errorf("work's success was lost: %v", err)
	}
	if !strings.Contains(restore, "no longer in the registry") {
		t.Errorf("restore = %q", restore)
	}
	if st := s.process.GetStatus().State; st != process.StateStopped {
		t.Errorf("engine is %s; nothing should have been restarted", st)
	}
}

func TestBorrowIsExclusiveAndGuardsTheHandlers(t *testing.T) {
	s := leaseServer(t)
	s.initTemplates()
	inside := make(chan struct{})
	release := make(chan struct{})
	go s.borrowEngine(context.Background(), "test", helperLoan, nil,
		func(context.Context, string) error { close(inside); <-release; return nil })
	<-inside

	if busy := s.engineBusy(); !strings.Contains(busy, "Autoconfigure") {
		t.Errorf("engineBusy = %q", busy)
	}
	if err, _ := s.borrowEngine(context.Background(), "again", helperLoan, nil,
		func(context.Context, string) error { return nil }); err == nil {
		t.Error("a second loan was granted while one was held")
	}
	for name, h := range map[string]func(w http.ResponseWriter, r *http.Request){
		"start": s.handleServiceStart, "stop": s.handleServiceStop, "restart": s.handleServiceRestart,
	} {
		rec := httptest.NewRecorder()
		h(rec, httptest.NewRequest("POST", "/", nil))
		if rec.Code != http.StatusConflict {
			t.Errorf("%s while borrowed answered %d", name, rec.Code)
		}
	}
	close(release)
	testutil.Eventually(t, 5*time.Second, func() bool { return s.lease.heldBy() == "" },
		"the lease was never released after the loan")
	if busy := s.engineBusy(); busy != "" {
		t.Errorf("still busy after the loan: %q", busy)
	}
}

func TestBorrowRecordsNoMeasurementForTheLoan(t *testing.T) {
	s := leaseServer(t)
	s.borrowEngine(context.Background(), "test", helperLoan, nil,
		func(context.Context, string) error { return nil })
	if m, ok := s.registry.Get("helper"); ok && m.Measured != nil {
		t.Error("a measurement was recorded against the loan")
	}
}

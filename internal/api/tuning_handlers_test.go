package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
	"github.com/tmac1973/vllm-toolchest/internal/tuning"
)

// tunModel is a golden fixture with a full HF config, so shapes derive.
const tunModel = "unsloth/Qwen3.8-27B-FP8"

// tunServer is a golden server whose tuner runs a shell script in place of
// python. It never stops vLLM: there is none to stop.
func tunServer(t *testing.T, script string) *Server {
	t.Helper()
	s := newGoldenServer(t, goldenEnvGeneric)
	s.tuner = tuning.NewManager(s.cfg.DataDir, "gfx-test", "tuner.py", nil)
	s.tuner.SetPython(testutil.WriteScript(t, script))
	s.tuner.SetConfigsDir(t.TempDir())
	t.Cleanup(func() {
		s.tuner.Cancel()
		testutil.Eventually(t, 15*time.Second, func() bool {
			j := s.tuner.ActiveJob()
			return j == nil || j.State != tuning.StateRunning
		}, "the tuning job did not stop")
	})
	return s
}

// tunSleeper prints two lines and then waits to be cancelled, like a tuner
// part-way through a long sweep.
const tunSleeper = "echo tuning-line-1\necho tuning-line-2\nexec sleep 60\n"

func tunStart(s *Server, body string, htmx bool) *httptest.ResponseRecorder {
	r := bhRequest(http.MethodPost, "/api/tuning/start", strings.NewReader(body), nil)
	r.Header.Set("Content-Type", "application/json")
	if htmx {
		r.Header.Set("HX-Request", "true")
	}
	return bhServe(s.handleStartTuning, r)
}

func tunStatus(t *testing.T, s *Server) (active *tuning.Job, models []modelTuningView) {
	t.Helper()
	w := bhServe(s.handleTuningStatus, bhRequest(http.MethodGet, "/api/tuning/status", nil, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	var body struct {
		DeviceName string            `json:"device_name"`
		Active     *tuning.Job       `json:"active"`
		Models     []modelTuningView `json:"models"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.DeviceName != "gfx-test" {
		t.Errorf("device_name %q", body.DeviceName)
	}
	return body.Active, body.Models
}

func tunLogs(t *testing.T, s *Server, query string) []string {
	t.Helper()
	w := bhServe(s.handleTuningLogs, bhRequest(http.MethodGet, "/api/tuning/logs"+query, nil, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	var lines []string
	if err := json.Unmarshal(w.Body.Bytes(), &lines); err != nil {
		t.Fatal(err)
	}
	return lines
}

func TestTuningStartWithMalformedJSONIsRefused(t *testing.T) {
	s := tunServer(t, tunSleeper)
	w := tunStart(s, `{"model_id":`, false)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid JSON") {
		t.Errorf("plain: status %d %q, want 400 invalid JSON", w.Code, w.Body)
	}
	// htmx is told why in a 200, or the button does nothing visible.
	w = tunStart(s, `{"model_id":`, true)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "invalid JSON") {
		t.Errorf("htmx: status %d %q, want 200 with the reason", w.Code, w.Body)
	}
	if s.tuner.ActiveJob() != nil {
		t.Error("a refused start began a job")
	}
}

func TestTuningStartWithoutAModelIsRefused(t *testing.T) {
	s := tunServer(t, tunSleeper)
	if w := tunStart(s, `{}`, false); w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
	if s.tuner.ActiveJob() != nil {
		t.Error("a refused start began a job")
	}
}

func TestTuningStartForAnUnknownModelIsNotFound(t *testing.T) {
	s := tunServer(t, tunSleeper)
	if w := tunStart(s, `{"model_id":"nobody/nothing"}`, false); w.Code != http.StatusNotFound {
		t.Errorf("status %d, want 404", w.Code)
	}
	if s.tuner.ActiveJob() != nil {
		t.Error("a refused start began a job")
	}
}

// With no hidden size there are no GEMM shapes, and a job over nothing would
// stop vLLM for no result.
func TestTuningStartForAModelWithNoShapesIsRefused(t *testing.T) {
	s := tunServer(t, tunSleeper)
	if err := s.registry.Register(&models.Model{ID: "bare/model", DisplayName: "bare"}); err != nil {
		t.Fatal(err)
	}
	if w := tunStart(s, `{"model_id":"bare/model"}`, false); w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
}

// Status reports nothing active before a job, then the job while it runs;
// the logs carry its output; cancel ends it. A second start while one runs is
// refused, since the tuner wants every card to itself.
func TestTuningStatusLogsAndCancelFollowAJob(t *testing.T) {
	s := tunServer(t, tunSleeper)

	if active, views := tunStatus(t, s); active != nil {
		t.Errorf("active before any job: %+v", active)
	} else if len(views) == 0 {
		t.Error("status lists no models")
	}

	w := tunStart(s, `{"model_id":"`+tunModel+`"}`, false)
	if w.Code != http.StatusAccepted {
		t.Fatalf("start: status %d: %s", w.Code, w.Body)
	}
	var started tuning.Job
	if err := json.Unmarshal(w.Body.Bytes(), &started); err != nil {
		t.Fatal(err)
	}
	if started.ModelID != tunModel || len(started.Shapes) == 0 {
		t.Errorf("started job %+v", started)
	}

	if active, _ := tunStatus(t, s); active == nil || active.ID != started.ID || active.State != tuning.StateRunning {
		t.Errorf("status while running: %+v", active)
	}

	testutil.Eventually(t, 5*time.Second, func() bool {
		return strings.Contains(strings.Join(tunLogs(t, s, ""), "\n"), "tuning-line-2")
	}, "the tuner's output never reached the logs")
	if got := tunLogs(t, s, "?limit=1"); len(got) != 1 || got[0] != "tuning-line-2" {
		t.Errorf("limit=1 gave %q, want the last line", got)
	}
	// An out-of-range limit falls back to the default rather than failing.
	if got := tunLogs(t, s, "?limit=0"); len(got) < 2 {
		t.Errorf("limit=0 gave %d lines", len(got))
	}

	if w := tunStart(s, `{"model_id":"`+tunModel+`"}`, false); w.Code != http.StatusConflict {
		t.Errorf("second start: status %d, want 409", w.Code)
	}

	w = bhServe(s.handleCancelTuning, bhRequest(http.MethodPost, "/api/tuning/cancel", nil, nil))
	if w.Code != http.StatusNoContent {
		t.Errorf("cancel: status %d, want 204", w.Code)
	}
	testutil.Eventually(t, 15*time.Second, func() bool {
		active, _ := tunStatus(t, s)
		return active != nil && active.State == tuning.StateCancelled
	}, "the job never reported cancelled")
}

// Cancel is safe to press with nothing running: the page shows the button
// from a stale status poll.
func TestTuningCancelWithNothingRunningIsHarmless(t *testing.T) {
	s := tunServer(t, tunSleeper)
	w := bhServe(s.handleCancelTuning, bhRequest(http.MethodPost, "/api/tuning/cancel", nil, nil))
	if w.Code != http.StatusNoContent {
		t.Errorf("status %d, want 204", w.Code)
	}
	if got := tunLogs(t, s, ""); len(got) != 0 {
		t.Errorf("logs before any job: %q", got)
	}
}

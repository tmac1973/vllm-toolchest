package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// bhModel is a model the golden server registers, for the handlers that
// refuse anything the registry does not know.
const bhModel = "TheBloke/Mixtral-8x7B-AWQ"

// bhRequest builds a request carrying chi's URL parameters, so a handler can
// be called directly without standing up the whole router.
func bhRequest(method, target string, body io.Reader, params map[string]string) *http.Request {
	r := httptest.NewRequest(method, target, body)
	rc := chi.NewRouteContext()
	for k, v := range params {
		rc.URLParams.Add(k, v)
	}
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
}

// bhServe runs one handler against r and returns the recorder.
func bhServe(h http.HandlerFunc, r *http.Request) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	h(w, r)
	return w
}

// bhSaveRun stores a finished ad-hoc run.
func bhSaveRun(t *testing.T, s *Server, id string) {
	t.Helper()
	if err := s.bench.Save(benchmark.BenchmarkRun{
		ID: id, JobID: benchmark.AdhocJobID, Status: benchmark.StatusCompleted,
		CreatedAt: time.Now(), ModelID: bhModel, ModelName: "Mixtral", Preset: "internal-quick",
	}); err != nil {
		t.Fatal(err)
	}
}

// bhStartActiveRun puts a run in flight against an engine that never answers,
// so the service reports it active for as long as the test needs. It is
// cancelled, and its last save waited for, before the temp dir is removed.
func bhStartActiveRun(t *testing.T, s *Server, id string) {
	t.Helper()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	t.Cleanup(engine.Close)

	run := benchmark.BenchmarkRun{
		ID: id, JobID: benchmark.AdhocJobID, Status: benchmark.StatusRunning,
		CreatedAt: time.Now(), ModelID: bhModel, Preset: "internal-quick",
	}
	if err := s.bench.Save(run); err != nil {
		t.Fatal(err)
	}
	preset, _ := benchmark.LookupPreset("internal-quick")
	if err := s.benchSvc.StartRun(benchmark.RunnerConfig{
		Run: run, Preset: preset, VLLMURL: engine.URL, ServedName: bhModel,
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		s.benchSvc.CancelRun(id)
		// The runner's final save comes before the service clears the run, so
		// once it is cleared nothing is writing into the directory.
		testutil.Eventually(t, 10*time.Second, func() bool {
			_, busy := s.benchSvc.ActiveRunID()
			return !busy
		}, "run %s did not finish after being cancelled", id)
	})
}

// bhStartActiveJob saves a job and submits it to a runner that blocks before
// loading anything, so it stays active until cancelled at cleanup.
func bhStartActiveJob(t *testing.T, s *Server, id string) {
	t.Helper()
	s.benchSvc.SetJobEnv(&stubJobEnv{})
	job := bhJob(id, benchmark.JobStatusPending)
	if err := s.bench.SaveJob(job); err != nil {
		t.Fatal(err)
	}
	if err := s.benchSvc.SubmitJob(job); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { bhStopJob(t, s, id) })
}

// bhJob is a one-cell job over the golden server's Mixtral.
func bhJob(id, status string) benchmark.BenchmarkJob {
	return benchmark.BenchmarkJob{
		ID: id, Name: "job " + id, Kind: benchmark.JobKindBatch, Status: status,
		CreatedAt: time.Now(), ModelIDs: []string{bhModel}, Presets: []string{"internal-quick"},
		Cells: benchmark.ExpandCells([]string{bhModel}, []string{"internal-quick"}, nil),
	}
}

// bhStartFakeEngine starts a stand-in vLLM process so the process manager
// reports it live: starting, or running once ready is true and it has said so.
func bhStartFakeEngine(t *testing.T, s *Server, modelID string, ready bool) {
	t.Helper()
	body := "exec sleep 60\n"
	if ready {
		body = "echo 'INFO: Application startup complete.'\nexec sleep 60\n"
	}
	s.process = process.NewManager("127.0.0.1", 0, 0)
	s.process.SetLauncher(process.Launcher{Bin: testutil.WriteScript(t, body)})
	if err := s.process.Start(modelID, t.TempDir(), nil, nil); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.process.Stop() })
	want := process.StateStarting
	if ready {
		want = process.StateRunning
	}
	testutil.Eventually(t, 5*time.Second, func() bool {
		return s.process.GetStatus().State == want
	}, "the fake engine never reached %s", want)
}

// Deleting a finished run removes it, and deleting one that is already gone
// is not an error: the UI's delete button can be clicked twice.
func TestDeletingARunRemovesItAndIsIdempotent(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveRun(t, s, "r1")

	for i := 0; i < 2; i++ {
		w := bhServe(s.handleDeleteBenchmark,
			bhRequest(http.MethodDelete, "/api/benchmarks/r1", nil, map[string]string{"id": "r1"}))
		if w.Code != http.StatusNoContent {
			t.Fatalf("delete #%d: status %d, want 204", i+1, w.Code)
		}
	}
	if _, err := s.bench.Get("r1"); err == nil {
		t.Error("the run is still in the store")
	}
}

// A run being measured is saved again on every progress update, so deleting
// it would only see it come back. The handler refuses instead.
func TestDeletingARunningRunIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhStartActiveRun(t, s, "live")

	w := bhServe(s.handleDeleteBenchmark,
		bhRequest(http.MethodDelete, "/api/benchmarks/live", nil, map[string]string{"id": "live"}))
	if w.Code != http.StatusConflict {
		t.Errorf("status %d, want 409", w.Code)
	}
	if _, err := s.bench.Get("live"); err != nil {
		t.Error("the running run was removed")
	}
}

func TestBatchDeleteRemovesTheSelectedRunsAndCountsTheMissing(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	for _, id := range []string{"a", "b", "c"} {
		bhSaveRun(t, s, id)
	}

	w := bhServe(s.handleBatchDeleteBenchmarks,
		bhRequest(http.MethodDelete, "/api/benchmarks/batch-delete?ids=a,%20b,,gone", nil, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	var got map[string]int
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["deleted"] != 2 || got["not_found"] != 1 {
		t.Errorf("got %v, want 2 deleted and 1 not found", got)
	}
	if runs := s.bench.List(); len(runs) != 1 || runs[0].ID != "c" {
		t.Errorf("left %v, want only c", runs)
	}
}

// An empty selection is a client bug, not a request to delete nothing.
func TestBatchDeleteWithNothingSelectedIsABadRequest(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	for _, q := range []string{"", "?ids=", "?ids=,%20,"} {
		w := bhServe(s.handleBatchDeleteBenchmarks,
			bhRequest(http.MethodDelete, "/api/benchmarks/batch-delete"+q, nil, nil))
		if w.Code != http.StatusBadRequest {
			t.Errorf("%q: status %d, want 400", q, w.Code)
		}
	}
}

// One running run in the selection refuses the whole batch, rather than
// deleting the rest and leaving the caller to work out which one survived.
func TestBatchDeleteIncludingARunningRunDeletesNothing(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveRun(t, s, "done")
	bhStartActiveRun(t, s, "live")

	w := bhServe(s.handleBatchDeleteBenchmarks,
		bhRequest(http.MethodDelete, "/api/benchmarks/batch-delete?ids=done,live", nil, nil))
	if w.Code != http.StatusConflict {
		t.Errorf("status %d, want 409", w.Code)
	}
	if len(s.bench.List()) != 2 {
		t.Error("a refused batch still deleted something")
	}
}

func TestCompareNeedsAtLeastTwoRuns(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveRun(t, s, "a")
	for _, q := range []string{"", "?ids=a", "?ids=a,,%20"} {
		w := bhServe(s.handleCompareBenchmarks,
			bhRequest(http.MethodGet, "/api/benchmarks/compare"+q, nil, nil))
		if w.Code != http.StatusBadRequest {
			t.Errorf("%q: status %d, want 400", q, w.Code)
		}
	}
}

// A run that has been deleted since the page was drawn is named in the
// error, so the user knows which selection went stale.
func TestCompareNamesAMissingRun(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveRun(t, s, "a")
	w := bhServe(s.handleCompareBenchmarks,
		bhRequest(http.MethodGet, "/api/benchmarks/compare?ids=a,gone", nil, nil))
	if w.Code != http.StatusNotFound {
		t.Errorf("status %d, want 404", w.Code)
	}
	if !strings.Contains(w.Body.String(), "gone") {
		t.Errorf("the missing id is not named: %q", w.Body)
	}
}

func TestCompareAnswersJSONOrHTMLByCaller(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveRun(t, s, "a")
	bhSaveRun(t, s, "b")

	w := bhServe(s.handleCompareBenchmarks,
		bhRequest(http.MethodGet, "/api/benchmarks/compare?ids=a,b", nil, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	var cmp benchmark.Comparison
	if err := json.Unmarshal(w.Body.Bytes(), &cmp); err != nil {
		t.Fatalf("plain caller did not get JSON: %v", err)
	}
	if len(cmp.Rows) != 2 {
		t.Errorf("%d rows, want 2", len(cmp.Rows))
	}

	r := bhRequest(http.MethodGet, "/api/benchmarks/compare?ids=a,b", nil, nil)
	r.Header.Set("HX-Request", "true")
	w = bhServe(s.handleCompareBenchmarks, r)
	if w.Code != http.StatusOK || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/html") {
		t.Errorf("htmx caller got status %d, type %q", w.Code, w.Header().Get("Content-Type"))
	}
	if strings.Contains(w.Body.String(), "partial not found") || strings.Contains(w.Body.String(), "render error") {
		t.Errorf("the comparison partial did not render: %q", w.Body)
	}
}

func TestCancellingARunThatIsNotRunningIsNotFound(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhSaveRun(t, s, "done")
	w := bhServe(s.handleCancelBenchmark,
		bhRequest(http.MethodPost, "/api/benchmarks/done/cancel", nil, map[string]string{"id": "done"}))
	if w.Code != http.StatusNotFound {
		t.Errorf("status %d, want 404", w.Code)
	}
}

// Cancelling stops the run and the service lets go of it, which is what frees
// vLLM for the next run or job.
func TestCancellingTheActiveRunStopsIt(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhStartActiveRun(t, s, "live")

	// Another id does not cancel the run that is going.
	w := bhServe(s.handleCancelBenchmark,
		bhRequest(http.MethodPost, "/api/benchmarks/other/cancel", nil, map[string]string{"id": "other"}))
	if w.Code != http.StatusNotFound {
		t.Errorf("wrong id: status %d, want 404", w.Code)
	}

	w = bhServe(s.handleCancelBenchmark,
		bhRequest(http.MethodPost, "/api/benchmarks/live/cancel", nil, map[string]string{"id": "live"}))
	if w.Code != http.StatusAccepted {
		t.Fatalf("status %d, want 202", w.Code)
	}
	testutil.Eventually(t, 10*time.Second, func() bool {
		_, busy := s.benchSvc.ActiveRunID()
		return !busy
	}, "the run was still active after cancel")
	run, err := s.bench.Get("live")
	if err != nil {
		t.Fatal(err)
	}
	if run.Status != benchmark.StatusFailed {
		t.Errorf("status %q, want failed after cancel", run.Status)
	}
}

// bhPostRun posts a start request with the given content type.
func bhPostRun(s *Server, contentType, body string) *httptest.ResponseRecorder {
	r := bhRequest(http.MethodPost, "/api/benchmarks/", strings.NewReader(body), nil)
	r.Header.Set("Content-Type", contentType)
	return bhServe(s.handleStartBenchmark, r)
}

// A typo in the preset is refused by name. Falling back to a default would
// measure something the caller did not ask for and label it as if they had.
func TestStartingARunWithAnUnknownPresetIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := bhPostRun(s, "application/json", `{"model_id":"`+bhModel+`","preset":"internal-quik"}`)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "unknown preset") {
		t.Errorf("body %q does not say why", w.Body)
	}
	if len(s.bench.List()) != 0 {
		t.Error("a refused start saved a run")
	}
}

func TestStartingARunWithoutAModelOrPresetIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	for _, body := range []string{
		`{}`,
		`{"model_id":"` + bhModel + `"}`,
		`{"preset":"internal-quick"}`,
	} {
		w := bhPostRun(s, "application/json", body)
		if w.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400", body, w.Code)
		}
	}
}

func TestStartingARunWithMalformedJSONIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := bhPostRun(s, "application/json", `{"model_id":`)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "invalid JSON") {
		t.Errorf("body %q does not say the JSON was bad", w.Body)
	}
}

// The page's form posts urlencoded fields, not JSON. If they were not read
// the request would fail as "required" no matter what was chosen; reaching
// the preset check and the engine check shows the fields arrived.
func TestStartingARunReadsAFormBody(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	form := url.Values{"model_id": {bhModel}, "preset": {"nope"}}
	w := bhPostRun(s, "application/x-www-form-urlencoded", form.Encode())
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "unknown preset") {
		t.Errorf("status %d %q, want the preset refused", w.Code, w.Body)
	}

	form.Set("preset", "internal-quick")
	w = bhPostRun(s, "application/x-www-form-urlencoded", form.Encode())
	if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "not running") {
		t.Errorf("status %d %q, want refusal because vLLM is stopped", w.Code, w.Body)
	}
}

func TestStartingARunForAnUnregisteredModelIsNotFound(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := bhPostRun(s, "application/json", `{"model_id":"nobody/nothing","preset":"internal-quick"}`)
	if w.Code != http.StatusNotFound {
		t.Errorf("status %d, want 404", w.Code)
	}
}

// Ad-hoc runs measure whatever vLLM is serving, so asking for a different
// model than the loaded one is refused rather than benchmarking the wrong one.
func TestStartingARunForAModelThatIsNotLoadedIsRefused(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	bhStartFakeEngine(t, s, "unsloth/Qwen3.8-27B-FP8", true)

	w := bhPostRun(s, "application/json", `{"model_id":"`+bhModel+`","preset":"internal-quick"}`)
	if w.Code != http.StatusConflict {
		t.Errorf("status %d, want 409", w.Code)
	}
	if !strings.Contains(w.Body.String(), "currently serving") {
		t.Errorf("body %q does not say what is loaded", w.Body)
	}
}

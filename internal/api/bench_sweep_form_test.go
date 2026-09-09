package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
)

// postJob submits the batch-job form the way the browser does.
//
// The job is persisted before it is handed to the runner, and these tests have
// no JobEnv — there is no vLLM to load models into — so a well-formed request
// saves the job and then fails to start it. That is the right split for what
// is under test here: parsing and persistence happen before the runner is
// involved at all. A request rejected during parsing never reaches the save,
// which is what the bad-value test checks.
func postJob(t *testing.T, s *Server, form url.Values) (int, string) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/api/benchmark-jobs/", strings.NewReader(form.Encode()))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	r.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.handleCreateJob(w, r)
	return w.Code, w.Body.String()
}

// sweepOf finds one axis of the most recently created job.
func sweepOf(t *testing.T, s *Server, field string) []string {
	t.Helper()
	jobs := s.bench.ListJobs()
	if len(jobs) == 0 {
		t.Fatal("no job was created")
	}
	for _, a := range jobs[0].Sweeps {
		if a.Field == field {
			return a.Values
		}
	}
	return nil
}

func jobForm(t *testing.T, s *Server, query string) string {
	t.Helper()
	r := httptest.NewRequest(http.MethodGet, "/api/benchmark-jobs/form"+query, nil)
	r.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.handleJobForm(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	return w.Body.String()
}

// Ticking several boxes for one parameter submits the field repeatedly. That
// is the whole point of the multi-select menus, and it is not the shape the
// handler originally read.
func TestSweepAcceptsRepeatedFormValues(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	postJob(t, s, url.Values{
		"name":                       {"repeated"},
		"model_ids":                  {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":                    {"internal-quick"},
		"sweep_max_model_len":        {"8192", "32768"},
		"sweep_tensor_parallel_size": {"1", "2"},
	})
	if got := sweepOf(t, s, "max_model_len"); strings.Join(got, ",") != "8192,32768" {
		t.Errorf("max_model_len = %v", got)
	}
	if got := sweepOf(t, s, "tensor_parallel_size"); strings.Join(got, ",") != "1,2" {
		t.Errorf("tensor_parallel_size = %v", got)
	}
}

// The comma-separated shape is what a script or an older client sends, and
// dropping it would break them silently — the field would simply be ignored.
func TestSweepStillAcceptsACommaSeparatedValue(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	postJob(t, s, url.Values{
		"name":                {"comma"},
		"model_ids":           {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":             {"internal-quick"},
		"sweep_max_model_len": {"8192, 32768, 65536"},
	})
	if got := sweepOf(t, s, "max_model_len"); strings.Join(got, ",") != "8192,32768,65536" {
		t.Errorf("got %v", got)
	}
}

// Both shapes in one request is what a form with a restored custom value plus
// freshly ticked boxes could produce. Duplicates across them must collapse, or
// the same configuration is loaded twice.
func TestSweepMergesBothShapesAndDeduplicates(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	postJob(t, s, url.Values{
		"name":                {"mixed"},
		"model_ids":           {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":             {"internal-quick"},
		"sweep_max_model_len": {"8192", "8192, 32768"},
	})
	if got := sweepOf(t, s, "max_model_len"); strings.Join(got, ",") != "8192,32768" {
		t.Errorf("got %v, want the duplicate collapsed", got)
	}
}

// A value the curated list does not offer has to come back ticked when the job
// is re-run. Dropping it would quietly change what the re-run measures.
func TestJobFormRestoresACustomValueAsATickedOption(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	postJob(t, s, url.Values{
		"name":                {"custom"},
		"model_ids":           {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":             {"internal-quick"},
		"sweep_max_model_len": {"12000", "8192"},
	})
	jobs := s.bench.ListJobs()
	body := jobForm(t, s, "?from="+jobs[0].ID)

	if !strings.Contains(body, `value="12000"`) {
		t.Error("the custom value is missing from the re-run form")
	}
	if !strings.Contains(body, "(custom)") {
		t.Error("the custom value is not marked as one")
	}
	// And the curated values it shares the row with are still offered.
	for _, v := range []string{"4096", "8192", "131072"} {
		if !strings.Contains(body, `value="`+v+`"`) {
			t.Errorf("curated value %s went missing", v)
		}
	}
}

// The checkbox name decides which parameter a value lands on. Getting this
// wrong is silent: every box would submit under one name and the job would
// sweep the wrong axis.
func TestJobFormCheckboxesCarryTheirOwnFieldName(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	body := jobForm(t, s, "")
	for _, f := range benchmark.SweepFields() {
		if !strings.Contains(body, `name="sweep_`+f.Name+`"`) {
			t.Errorf("no checkbox is named sweep_%s", f.Name)
		}
	}
	if strings.Contains(body, `name="sweep_"`) {
		t.Error(`a checkbox was rendered with a bare name="sweep_"`)
	}
}

// The closed menu has to read correctly before any JavaScript runs, or a
// re-opened job shows "use the saved value" over three ticked boxes.
func TestJobFormRendersTheSummaryServerSide(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	postJob(t, s, url.Values{
		"name":                       {"summary"},
		"model_ids":                  {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":                    {"internal-quick"},
		"sweep_max_model_len":        {"8192", "32768"},
		"sweep_tensor_parallel_size": {"2"},
	})
	jobs := s.bench.ListJobs()
	body := jobForm(t, s, "?from="+jobs[0].ID)

	if !strings.Contains(body, "8192, 32768 — sweep") {
		t.Error("a swept parameter should say so in its summary")
	}
	if !strings.Contains(body, `data-active="sweep"`) {
		t.Error("a swept parameter should be marked as swept")
	}
	if !strings.Contains(body, `data-active="fixed"`) {
		t.Error("a single value fixes the parameter and should be marked fixed")
	}
}

func TestSweepSummary(t *testing.T) {
	cases := []struct {
		in   []string
		want string
	}{
		{nil, "Use model's saved value"},
		{[]string{"8192"}, "8192"},
		{[]string{"8192", "32768"}, "8192, 32768 — sweep"},
	}
	for _, tc := range cases {
		if got := sweepSummary(tc.in); got != tc.want {
			t.Errorf("sweepSummary(%v) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// An unparseable value is rejected at submit with the parameter named, rather
// than stored and discovered several engine loads into the job.
//
// htmx does not swap a non-2xx response, so a rejection sent as an HTTP error
// is invisible to the page: the button clicks and nothing happens. The htmx
// caller therefore gets 200 and the message; everyone else keeps the status
// code.
func TestSweepRejectsABadValue(t *testing.T) {
	bad := url.Values{
		"name":                {"bad"},
		"model_ids":           {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":             {"internal-quick"},
		"sweep_max_model_len": {"not-a-number"},
	}

	t.Run("htmx sees the reason", func(t *testing.T) {
		s := newGoldenServer(t, goldenEnvGeneric)
		code, body := postJob(t, s, bad)
		if code != http.StatusOK {
			t.Errorf("status %d, want 200 so htmx swaps it", code)
		}
		if !strings.Contains(body, "Context length") {
			t.Errorf("the message should name the parameter: %q", body)
		}
		if len(s.bench.ListJobs()) != 0 {
			t.Error("a rejected submission must not create a job")
		}
	})

	t.Run("a plain client keeps the status code", func(t *testing.T) {
		s := newGoldenServer(t, goldenEnvGeneric)
		r := httptest.NewRequest(http.MethodPost, "/api/benchmark-jobs/",
			strings.NewReader(bad.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		w := httptest.NewRecorder()
		s.handleCreateJob(w, r)
		if w.Code != http.StatusBadRequest {
			t.Errorf("status %d, want 400", w.Code)
		}
	})
}

// stubJobEnv is the least that lets SubmitJob accept a job. It never gets as
// far as loading anything: the runner blocks on EnsureModelLoaded, and the
// test only cares what the handler returned.
type stubJobEnv struct{ blocked chan struct{} }

func (e *stubJobEnv) ResolveModel(id string) (benchmark.ModelInfo, error) {
	return benchmark.ModelInfo{HFRepoID: id, ServedName: id}, nil
}
func (e *stubJobEnv) CurrentLoadedModel() string { return "" }
func (e *stubJobEnv) EnsureModelLoaded(ctx context.Context, _ string, _ benchmark.ConfigSnapshot) error {
	<-ctx.Done()
	return ctx.Err()
}
func (e *stubJobEnv) CurrentMetrics() monitor.Metrics { return monitor.Metrics{} }
func (e *stubJobEnv) VLLMURL() string                 { return "http://127.0.0.1:1" }
func (e *stubJobEnv) HFToken() string                 { return "" }
func (e *stubJobEnv) HFCacheDir() string              { return "" }
func (e *stubJobEnv) VLLMVersion() string             { return "" }

// The page closes the editor on this trigger, so its absence on a rejection is
// what keeps a bad submission's form open with its error showing.
func TestJobSubmissionSignalsSuccessOnlyWhenItSucceeded(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.benchSvc.SetJobEnv(&stubJobEnv{})
	// Cancelling only signals; the runner's goroutine keeps writing to the
	// store for a moment after. t.TempDir removes the directory as soon as the
	// test returns, and deleting one the store is still saving into fails with
	// "directory not empty" — a flake with nothing to do with what is asserted.
	t.Cleanup(func() {
		if id, busy := s.benchSvc.ActiveJobID(); busy {
			s.benchSvc.CancelJob(id)
		}
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			if _, busy := s.benchSvc.ActiveJobID(); !busy {
				// The final save lands just after the flag clears; give it the
				// scheduler slot rather than racing it.
				time.Sleep(20 * time.Millisecond)
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
		t.Error("job did not finish within 5s of being cancelled")
	})

	post := func(form url.Values) *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodPost, "/api/benchmark-jobs/",
			strings.NewReader(form.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		r.Header.Set("HX-Request", "true")
		w := httptest.NewRecorder()
		s.handleCreateJob(w, r)
		return w
	}

	rejected := post(url.Values{"name": {"no models"}, "presets": {"internal-quick"}})
	if got := rejected.Result().Header.Get("HX-Trigger"); got != "" {
		t.Errorf("a rejected submission signalled %q; the form would close over the error", got)
	}

	accepted := post(url.Values{
		"name":      {"good"},
		"model_ids": {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":   {"internal-quick"},
	})
	if got := accepted.Result().Header.Get("HX-Trigger"); got != "jobSubmitted" {
		t.Errorf("HX-Trigger = %q, want jobSubmitted", got)
	}
	// The confirmation is swapped out-of-band, since the form it was
	// submitted from is about to close.
	if body := accepted.Body.String(); !strings.Contains(body, `hx-swap-oob="true"`) ||
		!strings.Contains(body, "bench-notice") {
		t.Errorf("confirmation is not an out-of-band notice: %q", body)
	}
}

// The form's cell cap and the server's are the same number; a form that let
// you build a job the server then refuses would be worse than no preview.
func TestJobFormAdvertisesTheServersCap(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	body := jobForm(t, s, "")
	if !strings.Contains(body, `data-max-cells="64"`) {
		t.Error("the form does not carry the combination cap")
	}
	if benchmark.MaxSweepCombinations != 64 {
		t.Errorf("cap changed to %d; update this test and the form's expectation",
			benchmark.MaxSweepCombinations)
	}
}

// The Ad-Hoc Runs row is synthesized with a fixed "completed" status, so it
// claimed completed while a run inside it was still going — and expanding the
// row showed a running run directly under the badge contradicting it.
func TestAdhocRowStatusFollowsItsRuns(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)

	save := func(id, status string) {
		if err := s.bench.Save(benchmark.BenchmarkRun{
			ID: id, JobID: benchmark.AdhocJobID, Status: status,
			CreatedAt: time.Now(), ModelID: "m", ModelName: "m",
		}); err != nil {
			t.Fatal(err)
		}
	}
	adhoc := func() jobRow {
		for _, j := range s.bench.ListJobs() {
			if j.ID == benchmark.AdhocJobID {
				return s.jobSummary(j)
			}
		}
		t.Fatal("no ad-hoc job")
		return jobRow{}
	}

	save("r1", benchmark.StatusCompleted)
	if got := adhoc().Status; got != benchmark.JobStatusCompleted {
		t.Errorf("with nothing running the row should read completed, got %q", got)
	}

	save("r2", benchmark.StatusRunning)
	row := adhoc()
	if row.Status != benchmark.JobStatusRunning {
		t.Errorf("a running run should make the row read running, got %q", row.Status)
	}
	if row.RunCount != 2 {
		t.Errorf("RunCount = %d, want 2", row.RunCount)
	}

	// And it goes back once the run lands.
	save("r2", benchmark.StatusCompleted)
	if got := adhoc().Status; got != benchmark.JobStatusCompleted {
		t.Errorf("after the run finished the row should read completed, got %q", got)
	}
}

// The list only polls while something is running: an idle history of hundreds
// of runs should not re-render itself every two seconds forever.
func TestAdhocListPollsOnlyWhileRunning(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)

	render := func() string {
		var b strings.Builder
		s.renderRunList(&stringWriter{&b}, s.bench.RunsForJob(benchmark.AdhocJobID))
		return b.String()
	}

	s.bench.Save(benchmark.BenchmarkRun{ID: "r1", JobID: benchmark.AdhocJobID,
		Status: benchmark.StatusCompleted, CreatedAt: time.Now(), ModelID: "m"})
	if strings.Contains(render(), "every 2s") {
		t.Error("an idle list should not poll")
	}

	s.bench.Save(benchmark.BenchmarkRun{ID: "r2", JobID: benchmark.AdhocJobID,
		Status: benchmark.StatusRunning, CreatedAt: time.Now(), ModelID: "m"})
	out := render()
	if !strings.Contains(out, "every 2s") {
		t.Error("a running list should poll so the rows and badge stay current")
	}
	if !strings.Contains(out, `id="job-status-adhoc"`) {
		t.Error("the list should carry the out-of-band badge update")
	}
}

// stringWriter adapts a strings.Builder to http.ResponseWriter for the render
// helpers, which write through one.
type stringWriter struct{ b *strings.Builder }

func (w *stringWriter) Header() http.Header         { return http.Header{} }
func (w *stringWriter) Write(p []byte) (int, error) { return w.b.Write(p) }
func (w *stringWriter) WriteHeader(int)             {}

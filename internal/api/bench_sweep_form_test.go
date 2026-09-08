package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
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
func TestSweepRejectsABadValue(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	code, body := postJob(t, s, url.Values{
		"name":                {"bad"},
		"model_ids":           {"TheBloke/Mixtral-8x7B-AWQ"},
		"presets":             {"internal-quick"},
		"sweep_max_model_len": {"not-a-number"},
	})
	if code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", code)
	}
	if !strings.Contains(body, "Context length") {
		t.Errorf("the message should name the parameter: %q", body)
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

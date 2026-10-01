package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// finishedRun runs autoconfigure for org/model with a stand-in helper and waits for
// the result.
func finishedRun(t *testing.T) *Server {
	t.Helper()
	s, _ := autoconfigServer(t)
	s.initTemplates()
	s.registry.Register(&models.Model{ID: "tcclaviger/Qwen3.8-27B-DFlash2-FP8", LocalPath: "/data/models/draft",
		HFConfig: models.HFConfig{HiddenSize: 5120, VocabSize: 248320,
			Draft: &models.DraftMeta{Method: "dflash", BlockSize: 8, TargetLayers: 64}}})
	if err := s.startAutoconfig("org/model", models.ContextMax, false); err != nil {
		t.Fatal(err)
	}
	if run := waitForRun(t, s); run.err != nil {
		t.Fatal(run.err)
	}
	return s
}

func post(s *Server, path string, form url.Values) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", path, strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.router.ServeHTTP(rec, req)
	return rec
}

func defaultRows(s *Server) url.Values {
	run, _ := s.autoconfigSnapshot()
	form := url.Values{}
	for _, r := range run.result.Rows {
		if r.Ticked {
			form.Add("row", r.Key)
		}
	}
	return form
}

func TestSaveWithoutApplying(t *testing.T) {
	s := finishedRun(t)
	s.router = s.buildRouter()
	before, _ := s.registry.Get("org/model")
	live := before.VLLMConfig

	form := defaultRows(s)
	form.Set("width", "all")
	form.Set("apply", "0")
	rec := post(s, "/api/models/autoconfig/save?id=org/model", form)
	if !strings.Contains(rec.Body.String(), "Saved as the Autoconfig profile.") {
		t.Fatalf("response: %s", rec.Body.String())
	}
	p, ok := s.registry.Profile("org/model", models.AutoconfigProfileName)
	if !ok || p.Source != models.ProfileSourceAutoconfig || p.Config.TensorParallelSize != 4 {
		t.Fatalf("profile: %+v", p)
	}
	if m, _ := s.registry.Get("org/model"); m.VLLMConfig != live {
		t.Error("saving without applying changed the live config")
	}
	if name, _ := s.registry.ActiveProfile("org/model"); name != "" {
		t.Errorf("active profile = %q", name)
	}
	if rec := post(s, "/api/models/autoconfig/save?id=org/model", form); !strings.Contains(rec.Body.String(), "no finished Autoconfigure result") {
		t.Error("a result was saved twice")
	}
}

func TestSaveAndApplyWithARowUnticked(t *testing.T) {
	s := finishedRun(t)
	s.router = s.buildRouter()

	form := defaultRows(s)
	var kept []string
	for _, k := range form["row"] {
		if k != "field:speculative_config" {
			kept = append(kept, k)
		}
	}
	form["row"] = kept
	form.Set("width", "narrow")
	form.Set("apply", "1")
	rec := post(s, "/api/models/autoconfig/save?id=org/model", form)
	if !strings.Contains(rec.Body.String(), "now in use") || rec.Header().Get("HX-Trigger") != "modelsChanged" {
		t.Fatalf("response: %s", rec.Body.String())
	}
	m, _ := s.registry.Get("org/model")
	if m.VLLMConfig.SpeculativeConfig != "" {
		t.Error("an unticked row was applied")
	}
	if m.VLLMConfig.ReasoningParser != "qwen3" || m.VLLMConfig.KVCacheDtype != "fp8" {
		t.Errorf("live config: %+v", m.VLLMConfig)
	}
	if name, modified := s.registry.ActiveProfile("org/model"); name != models.AutoconfigProfileName || modified {
		t.Errorf("active = %q modified = %v", name, modified)
	}
	if m.VRAMEstimate.TotalRequiredGB == 0 {
		t.Error("the stored estimate was not recomputed")
	}
	p, _ := s.registry.Profile("org/model", models.AutoconfigProfileName)
	if p.Autoconfig == nil || p.Autoconfig.Class != models.ContextMax {
		t.Errorf("record: %+v", p.Autoconfig)
	}
}

func TestSaveRefusals(t *testing.T) {
	s := finishedRun(t)
	s.router = s.buildRouter()
	rec := post(s, "/api/models/autoconfig/save?id=org/other", url.Values{})
	if !strings.Contains(rec.Body.String(), "no finished Autoconfigure result") {
		t.Errorf("save for another model: %s", rec.Body.String())
	}
	rec = post(s, "/api/models/autoconfig/start?id="+url.QueryEscape(models.HelperRepo), url.Values{})
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "Not started") {
		t.Errorf("start for the helper: %d %s", rec.Code, rec.Body.String())
	}
	req := httptest.NewRequest("GET", "/api/models/autoconfig/status?id=org/other", nil)
	out := httptest.NewRecorder()
	s.router.ServeHTTP(out, req)
	if strings.TrimSpace(out.Body.String()) != "" {
		t.Errorf("status for another model: %s", out.Body.String())
	}
}

func TestTheDialogAndTheReview(t *testing.T) {
	s := finishedRun(t)
	s.router = s.buildRouter()
	req := httptest.NewRequest("GET", "/api/models/autoconfig?id=org/model", nil)
	rec := httptest.NewRecorder()
	s.router.ServeHTTP(rec, req)
	body := rec.Body.String()
	if !strings.Contains(body, "From the model card") || !strings.Contains(body, `name="width" value="narrow"`) {
		t.Errorf("a held result did not render its review:\n%s", body)
	}
	// The helper's answer is shown as given, so a reading that came to
	// nothing can be told from one the checks dropped.
	if !strings.Contains(body, "The helper's answer, as it gave it") {
		t.Errorf("the helper's answer is not in the review:\n%s", body[strings.Index(body, "Notes"):])
	}
	run, _ := s.autoconfigSnapshot()
	res := *run.result
	res.Advice = []byte(`{"temperature":null,"top_p":0.95}`)
	m, _ := s.registry.Get("org/model")
	if v := s.autoconfigReview(m, &res); !strings.Contains(v.HelperAnswer, `"temperature": null`) {
		t.Errorf("helper answer: %q", v.HelperAnswer)
	}
	res.Advice = nil
	if v := s.autoconfigReview(m, &res); v.HelperAnswer != "" {
		t.Error("a run without a reading showed one")
	}
	post(s, "/api/models/autoconfig/discard?id=org/model", url.Values{})
	rec = httptest.NewRecorder()
	s.router.ServeHTTP(rec, httptest.NewRequest("GET", "/api/models/autoconfig?id=org/model", nil))
	if !strings.Contains(rec.Body.String(), "How much context do you need?") || !strings.Contains(rec.Body.String(), "Read the card again") {
		t.Errorf("after discarding, the start panel with a reading to reuse:\n%s", rec.Body.String())
	}
}

func TestDraftDownloadOnlyForASuggestion(t *testing.T) {
	s := finishedRun(t)
	s.router = s.buildRouter()
	rec := post(s, "/api/models/autoconfig/draft?id=org/model&repo=evil/anything", url.Values{})
	if !strings.Contains(rec.Body.String(), "not one this result suggests") {
		t.Errorf("an arbitrary repository: %s", rec.Body.String())
	}
	rec = post(s, "/api/models/autoconfig/draft?id=org/other&repo=org/drafter", url.Values{})
	if !strings.Contains(rec.Body.String(), "not one this result suggests") {
		t.Errorf("a model with no held result: %s", rec.Body.String())
	}
}

package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// prbServer is a golden server with the probe manager wired, as the real
// constructor does. None of these tests gets as far as spawning a probe.
func prbServer(t *testing.T) *Server {
	t.Helper()
	s := newGoldenServer(t, goldenEnvGeneric)
	s.probe = newProbeManager(s)
	return s
}

func prbStart(s *Server, contentType, body string, htmx bool) *httptest.ResponseRecorder {
	r := bhRequest(http.MethodPost, "/api/benchmarks/probe-context", strings.NewReader(body), nil)
	r.Header.Set("Content-Type", contentType)
	if htmx {
		r.Header.Set("HX-Request", "true")
	}
	return bhServe(s.handleStartContextProbe, r)
}

// prbAssertNoProbe fails if a refused start left a probe registered: the
// next attempt would be refused as "already in progress" forever.
func prbAssertNoProbe(t *testing.T, s *Server) {
	t.Helper()
	s.probe.mu.Lock()
	defer s.probe.mu.Unlock()
	if s.probe.active != nil {
		t.Error("a refused start left a probe active")
	}
}

func TestProbeStartWithoutAModelIsRefused(t *testing.T) {
	s := prbServer(t)
	if w := prbStart(s, "application/json", `{"tp_size":1}`, false); w.Code != http.StatusBadRequest {
		t.Errorf("JSON: status %d, want 400", w.Code)
	}
	if w := prbStart(s, "application/x-www-form-urlencoded", "tp_size=1", false); w.Code != http.StatusBadRequest {
		t.Errorf("form: status %d, want 400", w.Code)
	}
	prbAssertNoProbe(t, s)
}

func TestProbeStartWithMalformedJSONIsRefused(t *testing.T) {
	s := prbServer(t)
	w := prbStart(s, "application/json", `{"model_id":`, false)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid JSON") {
		t.Errorf("status %d %q, want 400 invalid JSON", w.Code, w.Body)
	}
}

func TestProbeStartForAnUnregisteredModelIsNotFound(t *testing.T) {
	s := prbServer(t)
	w := prbStart(s, "application/json", `{"model_id":"nobody/nothing"}`, false)
	if w.Code != http.StatusNotFound {
		t.Errorf("status %d, want 404", w.Code)
	}
	prbAssertNoProbe(t, s)
}

// The probe launches its own vLLM over and over and needs every card's VRAM.
// A main engine that is up -- or still coming up, and about to claim the
// memory -- would make every attempt fail as an OOM and the probe would
// report a ceiling of nothing.
func TestProbeStartIsRefusedWhileVLLMIsLive(t *testing.T) {
	for _, tc := range []struct {
		name  string
		ready bool
	}{
		{"starting", false},
		{"running", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := prbServer(t)
			bhStartFakeEngine(t, s, bhModel, tc.ready)

			// The form is how the page sends it; the model id must have been
			// read for the request to get as far as the engine check.
			form := url.Values{"model_id": {bhModel}, "tp_size": {"1"}}
			w := prbStart(s, "application/x-www-form-urlencoded", form.Encode(), false)
			if w.Code != http.StatusConflict {
				t.Errorf("status %d, want 409", w.Code)
			}
			if !strings.Contains(w.Body.String(), "stop the main vLLM") {
				t.Errorf("body %q does not say what to do", w.Body)
			}
			prbAssertNoProbe(t, s)
		})
	}
}

// One probe at a time: two would fight over the probe port and the cards.
func TestProbeStartIsRefusedWhileAnotherProbeRuns(t *testing.T) {
	s := prbServer(t)
	s.probe.active = &activeProbe{id: "other", modelID: bhModel}

	w := prbStart(s, "application/json", `{"model_id":"`+bhModel+`"}`, false)
	if w.Code != http.StatusConflict {
		t.Errorf("status %d, want 409", w.Code)
	}
	if s.probe.active.id != "other" {
		t.Error("the running probe was replaced")
	}
}

// The probe form is an htmx form, and htmx does not swap a non-2xx response:
// a refusal sent with http.Error leaves the form sitting there with no
// reason given. Every other benchmark form answers htmx through s.fail.
func TestProbeStartRefusalIsVisibleToHTMX(t *testing.T) {
	s := prbServer(t)
	w := prbStart(s, "application/x-www-form-urlencoded", "model_id=", true)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "model_id is required") {
		t.Errorf("status %d %q, want 200 with the reason", w.Code, w.Body)
	}
}

func prbApply(s *Server, contentType, body string, htmx bool) *httptest.ResponseRecorder {
	r := bhRequest(http.MethodPost, "/api/benchmarks/probe-context/apply", strings.NewReader(body), nil)
	r.Header.Set("Content-Type", contentType)
	if htmx {
		r.Header.Set("HX-Request", "true")
	}
	return bhServe(s.handleApplyProbe, r)
}

func TestProbeApplyValidatesItsBody(t *testing.T) {
	s := prbServer(t)
	for _, tc := range []struct {
		body string
		want int
	}{
		{`{"model_id":`, http.StatusBadRequest},
		{`{"max_model_len":8192}`, http.StatusBadRequest},
		{`{"model_id":"` + bhModel + `"}`, http.StatusBadRequest},
		{`{"model_id":"` + bhModel + `","max_model_len":-1}`, http.StatusBadRequest},
		{`{"model_id":"nobody/nothing","max_model_len":8192}`, http.StatusNotFound},
	} {
		if w := prbApply(s, "application/json", tc.body, false); w.Code != tc.want {
			t.Errorf("%s: status %d, want %d", tc.body, w.Code, tc.want)
		}
	}
}

// Apply writes what it is given onto the saved config. It does not look for
// a stored probe result: the values travel in the button's hx-vals, so a
// result that has since been replaced (or lost with a restart) does not stop
// the user applying the row they are looking at.
func TestProbeApplyWritesTheValuesWithoutAStoredResult(t *testing.T) {
	s := prbServer(t)
	if _, ok := s.probe.store.Get(bhModel); ok {
		t.Fatal("precondition: there should be no stored result")
	}

	w := prbApply(s, "application/json",
		`{"model_id":"`+bhModel+`","max_model_len":12345,"gpu_memory_utilization":0.85,"max_num_seqs":7}`, false)
	if w.Code != http.StatusNoContent {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	m, _ := s.registry.Get(bhModel)
	if m.VLLMConfig.MaxModelLen != 12345 || m.VLLMConfig.GPUMemoryUtilization != 0.85 || m.VLLMConfig.MaxNumSeqs != 7 {
		t.Errorf("saved config %+v", m.VLLMConfig)
	}
}

// Omitted optional values leave the saved ones alone: the concurrency row
// applies a context length and a sequence count, not a memory fraction.
func TestProbeApplyLeavesOmittedValuesAlone(t *testing.T) {
	s := prbServer(t)
	m, _ := s.registry.Get(bhModel)
	util := m.VLLMConfig.GPUMemoryUtilization

	w := prbApply(s, "application/json", `{"model_id":"`+bhModel+`","max_model_len":4096,"max_num_seqs":2}`, false)
	if w.Code != http.StatusNoContent {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	m, _ = s.registry.Get(bhModel)
	if m.VLLMConfig.GPUMemoryUtilization != util {
		t.Errorf("gpu_memory_utilization changed from %v to %v", util, m.VLLMConfig.GPUMemoryUtilization)
	}
}

// The Apply button is htmx with hx-ext="json-enc", but no page loads the
// json-enc extension (web/templates/layout.html loads htmx and htmx-sse
// only). htmx therefore posts the hx-vals form-encoded, the handler insists on
// JSON, and every click is a 400 that htmx does not display.
func TestProbeApplyAcceptsWhatTheApplyButtonSends(t *testing.T) {
	s := prbServer(t)
	form := url.Values{"model_id": {bhModel}, "max_model_len": {"12345"}}
	w := prbApply(s, "application/x-www-form-urlencoded", form.Encode(), true)
	if w.Code >= 300 {
		t.Errorf("status %d %q", w.Code, w.Body)
	}
	if m, _ := s.registry.Get(bhModel); m.VLLMConfig.MaxModelLen != 12345 {
		t.Errorf("max_model_len = %d, want 12345", m.VLLMConfig.MaxModelLen)
	}
}

package api

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

// Before the boot goroutine has read anything, the endpoint says so with a
// 200, not an error; afterwards it lists the archs and where they came from.
func TestArchRegistryEndpoint(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.router = s.buildRouter()
	get := func() map[string]any {
		rec := httptest.NewRecorder()
		s.router.ServeHTTP(rec, httptest.NewRequest("GET", "/api/arch-registry", nil))
		if rec.Code != 200 {
			t.Fatalf("status %d", rec.Code)
		}
		var out map[string]any
		json.Unmarshal(rec.Body.Bytes(), &out)
		return out
	}
	if out := get(); out["known"] != false || out["count"] != 0.0 || out["source"] != "none" {
		t.Errorf("before: %v", out)
	}
	s.SetSupportedArchs([]string{"Qwen4ExpForConditionalGeneration", "Gemma4ForCausalLM"}, "probe")
	out := get()
	archs, _ := out["archs"].([]any)
	if out["known"] != true || out["count"] != 2.0 || out["source"] != "probe" || len(archs) != 2 || archs[0] != "Gemma4ForCausalLM" {
		t.Errorf("after: %v", out)
	}
	if m, known := s.SupportedArchs(); !known || !m["Qwen4ExpForConditionalGeneration"] {
		t.Error("the set is not readable")
	}
}

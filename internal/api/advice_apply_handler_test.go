package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// The engine hands over its own pool size; applying it writes that one field
// and says so.
func TestApplyingWritesTheValueAndReports(t *testing.T) {
	const id = "test/apply"
	s := testServerWithEngine(t, id, realStartOutput, models.VLLMConfig{
		TensorParallelSize: 4, MaxModelLen: 262144, KVCacheDtype: "fp8",
	})
	s.initTemplates()

	form := url.Values{"model_id": {id}, "field": {"kv_cache_memory"}, "value": {"4545545954"}}
	req := httptest.NewRequest("POST", "/api/service/advice/apply", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()

	s.handleApplyAdvice(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	m, _ := s.registry.Get(id)
	if m.VLLMConfig.KVCacheMemory != 4545545954 {
		t.Errorf("kv_cache_memory = %d, want 4545545954", m.VLLMConfig.KVCacheMemory)
	}
	// Untouched fields survive.
	if m.VLLMConfig.MaxModelLen != 262144 || m.VLLMConfig.TensorParallelSize != 4 {
		t.Errorf("applying one field disturbed the rest: %+v", m.VLLMConfig)
	}
	// And the reply is the advice panel, not some other page's fragment.
	body := rec.Body.String()
	if !strings.Contains(body, "kv_cache_memory") {
		t.Errorf("the reply does not look like the advice panel:\n%.400s", body)
	}
}

// A field no rule can name is refused, and nothing is written.
func TestApplyingRefusesAnUnknownField(t *testing.T) {
	const id = "test/apply-unknown"
	s := testServerWithEngine(t, id, realStartOutput, models.VLLMConfig{
		TensorParallelSize: 4, MaxModelLen: 262144,
	})
	s.initTemplates()

	form := url.Values{"model_id": {id}, "field": {"tensor_parallel_size"}, "value": {"1"}}
	req := httptest.NewRequest("POST", "/api/service/advice/apply", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()

	s.handleApplyAdvice(rec, req)

	m, _ := s.registry.Get(id)
	if m.VLLMConfig.TensorParallelSize != 4 {
		t.Error("a field outside the allowed set was written anyway")
	}
	if !strings.Contains(rec.Body.String(), "not a setting this can change") {
		t.Errorf("the refusal was not explained:\n%.300s", rec.Body.String())
	}
}

// A model that has gone from the registry is reported rather than panicked on.
func TestApplyingToAMissingModel(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.initTemplates()

	form := url.Values{"model_id": {"gone/away"}, "field": {"max_model_len"}, "value": {"8192"}}
	req := httptest.NewRequest("POST", "/api/service/advice/apply", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()

	s.handleApplyAdvice(rec, req)

	if rec.Code >= 500 {
		t.Fatalf("status %d on a missing model: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "no longer in the registry") {
		t.Errorf("the missing model was not reported:\n%.300s", rec.Body.String())
	}
}

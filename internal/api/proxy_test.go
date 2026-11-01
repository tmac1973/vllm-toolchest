package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// The proxy is a pass-through. It used to buffer non-streaming chat
// completions to time them; that is read from the engine now, and a request
// must reach the engine and come back exactly as it was either way.
func TestProxyForwardsARequestUntouched(t *testing.T) {
	var gotBody, gotPath string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, r.ContentLength)
		r.Body.Read(body)
		gotBody, gotPath = string(body), r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer backend.Close()

	s := newTestServer(t, backend.URL)
	const body = `{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.newProxyHandler().ServeHTTP(rec, req)

	if gotPath != "/v1/chat/completions" || gotBody != body {
		t.Errorf("engine received %q at %q", gotBody, gotPath)
	}
	if rec.Body.String() != "data: [DONE]\n\n" {
		t.Errorf("client received %q", rec.Body.String())
	}
	// Nothing is inferred from traffic passing through.
	if n := len(s.bench.RecentTimings("x", 10)); n != 0 {
		t.Errorf("the proxy recorded %d timings", n)
	}
}

// The engine requires the key it was started with. After the key is changed
// in Settings, a client sends the new one -- which apiKeyAuth accepts -- and
// the proxy must hand the engine the old one, or every request fails until a
// restart.
func TestProxySendsTheEngineItsOwnKey(t *testing.T) {
	var gotAuth string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
	}))
	defer backend.Close()

	s := newTestServer(t, backend.URL)
	dir := t.TempDir()
	fake := testutil.WriteScript(t, "exec sleep 30\n")
	s.process = process.NewManager("127.0.0.1", 0, 0)
	s.process.SetLauncher(process.Launcher{Bin: fake})
	if err := s.process.Start("org/m", dir, nil, []string{"VLLM_API_KEY=engine-key"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.process.Stop() })

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{}"))
	req.Header.Set("Authorization", "Bearer changed-in-settings")
	s.newProxyHandler().ServeHTTP(httptest.NewRecorder(), req)
	if gotAuth != "Bearer engine-key" {
		t.Errorf("engine received Authorization %q", gotAuth)
	}
}

// /metrics on the vllmctl port reaches the engine's own /metrics, behind the
// same key as /v1. It used to fall through to the router's 404.
func TestMetricsReachTheEngine(t *testing.T) {
	var gotPath string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Write([]byte("vllm:num_requests_running 0\n"))
	}))
	defer backend.Close()

	s := newTestServer(t, backend.URL)
	s.cfg.APIKey = "k"

	rec := httptest.NewRecorder()
	s.buildRouter().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	if rec.Code != http.StatusUnauthorized || gotPath != "" {
		t.Fatalf("without a key: status %d, engine saw %q", rec.Code, gotPath)
	}

	req := httptest.NewRequest("GET", "/metrics", nil)
	req.Header.Set("Authorization", "Bearer k")
	rec = httptest.NewRecorder()
	s.buildRouter().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || gotPath != "/metrics" {
		t.Fatalf("with a key: status %d, engine saw %q", rec.Code, gotPath)
	}
	if !strings.Contains(rec.Body.String(), "vllm:num_requests_running") {
		t.Errorf("client received %q", rec.Body.String())
	}
}

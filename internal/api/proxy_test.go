package api

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseChatRequest(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantStream bool
		wantModel  string
	}{
		{"non-streaming", `{"stream":false,"model":"m"}`, false, "m"},
		{"streaming", `{"stream":true,"model":"x"}`, true, "x"},
		{"omitted-stream defaults false", `{"model":"y"}`, false, "y"},
		{"malformed → defaults", `{not json`, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotStream, gotModel := parseChatRequest([]byte(tc.body))
			if gotStream != tc.wantStream {
				t.Errorf("stream: want %v, got %v", tc.wantStream, gotStream)
			}
			if gotModel != tc.wantModel {
				t.Errorf("model: want %q, got %q", tc.wantModel, gotModel)
			}
		})
	}
}

func TestParseUsage(t *testing.T) {
	prompt, completion, ok := parseUsage([]byte(`{"usage":{"prompt_tokens":42,"completion_tokens":13}}`))
	if !ok || prompt != 42 || completion != 13 {
		t.Errorf("want 42/13/true, got %d/%d/%v", prompt, completion, ok)
	}

	_, _, ok = parseUsage([]byte(`{"choices":[{}]}`)) // no usage
	if ok {
		t.Error("expected ok=false when usage missing")
	}

	_, _, ok = parseUsage([]byte(`not json`))
	if ok {
		t.Error("expected ok=false on malformed JSON")
	}
}

func TestIsChatCompletion(t *testing.T) {
	cases := []struct {
		method string
		path   string
		want   bool
	}{
		{"POST", "/v1/chat/completions", true},
		{"POST", "/chat/completions", true},
		{"POST", "/v1/embeddings", false},
		{"POST", "/v1/models", false},
		{"GET", "/v1/chat/completions", false},
	}
	for _, tc := range cases {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		if got := isChatCompletion(req); got != tc.want {
			t.Errorf("%s %s: want %v, got %v", tc.method, tc.path, tc.want, got)
		}
	}
}

func TestCaptureWriterBuffersUpToLimit(t *testing.T) {
	rec := httptest.NewRecorder()
	cw := &captureWriter{ResponseWriter: rec, buf: &bytes.Buffer{}, limit: 100}
	cw.WriteHeader(200)
	cw.Write([]byte(`{"hello":"world"}`))
	cw.Write([]byte(strings.Repeat("x", 50)))

	if cw.truncated {
		t.Error("expected non-truncated at 67 bytes (under 100 limit)")
	}
	if cw.status != 200 {
		t.Errorf("expected status 200, got %d", cw.status)
	}
	if !bytes.Contains(cw.buf.Bytes(), []byte(`"hello"`)) {
		t.Error("expected buffered body to contain captured JSON")
	}

	// Now overflow.
	cw.Write([]byte(strings.Repeat("y", 200)))
	if !cw.truncated {
		t.Error("expected truncated after exceeding limit")
	}
}

// TestProxyCapturesTimingForNonStreaming verifies the end-to-end path:
// hit the proxy with a non-streaming /v1/chat/completions request,
// confirm a TimingSample lands in the store.
func TestProxyCapturesTimingForNonStreaming(t *testing.T) {
	// Backend that returns an OpenAI-shaped completion with usage.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write([]byte(`{
  "id":"chatcmpl-1",
  "choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":42,"completion_tokens":13,"total_tokens":55}
}`))
	}))
	defer backend.Close()

	s := newTestServer(t, backend.URL)
	proxy := s.newProxyHandler()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"test-model","stream":false,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	samples := s.bench.RecentTimings("test-model", 10)
	if len(samples) != 1 {
		t.Fatalf("expected 1 captured timing, got %d", len(samples))
	}
	if samples[0].GenTokens != 13 || samples[0].PromptTokens != 42 {
		t.Errorf("unexpected sample: %+v", samples[0])
	}
}

func TestProxySkipsStreamingResponses(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer backend.Close()

	s := newTestServer(t, backend.URL)
	proxy := s.newProxyHandler()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)

	// Streaming should not capture (we don't buffer SSE).
	if len(s.bench.RecentTimings("x", 10)) != 0 {
		t.Error("expected no timing samples for streaming responses")
	}
}

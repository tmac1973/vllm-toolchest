package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// maxCapturedRespBody is the upper bound on response body we'll buffer
// for passive timing capture. Typical non-streaming chat completion
// responses are well under 64 KiB even for large generations; if a
// response exceeds this, we skip capture rather than blow memory.
const maxCapturedRespBody = 1 << 20 // 1 MiB

func (s *Server) newProxyHandler() http.Handler {
	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("%s:%d", s.cfg.VLLMHost, s.cfg.VLLMPort),
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		fmt.Fprintf(w, `{"error":{"message":"vLLM is not available: %s","type":"proxy_error"}}`, err)
	}

	// Wrap to capture passive timing samples from non-streaming chat
	// completion responses.
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.bench != nil && isChatCompletion(r) {
			s.captureChatCompletion(w, r, proxy)
			return
		}
		proxy.ServeHTTP(w, r)
	})
}

// isChatCompletion identifies POST /v1/chat/completions requests. We
// only capture on this endpoint — other proxied paths (embeddings,
// /v1/models, etc.) don't have the usage shape we record.
func isChatCompletion(r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}
	// chi mounts the proxy under /v1/*, so the path can be either form.
	return strings.HasSuffix(r.URL.Path, "/chat/completions")
}

// captureChatCompletion forwards the request and, if it's non-streaming,
// captures the response body to extract usage tokens for the timing
// store. Streaming requests are forwarded unchanged — capturing SSE
// would require buffering the entire stream and double the bandwidth.
func (s *Server) captureChatCompletion(w http.ResponseWriter, r *http.Request, proxy *httputil.ReverseProxy) {
	// Buffer the request body so we can inspect it for stream=true and
	// re-attach it for the proxy.
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read request body: "+err.Error(), http.StatusBadRequest)
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(reqBody))
	r.ContentLength = int64(len(reqBody))

	streaming, modelID := parseChatRequest(reqBody)
	if streaming {
		// Streaming: forward as-is without capture.
		proxy.ServeHTTP(w, r)
		return
	}

	// Non-streaming: wrap the response writer so we can buffer the body
	// after the proxy writes it, then forward to the real writer.
	start := time.Now()
	cap := &captureWriter{ResponseWriter: w, buf: &bytes.Buffer{}, limit: maxCapturedRespBody}
	proxy.ServeHTTP(cap, r)
	elapsed := time.Since(start)

	// Only capture on 2xx with a successfully buffered body.
	if cap.status < 200 || cap.status >= 300 || cap.truncated {
		return
	}

	prompt, completion, ok := parseUsage(cap.buf.Bytes())
	if !ok || completion <= 0 {
		return
	}

	// genTPS = completion tokens / elapsed time. We don't have TTFT
	// split for non-streaming responses, so PromptTPS is left at 0.
	genTPS := float64(completion) / elapsed.Seconds()
	sample := benchmark.TimingSample{
		Timestamp:    start,
		ModelID:      resolveSampleModelID(s, modelID),
		PromptTokens: prompt,
		GenTokens:    completion,
		GenTokPerSec: genTPS,
	}
	s.bench.AddTiming(sample)
	slog.Debug("captured timing", "model", sample.ModelID, "gen_tps", genTPS,
		"prompt", prompt, "completion", completion, "elapsed_ms", elapsed.Milliseconds())
}

// parseChatRequest extracts the streaming flag and the requested model
// from a chat completion request body. Best-effort: malformed JSON
// returns (false, "") and the timing capture is skipped.
func parseChatRequest(body []byte) (streaming bool, model string) {
	var req struct {
		Stream bool   `json:"stream"`
		Model  string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false, ""
	}
	return req.Stream, req.Model
}

// parseUsage pulls prompt_tokens and completion_tokens out of a JSON
// chat-completion response.
func parseUsage(body []byte) (prompt, completion int, ok bool) {
	var resp struct {
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &resp); err != nil || resp.Usage == nil {
		return 0, 0, false
	}
	return resp.Usage.PromptTokens, resp.Usage.CompletionTokens, true
}

// resolveSampleModelID maps the request's "model" field onto a stable
// model id for the timing store. The request typically carries the
// served-model-name (which for vLLM is a filesystem path); we prefer
// to attribute to the registry's HF repo id so the dashboard's
// per-model card matches.
func resolveSampleModelID(s *Server, requested string) string {
	if requested == "" {
		// Fallback: use whatever vLLM is currently serving.
		st := s.process.GetStatus()
		if st.State == process.StateRunning {
			return st.ModelID
		}
		return ""
	}
	// Common case: the request's "model" already equals the registry id.
	if _, ok := s.registry.Get(requested); ok {
		return requested
	}
	// vLLM serves a filesystem path: walk the registry looking for a
	// matching LocalPath suffix.
	for _, m := range s.registry.List() {
		if strings.HasSuffix(requested, m.LocalPath) || requested == m.LocalPath {
			return m.ID
		}
	}
	return requested
}

// captureWriter is a ResponseWriter that mirrors writes into an internal
// buffer (up to limit bytes) while still forwarding to the underlying
// writer. Used to capture the proxied response body for timing parse.
type captureWriter struct {
	http.ResponseWriter
	buf       *bytes.Buffer
	limit     int
	status    int
	truncated bool
}

func (c *captureWriter) WriteHeader(status int) {
	c.status = status
	c.ResponseWriter.WriteHeader(status)
}

func (c *captureWriter) Write(p []byte) (int, error) {
	if c.status == 0 {
		c.status = http.StatusOK
	}
	// Buffer at most limit bytes; once exceeded, mark truncated and stop.
	if !c.truncated {
		if c.buf.Len()+len(p) > c.limit {
			c.truncated = true
		} else {
			c.buf.Write(p)
		}
	}
	return c.ResponseWriter.Write(p)
}

// Flush passes through if the underlying writer supports it. Required so
// non-streaming responses still flush correctly through the wrapper.
func (c *captureWriter) Flush() {
	if f, ok := c.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// handleV1Models answers the OpenAI-compatible model listing.
//
// It reports the model actually being served, and nothing else. vLLM serves
// one model per process, so the list is either one entry or none.
//
// This used to fall back to listing the whole registry when the engine was
// stopped, which advertised models no request could be served by: a client
// discovering a name there got a 404 for it, and the same endpoint answered
// with a different identifier depending on whether the engine happened to be
// up. Nothing loaded now means an empty list, which is the honest answer and
// the one a client can act on.
func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	if s.process.GetStatus().State == process.StateRunning {
		s.newProxyHandler().ServeHTTP(w, r)
		return
	}
	respondJSON(w, struct {
		Object string `json:"object"`
		Data   []any  `json:"data"`
	}{Object: "list", Data: []any{}})
}

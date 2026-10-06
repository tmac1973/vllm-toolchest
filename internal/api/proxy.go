package api

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

func (s *Server) newProxyHandler() http.Handler {
	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("%s:%d", s.cfg.VLLMHost, s.cfg.VLLMPort),
	}

	// A plain pass-through. Request timings used to be captured here; they
	// are read from the engine's own counters now, which see every request
	// rather than the non-streaming ones that happen to come this way. See
	// engine_timings.go.
	proxy := httputil.NewSingleHostReverseProxy(target)
	// The client's key has been checked against the configured one by
	// apiKeyAuth. The engine requires the key it was started with, which
	// differs after a change in Settings until the next start, so that is
	// the one sent on.
	direct := proxy.Director
	proxy.Director = func(r *http.Request) {
		direct(r)
		if key := s.process.EngineAPIKey(); key != "" {
			r.Header.Set("Authorization", "Bearer "+key)
		}
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		respondJSONStatus(w, http.StatusBadGateway, openAIError("vLLM is not available: "+err.Error(), "proxy_error"))
	}
	return proxy
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

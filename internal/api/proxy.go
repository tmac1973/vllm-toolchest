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

	proxy := httputil.NewSingleHostReverseProxy(target)

	// Customize error handler for when vLLM is not running
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		fmt.Fprintf(w, `{"error":{"message":"vLLM is not available: %s","type":"proxy_error"}}`, err)
	}

	return proxy
}

func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	// When vLLM is running, proxy through to get actual model paths
	if s.process.GetStatus().State == process.StateRunning {
		s.newProxyHandler().ServeHTTP(w, r)
		return
	}

	// Fallback: return registry list when vLLM is not running
	list := s.registry.List()
	type modelObj struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	type response struct {
		Object string     `json:"object"`
		Data   []modelObj `json:"data"`
	}

	resp := response{Object: "list"}
	for _, m := range list {
		resp.Data = append(resp.Data, modelObj{
			ID:      m.ID,
			Object:  "model",
			OwnedBy: "vllm-toolchest",
		})
	}

	respondJSON(w, resp)
}

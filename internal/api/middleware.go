package api

import (
	"crypto/subtle"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5/middleware"
)

// apiKeyAuth returns middleware that checks for a valid Bearer token.
// If no API key is configured, all requests are allowed.
//
// A CORS preflight is let through without one. A browser asks it before a
// page's cross-origin request and never attaches credentials to it, so
// refusing it locks every browser client out however its key is set. It
// carries no body and reaches nothing but vLLM's CORS handling, which
// answers it; the request that follows is checked as usual.
func (s *Server) apiKeyAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.APIKey == "" || isPreflight(r) {
			next.ServeHTTP(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			allowAnyOrigin(w, r)
			respondJSONStatus(w, http.StatusUnauthorized, openAIError("missing API key", "auth_error"))
			return
		}

		token := strings.TrimPrefix(auth, "Bearer ")
		// Constant time, so response timing says nothing about how much of
		// a guess was right.
		if subtle.ConstantTimeCompare([]byte(token), []byte(s.cfg.APIKey)) != 1 {
			allowAnyOrigin(w, r)
			respondJSONStatus(w, http.StatusUnauthorized, openAIError("invalid API key", "auth_error"))
			return
		}

		next.ServeHTTP(w, r)
	})
}

// isPreflight reports a CORS preflight: an OPTIONS request naming the method
// the browser wants to send. A bare OPTIONS is not one.
func isPreflight(r *http.Request) bool {
	return r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != ""
}

// allowAnyOrigin lets a browser page read a /v1 response this server writes
// itself, as vLLM lets it read the ones it writes. Without it a page sees a
// refused key, or an engine that is down, only as "Failed to fetch". It is
// what vLLM answers with by default, and what it exposes -- an error, or an
// empty model list -- is no more than any client could learn by asking.
func allowAnyOrigin(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Origin") != "" {
		w.Header().Set("Access-Control-Allow-Origin", "*")
	}
}

// answerPreflight answers a preflight that vLLM cannot, because it is not
// running, as vLLM would: any origin, the method and headers asked for. Left
// failing, it would stop the browser before the request whose answer says
// the engine is down.
func answerPreflight(w http.ResponseWriter, r *http.Request) {
	h := w.Header()
	h.Set("Access-Control-Allow-Origin", "*")
	h.Set("Access-Control-Allow-Methods", r.Header.Get("Access-Control-Request-Method"))
	if asked := r.Header.Get("Access-Control-Request-Headers"); asked != "" {
		h.Set("Access-Control-Allow-Headers", asked)
	}
	w.WriteHeader(http.StatusNoContent)
}

// requestLogger logs each request when it completes, through slog like the
// rest of vllmctl. A GET or HEAD is logged at debug: the UI polls status, the
// monitor and the logs every few seconds, and at info those lines buried the
// ones that say something happened. Everything that changes state stays at
// info, with the time it finished.
func requestLogger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
		start := time.Now()
		next.ServeHTTP(ww, r)

		level := slog.LevelInfo
		if r.Method == http.MethodGet || r.Method == http.MethodHead {
			level = slog.LevelDebug
		}
		slog.Log(r.Context(), level, "http request",
			"method", r.Method, "path", r.URL.RequestURI(), "status", ww.Status(),
			"bytes", ww.BytesWritten(), "duration", time.Since(start).Round(time.Microsecond),
			"remote", r.RemoteAddr)
	})
}

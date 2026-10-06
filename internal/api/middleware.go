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
func (s *Server) apiKeyAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.APIKey == "" {
			next.ServeHTTP(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			respondJSONStatus(w, http.StatusUnauthorized, openAIError("missing API key", "auth_error"))
			return
		}

		token := strings.TrimPrefix(auth, "Bearer ")
		// Constant time, so response timing says nothing about how much of
		// a guess was right.
		if subtle.ConstantTimeCompare([]byte(token), []byte(s.cfg.APIKey)) != 1 {
			respondJSONStatus(w, http.StatusUnauthorized, openAIError("invalid API key", "auth_error"))
			return
		}

		next.ServeHTTP(w, r)
	})
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

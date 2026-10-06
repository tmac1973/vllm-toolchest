package api

import (
	"crypto/subtle"
	"net/http"
	"strings"
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
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":{"message":"missing API key","type":"auth_error"}}`,
				http.StatusUnauthorized)
			return
		}

		token := strings.TrimPrefix(auth, "Bearer ")
		// Constant time, so response timing says nothing about how much of
		// a guess was right.
		if subtle.ConstantTimeCompare([]byte(token), []byte(s.cfg.APIKey)) != 1 {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":{"message":"invalid API key","type":"auth_error"}}`,
				http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}

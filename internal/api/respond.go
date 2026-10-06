package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

func isHTMX(r *http.Request) bool {
	return r.Header.Get("HX-Request") == "true"
}

func respondJSON(w http.ResponseWriter, v any) {
	respondJSONStatus(w, http.StatusOK, v)
}

func respondJSONStatus(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	// The status is already sent, so an encode failure -- a client gone
	// away, as a rule -- has nowhere left to be reported.
	_ = json.NewEncoder(w).Encode(v)
}

// openAIError is an error body in the shape OpenAI clients parse, for the
// /v1 routes.
func openAIError(msg, typ string) any {
	return map[string]any{"error": map[string]string{"message": msg, "type": typ}}
}

// fail reports a refused request. htmx does not swap a non-2xx response, so
// an htmx caller given http.Error sees nothing at all -- the button clicks,
// the form sits there, and the reason is only in the network tab. It gets
// 200 and the error partial instead; everything else keeps the real status.
func (s *Server) fail(w http.ResponseWriter, r *http.Request, status int, msg string) {
	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "error_message", msg)
		return
	}
	http.Error(w, msg, status)
}

// modelFromQuery resolves the model named by the id query parameter. When
// there is none it has already answered 404 and returns false.
func (s *Server) modelFromQuery(w http.ResponseWriter, r *http.Request) (*models.Model, bool) {
	m, ok := s.registry.Get(r.URL.Query().Get("id"))
	if !ok {
		s.fail(w, r, http.StatusNotFound, "model not found")
	}
	return m, ok
}

// vllmBaseURL is the served engine's origin, for the calls vllmctl makes to
// it directly rather than through the /v1 proxy.
func (s *Server) vllmBaseURL() string {
	return fmt.Sprintf("http://%s:%d", s.cfg.VLLMHost, s.cfg.VLLMPort)
}

// parseForm parses r's form. On a malformed body it has already answered 400
// and returns false.
func (s *Server) parseForm(w http.ResponseWriter, r *http.Request) bool {
	if err := r.ParseForm(); err != nil {
		s.fail(w, r, http.StatusBadRequest, "invalid form: "+err.Error())
		return false
	}
	return true
}

// readBody decodes a JSON body into v when the Content-Type says JSON, and
// otherwise parses the form for the caller to read with FormValue; isJSON
// says which happened. On a malformed body it has already answered 400 and
// ok is false.
func (s *Server) readBody(w http.ResponseWriter, r *http.Request, v any) (isJSON, ok bool) {
	if strings.Contains(r.Header.Get("Content-Type"), "json") {
		if err := json.NewDecoder(r.Body).Decode(v); err != nil {
			s.fail(w, r, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return true, false
		}
		return true, true
	}
	return false, s.parseForm(w, r)
}

func respondHTML(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
}

func toFloat64(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case int32:
		return float64(n)
	default:
		return 0
	}
}

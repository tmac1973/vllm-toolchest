package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestReadBodyRefusesMalformedJSON(t *testing.T) {
	s := settingsServer(t)
	r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader("{not json"))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	var v struct{ A int }
	if _, ok := s.readBody(w, r, &v); ok {
		t.Fatal("malformed JSON accepted")
	}
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid JSON") {
		t.Errorf("got %d %q", w.Code, w.Body.String())
	}
}

func TestReadBodyReadsAFormAsAForm(t *testing.T) {
	s := settingsServer(t)
	r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader("model_id=org%2Fm"))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()

	var v struct{ ModelID string }
	isJSON, ok := s.readBody(w, r, &v)
	if !ok || isJSON || r.FormValue("model_id") != "org/m" {
		t.Errorf("isJSON=%v ok=%v model_id=%q", isJSON, ok, r.FormValue("model_id"))
	}
}

// htmx does not swap a non-2xx response, so its refusal must arrive as a 200
// carrying the message, where any other caller gets the real status.
func TestFailSpeaksEachCallersLanguage(t *testing.T) {
	s := settingsServer(t)

	r := httptest.NewRequest(http.MethodPost, "/", nil)
	r.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.fail(w, r, http.StatusConflict, "busy right now")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "busy right now") {
		t.Errorf("htmx: got %d %q", w.Code, w.Body.String())
	}

	w = httptest.NewRecorder()
	s.fail(w, httptest.NewRequest(http.MethodPost, "/", nil), http.StatusConflict, "busy right now")
	if w.Code != http.StatusConflict {
		t.Errorf("plain: got %d", w.Code)
	}
}

// The error text is arbitrary -- a dial error can carry quotes -- and the body
// must still parse.
func TestOpenAIErrorIsValidJSON(t *testing.T) {
	w := httptest.NewRecorder()
	respondJSONStatus(w, http.StatusBadGateway, openAIError(`dial "vllm": refused`, "proxy_error"))
	var body struct {
		Error struct{ Message, Type string }
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("body %q: %v", w.Body.String(), err)
	}
	if body.Error.Message != `dial "vllm": refused` || w.Header().Get("Content-Type") != "application/json" {
		t.Errorf("got %+v, content type %q", body, w.Header().Get("Content-Type"))
	}
}

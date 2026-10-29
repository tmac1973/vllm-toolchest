package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A web page in the operator's browser must not be able to drive /api -- a
// restore from one would replace the secrets -- while the UI itself, scripts
// and curl keep working.
func TestAPIRefusesCrossOriginWrites(t *testing.T) {
	s := settingsServer(t)
	router := s.buildRouter()

	cases := []struct {
		name    string
		headers map[string]string
		refused bool
	}{
		{"cross-site browser post", map[string]string{"Sec-Fetch-Site": "cross-site"}, true},
		{"another port on the same host", map[string]string{"Sec-Fetch-Site": "same-site"}, true},
		{"older browser, foreign Origin", map[string]string{"Origin": "http://evil.example"}, true},
		{"the UI itself", map[string]string{"Sec-Fetch-Site": "same-origin"}, false},
		{"curl or a script", nil, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// An unrouted path: the guard answers 403, or the request falls
			// through to a 404. Only the guard is under test.
			req := httptest.NewRequest(http.MethodPost, "/api/no-such-route", strings.NewReader(""))
			for k, v := range c.headers {
				req.Header.Set(k, v)
			}
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if refused := rec.Code == http.StatusForbidden; refused != c.refused {
				t.Errorf("status %d, refused = %v, want %v", rec.Code, refused, c.refused)
			}
		})
	}
}

// /v1 is called cross-origin by browser chat front-ends on purpose.
func TestV1AllowsCrossOriginCalls(t *testing.T) {
	s := settingsServer(t)
	s.cfg.APIKey = "sk-test"
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}"))
	req.Header.Set("Sec-Fetch-Site", "cross-site")
	rec := httptest.NewRecorder()
	// No key: refused by apiKeyAuth with 401, which proves the request got
	// past any origin check to the /v1 group's own.
	s.buildRouter().ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status %d, want 401 from the key check", rec.Code)
	}
}

// A browser asks a preflight before a page's cross-origin call and never puts
// the key on it. With a key set, the preflight must still reach vLLM, which
// answers it, or no browser client can get as far as sending its key.
func TestV1PassesPreflightsWithoutTheKey(t *testing.T) {
	var reached []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = append(reached, r.Method+" "+r.URL.Path)
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", r.Header.Get("Access-Control-Request-Headers"))
	}))
	defer backend.Close()
	s := newTestServer(t, backend.URL)
	s.cfg.APIKey = "sk-test"
	router := s.buildRouter()

	for _, path := range []string{"/v1/chat/completions", "/v1/models"} {
		req := httptest.NewRequest(http.MethodOptions, path, nil)
		req.Header.Set("Origin", "http://chat.lan:8080")
		req.Header.Set("Access-Control-Request-Method", "POST")
		req.Header.Set("Access-Control-Request-Headers", "authorization, content-type")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK || rec.Header().Get("Access-Control-Allow-Headers") != "authorization, content-type" {
			t.Errorf("preflight to %s: %d %v", path, rec.Code, rec.Header())
		}
	}
	if len(reached) != 2 {
		t.Errorf("vLLM received %v, want both preflights", reached)
	}

	// What follows a preflight still needs the key, and an OPTIONS that is
	// not a preflight is not let through on the strength of its method.
	reached = nil
	for _, method := range []string{http.MethodPost, http.MethodOptions} {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(method, "/v1/chat/completions", strings.NewReader("{}")))
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s without the key: %d, want 401", method, rec.Code)
		}
	}
	if len(reached) != 0 {
		t.Errorf("vLLM received %v from requests without the key", reached)
	}
}

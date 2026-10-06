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

package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

// The settings page carries the most template-to-struct coupling in the app,
// and html/template resolves field names at execute time, not parse time — so
// a renamed field renders a blank page in production and nothing catches it
// until someone opens the browser. Execute it for both image variants.
func TestSettingsPageRendersPerVariant(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  vllmenv.Env
		want []string
		skip []string
	}{
		{
			name: "generic",
			env:  vllmenv.Env{Variant: vllmenv.VariantGeneric, VenvRoot: "/opt/vllm-venv"},
			want: []string{"Image variant", "generic", "attention_backend"},
			// The radiance panel and its R4D backend must not appear on an
			// image that has neither.
			skip: []string{"radiance_use_r4d", `value="R4D"`},
		},
		{
			name: "radiance",
			env: vllmenv.Env{
				Variant:         vllmenv.VariantRadiance,
				RadianceVersion: "0.9.3",
				VenvRoot:        "/opt/vllm",
			},
			want: []string{"radiance_use_r4d", "radiance_draft_tau", "0.9.3", `value="R4D"`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &Server{cfg: &config.Config{}, vllmEnv: tc.env}
			s.pages = s.parseTemplates()

			w := httptest.NewRecorder()
			s.handleSettingsPage(w, httptest.NewRequest("GET", "/settings", nil))

			if w.Code != 200 {
				t.Fatalf("status = %d, want 200", w.Code)
			}
			body := w.Body.String()
			for _, want := range tc.want {
				if !strings.Contains(body, want) {
					t.Errorf("page is missing %q", want)
				}
			}
			for _, skip := range tc.skip {
				if strings.Contains(body, skip) {
					t.Errorf("page unexpectedly contains %q", skip)
				}
			}
		})
	}
}

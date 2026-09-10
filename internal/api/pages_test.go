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
			name: "rocm-source",
			env:  vllmenv.Env{Variant: "rocm-source", VenvRoot: "/opt/vllm-venv"},
			want: []string{"Image variant", "rocm-source", "attention_backend"},
			// A from-source image declares no knobs, so the whole panel is
			// absent rather than an empty box. R4D belongs to radiance, and
			// FLASHINFER to NVIDIA: offering either here would abort the
			// engine minutes into a load.
			skip: []string{"knob_", `value="R4D"`, `value="FLASHINFER"`},
		},
		{
			name: "radiance",
			env: vllmenv.Env{
				Variant:        "radiance",
				VariantVersion: "0.9.3",
				VenvRoot:       "/opt/vllm",
			},
			want: []string{"knob_use_r4d", "knob_draft_tau", "0.9.3", `value="R4D"`,
				// Its vendor's backends, and not the other vendor's.
				`value="ROCM_FLASH"`,
				// Rendered from the manifest, not from template markup:
				// the four-state select and a per-option label.
				`value="all"`, "all shapes"},
		},
		{
			// An image whose variant no manifest describes — built before
			// its manifest existed, or an operator override naming
			// something we do not ship. The panel must be absent rather
			// than empty, and the page must still render.
			name: "unknown variant",
			env:  vllmenv.Env{Variant: "some-future-image", VenvRoot: "/opt/vllm-venv"},
			want: []string{"Image variant", "some-future-image"},
			// No manifest means no knobs and no vendor, so only the backends
			// that work anywhere are offered.
			skip: []string{"knob_", `value="ROCM_FLASH"`, `value="FLASHINFER"`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &Server{cfg: &config.Config{}, vllmEnv: tc.env}
			s.initTemplates()

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

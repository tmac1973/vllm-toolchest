package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

// renderHelp renders the help page for one image variant.
func renderHelp(t *testing.T, variant string) string {
	t.Helper()
	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: variant}}
	s.initTemplates()

	w := httptest.NewRecorder()
	s.handleHelpPage(w, httptest.NewRequest(http.MethodGet, "/help", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	body := w.Body.String()
	// render() writes a 500 body only after having already written a 200
	// header, so the status alone does not prove the template executed.
	if strings.Contains(body, "template error") {
		t.Fatalf("template failed to execute: %s", body[max(0, len(body)-400):])
	}
	return body
}

// Every anchor the table of contents links to has to exist, or a heading
// rename leaves a link that silently goes nowhere.
func TestHelpTOCAnchorsResolve(t *testing.T) {
	body := renderHelp(t, "radiance")

	// The TOC is the first nav on the page; collect its hrefs.
	start := strings.Index(body, `class="help-toc"`)
	if start < 0 {
		t.Fatal("no table of contents on the help page")
	}
	end := strings.Index(body[start:], "</nav>")
	if end < 0 {
		t.Fatal("unterminated table of contents")
	}
	toc := body[start : start+end]

	var anchors []string
	for _, part := range strings.Split(toc, `href="#`)[1:] {
		anchors = append(anchors, part[:strings.IndexByte(part, '"')])
	}
	if len(anchors) < 8 {
		t.Fatalf("expected a full table of contents, found %d entries", len(anchors))
	}
	for _, a := range anchors {
		if !strings.Contains(body, `id="`+a+`"`) {
			t.Errorf("table of contents links to #%s, which no heading defines", a)
		}
	}
}

// The radiance-only sections describe a kernel library the generic image does
// not ship. Showing them there would document a ceiling that does not exist.
func TestHelpRadianceSectionsAreGated(t *testing.T) {
	generic := renderHelp(t, "rocm-source")
	radiance := renderHelp(t, "radiance")

	for _, phrase := range []string{"all-reduce token ceiling", "image default"} {
		if strings.Contains(generic, phrase) {
			t.Errorf("generic image shows radiance-only text: %q", phrase)
		}
		if !strings.Contains(radiance, phrase) {
			t.Errorf("radiance image is missing %q", phrase)
		}
	}
}

// The help page links into the app. A typo'd path is a dead link on the one
// page whose whole job is pointing people at the right place.
func TestHelpInternalLinksArePagePaths(t *testing.T) {
	body := renderHelp(t, "radiance")

	pages := map[string]bool{
		"/": true, "/server": true, "/models": true, "/models/browse": true,
		"/benchmarks": true, "/benchmarks/visualize": true, "/tuning": true,
		"/settings": true, "/help": true,
	}
	// Only the help page's own body, so the layout's sidebar is not counted.
	start := strings.Index(body, `class="help-page"`)
	if start < 0 {
		t.Fatal("help page body not found")
	}
	for _, part := range strings.Split(body[start:], `href="/`)[1:] {
		path := "/" + part[:strings.IndexByte(part, '"')]
		if !pages[path] {
			t.Errorf("help page links to %q, which is not a page route", path)
		}
	}
}

// The Tuning section answers three questions people actually arrive with:
// what it tunes, whether it applies to their model, and whether it is safe to
// start right now. The last one matters most — a tuning run stops the serving
// vLLM to get the GPU, and nothing else on the page says so.
func TestHelpTuningCoversApplicabilityAndRisk(t *testing.T) {
	page := renderHelp(t, "radiance")

	for _, want := range []struct{ phrase, why string }{
		{"stops the running vLLM server", "a tuning run takes the server down and that must be stated"},
		{"block-quantized FP8", "the format it applies to"},
		{"block_structure", "how eligibility is actually determined, rather than by format label"},
		{"per tensor, per channel or per group", "the FP8 scalings that do not qualify"},
		{"tensor-parallel factor", "shapes depend on TP, so results do not carry across it"},
		{"different subsystem from the attention backend", "the two are routinely confused"},
		{"block size (128)", "why the shape count is smaller than the matmul count"},
	} {
		if !strings.Contains(page, want.phrase) {
			t.Errorf("tuning help is missing %q — %s", want.phrase, want.why)
		}
	}
}

// Formats that can never benefit should be named, so somebody with an AWQ
// model does not go looking for a tuning win that cannot exist.
func TestHelpTuningNamesIneligibleFormats(t *testing.T) {
	page := renderHelp(t, "radiance")
	for _, f := range []string{"AWQ", "GPTQ", "bitsandbytes", "FP16"} {
		if !strings.Contains(page, f) {
			t.Errorf("tuning help never mentions %s, which cannot benefit from it", f)
		}
	}
}

// The warning belongs where the button is, not only on the help page: by the
// time someone is reading about tuning they have usually already started one.
func TestTuningPageWarnsItStopsTheServer(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := httptest.NewRecorder()
	s.render(w, "tuning.html", tuningPageData{
		Title: "Kernel Tuning", Nav: "tuning", DeviceName: "AMD-gfx1100",
	})
	body := w.Body.String()
	if !strings.Contains(body, "stops the vLLM server") {
		t.Error("the tuning page does not say that starting a run takes the server down")
	}
	if !strings.Contains(body, "/help#tuning") {
		t.Error("the tuning page should link to the fuller explanation")
	}
}

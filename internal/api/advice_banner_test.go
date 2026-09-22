package api

import (
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// The strip has to say how much there is and how bad it is, because that is the
// only thing standing between the reader and a dialog they have no reason to
// open. A bare total would hide an error among three notes.
func TestTheBannerCountsBySeverity(t *testing.T) {
	v := newAdviceView([]advice.Item{
		{Severity: advice.Info, Message: "chunked prefill"},
		{Severity: advice.Warning, Message: "deprecated variable"},
		{Severity: advice.Error, Message: "could not reserve the fraction configured"},
		{Severity: advice.Warning, Message: "another deprecation"},
	})

	if v.Level != "error" {
		t.Errorf("level = %q; an error present outranks everything else", v.Level)
	}
	if want := "1 error, 2 warnings, 1 note from this start"; v.Summary != want {
		t.Errorf("summary = %q, want %q", v.Summary, want)
	}
}

// Warnings alone must not paint the strip red: the tint is the whole signal,
// and one that cries wolf on a deprecation notice stops being read.
func TestTheBannerLevelIsTheWorstPresent(t *testing.T) {
	for _, tc := range []struct {
		name  string
		items []advice.Item
		level string
	}{
		{"quiet", nil, "quiet"},
		{"notes only", []advice.Item{{Severity: advice.Info}}, "note"},
		{"warning", []advice.Item{{Severity: advice.Info}, {Severity: advice.Warning}}, "warning"},
		{"error", []advice.Item{{Severity: advice.Warning}, {Severity: advice.Error}}, "error"},
	} {
		if got := newAdviceView(tc.items).Level; got != tc.level {
			t.Errorf("%s: level = %q, want %q", tc.name, got, tc.level)
		}
	}
}

// A quiet panel says nothing in the strip. The template supplies its own
// wording there, and a summary would render beside it.
func TestAQuietBannerHasNoSummary(t *testing.T) {
	v := newAdviceView(nil)
	if v.Summary != "" {
		t.Errorf("a quiet start summarised itself as %q", v.Summary)
	}
	if v.Sig != "quiet" {
		t.Errorf("sig = %q, want the quiet marker", v.Sig)
	}
}

// The page pulses the strip when the fingerprint changes and leaves it alone
// otherwise. A fingerprint that moved on its own would pulse every ten seconds
// forever; one that did not move on new advice would never pulse at all.
func TestTheBannerFingerprintTracksContent(t *testing.T) {
	base := []advice.Item{{Severity: advice.Warning, Message: "deprecated variable", Field: "env"}}
	first := newAdviceView(base).Sig
	if again := newAdviceView(base).Sig; again != first {
		t.Errorf("the same advice fingerprinted twice: %q then %q", first, again)
	}

	more := append(append([]advice.Item{}, base...),
		advice.Item{Severity: advice.Error, Message: "could not reserve the fraction configured"})
	if got := newAdviceView(more).Sig; got == first {
		t.Error("new advice arrived and the fingerprint did not move; the strip would never pulse")
	}
}

// No process is not a crash and not an empty strip: it is the quiet one.
func TestNoProcessStillFillsTheBanner(t *testing.T) {
	s := &Server{}
	v := s.adviceSnapshot()
	if v.Level != "quiet" || v.Sig != "quiet" {
		t.Errorf("level/sig = %q/%q with no process; the strip would render untinted and pulse", v.Level, v.Sig)
	}
}

// oobTarget is the selector the panel swaps its list into out of band.
var oobTarget = regexp.MustCompile(`hx-swap-oob="innerHTML:(#[A-Za-z0-9_-]+)"`)

// The panel renders in two pieces that land in two different places, and the
// dialog is the piece that does not travel with them: it is written into
// server.html so the ten-second poll cannot swap it away mid-read. That leaves
// the selector and the element's id coupled across two files with nothing
// connecting them, and a rename on either side is silent -- the strip keeps
// working and the dialog stays on "Loading..." forever.
func TestThePanelsOutOfBandTargetExistsOnTheServerPage(t *testing.T) {
	s := testServerWithEngine(t, "test/advice",
		"vllm: error: unrecognized arguments: --enable-expert-offload",
		models.VLLMConfig{TensorParallelSize: 1, MaxModelLen: 8192})
	s.initTemplates()
	waitForObserved(t, s, func() bool { return len(s.process.Advice()) > 0 })

	req := httptest.NewRequest("GET", "/api/service/advice", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleServiceAdvice(rec, req)

	m := oobTarget.FindStringSubmatch(rec.Body.String())
	if m == nil {
		t.Fatalf("the panel renders no out-of-band swap, so the dialog would never fill:\n%.400s", rec.Body.String())
	}

	pageRec := httptest.NewRecorder()
	s.handleServerPage(pageRec, httptest.NewRequest("GET", "/server", nil))
	if pageRec.Code != 200 {
		t.Fatalf("server page: status %d", pageRec.Code)
	}
	if !strings.Contains(pageRec.Body.String(), `id="`+strings.TrimPrefix(m[1], "#")+`"`) {
		t.Errorf("the panel swaps into %s, which the server page does not contain", m[1])
	}
}

// The strip is the only thing on the page telling the reader anything is there,
// so it has to carry the level it is tinted by, the fingerprint the pulse keys
// off, and a way in.
func TestTheStripCarriesWhatThePageNeeds(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.process = process.NewManager("127.0.0.1", 0, 0)
	s.initTemplates()

	req := httptest.NewRequest("GET", "/api/service/advice", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleServiceAdvice(rec, req)

	body := rec.Body.String()
	for _, want := range []string{`class="engine-notes"`, `data-level="quiet"`, `data-sig=`, "openEngineNotes()"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q from the strip:\n%.400s", want, body)
		}
	}
}

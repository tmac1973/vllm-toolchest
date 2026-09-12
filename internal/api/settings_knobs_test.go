package api

import (
	"html"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
	"github.com/tmac1973/vllm-toolchest/variants"
)

// knobServer is settingsServer running as a named image variant, so the form
// handler has a manifest to render and validate against.
func knobServer(t *testing.T, variant string) *Server {
	t.Helper()
	s := settingsServer(t)
	s.vllmEnv = vllmenv.Env{Variant: variant, VenvRoot: "/opt/vllm"}
	return s
}

func TestKnobIsSavedFromTheForm(t *testing.T) {
	s := knobServer(t, "radiance")
	code, body := put(t, s, url.Values{
		"knob_use_r4d":       {"0"},
		"knob_skinny_gemm":   {"all"},
		"knob_draft_tau":     {"0.31"},
		"knob_dynamic_draft": {""},
	})
	if code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	got := s.cfg.KnobValues("radiance")
	for id, want := range map[string]string{"use_r4d": "0", "skinny_gemm": "all", "draft_tau": "0.31"} {
		if got[id] != want {
			t.Errorf("%s = %q, want %q", id, got[id], want)
		}
	}
	// An empty submitted value is "image default", stored as absence.
	if _, present := got["dynamic_draft"]; present {
		t.Errorf("an empty value should be stored as absent, got %v", got)
	}
}

// The distinction the twelve hand-written blocks encoded, and the one most
// likely to be lost in a loop: an empty submitted value clears a stored knob,
// but an absent field means the form never carried it and the stored value has
// to survive. Collapsing the two lets any POST that omits the section wipe
// every switch.
func TestAbsentKnobFieldKeepsTheStoredValue(t *testing.T) {
	s := knobServer(t, "radiance")

	if _, body := put(t, s, url.Values{"knob_use_r4d": {"1"}, "knob_draft_tau": {"0.28"}}); body == "" {
		t.Fatal("no response")
	}
	if s.cfg.KnobValues("radiance")["use_r4d"] != "1" {
		t.Fatal("setup failed: knob was not stored")
	}

	// A form that carries one knob and not the other.
	put(t, s, url.Values{"knob_use_r4d": {"0"}})

	got := s.cfg.KnobValues("radiance")
	if got["use_r4d"] != "0" {
		t.Errorf("submitted knob = %q, want the new value 0", got["use_r4d"])
	}
	if got["draft_tau"] != "0.28" {
		t.Errorf("omitted knob = %q, want it untouched at 0.28", got["draft_tau"])
	}
}

func TestEmptyKnobFieldClearsTheStoredValue(t *testing.T) {
	s := knobServer(t, "radiance")
	put(t, s, url.Values{"knob_skinny_gemm": {"all"}})
	if s.cfg.KnobValues("radiance")["skinny_gemm"] != "all" {
		t.Fatal("setup failed")
	}

	put(t, s, url.Values{"knob_skinny_gemm": {""}})
	if v, present := s.cfg.KnobValues("radiance")["skinny_gemm"]; present {
		t.Errorf("choosing image default should clear the knob, got %q", v)
	}
}

// Validation runs before the save, so a bad value never reaches the file. The
// htmx path renders the message as a partial rather than setting a status, the
// same way the runtime-env table reports a rejected value.
func TestInvalidKnobValueIsRejected(t *testing.T) {
	s := knobServer(t, "radiance")
	_, body := put(t, s, url.Values{"knob_use_r4d": {"perhaps"}})

	if !strings.Contains(body, "not a valid value") {
		t.Errorf("expected a validation message, got %q", body)
	}
	// The message names the control and what it would have accepted, so the
	// operator can fix it without reading the manifest.
	if !strings.Contains(body, "R4D kernel library") {
		t.Errorf("the message should name the control, got %q", body)
	}
	if _, stored := s.cfg.KnobValues("radiance")["use_r4d"]; stored {
		t.Errorf("nothing should have been stored; got %v", s.cfg.Knobs)
	}
}

// Text knobs are not second-guessed: the image is the authority on what it
// parses, and refusing an unfamiliar value here would make a knob unusable the
// moment upstream extends it.
func TestFreeTextKnobIsNotValidatedAgainstAList(t *testing.T) {
	s := knobServer(t, "radiance")
	code, body := put(t, s, url.Values{"knob_draft_schedule": {"1:9,2:9"}})
	if code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	if s.cfg.KnobValues("radiance")["draft_schedule"] != "1:9,2:9" {
		t.Error("free-text knob was not stored verbatim")
	}
}

// A variant with no manifest has no knobs to set, and a stray knob_* field
// must not create config for it.
func TestKnobFieldsAreIgnoredOnAnUndescribedVariant(t *testing.T) {
	s := knobServer(t, "some-future-image")
	code, body := put(t, s, url.Values{"knob_use_r4d": {"1"}})
	if code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	if len(s.cfg.Knobs) != 0 {
		t.Errorf("knobs were stored for a variant with no manifest: %v", s.cfg.Knobs)
	}
}

// Saving on one image must not disturb the switches belonging to another --
// the reason Knobs is keyed by variant rather than flat.
func TestSavingDoesNotTouchAnotherVariantsKnobs(t *testing.T) {
	s := knobServer(t, "radiance")
	s.cfg.SetKnobs("generic", map[string]string{"kept": "yes"})

	put(t, s, url.Values{"knob_use_r4d": {"1"}})

	if s.cfg.KnobValues("generic")["kept"] != "yes" {
		t.Errorf("another variant's knobs were disturbed: %v", s.cfg.Knobs)
	}
}

// Every manifest must render. This is the guard that lets someone add variant
// number eleven by writing one .conf file: if its knobs describe a control the
// template cannot draw, or a label the view builder cannot resolve, it fails
// here rather than as a blank box on somebody's Settings page.
//
// html/template resolves field names at execute time, not parse time, so a
// lookup it cannot perform renders nothing and reports nothing.
func TestEverySettingsPanelRenders(t *testing.T) {
	for _, d := range variants.All() {
		t.Run(d.ID, func(t *testing.T) {
			s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: d.ID, VenvRoot: "/opt/vllm"}}
			s.initTemplates()

			w := httptest.NewRecorder()
			s.handleSettingsPage(w, httptest.NewRequest("GET", "/settings", nil))
			if w.Code != 200 {
				t.Fatalf("status = %d, want 200", w.Code)
			}
			// Unescaped before matching: html/template escapes characters
			// like "+" to numeric references ("&#43;") when they arrive as
			// data rather than as literal template text. The browser renders
			// them identically, so this is a difference in the source only,
			// but a raw substring match would trip over it.
			body := html.UnescapeString(w.Body.String())

			for _, k := range d.Knobs {
				if !strings.Contains(body, `name="knob_`+k.ID+`"`) {
					t.Errorf("knob %s has no control on the page", k.ID)
				}
				if k.Label != "" && !strings.Contains(body, k.Label) {
					t.Errorf("knob %s: label %q is missing", k.ID, k.Label)
				}
				// Selects must offer every declared option, or a stored
				// value silently has no way back.
				for _, o := range k.Options {
					if o.Value != "" && !strings.Contains(body, `value="`+o.Value+`"`) {
						t.Errorf("knob %s: option %q is missing", k.ID, o.Value)
					}
				}
			}

			// A variant declaring no knobs draws no panel at all.
			if len(d.Knobs) == 0 && strings.Contains(body, "knob_") {
				t.Error("a variant with no knobs rendered a control")
			}
		})
	}
}

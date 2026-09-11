package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

// postProfile sends a profile action the way the panel does and returns the
// re-rendered panel.
func postProfile(t *testing.T, h http.HandlerFunc, id, name string) string {
	t.Helper()
	form := url.Values{"name": {name}}
	req := httptest.NewRequest("POST", "/x?id="+url.QueryEscape(id), strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	h(w, req)
	// Refusals are 200 too: htmx only swaps a 2xx, and a refusal is shown in
	// the panel rather than lost.
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	return w.Body.String()
}

func assertContains(t *testing.T, body string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(body, want) {
			t.Errorf("the panel should contain %q", want)
		}
	}
}

func TestSaveProfileFromThePanel(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})

	body := postProfile(t, s.handleSaveModelProfile, m.ID, "nightly")
	// The panel survives, with the new profile selected and named as active.
	assertContains(t, body, `name="dtype"`, "Saved profile nightly.",
		`<option value="nightly" selected>nightly — saved `, "From profile <strong>nightly</strong>.")
	if _, ok := s.registry.Profile(m.ID, "nightly"); !ok {
		t.Error("the profile was not stored")
	}

	body = postProfile(t, s.handleSaveModelProfile, m.ID, "Nightly")
	assertContains(t, body, "Replaced profile Nightly.")
	if n := len(s.registry.Profiles(m.ID)); n != 1 {
		t.Errorf("saving over a name left %d profiles, want 1", n)
	}
}

func TestSaveProfileNeedsAName(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})

	body := postProfile(t, s.handleSaveModelProfile, m.ID, "   ")
	assertContains(t, body, `name="dtype"`, "Type a name for these settings first.")
	if got := s.registry.Profiles(m.ID); len(got) != 0 {
		t.Errorf("a blank name was stored: %+v", got)
	}
}

func TestRestoreProfileFromThePanel(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	s.registry.SaveProfile(m.ID, "base", models.ProfileMeta{Variant: "rocm-source"})
	changed := m.VLLMConfig
	changed.MaxModelLen = 2048
	s.registry.UpdateConfig(m.ID, changed)

	body := postProfile(t, s.handleApplyModelProfile, m.ID, "base")

	// Checked through the registry, not only in the HTML.
	got, _ := s.registry.Get(m.ID)
	if got.VLLMConfig.MaxModelLen != 4096 {
		t.Errorf("MaxModelLen = %d after restore, want 4096", got.VLLMConfig.MaxModelLen)
	}
	assertContains(t, body, "Restored profile base.", `<option value="4096" selected>`)
	if strings.Contains(body, "edited since") || strings.Contains(body, "&#9888;") {
		t.Error("a clean restore on the same image should neither warn nor show drift")
	}
}

// The reason restore validates at all: a profile saved on one image, naming a
// backend in the free-text speculative config, restored onto an image without
// it. Applied, it would abort the engine minutes into a load.
func TestRestoreRefusesABackendTheImageLacks(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	withSpec := m.VLLMConfig
	withSpec.SpeculativeConfig = `{"method":"mtp","attention_backend":"ROCM_AITER_UNIFIED_ATTN"}`
	s.registry.UpdateConfig(m.ID, withSpec)
	s.registry.SaveProfile(m.ID, "from radiance", models.ProfileMeta{Variant: "radiance"})
	live := withSpec
	live.SpeculativeConfig = ""
	s.registry.UpdateConfig(m.ID, live)

	s.vllmEnv = vllmenv.Env{Variant: "rocm"} // declares ROCM_ATTN and TRITON_ATTN only
	body := postProfile(t, s.handleApplyModelProfile, m.ID, "from radiance")

	if got, _ := s.registry.Get(m.ID); got.VLLMConfig != live {
		t.Errorf("a refused restore changed the config:\n got %+v\nwant %+v", got.VLLMConfig, live)
	}
	assertContains(t, body, `name="dtype"`, "Profile from radiance was not restored",
		"ROCM_AITER_UNIFIED_ATTN", "rocm image")
}

// The picker's own backend is restored with a warning instead: the picker keeps
// it as a labelled option, so it is one click from fixed.
func TestRestoreWarnsOnAnUnofferedPickerBackend(t *testing.T) {
	// configTestServer's model picks ROCM_AITER_UNIFIED_ATTN, which rocm lacks.
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm"})
	s.registry.SaveProfile(m.ID, "aiter", models.ProfileMeta{Variant: "rocm"})
	live := m.VLLMConfig
	live.AttentionBackend = ""
	s.registry.UpdateConfig(m.ID, live)

	body := postProfile(t, s.handleApplyModelProfile, m.ID, "aiter")

	if got, _ := s.registry.Get(m.ID); got.VLLMConfig.AttentionBackend != "ROCM_AITER_UNIFIED_ATTN" {
		t.Errorf("AttentionBackend = %q, want it restored", got.VLLMConfig.AttentionBackend)
	}
	assertContains(t, body, "Restored profile aiter.", "is not one the rocm image offers",
		`value="ROCM_AITER_UNIFIED_ATTN" selected`)
}

func TestRestoreFromAnotherImageSaysSo(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	s.registry.SaveProfile(m.ID, "elsewhere", models.ProfileMeta{Variant: "cuda"})

	body := postProfile(t, s.handleApplyModelProfile, m.ID, "elsewhere")
	assertContains(t, body, "Restored profile elsewhere.", "It was saved on the cuda image, and this is rocm-source.")
}

func TestDeleteProfileFromThePanel(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	s.registry.SaveProfile(m.ID, "gone", models.ProfileMeta{})

	body := postProfile(t, s.handleDeleteModelProfile, m.ID, "gone")
	assertContains(t, body, "Deleted profile gone.", "(no saved profiles)")
	if len(s.registry.Profiles(m.ID)) != 0 {
		t.Error("the profile is still stored")
	}

	body = postProfile(t, s.handleDeleteModelProfile, m.ID, "gone")
	assertContains(t, body, `name="dtype"`, "There is no profile named gone.")
}

func TestProfileNamesAreEscaped(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})

	body := postProfile(t, s.handleSaveModelProfile, m.ID, `"><script>alert(1)</script>`)
	if strings.Contains(body, "<script>alert(1)") {
		t.Error("a profile name reached the page unescaped")
	}
	assertContains(t, body, "&lt;script&gt;alert(1)")
}

// The regression test for keeping the profile controls outside the form. The
// form autosaves with hx-include="closest form", so a profile control inside it
// would be posted with every save — and a select would trigger a save merely by
// being browsed.
func TestProfileControlsAreOutsideTheAutosavingForm(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	s.registry.SaveProfile(m.ID, "one", models.ProfileMeta{})

	w := httptest.NewRecorder()
	s.handleModelConfigPanel(w, httptest.NewRequest("GET", "/x?id="+url.QueryEscape(m.ID), nil))
	body := w.Body.String()

	open, end := strings.Index(body, "<form"), strings.Index(body, "</form>")
	if open < 0 || end < open {
		t.Fatal("no config form in the panel")
	}
	if form := body[open:end]; strings.Contains(form, `name="name"`) || strings.Contains(form, "profile-") {
		t.Error("a profile control is inside the autosaving config form")
	}
	if bar := strings.Index(body, `class="model-profiles"`); bar < 0 || bar > open {
		t.Error("the profile bar should render before the form")
	}
}

// A read-only registry says why on the panel before anyone tries to save.
func TestReadOnlyRegistryShowsOnThePanel(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	path := filepath.Join(s.cfg.DataDir, "config", "models.json")
	if err := os.WriteFile(path, []byte(`{"schema_version": 99, "models": {"org/model": {"id": "org/model"}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	s.registry = models.NewRegistry(s.cfg.DataDir, filepath.Join(s.cfg.DataDir, "models"))

	body := postProfile(t, s.handleSaveModelProfile, m.ID, "x")
	assertContains(t, body, "models.json is schema version 99", "Not saved: refusing to write")
}

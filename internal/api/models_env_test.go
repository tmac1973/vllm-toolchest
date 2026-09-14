package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

// putConfig submits the config form the way the panel does and returns the
// re-rendered panel.
func putConfig(t *testing.T, s *Server, id string, form url.Values) string {
	t.Helper()
	req := httptest.NewRequest("PUT", "/x?id="+url.QueryEscape(id), strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.handleUpdateModelConfig(w, req)
	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	return w.Body.String()
}

func TestModelEnvRoundTripsThroughThePanel(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})

	body := putConfig(t, s, m.ID, url.Values{
		"dtype": {"auto"}, "max_model_len": {"4096"},
		"env": {"VLLM_PLE_CPU_OFFLOAD=1\nCLAV_GDN=1"},
	})

	got, _ := s.registry.Get(m.ID)
	if got.VLLMConfig.Env != "VLLM_PLE_CPU_OFFLOAD=1\nCLAV_GDN=1" {
		t.Errorf("stored env = %q", got.VLLMConfig.Env)
	}
	// The textarea comes back populated, and the effective block shows what
	// the launch will actually apply.
	assertContains(t, body, `name="env"`, "VLLM_PLE_CPU_OFFLOAD=1", "CLAV_GDN=1", "Effective environment")

	// And the launch itself carries them.
	env := s.launchEnv(got)
	if v, _ := envValue(env, "VLLM_PLE_CPU_OFFLOAD"); v != "1" {
		t.Errorf("VLLM_PLE_CPU_OFFLOAD = %q in the launch environment, want 1", v)
	}
}

// A half-typed line must not refuse the save: the panel autosaves on every
// change, so blocking would discard the operator's other edits.
func TestModelEnvWarnsRatherThanBlocking(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})

	body := putConfig(t, s, m.ID, url.Values{
		"dtype": {"auto"}, "max_model_len": {"8192"},
		"env": {"VLLM_PLE_CPU_OFF\nCLAV_GDN=1"},
	})

	got, _ := s.registry.Get(m.ID)
	if got.VLLMConfig.MaxModelLen != 8192 {
		t.Error("a malformed env line blocked the rest of the config from saving")
	}
	if !strings.Contains(body, "ignored") {
		t.Error("the panel said nothing about the line that will not apply")
	}
	// The good line still reaches the launch; the junk one does not.
	env := s.launchEnv(got)
	if v, _ := envValue(env, "CLAV_GDN"); v != "1" {
		t.Errorf("CLAV_GDN = %q, want 1", v)
	}
	for _, kv := range env {
		if strings.HasPrefix(kv, "VLLM_PLE_CPU_OFF=") || kv == "VLLM_PLE_CPU_OFF" {
			t.Errorf("a malformed line reached the launch: %q", kv)
		}
	}
}

// Names that fight a mechanism this tool relies on warn, and still apply.
func TestModelEnvWarnsAboutRiskyNames(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})

	body := putConfig(t, s, m.ID, url.Values{
		"dtype": {"auto"}, "max_model_len": {"4096"},
		"env": {"HIP_VISIBLE_DEVICES=0"},
	})
	if !strings.Contains(body, "HIP_VISIBLE_DEVICES") || !strings.Contains(body, "hides GPUs") {
		t.Error("setting HIP_VISIBLE_DEVICES on a model should warn")
	}
	if got, _ := s.registry.Get(m.ID); got.VLLMConfig.Env == "" {
		t.Error("the warning should not have stopped the value being saved")
	}
}

// The preview is the reason this feature is usable: a name set both
// machine-wide and on the model appears once, with the value that wins.
func TestEffectiveEnvShowsTheModelWinning(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	s.cfg.RuntimeEnvExtra = "VLLM_PLE_CPU_OFFLOAD=0"

	cfg := m.VLLMConfig
	cfg.Env = "VLLM_PLE_CPU_OFFLOAD=1"
	if err := s.registry.UpdateConfig(m.ID, cfg); err != nil {
		t.Fatal(err)
	}

	lines := s.effectiveEnvLines(m)
	var seen []string
	for _, l := range lines {
		if strings.HasPrefix(l.Text, "VLLM_PLE_CPU_OFFLOAD=") {
			seen = append(seen, l.Text)
		}
	}
	if len(seen) != 1 || seen[0] != "VLLM_PLE_CPU_OFFLOAD=1" {
		t.Errorf("effective environment shows %v, want exactly VLLM_PLE_CPU_OFFLOAD=1", seen)
	}
}

// The Settings page has no model in view, so its preview must be unaffected by
// whatever any model sets.
func TestSettingsEffectiveEnvIgnoresModelEnv(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	cfg := m.VLLMConfig
	cfg.Env = "MODEL_ONLY=1"
	if err := s.registry.UpdateConfig(m.ID, cfg); err != nil {
		t.Fatal(err)
	}

	for _, l := range s.effectiveEnvLines(nil) {
		if strings.HasPrefix(l.Text, "MODEL_ONLY") {
			t.Errorf("a model's variable leaked into the machine-wide preview: %q", l.Text)
		}
	}
}

// Profiles snapshot VLLMConfig, so the environment has to travel with them —
// which is the payoff of storing it as a string rather than a map.
func TestProfilesCaptureTheModelEnv(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})

	cfg := m.VLLMConfig
	cfg.Env = "CLAV_GDN=1"
	if err := s.registry.UpdateConfig(m.ID, cfg); err != nil {
		t.Fatal(err)
	}
	if _, err := s.registry.SaveProfile(m.ID, "gdn on", models.ProfileMeta{}); err != nil {
		t.Fatal(err)
	}

	cleared := cfg
	cleared.Env = ""
	if err := s.registry.UpdateConfig(m.ID, cleared); err != nil {
		t.Fatal(err)
	}
	if _, err := s.registry.ApplyProfile(m.ID, "gdn on"); err != nil {
		t.Fatal(err)
	}

	got, _ := s.registry.Get(m.ID)
	if got.VLLMConfig.Env != "CLAV_GDN=1" {
		t.Errorf("restored env = %q, want CLAV_GDN=1", got.VLLMConfig.Env)
	}
}

// Guard the storage decision itself: EnvSet is what parses both scopes, so a
// model block and the machine-wide block must resolve identically.
func TestModelEnvUsesTheSameParserAsSettings(t *testing.T) {
	const block = "# comment\n\nA=1\nB=two words\n"
	want := config.EnvSet{Extra: block}.Pairs()

	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	cfg := m.VLLMConfig
	cfg.Env = block
	if err := s.registry.UpdateConfig(m.ID, cfg); err != nil {
		t.Fatal(err)
	}
	got, _ := s.registry.Get(m.ID)

	env := s.launchEnv(got)
	for _, pair := range want {
		name, value, _ := strings.Cut(pair, "=")
		if v, ok := envValue(env, name); !ok || v != value {
			t.Errorf("%s = %q (present=%v), want %q", name, v, ok, value)
		}
	}
}

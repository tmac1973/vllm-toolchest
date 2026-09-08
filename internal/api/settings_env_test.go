package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
)

// settingsServer is a *Server with just enough wired up to drive
// handleUpdateSettings: a config that saves to a temp path, and the templates.
func settingsServer(t *testing.T) *Server {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Loaded rather than constructed: Config remembers the path it came from
	// privately, and handleUpdateSettings saves through that. A hand-built
	// Config has no path and every save fails.
	path := filepath.Join(dir, "config", "vllmctl.yaml")
	if err := os.WriteFile(path, []byte("data_dir: "+dir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: cfg}
	s.initTemplates()
	return s
}

// put drives handleUpdateSettings with a form body, as htmx does.
func put(t *testing.T, s *Server, form url.Values) (int, string) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(form.Encode()))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	r.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.handleUpdateSettings(w, r)
	return w.Code, w.Body.String()
}

// The settings page submits every field on any change. Without the marker
// gate, a form that doesn't carry the environment table — the Server page's
// model picker, say — would read every row as empty and wipe the lot.
func TestRuntimeEnvSurvivesAFormThatDoesNotCarryIt(t *testing.T) {
	s := settingsServer(t)

	put(t, s, url.Values{
		"runtime_env_touched":    {"1"},
		"env_VLLM_LOGGING_LEVEL": {"DEBUG"},
		"runtime_env_extra":      {"MY_VAR=1"},
	})
	if got := s.cfg.RuntimeEnv["VLLM_LOGGING_LEVEL"]; got != "DEBUG" {
		t.Fatalf("setup failed: VLLM_LOGGING_LEVEL = %q", got)
	}

	// A form from somewhere else entirely.
	put(t, s, url.Values{"active_model": {"some/model"}})

	if got := s.cfg.RuntimeEnv["VLLM_LOGGING_LEVEL"]; got != "DEBUG" {
		t.Errorf("unrelated form cleared the curated value; got %q", got)
	}
	if s.cfg.RuntimeEnvExtra != "MY_VAR=1" {
		t.Errorf("unrelated form cleared the extra block; got %q", s.cfg.RuntimeEnvExtra)
	}
}

// Clearing has to remain possible: with the marker present, an empty field is
// the user setting the row back to unset.
func TestRuntimeEnvClearsWhenTheFormOwnsIt(t *testing.T) {
	s := settingsServer(t)
	put(t, s, url.Values{
		"runtime_env_touched":    {"1"},
		"env_VLLM_LOGGING_LEVEL": {"DEBUG"},
	})
	put(t, s, url.Values{
		"runtime_env_touched":    {"1"},
		"env_VLLM_LOGGING_LEVEL": {""},
	})
	if v, ok := s.cfg.RuntimeEnv["VLLM_LOGGING_LEVEL"]; ok {
		t.Errorf("expected the key to be gone, got %q", v)
	}
}

// A value outside the option's allowed set is a bug or a hand-rolled request,
// not something to store and puzzle over later.
func TestRuntimeEnvRejectsAValueOutsideTheAllowedSet(t *testing.T) {
	s := settingsServer(t)
	_, body := put(t, s, url.Values{
		"runtime_env_touched":    {"1"},
		"env_VLLM_LOGGING_LEVEL": {"LOUD"},
	})
	if !strings.Contains(body, "not one of") {
		t.Errorf("expected a validation message, got %q", body)
	}
	if len(s.cfg.RuntimeEnv) != 0 {
		t.Errorf("nothing should have been stored; got %v", s.cfg.RuntimeEnv)
	}
}

// A risky variable is saved and reported. Refusing it would be wrong — each
// has a legitimate use — but saving it silently would be worse.
func TestRiskyVariableSavesWithAWarning(t *testing.T) {
	s := settingsServer(t)
	_, body := put(t, s, url.Values{
		"runtime_env_touched": {"1"},
		"runtime_env_extra":   {"HIP_VISIBLE_DEVICES=0"},
	})
	if !strings.Contains(body, "Settings saved") {
		t.Errorf("the save should be confirmed, not reported as a failure: %q", body)
	}
	if !strings.Contains(body, "HIP_VISIBLE_DEVICES") {
		t.Errorf("expected a warning naming the variable, got %q", body)
	}
	if s.cfg.RuntimeEnvExtra != "HIP_VISIBLE_DEVICES=0" {
		t.Errorf("the value should still have been stored; got %q", s.cfg.RuntimeEnvExtra)
	}
}

func TestModelsDirValidation(t *testing.T) {
	existing := t.TempDir()
	file := filepath.Join(existing, "a-file")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name, dir, wantMsg string
	}{
		{"relative", "relative/path", "absolute path"},
		{"missing", "/no/such/directory/here", "no such file"},
		{"a file", file, "not a directory"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := settingsServer(t)
			_, body := put(t, s, url.Values{"models_dir": {tc.dir}})
			if !strings.Contains(body, tc.wantMsg) {
				t.Errorf("expected %q in %q", tc.wantMsg, body)
			}
			if s.cfg.ModelDir != "" {
				t.Errorf("a rejected path must not be stored; got %q", s.cfg.ModelDir)
			}
		})
	}

	t.Run("accepted", func(t *testing.T) {
		s := settingsServer(t)
		_, body := put(t, s, url.Values{"models_dir": {existing}})
		if s.cfg.ModelDir != existing {
			t.Errorf("ModelDir = %q, want %q", s.cfg.ModelDir, existing)
		}
		if !strings.Contains(body, "restart") {
			t.Errorf("a change to the models dir needs the restart note: %q", body)
		}
	})

	// Re-submitting the same value is what the settings page does on every
	// unrelated edit, and it must not claim a restart is due.
	t.Run("unchanged is quiet", func(t *testing.T) {
		s := settingsServer(t)
		put(t, s, url.Values{"models_dir": {existing}})
		_, body := put(t, s, url.Values{"models_dir": {existing}})
		if strings.Contains(body, "restart") {
			t.Errorf("re-saving the same path should not mention a restart: %q", body)
		}
	})
}

// Auto-start is a checkbox: an unchecked box submits nothing, so without its
// own marker any form that omitted it would turn the setting off.
func TestAutoStartNeedsItsMarker(t *testing.T) {
	s := settingsServer(t)

	put(t, s, url.Values{"auto_start_touched": {"1"}, "auto_start": {"on"}})
	if !s.cfg.AutoStart {
		t.Fatal("setup failed: auto-start not enabled")
	}

	put(t, s, url.Values{"active_model": {"some/model"}})
	if !s.cfg.AutoStart {
		t.Error("a form without the marker turned auto-start off")
	}

	put(t, s, url.Values{"auto_start_touched": {"1"}})
	if s.cfg.AutoStart {
		t.Error("the owning form should be able to turn it off")
	}
}

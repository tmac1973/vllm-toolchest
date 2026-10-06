package api

import (
	"net/url"
	"testing"
)

// The settings page fills a stored secret's field with secretMask and submits
// the whole form on any change. Changing an unrelated setting must leave the
// stored secrets alone, not save the mask as the new key and token.
func TestSecretMaskLeavesStoredSecretsAlone(t *testing.T) {
	s := settingsServer(t)
	s.cfg.APIKey = "real-key"
	s.cfg.HFToken = "hf_real"

	code, body := put(t, s, url.Values{
		"settings_form": {"1"},
		"api_key":       {secretMask},
		"hf_token":      {secretMask},
		"theme":         {"dark"},
	})
	if code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	if s.cfg.APIKey != "real-key" {
		t.Errorf("APIKey = %q, want it unchanged", s.cfg.APIKey)
	}
	if s.cfg.HFToken != "hf_real" {
		t.Errorf("HFToken = %q, want it unchanged", s.cfg.HFToken)
	}
	if s.cfg.Theme != "dark" {
		t.Errorf("Theme = %q, want the unrelated change applied", s.cfg.Theme)
	}
}

// An emptied field is still an explicit clear.
func TestEmptySecretFieldClearsIt(t *testing.T) {
	s := settingsServer(t)
	s.cfg.APIKey = "real-key"

	put(t, s, url.Values{"settings_form": {"1"}, "api_key": {""}})
	if s.cfg.APIKey != "" {
		t.Errorf("APIKey = %q, want it cleared", s.cfg.APIKey)
	}
}

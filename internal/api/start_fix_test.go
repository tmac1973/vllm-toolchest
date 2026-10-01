package api

import (
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// failingServer is a server whose engine prints output and exits 1, as a
// start that fails does.
func failingServer(t *testing.T, output string, cfg models.VLLMConfig) (*Server, *models.Model) {
	t.Helper()
	dir := t.TempDir()
	out := filepath.Join(dir, "out.txt")
	os.WriteFile(out, []byte(output+"\n"), 0o644)
	fake := filepath.Join(dir, "fakevllm")
	os.WriteFile(fake, []byte("#!/bin/sh\ncat "+out+"\nexit 1\n"), 0o755)

	s := newGoldenServer(t, goldenEnvGeneric)
	s.process = process.NewManager("127.0.0.1", 0, 0)
	s.process.SetLauncher(process.Launcher{Bin: fake})
	s.router = s.buildRouter()
	m := &models.Model{ID: "org/failing", DisplayName: "Failing", LocalPath: dir, VLLMConfig: cfg}
	s.registry.Register(m)
	if err := s.startModel(m); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200 && s.process.GetStatus().State != process.StateError; i++ {
		time.Sleep(10 * time.Millisecond)
	}
	for i := 0; i < 100 && len(s.process.Advice()) == 0; i++ {
		time.Sleep(10 * time.Millisecond)
	}
	return s, m
}

func TestAStartThatRanOutOfContext(t *testing.T) {
	s, m := failingServer(t,
		"ValueError: The model's max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache (187432).",
		models.VLLMConfig{MaxModelLen: 262144, MaxNumSeqs: 16})
	fix := s.startFixFor(m)
	if fix == nil || len(fix.Changes) != 1 || fix.Changes[0].Field != "max_model_len" || fix.Changes[0].Proposed != "187392" {
		t.Fatalf("fix = %+v", fix)
	}
	if !strings.Contains(fix.Why, "187432") {
		t.Errorf("why = %q", fix.Why)
	}

	panel := httptest.NewRecorder()
	s.router.ServeHTTP(panel, httptest.NewRequest("GET", "/api/models/config-panel?id=org/failing", nil))
	if !strings.Contains(panel.Body.String(), "The last start failed") || !strings.Contains(panel.Body.String(), "Apply this fix") {
		t.Errorf("the config panel does not show the fix:\n%s", panel.Body.String()[:min(800, len(panel.Body.String()))])
	}

	rec := post(s, "/api/models/autoconfig/fix?id=org/failing", url.Values{})
	if !strings.Contains(rec.Body.String(), "Applied.") {
		t.Fatalf("response: %s", rec.Body.String())
	}
	m, _ = s.registry.Get("org/failing")
	if m.VLLMConfig.MaxModelLen != 187392 {
		t.Errorf("context = %d", m.VLLMConfig.MaxModelLen)
	}
	if s.startFixFor(m) != nil {
		t.Error("the notice outlived the change it asked for")
	}
	if strings.Contains(rec.Body.String(), "The last start failed") {
		t.Error("the re-rendered panel still shows the notice")
	}
}

func TestAnUnrecognisedFlagIsRemoved(t *testing.T) {
	s, m := failingServer(t,
		"vllm: error: unrecognized arguments: --enable-reasoning",
		models.VLLMConfig{MaxModelLen: 8192, MaxNumSeqs: 16, ExtraFlags: "--enable-reasoning --keep-me 1"})
	fix := s.startFixFor(m)
	if fix == nil || len(fix.Changes) != 1 || fix.Changes[0].Current != "--enable-reasoning" {
		t.Fatalf("fix = %+v", fix)
	}
	post(s, "/api/models/autoconfig/fix?id=org/failing", url.Values{})
	m, _ = s.registry.Get("org/failing")
	if m.VLLMConfig.ExtraFlags != "--keep-me 1" {
		t.Errorf("extra flags = %q", m.VLLMConfig.ExtraFlags)
	}
}

func TestABareOutOfMemoryHalvesTheContext(t *testing.T) {
	s, m := failingServer(t, "torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB",
		models.VLLMConfig{MaxModelLen: 131072, MaxNumSeqs: 16})
	if fix := s.startFixFor(m); fix == nil || len(fix.Changes) != 1 || fix.Changes[0].Proposed != "65536" {
		t.Fatalf("fix = %+v", fix)
	}
	s, m = failingServer(t, "torch.cuda.OutOfMemoryError: HIP out of memory.", models.VLLMConfig{MaxModelLen: 2048, MaxNumSeqs: 16})
	fix := s.startFixFor(m)
	if fix == nil || len(fix.Changes) != 0 || fix.Why == "" {
		t.Errorf("at the floor: %+v; want the explanation and no change", fix)
	}
}

func TestTheFixKeepsTheAutoconfigProfileInStep(t *testing.T) {
	s, m := failingServer(t,
		"ValueError: The model's max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache (187432).",
		models.VLLMConfig{MaxModelLen: 262144, MaxNumSeqs: 16})
	s.registry.SaveProfileFrom(m.ID, models.AutoconfigProfileName, models.ConfigProfile{
		Config: m.VLLMConfig, Source: models.ProfileSourceAutoconfig, Autoconfig: &models.AutoconfigRecord{Class: models.ContextMax},
	})
	s.registry.ApplyProfile(m.ID, models.AutoconfigProfileName)
	post(s, "/api/models/autoconfig/fix?id=org/failing", url.Values{})

	p, _ := s.registry.Profile(m.ID, models.AutoconfigProfileName)
	if p.Config.MaxModelLen != 187392 || len(p.Notes) != 1 || !strings.Contains(p.Notes[0].Reason, "room for 187432") {
		t.Errorf("profile: %+v", p)
	}
	if name, modified := s.registry.ActiveProfile(m.ID); name != models.AutoconfigProfileName || modified {
		t.Errorf("active = %q modified = %v", name, modified)
	}
}

func TestNoFixForAnotherModelOrASuccess(t *testing.T) {
	s, _ := failingServer(t,
		"ValueError: The model's max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache (187432).",
		models.VLLMConfig{MaxModelLen: 262144})
	other := &models.Model{ID: "org/other"}
	s.registry.Register(other)
	if s.startFixFor(other) != nil {
		t.Error("a fix was offered for a model that was not started")
	}
	rec := post(s, "/api/models/autoconfig/fix?id=org/other", url.Values{})
	if !strings.Contains(rec.Body.String(), "Nothing to apply") {
		t.Errorf("response: %s", rec.Body.String())
	}
}

// The same failure as vLLM 0.29 reports it, as seen on compute.
func TestAStartThatRanOutOfContextOnVLLM029(t *testing.T) {
	s, m := failingServer(t,
		"(EngineCore pid=7579) ERROR 10-01 04:29:49 [core.py:1385] ValueError: To serve at least one request with the model's max seq len (262144), (5.84 GiB KV cache is needed, which is larger than the available KV cache memory (4.92 GiB). Based on the available memory, the estimated maximum model length is 214240.",
		models.VLLMConfig{MaxModelLen: 262144, MaxNumSeqs: 16})
	fix := s.startFixFor(m)
	if fix == nil || len(fix.Changes) != 1 || fix.Changes[0].Proposed != "214016" {
		t.Fatalf("fix = %+v", fix)
	}
}

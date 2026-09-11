package models

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
)

// profileRegistry is a registry with one model registered under org/model.
func profileRegistry(t *testing.T, cfg VLLMConfig) (*Registry, string) {
	t.Helper()
	reg, dir := pendingRegistry(t)
	if err := reg.Register(&Model{ID: "org/model", VLLMConfig: cfg}); err != nil {
		t.Fatal(err)
	}
	return reg, dir
}

func TestSaveProfileRoundTrip(t *testing.T) {
	cfg := VLLMConfig{MaxModelLen: 131072, TensorParallelSize: 4, SpeculativeConfig: `{"method":"mtp"}`}
	reg, _ := profileRegistry(t, cfg)

	replaced, err := reg.SaveProfile("org/model", "  tp4   mtp ", ProfileMeta{Variant: "radiance", VariantVersion: "0.27"})
	if err != nil {
		t.Fatal(err)
	}
	if replaced {
		t.Error("a new name should not report a replacement")
	}

	p, ok := reg.Profile("org/model", "TP4 MTP")
	if !ok {
		t.Fatal("profile not found by its folded name")
	}
	if p.Name != "tp4 mtp" {
		t.Errorf("stored name = %q, want the normalised %q", p.Name, "tp4 mtp")
	}
	if p.Config != cfg {
		t.Errorf("config not snapshotted:\n got %+v\nwant %+v", p.Config, cfg)
	}
	if p.Variant != "radiance" || p.VariantVersion != "0.27" || p.SavedAt.IsZero() {
		t.Errorf("provenance not recorded: %+v", p)
	}
}

// Overwriting is the operator's call and is not asked about, but it must be a
// replacement: two entries a dropdown renders identically are worse than one.
func TestSaveProfileReplacesAFoldedName(t *testing.T) {
	reg, _ := profileRegistry(t, VLLMConfig{MaxModelLen: 1})
	reg.SaveProfile("org/model", "long ctx", ProfileMeta{})
	reg.UpdateConfig("org/model", VLLMConfig{MaxModelLen: 2})

	replaced, err := reg.SaveProfile("org/model", "Long  Ctx", ProfileMeta{})
	if err != nil {
		t.Fatal(err)
	}
	if !replaced {
		t.Error("saving over an existing name should report a replacement")
	}
	got := reg.Profiles("org/model")
	if len(got) != 1 {
		t.Fatalf("expected one profile, got %d: %+v", len(got), got)
	}
	if got[0].Name != "Long Ctx" || got[0].Config.MaxModelLen != 2 {
		t.Errorf("the entry was not replaced: %+v", got[0])
	}
}

func TestProfilesAreScopedAndOrdered(t *testing.T) {
	reg, _ := profileRegistry(t, VLLMConfig{})
	reg.Register(&Model{ID: "org/other"})
	for _, name := range []string{"zeta", "Alpha", "mid"} {
		reg.SaveProfile("org/model", name, ProfileMeta{})
	}
	reg.SaveProfile("org/other", "alpha", ProfileMeta{})

	got := reg.Profiles("org/model")
	if len(got) != 3 {
		t.Fatalf("expected three profiles for org/model, got %+v", got)
	}
	for i, want := range []string{"Alpha", "mid", "zeta"} {
		if got[i].Name != want || got[i].ModelID != "org/model" {
			t.Errorf("position %d: got %s/%s, want org/model/%s", i, got[i].ModelID, got[i].Name, want)
		}
	}
	if _, ok := reg.Profile("org/other", "Alpha"); !ok {
		t.Error("the same name on another model should be its own profile")
	}
}

func TestProfilesSurviveAReload(t *testing.T) {
	reg, dir := profileRegistry(t, VLLMConfig{MaxModelLen: 8192})
	reg.SaveProfile("org/model", "nightly", ProfileMeta{})

	reloaded := NewRegistry(dir, filepath.Join(dir, "models"))
	p, ok := reloaded.Profile("org/model", "nightly")
	if !ok || p.Config.MaxModelLen != 8192 {
		t.Errorf("profile did not survive the reload: %+v", p)
	}
	if name, _ := reloaded.ActiveProfile("org/model"); name != "nightly" {
		t.Errorf("active profile after reload = %q, want nightly", name)
	}
}

func TestApplyProfileRestoresTheConfig(t *testing.T) {
	want := VLLMConfig{MaxModelLen: 131072, AttentionBackend: "R4D"}
	reg, _ := profileRegistry(t, want)
	reg.SaveProfile("org/model", "baseline", ProfileMeta{})
	reg.UpdateConfig("org/model", VLLMConfig{MaxModelLen: 4096})

	got, err := reg.ApplyProfile("org/model", "BASELINE")
	if err != nil {
		t.Fatal(err)
	}
	m, _ := reg.Get("org/model")
	if got != want || m.VLLMConfig != want {
		t.Errorf("config not restored:\n returned %+v\n   stored %+v\n     want %+v", got, m.VLLMConfig, want)
	}
	if name, modified := reg.ActiveProfile("org/model"); name != "baseline" || modified {
		t.Errorf("ActiveProfile() = %q, %v; want baseline, false", name, modified)
	}
}

func TestApplyMissingProfileChangesNothing(t *testing.T) {
	reg, _ := profileRegistry(t, VLLMConfig{MaxModelLen: 4096})

	if _, err := reg.ApplyProfile("org/model", "nope"); !errors.Is(err, ErrProfileNotFound) {
		t.Errorf("err = %v, want ErrProfileNotFound", err)
	}
	if m, _ := reg.Get("org/model"); m.VLLMConfig.MaxModelLen != 4096 {
		t.Errorf("config changed: %+v", m.VLLMConfig)
	}
}

// Drift is what the "edited since" line and a benchmark's label both rely on.
func TestActiveProfileReportsDrift(t *testing.T) {
	reg, _ := profileRegistry(t, VLLMConfig{MaxModelLen: 4096})

	if name, modified := reg.ActiveProfile("org/model"); name != "" || modified {
		t.Errorf("a model with no profile: got %q, %v", name, modified)
	}

	reg.SaveProfile("org/model", "base", ProfileMeta{})
	if _, modified := reg.ActiveProfile("org/model"); modified {
		t.Error("just saved, but reported as modified")
	}

	reg.UpdateConfig("org/model", VLLMConfig{MaxModelLen: 8192})
	if name, modified := reg.ActiveProfile("org/model"); name != "base" || !modified {
		t.Errorf("after an edit: got %q, %v; want base, true", name, modified)
	}
}

func TestDeleteProfileLeavesTheLiveConfig(t *testing.T) {
	cfg := VLLMConfig{MaxModelLen: 4096}
	reg, _ := profileRegistry(t, cfg)
	reg.SaveProfile("org/model", "base", ProfileMeta{})

	if !reg.DeleteProfile("org/model", "Base") {
		t.Fatal("deleting a profile that exists should report true")
	}
	if len(reg.Profiles("org/model")) != 0 {
		t.Error("the profile is still listed")
	}
	m, _ := reg.Get("org/model")
	if m.VLLMConfig != cfg {
		t.Errorf("deleting a profile changed the live config: %+v", m.VLLMConfig)
	}
	if name, _ := reg.ActiveProfile("org/model"); name != "" {
		t.Errorf("the label still names the deleted profile: %q", name)
	}
	if reg.DeleteProfile("org/model", "base") {
		t.Error("deleting a missing profile should report false")
	}
}

// The case that decided where profiles are stored: a model removed to free
// disk and pulled again keeps them.
func TestProfilesSurviveARedownload(t *testing.T) {
	reg, _ := profileRegistry(t, VLLMConfig{MaxModelLen: 131072})
	reg.SaveProfile("org/model", "long", ProfileMeta{})

	reg.Delete("org/model", false)
	reg.Register(&Model{ID: "org/model", VLLMConfig: VLLMConfig{MaxModelLen: 4096}})

	if _, ok := reg.Profile("org/model", "long"); !ok {
		t.Fatal("the profile was lost with the model record")
	}
	// The fresh record's config came from defaults, not from the profile.
	if name, _ := reg.ActiveProfile("org/model"); name != "" {
		t.Errorf("a fresh record claims profile %q", name)
	}
}

func TestSaveProfileRejectsBadNames(t *testing.T) {
	reg, _ := profileRegistry(t, VLLMConfig{})
	for _, name := range []string{"", "   ", strings.Repeat("x", MaxProfileNameLen+1)} {
		if _, err := reg.SaveProfile("org/model", name, ProfileMeta{}); err == nil {
			t.Errorf("name %q was accepted", name)
		}
	}
	if got := reg.Profiles("org/model"); len(got) != 0 {
		t.Errorf("a rejected name was stored: %+v", got)
	}
	if _, err := reg.SaveProfile("org/model", strings.Repeat("é", MaxProfileNameLen), ProfileMeta{}); err != nil {
		t.Errorf("the limit counts characters, not bytes: %v", err)
	}
}

func TestProfilesOnAReadOnlyRegistry(t *testing.T) {
	dir := t.TempDir()
	writeRegistryFile(t, dir, fmt.Sprintf(`{"schema_version": %d,
		"models": {"org/model": {"id": "org/model"}},
		"config_profiles": [{"model_id": "org/model", "name": "kept"}]}`, schemaVersion+1))
	reg := NewRegistry(dir, filepath.Join(dir, "models"))

	if _, err := reg.SaveProfile("org/model", "new", ProfileMeta{}); err == nil {
		t.Error("SaveProfile succeeded on a read-only registry")
	}
	if _, err := reg.ApplyProfile("org/model", "kept"); err == nil {
		t.Error("ApplyProfile succeeded on a read-only registry")
	}
	if reg.DeleteProfile("org/model", "kept") {
		t.Error("DeleteProfile reported success on a read-only registry")
	}
	if _, ok := reg.Profile("org/model", "kept"); !ok {
		t.Error("profiles from a read-only file should still be listed")
	}
}

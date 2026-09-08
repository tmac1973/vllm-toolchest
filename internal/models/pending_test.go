package models

import (
	"path/filepath"
	"testing"
	"time"
)

func pendingRegistry(t *testing.T) (*Registry, string) {
	t.Helper()
	dir := t.TempDir()
	return NewRegistry(dir, filepath.Join(dir, "models")), dir
}

// The payoff of a pending config is entirely in the claim: restoring one is
// pointless if the model arriving later doesn't pick it up.
func TestPendingConfigIsClaimedOnRegistration(t *testing.T) {
	reg, _ := pendingRegistry(t)
	want := VLLMConfig{MaxModelLen: 131072, TensorParallelSize: 2, KVCacheDtype: "fp8"}
	if err := reg.SetPendingConfig(PendingConfig{ModelID: "org/model", Config: want, SavedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}

	// A model registers with the defaults it was born with...
	if err := reg.Register(&Model{ID: "org/model", VLLMConfig: VLLMConfig{MaxModelLen: 4096}}); err != nil {
		t.Fatal(err)
	}

	m, ok := reg.Get("org/model")
	if !ok {
		t.Fatal("model not registered")
	}
	if m.VLLMConfig != want {
		t.Errorf("config was not claimed:\n got %+v\nwant %+v", m.VLLMConfig, want)
	}
	if len(reg.PendingConfigs()) != 0 {
		t.Error("the claimed entry should be gone from the pending list")
	}
}

// A held config must not attach itself to some other model.
func TestPendingConfigOnlyClaimsItsOwnModel(t *testing.T) {
	reg, _ := pendingRegistry(t)
	reg.SetPendingConfig(PendingConfig{ModelID: "org/wanted", Config: VLLMConfig{MaxModelLen: 999}})

	reg.Register(&Model{ID: "org/other", VLLMConfig: VLLMConfig{MaxModelLen: 4096}})

	m, _ := reg.Get("org/other")
	if m.VLLMConfig.MaxModelLen != 4096 {
		t.Errorf("an unrelated model was given the held config: %+v", m.VLLMConfig)
	}
	if len(reg.PendingConfigs()) != 1 {
		t.Error("the entry should still be waiting for its own model")
	}
}

// Restoring the same backup twice is normal — it must refresh, not accumulate.
func TestSetPendingConfigUpserts(t *testing.T) {
	reg, _ := pendingRegistry(t)
	reg.SetPendingConfig(PendingConfig{ModelID: "org/model", Config: VLLMConfig{MaxModelLen: 1}})
	reg.SetPendingConfig(PendingConfig{ModelID: "org/model", Config: VLLMConfig{MaxModelLen: 2}})

	got := reg.PendingConfigs()
	if len(got) != 1 {
		t.Fatalf("expected one entry, got %d", len(got))
	}
	if got[0].Config.MaxModelLen != 2 {
		t.Errorf("the entry was not refreshed: %+v", got[0].Config)
	}
}

func TestPendingConfigsAreOrdered(t *testing.T) {
	reg, _ := pendingRegistry(t)
	for _, id := range []string{"zeta/m", "alpha/m", "mid/m"} {
		reg.SetPendingConfig(PendingConfig{ModelID: id})
	}
	got := reg.PendingConfigs()
	for i, want := range []string{"alpha/m", "mid/m", "zeta/m"} {
		if got[i].ModelID != want {
			t.Errorf("position %d: got %s, want %s", i, got[i].ModelID, want)
		}
	}
}

// Pending entries have to outlive a restart: the model they are waiting for
// may take a long download to arrive.
func TestPendingConfigsSurviveAReload(t *testing.T) {
	reg, dir := pendingRegistry(t)
	reg.SetPendingConfig(PendingConfig{ModelID: "org/model", Config: VLLMConfig{MaxModelLen: 8192}})

	reloaded := NewRegistry(dir, filepath.Join(dir, "models"))
	got := reloaded.PendingConfigs()
	if len(got) != 1 || got[0].Config.MaxModelLen != 8192 {
		t.Errorf("pending entries did not survive the reload: %+v", got)
	}
}

func TestDiscardPendingConfig(t *testing.T) {
	reg, _ := pendingRegistry(t)
	reg.SetPendingConfig(PendingConfig{ModelID: "org/model"})

	if !reg.DiscardPendingConfig("org/model") {
		t.Error("discarding an entry that exists should report true")
	}
	if len(reg.PendingConfigs()) != 0 {
		t.Error("the entry is still there")
	}
	if reg.DiscardPendingConfig("org/model") {
		t.Error("discarding a missing entry should report false")
	}
}

// The returned slice is a copy; a caller mutating it must not reach into the
// registry's state.
func TestPendingConfigsReturnsACopy(t *testing.T) {
	reg, _ := pendingRegistry(t)
	reg.SetPendingConfig(PendingConfig{ModelID: "org/model", Config: VLLMConfig{MaxModelLen: 1}})

	got := reg.PendingConfigs()
	got[0].Config.MaxModelLen = 999

	if reg.PendingConfigs()[0].Config.MaxModelLen != 1 {
		t.Error("mutating the returned slice changed the registry")
	}
}

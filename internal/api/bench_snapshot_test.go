package api

import (
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

func TestConfigSnapshotRecordsTheProfileAndTheWiderConfig(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: "rocm-source"})
	s.registry.SaveProfile(m.ID, "mtp", models.ProfileMeta{})

	snap := s.configSnapshotFromModel(m)
	if snap.ProfileName != "mtp" || snap.ProfileModified {
		t.Errorf("profile = %q, modified = %v; want mtp, false", snap.ProfileName, snap.ProfileModified)
	}
	// The settings a profile exists to A/B, which the snapshot used to drop.
	c := m.VLLMConfig
	if snap.SpeculativeConfig != c.SpeculativeConfig || snap.AttentionBackend != c.AttentionBackend ||
		snap.CompilationConfig != c.CompilationConfig || snap.MambaCacheMode != c.MambaCacheMode ||
		snap.KVCacheMemory != c.KVCacheMemory || !snap.DisableAsyncScheduling {
		t.Errorf("the wider config was not recorded: %+v", snap)
	}

	edited := m.VLLMConfig
	edited.MaxNumSeqs = 8
	s.registry.UpdateConfig(m.ID, edited)
	snap = s.configSnapshotFromModel(m)
	if !snap.ProfileModified {
		t.Error("a config edited since its profile should be recorded as modified")
	}
	// Neither of the two builders this replaced set it.
	if snap.MaxNumSeqs != 8 {
		t.Errorf("MaxNumSeqs = %d, want 8", snap.MaxNumSeqs)
	}
}

// runColumns and runValues are index-aligned; a column added to one and not
// the other mislabels every column after it without failing anything else.
func TestCSVColumnsAlignWithValues(t *testing.T) {
	run := benchmark.BenchmarkRun{Config: benchmark.ConfigSnapshot{
		ProfileName: "mtp", ProfileModified: true, AttentionBackend: "R4D", ExtraFlags: "--swap-space 4",
	}}
	vals := runValues(run)
	if len(vals) != len(runColumns) {
		t.Fatalf("%d values for %d columns", len(vals), len(runColumns))
	}
	idx := map[string]int{}
	for i, name := range runColumns {
		idx[name] = i
	}
	for col, want := range map[string]string{
		"profile": "mtp", "profile_edited": "true", "attention_backend": "R4D", "extra_flags": "--swap-space 4",
	} {
		if got := vals[idx[col]]; got != want {
			t.Errorf("%s = %q, want %q", col, got, want)
		}
	}
}

package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "vllmctl.yaml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// The pre-manifest shape. Its yaml keys are the knob ids, so the fold is a
// reparent and cannot lose a value — which is the whole reason the manifest
// slugs were named after these keys.
const legacyYAML = `
listen_addr: ":3000"
radiance:
    use_r4d: "1"
    skinny_gemm: all
    draft_tau: "0.28"
`

func TestLegacyRadianceBlockIsReparented(t *testing.T) {
	cfg, err := Load(writeConfig(t, legacyYAML))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"use_r4d": "1", "skinny_gemm": "all", "draft_tau": "0.28"}
	got := cfg.KnobValues("radiance")
	for id, v := range want {
		if got[id] != v {
			t.Errorf("knobs.radiance.%s = %q, want %q", id, got[id], v)
		}
	}
	if len(got) != len(want) {
		t.Errorf("knobs.radiance = %v, want exactly %v", got, want)
	}
	if cfg.LegacyRadiance != nil {
		t.Error("the legacy block should be cleared once folded, so Save stops writing it")
	}
}

// The file has to rewrite itself into the new shape, or every load pays the
// migration again and the old block lingers as a second source of truth.
func TestSaveRetiresTheLegacyBlock(t *testing.T) {
	path := writeConfig(t, legacyYAML)
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.Save(""); err != nil {
		t.Fatal(err)
	}
	out, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	if strings.Contains(s, "\nradiance:") {
		t.Errorf("the legacy radiance: key survived a save:\n%s", s)
	}
	if !strings.Contains(s, "knobs:") || !strings.Contains(s, "use_r4d") {
		t.Errorf("the knobs block is missing after save:\n%s", s)
	}

	// And the rewritten file loads back to the same values.
	again, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if again.KnobValues("radiance")["skinny_gemm"] != "all" {
		t.Errorf("round trip lost a value: %v", again.KnobValues("radiance"))
	}
}

// A value already under knobs is authoritative: a file written by a current
// build beats a stale legacy block someone left behind by hand.
func TestKnobsWinOverTheLegacyBlock(t *testing.T) {
	cfg, err := Load(writeConfig(t, `
knobs:
    radiance:
        use_r4d: "0"
radiance:
    use_r4d: "1"
    preshuffle: "1"
`))
	if err != nil {
		t.Fatal(err)
	}
	got := cfg.KnobValues("radiance")
	if got["use_r4d"] != "0" {
		t.Errorf("use_r4d = %q, want the knobs value 0", got["use_r4d"])
	}
	// A key only the legacy block had still comes across.
	if got["preshuffle"] != "1" {
		t.Errorf("preshuffle = %q, want 1 folded from the legacy block", got["preshuffle"])
	}
}

// Precedence the hand-written version had, and the reason migrateLegacyKnobs
// runs before applyEnvOverrides: the container environment beats the file.
func TestKnobEnvOverrideBeatsTheFile(t *testing.T) {
	t.Setenv("RADIANCE_USE_R4D", "0")
	cfg, err := Load(writeConfig(t, legacyYAML))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.KnobValues("radiance")["use_r4d"]; got != "0" {
		t.Errorf("use_r4d = %q, want 0 from the environment", got)
	}
}

// Unset means "leave the image's own default in place". Emitting NAME= would
// export an empty value, which these images read as "off" — silently turning
// off the very feature the variant exists for.
func TestKnobEnvSkipsUnset(t *testing.T) {
	cfg := &Config{Knobs: map[string]map[string]string{
		"radiance": {"use_r4d": "1", "draft_tau": "  "},
	}}
	env := cfg.KnobEnv("radiance")
	joined := strings.Join(env, "\n")
	if !strings.Contains(joined, "RADIANCE_USE_R4D=1") {
		t.Errorf("set knob missing from %v", env)
	}
	if strings.Contains(joined, "RADIANCE_DRAFT_TAU") {
		t.Errorf("whitespace-only knob was exported: %v", env)
	}
}

// A variant we cannot describe contributes nothing rather than erroring: there
// is no environment we could correctly emit for it.
func TestKnobEnvUnknownVariantIsEmpty(t *testing.T) {
	cfg := &Config{Knobs: map[string]map[string]string{"radiance": {"use_r4d": "1"}}}
	if env := cfg.KnobEnv("some-future-image"); env != nil {
		t.Errorf("got %v, want nothing for a variant with no manifest", env)
	}
}

// Knobs for the variant you are not running must survive. A config outlives
// the image it was written on.
func TestSetKnobsLeavesOtherVariantsAlone(t *testing.T) {
	cfg := &Config{}
	cfg.SetKnobs("radiance", map[string]string{"use_r4d": "1"})
	cfg.SetKnobs("generic", map[string]string{})
	if cfg.KnobValues("radiance")["use_r4d"] != "1" {
		t.Error("writing one variant's knobs disturbed another's")
	}
	// An empty set removes the variant rather than storing an empty map, so
	// the saved yaml does not accumulate empty blocks.
	if _, present := cfg.Knobs["generic"]; present {
		t.Error("an empty knob set should not be stored")
	}
}

func TestKnobSetValidate(t *testing.T) {
	for _, tc := range []struct {
		name    string
		set     KnobSet
		wantErr string
	}{
		{"valid select", KnobSet{"radiance", map[string]string{"skinny_gemm": "all"}}, ""},
		{"empty is the image default", KnobSet{"radiance", map[string]string{"skinny_gemm": ""}}, ""},
		{"free text is not second-guessed", KnobSet{"radiance", map[string]string{"draft_tau": "0.31"}}, ""},
		{"value outside the option list", KnobSet{"radiance", map[string]string{"use_r4d": "maybe"}}, "not a valid value"},
		{"unknown knob", KnobSet{"radiance", map[string]string{"nonesuch": "1"}}, `no knob "nonesuch"`},
		{"unknown variant with no values", KnobSet{"future", nil}, ""},
		{"unknown variant with values", KnobSet{"future", map[string]string{"x": "1"}}, "no manifest describes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.set.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("got %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
}

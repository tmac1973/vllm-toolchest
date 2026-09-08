package config

import (
	"strings"
	"testing"
)

// The curated table is data, and a wrong entry there is a control that does
// nothing (or the wrong thing) with no error anywhere. These checks are cheap
// insurance against the table drifting.
func TestRuntimeEnvOptionsAreWellFormed(t *testing.T) {
	seen := map[string]bool{}
	for _, o := range RuntimeEnvOptions() {
		if o.Name == "" || o.Label == "" || o.Help == "" {
			t.Errorf("%q: name, label and help are all required", o.Name)
		}
		if seen[o.Name] {
			t.Errorf("%s: listed twice", o.Name)
		}
		seen[o.Name] = true
		if !validEnvName(o.Name) {
			t.Errorf("%s: not a valid environment variable name", o.Name)
		}
		// A value set must offer "unset" as its first entry, or there is no
		// way back to "leave the environment alone" once a value is picked.
		if len(o.Values) > 0 && o.Values[0] != "" {
			t.Errorf("%s: first allowed value must be \"\" (unset), got %q", o.Name, o.Values[0])
		}
		// An option cannot be both a picker and a free-text field.
		if len(o.Values) > 0 && o.Example != "" {
			t.Errorf("%s: has both a value list and a free-text example", o.Name)
		}
	}
}

// The curated set must not offer a variable the warnings call harmful, or the
// UI would recommend and warn about the same thing.
func TestCuratedOptionsAreNotRisky(t *testing.T) {
	for _, o := range RuntimeEnvOptions() {
		if _, bad := riskyEnvVars[o.Name]; bad {
			t.Errorf("%s is both curated and listed as risky", o.Name)
		}
		if strings.HasPrefix(o.Name, radiancePrefix) {
			t.Errorf("%s is owned by the Radiance section and must not be curated here", o.Name)
		}
	}
}

func TestPairsSkipsUnsetAndSorts(t *testing.T) {
	e := EnvSet{Curated: map[string]string{
		"VLLM_LOGGING_LEVEL":         "DEBUG",
		"VLLM_ROCM_USE_AITER":        "",   // unset: must not appear
		"VLLM_DISABLE_COMPILE_CACHE": "  ", // whitespace is still unset
	}}
	got := e.Pairs()
	want := []string{"VLLM_LOGGING_LEVEL=DEBUG"}
	if len(got) != len(want) || got[0] != want[0] {
		t.Errorf("got %v, want %v", got, want)
	}
}

// An unset curated option means "don't touch the environment". Emitting
// NAME= instead would export an empty value, which for vLLM's truthiness
// parsing is a real (false) value, silently turning a default-on feature off.
func TestUnsetIsNotEmptyValue(t *testing.T) {
	e := EnvSet{Curated: map[string]string{"VLLM_ROCM_USE_AITER_MHA": ""}}
	for _, p := range e.Pairs() {
		if strings.HasPrefix(p, "VLLM_ROCM_USE_AITER_MHA=") {
			t.Errorf("unset option was exported as %q", p)
		}
	}
}

func TestExtraOverridesCurated(t *testing.T) {
	e := EnvSet{
		Curated: map[string]string{"VLLM_LOGGING_LEVEL": "INFO"},
		Extra:   "VLLM_LOGGING_LEVEL=DEBUG\n# a comment\n\n",
	}
	got := e.Pairs()
	if len(got) != 1 || got[0] != "VLLM_LOGGING_LEVEL=DEBUG" {
		t.Errorf("extra should win over curated; got %v", got)
	}
}

func TestParseExtraEnv(t *testing.T) {
	pairs, malformed := parseExtraEnv(strings.Join([]string{
		"GOOD=1",
		"  SPACED = value with spaces  ", // name trimmed, value kept verbatim after =
		"# comment",
		"",
		"NOEQUALS",
		"=novalue",
		"9BAD=1",
		"WITH_EQUALS=a=b",
	}, "\n"))

	byName := map[string]string{}
	for _, p := range pairs {
		byName[p.Name] = p.Value
	}
	if byName["GOOD"] != "1" {
		t.Errorf("GOOD: got %q", byName["GOOD"])
	}
	// An = inside the value is part of the value; only the first splits.
	if byName["WITH_EQUALS"] != "a=b" {
		t.Errorf("WITH_EQUALS: got %q", byName["WITH_EQUALS"])
	}
	// "SPACED " has a trailing space in its name, which is not a valid name.
	if len(malformed) != 4 {
		t.Errorf("expected 4 malformed lines, got %d: %v", len(malformed), malformed)
	}
}

func TestValidateRejectsUnknownAndBadValues(t *testing.T) {
	if err := (EnvSet{Curated: map[string]string{"NOT_A_REAL_VAR": "1"}}).Validate(); err == nil {
		t.Error("unknown curated variable should be rejected")
	}
	if err := (EnvSet{Curated: map[string]string{"VLLM_LOGGING_LEVEL": "LOUD"}}).Validate(); err == nil {
		t.Error("value outside the allowed set should be rejected")
	}
	if err := (EnvSet{Curated: map[string]string{"VLLM_LOGGING_LEVEL": "DEBUG"}}).Validate(); err != nil {
		t.Errorf("allowed value rejected: %v", err)
	}
	// Extra is free-form by design: an unknown name there is fine.
	if err := (EnvSet{Extra: "SOMETHING_ELSE=1"}).Validate(); err != nil {
		t.Errorf("unknown name in extra should be allowed: %v", err)
	}
	if err := (EnvSet{Extra: "NOT KEY VALUE"}).Validate(); err == nil {
		t.Error("malformed extra line should be rejected")
	}
}

func TestWarnings(t *testing.T) {
	e := EnvSet{Extra: "HIP_VISIBLE_DEVICES=0\nRADIANCE_USE_R4D=0\nHARMLESS=1"}
	w := e.Warnings()
	if len(w) != 2 {
		t.Fatalf("expected 2 warnings, got %d: %v", len(w), w)
	}
	joined := strings.Join(w, "\n")
	if !strings.Contains(joined, "HIP_VISIBLE_DEVICES") {
		t.Error("risky variable not reported")
	}
	// A RADIANCE_* variable is not in riskyEnvVars; it warns because the
	// Radiance section is applied after this one and therefore wins.
	if !strings.Contains(joined, "RADIANCE_USE_R4D") || !strings.Contains(joined, "wins") {
		t.Errorf("radiance override not explained: %v", w)
	}
}

// Warnings must never block a save, however alarming.
func TestWarningsDoNotInvalidate(t *testing.T) {
	e := EnvSet{Extra: "HSA_OVERRIDE_GFX_VERSION=11.0.0"}
	if len(e.Warnings()) == 0 {
		t.Fatal("expected a warning")
	}
	if err := e.Validate(); err != nil {
		t.Errorf("a risky variable must warn, not fail validation: %v", err)
	}
}

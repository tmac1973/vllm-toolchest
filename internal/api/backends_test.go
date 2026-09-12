package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// A backend the running stack does not have is not a harmless extra option:
// picking it aborts the engine minutes into a load. So a variant nothing
// describes offers "auto" and nothing else, rather than a guess.
func TestUndescribedVariantOffersOnlyAuto(t *testing.T) {
	opts := attentionBackendOptions(variants.Descriptor{}, false)
	if len(opts) != 1 || opts[0].Val != "" {
		t.Errorf("got %+v, want the auto option alone", opts)
	}
}

// Every offered backend comes from the manifest. This is the check that stops
// the list drifting back into Go, where it was wrong twice: once by offering
// FLASHINFER on ROCm, and once by outliving the upstream names it used
// (ROCM_FLASH, TRITON_FLASH_ATTN and XFORMERS were all offered long after they
// stopped existing).
func TestOfferedBackendsComeFromTheManifest(t *testing.T) {
	for _, d := range variants.All() {
		t.Run(d.ID, func(t *testing.T) {
			opts := attentionBackendOptions(d, true)
			if len(opts) == 0 || opts[0].Val != "" {
				t.Fatal("auto must be first and is always offered")
			}
			declared := map[string]bool{}
			for _, b := range d.AttentionBackends {
				declared[b.Value] = true
			}
			for _, o := range opts[1:] {
				if !declared[o.Val] {
					t.Errorf("%q is offered but not declared in the manifest", o.Val)
				}
			}
			if len(opts)-1 != len(d.AttentionBackends) {
				t.Errorf("offered %d backends, manifest declares %d", len(opts)-1, len(d.AttentionBackends))
			}
		})
	}
}

// Rebuilding onto a variant with a different list must not quietly clear a
// backend the operator chose. It is kept, selected, and labelled so the reason
// it looks odd is on screen rather than inferred.
func TestConfiguredBackendSurvivesAVariantThatLacksIt(t *testing.T) {
	d, ok := variants.Get("radiance")
	if !ok {
		t.Fatal("radiance manifest missing")
	}

	opts := backendOptionsFor(d, true, "FLASHINFER")
	last := opts[len(opts)-1]
	if last.Val != "FLASHINFER" {
		t.Fatalf("configured backend was dropped: %+v", opts)
	}
	if !strings.Contains(last.Label, "not one this image offers") {
		t.Errorf("label %q should explain why it is listed", last.Label)
	}

	// A backend the image does declare is not duplicated.
	opts = backendOptionsFor(d, true, "R4D")
	seen := 0
	for _, o := range opts {
		if o.Val == "R4D" {
			seen++
		}
	}
	if seen != 1 {
		t.Errorf("R4D appears %d times, want 1", seen)
	}

	// Nothing configured adds nothing.
	if got, want := len(backendOptionsFor(d, true, "")), len(attentionBackendOptions(d, true)); got != want {
		t.Errorf("empty config changed the list: %d vs %d", got, want)
	}
}

// The names are vLLM's, and they drift between releases. Anything obviously
// not a backend identifier is a typo that would only surface as a failed
// launch on somebody else's hardware.
func TestDeclaredBackendNamesLookLikeIdentifiers(t *testing.T) {
	for _, d := range variants.All() {
		for _, b := range d.AttentionBackends {
			if b.Value == "" {
				t.Errorf("%s: empty backend value", d.ID)
			}
			if b.Value != strings.ToUpper(b.Value) || strings.ContainsAny(b.Value, " \t") {
				t.Errorf("%s: %q is not an upper-case identifier", d.ID, b.Value)
			}
			if b.Label == "" {
				t.Errorf("%s: %s has no label", d.ID, b.Value)
			}
		}
	}
}

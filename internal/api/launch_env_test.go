package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
	"github.com/tmac1973/vllm-toolchest/variants"
)

func envValue(env []string, name string) (string, bool) {
	// Last occurrence wins, exactly as os/exec resolves it.
	val, found := "", false
	for _, kv := range env {
		if n, v, ok := strings.Cut(kv, "="); ok && n == name {
			val, found = v, true
		}
	}
	return val, found
}

// The image's own environment has to reach the server, or a variant whose
// kernel routing lives in its manifest runs with whatever the base happened to
// bake -- which for radiance means AITER paths that crash on gfx1201.
func TestVariantImageEnvIsApplied(t *testing.T) {
	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: "radiance"}}
	env := s.launchEnv("")

	d, _ := variants.Get("radiance")
	if len(d.ImageEnv) == 0 {
		t.Skip("radiance declares no image env")
	}
	for _, pair := range d.ImageEnv {
		name, want, _ := strings.Cut(pair, "=")
		got, ok := envValue(env, name)
		if !ok {
			t.Errorf("%s from the manifest never reached the launch environment", name)
			continue
		}
		if got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
}

// Layering is the contract: image env is the author's default, the runtime
// environment and the knobs are the operator's choices, and the operator wins.
func TestOperatorSettingsOverrideTheImageEnv(t *testing.T) {
	d, _ := variants.Get("radiance")
	if len(d.ImageEnv) == 0 {
		t.Skip("radiance declares no image env")
	}
	name, imageValue, _ := strings.Cut(d.ImageEnv[0], "=")

	cfg := &config.Config{RuntimeEnvExtra: name + "=operator-wins"}
	s := &Server{cfg: cfg, vllmEnv: vllmenv.Env{Variant: "radiance"}}

	got, ok := envValue(s.launchEnv(""), name)
	if !ok {
		t.Fatalf("%s missing entirely", name)
	}
	if got == imageValue {
		t.Errorf("%s = %q: the image default beat the operator's runtime env", name, got)
	}
	if got != "operator-wins" {
		t.Errorf("%s = %q, want the operator's value", name, got)
	}
}

// The process manager depends on spawn to capture worker logs. A manifest must
// not be able to take it over, and the manifest validator rejects one that
// tries -- but check the resulting environment too, since that is what breaks.
func TestProcessManagerRequirementsSurvive(t *testing.T) {
	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: "radiance"}}
	env := s.launchEnv("")
	for name, want := range map[string]string{
		"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
		"PYTHONUNBUFFERED":             "1",
	} {
		if got, _ := envValue(env, name); got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
}

// A variant with no manifest contributes nothing rather than panicking: the
// binary must still serve on an image built before its manifest existed.
func TestUnknownVariantContributesNoImageEnv(t *testing.T) {
	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: "some-future-image"}}
	if got := s.variantImageEnv(); got != nil {
		t.Errorf("got %v, want nothing", got)
	}
	if len(s.launchEnv("")) == 0 {
		t.Error("the process manager's own defaults should still be there")
	}
}

// variantImageEnv returns a fresh slice each call. Returning the manifest's
// own backing array would let one launch's appends corrupt the next.
func TestVariantImageEnvIsCopied(t *testing.T) {
	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: "radiance"}}
	first := s.variantImageEnv()
	if len(first) == 0 {
		t.Skip("radiance declares no image env")
	}
	first[0] = "CLOBBERED=1"

	if second := s.variantImageEnv(); second[0] == "CLOBBERED=1" {
		t.Error("the manifest's slice was handed out directly and got mutated")
	}
}

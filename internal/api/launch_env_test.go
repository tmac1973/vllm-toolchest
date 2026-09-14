package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
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
	env := s.launchEnv(&models.Model{})

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

	got, ok := envValue(s.launchEnv(&models.Model{}), name)
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
	env := s.launchEnv(&models.Model{})
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
	if len(s.launchEnv(&models.Model{})) == 0 {
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

// The case per-model environment exists for: a variable that describes one
// checkpoint must beat the machine-wide value of the same name, and must not
// disturb the names that model says nothing about.
func TestModelEnvOverridesTheMachineWideOne(t *testing.T) {
	s := &Server{
		cfg:     &config.Config{RuntimeEnvExtra: "VLLM_PLE_CPU_OFFLOAD=0\nNCCL_DEBUG=WARN"},
		vllmEnv: vllmenv.Env{Variant: "radiance"},
	}
	m := &models.Model{VLLMConfig: models.VLLMConfig{Env: "VLLM_PLE_CPU_OFFLOAD=1"}}

	env := s.launchEnv(m)
	if got, _ := envValue(env, "VLLM_PLE_CPU_OFFLOAD"); got != "1" {
		t.Errorf("VLLM_PLE_CPU_OFFLOAD = %q, want the model's 1", got)
	}
	if got, _ := envValue(env, "NCCL_DEBUG"); got != "WARN" {
		t.Errorf("NCCL_DEBUG = %q, want the machine-wide WARN", got)
	}
}

// The model sits above the image manifest as well, which is the layer a
// model-specific variable most often has to argue with.
func TestModelEnvOverridesTheImageManifest(t *testing.T) {
	d, _ := variants.Get("radiance")
	if len(d.ImageEnv) == 0 {
		t.Skip("radiance declares no image env")
	}
	name, imageValue, _ := strings.Cut(d.ImageEnv[0], "=")

	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: "radiance"}}
	m := &models.Model{VLLMConfig: models.VLLMConfig{Env: name + "=model-wins"}}

	got, ok := envValue(s.launchEnv(m), name)
	if !ok {
		t.Fatalf("%s missing entirely", name)
	}
	if got == imageValue {
		t.Errorf("%s = %q: the image default beat the model", name, got)
	}
	if got != "model-wins" {
		t.Errorf("%s = %q, want the model's value", name, got)
	}
}

// Comments and malformed lines are dropped rather than breaking the launch,
// the same way the machine-wide block treats them. The panel warns; the launch
// carries on with what parsed.
func TestModelEnvDropsMalformedLines(t *testing.T) {
	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: "radiance"}}
	m := &models.Model{VLLMConfig: models.VLLMConfig{
		Env: "# the drafter needs this\nnot a pair\nCLAV_GDN=1\n",
	}}

	env := s.launchEnv(m)
	if got, _ := envValue(env, "CLAV_GDN"); got != "1" {
		t.Errorf("CLAV_GDN = %q, want 1 despite the junk line above it", got)
	}
	for _, kv := range env {
		if strings.HasPrefix(kv, "not a pair") || strings.HasPrefix(kv, "#") {
			t.Errorf("a malformed line reached the launch environment: %q", kv)
		}
	}
}

// launchEnv now takes the model rather than a quantization method, so check
// the quant-derived part still arrives: AWQ needs the Triton kernel switch.
func TestQuantMethodStillReachesTheEnvironment(t *testing.T) {
	s := &Server{cfg: &config.Config{}, vllmEnv: vllmenv.Env{Variant: "radiance"}}
	m := &models.Model{Quantization: models.QuantMeta{Method: "awq"}}

	if got, ok := envValue(s.launchEnv(m), "VLLM_USE_TRITON_AWQ"); !ok || got != "1" {
		t.Errorf("VLLM_USE_TRITON_AWQ = %q (present=%v), want 1 for an AWQ model", got, ok)
	}
}

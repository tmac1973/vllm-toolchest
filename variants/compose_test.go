package variants_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// Each vendor's compose file lists the knob variables of every variant of that
// vendor as bare `- NAME` entries, so a value set in .env reaches the
// container and an unset one stays genuinely unset.
//
// That list is hand-maintained and there is nothing to stop it drifting from
// the manifests. When it does, the failure is silent in the worst way: the
// Settings page offers a control, the operator sets it, the config stores it,
// and the variable never crosses into the container. Nothing errors.
//
// So check the two against each other.
func TestComposePassthroughMatchesManifests(t *testing.T) {
	byVendor := map[string][]variants.Descriptor{}
	for _, d := range variants.All() {
		if d.Vendor == "" {
			// `generic` names no vendor on purpose: it is two variants
			// wearing one name until it splits. It declares no knobs
			// either, so there is nothing to check.
			if len(d.Knobs) > 0 {
				t.Errorf("%s declares knobs but no VARIANT_VENDOR, so no compose file can carry them", d.ID)
			}
			continue
		}
		byVendor[d.Vendor] = append(byVendor[d.Vendor], d)
	}

	for vendor, ds := range byVendor {
		t.Run(vendor, func(t *testing.T) {
			path := filepath.Join(repoRoot(t), "docker-compose."+vendor+".yml")
			declared := composeEnvNames(t, path)

			// Names the compose file passes through that are not knobs --
			// HIP_VISIBLE_DEVICES and friends -- are legitimate, so this is
			// a one-way check: every knob must be listed, not every listed
			// name must be a knob.
			for _, d := range ds {
				for _, name := range d.EnvNames() {
					if !declared[name] {
						t.Errorf("%s declares knob variable %s, but %s does not pass it through; "+
							"the control would appear in Settings and do nothing",
							d.ID, name, filepath.Base(path))
					}
				}
			}
		})
	}
}

// composeEnvNames reads the service's `environment:` list. Only the bare
// `- NAME` form counts: `NAME: "${NAME:-}"` would define the variable as an
// empty string inside the container, which these images read as "off".
func composeEnvNames(t *testing.T, path string) map[string]bool {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading compose file: %v", err)
	}

	var doc struct {
		Services map[string]struct {
			Environment []string `yaml:"environment"`
		} `yaml:"services"`
	}
	if err := yaml.Unmarshal(data, &doc); err != nil {
		// A mapping-form environment: block fails to unmarshal into
		// []string, which is itself the thing this test exists to catch.
		t.Fatalf("parsing %s: %v (is the environment: block in list form?)", path, err)
	}

	out := map[string]bool{}
	for _, svc := range doc.Services {
		for _, entry := range svc.Environment {
			if name, _, hasValue := strings.Cut(entry, "="); !hasValue {
				out[entry] = true
			} else {
				t.Errorf("%s: %q assigns a value; a bare `- NAME` is required so an "+
					"unset variable stays unset rather than becoming empty", filepath.Base(path), name)
			}
		}
	}
	return out
}

// Every variant must name a compose file that exists, and a Dockerfile that
// exists. Both are strings in a config file with nothing to catch a typo.
func TestVariantBuildInputsExist(t *testing.T) {
	root := repoRoot(t)
	for _, d := range variants.All() {
		t.Run(d.ID, func(t *testing.T) {
			if d.Vendor != "" {
				compose := filepath.Join(root, "docker-compose."+d.Vendor+".yml")
				if _, err := os.Stat(compose); err != nil {
					t.Errorf("VARIANT_VENDOR=%s but %s does not exist", d.Vendor, filepath.Base(compose))
				}
			}
			if d.Dockerfile != "" {
				if _, err := os.Stat(filepath.Join(root, d.Dockerfile)); err != nil {
					t.Errorf("VARIANT_DOCKERFILE=%s does not exist", d.Dockerfile)
				}
			}
			// A variant that layers onto a published image needs both a base
			// and somewhere for vllmctl to find vLLM inside it; one without a
			// base builds from source and needs neither.
			if d.BaseImage != "" && d.VenvRoot == "" {
				t.Error("declares a base image but no VARIANT_VENV_ROOT, so the build assertion cannot check it")
			}
		})
	}
}

// The manifests are the only description of these images we have, so the
// fields the build actually consumes must be present and plausible.
func TestPrebuiltVariantsArePinned(t *testing.T) {
	for _, d := range variants.All() {
		if d.BaseImage == "" {
			continue
		}
		t.Run(d.ID, func(t *testing.T) {
			// Fully qualified: podman resolves an unqualified name against
			// its unqualified-search-registries, which on Fedora/RHEL tries
			// registry.fedoraproject.org first and fails with "manifest
			// unknown" before it ever reaches Docker Hub.
			if !strings.Contains(d.BaseImage, "/") || !strings.Contains(strings.SplitN(d.BaseImage, "/", 2)[0], ".") {
				t.Errorf("base image %q is not fully qualified; podman resolves bare names against "+
					"its own search registries and fails before reaching Docker Hub", d.BaseImage)
			}
			// The pin is surfaced in the UI as the model-support ceiling and
			// is what the tuner script is fetched against.
			if d.VLLMPin == "" {
				t.Error("a prebuilt base pins vLLM; VARIANT_VLLM_PIN says which release, and the build asserts it")
			}
		})
	}
}

// The vLLM pin doubles as the git ref the tuner benchmark script is fetched
// from, unless the manifest names one separately. A version like
// "0.23.1.dev1+g9ddef7117" has no tag behind it, so a pin in that shape would
// 404 the fetch -- which is how this was found, halfway through a real build.
func TestPinIsAFetchableRefOrTunerRefIsSet(t *testing.T) {
	for _, d := range variants.All() {
		if d.BaseImage == "" || d.VLLMPin == "" || d.VLLMPin == "main" {
			continue
		}
		t.Run(d.ID, func(t *testing.T) {
			if d.TunerRef != "" {
				return // an explicit ref settles it
			}
			// Release tags look like v1.2.3. A "+" local-version segment or a
			// ".dev" counter means this came from `pip show`, not from a tag.
			for _, marker := range []string{"+", ".dev", ".post", ".rc"} {
				if strings.Contains(d.VLLMPin, marker) {
					t.Errorf("VARIANT_VLLM_PIN=%q contains %q, so it is a reported version rather than "+
						"a git ref and the tuner fetch would 404. Set VARIANT_VLLM_PIN to the nearest "+
						"release and VARIANT_TUNER_REF to the commit (the part after +g).",
						d.VLLMPin, marker)
				}
			}
		})
	}
}

// A tuner ref is a git ref: a tag, a branch or a commit sha. Never a version.
func TestTunerRefLooksLikeAGitRef(t *testing.T) {
	for _, d := range variants.All() {
		if d.TunerRef == "" {
			continue
		}
		if strings.ContainsAny(d.TunerRef, "+ ") {
			t.Errorf("%s: VARIANT_TUNER_REF=%q is not a git ref", d.ID, d.TunerRef)
		}
	}
}

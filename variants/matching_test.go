package variants_test

import (
	"strings"
	"testing"
)

// setup.sh decides which variants fit a machine, which one to recommend, and
// what an existing install's variant name migrates to. All three run on
// hardware nobody here can test, so drive them against synthetic detection
// state rather than whatever GPU happens to be in the build machine.
//
// runMatch sources setup.sh, sets the globals detect_gpu would have set, and
// evaluates snippet.
func runMatch(t *testing.T, state, snippet string) string {
	t.Helper()
	return runSetupSh(t, state+"\n"+snippet)
}

func TestVariantMatching(t *testing.T) {
	for _, tc := range []struct {
		name    string
		state   string
		want    []string
		notWant []string
	}{
		{
			name:  "RDNA4 gets the tuned image and the portable one",
			state: `GPU_VENDOR=rocm; AMD_GFX_TARGET=gfx1201`,
			want:  []string{"radiance", "rocm-source"},
		},
		{
			// The one host check that already existed and must keep working:
			// the radiance image is compiled for gfx1201 and nothing else.
			name:    "RDNA3 is not offered the gfx1201 image",
			state:   `GPU_VENDOR=rocm; AMD_GFX_TARGET=gfx1100`,
			want:    []string{"rocm-source"},
			notWant: []string{"radiance"},
		},
		{
			name:    "NVIDIA gets no AMD variant",
			state:   `GPU_VENDOR=cuda; AMD_GFX_TARGET=""`,
			want:    []string{"cuda-source"},
			notWant: []string{"radiance", "rocm-source"},
		},
		{
			name:    "an undetected GPU matches nothing",
			state:   `GPU_VENDOR=""; AMD_GFX_TARGET=""`,
			notWant: []string{"radiance", "rocm-source", "cuda-source"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := splitLines(runMatch(t, tc.state, "matching_variants"))
			for _, w := range tc.want {
				if !contains(got, w) {
					t.Errorf("%s should be offered here; got %v", w, got)
				}
			}
			for _, n := range tc.notWant {
				if contains(got, n) {
					t.Errorf("%s must not be offered here; got %v", n, got)
				}
			}
		})
	}
}

// The recommendation is the most specific match: a variant built for the exact
// card beats one that runs on anything of that vendor. Getting this backwards
// would hand every RDNA4 owner the slow generic build by default.
func TestRecommendationPrefersTheSpecificVariant(t *testing.T) {
	got := strings.TrimSpace(runMatch(t,
		`GPU_VENDOR=rocm; AMD_GFX_TARGET=gfx1201`, "recommended_variant"))
	if got != "radiance" {
		t.Errorf("recommended = %q, want radiance (it names gfx1201; rocm-source names nothing)", got)
	}

	// On RDNA3 the tuned gfx1201 images do not apply, but AMD's own prebuilt
	// does -- and it names ten architectures where the source build names
	// none, so it wins on specificity. That is the right default: an hours-long
	// compile should be something someone opts into, not what they get for
	// pressing Enter.
	got = strings.TrimSpace(runMatch(t,
		`GPU_VENDOR=rocm; AMD_GFX_TARGET=gfx1100`, "recommended_variant"))
	if got != "rocm" {
		t.Errorf("recommended = %q, want rocm (it names gfx1100; rocm-source names nothing)", got)
	}

	// The source build is still offered, just not preselected.
	all := splitLines(runMatch(t, `GPU_VENDOR=rocm; AMD_GFX_TARGET=gfx1100`, "matching_variants"))
	if !contains(all, "rocm-source") {
		t.Errorf("rocm-source should still be on the menu; got %v", all)
	}
}

// "generic" was two variants wearing one name. An install predating the
// manifests has it in .env, and every later command reads that name back, so
// it has to resolve to whichever of the two that install has been running.
func TestGenericMigratesByVendor(t *testing.T) {
	for _, tc := range []struct{ vendor, want string }{
		{"rocm", "rocm-source"},
		{"cuda", "cuda-source"},
	} {
		got := strings.TrimSpace(runMatch(t,
			"GPU_VENDOR="+tc.vendor, `migrate_variant_name generic`))
		if got != tc.want {
			t.Errorf("generic on %s migrated to %q, want %q", tc.vendor, got, tc.want)
		}
	}

	// An undetectable vendor must not guess: building the wrong image wastes
	// an hour and fails at the end. Leaving the old name makes the error name it.
	if got := strings.TrimSpace(runMatch(t, `GPU_VENDOR=""`, `migrate_variant_name generic`)); got != "generic" {
		t.Errorf("with no vendor detected, migration returned %q; it should not guess", got)
	}

	// Every other name passes through untouched.
	if got := strings.TrimSpace(runMatch(t, `GPU_VENDOR=rocm`, `migrate_variant_name radiance`)); got != "radiance" {
		t.Errorf("radiance migrated to %q, want it untouched", got)
	}
}

// The compose file and Dockerfile a variant resolves to are what the build
// actually uses, and they come from the manifest rather than from a rule
// duplicated in bash.
func TestDispatchResolvesFromTheManifest(t *testing.T) {
	for _, tc := range []struct{ variant, vendor, compose, dockerfile string }{
		{"radiance", "rocm", "docker-compose.amd.yml", "Dockerfile.prebuilt"},
		{"rocm-source", "rocm", "docker-compose.amd.yml", "Dockerfile.rocm"},
		{"cuda-source", "cuda", "docker-compose.nvidia.yml", "Dockerfile.cuda"},
	} {
		t.Run(tc.variant, func(t *testing.T) {
			out := runMatch(t,
				"GPU_VENDOR="+tc.vendor+"; BUILD_VARIANT="+tc.variant,
				`printf '%s\n%s\n' "$(compose_file)" "$(dockerfile)"`)
			lines := splitLines(out)
			if len(lines) != 2 {
				t.Fatalf("got %v", lines)
			}
			if lines[0] != tc.compose {
				t.Errorf("compose = %q, want %q", lines[0], tc.compose)
			}
			if lines[1] != tc.dockerfile {
				t.Errorf("dockerfile = %q, want %q", lines[1], tc.dockerfile)
			}
		})
	}
}

// A host check that fails with severity "block" must stop the install, and the
// escape hatch must lift it -- that hatch is what makes a wrong check on
// untestable hardware a manifest edit rather than a release.
func TestHostRequirementBlocksAndCanBeSkipped(t *testing.T) {
	state := `GPU_VENDOR=rocm; AMD_GFX_TARGET=gfx1100; BUILD_VARIANT=radiance; GPU_INFO="a card"`

	out := runSetupShAllowFail(t, state+"\ncheck_host_requirements && echo PASSED")
	if strings.Contains(out, "PASSED") {
		t.Error("the gfx1201-only check passed on gfx1100")
	}
	if !strings.Contains(out, "gfx1201") {
		t.Errorf("the failure should name what is required; got %q", out)
	}

	out = runSetupSh(t, `VLLMCTL_SKIP_HOSTCHECK=1
`+state+"\ncheck_host_requirements && echo PASSED")
	if !strings.Contains(out, "PASSED") {
		t.Errorf("VLLMCTL_SKIP_HOSTCHECK=1 did not lift the check; got %q", out)
	}
}

func contains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}

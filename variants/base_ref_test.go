package variants

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The base image a build runs on is decided by setup.sh, and getting it wrong
// is close to undetectable: the image still boots, still reports the same vLLM
// version whenever two tags share a build, and only fails later on a flag the
// older tag has never heard of. So pin the precedence here.
//
// These source setup.sh for its functions, the way bash_parity_test.go does.
func runBaseRef(t *testing.T, envFile string, environ []string, variant string) string {
	t.Helper()
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}

	// SCRIPT_DIR is readonly and derived from setup.sh's own location, so it
	// cannot be preset or reassigned. Copy the script and the manifests into a
	// temp directory and source that copy instead: SCRIPT_DIR then resolves
	// there on its own, and the .env this case is about sits beside it.
	dir := t.TempDir()
	script, err := os.ReadFile(filepath.Join("..", "setup.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "setup.sh"), script, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "variants"), 0o755); err != nil {
		t.Fatal(err)
	}
	confs, err := filepath.Glob("*.conf")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range confs {
		body, err := os.ReadFile(c)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "variants", filepath.Base(c)), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if envFile != "" {
		if err := os.WriteFile(filepath.Join(dir, ".env"), []byte(envFile), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	snippet := "BUILD_VARIANT=" + variant + "\nvariant_base_ref\n"
	cmd := exec.Command("bash", "-c", ". ./setup.sh\n"+snippet)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), environ...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("bash failed: %v\n%s", err, out)
	}
	return strings.TrimSpace(string(out))
}

// The manifest is the source of truth. This is the case that broke: .env still
// held the image the first install wrote there, so a manifest bump resolved to
// the old tag, and the build ran on it while .env was rewritten to the new one.
func TestManifestBeatsAStaleEnvFile(t *testing.T) {
	d, ok := Get("rdna4-clav")
	if !ok {
		t.Skip("rdna4-clav manifest missing")
	}

	got := runBaseRef(t,
		"VLLMCTL_BASE_IMAGE=docker.io/tcclaviger/vllm:28.02.2\nVLLMCTL_PORT=3000\n",
		nil, "rdna4-clav")

	if got != d.BaseImage {
		t.Errorf("base image = %q, want the manifest's %q", got, d.BaseImage)
	}
}

// The environment variable is the deliberate one-off override and still wins,
// because that is how an operator tries an alternate build without editing a
// tracked file.
func TestEnvironmentOverrideStillWins(t *testing.T) {
	const pinned = "docker.io/someone/their-vllm:testing"

	got := runBaseRef(t, "", []string{"VLLMCTL_BASE_IMAGE=" + pinned}, "rdna4-clav")
	if got != pinned {
		t.Errorf("base image = %q, want the environment's %q", got, pinned)
	}
}

// With no .env and no override, the manifest answers — the ordinary install.
func TestManifestAnswersWithNothingElseSet(t *testing.T) {
	d, ok := Get("radiance")
	if !ok {
		t.Skip("radiance manifest missing")
	}

	if got := runBaseRef(t, "", nil, "radiance"); got != d.BaseImage {
		t.Errorf("base image = %q, want %q", got, d.BaseImage)
	}
}

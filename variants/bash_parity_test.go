package variants_test

import (
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// A manifest is read twice by two different programs: setup.sh sources it as
// bash, and the Go binary parses it with a hand-written reader. Nothing forces
// those to agree, and a disagreement is close to undetectable in production --
// the install would build one thing and the UI would describe another.
//
// So exercise setup.sh's own reader functions against every manifest and
// compare the answers to the Go parser's. This is the test that lets a
// contributor add variant number eleven by writing one file.

func repoRoot(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate this test file")
	}
	return filepath.Dir(filepath.Dir(self))
}

// runSetupSh sources setup.sh for its functions -- the guard at the bottom of
// the script keeps main from running -- and evaluates snippet.
func runSetupSh(t *testing.T, snippet string) string {
	t.Helper()
	root := repoRoot(t)
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}
	cmd := exec.Command("bash", "-c", ". ./setup.sh\n"+snippet)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("bash failed: %v\n%s", err, out)
	}
	return string(out)
}

// runSetupShAllowFail is runSetupSh for snippets expected to exit non-zero --
// a host check that blocks, for instance.
func runSetupShAllowFail(t *testing.T, snippet string) string {
	t.Helper()
	root := repoRoot(t)
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}
	cmd := exec.Command("bash", "-c", ". ./setup.sh\n"+snippet)
	cmd.Dir = root
	out, _ := cmd.CombinedOutput()
	return string(out)
}

func TestBashAndGoReadTheSameManifests(t *testing.T) {
	for _, d := range variants.All() {
		t.Run(d.ID, func(t *testing.T) {
			out := runSetupSh(t,
				"load_variant_manifest "+d.ID+"\n"+
					"printf '%s\\n' \"$VARIANT_ID\"\n"+
					"knob_env_names\n")

			lines := splitLines(out)
			if len(lines) == 0 {
				t.Fatal("bash produced nothing; is the manifest sourceable?")
			}
			if lines[0] != d.ID {
				t.Errorf("VARIANT_ID: bash says %q, Go says %q", lines[0], d.ID)
			}

			gotEnv := lines[1:]
			wantEnv := d.EnvNames()
			if len(gotEnv) == 0 && len(wantEnv) == 0 {
				return
			}
			if !reflect.DeepEqual(gotEnv, wantEnv) {
				t.Errorf("knob env names disagree\n bash: %v\n   go: %v", gotEnv, wantEnv)
			}
		})
	}
}

// The tooltips are the values most likely to break a naive bash reader: they
// carry commas, double quotes and em-dashes. Compare them byte for byte.
func TestBashAndGoReadTheSameHelpText(t *testing.T) {
	d, ok := variants.Get("radiance")
	if !ok {
		t.Fatal("radiance manifest missing")
	}
	for _, k := range d.Knobs {
		slug := strings.ToUpper(k.ID)
		out := runSetupSh(t,
			"load_variant_manifest radiance\n"+
				"printf '%s' \"$(knob_attr "+slug+" HELP)\"\n")
		if out != k.Help {
			t.Errorf("knob %s help text disagrees\n bash: %q\n   go: %q", k.ID, out, k.Help)
		}
	}
}

// "-" is the sentinel for "the recommendation is the image default", so it
// must produce no environment line at all -- in either reader.
func TestBashRecommendedEnvMatchesGo(t *testing.T) {
	for _, d := range variants.All() {
		t.Run(d.ID, func(t *testing.T) {
			out := runSetupSh(t,
				"load_variant_manifest "+d.ID+"\nknob_recommended_env\n")

			got := map[string]string{}
			for _, line := range splitLines(out) {
				k, v, _ := strings.Cut(line, "=")
				got[k] = v
			}

			want := map[string]string{}
			for id, v := range d.Recommended() {
				k, _ := d.Knob(id)
				want[k.Env] = v
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("recommended env disagrees\n bash: %v\n   go: %v", got, want)
			}
		})
	}
}

func TestBashListsEveryManifest(t *testing.T) {
	got := splitLines(runSetupSh(t, "list_variants\n"))
	var want []string
	for _, d := range variants.All() {
		want = append(want, d.ID)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("variant list disagrees\n bash: %v\n   go: %v", got, want)
	}
}

func splitLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if l = strings.TrimRight(l, "\r"); l != "" {
			out = append(out, l)
		}
	}
	return out
}

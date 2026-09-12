package variants_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// .env.example documents the feature switches, and it used to do so by hand --
// one of the six places the same twelve names were written out. The block
// between the markers is generated from the manifests now, and this is what
// stops it drifting back: add a knob, forget `make env-example`, and the file
// quietly describes an older set of switches than the software has.
func TestEnvExampleIsUpToDate(t *testing.T) {
	root := repoRoot(t)
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}

	original, err := os.ReadFile(filepath.Join(root, ".env.example"))
	if err != nil {
		t.Fatal(err)
	}

	// Regenerate into a copy so the test never writes to the repo.
	tmp := filepath.Join(t.TempDir(), ".env.example")
	if err := os.WriteFile(tmp, original, 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("./setup.sh", "--write-env-example", tmp)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("regenerating: %v\n%s", err, out)
	}

	regenerated, err := os.ReadFile(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if string(regenerated) != string(original) {
		t.Errorf(".env.example is out of date with variants/*.conf — run `make env-example`")
	}
}

// The generated block has to actually document every switch. A generator that
// silently emitted nothing would pass a byte comparison against a file
// generated the same way.
func TestEnvExampleDocumentsEveryKnob(t *testing.T) {
	body, err := os.ReadFile(filepath.Join(repoRoot(t), ".env.example"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(body)

	begin := strings.Index(s, "# >>> BEGIN GENERATED KNOBS")
	end := strings.Index(s, "# <<< END GENERATED KNOBS")
	if begin < 0 || end < 0 || end < begin {
		t.Fatal("the generated-knob markers are missing or out of order")
	}
	block := s[begin:end]

	total := 0
	for _, d := range variants.All() {
		for _, k := range d.Knobs {
			total++
			if !strings.Contains(block, "#"+k.Env+"=") {
				t.Errorf("%s declares %s but .env.example does not document it", d.ID, k.Env)
			}
			// The help text is what makes the line worth reading; a bare
			// KEY= tells nobody what the switch does.
			if k.Help != "" && !strings.Contains(block, strings.Fields(k.Help)[0]) {
				t.Errorf("%s: help text for %s is missing from the block", d.ID, k.ID)
			}
		}
	}
	if total == 0 {
		t.Fatal("no variant declares knobs, so this test proves nothing")
	}

	// An example line must never set a switch to prose. A placeholder is grey
	// hint text in the UI ("off — try: auto") and only sometimes a real value.
	for _, line := range strings.Split(block, "\n") {
		if !strings.HasPrefix(line, "#") || !strings.Contains(line, "=") || strings.HasPrefix(line, "# ") {
			continue
		}
		_, val, _ := strings.Cut(line, "=")
		if strings.Contains(val, " ") {
			t.Errorf("example line sets a switch to prose: %q", line)
		}
	}
}

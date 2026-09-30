package variants

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// lifecycleBed runs setup.sh's container functions against a container
// runtime, a compose and a systemctl that are all one script: it keeps a
// single fact -- whether a container named vllm-toolchest exists -- and logs
// what it was asked.
type lifecycleBed struct {
	t   *testing.T
	dir string
}

const fakeLifecycle = `#!/usr/bin/env bash
echo "$*" >> "$BED/calls.log"
case "$1 $2" in
  "container inspect") [ -f "$BED/container" ] ;;
  "stop "*)            : ;;
  "rm "*)              rm -f "$BED/container" ;;
  "compose up")        touch "$BED/container" ;;
  "compose down")      rm -f "$BED/container" ;;
  "compose build")     : ;;
esac
`

func newLifecycleBed(t *testing.T) *lifecycleBed {
	t.Helper()
	b := &lifecycleBed{t: t, dir: t.TempDir()}
	if err := os.WriteFile(filepath.Join(b.dir, "fake"), []byte(fakeLifecycle), 0o755); err != nil {
		t.Fatal(err)
	}
	return b
}

// run evaluates a snippet with the fakes in place. quadlet says whether
// auto-start is enabled; unitOwns says whether the container that exists was
// started by the unit, in which case stopping the unit removes it -- as
// Quadlet does -- and otherwise does nothing at all.
func (b *lifecycleBed) run(quadlet, unitOwns bool, snippet string) {
	b.t.Helper()
	prelude := `
CONTAINER_CMD="$BED/fake"
compose_cmd() { echo "$BED/fake compose"; }
has_quadlet() { ` + boolExit(quadlet) + `; }
systemctl_cmd() {
    echo "systemctl $*" >> "$BED/calls.log"
    case "$1" in
      stop)  ` + map[bool]string{true: `rm -f "$BED/container"`, false: `:`}[unitOwns] + ` ;;
      start) touch "$BED/container" ;;
    esac
}
ensure_base_image() { :; }
write_env_file() { :; }
refresh_quadlet() { :; }
`
	b.t.Setenv("BED", b.dir)
	sourceSetupSh(b.t, prelude+snippet)
}

func boolExit(v bool) string {
	if v {
		return "return 0"
	}
	return "return 1"
}

func (b *lifecycleBed) containerExists() bool {
	_, err := os.Stat(filepath.Join(b.dir, "container"))
	return err == nil
}

func (b *lifecycleBed) startContainer() {
	if err := os.WriteFile(filepath.Join(b.dir, "container"), nil, 0o644); err != nil {
		b.t.Fatal(err)
	}
}

func (b *lifecycleBed) calls() []string {
	data, _ := os.ReadFile(filepath.Join(b.dir, "calls.log"))
	var out []string
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		// Existence checks are how the functions look, not what they do.
		if line != "" && !strings.HasPrefix(line, "container inspect") {
			out = append(out, line)
		}
	}
	return out
}

func has(calls []string, prefix string) bool {
	for _, c := range calls {
		if strings.HasPrefix(c, prefix) {
			return true
		}
	}
	return false
}

// What happened on compute. Auto-start was on, the running container had been
// started by compose, and `quick` stopped the unit -- which was not running,
// so that did nothing and said nothing. The build succeeded, compose found a
// container of that name already up and left it alone, and setup.sh reported
// that vllm-toolchest was running. It was: the previous build.
func TestDownRemovesAContainerSystemdDidNotStart(t *testing.T) {
	b := newLifecycleBed(t)
	b.startContainer()

	b.run(true, false, "container_down\n")

	if b.containerExists() {
		t.Errorf("the container survived container_down; calls: %v", b.calls())
	}
}

// The ordinary case is not disturbed: a container the unit owns is gone once
// the unit stops, and nothing else is touched.
func TestDownLeavesAUnitOwnedStopToSystemd(t *testing.T) {
	b := newLifecycleBed(t)
	b.startContainer()

	b.run(true, true, "container_down\n")

	calls := b.calls()
	if len(calls) != 1 || calls[0] != "systemctl stop vllm-toolchest.service" {
		t.Errorf("calls = %v, want only the unit being stopped", calls)
	}
}

// After a quick rebuild with auto-start on, the container has to be the
// unit's again. Left as a compose container it is not restarted on failure,
// is not what up/down/logs act on, and was what the next quick failed to
// replace.
func TestQuickRebuildHandsTheContainerBackToSystemd(t *testing.T) {
	b := newLifecycleBed(t)
	b.startContainer()

	b.run(true, false, "container_quick_rebuild\n")

	calls := b.calls()
	if !has(calls, "compose build") {
		t.Errorf("the image was not built; calls: %v", calls)
	}
	if has(calls, "compose up") {
		t.Errorf("the container was started through compose with auto-start on; calls: %v", calls)
	}
	if last := calls[len(calls)-1]; last != "systemctl start vllm-toolchest.service" {
		t.Errorf("last call = %q, want the unit being started; calls: %v", last, calls)
	}
	if !b.containerExists() {
		t.Error("nothing is running after the rebuild")
	}
}

// Twice in a row, which is the sequence that failed: the second rebuild has
// to stop what the first one started.
func TestTwoQuickRebuildsInARowBothReplaceTheContainer(t *testing.T) {
	b := newLifecycleBed(t)
	b.startContainer()

	// After the first, the container is the unit's, so stopping the unit
	// removes it.
	b.run(true, false, "container_quick_rebuild\n")
	b.run(true, true, "container_quick_rebuild\n")

	var starts int
	for _, c := range b.calls() {
		if c == "systemctl start vllm-toolchest.service" {
			starts++
		}
	}
	if starts != 2 {
		t.Errorf("the unit was started %d times over two rebuilds, want 2; calls: %v", starts, b.calls())
	}
}

// Without auto-start there is no unit, and compose both stops and starts.
func TestQuickRebuildWithoutAutoStartUsesCompose(t *testing.T) {
	b := newLifecycleBed(t)
	b.startContainer()

	b.run(false, false, "container_quick_rebuild\n")

	calls := b.calls()
	if !has(calls, "compose down") || !has(calls, "compose up -d --build") {
		t.Errorf("calls = %v, want compose down then up", calls)
	}
	if has(calls, "systemctl") {
		t.Errorf("systemd was used with auto-start off; calls: %v", calls)
	}
}

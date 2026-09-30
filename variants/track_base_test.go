package variants

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A variant that follows a moving tag is resolved by setup.sh against a
// container runtime and a registry, neither of which a test can have. So the
// runtime here is a script: an image store made of files, a "registry" the
// test can move a tag in, and a log of what was asked of it.
//
// What it stands in for was checked once against the real thing -- podman and
// tcclaviger/vllm:latest -- and these tests pin the decisions that run made.
const fakeRuntime = `#!/usr/bin/env bash
# Image names become file names; the content is the image id.
enc() { printf '%s' "$1" | sed 's|/|%|g'; }
store="$FAKE_STATE/images"; registry="$FAKE_STATE/registry"
mkdir -p "$store"
echo "$*" >> "$FAKE_STATE/calls.log"

named() { for a in "$@"; do [ -f "$store/$(enc "$a")" ] && { echo "$a"; return 0; }; done; return 1; }

case "$1" in
  pull)
    [ -f "$registry/$(enc "$2")" ] || { echo "manifest unknown" >&2; exit 1; }
    cp "$registry/$(enc "$2")" "$store/$(enc "$2")" ;;
  tag)
    src="$2"; shift 2
    for dst in "$@"; do cp "$store/$(enc "$src")" "$store/$(enc "$dst")" || exit 1; done ;;
  rmi)
    shift; for a in "$@"; do [ "$a" = "-f" ] || rm -f "$store/$(enc "$a")"; done ;;
  images)
    for f in "$store"/*; do [ -e "$f" ] && basename "$f" | sed 's|%|/|g'; done ;;
  build)
    exit 0 ;;
  image)
    img="$3"; [ -f "$store/$(enc "$img")" ] || exit 1
    id="$(cat "$store/$(enc "$img")")"
    case "$*" in
      *RepoDigests*)
        # An image is known by registry digest only under a registry name.
        for f in "$store"/*; do
          n="$(basename "$f" | sed 's|%|/|g')"
          if [ "$(cat "$f")" = "$id" ]; then echo "${n%:*}@sha256:$id"; fi
        done ;;
      *.Digest*) echo "sha256:$id" ;;
      *.Id*)     echo "sha256:$id" ;;
    esac ;;
  run)
    img="$(named "$@")" || exit 1
    id="$(cat "$store/$(enc "$img")")"
    echo "vllm=$(cat "$FAKE_STATE/vllm/$id" 2>/dev/null)"
    echo "release=$(cat "$FAKE_STATE/release/$id" 2>/dev/null)" ;;
esac
`

const (
	clavTag = "docker.io/tcclaviger/vllm:latest"
	// Image ids, which double as registry digests in the fake.
	oldBase = "aaaaaaaaaaaa1111111111111111111111111111111111111111111111111111"
	newBase = "bbbbbbbbbbbb2222222222222222222222222222222222222222222222222222"

	oldAlias = "localhost/vllmctl-rdna4-clav-base:aaaaaaaaaaaa"
	newAlias = "localhost/vllmctl-rdna4-clav-base:bbbbbbbbbbbb"
)

// trackBed is a copy of setup.sh and the manifests beside a fake runtime.
type trackBed struct {
	t     *testing.T
	dir   string
	state string
}

func newTrackBed(t *testing.T) *trackBed {
	t.Helper()
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}
	b := &trackBed{t: t, dir: t.TempDir()}
	b.state = filepath.Join(b.dir, "state")

	// SCRIPT_DIR is derived from setup.sh's own location, so the copy is what
	// makes the .env these cases are about sit beside it. See runBaseRef.
	script, err := os.ReadFile(filepath.Join("..", "setup.sh"))
	if err != nil {
		t.Fatal(err)
	}
	b.write("setup.sh", string(script), 0o755)
	confs, _ := filepath.Glob("*.conf")
	for _, c := range confs {
		body, err := os.ReadFile(c)
		if err != nil {
			t.Fatal(err)
		}
		b.write(filepath.Join("variants", c), string(body), 0o644)
	}
	b.write("fakectl", fakeRuntime, 0o755)

	b.publish(oldBase, "0.27.0.dev0+g55c98e370a", "28.04.9")
	b.publish(newBase, "0.29.0.dev0+g2bdbbc8080", "29.06.1")
	return b
}

func (b *trackBed) write(rel, body string, mode os.FileMode) {
	b.t.Helper()
	path := filepath.Join(b.dir, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		b.t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), mode); err != nil {
		b.t.Fatal(err)
	}
}

// publish describes an image the fake can serve: what vLLM is inside it and
// what its author calls the release.
func (b *trackBed) publish(id, vllm, release string) {
	b.write(filepath.Join("state", "vllm", id), vllm, 0o644)
	b.write(filepath.Join("state", "release", id), release, 0o644)
}

// point moves a registry tag, the way a maintainer's push does.
func (b *trackBed) point(tag, id string) {
	b.write(filepath.Join("state", "registry", strings.ReplaceAll(tag, "/", "%")), id, 0o644)
}

// run sources setup.sh, points it at the fake runtime and runs a snippet for
// the rdna4-clav variant. The tuner-ref reachability probe is stubbed: it is
// a network call, and what it decides is tested on its own.
func (b *trackBed) run(environ []string, snippet string) string {
	b.t.Helper()
	prelude := ". ./setup.sh\n" +
		"CONTAINER_CMD=./fakectl\nBUILD_VARIANT=rdna4-clav\nGPU_VENDOR=rocm\n" +
		"tuner_ref_fetchable() { [[ \"$1\" != " + unreachableCommit + " ]]; }\n"
	cmd := exec.Command("bash", "-c", prelude+snippet)
	cmd.Dir = b.dir
	cmd.Env = append(os.Environ(), "FAKE_STATE="+b.state)
	cmd.Env = append(cmd.Env, environ...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		b.t.Fatalf("bash failed: %v\n%s", err, out)
	}
	return string(out)
}

const unreachableCommit = "0123456789ab"

// sourceSetupSh runs a snippet against the real setup.sh, for the functions
// that need neither a runtime nor an .env.
func sourceSetupSh(t *testing.T, snippet string) string {
	t.Helper()
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}
	cmd := exec.Command("bash", "-c", ". ./setup.sh\n"+snippet)
	cmd.Dir = ".."
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("bash failed: %v\n%s", err, out)
	}
	return string(out)
}

// build does what install, rebuild, pull and quick all do about the base
// image: resolve it, then record it.
func (b *trackBed) build(mode string, environ ...string) string {
	return b.run(environ, "ensure_base_image "+mode+"\nwrite_env_file\n")
}

func (b *trackBed) env() map[string]string {
	b.t.Helper()
	data, err := os.ReadFile(filepath.Join(b.dir, ".env"))
	if err != nil {
		b.t.Fatal(err)
	}
	out := map[string]string{}
	for _, line := range strings.Split(string(data), "\n") {
		if k, v, ok := strings.Cut(line, "="); ok {
			out[k] = v
		}
	}
	return out
}

func (b *trackBed) calls() string {
	data, _ := os.ReadFile(filepath.Join(b.state, "calls.log"))
	return string(data)
}

func (b *trackBed) images() string {
	return b.run(nil, "./fakectl images\n")
}

func wantEnv(t *testing.T, env map[string]string, want map[string]string) {
	t.Helper()
	for k, v := range want {
		if env[k] != v {
			t.Errorf("%s = %q, want %q", k, env[k], v)
		}
	}
}

// The three things a bare ":latest" would get wrong, in one pass: the image
// is pulled, the pin and the tuner ref come out of it, and what it resolved to
// is on record under a name that does not move.
func TestTrackedBaseIsResolvedAndRecorded(t *testing.T) {
	b := newTrackBed(t)
	b.point(clavTag, newBase)

	b.build("refresh")

	wantEnv(t, b.env(), map[string]string{
		"VLLMCTL_BASE_IMAGE":   newAlias,
		"VLLMCTL_BASE_DIGEST":  "docker.io/tcclaviger/vllm@sha256:" + newBase,
		"VLLMCTL_VLLM_PIN":     "v0.29.0",
		"VLLMCTL_TUNER_REF":    "2bdbbc8080",
		"VLLMCTL_BASE_VLLM":    "0.29.0.dev0+g2bdbbc8080",
		"VLLMCTL_BASE_RELEASE": "29.06.1",
	})
	if !strings.Contains(b.calls(), "pull "+clavTag) {
		t.Errorf("the tag was never pulled:\n%s", b.calls())
	}
}

// The reason a moving tag needs declaring at all: once it exists locally,
// nothing would ever ask the registry about it again.
func TestRefreshPullsEvenWhenTheTagIsAlreadyLocal(t *testing.T) {
	b := newTrackBed(t)
	b.point(clavTag, oldBase)
	b.build("refresh")

	b.point(clavTag, newBase)
	out := b.build("refresh")

	env := b.env()
	wantEnv(t, env, map[string]string{
		"VLLMCTL_BASE_IMAGE":    newAlias,
		"VLLMCTL_VLLM_PIN":      "v0.29.0",
		"VLLMCTL_TUNER_REF":     "2bdbbc8080",
		"VLLMCTL_BASE_PREVIOUS": oldAlias,
	})
	// The way back has to be on disk, and the operator has to be told its name.
	if !strings.Contains(b.images(), oldAlias) {
		t.Errorf("the previous base was not kept:\n%s", b.images())
	}
	if !strings.Contains(out, "VLLMCTL_BASE_IMAGE="+oldAlias+" ./setup.sh pull") {
		t.Errorf("the move was not reported with the command to undo it:\n%s", out)
	}
}

// quick is run after every code change. If it followed the tag, an engine
// upgrade would arrive as a side effect of fixing a typo in a template.
func TestQuickStaysOnTheRecordedBase(t *testing.T) {
	b := newTrackBed(t)
	b.point(clavTag, oldBase)
	b.build("refresh")
	pullsBefore := strings.Count(b.calls(), "pull ")

	// Upstream moves, and something else on the machine pulls the tag.
	b.point(clavTag, newBase)
	b.run(nil, "./fakectl pull "+clavTag+"\n")

	b.build("keep")

	wantEnv(t, b.env(), map[string]string{
		"VLLMCTL_BASE_IMAGE":   oldAlias,
		"VLLMCTL_VLLM_PIN":     "v0.27.0",
		"VLLMCTL_TUNER_REF":    "55c98e370a",
		"VLLMCTL_BASE_RELEASE": "28.04.9",
		"VLLMCTL_BASE_DIGEST":  "docker.io/tcclaviger/vllm@sha256:" + oldBase,
	})
	if got := strings.Count(b.calls(), "pull ") - pullsBefore; got != 1 {
		t.Errorf("quick pulled the base image itself (%d pulls, 1 of them the test's own)", got)
	}
}

// Checked against the function body, like the Quadlet test beside it: the
// default is what matters, and running the function builds an image.
func TestQuickRebuildKeepsTheBaseByDefault(t *testing.T) {
	body := sourceSetupSh(t, "declare -f container_quick_rebuild\n")
	if !strings.Contains(body, `ensure_base_image "${1:-keep}"`) {
		t.Errorf("container_quick_rebuild does not default to keeping the base image:\n%s", body)
	}
}

// Going back is the override, and it has to survive the quick rebuilds that
// follow -- otherwise the first one after a rollback undoes it.
func TestGoingBackSticksUntilTheNextPull(t *testing.T) {
	b := newTrackBed(t)
	b.point(clavTag, oldBase)
	b.build("refresh")
	b.point(clavTag, newBase)
	b.build("refresh")

	b.build("refresh", "VLLMCTL_BASE_IMAGE="+oldAlias)
	wantEnv(t, b.env(), map[string]string{
		"VLLMCTL_BASE_IMAGE":    oldAlias,
		"VLLMCTL_VLLM_PIN":      "v0.27.0",
		"VLLMCTL_BASE_PREVIOUS": newAlias,
		// Known only by its local name by now, since the tag has moved off
		// it. The digest is still the image's own.
		"VLLMCTL_BASE_DIGEST": "docker.io/tcclaviger/vllm@sha256:" + oldBase,
	})

	b.build("keep")
	wantEnv(t, b.env(), map[string]string{"VLLMCTL_BASE_IMAGE": oldAlias, "VLLMCTL_VLLM_PIN": "v0.27.0"})

	b.build("refresh")
	wantEnv(t, b.env(), map[string]string{"VLLMCTL_BASE_IMAGE": newAlias, "VLLMCTL_BASE_PREVIOUS": oldAlias})
}

// Each base is ten gigabytes. The one in use and the one before it are kept;
// anything older is removed, and nothing belonging to another variant is.
func TestOlderBasesArePruned(t *testing.T) {
	b := newTrackBed(t)
	const third = "cccccccccccc3333333333333333333333333333333333333333333333333333"
	b.publish(third, "0.30.0", "30.01.1")

	for _, id := range []string{oldBase, newBase, third} {
		b.point(clavTag, id)
		b.build("refresh")
	}
	b.run(nil, "./fakectl tag "+clavTag+" localhost/vllmctl-radiance-flat:0.9.3\n")
	b.build("keep")

	images := b.images()
	if strings.Contains(images, oldAlias) {
		t.Errorf("a base two moves old is still on disk:\n%s", images)
	}
	for _, keep := range []string{newAlias, "localhost/vllmctl-rdna4-clav-base:cccccccccccc", "localhost/vllmctl-radiance-flat:0.9.3"} {
		if !strings.Contains(images, keep) {
			t.Errorf("%s was removed:\n%s", keep, images)
		}
	}
}

// An install made before the variant followed a tag has a release tag on
// record and nothing else. quick must not take that as licence to upgrade.
func TestAnInstallFromBeforeTrackingStaysWhereItWas(t *testing.T) {
	b := newTrackBed(t)
	const pinned = "docker.io/tcclaviger/vllm:28.04.9"
	b.point(pinned, oldBase)
	b.point(clavTag, newBase)
	b.run(nil, "./fakectl pull "+pinned+"\n")
	b.write(".env", "VLLMCTL_BASE_IMAGE="+pinned+"\nHF_TOKEN=keepme\n", 0o644)

	out := b.build("keep")

	env := b.env()
	wantEnv(t, env, map[string]string{
		"VLLMCTL_BASE_IMAGE": pinned,
		"VLLMCTL_VLLM_PIN":   "v0.27.0",
		"VLLMCTL_TUNER_REF":  "55c98e370a",
		"HF_TOKEN":           "keepme",
	})
	if strings.Contains(b.calls(), "pull "+clavTag) {
		t.Error("quick pulled the moving tag on an install that had never resolved it")
	}
	// The stale-pin warning is for variants that name a release. Here the
	// difference between .env and the manifest is the design.
	if strings.Contains(out, "Using the manifest") {
		t.Errorf("warned about a stale .env pin on a tracking variant:\n%s", out)
	}
}

// A commit that exists only in the image author's fork cannot be fetched from
// vllm-project/vllm. The release it was cut from is the same minor, which is
// as close as the build's own assertion asks for.
func TestAForkOnlyCommitFallsBackToTheRelease(t *testing.T) {
	b := newTrackBed(t)
	const forked = "dddddddddddd4444444444444444444444444444444444444444444444444444"
	b.publish(forked, "0.29.0.dev0+g"+unreachableCommit, "29.07.0")
	b.point(clavTag, forked)

	b.build("refresh")

	env := b.env()
	wantEnv(t, env, map[string]string{"VLLMCTL_VLLM_PIN": "v0.29.0"})
	if ref, set := env["VLLMCTL_TUNER_REF"]; set {
		t.Errorf("VLLMCTL_TUNER_REF=%q for a commit that is not upstream; the build would 404 on it", ref)
	}
}

// The pin is what the build asserts and the tuner ref is what it fetches, so
// the rule that derives them is the one TestPinIsAFetchableRefOrTunerRefIsSet
// asks a manifest author to apply by hand.
func TestVersionsBecomePinAndTunerRef(t *testing.T) {
	for _, tc := range []struct{ version, pin, ref string }{
		{"0.29.0", "v0.29.0", ""},
		{"0.29.0.dev0+g2bdbbc8080", "v0.29.0", "2bdbbc8080"},
		{"0.27.0.dev0+g55c98e370a", "v0.27.0", "55c98e370a"},
		{"0.23.1.dev1+g9ddef7117", "v0.23.1", "9ddef7117"},
		// setuptools-scm appends the date to a dirty tree.
		{"0.29.0.dev0+g2bdbbc8.d20260928", "v0.29.0", "2bdbbc8"},
		{"0.28.1.post1", "v0.28.1", ""},
		// A local version that is not a commit names nothing fetchable.
		{"0.28.0+rocm72", "v0.28.0", ""},
		{"0.23.1rc0", "v0.23.1rc0", ""},
	} {
		out := strings.TrimRight(sourceSetupSh(t, "vllm_refs_from_version '"+tc.version+"'\n"), "\n")
		pin, ref, _ := strings.Cut(out, " ")
		if pin != tc.pin || ref != tc.ref {
			t.Errorf("%s -> pin %q ref %q, want pin %q ref %q", tc.version, pin, ref, tc.pin, tc.ref)
		}
	}
}

// A variant that names one release is not touched by any of this: it is
// pulled once, built on under its own name, and its pin is the manifest's.
func TestPinnedVariantsAreLeftAlone(t *testing.T) {
	b := newTrackBed(t)
	d, ok := Get("radiance")
	if !ok {
		t.Skip("radiance manifest missing")
	}
	b.point(d.BaseImage, oldBase)

	for _, mode := range []string{"refresh", "refresh", "keep"} {
		b.run(nil, "BUILD_VARIANT=radiance\nensure_base_image "+mode+"\nwrite_env_file\n")
	}

	env := b.env()
	wantEnv(t, env, map[string]string{
		"VLLMCTL_BASE_IMAGE": d.BaseImage,
		"VLLMCTL_VLLM_PIN":   d.VLLMPin,
	})
	for _, k := range []string{"VLLMCTL_BASE_DIGEST", "VLLMCTL_BASE_PREVIOUS", "VLLMCTL_BASE_VLLM"} {
		if v, set := env[k]; set {
			t.Errorf("%s=%q was recorded for a variant that does not track its base", k, v)
		}
	}
	if got := strings.Count(b.calls(), "pull "); got != 1 {
		t.Errorf("a pinned base was pulled %d times, want once", got)
	}
	if strings.Contains(b.calls(), "run ") {
		t.Error("a container was run to read a pin the manifest already states")
	}
}

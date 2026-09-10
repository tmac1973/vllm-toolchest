package vllmenv

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// mkVenv builds a fake venv layout under root and returns its path.
func mkVenv(t *testing.T, root, pyVersion string) string {
	t.Helper()
	sp := filepath.Join(root, "lib", "python"+pyVersion, "site-packages", "vllm")
	if err := os.MkdirAll(sp, 0o755); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(root, "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bin, "python"), []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestDetectFindsVenvViaOverride(t *testing.T) {
	root := mkVenv(t, filepath.Join(t.TempDir(), "vllm"), "3.12")
	t.Setenv("VLLMCTL_VLLM_VENV", root)

	e := Detect()

	if e.VenvRoot != root {
		t.Errorf("VenvRoot = %q, want %q", e.VenvRoot, root)
	}
	if want := filepath.Join(root, "bin", "python"); e.Python != want {
		t.Errorf("Python = %q, want %q", e.Python, want)
	}
	wantCfg := filepath.Join(root, "lib/python3.12/site-packages",
		"vllm/model_executor/layers/quantization/utils/configs")
	if e.BlockFP8ConfigsDir != wantCfg {
		t.Errorf("BlockFP8ConfigsDir = %q, want %q", e.BlockFP8ConfigsDir, wantCfg)
	}
}

// The two images ship different Python minor versions over time, so the venv
// probe globs rather than hardcoding 3.12.
func TestDetectMatchesAnyPython3Minor(t *testing.T) {
	root := mkVenv(t, filepath.Join(t.TempDir(), "vllm"), "3.14")
	t.Setenv("VLLMCTL_VLLM_VENV", root)

	if e := Detect(); e.VenvRoot != root {
		t.Errorf("VenvRoot = %q, want %q", e.VenvRoot, root)
	}
}

// A directory that looks like a venv but has no vLLM in it must not be taken:
// otherwise the tuner would symlink configs into a package that isn't there.
func TestDetectSkipsVenvWithoutVLLM(t *testing.T) {
	root := filepath.Join(t.TempDir(), "empty")
	if err := os.MkdirAll(filepath.Join(root, "lib/python3.12/site-packages"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLMCTL_VLLM_VENV", root)

	e := Detect()
	if e.VenvRoot != "" {
		t.Errorf("VenvRoot = %q, want empty", e.VenvRoot)
	}
	if e.Python != "python" {
		t.Errorf("Python = %q, want the bare fallback", e.Python)
	}
}

// Which image this is decides which knobs the Settings page offers and which
// attention backends the model config panel lists, so getting it wrong is not
// cosmetic: it offers a backend the stack does not have, and the engine aborts
// minutes into a load.
func TestDetectVariant(t *testing.T) {
	for _, tc := range []struct {
		name        string
		env         map[string]string
		wantVariant string
	}{
		{
			// Outside a container, nothing identifies the image. Reporting
			// "unknown" is what makes the UI degrade to its generic form
			// rather than claiming capabilities it cannot verify.
			name:        "nothing identifies the image",
			env:         nil,
			wantVariant: VariantUnknown,
		},
		{
			// Dockerfile.prebuilt stamps this from the manifest, and the
			// from-source Dockerfiles set it too, so it is the normal path
			// rather than an escape hatch.
			name:        "the image names itself",
			env:         map[string]string{"VLLMCTL_IMAGE_VARIANT": "radiance"},
			wantVariant: "radiance",
		},
		{
			name:        "a from-source image names itself",
			env:         map[string]string{"VLLMCTL_IMAGE_VARIANT": "rocm-source"},
			wantVariant: "rocm-source",
		},
		{
			// Honoured rather than corrected: the operator set it, and the
			// Settings page shows what it reports. Nothing matches it, so no
			// knobs and no vendor backends are offered.
			name:        "an unrecognised name is reported as given",
			env:         map[string]string{"VLLMCTL_IMAGE_VARIANT": "some-future-image"},
			wantVariant: "some-future-image",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("VLLMCTL_IMAGE_VARIANT", "")
			for k, v := range tc.env {
				t.Setenv(k, v)
			}

			e := Detect()
			if e.Variant != tc.wantVariant {
				t.Errorf("Variant = %q, want %q", e.Variant, tc.wantVariant)
			}
			// Capabilities follow the variant, and only a described one has any.
			if got, want := e.Has("r4d_allreduce"), tc.wantVariant == "radiance"; got != want {
				t.Errorf("Has(r4d_allreduce) = %v, want %v", got, want)
			}
			if _, known := e.Descriptor(); known != (tc.wantVariant == "radiance" || tc.wantVariant == "rocm-source") {
				t.Errorf("Descriptor() resolved unexpectedly for %q", tc.wantVariant)
			}
		})
	}
}

// The stamp file is how an image is identified when nothing names it: a marker
// its base ships that no other image has. Detect reads absolute paths, so this
// exercises the reader rather than the real /opt locations.
func TestStampVersion(t *testing.T) {
	dir := t.TempDir()

	missing := filepath.Join(dir, "absent")
	if v := stampVersion(missing); v != "" {
		t.Errorf("a missing stamp = %q, want empty", v)
	}
	if v := stampVersion(""); v != "" {
		t.Errorf("no stamp path = %q, want empty", v)
	}

	withVersion := filepath.Join(dir, "version")
	os.WriteFile(withVersion, []byte("  0.9.3\n"), 0o644)
	if v := stampVersion(withVersion); v != "0.9.3" {
		t.Errorf("stamp = %q, want the trimmed version", v)
	}

	// A stamp with no readable content still identifies the image, so it must
	// not read as absent -- otherwise detection falls through to the next
	// variant and reports the wrong one.
	empty := filepath.Join(dir, "empty")
	os.WriteFile(empty, []byte("\n"), 0o644)
	if v := stampVersion(empty); v == "" {
		t.Error("an empty-but-present stamp must still identify the image")
	}
}

// The venv is where every tuned-kernel path, the bitsandbytes check and the
// tuner all point. Probing must cover the running variant's declared root and
// every other one, because an operator who overrides the variant should still
// end up with a working install.
func TestVenvCandidatesCoverEveryDeclaredRoot(t *testing.T) {
	t.Setenv("VLLMCTL_VLLM_VENV", "")
	t.Setenv("VIRTUAL_ENV", "")

	d, ok := variants.Get("radiance")
	if !ok {
		t.Fatal("radiance manifest missing")
	}
	got := venvCandidates(d, true)

	if len(got) == 0 || got[0] != d.VenvRoot {
		t.Errorf("candidates = %v, want the running variant's root first", got)
	}
	for _, other := range variants.All() {
		if other.VenvRoot == "" {
			continue
		}
		if !slices.Contains(got, other.VenvRoot) {
			t.Errorf("%s declares venv %s but it is never probed", other.ID, other.VenvRoot)
		}
	}
	// No duplicates: probing the same directory twice is harmless but says
	// the de-duplication is broken.
	seen := map[string]bool{}
	for _, c := range got {
		if seen[c] {
			t.Errorf("candidate %q listed twice", c)
		}
		seen[c] = true
	}
}

// An explicit override beats every declared root, so an operator can point
// vllmctl at an install nothing knows about.
func TestVenvOverrideComesFirst(t *testing.T) {
	t.Setenv("VLLMCTL_VLLM_VENV", "/somewhere/else")
	t.Setenv("VIRTUAL_ENV", "")
	d, _ := variants.Get("radiance")
	if got := venvCandidates(d, true); got[0] != "/somewhere/else" {
		t.Errorf("candidates = %v, want the override first", got)
	}
}

func TestServeCommand(t *testing.T) {
	for _, tc := range []struct {
		name     string
		launcher []string
		wantBin  string
		wantArgs []string
	}{
		{
			name:     "generic inserts the serve subcommand",
			launcher: []string{"vllm", "serve"},
			wantBin:  "vllm",
			wantArgs: []string{"serve", "/models/m", "--port", "8000"},
		},
		{
			// The radiance entrypoint already execs `vllm serve`, so passing
			// "serve" again would make the model path an unknown argument.
			name:     "radiance entrypoint takes the model path directly",
			launcher: []string{"/opt/radiance_entrypoint.sh"},
			wantBin:  "/opt/radiance_entrypoint.sh",
			wantArgs: []string{"/models/m", "--port", "8000"},
		},
		{
			name:     "empty launcher falls back to vllm serve",
			launcher: nil,
			wantBin:  "vllm",
			wantArgs: []string{"serve", "/models/m", "--port", "8000"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := Env{Launcher: tc.launcher}
			bin, args := e.ServeCommand("/models/m", []string{"--port", "8000"})
			if bin != tc.wantBin {
				t.Errorf("bin = %q, want %q", bin, tc.wantBin)
			}
			if !reflect.DeepEqual(args, tc.wantArgs) {
				t.Errorf("args = %#v, want %#v", args, tc.wantArgs)
			}
		})
	}
}

// ServeCommand must not alias or mutate the launcher slice: it is shared, and
// a second call would otherwise see arguments left behind by the first.
func TestServeCommandDoesNotMutateLauncher(t *testing.T) {
	e := Env{Launcher: []string{"vllm", "serve"}}
	e.ServeCommand("/models/a", []string{"--x"})
	_, args := e.ServeCommand("/models/b", []string{"--y"})

	if !reflect.DeepEqual(e.Launcher, []string{"vllm", "serve"}) {
		t.Errorf("launcher was mutated: %#v", e.Launcher)
	}
	if !reflect.DeepEqual(args, []string{"serve", "/models/b", "--y"}) {
		t.Errorf("second call leaked state: %#v", args)
	}
}

// stubPython writes an executable that ignores its arguments and emits body on
// stdout, standing in for the interpreter ProbeDeviceName shells out to.
func stubPython(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "python")
	script := "#!/bin/sh\n" + body + "\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestProbeDeviceName(t *testing.T) {
	for _, tc := range []struct {
		name    string
		body    string
		want    string
		wantErr bool
	}{
		{
			// vLLM reports a spaced marketing name on the radiance image; the
			// filenames it keys tuned configs by use underscores.
			name: "spaces become underscores",
			body: `printf 'AMD Radeon R9700\n'`,
			want: "AMD_Radeon_R9700",
		},
		{
			// The generic image's patched get_device_name has no spaces.
			name: "already-underscored name passes through",
			body: `printf 'AMD-gfx1201'`,
			want: "AMD-gfx1201",
		},
		{
			name: "surrounding whitespace is trimmed",
			body: `printf '  AMD Radeon R9700  \n\n'`,
			want: "AMD_Radeon_R9700",
		},
		{
			// Better to keep the architecture-derived fallback than to key
			// every tuned config on an empty string.
			name:    "empty output is an error",
			body:    `printf ''`,
			wantErr: true,
		},
		{
			name:    "non-zero exit is an error",
			body:    `exit 1`,
			wantErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := Env{Python: stubPython(t, tc.body)}
			got, err := e.ProbeDeviceName(10 * time.Second)

			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("ProbeDeviceName() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestProbeDeviceNameHonoursTimeout(t *testing.T) {
	e := Env{Python: stubPython(t, "sleep 10")}

	start := time.Now()
	if _, err := e.ProbeDeviceName(200 * time.Millisecond); err == nil {
		t.Fatal("expected a timeout error")
	}
	// The probe must not hold up boot: it runs in the background, but a
	// runaway child would still pin a HIP context and its VRAM.
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("probe took %v; the timeout did not fire", elapsed)
	}
}

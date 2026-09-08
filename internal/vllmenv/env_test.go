package vllmenv

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
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

func TestDetectVariant(t *testing.T) {
	for _, tc := range []struct {
		name        string
		env         map[string]string
		wantVariant string
		wantVersion string
	}{
		{"default is generic", nil, VariantGeneric, ""},
		{"RADIANCE_VERSION marks radiance",
			map[string]string{"RADIANCE_VERSION": "0.9.3"}, VariantRadiance, "0.9.3"},
		{"explicit override wins",
			map[string]string{"VLLMCTL_IMAGE_VARIANT": "radiance"}, VariantRadiance, ""},
		{"override can force generic on a radiance image",
			map[string]string{"RADIANCE_VERSION": "0.9.3", "VLLMCTL_IMAGE_VARIANT": "generic"},
			VariantGeneric, "0.9.3"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Clear both so the host environment can't leak in.
			t.Setenv("RADIANCE_VERSION", "")
			t.Setenv("VLLMCTL_IMAGE_VARIANT", "")
			for k, v := range tc.env {
				t.Setenv(k, v)
			}

			e := Detect()
			if e.Variant != tc.wantVariant {
				t.Errorf("Variant = %q, want %q", e.Variant, tc.wantVariant)
			}
			if e.RadianceVersion != tc.wantVersion {
				t.Errorf("RadianceVersion = %q, want %q", e.RadianceVersion, tc.wantVersion)
			}
			if got := e.IsRadiance(); got != (tc.wantVariant == VariantRadiance) {
				t.Errorf("IsRadiance() = %v", got)
			}
		})
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

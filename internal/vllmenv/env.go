// Package vllmenv discovers the vLLM installation vllmctl is managing.
//
// vllmctl ships in more than one image variant, and the variants do not agree
// on where things live:
//
//	generic (Dockerfile.rocm / Dockerfile.cuda)  venv at /opt/vllm-venv
//	radiance (Dockerfile.radiance)               venv at /opt/vllm
//
// Everything that used to be a hardcoded path is resolved here once at boot so
// the same binary works in either image (and degrades to a harmless no-op when
// run outside a container during development).
package vllmenv

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// Variant names the image this binary is running inside.
const (
	VariantGeneric  = "generic"
	VariantRadiance = "radiance"
)

// Env is the resolved layout of the vLLM install.
type Env struct {
	// Variant is "generic" or "radiance".
	Variant string

	// RadianceVersion is the version string from /opt/radiance_version,
	// empty on a generic image.
	RadianceVersion string

	// VenvRoot is the virtualenv prefix (e.g. /opt/vllm-venv), empty if no
	// vLLM install was found.
	VenvRoot string

	// Python is the interpreter to run tuner/probe subprocesses with.
	// Falls back to "python" when no venv was located.
	Python string

	// SitePackages is the venv's site-packages directory.
	SitePackages string

	// BlockFP8ConfigsDir is where vLLM looks up tuned block-FP8 GEMM configs.
	BlockFP8ConfigsDir string

	// Launcher is the argv prefix that starts a server. On a generic image
	// this is ["vllm", "serve"]; on radiance it is the radiance entrypoint,
	// which prints the startup banner, optionally applies NUMA binding, and
	// then execs `vllm serve` with the same arguments.
	Launcher []string

	// TunerScript is the fp8 tuning wrapper, empty when not installed.
	TunerScript string

	// HasBitsAndBytes reports whether the bitsandbytes package is installed.
	//
	// It is not on every image: the ROCm one dropped it when its only ROCm
	// fork stopped compiling for wave32 (see Dockerfile.rocm), and radiance
	// has never carried it. Offering BitsAndBytes quantization on an image
	// without it produces a launch that fails on an import, minutes after the
	// operator picked it.
	HasBitsAndBytes bool
}

// venvCandidates are probed in order. VIRTUAL_ENV comes first so an operator
// can point vllmctl at a venv we don't know about.
func venvCandidates() []string {
	var c []string
	if v := os.Getenv("VLLMCTL_VLLM_VENV"); v != "" {
		c = append(c, v)
	}
	if v := os.Getenv("VIRTUAL_ENV"); v != "" {
		c = append(c, v)
	}
	return append(c, "/opt/vllm-venv", "/opt/vllm")
}

// Detect resolves the environment. It only touches the filesystem — no GPU is
// initialized and no Python is executed, so it is safe to call at boot.
func Detect() Env {
	e := Env{
		Variant:  VariantGeneric,
		Python:   "python",
		Launcher: []string{"vllm", "serve"},
	}

	// A radiance image stamps its version into /opt. RADIANCE_VERSION is the
	// same value as an env var, and VLLMCTL_IMAGE_VARIANT is the explicit
	// operator override for anything the two miss.
	if b, err := os.ReadFile("/opt/radiance_version"); err == nil {
		e.Variant = VariantRadiance
		e.RadianceVersion = strings.TrimSpace(string(b))
	} else if v := os.Getenv("RADIANCE_VERSION"); v != "" {
		e.Variant = VariantRadiance
		e.RadianceVersion = v
	}
	if v := os.Getenv("VLLMCTL_IMAGE_VARIANT"); v != "" {
		e.Variant = v
	}

	for _, root := range venvCandidates() {
		sp, ok := sitePackagesWithVLLM(root)
		if !ok {
			continue
		}
		e.VenvRoot = root
		e.SitePackages = sp
		e.BlockFP8ConfigsDir = filepath.Join(sp,
			"vllm/model_executor/layers/quantization/utils/configs")
		e.HasBitsAndBytes = fileExists(filepath.Join(sp, "bitsandbytes"))
		if py := filepath.Join(root, "bin/python"); isExecutable(py) {
			e.Python = py
		}
		break
	}

	// The radiance entrypoint takes the same arguments `vllm serve` does and
	// execs it, so using it as the launcher keeps the arch/P2P/version banner
	// and the bandwidth sweep in our log stream. Fall back to plain `vllm`
	// if it isn't there (e.g. someone ran vllmctl against a stock image).
	if e.Variant == VariantRadiance && isExecutable("/opt/radiance_entrypoint.sh") {
		e.Launcher = []string{"/opt/radiance_entrypoint.sh"}
	}

	if p := "/opt/vllm-tuner/tune_fp8_wrapper.py"; fileExists(p) {
		e.TunerScript = p
	}

	return e
}

// IsRadiance reports whether this is the radiance image variant.
func (e Env) IsRadiance() bool { return e.Variant == VariantRadiance }

// Describe returns a one-line human-readable summary for logs and the UI.
func (e Env) Describe() string {
	s := e.Variant
	if e.RadianceVersion != "" {
		s += " " + e.RadianceVersion
	}
	if e.VenvRoot != "" {
		s += " (" + e.VenvRoot + ")"
	}
	return s
}

// ServeCommand returns the argv for serving modelPath with args.
func (e Env) ServeCommand(modelPath string, args []string) (string, []string) {
	launcher := e.Launcher
	if len(launcher) == 0 {
		launcher = []string{"vllm", "serve"}
	}
	argv := append(append([]string{}, launcher[1:]...), modelPath)
	argv = append(argv, args...)
	return launcher[0], argv
}

// deviceNameProbe asks the installed vLLM what it calls this GPU. That string
// is what vLLM interpolates into tuned-kernel filenames, so guessing it from
// the gfx target is wrong on any image that doesn't patch get_device_name --
// which is exactly the difference between the generic image (patched to report
// "AMD-gfx1201") and radiance (unpatched, reports "AMD Radeon R9700").
const deviceNameProbe = `
import sys
try:
    from vllm.platforms import current_platform
    sys.stdout.write(current_platform.get_device_name())
except Exception:
    import torch
    sys.stdout.write(torch.cuda.get_device_name(0))
`

// ProbeDeviceName runs a short-lived subprocess to read vLLM's device name.
//
// It is a subprocess on purpose: resolving the name initializes HIP/CUDA and
// costs ~2 GiB of VRAM for the context, which we do not want held for the life
// of vllmctl. The child exits and the driver reclaims it.
//
// Returns the name with spaces replaced by underscores, matching the form vLLM
// uses in config filenames.
func (e Env) ProbeDeviceName(timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, e.Python, "-c", deviceNameProbe)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	// Importing vLLM spawns children of its own. Two problems follow from
	// that, and both bite exactly when the probe has gone wrong:
	//
	//  - Killing only the direct child orphans the rest, and an orphan that
	//    reached the GPU keeps a HIP context (and its VRAM) alive. Putting the
	//    probe in its own process group lets us signal the whole tree.
	//  - Output() waits for EOF on the stdout pipe, which any surviving
	//    grandchild holds open. Without WaitDelay the call blocks long past
	//    the timeout it was given.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 5 * time.Second

	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	name := strings.TrimSpace(string(out))
	if name == "" {
		return "", errEmptyDeviceName
	}
	return strings.ReplaceAll(name, " ", "_"), nil
}

type probeError string

func (e probeError) Error() string { return string(e) }

const errEmptyDeviceName = probeError("vllm reported an empty device name")

func sitePackagesWithVLLM(root string) (string, bool) {
	matches, _ := filepath.Glob(filepath.Join(root, "lib/python3.*/site-packages"))
	for _, sp := range matches {
		if fileExists(filepath.Join(sp, "vllm")) {
			return sp, true
		}
	}
	return "", false
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func isExecutable(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir() && fi.Mode()&0o111 != 0
}

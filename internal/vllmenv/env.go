// Package vllmenv discovers the vLLM installation vllmctl is managing.
//
// vllmctl ships in a dozen image variants and they do not agree on where
// anything lives: the venv root, the launcher to start a server with, the file
// that proves which image this is. Each variant states its own layout in
// variants/<id>.conf, and this package resolves it once at boot, so the same
// binary works in every image and degrades to a harmless no-op when run
// outside a container during development.
package vllmenv

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// VariantUnknown is what Detect reports when nothing identifies the image: a
// development checkout, or a container built before its manifest existed.
// Every consumer treats it as "no variant-specific behaviour", which is the
// safe reading — offering a tuned backend on an image that lacks it produces a
// startup abort minutes after someone picks it.
const VariantUnknown = ""

// Env is the resolved layout of the vLLM install.
type Env struct {
	// Variant is "generic" or "radiance".
	Variant string

	// VariantVersion is the version string from the variant's stamp file,
	// empty when the variant does not ship one.
	VariantVersion string

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

// venvCandidates are probed in order: the explicit override first, so an
// operator can point vllmctl at a venv nothing knows about, then the variant's
// own declared root, then every other variant's, then the two historical
// defaults.
//
// The wide net is deliberate. Getting this wrong means every tuned-kernel
// path, the bitsandbytes check and the tuner all silently target the wrong
// interpreter, so it is worth probing a few directories that will not exist.
func venvCandidates(d variants.Descriptor, known bool) []string {
	var c []string
	add := func(v string) {
		if v == "" {
			return
		}
		for _, seen := range c {
			if seen == v {
				return
			}
		}
		c = append(c, v)
	}

	add(os.Getenv("VLLMCTL_VLLM_VENV"))
	add(os.Getenv("VIRTUAL_ENV"))
	if known {
		add(d.VenvRoot)
	}
	for _, other := range variants.All() {
		add(other.VenvRoot)
	}
	add("/opt/vllm-venv")
	add("/opt/vllm")
	return c
}

// detectVariant works out which image this is.
//
// The explicit override wins: Dockerfile.prebuilt stamps VLLMCTL_IMAGE_VARIANT
// from the manifest, which makes it the normal path rather than an escape
// hatch. Failing that, a variant is identified by its stamp file -- a marker
// its base image ships, which no other image has.
func detectVariant() (string, string) {
	if v := os.Getenv("VLLMCTL_IMAGE_VARIANT"); v != "" {
		if d, ok := variants.Get(v); ok {
			return v, stampVersion(d.StampFile)
		}
		// Honour an unrecognised name rather than silently reporting
		// something else: the operator set it, and the UI says so.
		return v, ""
	}
	for _, d := range variants.All() {
		if d.StampFile == "" {
			continue
		}
		if version := stampVersion(d.StampFile); version != "" {
			return d.ID, version
		}
	}
	return VariantUnknown, ""
}

// stampVersion reads a variant's stamp file. A stamp with no readable content
// still identifies the image, so an empty-but-present file reports a space
// rather than nothing -- callers test the version for display, not identity.
func stampVersion(path string) string {
	if path == "" {
		return ""
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	if v := strings.TrimSpace(string(b)); v != "" {
		return v
	}
	return " "
}

// Detect resolves the environment. It only touches the filesystem — no GPU is
// initialized and no Python is executed, so it is safe to call at boot.
func Detect() Env {
	e := Env{
		Python:   "python",
		Launcher: []string{"vllm", "serve"},
	}

	e.Variant, e.VariantVersion = detectVariant()
	if strings.TrimSpace(e.VariantVersion) == "" {
		e.VariantVersion = ""
	}
	d, known := variants.Get(e.Variant)

	for _, root := range venvCandidates(d, known) {
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

	// A variant may ship a launcher that takes the same arguments `vllm serve`
	// does and execs it. Using it keeps whatever that script contributes --
	// an arch/P2P banner, NUMA binding, a bandwidth sweep -- in our log
	// stream. Fall back to plain `vllm` if it is not there, which is what
	// happens when vllmctl is run against a stock image.
	if known && len(d.Launcher) > 0 && isExecutable(d.Launcher[0]) {
		e.Launcher = d.Launcher
	}

	if p := "/opt/vllm-tuner/tune_fp8_wrapper.py"; fileExists(p) {
		e.TunerScript = p
	}

	return e
}

// Descriptor returns the manifest describing this image, if one does.
func (e Env) Descriptor() (variants.Descriptor, bool) { return variants.Get(e.Variant) }

// Has reports whether this image's variant declares a capability. An image no
// manifest describes has none, which is what makes the UI degrade to its
// generic form rather than offering something the stack cannot do.
func (e Env) Has(capability string) bool {
	d, ok := variants.Get(e.Variant)
	return ok && d.Has(capability)
}

// Describe returns a one-line human-readable summary for logs and the UI.
func (e Env) Describe() string {
	s := e.Variant
	if s == "" {
		s = "unknown"
	}
	if e.VariantVersion != "" {
		s += " " + e.VariantVersion
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

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
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/fsutil"
	"github.com/tmac1973/vllm-toolchest/internal/procgroup"
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

	// Python is the interpreter to run probe subprocesses with.
	// Falls back to "python" when no venv was located.
	Python string

	// SitePackages is the venv's site-packages directory.
	SitePackages string

	// Launcher is the argv prefix that starts a server. On a generic image
	// this is ["vllm", "serve"]; on radiance it is the radiance entrypoint,
	// which prints the startup banner, optionally applies NUMA binding, and
	// then execs `vllm serve` with the same arguments.
	Launcher []string

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
// The wide net is deliberate. Getting this wrong means the probes and the
// bitsandbytes check all silently target the wrong interpreter, so it is
// worth probing a few directories that will not exist.
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
		e.HasBitsAndBytes = fsutil.Exists(filepath.Join(sp, "bitsandbytes"))
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
	out, err := e.runProbe(timeout, deviceNameProbe)
	if err != nil {
		return "", err
	}
	name := strings.TrimSpace(string(out))
	if name == "" {
		return "", errEmptyDeviceName
	}
	return strings.ReplaceAll(name, " ", "_"), nil
}

// supportedArchsProbe lists the model architectures the installed vLLM can
// load. An import and a registry read: no GPU, no engine. Each name is
// marked, because importing vLLM can log to stdout too.
const supportedArchsProbe = `
from vllm.model_executor.models import ModelRegistry
for a in sorted(ModelRegistry.get_supported_archs()):
    print("ARCH " + a)
`

// ProbeSupportedArchs asks the installed vLLM which model architectures it
// can load. It is version-specific and lives only inside the image: the 28.04.9
// rdna4-clav image was needed for qwen4_exp, and a model whose architecture
// the image lacks fails only at start, after its download. On compute's image
// it lists 386 in about twenty seconds.
//
// An empty list is an error, not a vLLM that supports nothing: treating it as
// data would mark every model unverified.
func (e Env) ProbeSupportedArchs(timeout time.Duration) ([]string, error) {
	out, err := e.runProbe(timeout, supportedArchsProbe)
	if err != nil {
		return nil, err
	}
	return parseArchs(string(out))
}

// RegistryFingerprint identifies the model registry the image's vLLM reads
// its supported architectures from: a hash of
// vllm/model_executor/models/registry.py, "" when it cannot be read.
//
// It keys the cached list rather than the variant's stamp version, which
// rdna4-clav does not have, or the vLLM version, which the clav releases do
// not always move: 28.02.2 and 28.04.9 both carried 0.27.0.dev0, and the
// second added qwen4_exp. The registry file changes exactly when the answer
// can.
func (e Env) RegistryFingerprint() string {
	if e.SitePackages == "" {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(e.SitePackages, "vllm/model_executor/models/registry.py"))
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}

// parseArchs reads the probe's marked lines, sorted and without duplicates.
func parseArchs(out string) ([]string, error) {
	seen := map[string]bool{}
	var archs []string
	for _, line := range strings.Split(out, "\n") {
		name, ok := strings.CutPrefix(strings.TrimSpace(line), "ARCH ")
		if name = strings.TrimSpace(name); ok && name != "" && !seen[name] {
			seen[name] = true
			archs = append(archs, name)
		}
	}
	if len(archs) == 0 {
		return nil, errNoArchs
	}
	sort.Strings(archs)
	return archs, nil
}

// runProbe runs a Python snippet in the image's interpreter and returns its
// stdout.
func (e Env) runProbe(timeout time.Duration, script string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Importing vLLM spawns children of its own; see procgroup for why the
	// whole group goes on timeout.
	cmd := procgroup.Command(ctx, syscall.SIGKILL, 5*time.Second, e.Python, "-c", script)
	cmd.Env = append(procgroup.Environ(), "PYTHONUNBUFFERED=1")

	return cmd.Output()
}

type probeError string

func (e probeError) Error() string { return string(e) }

const (
	errEmptyDeviceName = probeError("vllm reported an empty device name")
	errNoArchs         = probeError("vllm listed no supported model architectures")
)

func sitePackagesWithVLLM(root string) (string, bool) {
	matches, _ := filepath.Glob(filepath.Join(root, "lib/python3.*/site-packages"))
	for _, sp := range matches {
		if fsutil.Exists(filepath.Join(sp, "vllm")) {
			return sp, true
		}
	}
	return "", false
}

func isExecutable(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir() && fi.Mode()&0o111 != 0
}

package tuning

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultVLLMConfigsDir is where the generic image's venv puts vLLM. It is
// only a fallback: the real path is discovered at boot (the radiance image
// installs its venv at /opt/vllm, not /opt/vllm-venv) and passed in.
const DefaultVLLMConfigsDir = "/opt/vllm-venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/configs"

// InstallTunedConfigs symlinks every JSON in /{dataDir}/tuned-kernels/{deviceName}/
// into vLLM's configs directory so the runtime picks them up. Idempotent and
// safe to call on every vllmctl boot — overwrites stale symlinks, leaves the
// upstream-shipped JSONs (regular files) alone.
//
// configsDir is vLLM's block-FP8 config directory; empty falls back to
// DefaultVLLMConfigsDir.
//
// Returns the number of symlinks installed and any non-fatal warning text.
// Missing source dir is not an error (first-run case).
func InstallTunedConfigs(dataDir, deviceName, configsDir string) (int, string, error) {
	if configsDir == "" {
		configsDir = DefaultVLLMConfigsDir
	}

	srcDir := filepath.Join(dataDir, "tuned-kernels", deviceName)
	if _, err := os.Stat(srcDir); os.IsNotExist(err) {
		return 0, "", nil
	} else if err != nil {
		return 0, "", err
	}

	if _, err := os.Stat(configsDir); err != nil {
		// vLLM venv not present (dev environment) — quietly skip.
		return 0, fmt.Sprintf("skipped: %s not found", configsDir), nil
	}

	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return 0, "", err
	}

	installed := 0
	var warnings []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		src := filepath.Join(srcDir, e.Name())
		dst := filepath.Join(configsDir, e.Name())

		if info, err := os.Lstat(dst); err == nil {
			// Replace existing symlinks, but never overwrite a regular file
			// shipped by upstream vLLM — that's data loss. On the radiance
			// image this also protects its own tuned configs, which are
			// baked into site-packages as regular files.
			if info.Mode()&os.ModeSymlink == 0 {
				warnings = append(warnings, fmt.Sprintf("skipped %s: upstream regular file present", e.Name()))
				continue
			}
			if err := os.Remove(dst); err != nil {
				warnings = append(warnings, fmt.Sprintf("remove old symlink %s: %v", e.Name(), err))
				continue
			}
		}
		if err := os.Symlink(src, dst); err != nil {
			warnings = append(warnings, fmt.Sprintf("symlink %s: %v", e.Name(), err))
			continue
		}
		installed++
	}

	return installed, strings.Join(warnings, "; "), nil
}

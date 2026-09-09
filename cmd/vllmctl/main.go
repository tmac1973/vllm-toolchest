package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/api"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/tuning"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

// version is stamped at build time via -ldflags "-X main.version=...". The
// Makefile fills it from git describe; a plain `go build` leaves it at "dev".
var version = "dev"

func main() {
	configPath := flag.String("config", "/data/config/vllmctl.yaml", "config file path")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	if err := initDataDir(cfg.DataDir); err != nil {
		slog.Warn("could not init data dir (expected in local dev)", "error", err)
	}

	// Resolve which image variant we're in and where its vLLM lives. Purely
	// filesystem inspection — no GPU is touched.
	env := vllmenv.Detect()

	// A device name resolved on an earlier boot, either saved into the config
	// or left in the cache. Applied before the server starts so nothing else
	// is concurrently reading the config.
	//
	// The saved one is checked first because a name that contradicts the build
	// is worse than none: the probe is skipped entirely while a stored name is
	// present, so a stale one persists across every restart and every rebuild.
	if arch, ok := archFromDeviceName(cfg.VLLMDeviceName); ok && arch != cfg.GPUArch {
		slog.Warn("stored vLLM device name disagrees with this build; re-probing",
			"stored", cfg.VLLMDeviceName, "gpu_arch", cfg.GPUArch)
		cfg.VLLMDeviceName = ""
	}
	if cfg.VLLMDeviceName == "" {
		if name := readCachedDeviceName(cfg.DataDir, cfg.GPUArch); name != "" {
			cfg.VLLMDeviceName = name
			slog.Info("using cached vLLM device name", "device", name)
		}
	}

	slog.Info("vLLM environment", "variant", env.Variant, "venv", env.VenvRoot,
		"launcher", strings.Join(env.Launcher, " "), "radiance_version", env.RadianceVersion)

	// Hot-link any operator-tuned kernel configs into vLLM's site-packages
	// so vLLM picks them up on next launch. No-op when nothing's been tuned
	// or when running outside the container.
	installTuned(cfg.DataDir, cfg.DeviceNameSuffix(), env)

	srv := api.NewServerWithEnv(cfg, env, version)

	// Resolve the device name vLLM actually reports, in the background: the
	// probe imports vLLM and initializes the GPU, which takes tens of seconds
	// and must not hold up the listener. Until it lands, the architecture
	// -derived fallback stands.
	go resolveDeviceName(cfg, env, srv)

	httpSrv := &http.Server{
		Addr:    cfg.ListenAddr,
		Handler: srv.Router(),
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("listening", "addr", cfg.ListenAddr)
		if err := httpSrv.ListenAndServe(); err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	// Auto-start after the listener is up, and in the background: loading a
	// model takes minutes, and doing it before serving would leave the UI
	// unreachable for exactly as long as the operator most wants to watch it.
	go srv.AutoStart()

	<-ctx.Done()
	slog.Info("shutting down")
	httpSrv.Shutdown(context.Background())
}

func initDataDir(dataDir string) error {
	dirs := []string{
		filepath.Join(dataDir, "config"),
		filepath.Join(dataDir, "models"),
		filepath.Join(dataDir, "tuned-kernels"),
	}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("creating %s: %w", dir, err)
		}
	}
	return nil
}

// installTuned links operator-tuned kernel configs into vLLM's config dir.
func installTuned(dataDir, deviceName string, env vllmenv.Env) {
	if deviceName == "" {
		return
	}
	n, warn, err := tuning.InstallTunedConfigs(dataDir, deviceName, env.BlockFP8ConfigsDir)
	if err != nil {
		slog.Warn("install tuned kernel configs", "error", err)
		return
	}
	if n > 0 {
		slog.Info("installed tuned kernel configs", "count", n, "device", deviceName, "warnings", warn)
	}
}

// archFromDeviceName extracts the gfx target from a device name that encodes
// one, reporting whether it did.
//
// Only the patched form carries a target — "AMD-gfx1201". A marketing name
// like "AMD_Radeon_R9700" or "AMD_Radeon_RX_7900_XTX" says nothing about the
// architecture and must not be second-guessed, so it reports false and is left
// alone.
//
// This exists because the two can disagree. An image built for one gfx target
// with another target's patches applied reports a device name from the patches
// rather than from the hardware; the name is then saved and reused forever,
// and tuned kernel configs keyed by it silently stop being found.
func archFromDeviceName(name string) (string, bool) {
	const prefix = "AMD-gfx"
	if !strings.HasPrefix(name, prefix) {
		return "", false
	}
	arch := "gfx" + strings.TrimPrefix(name, prefix)
	for _, r := range strings.TrimPrefix(arch, "gfx") {
		// gfx targets are digits with an optional trailing letter (gfx90a).
		if (r < '0' || r > '9') && (r < 'a' || r > 'z') {
			return "", false
		}
	}
	if arch == "gfx" {
		return "", false
	}
	return arch, true
}

// deviceNameCache is where a resolved device name is remembered between boots,
// so the expensive probe runs once per image rather than once per restart.
func deviceNameCache(dataDir string) string {
	return filepath.Join(dataDir, "config", "device-name")
}

// The cache records the architecture the name was resolved under, as
// "<gfx target> <device name>". The data directory outlives the image, and the
// name is a property of the build rather than of the machine: the same card
// reports differently depending on which patches the image was built with, and
// rebuilding for a different GPU_ARCH changes it outright. Without the arch
// there is no way to tell a still-valid cache from one the build has
// invalidated, and a stale name is the silent failure the probe exists to
// prevent — tuned kernel configs are keyed by it.
//
// A file without the arch prefix is from before this was recorded and is
// treated as stale, which costs exactly one re-probe.
func readCachedDeviceName(dataDir, gpuArch string) string {
	b, err := os.ReadFile(deviceNameCache(dataDir))
	if err != nil {
		return ""
	}
	arch, name, ok := strings.Cut(strings.TrimSpace(string(b)), " ")
	if !ok || arch != gpuArch || name == "" {
		return ""
	}
	return name
}

func writeCachedDeviceName(dataDir, gpuArch, name string) error {
	return os.WriteFile(deviceNameCache(dataDir), []byte(gpuArch+" "+name+"\n"), 0o644)
}

// resolveDeviceName asks the running vLLM what it calls this GPU.
//
// The name cannot be derived from the gfx target. On a gfx1201 build the
// generic image patches get_device_name to return "AMD-gfx1201", while the
// radiance image leaves it reporting "AMD_Radeon_R9700"; on any other target
// no patch applies and vLLM reports whatever the driver says. Tuning against
// the wrong one writes correctly-formatted JSON that vLLM never reads,
// silently.
//
// The result is handed to the tuner (which guards it with its own mutex) and
// written to a cache file rather than back into the shared Config: the HTTP
// handlers mutate that config, and this runs concurrently with them.
func resolveDeviceName(cfg *config.Config, env vllmenv.Env, srv *api.Server) {
	// Nothing to ask, or the answer is already known from config/cache.
	if cfg.VLLMDeviceName != "" || env.VenvRoot == "" {
		return
	}

	fallback := cfg.DeviceNameSuffix()
	name, err := env.ProbeDeviceName(3 * time.Minute)
	if err != nil {
		slog.Warn("could not probe vLLM device name; keeping the "+
			"architecture-derived one", "fallback", fallback, "error", err)
		return
	}
	if name == fallback {
		slog.Info("vLLM device name confirmed", "device", name)
		return
	}

	slog.Info("vLLM device name resolved", "device", name, "was", fallback)
	srv.SetDeviceName(name)

	if err := writeCachedDeviceName(cfg.DataDir, cfg.GPUArch, name); err != nil {
		slog.Warn("could not cache device name", "error", err)
	}
	// Tuned configs are keyed by device name, so the corrected name may make
	// previously-unlinkable results linkable.
	installTuned(cfg.DataDir, name, env)
}

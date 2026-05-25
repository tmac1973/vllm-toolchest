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
	"syscall"

	"github.com/tmac1973/vllm-toolchest/internal/api"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/tuning"
)

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

	// Hot-link any operator-tuned kernel configs into vLLM's site-packages
	// so vLLM picks them up on next launch. No-op when nothing's been tuned
	// or when running outside the container.
	if dev := cfg.DeviceNameSuffix(); dev != "" {
		n, warn, err := tuning.InstallTunedConfigs(cfg.DataDir, dev)
		if err != nil {
			slog.Warn("install tuned kernel configs", "error", err)
		} else if n > 0 {
			slog.Info("installed tuned kernel configs", "count", n, "warnings", warn)
		}
	}

	srv := api.NewServer(cfg)

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

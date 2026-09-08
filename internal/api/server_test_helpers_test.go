package api

import (
	"net/url"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
)

// newTestServer constructs a minimal *Server suitable for testing the
// proxy + benchmark store wiring. backendURL is parsed for host:port so
// the proxy targets the test backend.
func newTestServer(t *testing.T, backendURL string) *Server {
	t.Helper()
	u, err := url.Parse(backendURL)
	if err != nil {
		t.Fatal(err)
	}
	port, _ := strconv.Atoi(u.Port())

	dir := t.TempDir()
	cfg := &config.Config{
		DataDir:  dir,
		VLLMHost: u.Hostname(),
		VLLMPort: port,
	}
	// Pre-create the config subdirectory the Store writes into.
	_ = filepath.Join(dir, "config")

	s := &Server{
		cfg:      cfg,
		registry: models.NewRegistry(dir, filepath.Join(dir, "models")),
		monitor:  monitor.New(0),
	}
	s.bench = benchmark.NewStore(dir)
	s.benchSvc = benchmark.NewService(s.bench)
	return s
}

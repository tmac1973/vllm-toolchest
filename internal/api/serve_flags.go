package api

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// serveFlagsProbeTimeout bounds the help probe. A first run on an image that
// builds kernels at import has been seen to take over a minute.
const serveFlagsProbeTimeout = 3 * time.Minute

// flagSupport says whether the installed vLLM accepts a serve flag.
//
// Unknown -- no probe has succeeded -- passes every flag, as
// validateNamedBackends passes every backend on a variant it knows nothing
// about: refusing on no information would untick every flag a card names.
type flagSupport struct {
	known bool
	set   map[string]bool
}

func (f flagSupport) Known() bool { return f.known }

// Has reports whether the image lists flag. A "=value" suffix is ignored.
func (f flagSupport) Has(flag string) bool {
	if !f.known {
		return true
	}
	if i := strings.IndexByte(flag, '='); i >= 0 {
		flag = flag[:i]
	}
	return f.set[flag]
}

func newFlagSupport(flags []string) flagSupport {
	set := make(map[string]bool, len(flags))
	for _, f := range flags {
		set[f] = true
	}
	return flagSupport{known: true, set: set}
}

// serveFlagsState holds this process's probe result. The file cache is for
// the next boot; this is what requests read.
type serveFlagsState struct {
	mu      sync.Mutex
	probing bool
	flags   []string
}

// serveFlagsCache is the file a probe result is kept in, keyed by the image
// it describes.
type serveFlagsCache struct {
	Variant        string    `json:"variant"`
	VariantVersion string    `json:"variant_version"`
	ProbedAt       time.Time `json:"probed_at"`
	Flags          []string  `json:"flags"`
}

func (s *Server) serveFlagsPath() string {
	return filepath.Join(s.cfg.DataDir, "cache", "serve-flags.json")
}

// cachedServeFlags reads the file, and returns its flags only while it
// describes this image. An image with no version stamp never matches: nothing
// would say when its list went stale.
func (s *Server) cachedServeFlags() []string {
	if s.vllmEnv.VariantVersion == "" {
		return nil
	}
	data, err := os.ReadFile(s.serveFlagsPath())
	if err != nil {
		return nil
	}
	var c serveFlagsCache
	if json.Unmarshal(data, &c) != nil {
		return nil
	}
	if c.Variant != s.vllmEnv.Variant || c.VariantVersion != s.vllmEnv.VariantVersion {
		return nil
	}
	return c.Flags
}

// serveFlags is what the installed vLLM accepts, as far as is known. It never
// probes: a request must not wait on a Python import.
func (s *Server) serveFlags() flagSupport {
	s.flagsState.mu.Lock()
	flags := s.flagsState.flags
	s.flagsState.mu.Unlock()
	if flags == nil {
		flags = s.cachedServeFlags()
	}
	if flags == nil {
		return flagSupport{}
	}
	return newFlagSupport(flags)
}

// RefreshServeFlags loads the cached list, or probes when there is none for
// this image. It is meant to run once in the background at boot. A second
// call while one is running returns at once.
//
// The probe runs whatever the engine is doing: printing help returns from
// argument parsing before any device is touched.
func (s *Server) RefreshServeFlags() {
	s.flagsState.mu.Lock()
	if s.flagsState.probing {
		s.flagsState.mu.Unlock()
		return
	}
	s.flagsState.probing = true
	s.flagsState.mu.Unlock()
	defer func() {
		s.flagsState.mu.Lock()
		s.flagsState.probing = false
		s.flagsState.mu.Unlock()
	}()

	if flags := s.cachedServeFlags(); flags != nil {
		s.setServeFlags(flags)
		return
	}
	if s.vllmEnv.VenvRoot == "" && len(s.vllmEnv.Launcher) == 0 {
		return // no vLLM here to ask
	}

	flags, err := s.vllmEnv.ProbeServeFlags(serveFlagsProbeTimeout)
	if err != nil {
		slog.Warn("could not list the serve flags this vLLM accepts; card flags will not be checked", "error", err)
		return
	}
	s.setServeFlags(flags)
	slog.Info("listed the serve flags this vLLM accepts", "count", len(flags))

	if s.vllmEnv.VariantVersion == "" {
		return
	}
	data, _ := json.MarshalIndent(serveFlagsCache{
		Variant: s.vllmEnv.Variant, VariantVersion: s.vllmEnv.VariantVersion,
		ProbedAt: time.Now().UTC(), Flags: flags,
	}, "", "  ")
	path := s.serveFlagsPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		slog.Warn("could not cache the serve flags", "error", err)
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		slog.Warn("could not cache the serve flags", "error", err)
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		slog.Warn("could not cache the serve flags", "error", err)
	}
}

func (s *Server) setServeFlags(flags []string) {
	s.flagsState.mu.Lock()
	s.flagsState.flags = flags
	s.flagsState.mu.Unlock()
}

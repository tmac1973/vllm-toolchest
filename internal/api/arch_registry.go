package api

import (
	"net/http"
	"sort"
	"sync"
)

// archRegistry is the model architectures the running image's vLLM can load,
// once the boot goroutine has read them. Known is false until then, and stays
// false when it could not: the recommendation feed then calls a model's
// architecture unchecked rather than unsupported.
type archRegistry struct {
	mu     sync.RWMutex
	archs  map[string]bool
	known  bool
	source string // "cache" or "probe"
}

// SetSupportedArchs records the architectures the image supports and where
// the list came from.
func (s *Server) SetSupportedArchs(archs []string, source string) {
	m := make(map[string]bool, len(archs))
	for _, a := range archs {
		m[a] = true
	}
	s.archs.mu.Lock()
	defer s.archs.mu.Unlock()
	s.archs.archs, s.archs.known, s.archs.source = m, true, source
}

// SupportedArchs is the set of supported architectures, and whether it has
// been read at all.
func (s *Server) SupportedArchs() (map[string]bool, bool) {
	s.archs.mu.RLock()
	defer s.archs.mu.RUnlock()
	return s.archs.archs, s.archs.known
}

// ArchSource says where the list came from: "probe", "cache" or "none".
func (s *Server) ArchSource() string {
	s.archs.mu.RLock()
	defer s.archs.mu.RUnlock()
	if s.archs.source == "" {
		return "none"
	}
	return s.archs.source
}

// handleArchRegistry reports what the image supports: the way to see the
// probe worked without reading logs.
func (s *Server) handleArchRegistry(w http.ResponseWriter, r *http.Request) {
	m, known := s.SupportedArchs()
	archs := make([]string, 0, len(m))
	for a := range m {
		archs = append(archs, a)
	}
	sort.Strings(archs)
	respondJSON(w, map[string]any{
		"known":  known,
		"count":  len(archs),
		"archs":  archs,
		"source": s.ArchSource(),
	})
}

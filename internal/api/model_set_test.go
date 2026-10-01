package api

import (
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// The models page reloads its list when this moves, so it must move when a
// model arrives or goes, and not when a config is edited: a reload throws
// away an open configure panel.
func TestModelSetMovesWithTheSetOnly(t *testing.T) {
	s := newTestServer(t, "")
	before := s.modelSet()

	if err := s.registry.Register(&models.Model{ID: "org/new", LocalPath: t.TempDir()}); err != nil {
		t.Fatal(err)
	}
	added := s.modelSet()
	if added == before {
		t.Error("a new model did not move the fingerprint")
	}

	if err := s.registry.UpdateConfig("org/new", models.VLLMConfig{MaxModelLen: 4096}); err != nil {
		t.Fatal(err)
	}
	if s.modelSet() != added {
		t.Error("a config edit moved the fingerprint")
	}

	if err := s.registry.Delete("org/new", false); err != nil {
		t.Fatal(err)
	}
	if s.modelSet() != before {
		t.Error("deleting the model did not bring it back")
	}
}

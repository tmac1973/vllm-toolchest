package api

import (
	"os"
	"path/filepath"
	"strings"
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

// A drafter in a model's own folder is named on its card, with its method and
// size, since it is not a model of its own to be listed.
func TestTheModelCardNamesABundledDrafter(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Gemma4ForConditionalGeneration"],"num_hidden_layers":2,"hidden_size":64,"num_attention_heads":4}`), 0o644)
	sub := filepath.Join(dir, "gemma-4-31B-it-speculator.eagle3")
	os.Mkdir(sub, 0o755)
	os.WriteFile(filepath.Join(sub, "config.json"), []byte(`{"architectures":["Eagle3DraftModel"],"speculators_config":{"algorithm":"eagle3"}}`), 0o644)
	os.WriteFile(filepath.Join(sub, "model.safetensors"), make([]byte, 2048), 0o644)
	if err := s.registry.Register(&models.Model{ID: "tcclaviger/gemma-4-31B-it-MXFP416-MTP", LocalPath: dir}); err != nil {
		t.Fatal(err)
	}
	for _, r := range s.modelListView().Rows {
		if r.ID == "tcclaviger/gemma-4-31B-it-MXFP416-MTP" {
			if len(r.Bundled) != 1 || !strings.HasPrefix(r.Bundled[0], "gemma-4-31B-it-speculator.eagle3 · eagle3 · ") {
				t.Errorf("bundled: %q", r.Bundled)
			}
			return
		}
	}
	t.Error("the model is not listed")
}

package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// With nothing loaded the listing is empty, not the registry.
//
// It used to fall back to every registered model, which advertised names no
// request could be served by — a client discovering one there got a 404 for
// it — and made the same endpoint answer with a different kind of identifier
// depending on whether the engine happened to be up.
func TestV1ModelsIsEmptyWhenNothingIsServed(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	if len(s.registry.List()) == 0 {
		t.Fatal("fixture registry is empty; the test would pass vacuously")
	}

	w := httptest.NewRecorder()
	s.handleV1Models(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	var body struct {
		Object string `json:"object"`
		Data   []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("response is not JSON: %v (%s)", err, w.Body.String())
	}
	if body.Object != "list" {
		t.Errorf("object = %q, want \"list\"", body.Object)
	}
	if len(body.Data) != 0 {
		t.Errorf("nothing is loaded, so nothing should be listed; got %+v", body.Data)
	}
	// An OpenAI client iterates data; it must be [] rather than null.
	if !strings.Contains(w.Body.String(), `"data":[]`) {
		t.Errorf("data should serialize as an empty array: %s", w.Body.String())
	}
}

// Every launch path names the model, so clients get the repo id rather than
// the container-local path vLLM would otherwise report.
func TestEveryLaunchPathNamesTheModel(t *testing.T) {
	m := &models.Model{
		ID:         "cyankiwi/Qwen3.5-4B-AWQ-4bit",
		LocalPath:  "/models/cyankiwi/Qwen3.5-4B-AWQ-4bit",
		VLLMConfig: models.VLLMConfig{MaxModelLen: 8192, Dtype: "auto"},
	}

	cfg := m.StartConfig()
	if cfg.ServedModelName != m.ID {
		t.Errorf("ServedModelName = %q, want the repo id %q", cfg.ServedModelName, m.ID)
	}

	args := process.BuildArgs(cfg)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--served-model-name "+m.ID) {
		t.Errorf("launch args do not name the model: %s", joined)
	}
}

// The flag is only emitted when there is a name, so a config built without a
// model does not produce a bare --served-model-name.
func TestNoServedNameFlagWithoutAName(t *testing.T) {
	args := process.BuildArgs(process.VLLMStartConfig{MaxModelLen: 8192})
	if strings.Contains(strings.Join(args, " "), "--served-model-name") {
		t.Errorf("unexpected flag: %v", args)
	}
}

// The effective-command preview has to show what a start actually runs; a
// preview missing a flag the launch passes is worse than no preview.
func TestCommandPreviewIncludesTheServedName(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	list := s.registry.List()
	if len(list) == 0 {
		t.Fatal("no fixture models")
	}
	m := list[0]

	view := s.newModelConfigView(m)
	if !strings.Contains(view.EffectiveCommand, "--served-model-name "+m.ID) {
		t.Errorf("preview omits the served name:\n%s", view.EffectiveCommand)
	}
}

// A stopped engine must not be proxied to — that is what makes the empty
// listing reachable at all.
func TestV1ModelsDoesNotProxyWhenStopped(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	if got := s.process.GetStatus().State; got == process.StateRunning {
		t.Fatalf("fixture server should not be running, got %s", got)
	}
	w := httptest.NewRecorder()
	s.handleV1Models(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if w.Code != http.StatusOK {
		t.Errorf("status %d; a stopped engine should still answer the listing", w.Code)
	}
}

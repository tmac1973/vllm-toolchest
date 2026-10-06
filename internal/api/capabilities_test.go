package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// The launch flags of the ThinkingCap config on compute, as the process
// manager reports them.
var thinkingCapArgs = []string{
	"--served-model-name", "tcclaviger/ThinkingCap-3.8-27B-PARO5",
	"--max-model-len", "131072", "--tensor-parallel-size", "4",
	"--max-num-seqs", "8", "--kv-cache-dtype", "fp8",
	"--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
	`--speculative-config={"method": "dflash", "model": "/data/models/x", "num_speculative_tokens": 7}`,
	"--override-generation-config", `{"max_tokens": 65536, "temperature": 0.6, "top_p": 0.95, "top_k": 20}`,
}

func thinkingCapModel(t *testing.T) *models.Model {
	t.Helper()
	dir := t.TempDir()
	template := `{%- if enable_thinking is undefined or enable_thinking is true %}` +
		`{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}` +
		`{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}{{ raise_exception('bad') }}{%- endif %}{%- endif %}`
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(template), 0o644); err != nil {
		t.Fatal(err)
	}
	temp, topK := 1.0, 20
	rep := 1.05
	return &models.Model{
		ID: "tcclaviger/ThinkingCap-3.8-27B-PARO5", LocalPath: dir, Enabled: true,
		HFConfig:    models.HFConfig{MaxPositionEmbeddings: 262144},
		Vision:      models.VisionMeta{IsVisionModel: true},
		GenDefaults: models.GenDefaults{Temperature: &temp, TopK: &topK, RepetitionPenalty: &rep},
		VLLMConfig:  models.VLLMConfig{MaxModelLen: 262144, MaxNumSeqs: 16},
	}
}

// roundTrip is the object as a client receives it.
func roundTrip(t *testing.T, v any) map[string]any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestCapabilitiesDescribeWhatIsBeingServed(t *testing.T) {
	caps := roundTrip(t, buildCapabilities(thinkingCapModel(t), thinkingCapArgs))

	for key, want := range map[string]any{
		"schema_version": 1.0,
		// The flag the engine was started with, not the saved config's
		// 262144: that was edited after this start and is not what is served.
		"context_size":   131072.0,
		"context_length": 262144.0,
		"parallel":       8.0,
		// vLLM does not split the window between requests.
		"context_per_request": 131072.0,
		"vision":              true,
		"tools":               true,
		"embedding":           false,
		"max_output_tokens":   65536.0,
	} {
		if !reflect.DeepEqual(caps[key], want) {
			t.Errorf("%s = %v, want %v", key, caps[key], want)
		}
	}

	if want := map[string]any{
		"supported": true, "default_enabled": true,
		"toggle": "chat_template_kwargs", "kwarg": "enable_thinking",
		"effort_levels": []any{"low", "medium", "xhigh"}, "default_effort": "xhigh",
	}; !reflect.DeepEqual(caps["reasoning"], want) {
		t.Errorf("reasoning = %v\n     want   %v", caps["reasoning"], want)
	}

	sampling := caps["sampling"].(map[string]any)
	if sampling["source"] != "override-generation-config" {
		t.Errorf("sampling source = %v", sampling["source"])
	}
	// The launch override where it speaks, the checkpoint's own
	// generation_config.json where it does not, null where neither does.
	if want := map[string]any{
		"temperature": 0.6, "top_p": 0.95, "top_k": 20.0,
		"min_p": nil, "presence_penalty": nil, "repeat_penalty": 1.05,
	}; !reflect.DeepEqual(sampling["default"], want) {
		t.Errorf("sampling default = %v\n            want   %v", sampling["default"], want)
	}
}

// Absent is spelled out. A client told nothing about a key cannot tell a
// model without the feature from a server too old to mention it.
func TestCapabilitiesSayWhatIsAbsent(t *testing.T) {
	m := &models.Model{ID: "acme/plain", LocalPath: t.TempDir(), HFConfig: models.HFConfig{MaxPositionEmbeddings: 8192}}
	caps := roundTrip(t, buildCapabilities(m, nil))

	for _, key := range []string{
		"schema_version", "context_size", "context_length", "parallel", "context_per_request",
		"vision", "tools", "embedding", "reasoning", "sampling", "max_output_tokens",
	} {
		if _, present := caps[key]; !present {
			t.Errorf("%s is missing from the object", key)
		}
	}
	// No --max-model-len means the engine serves the trained maximum.
	if caps["context_size"] != 8192.0 || caps["parallel"] != 1.0 || caps["tools"] != false || caps["max_output_tokens"] != nil {
		t.Errorf("defaults = %v", caps)
	}
	reasoning := caps["reasoning"].(map[string]any)
	if reasoning["supported"] != false || reasoning["toggle"] != "none" || reasoning["effort_levels"] != nil || reasoning["default_effort"] != nil {
		t.Errorf("reasoning = %v", reasoning)
	}
	sampling := caps["sampling"].(map[string]any)
	if sampling["source"] != nil || len(sampling["presets"].([]any)) != 0 {
		t.Errorf("sampling = %v", sampling)
	}
}

func TestVisionAndToolsFollowTheLaunchFlags(t *testing.T) {
	m := thinkingCapModel(t)
	caps := buildCapabilities(m, []string{"--language-model-only"})
	if caps["vision"] != false {
		t.Error("a vision model served text-only was reported as taking images")
	}
	if caps["tools"] != false {
		t.Error("tools reported on a serve that does not parse tool calls")
	}
	// The saved figure, when the flag was left to the engine.
	if caps["parallel"] != 16 {
		t.Errorf("parallel = %v, want the saved 16", caps["parallel"])
	}
}

func loadedModels(t *testing.T, s *Server) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/service/loaded-models", nil)
	rec := httptest.NewRecorder()
	s.handleLoadedModels(rec, req)
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, rec.Body.String())
	}
	return out
}

// One model or none: what /v1 will answer for. The rest of the registry is
// not listed, because a request naming any of it fails.
func TestLoadedModelsListsOnlyWhatIsServed(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	mgr := process.NewManager("127.0.0.1", 0, 0)
	s.process = mgr
	m := thinkingCapModel(t)
	for _, reg := range []*models.Model{m, {ID: "acme/other", LocalPath: t.TempDir(), Enabled: true}} {
		if err := s.registry.Register(reg); err != nil {
			t.Fatal(err)
		}
	}

	out := loadedModels(t, s)
	if out["running"] != false || len(out["models"].([]any)) != 0 || out["schema_version"] != 1.0 {
		t.Errorf("with nothing running: %v", out)
	}

	// A fake engine that says it is up and then stays up.
	fake := testutil.WriteScript(t, "echo 'INFO:     Application startup complete.'\nsleep 60\n")
	mgr.SetLauncher(process.Launcher{Bin: fake})
	if err := mgr.Start(m.ID, "", thinkingCapArgs, nil); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mgr.Stop() })
	testutil.Eventually(t, 5*time.Second, func() bool { return mgr.GetStatus().State == process.StateRunning },
		"the fake engine never came up")

	out = loadedModels(t, s)
	list := out["models"].([]any)
	if out["running"] != true || len(list) != 1 {
		t.Fatalf("with one model running: %v", out)
	}
	entry := list[0].(map[string]any)
	if entry["id"] != m.ID || entry["status"] != "loaded" || entry["public_name"] != m.ID {
		t.Errorf("entry = %v", entry)
	}
	if caps := entry["capabilities"].(map[string]any); caps["context_per_request"] != 131072.0 {
		t.Errorf("capabilities = %v", caps)
	}
}

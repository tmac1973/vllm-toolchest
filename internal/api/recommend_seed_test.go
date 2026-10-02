package api

import (
	"context"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/recommend"
)

const seedConfig = `{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":4096,"num_hidden_layers":36,
"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960,"vocab_size":151936,"intermediate_size":12288}`

type seedHub struct{}

func (seedHub) Candidates(_ context.Context, q huggingface.CandidateQuery) ([]huggingface.ModelSearchResult, error) {
	if q.Tag != "" {
		return nil, nil
	}
	return []huggingface.ModelSearchResult{{
		ID: "org/seedme", Downloads: 1000, SHA: "abc", PipelineTag: "text-generation",
		Config:      &huggingface.ModelConfigMeta{Architectures: []string{"Qwen3ForCausalLM"}},
		Safetensors: &huggingface.Safetensors{Parameters: map[string]int64{"BF16": 8e9}},
		QuantFormat: huggingface.FormatFP16,
	}}, nil
}
func (seedHub) FetchConfigJSON(context.Context, string, string) ([]byte, error) {
	return []byte(seedConfig), nil
}
func (seedHub) GetFiles(_ context.Context, _, rev string) (string, []huggingface.ModelFile, error) {
	return rev, []huggingface.ModelFile{{Filename: "model.safetensors", Size: 16 << 30, Category: "weight"}}, nil
}

func downloadedDir(t *testing.T) string {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(seedConfig), 0o644)
	os.WriteFile(filepath.Join(dir, "model.safetensors"), make([]byte, 1024), 0o644)
	return dir
}

// A model downloaded while the feed lists it gets the hardware settings the
// planner gives for it here, marked as seeded; one the feed does not list, or
// one already registered, is left as it was.
func TestADownloadFromTheFeedIsSeeded(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.gpuInvOverride = &models.GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}
	s.recommend.engine = recommend.NewEngine(seedHub{}, t.TempDir())
	if r := s.recommendEngine().Result(context.Background(), s.recommendProfile(), "quality"); len(r.Verified) != 1 {
		t.Fatalf("pool: %+v", r)
	}

	s.onTransferComplete("dl1", "org/seedme", downloadedDir(t))
	m, ok := s.registry.Get("org/seedme")
	if !ok || m.ConfigSource != models.ConfigSeeded {
		t.Fatalf("not seeded: %+v", m)
	}
	// The same as the planner called directly, on the default config.
	d := models.Describe(m.ID, m.LocalPath)
	d.TotalSizeBytes = m.TotalSizeBytes
	plan := models.PlanFit(s.planInput(d, d.VLLMConfig, models.ContextMax, ""))
	want := recommend.SeedConfig(d.VLLMConfig, plan.All.Config)
	if m.VLLMConfig != want {
		t.Errorf("seeded\n%+v\nwant\n%+v", m.VLLMConfig, want)
	}
	if m.VLLMConfig.MaxModelLen != 40960 || m.VLLMConfig.TensorParallelSize < 1 {
		t.Errorf("config %+v", m.VLLMConfig)
	}

	// The config panel says where the values came from.
	s.router = s.buildRouter()
	rec := httptest.NewRecorder()
	s.router.ServeHTTP(rec, httptest.NewRequest("GET", "/api/models/config-panel?id=org/seedme", nil))
	if !strings.Contains(rec.Body.String(), "suggested for this machine when the model was downloaded") {
		t.Error("the config panel does not say the settings were suggested")
	}

	// Not listed by the feed: defaults, unmarked.
	s.onTransferComplete("dl2", "org/elsewhere", downloadedDir(t))
	if m, _ := s.registry.Get("org/elsewhere"); m.ConfigSource != "" || m.VLLMConfig.MaxModelLen != 8192 {
		t.Errorf("not from the feed: %q %d", m.ConfigSource, m.VLLMConfig.MaxModelLen)
	}

	// A re-download of a tuned model leaves the tuning alone.
	cfg := m.VLLMConfig
	cfg.MaxModelLen = 12345
	s.registry.UpdateConfig("org/seedme", cfg)
	s.onTransferComplete("dl3", "org/seedme", m.LocalPath)
	if m, _ := s.registry.Get("org/seedme"); m.VLLMConfig.MaxModelLen != 12345 || m.ConfigSource != "" {
		t.Errorf("a re-download changed a tuned config: %d %q", m.VLLMConfig.MaxModelLen, m.ConfigSource)
	}
}

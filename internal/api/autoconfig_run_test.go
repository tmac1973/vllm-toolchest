package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// autoconfigServer is a server with a scripted engine, a fake Hub serving the
// 27B's card, a fake helper answering on the engine's port, a model serving,
// and the helper installed.
func autoconfigServer(t *testing.T) (*Server, *atomic.Int32) {
	t.Helper()
	card, err := os.ReadFile("../autoconfig/testdata/thinkingcap-27b-paro5.md")
	if err != nil {
		t.Fatal(err)
	}
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/org/model/raw/main/README.md":
			w.Write(card)
		case "/api/models/org/model":
			w.Write([]byte(`{"cardData": {}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(hub.Close)

	var asked atomic.Int32
	helper := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked.Add(1)
		content := `{"command_index": 1, "other_notes": ["Serve with the latest image."]}`
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{
			"message": map[string]any{"content": content}, "finish_reason": "stop"}}})
	}))
	t.Cleanup(helper.Close)

	s := leaseServer(t) // scripted engine, org/served registered
	// The loan's work talks to the engine's port: point it at the fake helper.
	hs := newTestServer(t, helper.URL)
	s.cfg.VLLMHost, s.cfg.VLLMPort = hs.cfg.VLLMHost, hs.cfg.VLLMPort
	s.hfClient = huggingface.NewClient("")
	s.hfClient.SetBaseURL(hub.URL)

	s.registry.Register(&models.Model{
		ID: "org/model", DisplayName: "ThinkingCap", LocalPath: "/models/org/model",
		TotalSizeBytes: 25_247_995_844,
		HFConfig: models.HFConfig{
			NumHiddenLayers: 64, HiddenSize: 5120, IntermediateSize: 17408, VocabSize: 248320,
			NumAttentionHeads: 24, NumKeyValueHeads: 4, HeadDim: 256, AttentionLayers: 16,
			MaxPositionEmbeddings: 262144,
		},
		VLLMConfig: models.VLLMConfig{MaxModelLen: 8192, TensorParallelSize: 1, MaxNumSeqs: 16},
	})
	s.registry.Register(&models.Model{ID: models.HelperRepo, LocalPath: "/models/helper", Helper: true})
	s.gpuInvOverride = &models.GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}
	return s, &asked
}

func waitForRun(t *testing.T, s *Server) autoconfigRun {
	t.Helper()
	for i := 0; i < 500; i++ {
		if run, ok := s.autoconfigSnapshot(); ok && run.done {
			return run
		}
		time.Sleep(20 * time.Millisecond)
	}
	run, _ := s.autoconfigSnapshot()
	t.Fatalf("the run never finished; last progress %q", run.progress)
	return autoconfigRun{}
}

func TestAutoconfigureInterruptsAndRestores(t *testing.T) {
	s, asked := autoconfigServer(t)
	serve(t, s, "org/served")

	if err := s.startAutoconfig("org/model", models.ContextMax, false); err != nil {
		t.Fatal(err)
	}
	if err := s.startAutoconfig("org/model", models.ContextMax, false); err == nil {
		t.Error("a second run started while one was running")
	}
	run := waitForRun(t, s)
	if run.err != nil {
		t.Fatal(run.err)
	}
	res := run.result
	if asked.Load() != 1 || res.AdviceFrom != "helper" {
		t.Errorf("asked=%d from=%q", asked.Load(), res.AdviceFrom)
	}
	if !res.Plan.Known || res.Plan.All.TP != 4 || res.Plan.All.Config.KVCacheDtype != "fp8" {
		t.Errorf("plan: %+v", res.Plan.All)
	}
	waitFor(t, func() bool {
		st := s.process.GetStatus()
		return st.ModelID == "org/served" && st.State == process.StateRunning
	})

	// A second run on the unchanged card reuses the reading, and never
	// touches the engine.
	pid := s.process.GetStatus().PID
	s.clearAutoconfigRun("org/model")
	s.startAutoconfig("org/model", models.ContextMax, false)
	run = waitForRun(t, s)
	if run.result.AdviceFrom != "previous" || asked.Load() != 1 || s.process.GetStatus().PID != pid {
		t.Errorf("from=%q asked=%d; the engine was touched for a card already read", run.result.AdviceFrom, asked.Load())
	}
}

func TestAutoconfigureWhileTheEngineIsBusy(t *testing.T) {
	s, asked := autoconfigServer(t)
	s.lease.take("someone else")
	defer s.lease.release()

	if err := s.startAutoconfig("org/model", models.ContextLong, false); err != nil {
		t.Fatal(err)
	}
	run := waitForRun(t, s)
	if run.err != nil || asked.Load() != 0 {
		t.Fatalf("err=%v asked=%d", run.err, asked.Load())
	}
	found := false
	for _, n := range run.result.Notes {
		found = found || strings.Contains(n.Reason, "Autoconfigure is reading a model card")
	}
	if !found || len(run.result.Rows) == 0 {
		t.Errorf("the busy reason is missing, or the command was not used: %+v", run.result.Notes)
	}
}

func TestAutoconfigureRefuses(t *testing.T) {
	s, _ := autoconfigServer(t)
	s.registry.Register(&models.Model{ID: "org/draft", HFConfig: models.HFConfig{Draft: &models.DraftMeta{Method: "dflash"}}})
	for _, id := range []string{"org/missing", "org/draft", models.HelperRepo} {
		if err := s.startAutoconfig(id, models.ContextMax, false); err == nil {
			t.Errorf("%s: a run started", id)
		}
	}
}

func TestAutoconfigureWithoutAHelper(t *testing.T) {
	s, asked := autoconfigServer(t)
	s.registry.Delete(models.HelperRepo, false)
	s.startAutoconfig("org/model", models.ContextMax, false)
	run := waitForRun(t, s)
	if asked.Load() != 0 || run.result.AdviceFrom != "" {
		t.Error("a helper was asked with none installed")
	}
	found := false
	for _, n := range run.result.Notes {
		found = found || strings.Contains(n.Reason, "No helper model is installed")
	}
	if !found {
		t.Error("no note saying why the text was not read")
	}
}

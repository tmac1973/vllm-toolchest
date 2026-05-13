package benchmark

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/monitor"
)

func TestExpandCellsIsModelGrouped(t *testing.T) {
	cells := ExpandCells([]string{"m1", "m2", "m3"}, []string{"p1", "p2"})
	// Expect: m1/p1, m1/p2, m2/p1, m2/p2, m3/p1, m3/p2
	want := []struct{ model, preset string }{
		{"m1", "p1"}, {"m1", "p2"},
		{"m2", "p1"}, {"m2", "p2"},
		{"m3", "p1"}, {"m3", "p2"},
	}
	if len(cells) != len(want) {
		t.Fatalf("expected %d cells, got %d", len(want), len(cells))
	}
	for i, w := range want {
		if cells[i].ModelID != w.model || cells[i].Preset != w.preset {
			t.Errorf("cell[%d]: want %s/%s, got %s/%s", i, w.model, w.preset, cells[i].ModelID, cells[i].Preset)
		}
		if cells[i].Status != CellStatusPending {
			t.Errorf("cell[%d].Status: want pending, got %s", i, cells[i].Status)
		}
	}
}

func TestApplyOverridesNil(t *testing.T) {
	base := ConfigSnapshot{MaxModelLen: 8192, Dtype: "auto"}
	got := applyOverrides(base, nil)
	if got != base {
		t.Errorf("nil overrides should return base unchanged; got %+v", got)
	}
}

func TestApplyOverridesPartial(t *testing.T) {
	base := ConfigSnapshot{
		MaxModelLen:          8192,
		TensorParallelSize:   1,
		GPUMemoryUtilization: 0.90,
		Dtype:                "auto",
	}
	newMax := 4096
	newDtype := "bfloat16"
	got := applyOverrides(base, &ConfigOverrides{
		MaxModelLen: &newMax,
		Dtype:       &newDtype,
	})

	if got.MaxModelLen != 4096 {
		t.Errorf("MaxModelLen: want 4096, got %d", got.MaxModelLen)
	}
	if got.Dtype != "bfloat16" {
		t.Errorf("Dtype: want bfloat16, got %s", got.Dtype)
	}
	// Untouched fields stay at base values.
	if got.TensorParallelSize != 1 {
		t.Errorf("TP should be unchanged at 1; got %d", got.TensorParallelSize)
	}
	if got.GPUMemoryUtilization != 0.90 {
		t.Errorf("gpu_mem_util should be unchanged at 0.90; got %f", got.GPUMemoryUtilization)
	}
}

func TestJobFinalStatus(t *testing.T) {
	cases := []struct {
		name      string
		cells     []JobCell
		cancelled bool
		want      string
	}{
		{
			name:  "all completed",
			cells: []JobCell{{Status: CellStatusCompleted}, {Status: CellStatusCompleted}},
			want:  JobStatusCompleted,
		},
		{
			name:  "any failed",
			cells: []JobCell{{Status: CellStatusCompleted}, {Status: CellStatusFailed}},
			want:  JobStatusFailed,
		},
		{
			name:      "cancelled wins",
			cells:     []JobCell{{Status: CellStatusCompleted}, {Status: CellStatusCompleted}},
			cancelled: true,
			want:      JobStatusCanceled,
		},
		{
			name:  "all skipped",
			cells: []JobCell{{Status: CellStatusSkipped}, {Status: CellStatusSkipped}},
			want:  JobStatusCanceled,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			job := &BenchmarkJob{Cells: tc.cells}
			got := jobFinalStatus(job, tc.cancelled)
			if got != tc.want {
				t.Errorf("want %s, got %s", tc.want, got)
			}
		})
	}
}

// fakeJobEnv tracks how many times each model was loaded so the test can
// verify the model-grouped pass loads each model exactly once.
type fakeJobEnv struct {
	mu          sync.Mutex
	loaded      string
	loadCounts  map[string]*int32
	vllmURL     string
	models      map[string]ModelInfo
	vllmVersion string
}

func newFakeJobEnv(url string) *fakeJobEnv {
	return &fakeJobEnv{
		loadCounts: make(map[string]*int32),
		vllmURL:    url,
		models: map[string]ModelInfo{
			"a": {HFRepoID: "a", DisplayName: "Model A", ServedName: "a", Config: ConfigSnapshot{MaxModelLen: 4096}},
			"b": {HFRepoID: "b", DisplayName: "Model B", ServedName: "b", Config: ConfigSnapshot{MaxModelLen: 4096}},
		},
	}
}

func (f *fakeJobEnv) ResolveModel(id string) (ModelInfo, error) {
	m, ok := f.models[id]
	if !ok {
		return ModelInfo{}, fmt.Errorf("unknown model: %s", id)
	}
	return m, nil
}

func (f *fakeJobEnv) CurrentLoadedModel() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.loaded
}

func (f *fakeJobEnv) EnsureModelLoaded(ctx context.Context, modelID string, cfg ConfigSnapshot) error {
	f.mu.Lock()
	if f.loaded == modelID {
		f.mu.Unlock()
		return nil
	}
	f.loaded = modelID
	counter, ok := f.loadCounts[modelID]
	if !ok {
		var c int32
		counter = &c
		f.loadCounts[modelID] = counter
	}
	f.mu.Unlock()
	atomic.AddInt32(counter, 1)
	return nil
}

func (f *fakeJobEnv) CurrentMetrics() monitor.Metrics { return monitor.Metrics{} }
func (f *fakeJobEnv) VLLMURL() string                 { return f.vllmURL }
func (f *fakeJobEnv) HFToken() string                 { return "" }
func (f *fakeJobEnv) HFCacheDir() string              { return "" }
func (f *fakeJobEnv) VLLMVersion() string             { return f.vllmVersion }

func (f *fakeJobEnv) loadCount(model string) int32 {
	f.mu.Lock()
	defer f.mu.Unlock()
	c, ok := f.loadCounts[model]
	if !ok {
		return 0
	}
	return atomic.LoadInt32(c)
}

func TestJobLoadsEachModelExactlyOnce(t *testing.T) {
	fv := &fakeVLLM{
		firstDelay: 5 * time.Millisecond, followDelay: 1 * time.Millisecond,
		numChunks: 4, usagePrompt: 32, usageGen: 8,
	}
	srv := httptest.NewServer(fv.handler())
	defer srv.Close()

	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	store := NewStore(dir)
	env := newFakeJobEnv(srv.URL)
	svc := NewService(store)
	svc.SetJobEnv(env)

	// Two models × two presets = four cells, ordered: a/quick, a/std, b/quick, b/std
	job := BenchmarkJob{
		ID:        "test-job",
		Name:      "test",
		Kind:      JobKindBatch,
		Status:    JobStatusPending,
		CreatedAt: time.Now(),
		ModelIDs:  []string{"a", "b"},
		Presets:   []string{"internal-quick", "internal-quick"}, // duplicate preset OK
		Cells:     ExpandCells([]string{"a", "b"}, []string{"internal-quick", "internal-quick"}),
	}
	if err := store.SaveJob(job); err != nil {
		t.Fatal(err)
	}
	if err := svc.SubmitJob(job); err != nil {
		t.Fatal(err)
	}

	// Poll until the job clears the active slot.
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if _, busy := svc.ActiveJobID(); !busy {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if _, busy := svc.ActiveJobID(); busy {
		t.Fatal("job never finished within 15s")
	}

	if env.loadCount("a") != 1 {
		t.Errorf("model a should load once; got %d", env.loadCount("a"))
	}
	if env.loadCount("b") != 1 {
		t.Errorf("model b should load once; got %d", env.loadCount("b"))
	}

	final, err := store.GetJob("test-job")
	if err != nil {
		t.Fatal(err)
	}
	if final.Status != JobStatusCompleted {
		t.Errorf("expected job completed, got %s", final.Status)
	}
	for i, c := range final.Cells {
		if c.Status != CellStatusCompleted {
			t.Errorf("cell[%d] not completed: status=%s error=%q", i, c.Status, c.Error)
		}
		if c.BenchmarkRunID == "" {
			t.Errorf("cell[%d] has no run id", i)
		}
	}
	if got := len(store.RunsForJob("test-job")); got != 4 {
		t.Errorf("expected 4 runs persisted, got %d", got)
	}
}

func TestSubmitJobRejectsWhenAdHocRunActive(t *testing.T) {
	// Slow fake so the ad-hoc run stays active long enough to test the rejection.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
		fmt.Fprint(w, "data: {\"choices\":[{}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	store := NewStore(dir)
	svc := NewService(store)
	svc.SetJobEnv(newFakeJobEnv(srv.URL))

	run := BenchmarkRun{ID: "r1", Status: StatusRunning, CreatedAt: time.Now()}
	store.Save(run)

	if err := svc.StartRun(RunnerConfig{
		Run: run,
		Preset: Preset{Source: PresetSourceInternal,
			PromptTokens: []int{32}, GenTokens: 8, Repetitions: 1},
		VLLMURL: srv.URL, ServedName: "m", MaxModelLen: 4096,
	}); err != nil {
		t.Fatal(err)
	}
	defer svc.CancelRun("r1")

	// Job submit should be rejected while the run is active.
	err := svc.SubmitJob(BenchmarkJob{ID: "jx", Cells: []JobCell{{ModelID: "a", Preset: "internal-quick"}}})
	if err == nil {
		t.Fatal("expected ErrRunAlreadyActive, got nil")
	}
}

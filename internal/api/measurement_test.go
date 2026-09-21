package api

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// The lines a successful start really prints, taken from
// Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ on four R9700s, 2026-09-17.
const realStartOutput = `INFO [worker.py:222] PLE offload: locked 38.8 GiB of PLE weights in RAM
INFO [model_runner.py:415] Model loading took 19.07 GiB memory and 42.140533 seconds
INFO [gpu_worker.py:701] Available KV cache memory: 4.87 GiB
INFO [kv_cache_utils.py:2393] GPU KV cache size: 651,859 tokens, Maximum concurrency for 262,144 tokens per request: 2.49x
INFO [gpu_worker.py:935] Free memory on device (31.23/31.86 GiB) on startup. Desired GPU memory utilization is (0.97, 30.9 GiB). Actual usage is 24.57 GiB for consumed memory (weights + non-torch), 1.46 GiB for peak activation, and 0.49 GiB for CUDAGraph memory.
INFO [gpu_worker.py:872] CUDA graph pool memory: 0.49 GiB (actual), 0.4 GiB (estimated), difference: 0.09 GiB (18.6%).`

// startFakeEngine runs a process that prints the given output and then waits,
// so the manager observes it exactly as it would a real engine.
func startFakeEngine(t *testing.T, mgr *process.Manager, modelID, output string) {
	t.Helper()
	dir := t.TempDir()

	data := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(data, []byte(output+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	fake := filepath.Join(dir, "fakevllm")
	if err := os.WriteFile(fake, []byte("#!/bin/sh\ncat "+data+"\nsleep 60\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	mgr.SetLauncher(process.Launcher{Bin: fake})
	if err := mgr.Start(modelID, "", nil, nil); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() { _ = mgr.Stop() })
}

func testServerWithEngine(t *testing.T, modelID, output string, cfg models.VLLMConfig) *Server {
	t.Helper()
	s := newTestServer(t, "http://127.0.0.1:1")
	mgr := process.NewManager("127.0.0.1", 0, 0)
	s.process = mgr

	if err := s.registry.Register(&models.Model{
		ID:             modelID,
		DisplayName:    modelID,
		LocalPath:      "/models/" + modelID,
		TotalSizeBytes: 116_549_178_737,
		Quantization:   models.QuantMeta{Method: "compressed-tensors", BytesPerParam: 0.5625},
		VLLMConfig:     cfg,
	}); err != nil {
		t.Fatal(err)
	}

	startFakeEngine(t, mgr, modelID, output)
	return s
}

// waitForObserved gives the manager's scanner a moment to read the output.
func waitForObserved(t *testing.T, s *Server, want func() bool) {
	t.Helper()
	for i := 0; i < 200; i++ {
		if want() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("the engine's output was never observed: %+v", s.process.Measured())
}

// End to end: the engine prints, the manager observes, the registry keeps it.
func TestASuccessfulStartIsRecorded(t *testing.T) {
	const id = "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"
	cfg := models.VLLMConfig{
		TensorParallelSize: 4, MaxModelLen: 262144,
		KVCacheDtype: "fp8", MaxNumBatchedTokens: 8192,
		Env: "VLLM_PLE_CPU_OFFLOAD=1",
	}
	s := testServerWithEngine(t, id, realStartOutput, cfg)
	waitForObserved(t, s, func() bool { return s.process.Measured().KVCacheTokens != 0 })

	s.recordMeasurement(id)

	m, ok := s.registry.Get(id)
	if !ok {
		t.Fatal("model vanished from the registry")
	}
	if m.Measured == nil {
		t.Fatal("a complete start recorded nothing")
	}

	got := m.Measured
	if got.TP != 4 || got.ContextTokens != 262144 {
		t.Errorf("recorded TP=%d ctx=%d, want 4 and 262144", got.TP, got.ContextTokens)
	}
	if got.Engine.WeightsPerRankGB != 19.07 || got.Engine.ConsumedGB != 24.57 {
		t.Errorf("weights/consumed = %.2f/%.2f, want 19.07/24.57",
			got.Engine.WeightsPerRankGB, got.Engine.ConsumedGB)
	}
	if got.Engine.PLEOffloadGB != 38.8 {
		t.Errorf("PLE offload = %v, want 38.8", got.Engine.PLEOffloadGB)
	}
	// The figure the whole approach exists for: counted from the architecture
	// it was 12,288, and two real starts measured ~32,100.
	if b := got.KVBytesPerToken(); b < 32000 || b > 32200 {
		t.Errorf("KV bytes/token = %.0f, want ~32087", b)
	}
	// And it must be tied to the configuration it was taken under.
	if got.Fingerprint != models.MeasurementFingerprint(m) {
		t.Error("the measurement was stored without a matching fingerprint")
	}
	if !got.Applies(m, models.EngineIdentity{}) {
		t.Error("a measurement of this very configuration does not apply to it")
	}
}

// A start that dies during weight loading has one figure and no pool. Storing
// that would put a confident number on screen that no run produced.
func TestAPartialStartIsNotRecorded(t *testing.T) {
	const id = "test/dies-early"
	partial := `INFO [model_runner.py:415] Model loading took 19.07 GiB memory and 42.1 seconds
INFO [worker.py:222] PLE offload: locked 38.8 GiB of PLE weights in RAM`

	s := testServerWithEngine(t, id, partial, models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 8192})
	waitForObserved(t, s, func() bool { return s.process.Measured().WeightsPerRankGB != 0 })

	s.recordMeasurement(id)

	m, _ := s.registry.Get(id)
	if m.Measured != nil {
		t.Errorf("an incomplete start was recorded: %+v", m.Measured)
	}
}

// Changing the configuration retires the measurement rather than silently
// re-using it against settings it never described.
func TestAConfigChangeRetiresTheMeasurement(t *testing.T) {
	const id = "test/reconfigured"
	cfg := models.VLLMConfig{
		TensorParallelSize: 4, MaxModelLen: 262144,
		KVCacheDtype: "fp8", MaxNumBatchedTokens: 8192,
	}
	s := testServerWithEngine(t, id, realStartOutput, cfg)
	waitForObserved(t, s, func() bool { return s.process.Measured().KVCacheTokens != 0 })
	s.recordMeasurement(id)

	m, _ := s.registry.Get(id)
	if m.Measured == nil || !m.Measured.Applies(m, models.EngineIdentity{}) {
		t.Fatal("nothing recorded to retire")
	}

	// The context length is the one thing that may move freely: the KV term
	// scales with it exactly.
	cfg.MaxModelLen = 131072
	if err := s.registry.UpdateConfig(id, cfg); err != nil {
		t.Fatal(err)
	}
	m, _ = s.registry.Get(id)
	if !m.Measured.Applies(m, models.EngineIdentity{}) {
		t.Error("changing only the context length retired a measurement that still holds")
	}

	// Halving the tensor-parallel width does not.
	cfg.TensorParallelSize = 2
	if err := s.registry.UpdateConfig(id, cfg); err != nil {
		t.Fatal(err)
	}
	m, _ = s.registry.Get(id)
	if m.Measured.Applies(m, models.EngineIdentity{}) {
		t.Error("a measurement taken at TP=4 still claims to describe TP=2")
	}
}

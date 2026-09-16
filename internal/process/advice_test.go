package process

import "testing"

// The manager reads the engine's output as it streams. These drive observe
// directly rather than launching anything: the wiring being tested is the
// accumulation, not the process.
func TestManagerCapturesAdviceAndMeasurements(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)

	for _, line := range []string{
		"INFO 09-16 11:03:17 [core.py:193] init engine took 41.55 seconds",
		"INFO [gpu_model_runner.py:2653] model loading took 19.07 GiB and 172.5 seconds",
		"INFO [gpu_worker.py:298] Available KV cache memory: 10.34 GiB",
		"INFO [kv_cache_utils.py:833] Maximum concurrency for 262144 tokens per request: 2.48x",
		"vllm: error: unrecognized arguments: --enable-expert-offload",
	} {
		m.observe(line)
	}

	got := m.Advice()
	if len(got) != 1 {
		t.Fatalf("got %d advice items, want 1: %+v", len(got), got)
	}
	if got[0].Field != "extra_flags" {
		t.Errorf("field = %q, want extra_flags", got[0].Field)
	}

	measured := m.Measured()
	if measured.WeightsPerRankGB != 19.07 {
		t.Errorf("weights = %v, want 19.07", measured.WeightsPerRankGB)
	}
	if measured.KVCacheGB != 10.34 {
		t.Errorf("KV cache = %v, want 10.34", measured.KVCacheGB)
	}
	if measured.MaxConcurrency != 2.48 {
		t.Errorf("concurrency = %v, want 2.48", measured.MaxConcurrency)
	}
}

// A failing start prints the same complaint from every rank, and a
// tensor-parallel group of four should not fill the panel with four copies.
func TestManagerDeduplicatesAdvice(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	for i := 0; i < 4; i++ {
		m.observe("(Worker_TP" + string(rune('0'+i)) + ") torch.cuda.OutOfMemoryError: CUDA out of memory")
	}
	if got := m.Advice(); len(got) != 1 {
		t.Errorf("got %d items from four ranks reporting one problem", len(got))
	}
}

// Advice describing a configuration that is no longer loaded is worse than
// none, so a new run must not inherit the last one's output.
func TestStartClearsPreviousAdvice(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	m.observe("vllm: error: unrecognized arguments: --gone-in-the-next-image")
	m.observe("INFO [gpu_worker.py:298] Available KV cache memory: 10.34 GiB")

	if len(m.Advice()) == 0 || !m.Measured().Any() {
		t.Fatal("nothing captured, so the clearing is not being tested")
	}

	m.resetAdvice()

	if got := m.Advice(); len(got) != 0 {
		t.Errorf("advice survived the reset: %+v", got)
	}
	if m.Measured().Any() {
		t.Errorf("measurements survived the reset: %+v", m.Measured())
	}
}

// A pathological start can repeat a distinct warning per layer. The list is
// capped so the panel stays readable.
func TestAdviceIsCapped(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	for i := 0; i < adviceMax*2; i++ {
		// Distinct each time, so deduplication does not do the capping.
		m.observe("vllm: error: unrecognized arguments: --flag-" + string(rune('a'+i%26)) + string(rune('a'+i/26)))
	}
	if got := len(m.Advice()); got > adviceMax {
		t.Errorf("kept %d items, want no more than %d", got, adviceMax)
	}
}

// A line can be both a measurement and a piece of advice, and the partial
// offload failure is exactly that: it records how much was pinned and warns
// that the rest was not.
func TestPartialOffloadFailureIsBothAtOnce(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	m.observe("PLE offload: locked 0.0 GiB, FAILED to lock 47.7 GiB")

	if got := m.Advice(); len(got) != 1 || got[0].Field != "env" {
		t.Errorf("expected one piece of advice about the environment, got %+v", got)
	}
	measured := m.Measured()
	if !measured.PLEOffloadFailed {
		t.Error("the lock failure was not recorded as a measurement")
	}
	if measured.PLEOffloadWanted != 47.7 {
		t.Errorf("wanted = %v, want 47.7", measured.PLEOffloadWanted)
	}
}

package process

import (
	"fmt"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
)

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

// The ranks do not always agree on the number. Each measures its own KV pool,
// so a four-card start reports four byte counts differing in their last few
// digits -- and with Suggested in the dedup key those read as four separate
// pieces of advice about one thing. Seen on a real multi-GPU start.
//
// The existing dedup test cannot catch this: it feeds four *identical* lines,
// so the values match and any key collapses them.
func TestRanksDisagreeingOnANumberAreStillOneNote(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)

	const tmpl = "(Worker_TP%d) INFO [gpu_worker.py:789] Free memory on device (23.91/23.98 GiB) on startup. " +
		"Desired GPU memory utilization is (0.92, 22.07 GiB). Actual usage is 4.77 GiB for consumed memory " +
		"(weights + non-torch), 5.4 GiB for peak activation, and 3.74 GiB for CUDAGraph memory. Replace " +
		"gpu_memory_utilization config with `--kv-cache-memory=%d` (4.23 GiB) to fit into requested memory."

	// Deliberately not in ascending order: the smallest must win because it is
	// the smallest, not because it arrived last.
	for i, bytes := range []int64{4545302242, 4532719330, 4536913634} {
		m.observe(fmt.Sprintf(tmpl, i, bytes))
	}

	got := m.Advice()
	var kv []advice.Item
	for _, it := range got {
		if it.Field == "kv_cache_memory" {
			kv = append(kv, it)
		}
	}
	if len(kv) != 1 {
		t.Fatalf("three ranks reporting one pool produced %d notes: %+v", len(kv), kv)
	}

	// A per-rank pool has to fit the tightest rank, so the smallest is the
	// only value that is safe on every card.
	if kv[0].Suggested != "4532719330" {
		t.Errorf("kept %q; the smallest figure is the one that holds on every rank", kv[0].Suggested)
	}
}

// Reconciling numbers must not merge genuinely different advice, and two
// rules really do implicate max_model_len: the seq-len-vs-KV ceiling, which
// names the length that would fit, and a plain out-of-memory failure, which
// suggests lowering it among other things. They say different things about
// the same setting, so a key of field alone would silently drop one.
//
// The assertions check the premise as well as the result: pairing two rules
// with *different* fields would pass under either key and prove nothing.
func TestTwoRulesSharingAFieldStaySeparate(t *testing.T) {
	m := NewManager("127.0.0.1", 0, 0)
	m.observe("ValueError: max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache (65536)")
	m.observe("(Worker_TP0) torch.cuda.OutOfMemoryError: CUDA out of memory")

	got := m.Advice()
	if len(got) != 2 {
		t.Fatalf("got %d items, want 2 -- same field, different advice: %+v", len(got), got)
	}
	for _, it := range got {
		if it.Field != "max_model_len" {
			t.Errorf("field = %q, want max_model_len: this is no longer exercising a field collision", it.Field)
		}
	}
	if got[0].Message == got[1].Message {
		t.Error("both items carry the same message, so the separation proves nothing")
	}
}

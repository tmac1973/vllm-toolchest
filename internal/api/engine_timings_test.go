package api

import (
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// What vLLM 0.29 publishes after two short requests, as read from a real
// engine: the lines this reads, a few it must ignore, and the buckets and
// _created lines that surround them.
const engineMetricsSample = `# HELP vllm:request_prefill_time_seconds Histogram of time spent in PREFILL phase for request.
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_bucket{engine="0",le="0.3",model_name="acme/model"} 2.0
vllm:request_prefill_time_seconds_count{engine="0",model_name="acme/model"} 2.0
vllm:request_prefill_time_seconds_sum{engine="0",model_name="acme/model"} 0.177739146980457
vllm:request_prefill_time_seconds_created{engine="0",model_name="acme/model"} 1.7907703803e+09
vllm:request_decode_time_seconds_count{engine="0",model_name="acme/model"} 2.0
vllm:request_decode_time_seconds_sum{engine="0",model_name="acme/model"} 2.583450863021426
vllm:request_generation_tokens_count{engine="0",model_name="acme/model"} 2.0
vllm:request_generation_tokens_sum{engine="0",model_name="acme/model"} 500.0
vllm:request_prompt_tokens_sum{engine="0",model_name="acme/model"} 4150.0
vllm:request_prefill_kv_computed_tokens_count{engine="0",model_name="acme/model"} 2.0
vllm:request_prefill_kv_computed_tokens_sum{engine="0",model_name="acme/model"} 150.0
vllm:prompt_tokens_total{engine="0",model_name="acme/model"} 4150.0
vllm:generation_tokens_total{engine="0",model_name="acme/model"} 731.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="acme/model"} 5.9e-05
`

func TestEngineCountersAreReadFromTheMetrics(t *testing.T) {
	c, ok := parseEngineCounters(strings.NewReader(engineMetricsSample))
	if !ok {
		t.Fatal("the phase timers were not found")
	}
	want := engineCounters{
		requests: 2,
		// Computed, not submitted: 4000 of the 4150 prompt tokens were a
		// cached prefix and cost no prefill.
		promptTokens:   150,
		prefillSeconds: 0.177739146980457,
		// From the histogram observed when a request finishes, not from the
		// running counter, which is already counting tokens of requests
		// whose decode time has not been reported yet.
		genTokens:     500,
		decodeSeconds: 2.583450863021426,
	}
	if c != want {
		t.Errorf("counters = %+v\n    want   %+v", c, want)
	}
}

// An engine that predates the computed-token histogram still says how long
// the prompts were. That is exact without prefix caching and an overstatement
// with it, and either beats reporting no prefill speed.
func TestOlderEnginesFallBackToSubmittedPromptTokens(t *testing.T) {
	var older []string
	for _, line := range strings.Split(engineMetricsSample, "\n") {
		if !strings.Contains(line, "prefill_kv_computed") {
			older = append(older, line)
		}
	}
	c, ok := parseEngineCounters(strings.NewReader(strings.Join(older, "\n")))
	if !ok || c.promptTokens != 4150 {
		t.Errorf("prompt tokens = %v (ok %v), want the 4150 submitted", c.promptTokens, ok)
	}
}

func TestAnEngineWithoutPhaseTimersIsNotRead(t *testing.T) {
	if _, ok := parseEngineCounters(strings.NewReader("vllm:generation_tokens_total{engine=\"0\"} 731.0\n")); ok {
		t.Error("read timings from an engine that publishes no phase timers")
	}
}

func TestASampleIsWhatFinishedBetweenTwoScrapes(t *testing.T) {
	w := &engineTimingWatch{}

	// First sight of a process is measured against zero, which is where its
	// counters started: the requests before the first scrape still count.
	first, ok := w.observe(101, engineCounters{requests: 2, promptTokens: 150, prefillSeconds: 0.2, genTokens: 500, decodeSeconds: 2.5})
	if !ok || first.Requests != 2 || first.PromptTokens != 150 || first.GenTokens != 500 {
		t.Fatalf("first sample = %+v (ok %v)", first, ok)
	}

	// Nothing finished: a long think in progress is not a sample of zero.
	if _, ok := w.observe(101, engineCounters{requests: 2, promptTokens: 150, prefillSeconds: 0.2, genTokens: 500, decodeSeconds: 2.5}); ok {
		t.Error("an interval in which no request finished produced a sample")
	}

	// One request: a 12,000-token prompt in 3 s, 900 tokens out in 6 s.
	next, ok := w.observe(101, engineCounters{requests: 3, promptTokens: 12150, prefillSeconds: 3.2, genTokens: 1400, decodeSeconds: 8.5})
	if !ok {
		t.Fatal("no sample for a finished request")
	}
	if next.Requests != 1 || next.PromptTokens != 12000 || next.GenTokens != 900 ||
		math.Abs(next.PromptSeconds-3) > 1e-9 || math.Abs(next.GenSeconds-6) > 1e-9 {
		t.Errorf("sample = %+v, want 1 request, 12000 tokens in 3 s, 900 in 6 s", next)
	}
}

// A restarted engine counts from zero again. Measured against the old totals
// it would produce negative work, or none until it had caught up.
func TestARestartedEngineIsMeasuredFromZero(t *testing.T) {
	w := &engineTimingWatch{}
	w.observe(101, engineCounters{requests: 40, promptTokens: 9000, prefillSeconds: 3, genTokens: 20000, decodeSeconds: 100})

	s, ok := w.observe(202, engineCounters{requests: 1, promptTokens: 300, prefillSeconds: 0.1, genTokens: 200, decodeSeconds: 1})
	if !ok || s.Requests != 1 || s.PromptTokens != 300 || s.GenTokens != 200 {
		t.Errorf("sample after a restart = %+v (ok %v)", s, ok)
	}

	// The same, where the process id happens to be reused.
	w = &engineTimingWatch{}
	w.observe(101, engineCounters{requests: 40, genTokens: 20000, decodeSeconds: 100})
	if s, ok := w.observe(101, engineCounters{requests: 2, genTokens: 400, decodeSeconds: 2}); !ok || s.Requests != 2 || s.GenTokens != 400 {
		t.Errorf("sample after counters went backwards = %+v (ok %v)", s, ok)
	}
}

// End to end: a running engine, its metrics endpoint, and the panel's store.
func TestAPollRecordsWhatTheEngineReports(t *testing.T) {
	// The metrics endpoint stands in for the engine on the engine's port, so
	// it has to come up after the fake engine is started: Start refuses a
	// port something else is already listening on.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()

	s := newTestServer(t, "http://"+addr)
	mgr := process.NewManager(s.cfg.VLLMHost, s.cfg.VLLMPort, 0)
	s.process = mgr
	watch := &engineTimingWatch{}

	// Nothing running: nothing is asked of the engine and nothing recorded.
	s.pollEngineTimings(http.DefaultClient, watch)
	if n := len(s.bench.RecentTimings("acme/model", 10)); n != 0 {
		t.Fatalf("%d timings recorded with no engine running", n)
	}

	startFakeEngine(t, mgr, "acme/model", "INFO:     Application startup complete.")

	metrics := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(engineMetricsSample))
	}))
	if metrics.Listener, err = net.Listen("tcp", addr); err != nil {
		t.Fatal(err)
	}
	metrics.Start()
	defer metrics.Close()
	deadline := time.Now().Add(5 * time.Second)
	for mgr.GetStatus().State != process.StateRunning && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}

	s.pollEngineTimings(http.DefaultClient, watch)
	s.pollEngineTimings(http.DefaultClient, watch) // same totals: nothing new finished

	samples := s.bench.RecentTimings("acme/model", 10)
	if len(samples) != 1 {
		t.Fatalf("%d timings recorded, want 1", len(samples))
	}
	got := samples[0]
	if got.Requests != 2 || got.PromptTokens != 150 || got.GenTokens != 500 {
		t.Errorf("sample = %+v", got)
	}
	// 150 tokens in 0.178 s and 500 in 2.58 s: the engine's own figures,
	// neither of them diluted by the other phase.
	if math.Abs(got.PromptTokPerSec-843.9) > 0.5 || math.Abs(got.GenTokPerSec-193.5) > 0.5 {
		t.Errorf("PP = %.1f t/s, TG = %.1f t/s, want 843.9 and 193.5", got.PromptTokPerSec, got.GenTokPerSec)
	}
}

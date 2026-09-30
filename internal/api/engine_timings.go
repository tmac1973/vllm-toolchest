package api

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// The Live Performance panel reads the engine's own request timers.
//
// It used to time requests from the proxy, which could only see the ones that
// were not streaming and came through this server's port, and could only
// divide completion tokens by the whole request's wall time -- so a long
// prompt made generation look slow, and prefill could not be reported at all.
//
// The figures in vLLM's log are no better for this. "Avg prompt throughput"
// is tokens ingested during the last logging interval divided by the length
// of that interval: a throughput, and zero for any interval in which no new
// prompt arrived, which is every interval spent generating. A long think
// reads as a prefill speed of nothing.
//
// vLLM does time each phase of each request, and publishes the totals as
// Prometheus histograms when the request finishes:
//
//	request_prefill_time_seconds        time in prefill
//	request_prefill_kv_computed_tokens  prompt tokens actually computed
//	request_decode_time_seconds         time in decode
//	request_generation_tokens           tokens generated
//
// Tokens over seconds, between two scrapes, is the speed of the requests that
// finished in between -- every request, streaming or not, whichever port it
// arrived on -- with queueing and idle time in neither figure.

// engineCounters is the running totals one scrape reads.
type engineCounters struct {
	requests       float64
	promptTokens   float64
	prefillSeconds float64
	genTokens      float64
	decodeSeconds  float64
}

// engineTimingInterval is how often the engine is scraped. The panel
// refreshes every ten seconds, so this keeps it at most one refresh behind.
const engineTimingInterval = 5 * time.Second

// parseEngineCounters reads the totals out of a Prometheus exposition. ok is
// false when the engine does not publish the phase timers at all.
//
// Values are summed across label sets: there is one engine and one model, but
// nothing here depends on that.
func parseEngineCounters(r io.Reader) (engineCounters, bool) {
	var c engineCounters
	var cachedInclusiveTokens float64
	var sawTimers, sawComputed bool

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "vllm:request_") {
			continue
		}
		name, value, ok := splitMetricLine(line)
		if !ok {
			continue
		}
		switch name {
		case "vllm:request_prefill_time_seconds_count":
			c.requests += value
			sawTimers = true
		case "vllm:request_prefill_time_seconds_sum":
			c.prefillSeconds += value
		case "vllm:request_decode_time_seconds_sum":
			c.decodeSeconds += value
		case "vllm:request_generation_tokens_sum":
			c.genTokens += value
		case "vllm:request_prefill_kv_computed_tokens_sum":
			c.promptTokens += value
			sawComputed = true
		case "vllm:request_prompt_tokens_sum":
			cachedInclusiveTokens += value
		}
	}
	// An engine from before the computed-token histogram only says how long
	// the prompts were, cached prefix included. Better than nothing, and
	// exact whenever prefix caching is off.
	if !sawComputed {
		c.promptTokens = cachedInclusiveTokens
	}
	return c, sawTimers
}

// splitMetricLine takes `name{labels} value` apart.
func splitMetricLine(line string) (string, float64, bool) {
	sp := strings.LastIndexByte(line, ' ')
	if sp < 0 {
		return "", 0, false
	}
	value, err := strconv.ParseFloat(line[sp+1:], 64)
	if err != nil {
		return "", 0, false
	}
	name := line[:sp]
	if brace := strings.IndexByte(name, '{'); brace >= 0 {
		name = name[:brace]
	}
	return name, value, true
}

// engineTimingWatch remembers the last scrape, so the next one can be turned
// into what happened in between.
type engineTimingWatch struct {
	pid  int
	last engineCounters
}

// observe turns a scrape into a sample of the requests that finished since
// the previous one, or reports false when none did.
//
// The engine's counters start at zero with its process, so a scrape from a
// process not seen before is measured against zero rather than discarded:
// otherwise every request that finished before the first scrape after a start
// would be lost.
func (w *engineTimingWatch) observe(pid int, now engineCounters) (benchmark.TimingSample, bool) {
	if pid != w.pid || now.requests < w.last.requests {
		w.pid = pid
		w.last = engineCounters{}
	}
	prev := w.last
	w.last = now

	requests := int(now.requests - prev.requests + 0.5)
	if requests <= 0 {
		return benchmark.TimingSample{}, false
	}
	return benchmark.TimingSample{
		Requests:      requests,
		PromptTokens:  int(now.promptTokens - prev.promptTokens + 0.5),
		PromptSeconds: now.prefillSeconds - prev.prefillSeconds,
		GenTokens:     int(now.genTokens - prev.genTokens + 0.5),
		GenSeconds:    now.decodeSeconds - prev.decodeSeconds,
	}, true
}

// pollEngineTimings scrapes the running engine once and records what it
// finds. A stopped engine, or one that does not answer, is simply skipped.
func (s *Server) pollEngineTimings(client *http.Client, watch *engineTimingWatch) {
	st := s.process.GetStatus()
	if st.State != process.StateRunning {
		return
	}
	resp, err := client.Get(fmt.Sprintf("http://%s:%d/metrics", s.cfg.VLLMHost, s.cfg.VLLMPort))
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}
	counters, ok := parseEngineCounters(resp.Body)
	if !ok {
		return
	}
	sample, ok := watch.observe(st.PID, counters)
	if !ok {
		return
	}
	sample.ModelID = st.ModelID
	s.bench.AddTiming(sample)
}

// watchEngineTimings feeds the Live Performance panel for as long as the
// server runs.
func (s *Server) watchEngineTimings() {
	client := &http.Client{Timeout: 5 * time.Second}
	watch := &engineTimingWatch{}
	ticker := time.NewTicker(engineTimingInterval)
	defer ticker.Stop()
	for range ticker.C {
		s.pollEngineTimings(client, watch)
	}
}

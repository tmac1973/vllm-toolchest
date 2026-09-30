package benchmark

import (
	"sync"
	"time"
)

// maxTimingSamples caps the per-model ring buffer. 1000 entries × ~50
// bytes each = ~50 KB per model, which is fine even with dozens of
// loaded models. Drops the oldest sample on overflow.
const maxTimingSamples = 1000

// timingWindow is how many recent samples an average is taken over. It is a
// window rather than a decay because the average is a ratio of sums, which a
// decay cannot express -- see RunningAverage.
const timingWindow = 100

// timingMinSamplesForAverage is the threshold at which UI surfaces the
// running average. Below this we hide the badge — averaging 2 requests
// is too noisy to be meaningful.
const timingMinSamplesForAverage = 10

// TimingSample is what the engine reported for the requests that finished in
// one polling interval: how many tokens they took and how long the engine
// spent on each phase. Samples are kept in memory only (not persisted in
// benchmarks.json) — they're a live view of recent inference activity, not a
// benchmark history.
//
// The seconds are the engine's own per-request phase timers, summed. They are
// what make the two rates speeds rather than throughputs: time a request
// spent queued, or the server spent idle, is in neither.
type TimingSample struct {
	Timestamp time.Time `json:"ts"`
	ModelID   string    `json:"model"`
	// Requests is how many finished requests the sample covers.
	Requests int `json:"requests"`

	// PromptTokens is the prompt tokens actually computed. A prefix served
	// from the cache cost no prefill and is not counted.
	PromptTokens  int     `json:"prompt_n"`
	PromptSeconds float64 `json:"prompt_s"`
	GenTokens     int     `json:"gen_n"`
	GenSeconds    float64 `json:"gen_s"`

	PromptTokPerSec float64 `json:"prompt_tps"`
	GenTokPerSec    float64 `json:"gen_tps"`
}

// RunningAverage is the recent speed of one model.
//
// Each rate is tokens over seconds summed across the window, not a mean of
// per-sample rates. The difference matters most for prefill: a twenty-token
// prompt is dominated by fixed per-request overhead and reports a few hundred
// tokens a second on hardware that does several thousand, so averaging rates
// lets every trivial request drag the figure down by as much as a real one
// raises it. Summing weights each request by the work it actually was.
type RunningAverage struct {
	ModelID string `json:"model_id"`
	// Count is requests observed since the model started being tracked.
	Count        int       `json:"count"`
	AvgGenTPS    float64   `json:"avg_gen_tps"`
	AvgPromptTPS float64   `json:"avg_prompt_tps"`
	LastUpdated  time.Time `json:"last_updated"`
}

// timingStore holds per-model ring buffers and EMA running averages.
// Embedded into Store so callers see one type.
type timingStore struct {
	mu       sync.RWMutex
	ring     map[string][]TimingSample
	averages map[string]*RunningAverage
}

func newTimingStore() *timingStore {
	return &timingStore{
		ring:     make(map[string][]TimingSample),
		averages: make(map[string]*RunningAverage),
	}
}

// AddTiming records one sample. ModelID must be set; samples with zero
// GenTokens are dropped (no useful signal).
//
// A sample may state its rates or its seconds; whichever is missing is worked
// out from the other, so the average below has one thing to sum.
func (s *Store) AddTiming(sample TimingSample) {
	if sample.ModelID == "" || sample.GenTokens <= 0 {
		return
	}
	if sample.Timestamp.IsZero() {
		sample.Timestamp = time.Now()
	}
	if sample.Requests <= 0 {
		sample.Requests = 1
	}
	sample.GenSeconds, sample.GenTokPerSec = completeRate(sample.GenTokens, sample.GenSeconds, sample.GenTokPerSec)
	sample.PromptSeconds, sample.PromptTokPerSec = completeRate(sample.PromptTokens, sample.PromptSeconds, sample.PromptTokPerSec)

	s.timing.mu.Lock()
	defer s.timing.mu.Unlock()

	buf := s.timing.ring[sample.ModelID]
	buf = append(buf, sample)
	if len(buf) > maxTimingSamples {
		// Drop oldest. Keeping a ring index would be cheaper but the
		// trim cost amortizes to O(1) and the code stays simple.
		buf = buf[len(buf)-maxTimingSamples:]
	}
	s.timing.ring[sample.ModelID] = buf

	avg, ok := s.timing.averages[sample.ModelID]
	if !ok {
		avg = &RunningAverage{ModelID: sample.ModelID}
		s.timing.averages[sample.ModelID] = avg
	}
	avg.Count += sample.Requests
	avg.LastUpdated = sample.Timestamp

	window := buf
	if len(window) > timingWindow {
		window = window[len(window)-timingWindow:]
	}
	var genTok, genSec, promptTok, promptSec float64
	for _, w := range window {
		if w.GenSeconds > 0 {
			genTok += float64(w.GenTokens)
			genSec += w.GenSeconds
		}
		// A sample whose prompts were wholly cached did no prefill, and has
		// nothing to say about how fast prefill is.
		if w.PromptSeconds > 0 && w.PromptTokens > 0 {
			promptTok += float64(w.PromptTokens)
			promptSec += w.PromptSeconds
		}
	}
	avg.AvgGenTPS, avg.AvgPromptTPS = 0, 0
	if genSec > 0 {
		avg.AvgGenTPS = genTok / genSec
	}
	if promptSec > 0 {
		avg.AvgPromptTPS = promptTok / promptSec
	}
}

// completeRate fills in whichever of seconds and rate a sample left out.
func completeRate(tokens int, seconds, rate float64) (float64, float64) {
	switch {
	case tokens <= 0:
		return 0, 0
	case seconds > 0:
		return seconds, float64(tokens) / seconds
	case rate > 0:
		return float64(tokens) / rate, rate
	}
	return 0, 0
}

// RecentTimings returns the last n samples for a model, newest last.
// Returns up to all stored samples if n <= 0 or n exceeds buffer size.
func (s *Store) RecentTimings(modelID string, n int) []TimingSample {
	s.timing.mu.RLock()
	defer s.timing.mu.RUnlock()

	buf, ok := s.timing.ring[modelID]
	if !ok {
		return nil
	}
	if n <= 0 || n > len(buf) {
		n = len(buf)
	}
	out := make([]TimingSample, n)
	copy(out, buf[len(buf)-n:])
	return out
}

// RunningAverage returns the recent average for one model. The bool is false when
// the model has fewer than timingMinSamplesForAverage samples, so the
// UI knows whether to surface the badge.
func (s *Store) RunningAverage(modelID string) (RunningAverage, bool) {
	s.timing.mu.RLock()
	defer s.timing.mu.RUnlock()
	avg, ok := s.timing.averages[modelID]
	if !ok {
		return RunningAverage{}, false
	}
	out := *avg // copy
	return out, out.Count >= timingMinSamplesForAverage
}

// MinSamplesForAverage is the number of observed requests a model needs before
// its average is meaningful enough to show. Exported so the UI can say how far
// along it is rather than just withholding the row.
func MinSamplesForAverage() int { return timingMinSamplesForAverage }

// PendingAverages returns models that have samples but not yet enough for an
// average. The UI shows these as progress: "collecting, 6 of 10". Without it a
// panel that is working looks identical to one that is broken.
func (s *Store) PendingAverages() []RunningAverage {
	s.timing.mu.RLock()
	defer s.timing.mu.RUnlock()

	out := make([]RunningAverage, 0, len(s.timing.averages))
	for _, avg := range s.timing.averages {
		if avg.Count > 0 && avg.Count < timingMinSamplesForAverage {
			out = append(out, *avg)
		}
	}
	return out
}

// RunningAverages returns running averages for every model with at least
// timingMinSamplesForAverage samples.
func (s *Store) RunningAverages() []RunningAverage {
	s.timing.mu.RLock()
	defer s.timing.mu.RUnlock()

	out := make([]RunningAverage, 0, len(s.timing.averages))
	for _, avg := range s.timing.averages {
		if avg.Count >= timingMinSamplesForAverage {
			out = append(out, *avg)
		}
	}
	return out
}

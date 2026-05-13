package benchmark

import (
	"sync"
	"time"
)

// maxTimingSamples caps the per-model ring buffer. 1000 entries × ~50
// bytes each = ~50 KB per model, which is fine even with dozens of
// loaded models. Drops the oldest sample on overflow.
const maxTimingSamples = 1000

// timingEMAAlpha controls the exponential-moving-average's responsiveness.
// 0.1 means each new sample contributes 10% to the running average and
// the prior average decays at the same rate.
const timingEMAAlpha = 0.1

// timingMinSamplesForAverage is the threshold at which UI surfaces the
// running average. Below this we hide the badge — averaging 2 requests
// is too noisy to be meaningful.
const timingMinSamplesForAverage = 10

// TimingSample is one observed timing from real /v1/chat/completions
// usage, captured by the proxy. Samples are kept in memory only
// (not persisted in benchmarks.json) — they're a live view of recent
// inference activity, not a benchmark history.
type TimingSample struct {
	Timestamp       time.Time `json:"ts"`
	ModelID         string    `json:"model"`
	PromptTokens    int       `json:"prompt_n"`
	GenTokens       int       `json:"gen_n"`
	PromptTokPerSec float64   `json:"prompt_tps"`
	GenTokPerSec    float64   `json:"gen_tps"`
}

// RunningAverage is the smoothed view of recent timings for one model.
type RunningAverage struct {
	ModelID      string    `json:"model_id"`
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
func (s *Store) AddTiming(sample TimingSample) {
	if sample.ModelID == "" || sample.GenTokens <= 0 {
		return
	}
	if sample.Timestamp.IsZero() {
		sample.Timestamp = time.Now()
	}

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
	if avg.Count == 0 {
		avg.AvgGenTPS = sample.GenTokPerSec
		avg.AvgPromptTPS = sample.PromptTokPerSec
	} else {
		avg.AvgGenTPS = timingEMAAlpha*sample.GenTokPerSec + (1-timingEMAAlpha)*avg.AvgGenTPS
		if sample.PromptTokPerSec > 0 {
			avg.AvgPromptTPS = timingEMAAlpha*sample.PromptTokPerSec + (1-timingEMAAlpha)*avg.AvgPromptTPS
		}
	}
	avg.Count++
	avg.LastUpdated = sample.Timestamp
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

// RunningAverage returns the EMA for one model. The bool is false when
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

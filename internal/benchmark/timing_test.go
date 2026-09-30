package benchmark

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func newTimingStoreForTest(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	return NewStore(dir)
}

func TestAddTimingIgnoresInvalid(t *testing.T) {
	s := newTimingStoreForTest(t)
	s.AddTiming(TimingSample{ModelID: "", GenTokens: 10}) // missing model
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 0}) // no gen tokens
	if got := s.RecentTimings("m", 10); len(got) != 0 {
		t.Errorf("expected invalid samples to be dropped, got %d", len(got))
	}
}

func TestAddTimingAndRecent(t *testing.T) {
	s := newTimingStoreForTest(t)
	for i := 0; i < 5; i++ {
		s.AddTiming(TimingSample{
			ModelID:      "m",
			GenTokens:    10 + i,
			GenTokPerSec: float64(50 + i),
		})
	}
	got := s.RecentTimings("m", 0)
	if len(got) != 5 {
		t.Fatalf("expected 5 samples, got %d", len(got))
	}
	// Newest sample last.
	if got[4].GenTokPerSec != 54 {
		t.Errorf("expected last sample gen_tps=54, got %f", got[4].GenTokPerSec)
	}
}

func TestRingBufferTrims(t *testing.T) {
	s := newTimingStoreForTest(t)
	for i := 0; i < maxTimingSamples+50; i++ {
		s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: float64(i)})
	}
	got := s.RecentTimings("m", 0)
	if len(got) != maxTimingSamples {
		t.Fatalf("expected ring trimmed to %d, got %d", maxTimingSamples, len(got))
	}
	// Oldest retained sample should be #50 (we dropped the first 50).
	if got[0].GenTokPerSec != 50 {
		t.Errorf("expected oldest retained gen_tps=50, got %f", got[0].GenTokPerSec)
	}
}

func TestRunningAverageThreshold(t *testing.T) {
	s := newTimingStoreForTest(t)
	// Add 5 samples (below threshold) — average should not surface.
	for i := 0; i < 5; i++ {
		s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: 50})
	}
	avg, ok := s.RunningAverage("m")
	if ok {
		t.Errorf("expected average hidden below %d samples; got ok=true (count=%d)", timingMinSamplesForAverage, avg.Count)
	}
	// Add enough to clear the threshold.
	for i := 0; i < 6; i++ {
		s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: 50})
	}
	avg, ok = s.RunningAverage("m")
	if !ok {
		t.Fatalf("expected average visible after %d samples", timingMinSamplesForAverage)
	}
	if avg.Count != 11 {
		t.Errorf("expected count=11, got %d", avg.Count)
	}
	// All samples were 50 t/s, so the average is too.
	if math.Abs(avg.AvgGenTPS-50) > 0.001 {
		t.Errorf("expected average ≈ 50, got %f", avg.AvgGenTPS)
	}
}

// One fast request among slow ones moves the figure by its share of the work,
// not by a fixed fraction per sample.
func TestRunningAverageIsTokensOverTime(t *testing.T) {
	s := newTimingStoreForTest(t)
	// 20 requests of 100 tokens at 50 t/s: 2000 tokens in 40 s.
	for i := 0; i < 20; i++ {
		s.AddTiming(TimingSample{ModelID: "m", GenTokens: 100, GenSeconds: 2})
	}
	avg, _ := s.RunningAverage("m")
	if math.Abs(avg.AvgGenTPS-50) > 0.001 {
		t.Fatalf("expected baseline 50, got %f", avg.AvgGenTPS)
	}
	// One more at 200 t/s, the same size: 2100 tokens in 40.5 s.
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 100, GenSeconds: 0.5})
	avg, _ = s.RunningAverage("m")
	if want := 2100 / 40.5; math.Abs(avg.AvgGenTPS-want) > 0.001 {
		t.Errorf("average = %f, want %f", avg.AvgGenTPS, want)
	}
}

// The reason the average is a ratio of sums. A twenty-token prompt is mostly
// fixed overhead and reports a fraction of the real prefill speed; averaged as
// a rate it would count as much as a prompt a thousand times its size.
func TestATrivialPromptDoesNotDragPrefillSpeedDown(t *testing.T) {
	s := newTimingStoreForTest(t)
	// A 20,000-token prompt at 4000 t/s, and a 20-token one at 400 t/s.
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 100, GenSeconds: 1, PromptTokens: 20000, PromptSeconds: 5})
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 100, GenSeconds: 1, PromptTokens: 20, PromptSeconds: 0.05})

	avg, _ := s.RunningAverage("m")
	if avg.AvgPromptTPS < 3900 {
		t.Errorf("prefill average = %.0f t/s; the trivial prompt pulled it toward the mean of the two rates (2200)", avg.AvgPromptTPS)
	}
}

// A prompt answered wholly from the prefix cache did no prefill. It is not a
// request with a prefill speed of zero.
func TestACachedPromptSaysNothingAboutPrefillSpeed(t *testing.T) {
	s := newTimingStoreForTest(t)
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 100, GenSeconds: 1, PromptTokens: 4000, PromptSeconds: 1})
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 100, GenSeconds: 1, PromptTokens: 0, PromptSeconds: 0.002})

	avg, _ := s.RunningAverage("m")
	if math.Abs(avg.AvgPromptTPS-4000) > 0.001 {
		t.Errorf("prefill average = %f, want the 4000 of the one prompt that was computed", avg.AvgPromptTPS)
	}

	// And a model that has only ever been served from the cache has no
	// prefill speed to show, rather than one of zero.
	s.AddTiming(TimingSample{ModelID: "cached", GenTokens: 100, GenSeconds: 1})
	if avg, _ := s.RunningAverage("cached"); avg.AvgPromptTPS != 0 {
		t.Errorf("prefill average = %f for a model with no computed prompt", avg.AvgPromptTPS)
	}
}

// A sample covers every request that finished in one polling interval, and
// the count the panel shows is requests.
func TestCountIsRequestsNotSamples(t *testing.T) {
	s := newTimingStoreForTest(t)
	s.AddTiming(TimingSample{ModelID: "m", Requests: 7, GenTokens: 700, GenSeconds: 7})
	s.AddTiming(TimingSample{ModelID: "m", Requests: 4, GenTokens: 400, GenSeconds: 4})

	avg, ok := s.RunningAverage("m")
	if avg.Count != 11 || !ok {
		t.Errorf("count = %d (shown: %v), want 11 requests and an average on show", avg.Count, ok)
	}
}

func TestRunningAveragesFiltersBelowThreshold(t *testing.T) {
	s := newTimingStoreForTest(t)
	// Model "a" has plenty of samples.
	for i := 0; i < 15; i++ {
		s.AddTiming(TimingSample{ModelID: "a", GenTokens: 1, GenTokPerSec: 100})
	}
	// Model "b" has only a few.
	for i := 0; i < 3; i++ {
		s.AddTiming(TimingSample{ModelID: "b", GenTokens: 1, GenTokPerSec: 200})
	}
	avgs := s.RunningAverages()
	if len(avgs) != 1 {
		t.Fatalf("expected only 'a' visible, got %d entries", len(avgs))
	}
	if avgs[0].ModelID != "a" {
		t.Errorf("expected ModelID=a, got %s", avgs[0].ModelID)
	}
}

func TestTimingTimestampDefaultsToNow(t *testing.T) {
	s := newTimingStoreForTest(t)
	before := time.Now()
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: 1}) // no Timestamp
	got := s.RecentTimings("m", 1)
	if len(got) != 1 || got[0].Timestamp.Before(before) {
		t.Errorf("expected default Timestamp ≥ before(%v), got %v", before, got[0].Timestamp)
	}
}

// A model gathering samples must be distinguishable from no traffic at all.
// Collapsing the two is what made the Live Performance panel look broken while
// it was working: six samples had been captured and the panel said none had.
func TestPendingAveragesTracksProgressThenClears(t *testing.T) {
	s := NewStore("")

	for i := 0; i < MinSamplesForAverage()-1; i++ {
		s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: 50})
	}
	if got := len(s.RunningAverages()); got != 0 {
		t.Errorf("running averages = %d below the threshold, want 0", got)
	}
	pending := s.PendingAverages()
	if len(pending) != 1 || pending[0].Count != MinSamplesForAverage()-1 {
		t.Fatalf("pending = %+v, want one entry just short of the threshold", pending)
	}

	// Reaching the threshold promotes it and clears it from pending, so the
	// panel never shows a model as both averaged and still collecting.
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: 50})
	if got := len(s.RunningAverages()); got != 1 {
		t.Errorf("running averages = %d at the threshold, want 1", got)
	}
	if got := len(s.PendingAverages()); got != 0 {
		t.Errorf("pending = %d after promotion, want 0", got)
	}
}

// A model with no samples is not pending anything.
func TestPendingAveragesIgnoresModelsWithNoSamples(t *testing.T) {
	s := NewStore("")
	s.AddTiming(TimingSample{ModelID: "", GenTokens: 5})  // rejected: no model
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 0}) // rejected: no tokens
	if got := len(s.PendingAverages()); got != 0 {
		t.Errorf("pending = %d, want 0 — neither sample was recorded", got)
	}
}

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
	// All samples were 50 t/s — EMA should converge to 50 (within tolerance).
	if math.Abs(avg.AvgGenTPS-50) > 0.001 {
		t.Errorf("expected EMA ≈ 50, got %f", avg.AvgGenTPS)
	}
}

func TestRunningAverageEMASmoothes(t *testing.T) {
	s := newTimingStoreForTest(t)
	// Stable baseline.
	for i := 0; i < 20; i++ {
		s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: 50})
	}
	avg, _ := s.RunningAverage("m")
	if math.Abs(avg.AvgGenTPS-50) > 0.5 {
		t.Fatalf("expected baseline ≈ 50, got %f", avg.AvgGenTPS)
	}
	// Single 200-tps spike: EMA should move toward 200 but not jump all the way.
	s.AddTiming(TimingSample{ModelID: "m", GenTokens: 1, GenTokPerSec: 200})
	avg, _ = s.RunningAverage("m")
	// With alpha=0.1: new = 0.1*200 + 0.9*50 = 20 + 45 = 65
	if math.Abs(avg.AvgGenTPS-65) > 0.5 {
		t.Errorf("expected single spike to move EMA to ~65, got %f", avg.AvgGenTPS)
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

package benchmark

import (
	"math"
	"testing"
)

func TestPerSizeGroupsAndOrders(t *testing.T) {
	got := PerSize([]BenchmarkResult{
		{PromptTokens: 2048, Repetition: 1, PromptTokPerSec: 2000, GenTokPerSec: 40, TTFTMs: 300},
		{PromptTokens: 128, Repetition: 1, PromptTokPerSec: 500, GenTokPerSec: 50, TTFTMs: 100},
		{PromptTokens: 2048, Repetition: 2, PromptTokPerSec: 2200, GenTokPerSec: 42, TTFTMs: 320},
		{PromptTokens: 128, Repetition: 2, PromptTokPerSec: 540, GenTokPerSec: 52, TTFTMs: 110},
	})
	if len(got) != 2 {
		t.Fatalf("expected 2 sizes, got %d", len(got))
	}
	// Ascending by prompt length, not by whichever arrived first.
	if got[0].PromptTokens != 128 || got[1].PromptTokens != 2048 {
		t.Errorf("sizes out of order: %d then %d", got[0].PromptTokens, got[1].PromptTokens)
	}
	if got[0].Label != "pp128" || got[1].Label != "pp2048" {
		t.Errorf("labels: %q, %q", got[0].Label, got[1].Label)
	}
	if got[0].Count != 2 {
		t.Errorf("count = %d, want 2", got[0].Count)
	}
	if math.Abs(got[0].PromptMean-520) > 0.001 {
		t.Errorf("prompt mean = %v, want 520", got[0].PromptMean)
	}
	if math.Abs(got[0].GenMean-51) > 0.001 {
		t.Errorf("gen mean = %v, want 51", got[0].GenMean)
	}
	if math.Abs(got[0].TTFTMean-105) > 0.001 {
		t.Errorf("ttft mean = %v, want 105", got[0].TTFTMean)
	}
	// Population stddev of {500,540} about 520 is 20.
	if math.Abs(got[0].PromptStd-20) > 0.001 {
		t.Errorf("prompt stddev = %v, want 20", got[0].PromptStd)
	}
}

// The whole reason for this breakdown: a run measuring two very different
// prompt lengths has a headline average that describes neither.
func TestPerSizeSeparatesWhatAnAverageHides(t *testing.T) {
	got := PerSize([]BenchmarkResult{
		{PromptTokens: 128, PromptTokPerSec: 400},
		{PromptTokens: 8192, PromptTokPerSec: 3600},
	})
	if len(got) != 2 {
		t.Fatalf("expected 2 groups, got %d", len(got))
	}
	if got[0].PromptMean == got[1].PromptMean {
		t.Fatal("the two sizes should not share a value")
	}
	// The mean of the two (2000) sits nowhere near either measurement, which
	// is the point.
	for _, s := range got {
		if math.Abs(s.PromptMean-2000) < 100 {
			t.Errorf("size %d reported %v, suspiciously close to the overall mean",
				s.PromptTokens, s.PromptMean)
		}
	}
}

// One repetition has no spread. Reporting "± 0" would be noise; reporting a
// sample stddev would be a divide by zero.
func TestPerSizeSingleRepetitionHasZeroSpread(t *testing.T) {
	got := PerSize([]BenchmarkResult{{PromptTokens: 512, PromptTokPerSec: 900, GenTokPerSec: 40}})
	if len(got) != 1 || got[0].Count != 1 {
		t.Fatalf("got %+v", got)
	}
	if got[0].PromptStd != 0 || got[0].GenStd != 0 {
		t.Errorf("expected zero spread, got %v / %v", got[0].PromptStd, got[0].GenStd)
	}
	if got[0].PromptMean != 900 {
		t.Errorf("mean = %v, want 900", got[0].PromptMean)
	}
}

func TestPerSizeEmpty(t *testing.T) {
	if got := PerSize(nil); got != nil {
		t.Errorf("no results should give no breakdown, got %+v", got)
	}
	if got := PerSize([]BenchmarkResult{}); got != nil {
		t.Errorf("empty results should give no breakdown, got %+v", got)
	}
}

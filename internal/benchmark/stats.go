package benchmark

import "math"

// ComputeSummary aggregates per-test-point results into a summary row.
// Returns nil if there are no results.
func ComputeSummary(results []BenchmarkResult) *BenchmarkSummary {
	if len(results) == 0 {
		return nil
	}

	var sumPP, sumTG, sumTTFT float64
	minTG := math.MaxFloat64
	maxTG := 0.0

	for _, r := range results {
		sumPP += r.PromptTokPerSec
		sumTG += r.GenTokPerSec
		sumTTFT += r.TTFTMs
		if r.GenTokPerSec < minTG {
			minTG = r.GenTokPerSec
		}
		if r.GenTokPerSec > maxTG {
			maxTG = r.GenTokPerSec
		}
	}

	n := float64(len(results))
	return &BenchmarkSummary{
		AvgPromptTokPerSec: sumPP / n,
		AvgGenTokPerSec:    sumTG / n,
		AvgTTFTMs:          sumTTFT / n,
		MinGenTokPerSec:    minTG,
		MaxGenTokPerSec:    maxTG,
	}
}

// ComparisonData holds derived data for the comparison view: the runs
// themselves plus the largest values, used to scale bar widths.
type ComparisonData struct {
	Runs         []BenchmarkRun
	MaxGenTPS    float64
	MaxPromptTPS float64
	HasBenchy    bool
}

// BuildComparison prepares comparison data from a list of runs.
func BuildComparison(runs []BenchmarkRun) ComparisonData {
	c := ComparisonData{Runs: runs}
	for _, r := range runs {
		if r.Summary == nil {
			continue
		}
		if r.Summary.AvgGenTokPerSec > c.MaxGenTPS {
			c.MaxGenTPS = r.Summary.AvgGenTokPerSec
		}
		if r.Summary.AvgPromptTokPerSec > c.MaxPromptTPS {
			c.MaxPromptTPS = r.Summary.AvgPromptTokPerSec
		}
		if len(r.LlamaBenchy) > 0 {
			c.HasBenchy = true
		}
	}
	return c
}

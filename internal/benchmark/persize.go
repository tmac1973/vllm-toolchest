package benchmark

import (
	"fmt"
	"math"
	"sort"
)

// A run's headline averages cover every prompt length it measured, and that
// average matches nothing real: prompt throughput climbs steeply with prompt
// length, so the mean of a 128-token and an 8192-token measurement describes
// neither. Breaking the results out by size is the only form comparable with
// externally published figures, which are always quoted per length.
//
// This is derived from the stored results rather than recorded alongside them,
// so runs benchmarked before it existed get the breakdown too.

// PerSizeStats aggregates one prompt length's repetitions.
type PerSizeStats struct {
	// Label is the llama-bench style name for the size, e.g. "pp512", so a
	// number here can be lined up against one published elsewhere.
	Label        string
	PromptTokens int
	Count        int

	PromptMean float64
	PromptStd  float64
	GenMean    float64
	GenStd     float64
	TTFTMean   float64
}

// PerSize groups a run's results by prompt length, ascending. Returns nil when
// there is nothing to group, so a caller can treat "no breakdown" as absence
// rather than an empty table.
func PerSize(results []BenchmarkResult) []PerSizeStats {
	if len(results) == 0 {
		return nil
	}
	bySize := map[int][]BenchmarkResult{}
	for _, r := range results {
		bySize[r.PromptTokens] = append(bySize[r.PromptTokens], r)
	}
	sizes := make([]int, 0, len(bySize))
	for size := range bySize {
		sizes = append(sizes, size)
	}
	sort.Ints(sizes)

	out := make([]PerSizeStats, 0, len(sizes))
	for _, size := range sizes {
		group := bySize[size]
		s := PerSizeStats{
			Label:        fmt.Sprintf("pp%d", size),
			PromptTokens: size,
			Count:        len(group),
		}
		prompt := make([]float64, 0, len(group))
		gen := make([]float64, 0, len(group))
		var ttft float64
		for _, r := range group {
			prompt = append(prompt, r.PromptTokPerSec)
			gen = append(gen, r.GenTokPerSec)
			ttft += r.TTFTMs
		}
		s.PromptMean, s.PromptStd = meanStd(prompt)
		s.GenMean, s.GenStd = meanStd(gen)
		s.TTFTMean = ttft / float64(len(group))
		out = append(out, s)
	}
	return out
}

// meanStd returns the mean and the population standard deviation.
//
// Population rather than sample: these are all the repetitions that ran, not a
// sample drawn from a larger set, and with the two or three repetitions a
// preset typically does the sample correction would inflate the spread by a
// noticeable amount for no gain in meaning.
func meanStd(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	if len(values) == 1 {
		return mean, 0
	}
	var sq float64
	for _, v := range values {
		d := v - mean
		sq += d * d
	}
	return mean, math.Sqrt(sq / float64(len(values)))
}

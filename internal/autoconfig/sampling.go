package autoconfig

import (
	"regexp"
	"strconv"
	"strings"
)

// samplingPair is one key=value sampling setting as cards write it:
// temperature=0.7, top_p: 0.8.
var samplingPair = regexp.MustCompile(`(?i)\b(temperature|top_p|top_k|min_p|presence_penalty|repetition_penalty)\s*[=:]\s*([0-9]*\.?[0-9]+)`)

func noSampling(a Advice) bool {
	return a.Temperature == nil && a.TopP == nil && a.TopK == nil && a.MinP == nil &&
		a.PresencePenalty == nil && a.RepetitionPenalty == nil
}

// fillSampling reads the sampling values out of the helper's quote, for a
// helper that found the card's sampling advice and gave no values. Found on
// compute: Qwen3.5-35B-A3B-FP8's card lists four sets -- thinking and
// non-thinking, general and coding -- and the 4B helper quoted all four
// exactly and answered null for every value, rather than choose one.
//
// The choice is made here instead: the first set on a line that says
// "general" or "default", else the first set. The values come from a quote
// already checked against the card, so they are the card's.
func fillSampling(a Advice) Advice {
	var sets []map[string]float64
	var lines []string
	for _, line := range strings.Split(a.SamplingQuote, "\n") {
		set := map[string]float64{}
		for _, m := range samplingPair.FindAllStringSubmatch(line, -1) {
			if f, err := strconv.ParseFloat(m[2], 64); err == nil {
				if _, seen := set[strings.ToLower(m[1])]; !seen {
					set[strings.ToLower(m[1])] = f
				}
			}
		}
		if len(set) > 0 {
			sets, lines = append(sets, set), append(lines, strings.ToLower(line))
		}
	}
	if len(sets) == 0 {
		return a
	}
	chosen := sets[0]
	for i, l := range lines {
		if strings.Contains(l, "general") || strings.Contains(l, "default") {
			chosen = sets[i]
			break
		}
	}
	f := func(k string) *float64 {
		if v, ok := chosen[k]; ok {
			return &v
		}
		return nil
	}
	a.Temperature, a.TopP, a.MinP = f("temperature"), f("top_p"), f("min_p")
	a.PresencePenalty, a.RepetitionPenalty = f("presence_penalty"), f("repetition_penalty")
	if v, ok := chosen["top_k"]; ok {
		k := int(v)
		a.TopK = &k
	}
	return a
}

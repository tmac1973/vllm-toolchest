package recommend

import (
	"slices"
	"strings"
)

// The four aims a list is ordered by.
const (
	IntentQuality = "quality"
	IntentFastest = "fastest"
	IntentContext = "context"
	IntentNewest  = "newest"
)

var intents = []string{IntentQuality, IntentFastest, IntentContext, IntentNewest}

// orders works out each aim's order over the verified candidates, as indices,
// once per refresh: a chip press is a slice read.
//
//	quality = 0.70·params + 0.20·headroom + 0.10·accelerated
//	fastest = 0.50·(1 − width) + 0.30·headroom + 0.20·accelerated − 0.40·(weight-only or offloaded)
//	context = the context it holds here
//	newest  = recency
//
// Percentiles are over the verified set. Headroom is the spare memory's share
// of the cards at the recommended width; width is that width's place among
// the host's.
func orders(cands []Candidate, verified []int, cards int) map[string][]int {
	set := make([]Candidate, len(verified))
	for i, ix := range verified {
		set[i] = cands[ix]
	}
	params := percentiles(set, func(c Candidate) float64 {
		if c.ParamsB > 0 {
			return c.ParamsB
		}
		return c.WeightGB // no count: the size is the next best measure of scale
	})
	ctx := percentiles(set, func(c Candidate) float64 { return float64(c.AffordableTokens) })
	rec := percentiles(set, func(c Candidate) float64 { return float64(c.LastModified.Unix()) })

	score := map[string][]float64{}
	for _, in := range intents {
		score[in] = make([]float64, len(set))
	}
	for i, c := range set {
		headroom := 0.0
		if c.Available > 0 {
			headroom = c.Spare / c.Available
		}
		tpNorm := 0.0
		if cards > 1 && c.TP > 1 {
			tpNorm = float64(c.TP-1) / float64(cards-1)
		}
		slow := b2f(c.WeightOnly || c.Offload)
		score[IntentQuality][i] = 0.70*params[i] + 0.20*headroom + 0.10*b2f(c.Accelerated)
		score[IntentFastest][i] = max(0, min(1, 0.50*(1-tpNorm)+0.30*headroom+0.20*b2f(c.Accelerated)-0.40*slow))
		score[IntentContext][i] = ctx[i]
		score[IntentNewest][i] = rec[i]
	}

	out := map[string][]int{}
	for _, in := range intents {
		idx := make([]int, len(set))
		for i := range idx {
			idx[i] = i
		}
		s := score[in]
		slices.SortStableFunc(idx, func(a, b int) int {
			switch {
			case s[a] > s[b]:
				return -1
			case s[a] < s[b]:
				return 1
			case set[a].Downloads != set[b].Downloads:
				return set[b].Downloads - set[a].Downloads
			}
			return strings.Compare(set[a].ID, set[b].ID)
		})
		for i := range idx {
			idx[i] = verified[idx[i]]
		}
		out[in] = idx
	}
	return out
}

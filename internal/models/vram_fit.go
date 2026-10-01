package models

// GPUInventory is the hardware a requirement is compared against.
//
// PerCardGB is the smallest card, not the average: a tensor-parallel group is
// bounded by its weakest member, since every rank holds an equal shard.
//
// Known is false when nothing has been read from the driver yet. The estimator
// this replaces had no such state -- it invented a single 32 GiB card and
// judged every host against it, which is how a model running on four cards
// came to be labelled too large for the machine it was running on.
type GPUInventory struct {
	Count     int     `json:"count"`
	PerCardGB float64 `json:"per_card_gb"`
	// FreePerCardGB is what the emptiest card has left right now. vLLM checks
	// the fraction it was asked for against free memory rather than card size
	// and refuses to start when they disagree, so a fit computed from the card
	// size alone can promise a start that will not happen.
	//
	// Zero means "not reported", in which case PerCardGB stands in.
	FreePerCardGB float64 `json:"free_per_card_gb,omitempty"`
	Known         bool    `json:"known"`
}

// usableGB is what one card can actually offer: what is free, when that is
// known and smaller than the card.
func (inv GPUInventory) usableGB() float64 {
	if inv.FreePerCardGB > 0 && inv.FreePerCardGB < inv.PerCardGB {
		return inv.FreePerCardGB
	}
	return inv.PerCardGB
}

// replicationOverhead is what a rank holds beyond its arithmetic share of the
// weights: tensors that are replicated rather than sharded, quantization
// scales, and alignment padding.
//
// Calibrated against two checkpoints on a four-card host, of different
// quantization and different offload, whose sizes differ by 38%: predicted
// 19.2 and 14.0 GiB per rank against 19.07 and 14.15 reported by the engine.
//
// It is a cost of splitting and applies only from two ranks up; see
// weightsTotalGB.
const replicationOverhead = 1.10

// graphPoolLowPerRankGB and graphPoolHighPerRankGB bound the CUDA/HIP graph
// capture pool one rank allocates. Eager mode pays none of it.
//
// A band, because it is not a constant. What starts have reported, per rank:
//
//	MoE hybrid     TP=4   0.07   capture sizes [4]
//	MoE hybrid     TP=4   0.49
//	27B hybrid     TP=2   0.93   nine capture sizes, to 64
//	27B hybrid     TP=4   1.21   the same config
//	14B dense      TP=1   2.47
//	4B hybrid      TP=1   3.74
//
// This was a flat 0.9, the midpoint of the first two that were taken. The wide
// runs sitting low looked like sharding, and the 27B says otherwise: measured
// at two widths on one config, a rank's pool grew with the width. What
// separates the rows is more likely what was captured -- the smallest is a
// config that captures one size -- and compilation_config is not read here.
// Until it is, the band is the same at every width and for every config.
const (
	graphPoolLowPerRankGB  = 0.05
	graphPoolHighPerRankGB = 4.0
)

// rankOverheadLowGB and rankOverheadHighGB bound what one rank of a split
// holds beyond its weights before it serves anything: the memory the runtime
// takes outside the allocator, and what the allocator reserves over what it
// has handed out. The engine reports the two together with the weights as
// "consumed memory", and the projection had no term for either.
//
// Consumed less weights, per rank, on the one host that has run a split:
//
//	27B hybrid     TP=4   4.62   of it non-torch 3.08
//	MoE hybrid     TP=4   5.32   of it non-torch 2.63
//	MoE hybrid     TP=4   5.50   an earlier engine build
//	27B hybrid     TP=2   6.49   of it non-torch 2.65
//	35B-A3B MoE    TP=4   2.30   Qwen3.5 FP8, vLLM 0.29
//
// Without it the 27B's projection was 12 to 15 GB under what the engine
// needed, at its pessimistic end. It is a band across what was seen and no
// wider, because three models on one host cannot say what it depends on. The
// third sat well under the first two, and was what took the low end from 4.5
// to 2.3: its projection had been saved from overstating its room only by the
// KV cache being understated twice over.
//
// Charged from two ranks up, as the replication surcharge is. The one
// single-rank start on record consumed barely more than its weights, on a
// different host and image; whether a single rank here would is not known.
const (
	rankOverheadLowGB  = 2.3
	rankOverheadHighGB = 6.5
)

// TPOption is what this model costs at one tensor-parallel width, and whether
// that many cards can supply it.
//
// Every figure is a total across the cards in use, not a per-card share. What
// one rank holds is an implementation detail of the split; the question is
// whether the configuration can run.
type TPOption struct {
	TP          int  `json:"tp"`
	Configured  bool `json:"configured,omitempty"`
	Recommended bool `json:"recommended,omitempty"`

	WeightsGB    float64 `json:"weights_gb"`
	KVGB         float64 `json:"kv_gb"`
	OverheadGB   float64 `json:"overhead_gb"`
	RequiredGB   float64 `json:"required_gb"`
	RequiredHigh float64 `json:"required_high_gb,omitempty"`

	// AvailableGB is what this many cards can actually give, which is why it
	// scales with the width: choosing TP=2 on a four-card host puts two cards
	// to work, not four.
	AvailableGB float64 `json:"available_gb"`

	// Fits holds at the pessimistic end of the estimate, so it is a guarantee
	// rather than a hope. Uncertain is exactly the case where the bounds
	// straddle what is available -- it fits if the offload, the graph pool
	// and the working set land at the kind end of their bands and does not
	// if they do not.
	Fits      bool `json:"fits"`
	Uncertain bool `json:"uncertain,omitempty"`

	// SpareGB is what is left once the model and its configured context are
	// placed, and ConcurrentSeqs is how many full-length sequences that
	// buys -- the number to tune max_num_seqs against.
	SpareGB        float64 `json:"spare_gb,omitempty"`
	ConcurrentSeqs int     `json:"concurrent_seqs,omitempty"`

	// Measured marks the one row an actual start produced. The others are
	// projections of it onto a width nothing has run at, and the difference
	// matters: the projected arithmetic is the same arithmetic that was wrong
	// about this model by a third.
	Measured bool `json:"measured,omitempty"`
}

// VRAMFit compares a requirement against real hardware. It is computed at
// render time and never stored, so it cannot outlive the machine it describes.
type VRAMFit struct {
	Known   bool   `json:"known"`
	Unknown bool   `json:"unknown,omitempty"`
	Why     string `json:"why,omitempty"`

	Options       []TPOption `json:"options,omitempty"`
	RecommendedTP int        `json:"recommended_tp,omitempty"`

	// ConfiguredTP is the width the model is actually set to, and Configured
	// is that option when the host can supply that many cards.
	ConfiguredTP int       `json:"configured_tp"`
	Configured   *TPOption `json:"configured,omitempty"`

	// TotalVRAMGB is every card added up, and AvailableGB is the share of it
	// gpu_memory_utilization permits at the configured width.
	TotalVRAMGB float64 `json:"total_vram_gb,omitempty"`
	AvailableGB float64 `json:"available_gb,omitempty"`
}

// Fit compares what the model needs against what the host can give.
//
// It enumerates every tensor-parallel width the host can actually provide,
// because the config panel offers 1, 2, 4 and 8 and each buys a different
// amount of memory to work with.
func Fit(est VRAMEstimate, c VLLMConfig, inv GPUInventory) VRAMFit {
	fit := VRAMFit{ConfiguredTP: c.TensorParallelSize}
	if fit.ConfiguredTP < 1 {
		fit.ConfiguredTP = 1
	}

	if est.Unknown {
		fit.Unknown = true
		fit.Why = est.UnknownWhy
		return fit
	}

	util := c.GPUMemoryUtilization
	if util <= 0 || util > 1 {
		util = 0.90
	}

	if !inv.Known || inv.Count < 1 || inv.PerCardGB <= 0 {
		fit.Why = "no GPU has been read yet, so there is nothing to compare this against"
		return fit
	}
	fit.Known = true
	fit.TotalVRAMGB = float64(inv.Count) * inv.PerCardGB

	// Powers of two only, which is what the config panel offers and what the
	// shapes divide by. A host with three cards can run TP=2, not TP=3.
	for tp := 1; tp <= inv.Count; tp *= 2 {
		fit.Options = append(fit.Options, evaluateTP(est, c, inv, tp, util))
	}

	// The cheapest width that is guaranteed to work. Spending four cards where
	// two would do is a real cost on a shared box.
	for i := range fit.Options {
		o := &fit.Options[i]
		if o.TP == fit.ConfiguredTP {
			o.Configured = true
			fit.Configured = o
			fit.AvailableGB = o.AvailableGB
		}
		if fit.RecommendedTP == 0 && o.Fits {
			fit.RecommendedTP = o.TP
		}
	}
	for i := range fit.Options {
		fit.Options[i].Recommended = fit.Options[i].TP == fit.RecommendedTP
	}

	return fit
}

// projectWeights carries a measured weight total from the width it was taken
// at onto another.
//
// Splitting it is the whole job. Part of the weights shard across the ranks and
// part is replicated on every one of them, and only the second grows as the
// width grows. One measurement cannot separate them, but the replication
// surcharge is the ratio between them, and it is the figure the projected path
// already rests on.
//
// A measurement taken at one rank has no replication in it to find, so
// projecting from there onto a wider split understates. Said here rather than
// hidden: it is the one direction this cannot do better than guess, and the
// answer is to run the model at the width in question.
func projectWeights(measuredTotal float64, measuredTP, tp int) float64 {
	if measuredTP < 1 {
		measuredTP = 1
	}
	if tp < 1 {
		tp = 1
	}
	if tp == measuredTP || measuredTotal <= 0 {
		return measuredTotal
	}

	sharded, perRank := measuredTotal, 0.0
	if measuredTP >= 2 {
		sharded = measuredTotal / replicationOverhead
		perRank = (measuredTotal - sharded) / float64(measuredTP)
	}
	return sharded + perRank*float64(tp)
}

func evaluateTP(est VRAMEstimate, c VLLMConfig, inv GPUInventory, tp int, util float64) TPOption {
	o := TPOption{
		TP:          tp,
		AvailableGB: float64(tp) * inv.usableGB() * util,
	}

	// A measurement never goes through RequiredAt, at any width.
	//
	// At the width it was taken, re-deriving would corrupt it: RequiredAt
	// replaces the measured graph pool with its own constant and re-applies
	// the replication surcharge to a consumed figure that already carries the
	// allocator's overhead.
	//
	// At any *other* width it is worse, because the units disagree.
	// WeightsTotalGB is a total across ranks with replication already inside
	// it, and RequiredAt reads that field as a pre-replication figure -- so it
	// left TP=1 with no replication cost at all and charged TP=2 the surcharge
	// twice. On a real model that produced a table where two cards needed more
	// than four: 123.6 GB against 113.9.
	if est.Source == SourceMeasured && est.MeasuredTP > 0 && est.TotalRequiredGB > 0 {
		o.Measured = tp == est.MeasuredTP
		o.KVGB = est.KVAtContextGB * est.KVScale(tp)

		var low float64
		if o.Measured {
			o.WeightsGB = est.WeightsTotalGB
			o.OverheadGB = est.TotalRequiredGB - est.WeightsTotalGB - est.KVAtContextGB
			o.RequiredGB = est.TotalRequiredGB
			low, o.RequiredHigh = o.RequiredGB, o.RequiredGB
		} else {
			o.WeightsGB = projectWeights(est.WeightsTotalGB, est.MeasuredTP, tp)
			// The graph ladder is captured per rank, so it grows with the
			// width. How the working set moves with the width is the one thing
			// a single run cannot say, so off the measured width it is a band
			// and the row is no longer a single figure.
			graphs := est.MeasuredGraphPerRankGB * float64(tp)
			actLow, actHigh := activationAt(est, tp)

			low = o.WeightsGB + o.KVGB + graphs + actLow
			o.RequiredHigh = o.WeightsGB + o.KVGB + graphs + actHigh
			o.RequiredGB = (low + o.RequiredHigh) / 2
			o.OverheadGB = graphs + (actLow+actHigh)/2
		}
		if o.OverheadGB < 0 {
			o.OverheadGB = 0
		}

		o.Fits = o.RequiredHigh <= o.AvailableGB
		o.Uncertain = !o.Fits && low <= o.AvailableGB
		if spare := o.AvailableGB - o.RequiredHigh; spare > 0 {
			o.SpareGB = spare
			if o.KVGB > 0 {
				o.ConcurrentSeqs = 1 + int(spare/o.KVGB)
			}
		}
		return o
	}

	r := RequiredAt(est, c, tp)
	o.WeightsGB = (r.WeightsGB + r.WeightsHighGB) / 2
	o.KVGB = r.KVGB
	o.OverheadGB = (r.GraphsGB+r.GraphsHighGB)/2 + r.CacheGB +
		(r.ActivationGB+r.ActivationHighGB)/2 + (r.RankOverheadGB+r.RankOverheadHighGB)/2
	o.RequiredGB = (r.TotalGB + r.TotalHighGB) / 2
	o.RequiredHigh = r.TotalHighGB

	o.Fits = r.TotalHighGB <= o.AvailableGB
	o.Uncertain = !o.Fits && r.TotalGB <= o.AvailableGB

	// What the leftover buys, counted in whole sequences at the configured
	// context. The requirement already includes one, so the spare adds to it.
	if spare := o.AvailableGB - r.TotalHighGB; spare > 0 {
		o.SpareGB = spare
		if r.KVGB > 0 {
			o.ConcurrentSeqs = 1 + int(spare/r.KVGB)
		}
	}
	return o
}

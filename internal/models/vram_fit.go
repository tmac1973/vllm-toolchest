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

// graphPoolPerGPUGB is the CUDA/HIP graph capture pool one rank allocates.
// Measured on this machine at 0.49 and 1.32 GiB across two checkpoints; the
// midpoint is used, and eager mode pays none of it.
const graphPoolPerGPUGB = 0.9

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
	// straddle what is available -- it fits if the offload behaves and does
	// not if it does not.
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

func evaluateTP(est VRAMEstimate, c VLLMConfig, inv GPUInventory, tp int, util float64) TPOption {
	o := TPOption{
		TP:          tp,
		AvailableGB: float64(tp) * inv.usableGB() * util,
	}

	// At the width a real start ran at, the stored totals are the answer and
	// re-deriving them would corrupt them: RequiredAt would replace the
	// measured graph pool with its own constant, and re-apply the replication
	// surcharge to a consumed figure that already carries the allocator's
	// overhead. Both are corrections to a projection, and there is nothing
	// here left to correct.
	if est.Source == SourceMeasured && tp == est.MeasuredTP && est.TotalRequiredGB > 0 {
		o.Measured = true
		o.WeightsGB = est.WeightsTotalGB
		o.KVGB = est.KVAtContextGB
		o.OverheadGB = est.TotalRequiredGB - est.WeightsTotalGB - est.KVAtContextGB
		if o.OverheadGB < 0 {
			o.OverheadGB = 0
		}
		o.RequiredGB = est.TotalRequiredGB
		o.RequiredHigh = est.TotalRequiredGB

		o.Fits = o.RequiredGB <= o.AvailableGB
		if spare := o.AvailableGB - o.RequiredGB; spare > 0 {
			o.SpareGB = spare
			if est.KVAtContextGB > 0 {
				o.ConcurrentSeqs = 1 + int(spare/est.KVAtContextGB)
			}
		}
		return o
	}

	r := RequiredAt(est, c, tp)
	o.WeightsGB = (r.WeightsGB + r.WeightsHighGB) / 2
	o.KVGB = r.KVGB
	o.OverheadGB = r.GraphsGB + r.CacheGB + r.ActivationGB
	o.RequiredGB = (r.TotalGB + r.TotalHighGB) / 2
	o.RequiredHigh = r.TotalHighGB

	o.Fits = r.TotalHighGB <= o.AvailableGB
	o.Uncertain = !o.Fits && r.TotalGB <= o.AvailableGB

	// What the leftover buys, counted in whole sequences at the configured
	// context. The requirement already includes one, so the spare adds to it.
	if spare := o.AvailableGB - r.TotalHighGB; spare > 0 {
		o.SpareGB = spare
		if est.KVAtContextGB > 0 {
			o.ConcurrentSeqs = 1 + int(spare/est.KVAtContextGB)
		}
	}
	return o
}

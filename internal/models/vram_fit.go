package models

// GPUInventory is the hardware a fit is judged against.
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
	Known     bool    `json:"known"`
}

// replicationOverhead is what a rank holds beyond its arithmetic share of the
// weights: tensors that are replicated rather than sharded, quantization
// scales, and alignment padding.
//
// Calibrated against two checkpoints on a four-card host, of different
// quantization and different offload, whose sizes differ by 38%: predicted
// 19.2 and 14.0 GiB per rank against 19.07 and 14.15 reported by the engine.
//
// It is a tensor-parallel cost and applies only from two ranks up; see
// perRankWeights.
const replicationOverhead = 1.10

// TPOption is what one tensor-parallel width would cost per card, and whether
// it works.
type TPOption struct {
	TP int `json:"tp"`

	WeightsPerGPUGB float64 `json:"weights_per_gpu_gb"`
	// WeightsPerGPUHighGB is the upper bound when offload is unsized; equal to
	// WeightsPerGPUGB when the figure is certain.
	WeightsPerGPUHighGB float64 `json:"weights_per_gpu_high_gb,omitempty"`
	HostResidentGB      float64 `json:"host_resident_gb,omitempty"`

	ActivationGB float64 `json:"activation_gb"`
	GraphsGB     float64 `json:"graphs_gb"`
	LoadGB       float64 `json:"load_gb"`
	BudgetGB     float64 `json:"budget_gb"`

	// KVHeadroomGB is what is left for the KV cache once the weights,
	// activations and graphs are placed -- which is precisely what vLLM claims
	// at startup, so it is the number that decides how much context the engine
	// can actually serve.
	KVHeadroomGB float64 `json:"kv_headroom_gb"`
	MaxTokens    int     `json:"max_tokens"`

	// Loads is "the engine will start". ServesConfigured is "and it will serve
	// the context length this model is configured for". They are different
	// questions and conflating them is why a model could be called a fit and
	// then refuse the context it was set to.
	Loads            bool `json:"loads"`
	ServesConfigured bool `json:"serves_configured"`
	// ServesWorstCase holds when the option still serves the configured
	// context with nothing offloaded at all -- a claim that survives the band
	// being wrong in the worst direction.
	ServesWorstCase bool `json:"serves_worst_case,omitempty"`

	// Uncertain marks an option whose bounds straddle the budget: it fits if
	// offload behaves and does not if it does not. No verdict is offered.
	Uncertain bool `json:"uncertain,omitempty"`
}

// VRAMFit is the hardware verdict. It is computed at render time and never
// stored, so it cannot outlive the machine it describes.
type VRAMFit struct {
	Known   bool   `json:"known"`
	Unknown bool   `json:"unknown,omitempty"`
	Why     string `json:"why,omitempty"`

	Options       []TPOption `json:"options,omitempty"`
	RecommendedTP int        `json:"recommended_tp,omitempty"`

	// ConfiguredTP is the width the model is actually set to, and Configured
	// is that option when the inventory can supply it.
	ConfiguredTP int       `json:"configured_tp"`
	Configured   *TPOption `json:"configured,omitempty"`

	// Label is the short verdict for the card badge.
	Label string `json:"label"`
	// PerGPUGB is the figure to show beside the label: what one card holds at
	// the configured width.
	PerGPUGB float64 `json:"per_gpu_gb"`
}

// Fit judges an estimate against real hardware.
//
// It enumerates every tensor-parallel width the host can actually provide,
// rather than the one-or-two the old labels assumed, because the config panel
// offers 1, 2, 4 and 8 and a verdict that only considered two of them was
// wrong for half the choices a user can make.
func Fit(est VRAMEstimate, c VLLMConfig, inv GPUInventory) VRAMFit {
	fit := VRAMFit{ConfiguredTP: c.TensorParallelSize}
	if fit.ConfiguredTP < 1 {
		fit.ConfiguredTP = 1
	}

	if est.Unknown {
		fit.Unknown = true
		fit.Why = est.UnknownWhy
		fit.Label = "—"
		return fit
	}

	util := c.GPUMemoryUtilization
	if util <= 0 || util > 1 {
		util = 0.90
	}

	// Without an inventory there is still arithmetic worth showing -- what one
	// card would hold at the configured width -- but no verdict can be drawn.
	if !inv.Known || inv.Count < 1 || inv.PerCardGB <= 0 {
		fit.PerGPUGB = perRankWeights(est.DeviceWeightsGB, fit.ConfiguredTP)
		fit.Label = "no card inventory"
		fit.Why = "no GPU has been read yet, so there is nothing to judge this against"
		return fit
	}
	fit.Known = true

	// Powers of two only, which is what the config panel offers and what the
	// shapes divide by. A host with three cards can run TP=2, not TP=3.
	for tp := 1; tp <= inv.Count; tp *= 2 {
		fit.Options = append(fit.Options, evaluateTP(est, c, inv, tp, util))
	}

	// The recommendation is the cheapest width that actually serves, so it is
	// taken from the first qualifying option -- the options are built in
	// ascending order. Spending four cards where two would do is a real cost
	// on a shared box.
	//
	// Where every option is uncertain -- any unsized offload makes that so --
	// fall back to the cheapest width that serves on the *upper* bound, where
	// nothing moves off the cards. That is a guarantee rather than a guess,
	// and it beats offering no recommendation at all on exactly the models
	// that most need one.
	for i := range fit.Options {
		o := &fit.Options[i]
		if o.TP == fit.ConfiguredTP {
			fit.Configured = o
		}
		if fit.RecommendedTP == 0 && o.Loads && o.ServesConfigured && !o.Uncertain {
			fit.RecommendedTP = o.TP
		}
	}
	if fit.RecommendedTP == 0 {
		for i := range fit.Options {
			if o := fit.Options[i]; o.ServesWorstCase {
				fit.RecommendedTP = o.TP
				break
			}
		}
	}

	fit.Label, fit.PerGPUGB = label(fit, est)
	return fit
}

func evaluateTP(est VRAMEstimate, c VLLMConfig, inv GPUInventory, tp int, util float64) TPOption {
	o := TPOption{
		TP:                  tp,
		WeightsPerGPUGB:     perRankWeights(est.DeviceWeightsGB, tp),
		WeightsPerGPUHighGB: perRankWeights(est.DeviceWeightsHighGB, tp),
		HostResidentGB:      est.HostResidentGB,
		ActivationGB:        est.ActivationBaseGB / float64(tp),
		GraphsGB:            graphsGB(c),
		BudgetGB:            inv.PerCardGB * util,
	}

	o.LoadGB = o.WeightsPerGPUGB + o.ActivationGB + o.GraphsGB
	highLoad := o.WeightsPerGPUHighGB + o.ActivationGB + o.GraphsGB

	o.KVHeadroomGB = o.BudgetGB - o.LoadGB
	if o.KVHeadroomGB < 0 {
		o.KVHeadroomGB = 0
	}

	// KV heads shard with the ranks, so each rank caches its own slice.
	if perRank := float64(est.KVCachePerTokenB) / float64(tp); perRank > 0 {
		o.MaxTokens = int(o.KVHeadroomGB * 1024 * 1024 * 1024 / perRank)
	}

	o.Loads = o.LoadGB <= o.BudgetGB
	ctx := c.MaxModelLen
	o.ServesConfigured = o.Loads && (ctx <= 0 || o.MaxTokens >= ctx)

	// ServesWorstCase is the same question asked of the upper bound: it holds
	// when the option works even if every offloaded byte stays on the card.
	// That is the only claim a band can make without assuming its own best
	// case, so it is what a recommendation falls back to.
	if worstHeadroom := o.BudgetGB - highLoad; worstHeadroom > 0 && est.KVCachePerTokenB > 0 {
		worstTokens := int(worstHeadroom * 1024 * 1024 * 1024 / (float64(est.KVCachePerTokenB) / float64(tp)))
		o.ServesWorstCase = ctx <= 0 || worstTokens >= ctx
	}

	// Never let the worst case pass as a fit: if the upper bound does not
	// load, the option does not load, whatever the low bound says.
	if highLoad > o.BudgetGB {
		o.ServesConfigured = false
	}

	// A verdict that rests on an unsized offload is not a verdict, even where
	// both bounds land the same side of the budget: "fits" computed from the
	// optimistic bound quietly assumes every offloaded byte really does leave
	// the card. So this is taken from whether the offload is sized at all,
	// not from whether the arithmetic happened to straddle -- a band that
	// comes out narrow is still a guess, and a wide band sitting entirely
	// under the budget is the case most likely to be believed and least
	// entitled to be.
	o.Uncertain = est.OffloadUnsized || highLoad > o.LoadGB+0.05
	return o
}

// perRankWeights is what one card holds of the weights at a given width.
//
// The replication surcharge applies only from two ranks up. With one rank
// nothing is sharded, so there is nothing to replicate and no surcharge to
// pay -- charging it anyway inflated a 23.0 GB checkpoint to 25.3 GB and
// turned a model that fits on one card into one that supposedly could not
// serve its own configured context.
func perRankWeights(deviceGB float64, tp int) float64 {
	if tp < 2 {
		return deviceGB
	}
	return deviceGB / float64(tp) * replicationOverhead
}

// graphsGB is the CUDA/HIP graph capture pool. Measured on this machine at
// 0.49 and 1.32 GiB across two checkpoints; the midpoint is used, and eager
// mode pays none of it.
func graphsGB(c VLLMConfig) float64 {
	if c.EnforceEager {
		return 0
	}
	return 0.9
}

func label(fit VRAMFit, est VRAMEstimate) (string, float64) {
	if fit.Configured == nil {
		if fit.RecommendedTP > 0 {
			return "needs TP=" + itoa(fit.RecommendedTP), 0
		}
		return "too large", 0
	}

	// The label describes the width the model is configured for, since that is
	// what will happen if it is started now. Where a cheaper width would also
	// serve, it says so rather than leaving the banner claiming agreement with
	// a recommendation sitting elsewhere in the table.
	o := *fit.Configured
	switch {
	case o.Uncertain:
		return "depends on offload", o.WeightsPerGPUGB
	case o.ServesConfigured && fit.RecommendedTP > 0 && fit.RecommendedTP < o.TP:
		return "fits TP=" + itoa(o.TP) + ", TP=" + itoa(fit.RecommendedTP) + " would do", o.WeightsPerGPUGB
	case o.ServesConfigured:
		return "fits TP=" + itoa(o.TP), o.WeightsPerGPUGB
	case o.Loads:
		return "loads, context won't fit", o.WeightsPerGPUGB
	case fit.RecommendedTP > 0:
		return "needs TP=" + itoa(fit.RecommendedTP), o.WeightsPerGPUGB
	default:
		return "too large", o.WeightsPerGPUGB
	}
}

func itoa(n int) string {
	if n <= 0 {
		return "0"
	}
	var b [4]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

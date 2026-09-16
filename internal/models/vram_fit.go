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
	// CacheGB is a staging buffer held on the card, not offloaded from it.
	CacheGB  float64 `json:"cache_gb,omitempty"`
	LoadGB   float64 `json:"load_gb"`
	BudgetGB float64 `json:"budget_gb"`
	// LoadHighGB is the pessimistic end of the same figure.
	LoadHighGB float64 `json:"load_high_gb,omitempty"`

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
	// Loads is "the engine will start", judged at the pessimistic end of the
	// band so it is a guarantee rather than a hope. ServesConfigured adds
	// "and it will serve the context this model is configured for".
	Loads            bool `json:"loads"`
	ServesConfigured bool `json:"serves_configured"`

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
	// ServesConfigured is already judged at the pessimistic end, so the first
	// option satisfying it is the cheapest width that is guaranteed to work.
	for i := range fit.Options {
		o := &fit.Options[i]
		if o.TP == fit.ConfiguredTP {
			fit.Configured = o
		}
		if fit.RecommendedTP == 0 && o.Loads && o.ServesConfigured && !o.Uncertain {
			fit.RecommendedTP = o.TP
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

	// The staging cache sits on the card, so it is part of what a rank holds
	// at either end of the band.
	o.CacheGB = est.DeviceCacheGB
	o.LoadGB = o.WeightsPerGPUGB + o.ActivationGB + o.GraphsGB + o.CacheGB
	highLoad := o.WeightsPerGPUHighGB + o.ActivationGB + o.GraphsGB + o.CacheGB
	o.LoadHighGB = highLoad

	// Headroom is reported at the pessimistic end, so the KV figure on screen
	// is one the engine can actually deliver rather than the best case.
	o.KVHeadroomGB = o.BudgetGB - highLoad
	if o.KVHeadroomGB < 0 {
		o.KVHeadroomGB = 0
	}

	// KV heads shard with the ranks, so each rank caches its own slice.
	if perRank := float64(est.KVCachePerTokenB) / float64(tp); perRank > 0 {
		o.MaxTokens = int(o.KVHeadroomGB * 1024 * 1024 * 1024 / perRank)
	}

	// The whole band either agrees or it does not, and that is the only
	// question worth asking of it.
	//
	// Loads means guaranteed: it holds at the pessimistic end, so it survives
	// the estimate being wrong in the direction that costs memory. Uncertain
	// is exactly disagreement -- the bounds straddle the budget -- rather than
	// "an offload was involved". Deriving it from the latter withheld a
	// verdict from every model with PLE enabled, including the one whose
	// per-rank figure lands within a gigabyte of what the engine reports.
	ctx := c.MaxModelLen
	o.Loads = highLoad <= o.BudgetGB
	o.Uncertain = !o.Loads && o.LoadGB <= o.BudgetGB
	o.ServesConfigured = o.Loads && (ctx <= 0 || o.MaxTokens >= ctx)
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

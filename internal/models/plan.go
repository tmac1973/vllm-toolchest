package models

import (
	"fmt"
	"math"
)

// PlanDefaults are the machine-wide settings a plan starts from.
type PlanDefaults struct {
	// GPUMemoryUtilization is the fraction of each card vLLM may claim. Zero
	// or out of range means vLLM's own 0.90.
	GPUMemoryUtilization float64
	// MaxNumSeqs is the batch cap. It is not sized from memory: the KV cache
	// is shared, so the cap limits how many requests run at once, not how
	// much each may hold. Zero means 16.
	MaxNumSeqs int
}

func (d PlanDefaults) normalised() PlanDefaults {
	if d.GPUMemoryUtilization <= 0 || d.GPUMemoryUtilization > 1 {
		d.GPUMemoryUtilization = 0.90
	}
	if d.MaxNumSeqs <= 0 {
		d.MaxNumSeqs = 16
	}
	return d
}

// PlanInput is everything PlanFit needs. It holds no server state: the
// caller supplies the inventory and the estimate, so the planner is the same
// arithmetic whoever asks.
type PlanInput struct {
	Model     *Model
	Base      VLLMConfig // the config to build on
	Inventory GPUInventory
	Class     ContextClass
	Defaults  PlanDefaults
	// CardKVDtype is the KV cache dtype the model card's command names, ""
	// when it names none. The one hardware value a card decides: it is the
	// author's tested setting for that checkpoint.
	CardKVDtype string
	// FixedTP, when positive, plans that width only and keeps Base's
	// utilization, sequence cap, KV dtype and pinned pool, so that only the
	// context moves. That is refinement: the configuration has run, and a
	// measurement of it applies only while the rest stays as it was.
	FixedTP int
	// Estimate returns what the model needs under a candidate config:
	// measured when a measurement applies to it, projected otherwise.
	Estimate func(c VLLMConfig) VRAMEstimate
}

// WidthPlan is one tensor-parallel width, fully configured.
type WidthPlan struct {
	TP             int
	Config         VLLMConfig
	ContextTokens  int
	ContextReduced bool // less than the class asked for
	// FullContextRequests is how many requests at the full context the cache
	// holds at once. Zero when the model's KV shape is unknown.
	FullContextRequests int

	RequiredGB, RequiredHighGB, AvailableGB float64
	Measured                                bool

	// Notes say why each field has this width's value.
	Notes []ProfileNote
}

// FitPlan is the answer for one model on this machine: the all-cards width,
// and the narrowest that holds the context asked for when that is fewer
// cards.
type FitPlan struct {
	Known bool
	Why   string // when !Known

	All    WidthPlan
	Narrow *WidthPlan

	// FirstGuess is true when no measurement stands behind the figures.
	FirstGuess bool
	// Notes are true whichever width is chosen.
	Notes []ProfileNote
}

const (
	// minPlanContext is the least context worth configuring. A width that
	// cannot hold this much does not hold the model.
	minPlanContext = 2048
	// contextStep is what a planned context is rounded down to.
	contextStep = 1024
	// projectedMargin and measuredMargin are the share of the spare memory a
	// plan spends on context. A projection is a band whose pessimistic end is
	// often beyond the host, and the plan is drawn on its middle, so it keeps
	// a sixth back; a measurement is the engine's own figure and keeps only
	// enough to absorb rounding.
	projectedMargin = 0.85
	measuredMargin  = 0.97
	// unknownShapeContext is the context used when the KV shape cannot be
	// read and the base config names none.
	unknownShapeContext = 8192
	// maxClassFallback stands in for ContextMax when the model does not say
	// what its maximum is.
	maxClassFallback = 131072
)

// PlanFit works out, for a model on this machine, how many cards to use, how
// much context to configure, which KV dtype, and the memory fraction and
// sequence cap -- for all the cards and for the narrowest width that holds
// the context asked for.
//
// It plans on the expected figure, the middle of the estimate's band, not on
// its pessimistic end. For a large model split four ways the pessimistic end
// is usually beyond the host, and a planner bound to it would choose nothing.
// The margin taken off the context, and the refinement a real start makes
// possible, are what cover the difference.
func PlanFit(in PlanInput) FitPlan {
	if in.Model == nil {
		return FitPlan{Why: "no model"}
	}
	if in.Estimate == nil {
		return FitPlan{Why: "nothing to estimate the model with"}
	}
	inv := in.Inventory
	if !inv.Known || inv.Count < 1 || inv.PerCardGB <= 0 {
		return FitPlan{Why: "no GPU has been read yet"}
	}
	defaults := in.Defaults.normalised()
	target := planTarget(in.Model, in.Class)

	var widths []int
	if in.FixedTP > 0 {
		widths = []int{in.FixedTP}
	} else {
		for tp := 1; tp <= inv.Count; tp *= 2 {
			widths = append(widths, tp)
		}
	}

	dtype := in.CardKVDtype
	fixed := in.FixedTP > 0
	if fixed {
		dtype = in.Base.KVCacheDtype
	}
	if dtype == "" {
		dtype = "auto"
	}

	plans, why := planWidths(in, defaults, widths, target, dtype)
	var general []ProfileNote

	// The engine default first; fp8 only when the context asked for does not
	// fit without it. Never when the card or a configuration that has run
	// already decided.
	if in.CardKVDtype == "" && !fixed && dtype == "auto" {
		if all := widest(plans); all == nil || all.ContextTokens < target {
			if fp8, _ := planWidths(in, defaults, widths, target, "fp8"); widest(fp8) != nil {
				if all == nil || widest(fp8).ContextTokens > all.ContextTokens {
					plans = fp8
					general = append(general, ProfileNote{
						Field:  "kv_cache_dtype",
						Reason: "fp8, because the context asked for does not fit with the engine's default cache dtype. It halves what each token of context costs, at a small cost in accuracy.",
						Origin: "this machine",
					})
				}
			}
		}
	}

	all := widest(plans)
	if all == nil {
		if why == "" {
			why = "does not fit on this machine at any width"
		}
		return FitPlan{Why: why}
	}

	fp := FitPlan{Known: true, All: *all, FirstGuess: !all.Measured, Notes: general}
	if !fixed {
		for i := range plans {
			p := plans[i]
			if p.TP < all.TP && p.ContextTokens >= target && (fp.Narrow == nil || p.TP < fp.Narrow.TP) {
				fp.Narrow = &p
			}
		}
	}
	if fp.FirstGuess {
		fp.Notes = append(fp.Notes, ProfileNote{
			Reason: "This model has not run here yet, so these figures are an estimate. A real start will measure them, and the context can then be refined.",
			Origin: "this machine",
		})
	}
	return fp
}

// planTarget is the context a class asks for, for this model.
func planTarget(m *Model, class ContextClass) int {
	max := m.HFConfig.MaxPositionEmbeddings
	target := class.Tokens()
	if class == ContextMax {
		if max > 0 {
			return max
		}
		return maxClassFallback
	}
	if max > 0 && target > max {
		target = max
	}
	return target
}

// widest is the plan at the largest width, or nil when none holds the model.
func widest(plans []WidthPlan) *WidthPlan {
	var best *WidthPlan
	for i := range plans {
		if best == nil || plans[i].TP > best.TP {
			best = &plans[i]
		}
	}
	return best
}

// planWidths plans every width that holds the model with one KV dtype. why is
// set when the estimate itself could not be made.
func planWidths(in PlanInput, d PlanDefaults, widths []int, target int, dtype string) (plans []WidthPlan, why string) {
	for _, tp := range widths {
		p, ok, reason := planWidth(in, d, tp, target, dtype)
		if reason != "" {
			return nil, reason
		}
		if ok {
			plans = append(plans, p)
		}
	}
	return plans, ""
}

// planWidth plans one width. ok is false when the width does not hold the
// model; why is non-empty when nothing can be said at all.
func planWidth(in PlanInput, d PlanDefaults, tp, target int, dtype string) (p WidthPlan, ok bool, why string) {
	c := in.Base
	c.TensorParallelSize = tp
	c.MaxModelLen = target
	if in.FixedTP <= 0 {
		c.GPUMemoryUtilization = d.GPUMemoryUtilization
		c.MaxNumSeqs = d.MaxNumSeqs
		c.KVCacheDtype = dtype
		// A pinned pool would override the engine's own sizing, which is
		// what the context below is planned against.
		c.KVCacheMemory = 0
	}

	est := in.Estimate(c)
	if est.Unknown {
		why := est.UnknownWhy
		if why == "" {
			why = "the model could not be sized"
		}
		return WidthPlan{}, false, why
	}
	util := c.GPUMemoryUtilization
	if util <= 0 || util > 1 {
		util = 0.90
	}
	o := evaluateTP(est, c, in.Inventory, tp, util)
	measured := o.Measured && est.Source == SourceMeasured

	nonKV := o.RequiredGB - o.KVGB
	spare := o.AvailableGB - nonKV

	p = WidthPlan{
		TP:             tp,
		RequiredGB:     o.RequiredGB,
		RequiredHighGB: o.RequiredHigh,
		AvailableGB:    o.AvailableGB,
		Measured:       measured,
	}

	if est.KVCachePerTokenB <= 0 {
		// Nothing to size the context by. The width holds the model when the
		// weights and overhead fit without any cache.
		if spare <= 0 {
			return WidthPlan{}, false, ""
		}
		ctx := unknownShapeContext
		if in.Base.MaxModelLen > 0 {
			ctx = in.Base.MaxModelLen
		}
		p.ContextTokens = min(target, ctx)
		p.ContextReduced = p.ContextTokens < target
		c.MaxModelLen = p.ContextTokens
		p.Config = c
		p.Notes = widthNotes(c, p, target, dtype, in.CardKVDtype != "", in.FixedTP > 0, "the model's KV cache shape could not be read, so the context was not sized from memory")
		return p, true, ""
	}

	margin := projectedMargin
	if measured {
		margin = measuredMargin
	}
	tokens := 0
	if spare > 0 {
		perToken := float64(est.KVCachePerTokenB) * est.KVScale(tp)
		t := spare * (1 << 30) / perToken * margin
		tokens = int(math.Floor(t/contextStep)) * contextStep
	}
	ctx := min(target, tokens)
	if ctx < minPlanContext {
		return WidthPlan{}, false, ""
	}

	p.ContextTokens = ctx
	p.ContextReduced = ctx < target
	p.FullContextRequests = max(1, tokens/ctx)
	c.MaxModelLen = ctx
	p.Config = c
	p.Notes = widthNotes(c, p, target, dtype, in.CardKVDtype != "", in.FixedTP > 0, "")
	return p, true, ""
}

// widthNotes explains each hardware field of one width's config.
func widthNotes(c VLLMConfig, p WidthPlan, target int, dtype string, fromCard, fixed bool, contextWhy string) []ProfileNote {
	note := func(field, reason string) ProfileNote {
		return ProfileNote{Field: field, Reason: reason, Origin: "this machine"}
	}
	cards := "card"
	if p.TP > 1 {
		cards = "cards"
	}
	notes := []ProfileNote{
		note("tensor_parallel_size", fmt.Sprintf("Split across %d %s. It needs about %.1f GB here, of %.1f GB those cards offer.",
			p.TP, cards, p.RequiredGB, p.AvailableGB)),
	}

	switch {
	case contextWhy != "":
		notes = append(notes, note("max_model_len", fmt.Sprintf("%d tokens: %s.", p.ContextTokens, contextWhy)))
	case p.ContextReduced:
		notes = append(notes, note("max_model_len", fmt.Sprintf("%d tokens, the most that fits with room to spare; %d were asked for.",
			p.ContextTokens, target)))
	default:
		notes = append(notes, note("max_model_len", fmt.Sprintf("%d tokens, as asked for, with room for %s at once.",
			p.ContextTokens, requestsPhrase(p.FullContextRequests))))
	}

	if fixed {
		return notes
	}
	notes = append(notes,
		note("gpu_memory_utilization", fmt.Sprintf("%.2f of each card, the machine-wide setting.", c.GPUMemoryUtilization)),
		note("max_num_seqs", fmt.Sprintf("%d at once, the machine-wide setting. It caps the batch rather than reserving memory: shorter requests share the cache.", c.MaxNumSeqs)),
	)
	switch {
	case fromCard:
		notes = append(notes, note("kv_cache_dtype", dtype+", as the model card's command sets it: the author's tested setting for this checkpoint."))
	case dtype == "auto":
		notes = append(notes, note("kv_cache_dtype", "The engine's default: the context asked for fits without a smaller cache."))
	}
	return notes
}

func requestsPhrase(n int) string {
	switch n {
	case 0:
		return "an unknown number of full-length requests"
	case 1:
		return "one full-length request"
	}
	return fmt.Sprintf("%d full-length requests", n)
}

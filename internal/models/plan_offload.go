package models

import (
	"fmt"
	"regexp"
	"strings"
)

// ExpertOffloadFlag turns on the rdna4-clav image's expert offload.
const ExpertOffloadFlag = "--enable-expert-offload"

// What the image's expert-offload planner reserves on each card besides the
// weights, measured on compute (plan/autoconfigure/phase-15-expert-offload.md):
// Flash-Next at two cards was planned at 0.40 GiB of graphs and 8.48 of
// runtime overhead, and measured 6.38 to 8.43 for the latter. The KV reserve
// is one request at the configured context, 8% over.
const (
	offloadGraphsGB      = 0.4
	offloadRuntimeGB     = 8.5
	offloadKVReserve     = 1.08
	offloadRankProcessGB = 7.0 // host RAM per rank, the engine's own sum
	offloadEngineGB      = 5.0 // host RAM for the API server and engine
	offloadHostShare     = 0.9 // of MemTotal, for what the engine's margin keeps back
	// mxfp4BytesPerParam is MXFP4's 4-bit values with an 8-bit scale per 32.
	// Flash-Next's experts come to 59.7 GiB at it; the engine counted 59.9.
	mxfp4BytesPerParam = 4.25 / 8
	// residentExpertLayers is how many MoE layers the engine keeps wholly on
	// the cards by default (--expert-resident-layers 0,1,2).
	residentExpertLayers = 3
)

// offloadEligible reports a model the image's expert offload serves: a
// mixture of experts with weights under 8 bits. The image's docs name MXFP4
// and W4A16 experts.
func offloadEligible(m *Model) bool {
	if m == nil || !isMoE(m.HFConfig) || m.HFConfig.NumExperts <= 0 || m.HFConfig.MoEIntermediate <= 0 {
		return false
	}
	b := m.Quantization.BytesPerParam
	return b > 0 && b < 1
}

// expertWeightGB is what a model's routed experts weigh, in GiB.
func expertWeightGB(cfg HFConfig) float64 {
	layers := cfg.NumHiddenLayers - max(0, cfg.DenseLayers)
	if layers <= 0 {
		return 0
	}
	params := float64(layers) * float64(cfg.NumExperts) * 3 * float64(cfg.HiddenSize) * float64(cfg.MoEIntermediate)
	return params * mxfp4BytesPerParam / (1 << 30)
}

// planOffload plans a model with its experts in system RAM at the all-cards
// width, or says why it cannot be: the image plans the cards itself, giving
// the KV cache one request at the context and the experts' cache the rest,
// so the plan is the target context, one request at a time.
//
// It is feasible when the weights that are not experts, that one request's
// cache, the graphs and the runtime's overhead leave the experts a working
// cache -- the resident layers twice over, for a swap region beside them, an
// assumption until a smaller cache has been tried -- and all the experts fit
// in the host's RAM beside anything else already held there.
func planOffload(in PlanInput, d PlanDefaults, tp, target int, dtype string) (WidthPlan, bool, string) {
	m := in.Model
	if !in.ExpertOffload || in.FixedTP > 0 || !offloadEligible(m) {
		return WidthPlan{}, false, ""
	}
	if in.HostRAMGB <= 0 {
		return WidthPlan{}, false, "the host's memory has not been read"
	}

	c := in.Base
	c.TensorParallelSize = tp
	c.MaxModelLen = target
	c.GPUMemoryUtilization = d.GPUMemoryUtilization
	c.MaxNumSeqs = d.MaxNumSeqs
	c.KVCacheDtype = dtype
	c.KVCacheMemory = 0
	// Estimated without the flag: what is on the cards before offload, and
	// what is in RAM already (a PLE table).
	c.ExtraFlags = withoutFlag(c.ExtraFlags, ExpertOffloadFlag)
	droppedMTP := false
	if ref, ok := ParseSpeculative(c.SpeculativeConfig); ok && ref.LocalDraft() == "" && strings.Contains(strings.ToLower(ref.Method), "mtp") {
		c.SpeculativeConfig, droppedMTP = "", true
	}
	est := in.Estimate(c)
	if est.Unknown || est.KVCachePerTokenB <= 0 {
		return WidthPlan{}, false, ""
	}

	experts := expertWeightGB(m.HFConfig)
	nonExpert := max(est.DeviceWeightsGB-experts, 0) / float64(tp)
	if tp >= 2 {
		nonExpert *= replicationOverhead
	}
	budget := d.GPUMemoryUtilization * in.Inventory.usableGB()
	layers := max(1, m.HFConfig.NumHiddenLayers-max(0, m.HFConfig.DenseLayers))
	minCache := 2 * experts / float64(tp) * float64(residentExpertLayers) / float64(layers)

	// The context: the target, or as much as leaves the cache its minimum.
	kvPerTokenRank := float64(est.KVCachePerTokenB) * est.KVScale(tp) / float64(tp)
	room := budget - nonExpert - offloadGraphsGB - offloadRuntimeGB - minCache
	ctx := target
	if most := int(room * (1 << 30) / (kvPerTokenRank * offloadKVReserve)); most < ctx {
		ctx = most / contextStep * contextStep
	}
	if ctx < minPlanContext {
		return WidthPlan{}, false, "even with the experts in system RAM, the rest does not fit on the cards"
	}

	hostNeed := experts + est.HostResidentGB
	hostHas := offloadHostShare*in.HostRAMGB - offloadRankProcessGB*float64(tp) - offloadEngineGB
	if hostNeed > hostHas {
		return WidthPlan{}, false, "the experts do not fit in the host's memory"
	}

	c.MaxModelLen = ctx
	c.ExtraFlags = strings.TrimSpace(c.ExtraFlags + " " + ExpertOffloadFlag)
	kv := kvPerTokenRank * float64(ctx) * offloadKVReserve / (1 << 30)
	cache := budget - nonExpert - offloadGraphsGB - offloadRuntimeGB - kv
	p := WidthPlan{
		TP: tp, Config: c, ContextTokens: ctx, ContextReduced: ctx < target,
		FullContextRequests: 1, Offload: true,
		RequiredGB: budget * float64(tp), RequiredHighGB: budget * float64(tp), AvailableGB: budget * float64(tp),
	}
	p.Notes = widthNotes(c, p, target, dtype, in.CardKVDtype != "", false, "")
	p.Notes = append(p.Notes, ProfileNote{
		Field: "extra_flags", Origin: "this machine",
		Reason: offloadReason(experts, hostNeed-experts, cache*float64(tp)),
	})
	if droppedMTP {
		p.Notes = append(p.Notes, ProfileNote{
			Field: "speculative_config", Origin: "this machine",
			Reason: "MTP is left out with expert offload: this image sizes the offload's KV reserve without the MTP layer, and with it the start fails short of cache.",
		})
	}
	return p, true, ""
}

func offloadReason(experts, other, cacheGB float64) string {
	s := "Expert offload: the model's experts, " + gb(experts) + ", are held in system RAM"
	if other > 0 {
		s += " beside " + gb(other) + " already there"
	}
	return s + ", and the cards keep " + gb(cacheGB) + " of them cached. The cache holds one full-length request at a time, and generation is slower: Flash-Next decoded about 54 tokens a second on two cards with offload, against 153 on four without it."
}

func gb(v float64) string { return fmt.Sprintf("%.1f GB", v) }

// withoutFlag is an extra-flags string with a bare flag taken out wherever it
// stands alone, the rest left exactly as written: quoted JSON values in it
// keep their spacing.
func withoutFlag(flags, flag string) string {
	re := regexp.MustCompile(`(^|\s)` + regexp.QuoteMeta(flag) + `(\s|$)`)
	for re.MatchString(flags) {
		flags = re.ReplaceAllString(flags, " ")
	}
	return strings.TrimSpace(flags)
}

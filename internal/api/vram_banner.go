package api

import (
	"fmt"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// vramBanner is the headline above the config panel: what this configuration
// costs in GPU memory, how that figure is arrived at, and whether this host
// can supply it.
//
// The headline is a total across every card the model will be split over, not
// a per-card share. What one rank holds is an implementation detail of the
// split; the question is whether the configuration can run, and that is one
// number that moves as the settings beneath it are edited.
type vramBanner struct {
	Unknown bool
	Why     string

	ParamLabel  string
	ActiveLabel string
	// Caveat qualifies the parameter count when it is known to be understated
	// — an MoE whose expert shape the config does not describe. The memory
	// figures beside it still stand: those are measured from the files.
	Caveat string

	// Measured says the figures came from a real start rather than from
	// arithmetic over the checkpoint's shape. It is the most important thing
	// on the panel: the projected path is the same arithmetic that was wrong
	// about a real model by a third, and the two must not look alike.
	//
	// MeasuredWhen and MeasuredTP describe that run, so the reader can judge
	// how old it is and at what width it holds.
	Measured     bool
	MeasuredWhen string
	MeasuredTP   int

	// RequiredGB is the headline. Ranged marks it widened by an offload whose
	// size the configuration does not state.
	RequiredGB     float64
	RequiredLowGB  float64
	RequiredHighGB float64
	Ranged         bool

	// The working behind it.
	CheckpointGB   float64
	WeightsTotalGB float64
	KVAtContextGB  float64
	OverheadGB     float64
	ContextTokens  int

	// HostResidentGB is the part offload keeps in system RAM. Shown whenever
	// any offload is active, because it is the difference between the size on
	// disk and what the GPUs are actually asked for.
	HostResidentGB    float64
	HostResidentMinGB float64
	HostBanded        bool
	OffloadActive     bool
	OffloadLabel      string

	// Available is what this host permits at the configured width, and
	// HasInventory says whether there was anything to compare against.
	HasInventory bool
	AvailableGB  float64
	TotalVRAMGB  float64
	ConfiguredTP int
}

// vramTPRow is one tensor-parallel width: what the model costs in total at
// that width, and what that many cards can give it.
type vramTPRow struct {
	TP             int
	Configured     bool
	Recommended    bool
	WeightsGB      float64
	KVGB           float64
	OverheadGB     float64
	RequiredGB     float64
	AvailableGB    float64
	Fits           bool
	Uncertain      bool
	ConcurrentSeqs int
	// Measured marks the one row a real start produced. The others project it
	// onto a width nothing has run at.
	Measured bool
}

func newVRAMBanner(est models.VRAMEstimate, fit models.VRAMFit) vramBanner {
	if est.Unknown {
		return vramBanner{Unknown: true, Why: est.UnknownWhy}
	}

	b := vramBanner{
		Measured:       est.Source == models.SourceMeasured,
		MeasuredTP:     est.MeasuredTP,
		ParamLabel:     models.FormatParamCount(est.ParamCountBillion),
		RequiredGB:     est.TotalRequiredGB,
		RequiredLowGB:  est.TotalRequiredLowGB,
		RequiredHighGB: est.TotalRequiredHighGB,
		Ranged:         est.TotalRequiredHighGB > est.TotalRequiredLowGB+0.05,

		CheckpointGB:   est.CheckpointGB,
		WeightsTotalGB: est.WeightsTotalGB,
		KVAtContextGB:  est.KVAtContextGB,
		ContextTokens:  est.ContextTokens,

		HostResidentGB:    est.HostResidentGB,
		HostResidentMinGB: est.HostResidentMinGB,
		HostBanded:        est.HostResidentGB > est.HostResidentMinGB+0.05,
		OffloadActive:     est.Offload.Any(),
		OffloadLabel:      offloadLabel(est.Offload),
		Caveat:            est.Caveat,

		ConfiguredTP: fit.ConfiguredTP,
	}
	b.OverheadGB = est.TotalRequiredGB - b.WeightsTotalGB - b.KVAtContextGB
	if b.OverheadGB < 0 {
		b.OverheadGB = 0
	}
	if est.ActiveParamBillion > 0 {
		b.ActiveLabel = models.FormatParamCount(est.ActiveParamBillion)
	}
	if !est.MeasuredAt.IsZero() {
		b.MeasuredWhen = est.MeasuredAt.Format("2 Jan 15:04")
	}

	if fit.Known {
		b.HasInventory = true
		b.AvailableGB = fit.AvailableGB
		b.TotalVRAMGB = fit.TotalVRAMGB
	} else {
		b.Why = fit.Why
	}
	return b
}

func newVRAMTPRows(est models.VRAMEstimate, fit models.VRAMFit) []vramTPRow {
	if est.Unknown || !fit.Known {
		return nil
	}

	rows := make([]vramTPRow, 0, len(fit.Options))
	for _, o := range fit.Options {
		rows = append(rows, vramTPRow{
			TP:             o.TP,
			Configured:     o.Configured,
			Recommended:    o.Recommended,
			WeightsGB:      o.WeightsGB,
			KVGB:           o.KVGB,
			OverheadGB:     o.OverheadGB,
			RequiredGB:     o.RequiredGB,
			AvailableGB:    o.AvailableGB,
			Fits:           o.Fits,
			Uncertain:      o.Uncertain,
			ConcurrentSeqs: o.ConcurrentSeqs,
			Measured:       o.Measured,
		})
	}
	return rows
}

// offloadLabel names what is being kept off the cards, or "" when nothing is.
func offloadLabel(o models.Offload) string {
	var parts []string
	if o.PLE {
		// Estimated from the residual, never read from the configuration.
		parts = append(parts, "embedding table, estimated")
	}
	if o.Experts {
		switch {
		case o.ExpertCapSet:
			parts = append(parts, fmt.Sprintf("experts (up to %.0f GB)", o.ExpertHostCapGB))
		default:
			parts = append(parts, "experts (no ceiling set)")
		}
		if o.ExpertCacheSet {
			parts = append(parts, fmt.Sprintf("%.1f GB cache on card", o.ExpertCacheGB))
		}
	}
	if o.NVMe {
		parts = append(parts, "NVMe — not modelled")
	}

	switch len(parts) {
	case 0:
		return ""
	case 1:
		return parts[0]
	default:
		out := parts[0]
		for _, p := range parts[1:] {
			out += ", " + p
		}
		return out
	}
}

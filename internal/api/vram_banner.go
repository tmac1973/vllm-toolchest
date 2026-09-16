package api

import (
	"fmt"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// vramBanner is the headline above the config panel: what the checkpoint is,
// what of it the cards have to hold, and what this host makes of that.
type vramBanner struct {
	Unknown bool
	Why     string

	ParamLabel  string
	ActiveLabel string
	// Caveat qualifies the parameter count when it is known to be understated
	// — an MoE whose expert shape the config does not describe. The memory
	// figures beside it still stand: those are measured from the files.
	Caveat string

	CheckpointGB float64
	// HostResidentGB is the part offload keeps in system RAM, and
	// HostResidentMinGB the least it could be. Shown whenever any offload is
	// active, because it is the difference between the size on disk and what
	// the GPUs are actually asked for — the conflation that had a working
	// model reported as too large for the host running it.
	HostResidentGB    float64
	HostResidentMinGB float64
	HostBanded        bool
	OffloadActive     bool
	OffloadLabel      string

	DeviceWeightsGB float64
	// Ranged says the device figure is a band: offload is on but its size is
	// not knowable from the configuration alone.
	Ranged              bool
	DeviceWeightsHighGB float64

	// NoInventory means there are figures but no cards to judge them against.
	NoInventory bool
	Verdict     string
	VerdictHue  string
}

// vramTPRow is one tensor-parallel width: what each card would hold, what is
// left for the KV cache, and how much context that buys.
type vramTPRow struct {
	TP              int
	Configured      bool
	Recommended     bool
	WeightsPerGPUGB float64
	OverheadGB      float64
	LoadGB          float64
	BudgetGB        float64
	KVHeadroomGB    float64
	MaxTokens       int
	Verdict         string
	Hue             string
}

func newVRAMBanner(est models.VRAMEstimate, fit models.VRAMFit) vramBanner {
	if est.Unknown {
		return vramBanner{Unknown: true, Why: est.UnknownWhy}
	}

	b := vramBanner{
		ParamLabel:          models.FormatParamCount(est.ParamCountBillion),
		CheckpointGB:        est.CheckpointGB,
		HostResidentGB:      est.HostResidentGB,
		HostResidentMinGB:   est.HostResidentMinGB,
		HostBanded:          est.HostResidentGB > est.HostResidentMinGB+0.05,
		OffloadActive:       est.Offload.Any(),
		DeviceWeightsGB:     est.DeviceWeightsGB,
		DeviceWeightsHighGB: est.DeviceWeightsHighGB,
		Ranged:              est.Ranged(),
		OffloadLabel:        offloadLabel(est.Offload),
		Caveat:              est.Caveat,
	}
	if est.ActiveParamBillion > 0 {
		b.ActiveLabel = models.FormatParamCount(est.ActiveParamBillion)
	}

	switch {
	case !fit.Known:
		b.NoInventory = true
		b.Verdict = fit.Why
		b.VerdictHue = "#6b7280"
	default:
		b.Verdict = fit.Label
		b.VerdictHue = verdictHue(fit)
	}
	return b
}

func newVRAMTPRows(est models.VRAMEstimate, fit models.VRAMFit) []vramTPRow {
	if est.Unknown || !fit.Known {
		return nil
	}

	rows := make([]vramTPRow, 0, len(fit.Options))
	for _, o := range fit.Options {
		row := vramTPRow{
			TP:              o.TP,
			Configured:      o.TP == fit.ConfiguredTP,
			Recommended:     o.TP == fit.RecommendedTP,
			WeightsPerGPUGB: o.WeightsPerGPUGB,
			OverheadGB:      o.ActivationGB + o.GraphsGB,
			LoadGB:          o.LoadGB,
			BudgetGB:        o.BudgetGB,
			KVHeadroomGB:    o.KVHeadroomGB,
			MaxTokens:       o.MaxTokens,
		}

		switch {
		case o.Uncertain:
			row.Verdict, row.Hue = "depends on offload", "#6b7280"
		case o.ServesConfigured:
			row.Verdict, row.Hue = "fits", "#2d8a4e"
		case o.Loads:
			row.Verdict, row.Hue = "loads, context won't fit", "#b86e00"
		default:
			row.Verdict, row.Hue = "too large", "#b83d3d"
		}
		rows = append(rows, row)
	}
	return rows
}

func verdictHue(fit models.VRAMFit) string {
	switch {
	case fit.Configured == nil, fit.RecommendedTP == 0:
		return "#b83d3d"
	case fit.Configured.Uncertain:
		return "#6b7280"
	case !fit.Configured.ServesConfigured:
		return "#b86e00"
	}
	return "#2d8a4e"
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

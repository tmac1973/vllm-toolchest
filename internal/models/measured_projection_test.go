package models

import "testing"

// The invariant whose absence let the bug ship: adding ranks cannot reduce what
// the model needs in total. Every extra rank carries another copy of the
// replicated tensors and captures another graph ladder, so the total rises.
//
// What the panel actually showed for a measured model was TP=1 at 112.9,
// TP=2 at 123.6 and TP=4 at 113.9 -- two cards needing more than four. The
// cause was units: WeightsTotalGB is a total with replication already inside
// it, and RequiredAt read it as a pre-replication figure, so TP=1 paid no
// surcharge and TP=2 paid it twice.
func TestAddingRanksNeverReducesWhatIsNeeded(t *testing.T) {
	m := measuredModel()
	est, ok := MeasuredEstimate(m)
	if !ok {
		t.Fatal("no measured estimate")
	}

	fit := Fit(est, m.VLLMConfig, GPUInventory{
		Count: 8, PerCardGB: 31.86, FreePerCardGB: 31.86, Known: true,
	})
	if len(fit.Options) < 3 {
		t.Fatalf("got %d widths, need several for this to mean anything", len(fit.Options))
	}

	prev := 0.0
	for _, o := range fit.Options {
		t.Logf("TP=%d measured=%v weights=%.1f kv=%.1f overhead=%.1f required=%.1f",
			o.TP, o.Measured, o.WeightsGB, o.KVGB, o.OverheadGB, o.RequiredGB)
		if o.RequiredGB+0.01 < prev {
			t.Errorf("TP=%d needs %.1f GB, less than the narrower split's %.1f -- "+
				"more ranks cannot need less memory in total", o.TP, o.RequiredGB, prev)
		}
		prev = o.RequiredGB
	}
}

// The width that actually ran keeps its measured total exactly; the others are
// projections of it and must not equal it by accident.
func TestTheMeasuredWidthKeepsItsFigureAndOthersMove(t *testing.T) {
	m := measuredModel()
	est, _ := MeasuredEstimate(m)
	fit := Fit(est, m.VLLMConfig, GPUInventory{
		Count: 8, PerCardGB: 31.86, FreePerCardGB: 31.86, Known: true,
	})

	for _, o := range fit.Options {
		switch {
		case o.TP == est.MeasuredTP:
			if !o.Measured {
				t.Errorf("TP=%d ran, but is not marked measured", o.TP)
			}
			if o.RequiredGB != est.TotalRequiredGB {
				t.Errorf("TP=%d total %.2f != measured %.2f", o.TP, o.RequiredGB, est.TotalRequiredGB)
			}
		default:
			if o.Measured {
				t.Errorf("TP=%d claims to be measured; the run was at TP=%d", o.TP, est.MeasuredTP)
			}
			if o.RequiredGB == est.TotalRequiredGB {
				t.Errorf("TP=%d projects to exactly the measured total; it is not being projected", o.TP)
			}
		}
	}
}

// The split between what shards and what every rank replicates, which is the
// whole of the projection.
func TestProjectWeights(t *testing.T) {
	// A run at four ranks totalling 98.28 GB, with the calibrated 10%
	// surcharge: ~89.35 shards, ~2.23 is replicated onto each rank.
	const measured, measuredTP = 98.28, 4

	if got := projectWeights(measured, measuredTP, measuredTP); got != measured {
		t.Errorf("at the measured width got %.2f, want the measurement itself %.2f", got, measured)
	}

	narrower := projectWeights(measured, measuredTP, 2)
	wider := projectWeights(measured, measuredTP, 8)
	if !(narrower < measured && measured < wider) {
		t.Errorf("not monotonic in width: TP=2 %.2f, TP=4 %.2f, TP=8 %.2f", narrower, measured, wider)
	}

	// Every rank's replicated share is the same, so the step between widths is
	// constant.
	if stepDown, stepUp := measured-narrower, wider-measured; stepUp < stepDown*1.9 || stepUp > stepDown*2.1 {
		t.Errorf("steps are not proportional to the change in width: down %.2f over 2 ranks, up %.2f over 4",
			stepDown, stepUp)
	}

	// A measurement taken at one rank has no replication in it to find, so
	// there is nothing to scale and it is carried across flat. Understating,
	// and documented as such.
	if got := projectWeights(50, 1, 4); got != 50 {
		t.Errorf("projecting from a single rank gave %.2f, want the flat 50 it cannot improve on", got)
	}
}

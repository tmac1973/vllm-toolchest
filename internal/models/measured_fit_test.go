package models

import "testing"

// At the width a real start ran at, the stored totals are the answer. Passing
// them back through RequiredAt would corrupt them twice over: it replaces the
// measured graph pool (0.49/rank) with its own 0.9/rank constant, and applies
// the 1.10 replication surcharge to a consumed figure that already carries the
// allocator's overhead. Both are corrections to a projection, and a
// measurement has nothing left to correct.
func TestTheMeasuredWidthIsNotReDerived(t *testing.T) {
	m := measuredModel()
	est, ok := MeasuredEstimate(m, EngineIdentity{})
	if !ok {
		t.Fatal("no measured estimate")
	}

	fit := Fit(est, m.VLLMConfig, GPUInventory{Count: 4, PerCardGB: 31.86, Known: true})
	if fit.Configured == nil {
		t.Fatal("no option at the configured width")
	}

	o := *fit.Configured
	if !o.Measured {
		t.Fatal("the row for the width that actually ran is not marked as measured")
	}
	if o.RequiredGB != est.TotalRequiredGB {
		t.Errorf("row total %.2f != measured total %.2f -- it was re-derived",
			o.RequiredGB, est.TotalRequiredGB)
	}
	if o.RequiredHigh != o.RequiredGB {
		t.Error("a measured figure was given a band")
	}
	if o.Uncertain {
		t.Error("a measured row reports itself uncertain")
	}

	// The parts must add up to the whole, or the panel's breakdown lies.
	if sum := o.WeightsGB + o.KVGB + o.OverheadGB; sum < o.RequiredGB-0.05 || sum > o.RequiredGB+0.05 {
		t.Errorf("weights+kv+overhead = %.2f, total = %.2f", sum, o.RequiredGB)
	}

	// The corruption this guards against: RequiredAt would inflate the
	// measured figure by the graph constant and the replication surcharge.
	projected := RequiredAt(est, m.VLLMConfig, est.MeasuredTP)
	if projected.TotalGB <= est.TotalRequiredGB+1 {
		t.Skip("re-deriving no longer inflates; this guard has stopped testing anything")
	}
	t.Logf("re-deriving would have reported %.1f GB against the measured %.1f",
		projected.TotalGB, est.TotalRequiredGB)
}

// Other widths are projections of a measurement onto something nothing has run
// at, and must not claim otherwise.
func TestOtherWidthsStayProjections(t *testing.T) {
	m := measuredModel()
	est, _ := MeasuredEstimate(m, EngineIdentity{})
	fit := Fit(est, m.VLLMConfig, GPUInventory{Count: 4, PerCardGB: 31.86, Known: true})

	measured := 0
	for _, o := range fit.Options {
		if o.Measured {
			measured++
			if o.TP != est.MeasuredTP {
				t.Errorf("TP=%d claims to be measured; the run was at TP=%d", o.TP, est.MeasuredTP)
			}
		}
	}
	if measured != 1 {
		t.Errorf("%d rows claim to be measured, want exactly 1", measured)
	}
	if len(fit.Options) < 2 {
		t.Fatal("only one width was offered; this case needs several to mean anything")
	}
}

package benchmark

import "testing"

// The payoff of recording the profile: two runs of one model whose only
// difference is the config the operator named show that name as the varying
// column, rather than looking identical.
func TestCompareShowsTheProfileWhenRunsDiffer(t *testing.T) {
	a := cmpRun("a", "m", "fp8", "p", 8192, 1, 10)
	b := cmpRun("b", "m", "fp8", "p", 8192, 1, 12)
	a.Config.ProfileName = "baseline"
	b.Config.ProfileName = "mtp-8"
	b.Config.ProfileModified = true
	b.Config.SpeculativeConfig = `{"method":"mtp","num_speculative_tokens":8}`

	c := BuildCompare([]BenchmarkRun{a, b})
	varying := colKeys(c.Varying)
	for _, want := range []string{"profile", "speculative_config"} {
		if !has(varying, want) {
			t.Errorf("%s differs between the runs; varying = %v", want, varying)
		}
	}
	if c.Identical {
		t.Error("runs with different profiles were flagged identical")
	}

	col := -1
	for i, cc := range c.Columns {
		if cc.Key == "profile" {
			col = i
		}
	}
	for _, row := range c.Rows {
		if row.RunID == "b" && row.Cells[col] != "mtp-8 (edited)" {
			t.Errorf("an edited profile reads %q, want it marked as edited", row.Cells[col])
		}
	}
}

// A setting no selected run used must add no column, constant or otherwise:
// every dimension added is one more thing to read past.
func TestCompareAddsNoColumnForSettingsNoRunUsed(t *testing.T) {
	c := BuildCompare([]BenchmarkRun{
		cmpRun("a", "m", "fp8", "p", 8192, 1, 10),
		cmpRun("b", "m", "fp8", "p", 16384, 1, 12),
	})
	for _, key := range []string{
		"profile", "speculative_config", "attention_backend",
		"prefix_caching", "chunked_prefill", "max_num_batched_tokens",
	} {
		if has(colKeys(c.Columns), key) {
			t.Errorf("column %q appeared though no run set it", key)
		}
	}
}

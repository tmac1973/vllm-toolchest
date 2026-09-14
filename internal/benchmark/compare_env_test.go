package benchmark

import "testing"

// A kernel path switched on for one run and not the other is invisible in
// every other recorded field, so the environment has to be a dimension of its
// own — otherwise the two runs look identical and the difference looks like
// noise.
func TestCompareShowsTheEnvironmentWhenRunsDiffer(t *testing.T) {
	a := cmpRun("a", "m", "fp8", "p", 8192, 1, 10)
	b := cmpRun("b", "m", "fp8", "p", 8192, 1, 14)
	b.Config.Env = "CLAV_GDN=1"

	c := BuildCompare([]BenchmarkRun{a, b})
	if !has(colKeys(c.Varying), "env") {
		t.Errorf("the environment differs between the runs; varying = %v", colKeys(c.Varying))
	}

	col := -1
	for i, cc := range c.Columns {
		if cc.Key == "env" {
			col = i
		}
	}
	for _, row := range c.Rows {
		if row.RunID == "b" && row.Cells[col] != "CLAV_GDN=1" {
			t.Errorf("env cell = %q, want CLAV_GDN=1", row.Cells[col])
		}
		if row.RunID == "a" && row.Cells[col] != "" {
			t.Errorf("a run that set nothing shows %q, want empty", row.Cells[col])
		}
	}
}

// Runs that set no environment add no column at all.
func TestCompareAddsNoEnvironmentColumnWhenUnused(t *testing.T) {
	c := BuildCompare([]BenchmarkRun{
		cmpRun("a", "m", "fp8", "p", 8192, 1, 10),
		cmpRun("b", "m", "fp8", "p", 16384, 1, 12),
	})
	if has(colKeys(c.Columns), "env") {
		t.Error("an environment column appeared though neither run set one")
	}
}

// The stored block keeps its newlines; only the display form is flattened, so
// a table cell or CSV field stays on one line.
func TestEnvSummaryFlattensTheBlock(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"", ""},
		{"CLAV_GDN=1", "CLAV_GDN=1"},
		{"# why\nCLAV_GDN=1\n\nVLLM_PLE_CPU_OFFLOAD=1\n", "CLAV_GDN=1; VLLM_PLE_CPU_OFFLOAD=1"},
	} {
		if got := (ConfigSnapshot{Env: tc.in}).EnvSummary(); got != tc.want {
			t.Errorf("EnvSummary(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

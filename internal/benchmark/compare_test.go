package benchmark

import "testing"

func cmpRun(id, model, quant, preset string, ctx, tp int, gen float64) BenchmarkRun {
	r := BenchmarkRun{
		ID: id, ModelName: model, Quant: quant, Preset: preset,
		Config: ConfigSnapshot{
			MaxModelLen: ctx, TensorParallelSize: tp,
			GPUMemoryUtilization: 0.9, KVCacheDtype: "fp8", Dtype: "auto",
		},
		VLLMVersion: "0.9.3",
	}
	if gen > 0 {
		r.Summary = &BenchmarkSummary{AvgGenTokPerSec: gen, AvgPromptTokPerSec: gen * 3, AvgTTFTMs: 900}
	}
	return r
}

func colKeys(cols []CompareColumn) []string {
	out := make([]string, 0, len(cols))
	for _, c := range cols {
		out = append(out, c.Key)
	}
	return out
}

func has(keys []string, want string) bool {
	for _, k := range keys {
		if k == want {
			return true
		}
	}
	return false
}

// The point of a comparison is the thing that differs. Repeating the model,
// quantization, GPU and vLLM version down every row buries one varying
// parameter in a dozen identical columns.
func TestBuildCompareSplitsVaryingFromShared(t *testing.T) {
	runs := []BenchmarkRun{
		cmpRun("r1", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 110),
		cmpRun("r2", "Qwen3.8-27B", "fp8", "internal-quick", 32768, 2, 96),
		cmpRun("r3", "Qwen3.8-27B", "fp8", "internal-quick", 65536, 2, 88),
	}
	c := BuildCompare(runs)

	varying := colKeys(c.Varying)
	if len(varying) != 1 || varying[0] != "max_model_len" {
		t.Fatalf("varying columns = %v, want just max_model_len", varying)
	}
	common := colKeys(c.Common)
	for _, want := range []string{"model", "quant", "preset", "tensor_parallel_size", "vllm_version"} {
		if !has(common, want) {
			t.Errorf("%q is the same for every run and should be listed as shared", want)
		}
	}
	if has(common, "max_model_len") {
		t.Error("the varying column must not also be listed as shared")
	}
	if c.Identical {
		t.Error("these runs differ in context length, so Identical must be false")
	}
}

// Runs that differ in nothing recorded are worth saying so about: it usually
// means the wrong ones were selected, and any gap is variance.
func TestBuildCompareFlagsIdenticalRuns(t *testing.T) {
	a := cmpRun("r1", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 110)
	b := cmpRun("r2", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 108)
	c := BuildCompare([]BenchmarkRun{a, b})
	if !c.Identical {
		t.Errorf("runs differing only in their measurements should be flagged identical; varying = %v", colKeys(c.Varying))
	}
}

func TestBuildCompareRanksFastestFirst(t *testing.T) {
	runs := []BenchmarkRun{
		cmpRun("slow", "Qwen3.8-27B", "fp8", "internal-quick", 65536, 2, 88),
		cmpRun("fast", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 110),
		cmpRun("mid", "Qwen3.8-27B", "fp8", "internal-quick", 32768, 2, 96),
	}
	c := BuildCompare(runs)

	if c.Rows[0].RunID != "fast" || !c.Rows[0].Best {
		t.Fatalf("first row = %q best=%v, want fast", c.Rows[0].RunID, c.Rows[0].Best)
	}
	if c.Rows[0].Rank != 1 || c.Rows[1].Rank != 2 || c.Rows[2].Rank != 3 {
		t.Errorf("ranks = %d,%d,%d, want 1,2,3", c.Rows[0].Rank, c.Rows[1].Rank, c.Rows[2].Rank)
	}
	if c.Rows[0].GenBarPct != 100 {
		t.Errorf("the fastest run's bar = %.1f%%, want 100", c.Rows[0].GenBarPct)
	}
	if c.Rows[0].DeltaPct != 0 {
		t.Errorf("the fastest run's delta = %.1f, want 0", c.Rows[0].DeltaPct)
	}
	// 96 against 110 is 12.7% slower.
	if d := c.Rows[1].DeltaPct; d > -12.6 || d < -12.8 {
		t.Errorf("second row delta = %.2f%%, want about -12.7%%", d)
	}
}

// A failed run has no numbers to rank, but hiding it would misrepresent what
// was actually attempted.
func TestBuildCompareKeepsFailedRunsLast(t *testing.T) {
	runs := []BenchmarkRun{
		cmpRun("failed", "Qwen3.8-27B", "fp8", "internal-quick", 131072, 2, 0),
		cmpRun("ok", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 110),
	}
	c := BuildCompare(runs)

	if c.Rows[0].RunID != "ok" {
		t.Errorf("first row = %q, want the measured run", c.Rows[0].RunID)
	}
	last := c.Rows[len(c.Rows)-1]
	if last.RunID != "failed" || !last.Failed {
		t.Errorf("last row = %q failed=%v, want the failed run", last.RunID, last.Failed)
	}
	if last.Rank != 0 {
		t.Errorf("a failed run should have no rank, got %d", last.Rank)
	}
}

// The label names a run by what makes it distinct — the shared conditions are
// stated once above the table, so repeating them here wastes the width.
func TestCompareLabelUsesOnlyTheVaryingDimensions(t *testing.T) {
	runs := []BenchmarkRun{
		cmpRun("r1", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 110),
		cmpRun("r2", "Qwen3.8-27B", "fp8", "internal-quick", 32768, 2, 96),
	}
	c := BuildCompare(runs)
	if c.Rows[0].Label != "8192" {
		t.Errorf("label = %q, want just the context length", c.Rows[0].Label)
	}
}

// With nothing varying there is no distinguishing value to name, so the label
// falls back to identity rather than rendering blank.
func TestCompareLabelFallsBackToIdentity(t *testing.T) {
	a := cmpRun("r1", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 110)
	b := cmpRun("r2", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 108)
	c := BuildCompare([]BenchmarkRun{a, b})
	for _, row := range c.Rows {
		if row.Label == "" {
			t.Fatal("a row label must never be empty")
		}
	}
}

// An absent value is a difference. Two runs where one recorded its vLLM
// version and the other did not are not interchangeable, and folding them into
// "shared" would say they were.
func TestBuildCompareTreatsMissingValuesAsVariation(t *testing.T) {
	a := cmpRun("r1", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 110)
	b := cmpRun("r2", "Qwen3.8-27B", "fp8", "internal-quick", 8192, 2, 108)
	b.VLLMVersion = ""
	c := BuildCompare([]BenchmarkRun{a, b})
	if !has(colKeys(c.Varying), "vllm_version") {
		t.Errorf("vllm_version differs between the runs; varying = %v", colKeys(c.Varying))
	}
}

// The comparison table renders every dimension and hides the constant ones,
// so Cells has to line up with Columns rather than with Varying — a mismatch
// puts each row's values under the wrong headings.
func TestCompareCellsAlignWithColumns(t *testing.T) {
	runs := []BenchmarkRun{
		{ID: "a", ModelID: "m", Preset: "p", Config: ConfigSnapshot{MaxModelLen: 8192},
			Summary: &BenchmarkSummary{AvgGenTokPerSec: 40, AvgPromptTokPerSec: 900}},
		{ID: "b", ModelID: "m", Preset: "p", Config: ConfigSnapshot{MaxModelLen: 32768},
			Summary: &BenchmarkSummary{AvgGenTokPerSec: 30, AvgPromptTokPerSec: 1100}},
	}
	c := BuildCompare(runs)
	if len(c.Columns) == 0 {
		t.Fatal("no columns")
	}
	for _, row := range c.Rows {
		if len(row.Cells) != len(c.Columns) {
			t.Fatalf("%s has %d cells for %d columns", row.RunID, len(row.Cells), len(c.Columns))
		}
	}

	// Every varying dimension appears as a non-constant column, and the
	// constants are the rest.
	varying := map[string]bool{}
	for _, v := range c.Varying {
		varying[v.Key] = true
	}
	for _, col := range c.Columns {
		if col.Constant == varying[col.Key] {
			t.Errorf("%s: Constant=%v but varying=%v", col.Key, col.Constant, varying[col.Key])
		}
	}
}

// Prefill and decode rank independently. This is the case the second chart
// exists for: the run that generates fastest is the one that prefills slowest.
func TestCompareRanksPromptSeparatelyFromGeneration(t *testing.T) {
	runs := []BenchmarkRun{
		{ID: "short", Config: ConfigSnapshot{MaxModelLen: 8192},
			Summary: &BenchmarkSummary{AvgGenTokPerSec: 46, AvgPromptTokPerSec: 980}},
		{ID: "long", Config: ConfigSnapshot{MaxModelLen: 65536},
			Summary: &BenchmarkSummary{AvgGenTokPerSec: 31, AvgPromptTokPerSec: 1210}},
	}
	c := BuildCompare(runs)
	if !c.HasPrompt {
		t.Fatal("prompt measurements present but HasPrompt is false")
	}
	byID := map[string]CompareRow{}
	for _, r := range c.Rows {
		byID[r.RunID] = r
	}
	if !byID["short"].Best {
		t.Error("the fastest generator should be Best")
	}
	if byID["short"].PromptBest {
		t.Error("the fastest generator is not the fastest at prefill here")
	}
	if !byID["long"].PromptBest {
		t.Error("the fastest prefill should be PromptBest")
	}
	if byID["long"].PromptBarPct != 100 {
		t.Errorf("the best prompt bar should be full; got %.1f", byID["long"].PromptBarPct)
	}
}

// Nothing measured prompt speed means no second chart, rather than one with
// every bar at zero.
func TestCompareHasPromptIsFalseWithoutPromptNumbers(t *testing.T) {
	c := BuildCompare([]BenchmarkRun{
		{ID: "a", Summary: &BenchmarkSummary{AvgGenTokPerSec: 40}},
		{ID: "b", Summary: nil},
	})
	if c.HasPrompt {
		t.Error("HasPrompt should be false when no run recorded prompt throughput")
	}
}

// A dimension no run recorded says nothing, so it is not offered as a column
// to reveal.
func TestCompareOmitsEmptyConstantColumns(t *testing.T) {
	c := BuildCompare([]BenchmarkRun{
		{ID: "a", Config: ConfigSnapshot{MaxModelLen: 8192}},
		{ID: "b", Config: ConfigSnapshot{MaxModelLen: 32768}},
	})
	for _, col := range c.Columns {
		if col.Constant && col.Value == "" {
			t.Errorf("%s is a constant column with no value", col.Key)
		}
	}
}

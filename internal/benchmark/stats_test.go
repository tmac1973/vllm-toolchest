package benchmark

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

// finite fails the test if any of vals is NaN or ±Inf. Either would reach the
// page as JSON that does not parse, or as a bar of no width.
func finite(t *testing.T, what string, vals ...float64) {
	t.Helper()
	for i, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("%s[%d] = %v, want a finite number", what, i, v)
		}
	}
}

func summaryValues(s *BenchmarkSummary) []float64 {
	return []float64{s.AvgPromptTokPerSec, s.AvgGenTokPerSec, s.AvgTTFTMs, s.MinGenTokPerSec, s.MaxGenTokPerSec}
}

// A run that measured nothing has no summary. Returning a zero summary would
// rank a failure as a run that generated at 0 tok/s.
func TestComputeSummaryOfNothingIsNil(t *testing.T) {
	if s := ComputeSummary(nil); s != nil {
		t.Errorf("nil results: got %+v", s)
	}
	if s := ComputeSummary([]BenchmarkResult{}); s != nil {
		t.Errorf("empty results: got %+v", s)
	}
}

// One measurement is its own average, minimum and maximum.
func TestComputeSummaryOfOneResultIsThatResult(t *testing.T) {
	r := BenchmarkResult{PromptTokPerSec: 2500, GenTokPerSec: 42.5, TTFTMs: 180}
	got := ComputeSummary([]BenchmarkResult{r})
	want := &BenchmarkSummary{
		AvgPromptTokPerSec: 2500, AvgGenTokPerSec: 42.5, AvgTTFTMs: 180,
		MinGenTokPerSec: 42.5, MaxGenTokPerSec: 42.5,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// Three prompt sizes, as the default preset runs. The averages are plain
// means over the points; min and max are over generation speed only.
func TestComputeSummaryAveragesEveryPoint(t *testing.T) {
	results := []BenchmarkResult{
		{PromptTokens: 512, PromptTokPerSec: 1000, GenTokPerSec: 60, TTFTMs: 100},
		{PromptTokens: 2048, PromptTokPerSec: 2000, GenTokPerSec: 50, TTFTMs: 300},
		{PromptTokens: 8192, PromptTokPerSec: 3000, GenTokPerSec: 40, TTFTMs: 800},
	}
	got := ComputeSummary(results)
	want := &BenchmarkSummary{
		AvgPromptTokPerSec: 2000, // (1000+2000+3000)/3
		AvgGenTokPerSec:    50,   // (60+50+40)/3
		AvgTTFTMs:          400,  // (100+300+800)/3
		MinGenTokPerSec:    40,
		MaxGenTokPerSec:    60,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// A point whose request returned no tokens and took no time records zeros.
// The minimum starts at MaxFloat64; it must come down to the zero rather than
// leak out as 1.8e308, and nothing may turn into NaN.
func TestComputeSummaryOfZeroTimingsStaysFinite(t *testing.T) {
	results := []BenchmarkResult{{}, {}}
	got := ComputeSummary(results)
	if got == nil {
		t.Fatal("got nil for two results")
	}
	finite(t, "summary", summaryValues(got)...)
	if *got != (BenchmarkSummary{}) {
		t.Errorf("got %+v, want all zeros", got)
	}

	// A zero next to a real point still counts as the slowest.
	got = ComputeSummary([]BenchmarkResult{{GenTokPerSec: 30}, {}})
	if got.MinGenTokPerSec != 0 || got.MaxGenTokPerSec != 30 || got.AvgGenTokPerSec != 15 {
		t.Errorf("got %+v, want min 0, max 30, avg 15", got)
	}
}

func TestBuildComparisonOfNoRunsIsEmpty(t *testing.T) {
	c := BuildComparison(nil)
	if c.Runs != nil || c.MaxGenTPS != 0 || c.MaxPromptTPS != 0 || c.HasBenchy {
		t.Errorf("got %+v, want the zero value", c)
	}
}

// The maxima scale the bars, so they are taken over runs with numbers only. A
// failed run has no summary and must neither panic nor count.
func TestBuildComparisonScalesToTheFastestRun(t *testing.T) {
	runs := []BenchmarkRun{
		{ID: "a", Summary: &BenchmarkSummary{AvgGenTokPerSec: 40, AvgPromptTokPerSec: 3000}},
		{ID: "failed"},
		{ID: "b", Summary: &BenchmarkSummary{AvgGenTokPerSec: 55, AvgPromptTokPerSec: 2500}},
	}
	c := BuildComparison(runs)
	if len(c.Runs) != 3 {
		t.Errorf("runs = %d, want all 3 kept for display", len(c.Runs))
	}
	// The fastest generator and the fastest prefiller are different runs.
	if c.MaxGenTPS != 55 || c.MaxPromptTPS != 3000 {
		t.Errorf("max gen %v, max prompt %v; want 55 and 3000", c.MaxGenTPS, c.MaxPromptTPS)
	}
	if c.HasBenchy {
		t.Error("HasBenchy set with no llama-benchy results")
	}
}

func TestBuildComparisonOfOneRun(t *testing.T) {
	c := BuildComparison([]BenchmarkRun{{Summary: &BenchmarkSummary{AvgGenTokPerSec: 12, AvgPromptTokPerSec: 900}}})
	if c.MaxGenTPS != 12 || c.MaxPromptTPS != 900 {
		t.Errorf("got %+v", c)
	}
}

// All-zero summaries leave the maxima at zero. Whatever divides by them later
// must guard; here it is enough that nothing is NaN.
func TestBuildComparisonOfZeroRunsStaysFinite(t *testing.T) {
	c := BuildComparison([]BenchmarkRun{{Summary: &BenchmarkSummary{}}, {Summary: &BenchmarkSummary{}}})
	finite(t, "max", c.MaxGenTPS, c.MaxPromptTPS)
	if c.MaxGenTPS != 0 || c.MaxPromptTPS != 0 {
		t.Errorf("got %+v, want zero maxima", c)
	}
}

// The llama-benchy columns are shown when any run has them -- including a
// run whose summary is missing, which is skipped before the check.
func TestBuildComparisonNoticesBenchyResults(t *testing.T) {
	withBenchy := BenchmarkRun{
		Summary:     &BenchmarkSummary{AvgGenTokPerSec: 10},
		LlamaBenchy: []LlamaBenchyResult{{Concurrency: 1}},
	}
	if c := BuildComparison([]BenchmarkRun{{Summary: &BenchmarkSummary{}}, withBenchy}); !c.HasBenchy {
		t.Error("HasBenchy not set")
	}
}

// vizRun is a run measured at one context length.
func vizRun(id string, ctx int, gen, prompt, ttft float64) BenchmarkRun {
	return BenchmarkRun{
		ID: id, ModelID: "Qwen/Qwen3-8B", Preset: "quick",
		Config:  ConfigSnapshot{MaxModelLen: ctx},
		Summary: &BenchmarkSummary{AvgGenTokPerSec: gen, AvgPromptTokPerSec: prompt, AvgTTFTMs: ttft},
	}
}

// The page needs a valid payload to say "nothing selected": metrics present,
// no points, nothing skipped.
func TestBuildVisualizationOfNoRuns(t *testing.T) {
	v := BuildVisualization(nil)
	if len(v.Dimensions) != 0 || len(v.Points) != 0 || v.Skipped != 0 {
		t.Errorf("got %+v", v)
	}
	if len(v.Metrics) != 3 {
		t.Errorf("metrics = %v, want gen, prompt and ttft", v.Metrics)
	}
}

// A single run varies in nothing, so there is no axis to plot it on, but it
// is still a point.
func TestBuildVisualizationOfOneRunHasAPointAndNoAxes(t *testing.T) {
	v := BuildVisualization([]BenchmarkRun{vizRun("r1", 32768, 50, 2000, 150)})
	if len(v.Dimensions) != 0 {
		t.Errorf("dimensions = %+v, want none", v.Dimensions)
	}
	if len(v.Points) != 1 {
		t.Fatalf("points = %d, want 1", len(v.Points))
	}
	p := v.Points[0]
	if p.RunID != "r1" || p.Label != "Qwen/Qwen3-8B" || p.Detail != "Qwen/Qwen3-8B · quick" {
		t.Errorf("point = %+v", p)
	}
	want := map[string]float64{"gen": 50, "prompt": 2000, "ttft": 150}
	if !reflect.DeepEqual(p.Metrics, want) {
		t.Errorf("metrics = %v, want %v", p.Metrics, want)
	}
}

// Context lengths are numbers and must be ordered as numbers: 131072 after
// 32768, not before it as a string sort would put it. Failures are counted,
// not plotted.
func TestBuildVisualizationOrdersANumericAxisNumerically(t *testing.T) {
	failed := vizRun("x", 65536, 0, 0, 0)
	failed.Summary = nil
	runs := []BenchmarkRun{
		vizRun("a", 131072, 30, 1500, 400),
		vizRun("b", 8192, 60, 3000, 90),
		vizRun("c", 32768, 45, 2200, 200),
		failed,
	}
	v := BuildVisualization(runs)
	if v.Skipped != 1 {
		t.Errorf("skipped = %d, want 1", v.Skipped)
	}
	if len(v.Points) != 3 {
		t.Fatalf("points = %d, want 3", len(v.Points))
	}
	if len(v.Dimensions) != 1 {
		t.Fatalf("dimensions = %+v, want only the context length", v.Dimensions)
	}
	d := v.Dimensions[0]
	// The failed run's context is still a value on the axis: the axis is
	// built from every selected run.
	if d.Name != "max_model_len" || !d.Numeric ||
		!reflect.DeepEqual(d.Values, []string{"8192", "32768", "65536", "131072"}) {
		t.Errorf("dimension = %+v", d)
	}
	for _, p := range v.Points {
		if p.Dims["max_model_len"] == "" {
			t.Errorf("point %s has no position on the axis", p.RunID)
		}
		if p.Label != p.Dims["max_model_len"] {
			t.Errorf("point %s label %q, want its varying value", p.RunID, p.Label)
		}
	}
}

// Model names are categories, sorted as text.
func TestBuildVisualizationKeepsANameAxisCategorical(t *testing.T) {
	a := vizRun("a", 8192, 50, 1, 1)
	b := vizRun("b", 8192, 40, 1, 1)
	b.ModelID = "Meta/Llama-3.1-8B"
	v := BuildVisualization([]BenchmarkRun{a, b})
	if len(v.Dimensions) != 1 || v.Dimensions[0].Numeric ||
		!reflect.DeepEqual(v.Dimensions[0].Values, []string{"Meta/Llama-3.1-8B", "Qwen/Qwen3-8B"}) {
		t.Errorf("dimensions = %+v", v.Dimensions)
	}
}

// Runs that measured zero everywhere still produce finite metrics, and the
// payload still encodes: encoding/json refuses NaN outright.
func TestBuildVisualizationOfZeroTimingsEncodes(t *testing.T) {
	v := BuildVisualization([]BenchmarkRun{vizRun("a", 8192, 0, 0, 0), vizRun("b", 16384, 0, 0, 0)})
	for _, p := range v.Points {
		for k, m := range p.Metrics {
			finite(t, p.RunID+"."+k, m)
		}
	}
	if _, err := json.Marshal(v); err != nil {
		t.Errorf("payload does not encode: %v", err)
	}
}

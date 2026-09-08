package benchmark

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Comparing runs is mostly an exercise in subtraction. Two runs of the same
// model at two context lengths differ in one thing, and a table that repeats
// the model, the quantization, the preset, the GPU and the vLLM version across
// every row buries that one thing in fifteen columns of identical text.
//
// So: work out which dimensions actually vary across the selected runs, show
// those as columns, and state the rest once above the table as the conditions
// they were all measured under.

// compareDimension is one thing runs can differ in, and how to read it off a
// run.
type compareDimension struct {
	Key   string
	Label string
	value func(BenchmarkRun) string
}

var compareDimensions = []compareDimension{
	{"model", "Model", func(r BenchmarkRun) string { return firstNonEmpty(r.ModelName, r.ModelID) }},
	{"quant", "Quant", func(r BenchmarkRun) string { return r.Quant }},
	{"preset", "Preset", func(r BenchmarkRun) string { return r.Preset }},
	{"max_model_len", "Context", func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.MaxModelLen) }},
	{"tensor_parallel_size", "TP", func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.TensorParallelSize) }},
	{"gpu_memory_utilization", "GPU mem", func(r BenchmarkRun) string {
		if r.Config.GPUMemoryUtilization == 0 {
			return ""
		}
		return strconv.FormatFloat(r.Config.GPUMemoryUtilization, 'g', -1, 64)
	}},
	{"max_num_seqs", "Max seqs", func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.MaxNumSeqs) }},
	{"kv_cache_dtype", "KV cache", func(r BenchmarkRun) string { return r.Config.KVCacheDtype }},
	{"dtype", "dtype", func(r BenchmarkRun) string { return r.Config.Dtype }},
	{"enforce_eager", "Eager", func(r BenchmarkRun) string { return strconv.FormatBool(r.Config.EnforceEager) }},
	{"prompt_sizes", "Prompt sizes", func(r BenchmarkRun) string { return intsText(r.PromptTokens) }},
	{"gen_tokens", "Gen tokens", func(r BenchmarkRun) string { return itoaOrEmpty(r.GenTokens) }},
	{"vllm_version", "vLLM", func(r BenchmarkRun) string { return r.VLLMVersion }},
	{"gpus", "GPUs", func(r BenchmarkRun) string {
		names := make([]string, 0, len(r.GPUs))
		for _, g := range r.GPUs {
			names = append(names, g.Name)
		}
		return strings.Join(names, " + ")
	}},
}

// CompareColumn is one dimension's heading, and for a shared dimension the one
// value every run had.
type CompareColumn struct {
	Key   string
	Label string
	Value string
}

// CompareRow is one run in the comparison.
type CompareRow struct {
	RunID  string
	Label  string
	Cells  []string
	Failed bool

	GenTPS    float64
	PromptTPS float64
	TTFTMs    float64

	// GenBarPct is this run's generation rate as a percentage of the fastest,
	// which is what the bar length encodes.
	GenBarPct float64
	// DeltaPct is the shortfall against the fastest run: 0 for the winner,
	// negative for the rest.
	DeltaPct float64
	Rank     int
	Best     bool
}

// Comparison is the whole view: what varied, what did not, and the rows.
type Comparison struct {
	Varying []CompareColumn
	Common  []CompareColumn
	Rows    []CompareRow
	// Identical is true when the runs differ in nothing this knows how to
	// name — worth saying out loud, since it usually means the wrong runs
	// were selected.
	Identical bool
}

// BuildCompare assembles the comparison, ordered fastest first.
func BuildCompare(runs []BenchmarkRun) Comparison {
	var c Comparison

	// A dimension is "varying" when the selected runs do not all agree on it.
	// An empty value counts: a run that recorded no vLLM version differs from
	// one that did, and hiding that would make the two look interchangeable.
	for _, d := range compareDimensions {
		seen := map[string]bool{}
		for _, r := range runs {
			seen[d.value(r)] = true
		}
		switch {
		case len(seen) > 1:
			c.Varying = append(c.Varying, CompareColumn{Key: d.Key, Label: d.Label})
		case len(runs) > 0:
			var only string
			for v := range seen {
				only = v
			}
			if only != "" {
				c.Common = append(c.Common, CompareColumn{Key: d.Key, Label: d.Label, Value: only})
			}
		}
	}
	c.Identical = len(c.Varying) == 0

	var maxGen float64
	for _, r := range runs {
		if r.Summary != nil && r.Summary.AvgGenTokPerSec > maxGen {
			maxGen = r.Summary.AvgGenTokPerSec
		}
	}

	for _, r := range runs {
		row := CompareRow{RunID: r.ID, Failed: r.Summary == nil}
		for _, col := range c.Varying {
			row.Cells = append(row.Cells, valueFor(r, col.Key))
		}
		row.Label = compareLabel(r, c.Varying)
		if r.Summary != nil {
			row.GenTPS = r.Summary.AvgGenTokPerSec
			row.PromptTPS = r.Summary.AvgPromptTokPerSec
			row.TTFTMs = r.Summary.AvgTTFTMs
			if maxGen > 0 {
				row.GenBarPct = row.GenTPS / maxGen * 100
				row.DeltaPct = (row.GenTPS - maxGen) / maxGen * 100
			}
		}
		c.Rows = append(c.Rows, row)
	}

	// Fastest first, failures last — a run with no numbers has nothing to
	// contribute to a ranking but should still be visible.
	sort.SliceStable(c.Rows, func(i, j int) bool {
		if c.Rows[i].Failed != c.Rows[j].Failed {
			return !c.Rows[i].Failed
		}
		return c.Rows[i].GenTPS > c.Rows[j].GenTPS
	})
	rank := 0
	for i := range c.Rows {
		if c.Rows[i].Failed {
			continue
		}
		rank++
		c.Rows[i].Rank = rank
		c.Rows[i].Best = rank == 1
	}
	return c
}

// compareLabel names a run by what makes it distinct — the varying dimensions
// only, since the shared ones are stated once above the table.
func compareLabel(r BenchmarkRun, varying []CompareColumn) string {
	parts := make([]string, 0, len(varying))
	for _, col := range varying {
		if v := valueFor(r, col.Key); v != "" {
			parts = append(parts, v)
		}
	}
	if len(parts) == 0 {
		// Nothing varies, so the only honest label is the run's identity.
		return firstNonEmpty(r.ModelName, r.ModelID, r.ID)
	}
	return strings.Join(parts, " · ")
}

func valueFor(r BenchmarkRun, key string) string {
	for _, d := range compareDimensions {
		if d.Key == key {
			return d.value(r)
		}
	}
	return ""
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func itoaOrEmpty(v int) string {
	if v == 0 {
		return ""
	}
	return strconv.Itoa(v)
}

func intsText(vals []int) string {
	if len(vals) == 0 {
		return ""
	}
	parts := make([]string, 0, len(vals))
	for _, v := range vals {
		parts = append(parts, fmt.Sprintf("%d", v))
	}
	return strings.Join(parts, "/")
}

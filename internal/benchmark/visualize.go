package benchmark

import (
	"sort"
	"strconv"
)

// The payload the visualize page charts. It is deliberately a description of
// the data rather than of any chart: the page decides what a scatter or a
// heatmap needs, and the server only has to say what varied, what was
// measured, and where each run sits.

// VizDimension is one axis a set of runs varies on.
//
// Numeric matters for more than sorting. A context length of 131072 sits four
// times further from 32768 than 65536 does, and a scatter that spaces them
// evenly hides that. A model name has no such spacing and must stay a
// category.
type VizDimension struct {
	Name    string   `json:"name"`
	Label   string   `json:"label"`
	Values  []string `json:"values"`
	Numeric bool     `json:"numeric"`
}

// VizMetric is one measurement the page can plot.
type VizMetric struct {
	Key   string `json:"key"`
	Label string `json:"label"`
	Unit  string `json:"unit"`
	// HigherIsBetter tells the page which end of the scale is good, so a
	// heatmap can be coloured in the right direction rather than leaving the
	// reader to work out whether dark means fast or slow.
	HigherIsBetter bool `json:"higher_is_better"`
}

// VizPoint is one run: where it sits on every dimension, and what it measured.
type VizPoint struct {
	RunID   string             `json:"run_id"`
	Label   string             `json:"label"`
	Detail  string             `json:"detail"`
	Dims    map[string]string  `json:"dims"`
	Metrics map[string]float64 `json:"metrics"`
}

// VizData is the whole payload.
type VizData struct {
	Dimensions []VizDimension `json:"dimensions"`
	Metrics    []VizMetric    `json:"metrics"`
	Points     []VizPoint     `json:"points"`
	// Skipped counts selected runs that measured nothing — failures. Saying
	// how many were left out beats quietly plotting fewer points than were
	// selected.
	Skipped int `json:"skipped"`
}

var vizMetrics = []VizMetric{
	{Key: "gen", Label: "Generation speed", Unit: "tok/s", HigherIsBetter: true},
	{Key: "prompt", Label: "Prompt speed", Unit: "tok/s", HigherIsBetter: true},
	{Key: "ttft", Label: "Time to first token", Unit: "ms", HigherIsBetter: false},
}

// BuildVisualization turns runs into the payload: the dimensions they vary on,
// the metrics available, and one point per run that produced timings.
func BuildVisualization(runs []BenchmarkRun) VizData {
	data := VizData{Metrics: vizMetrics}

	// Only dimensions that vary are axes. One that every run shares cannot
	// separate them, and offering it as an axis produces a chart where every
	// point lands in the same column.
	cmp := BuildCompare(runs)
	for _, col := range cmp.Varying {
		values := map[string]bool{}
		for _, r := range runs {
			values[valueFor(r, col.Key)] = true
		}
		list := make([]string, 0, len(values))
		for v := range values {
			list = append(list, v)
		}
		numeric := allNumeric(list)
		sortDimensionValues(list, numeric)
		data.Dimensions = append(data.Dimensions, VizDimension{
			Name: col.Key, Label: col.Label, Values: list, Numeric: numeric,
		})
	}

	for _, r := range runs {
		if r.Summary == nil {
			data.Skipped++
			continue
		}
		p := VizPoint{
			RunID: r.ID,
			// Named by the same rules the comparison uses, so a point on a
			// chart and a row in the table name the same run the same way.
			Label:   compareLabel(r, cmp.Varying),
			Detail:  firstNonEmpty(r.ModelName, r.ModelID) + " · " + r.Preset,
			Dims:    map[string]string{},
			Metrics: map[string]float64{},
		}
		for _, d := range data.Dimensions {
			p.Dims[d.Name] = valueFor(r, d.Name)
		}
		p.Metrics["gen"] = r.Summary.AvgGenTokPerSec
		p.Metrics["prompt"] = r.Summary.AvgPromptTokPerSec
		p.Metrics["ttft"] = r.Summary.AvgTTFTMs
		data.Points = append(data.Points, p)
	}
	return data
}

// allNumeric reports whether every value parses as a number, which decides
// whether a dimension is spaced or categorical.
func allNumeric(values []string) bool {
	if len(values) == 0 {
		return false
	}
	for _, v := range values {
		if v == "" {
			return false
		}
		if _, err := strconv.ParseFloat(v, 64); err != nil {
			return false
		}
	}
	return true
}

// sortDimensionValues orders an axis: numerically where that means something,
// lexically otherwise. "1024" before "512" is wrong on an axis and right in a
// list of names.
func sortDimensionValues(values []string, numeric bool) {
	if numeric {
		sort.Slice(values, func(i, j int) bool {
			a, _ := strconv.ParseFloat(values[i], 64)
			b, _ := strconv.ParseFloat(values[j], 64)
			return a < b
		})
		return
	}
	sort.Strings(values)
}

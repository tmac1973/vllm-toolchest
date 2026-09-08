package api

import (
	"encoding/json"
	"html/template"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
)

// handleVisualizePage renders the chart page for a set of runs. The ids come
// from the benchmarks page's "Visualize selected".
func (s *Server) handleVisualizePage(w http.ResponseWriter, r *http.Request) {
	runs, err := s.runsByIDs(r.URL.Query().Get("ids"))

	data := struct {
		pageData
		RunCount int
		Skipped  int
		Error    string
		DataJSON template.JS
	}{
		pageData: pageData{Title: "Visualize", Nav: "benchmarks"},
	}

	switch {
	case err != nil:
		data.Error = err.Error()
	case len(runs) < 2:
		data.Error = "Select at least two runs to visualize."
	default:
		viz := benchmark.BuildVisualization(runs)
		data.RunCount = len(viz.Points)
		data.Skipped = viz.Skipped
		encoded, jsonErr := json.Marshal(viz)
		if jsonErr != nil {
			data.Error = "Could not encode the chart data: " + jsonErr.Error()
		} else {
			// Into a <script type="application/json"> block, so this is JSON
			// in a data island rather than executable script. template.JS
			// marks it as already-safe; json.Marshal has escaped the string
			// contents, and Go's JSON encoder escapes < > & by default, so a
			// model name cannot close the tag.
			data.DataJSON = template.JS(encoded)
		}
	}

	s.render(w, "visualize.html", data)
}

// handleVisualizeData returns the same payload as JSON, for anything that
// would rather plot it elsewhere.
func (s *Server) handleVisualizeData(w http.ResponseWriter, r *http.Request) {
	runs, err := s.runsByIDs(r.URL.Query().Get("ids"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	respondJSON(w, benchmark.BuildVisualization(runs))
}

// runsByIDs resolves a comma-separated id list, in the order given.
func (s *Server) runsByIDs(raw string) ([]benchmark.BenchmarkRun, error) {
	var runs []benchmark.BenchmarkRun
	for _, id := range strings.Split(raw, ",") {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		run, err := s.bench.Get(id)
		if err != nil {
			return nil, err
		}
		runs = append(runs, *run)
	}
	return runs, nil
}

package api

import (
	"fmt"
	"net/http"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
)

// handleListBenchmarks returns all benchmark runs, optionally filtered by
// model_id, preset, or job_id. Dual-mode (HTML/JSON) per HX-Request.
func (s *Server) handleListBenchmarks(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	modelFilter := q.Get("model_id")
	presetFilter := q.Get("preset")
	jobFilter := q.Get("job_id")

	all := s.bench.List()
	runs := make([]benchmark.BenchmarkRun, 0, len(all))
	for _, run := range all {
		if modelFilter != "" && run.ModelID != modelFilter {
			continue
		}
		if presetFilter != "" && run.Preset != presetFilter {
			continue
		}
		if jobFilter != "" && run.JobID != jobFilter {
			continue
		}
		runs = append(runs, run)
	}

	if !isHTMX(r) {
		respondJSON(w, map[string]any{
			"runs":  runs,
			"total": len(runs),
		})
		return
	}

	respondHTML(w)
	renderRunList(w, runs)
}

func renderRunList(w http.ResponseWriter, runs []benchmark.BenchmarkRun) {
	if len(runs) == 0 {
		fmt.Fprint(w, `<p style="opacity:0.7;">No benchmark runs yet. Load a model and start a run from the form above.</p>`)
		return
	}

	fmt.Fprint(w, `<table><thead><tr>
    <th>Model</th><th>Preset</th><th>Avg gen TPS</th><th>Avg TTFT</th><th>Status</th><th>When</th>
  </tr></thead><tbody>`)
	for _, run := range runs {
		var avgGen, avgTTFT string
		if run.Summary != nil {
			avgGen = fmt.Sprintf("%.1f t/s", run.Summary.AvgGenTokPerSec)
			avgTTFT = fmt.Sprintf("%.0f ms", run.Summary.AvgTTFTMs)
		} else {
			avgGen = "—"
			avgTTFT = "—"
		}
		when := run.CreatedAt.Format(time.RFC3339)
		fmt.Fprintf(w, `<tr>
      <td><strong>%s</strong><br><small style="opacity:0.6;">%s</small></td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
      <td><small>%s</small></td>
    </tr>`,
			htmlEscape(run.ModelName), htmlEscape(run.ModelID),
			run.Preset, avgGen, avgTTFT,
			statusBadge(run.Status),
			when,
		)
	}
	fmt.Fprint(w, `</tbody></table>`)
}

func statusBadge(s string) string {
	switch s {
	case benchmark.StatusRunning:
		return `<mark>running</mark>`
	case benchmark.StatusCompleted:
		return `<ins>completed</ins>`
	case benchmark.StatusFailed:
		return `<del>failed</del>`
	default:
		return s
	}
}

func htmlEscape(s string) string {
	// Lightweight escape — chi router gives us text-safe inputs but display
	// fields (model name, model id) can carry user-supplied characters.
	r := []byte{}
	for _, c := range []byte(s) {
		switch c {
		case '<':
			r = append(r, []byte("&lt;")...)
		case '>':
			r = append(r, []byte("&gt;")...)
		case '&':
			r = append(r, []byte("&amp;")...)
		case '"':
			r = append(r, []byte("&quot;")...)
		default:
			r = append(r, c)
		}
	}
	return string(r)
}

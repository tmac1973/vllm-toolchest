package api

import (
	"net/http"
	"sort"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
)

// adviceRow is one thing the engine said, as the panel renders it.
type adviceRow struct {
	Severity string
	Hue      string
	Message  string
	// Field is the setting the advice implicates, and Suggested the value the
	// engine itself named. Neither is acted on: the panel puts them in front
	// of a person, who decides.
	Field     string
	Suggested string
	// Line is what vLLM actually wrote, kept so the reader can check the
	// paraphrase against the source rather than taking our word for it.
	Line string
}

// adviceView is the whole panel.
type adviceView struct {
	Rows []adviceRow
	// Quiet means the engine said nothing worth acting on, which is the
	// ordinary case for a healthy start and deserves saying rather than
	// rendering an empty box.
	Quiet bool
}

// handleServiceAdvice renders what the engine said during this run.
//
// It reads the process manager's captured items rather than re-parsing the log,
// and that is not an optimisation. The log buffer holds 5000 lines, which on a
// host serving steadily is a few hours: a start from the 18th had been pushed
// out entirely by the 21st, advice and all. The items survive in memory for the
// life of the run, so they are the only place this is still available.
func (s *Server) handleServiceAdvice(w http.ResponseWriter, r *http.Request) {
	view := newAdviceView(s.process.Advice())

	if !isHTMX(r) {
		respondJSON(w, view)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "service_advice", view)
}

func newAdviceView(items []advice.Item) adviceView {
	v := adviceView{Quiet: len(items) == 0}
	for _, it := range items {
		v.Rows = append(v.Rows, adviceRow{
			Severity:  string(it.Severity),
			Hue:       adviceHue(it.Severity),
			Message:   it.Message,
			Field:     it.Field,
			Suggested: it.Suggested,
			Line:      it.Line,
		})
	}

	// Errors first, then warnings: on a failed start the reason is what the
	// reader came for, and it should not be below a note about a deprecated
	// environment variable. Stable within a severity, so the order the engine
	// said things in is preserved.
	sort.SliceStable(v.Rows, func(i, j int) bool {
		return adviceRank(v.Rows[i].Severity) < adviceRank(v.Rows[j].Severity)
	})
	return v
}

func adviceRank(severity string) int {
	switch advice.Severity(severity) {
	case advice.Error:
		return 0
	case advice.Warning:
		return 1
	}
	return 2
}

func adviceHue(severity advice.Severity) string {
	switch severity {
	case advice.Error:
		return "#b83d3d"
	case advice.Warning:
		return "#b86e00"
	}
	return "#6b7280"
}

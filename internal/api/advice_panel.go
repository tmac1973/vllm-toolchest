package api

import (
	"fmt"
	"hash/fnv"
	"net/http"
	"sort"
	"strconv"
	"strings"

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

	// Applicable means Suggested is a value that could be written to Field,
	// so the row offers a button. Most suggestions are not: an unrecognised
	// flag is the problem rather than the fix, and chunked prefill merely
	// echoes what is already set.
	Applicable bool
	// Ours marks a suggestion computed here rather than read out of the
	// engine's words. Both can be right and they do not deserve equal
	// confidence, so the row says which it is.
	Ours bool
	// Pins warns that applying this stops the pool from being measured again,
	// which is the one suggestion that costs something to take.
	Pins bool
}

// adviceView is the whole panel.
type adviceView struct {
	Rows []adviceRow
	// Quiet means the engine said nothing worth acting on, which is the
	// ordinary case for a healthy start and deserves saying rather than
	// rendering an empty box.
	Quiet bool
	// ModelID is the model the advice was captured for. Carried so an apply
	// names its target explicitly rather than letting the handler assume
	// whatever happens to be running when the button is pressed.
	ModelID string

	// OK and Error report what an apply did, shown above the rows. The panel
	// replaces itself on apply, so the outcome has to travel with it: there is
	// no separate element on the server page to put a notice in.
	OK    string
	Error string

	// Level, Summary and Sig are the banner: the one line that stands in for
	// the panel on the server page, which is now a strip above the logs rather
	// than a pane competing with them for height.
	//
	// Level tints it -- the worst severity present, since that is what decides
	// how loudly the strip should ask to be opened.
	Level string
	// Summary is what the strip says when there is something to say. Composed
	// here rather than in the template: counting by severity and pluralising is
	// not something Go templates do without becoming unreadable.
	Summary string
	// Sig fingerprints the rows so the page can tell a changed panel from the
	// same panel polled again. The strip pulses on a change and then stops;
	// without this it would pulse every ten seconds forever.
	Sig string
}

// applyTarget is the element an apply swaps into. The panel lives on the
// server page, so it replaces itself -- an earlier version targeted the config
// panel's element, which exists only on the models page and would have swapped
// a fragment into nothing.
const applyTarget = "#service-advice"

// handleServiceAdvice renders what the engine said during this run.
//
// It reads the process manager's captured items rather than re-parsing the log,
// and that is not an optimisation. The log buffer holds 5000 lines, which on a
// host serving steadily is a few hours: a start from the 18th had been pushed
// out entirely by the 21st, advice and all. The items survive in memory for the
// life of the run, so they are the only place this is still available.
func (s *Server) handleServiceAdvice(w http.ResponseWriter, r *http.Request) {
	view := s.adviceSnapshot()

	if !isHTMX(r) {
		respondJSON(w, view)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "service_advice", view)
}

// adviceSnapshot is the panel as it stands right now.
//
// The nil check is not defensive padding: the server page polls this every ten
// seconds, so a process manager that is not set takes the whole page down
// rather than degrading. It reached production as a nil-receiver panic in
// Manager.Advice, and gpuInventory had already established the guard for the
// same reason on the same kind of field.
//
// No process is the same answer as a process that said nothing.
func (s *Server) adviceSnapshot() adviceView {
	if s.process == nil {
		// Through newAdviceView rather than a bare literal, so the banner's
		// level and fingerprint are the quiet ones rather than empty strings
		// the strip would have to guess at.
		return newAdviceView(nil)
	}
	v := newAdviceView(s.process.Advice())
	v.ModelID = s.process.GetStatus().ModelID
	return v
}

func newAdviceView(items []advice.Item) adviceView {
	v := adviceView{Quiet: len(items) == 0}
	for _, it := range items {
		v.Rows = append(v.Rows, adviceRow{
			Severity:   string(it.Severity),
			Hue:        adviceHue(it.Severity),
			Message:    it.Message,
			Field:      it.Field,
			Suggested:  it.Suggested,
			Line:       it.Line,
			Applicable: it.Applicable,
			Ours:       it.Ours,
			// Only an applicable row can pin anything, since only it offers
			// the button that would.
			Pins: it.Applicable && it.Field == "kv_cache_memory",
		})
	}

	// Errors first, then warnings: on a failed start the reason is what the
	// reader came for, and it should not be below a note about a deprecated
	// environment variable. Stable within a severity, so the order the engine
	// said things in is preserved.
	sort.SliceStable(v.Rows, func(i, j int) bool {
		return adviceRank(v.Rows[i].Severity) < adviceRank(v.Rows[j].Severity)
	})

	v.Level, v.Summary = adviceBanner(v.Rows)
	v.Sig = adviceSig(v.Rows)
	return v
}

// adviceBanner reduces the rows to the one line the strip shows.
//
// The counts are per severity because "3 notes" and "1 error, 2 warnings" ask
// for very different amounts of attention, and a bare total would flatten the
// difference at exactly the moment it matters.
func adviceBanner(rows []adviceRow) (level, summary string) {
	if len(rows) == 0 {
		return "quiet", ""
	}

	var errs, warns, notes int
	for _, r := range rows {
		switch advice.Severity(r.Severity) {
		case advice.Error:
			errs++
		case advice.Warning:
			warns++
		default:
			notes++
		}
	}

	var parts []string
	for _, c := range []struct {
		n    int
		word string
	}{{errs, "error"}, {warns, "warning"}, {notes, "note"}} {
		if c.n > 0 {
			parts = append(parts, plural(c.n, c.word))
		}
	}

	switch {
	case errs > 0:
		level = "error"
	case warns > 0:
		level = "warning"
	default:
		level = "note"
	}
	return level, strings.Join(parts, ", ") + " from this start"
}

func plural(n int, word string) string {
	if n == 1 {
		return "1 " + word
	}
	return strconv.Itoa(n) + " " + word + "s"
}

// adviceSig fingerprints the panel's contents.
//
// Severity, field and message rather than the whole row: those are what the
// reader would notice changing. The verbatim line moves with the message, and
// hashing it too would only make the fingerprint more expensive to compute
// without making it more sensitive.
func adviceSig(rows []adviceRow) string {
	if len(rows) == 0 {
		return "quiet"
	}
	h := fnv.New64a()
	for _, r := range rows {
		fmt.Fprintf(h, "%s\x00%s\x00%s\x00", r.Severity, r.Field, r.Message)
	}
	return strconv.FormatUint(h.Sum64(), 16)
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

package api

import (
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
)

// The server page polls the advice endpoint every ten seconds. A server with
// no process manager must answer it rather than panicking: this reached the
// tests as a nil-receiver dereference in Manager.Advice, and in a browser it
// would have taken the page down on a timer.
func TestTheAdvicePanelToleratesNoProcess(t *testing.T) {
	s := &Server{}
	v := s.adviceSnapshot()
	if !v.Quiet {
		t.Errorf("no process, but the panel claims something was said: %+v", v)
	}
	if len(v.Rows) != 0 {
		t.Errorf("got %d rows with no process", len(v.Rows))
	}
}

// A healthy start says nothing, and that deserves saying rather than rendering
// an empty box.
func TestAQuietStartSaysSo(t *testing.T) {
	v := newAdviceView(nil)
	if !v.Quiet {
		t.Error("no advice, but the panel does not report itself quiet")
	}
	if len(v.Rows) != 0 {
		t.Errorf("got %d rows from nothing", len(v.Rows))
	}
}

// On a failed start the reason is what the reader came for. It must not sit
// below a note about a deprecated environment variable.
func TestErrorsComeFirst(t *testing.T) {
	v := newAdviceView([]advice.Item{
		{Severity: advice.Info, Message: "chunked prefill", Field: "max_num_batched_tokens"},
		{Severity: advice.Warning, Message: "deprecated variable", Field: "env"},
		{Severity: advice.Error, Message: "could not reserve the fraction configured", Field: "gpu_memory_utilization"},
		{Severity: advice.Info, Message: "kv pool reported", Field: "kv_cache_memory", Suggested: "4545545954"},
	})

	if v.Quiet {
		t.Fatal("four items, but the panel reports itself quiet")
	}
	if len(v.Rows) != 4 {
		t.Fatalf("got %d rows, want 4", len(v.Rows))
	}
	if v.Rows[0].Severity != string(advice.Error) {
		t.Errorf("first row is %q; the failure should lead", v.Rows[0].Severity)
	}
	if v.Rows[1].Severity != string(advice.Warning) {
		t.Errorf("second row is %q, want warning", v.Rows[1].Severity)
	}

	// Stable within a severity: the engine's own order is preserved.
	if v.Rows[2].Message != "chunked prefill" {
		t.Errorf("the two info items were reordered against each other: %q first", v.Rows[2].Message)
	}
}

// The suggested value and the field are what make an item actionable, and the
// verbatim line is what lets the reader check the paraphrase.
func TestRowsCarryTheEnginesOwnWords(t *testing.T) {
	const line = "Replace gpu_memory_utilization config with `--kv-cache-memory=4545545954` (4.23 GiB)"
	v := newAdviceView([]advice.Item{{
		Severity: advice.Info, Message: "the engine reports its pool",
		Field: "kv_cache_memory", Suggested: "4545545954", Line: line,
	}})

	r := v.Rows[0]
	if r.Field != "kv_cache_memory" || r.Suggested != "4545545954" {
		t.Errorf("field/suggested = %q/%q", r.Field, r.Suggested)
	}
	if r.Line != line {
		t.Error("the source line was not carried through verbatim")
	}
	if r.Hue == "" {
		t.Error("no hue, so severity is invisible on screen")
	}
}

package api

import (
	"net/http/httptest"
	"strings"
	"testing"
)

// browserData decodes an SSE frame the way the EventSource spec says a browser
// does: the event's data is its "data:" field values joined with "\n". Asserting
// on this rather than on the raw bytes is what makes the test about what the
// page receives.
func browserData(frame string) string {
	var parts []string
	for _, line := range strings.Split(strings.TrimSuffix(frame, "\n\n"), "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		parts = append(parts, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
	}
	return strings.Join(parts, "\n")
}

func TestSendLineCarriesNoTrailingNewline(t *testing.T) {
	// Every consumer appends its own "\n" when writing into the log pane, so
	// a newline arriving inside the data is a blank line between every line.
	w := httptest.NewRecorder()
	sse, err := NewSSEWriter(w)
	if err != nil {
		t.Fatal(err)
	}
	if err := sse.SendLine("INFO starting engine"); err != nil {
		t.Fatal(err)
	}

	got := browserData(w.Body.String())
	if got != "INFO starting engine" {
		t.Errorf("browser would receive %q, want the line with nothing appended", got)
	}
	if strings.Contains(got, "\n") {
		t.Error("the data carries a newline the consumer will double")
	}
}

// A blank line in vLLM's output should stay one blank line, not become two.
func TestSendLinePreservesABlankLineExactly(t *testing.T) {
	w := httptest.NewRecorder()
	sse, _ := NewSSEWriter(w)
	sse.SendLine("")

	if got := browserData(w.Body.String()); got != "" {
		t.Errorf("browser would receive %q, want an empty string", got)
	}
}

// The frame has to be one data field and one blank-line terminator: a second
// data field is what the browser joins with a newline.
func TestSendLineFrameShape(t *testing.T) {
	w := httptest.NewRecorder()
	sse, _ := NewSSEWriter(w)
	sse.SendLine("hello")

	frame := w.Body.String()
	if n := strings.Count(frame, "data:"); n != 1 {
		t.Errorf("frame has %d data fields, want 1: %q", n, frame)
	}
	if !strings.HasSuffix(frame, "\n\n") {
		t.Errorf("frame is not terminated by a blank line: %q", frame)
	}
}

// Several lines must arrive as separate events, so the pane can append them
// one at a time rather than receiving a run-together block.
func TestSendLineSeparatesEvents(t *testing.T) {
	w := httptest.NewRecorder()
	sse, _ := NewSSEWriter(w)
	for _, line := range []string{"first", "second", "third"} {
		sse.SendLine(line)
	}

	frames := strings.SplitAfter(w.Body.String(), "\n\n")
	var got []string
	for _, f := range frames {
		if strings.TrimSpace(f) == "" {
			continue
		}
		got = append(got, browserData(f))
	}
	want := []string{"first", "second", "third"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("got %v, want %v", got, want)
	}
}

// A named event still frames its data the same way; "done" is sent this way
// when a stream ends.
func TestSendEventCarriesTheEventName(t *testing.T) {
	w := httptest.NewRecorder()
	sse, _ := NewSSEWriter(w)
	sse.SendEvent("done", "tuning job ended")

	frame := w.Body.String()
	if !strings.HasPrefix(frame, "event: done\n") {
		t.Errorf("missing event name: %q", frame)
	}
	if got := browserData(frame); got != "tuning job ended" {
		t.Errorf("data = %q", got)
	}
}

func TestSSEHeaders(t *testing.T) {
	w := httptest.NewRecorder()
	if _, err := NewSSEWriter(w); err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string]string{
		"Content-Type":  "text/event-stream",
		"Cache-Control": "no-cache",
	} {
		if got := w.Header().Get(k); got != want {
			t.Errorf("%s = %q, want %q", k, got, want)
		}
	}
}

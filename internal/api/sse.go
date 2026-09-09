package api

import (
	"context"
	"fmt"
	"net/http"
)

// SSEWriter wraps an http.ResponseWriter for Server-Sent Events.
type SSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func NewSSEWriter(w http.ResponseWriter) (*SSEWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	return &SSEWriter{w: w, flusher: flusher}, nil
}

func (s *SSEWriter) SendEvent(event, data string) error {
	if event != "" {
		fmt.Fprintf(s.w, "event: %s\n", event)
	}
	fmt.Fprintf(s.w, "data: %s\n\n", data)
	s.flusher.Flush()
	return nil
}

func (s *SSEWriter) SendData(data string) error {
	return s.SendEvent("", data)
}

// SendLine sends one line of streamed output.
//
// One data field, not two. A second, empty "data:" used to follow, and the SSE
// spec joins an event's data fields with a newline — so the browser received
// "<line>\n" rather than "<line>", and every consumer appends its own newline
// on top. The result was a blank line between every log line in the server and
// tuning panes, and doubled blank lines wherever vLLM emitted one.
func (s *SSEWriter) SendLine(data string) error {
	return s.SendData(data)
}

// StreamLines streams log lines from a channel, sending each as an SSE line
// event. When the channel closes, a "done" event is sent with the given message.
func StreamLines(w http.ResponseWriter, ctx context.Context, ch <-chan string, doneMsg string) {
	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	for {
		select {
		case line, ok := <-ch:
			if !ok {
				sse.SendEvent("done", doneMsg)
				return
			}
			sse.SendLine(line)
		case <-ctx.Done():
			return
		}
	}
}

package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// The unit tests cover the view; this covers the template, which is where a
// mistake would otherwise surface as a 500 in front of someone rather than a
// build failure here.
func TestTheAdvicePanelRenders(t *testing.T) {
	const id = "test/advice"
	// One line that produces advice, and one that is merely a measurement.
	output := `vllm: error: unrecognized arguments: --enable-expert-offload
INFO [worker.py:222] PLE offload: locked 38.8 GiB of PLE weights in RAM`

	s := testServerWithEngine(t, id, output, models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 8192})
	s.initTemplates()
	waitForObserved(t, s, func() bool { return len(s.process.Advice()) > 0 })

	req := httptest.NewRequest("GET", "/api/service/advice", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleServiceAdvice(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if strings.Contains(body, "Nothing to report") {
		t.Error("advice was captured but the panel reported a quiet start")
	}
	if !strings.Contains(body, "does not recognise") {
		t.Errorf("the message is missing from the rendered panel:\n%s", body)
	}
	if !strings.Contains(body, "extra_flags") {
		t.Error("the implicated config field is not shown")
	}
	// The verbatim line is what lets a reader check the paraphrase.
	if !strings.Contains(body, "unrecognized arguments") {
		t.Error("what vLLM actually wrote is not in the panel")
	}
}

// A healthy start renders the quiet state rather than an empty list.
func TestTheAdvicePanelRendersQuiet(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.process = process.NewManager("127.0.0.1", 0, 0)
	s.initTemplates()

	req := httptest.NewRequest("GET", "/api/service/advice", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleServiceAdvice(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "Nothing to report") {
		t.Errorf("a quiet start did not say so:\n%s", rec.Body.String())
	}
}

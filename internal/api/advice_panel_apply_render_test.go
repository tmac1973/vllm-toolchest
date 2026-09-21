package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// The gap that let three defects through a green suite.
//
// The existing render test feeds an "unrecognized arguments" line, which is
// deliberately *not* applicable -- so the apply block was never evaluated and
// a reference to a field that does not exist on the view went unnoticed. Go
// templates resolve those at render time, so it would have become a 500 in
// front of whoever was reading a failed start.
//
// This drives a line that is applicable, which is the only way the form, its
// hidden inputs, its target and the pinning warning are exercised at all.
func TestAnApplicableRowRendersItsButton(t *testing.T) {
	const id = "test/applicable"
	// The engine handing over its own pool size: applicable, and the one
	// suggestion that costs something to take.
	output := `INFO [gpu_worker.py:935] Replace gpu_memory_utilization config with --kv-cache-memory=4545545954 (4.23 GiB) to fit into requested memory.`

	s := testServerWithEngine(t, id, output, models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 262144})
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

	for _, want := range []string{
		`hx-post="/api/service/advice/apply"`,
		`hx-target="#service-advice"`,
		`name="model_id"`,
		`name="field"`,
		`name="value"`,
		"4545545954",
		// Pinning the pool stops it being measured, and the button says so.
		"pins the pool",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q from the rendered row:\n%.700s", want, body)
		}
	}

	// The target must be an element that exists on the page the form is on.
	if strings.Contains(body, "#config-") {
		t.Error("the apply targets the config panel, which lives on a different page")
	}
}

// A suggestion computed here rather than read from the engine's words says so,
// because the two do not deserve equal confidence.
func TestOurOwnArithmeticIsLabelled(t *testing.T) {
	const id = "test/ours"
	output := `ERROR ValueError: Free memory on device cuda:0 (27.28/31.86 GiB) on startup is less than desired GPU memory utilization (0.97, 30.9 GiB). Decrease GPU memory utilization.`

	s := testServerWithEngine(t, id, output, models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 262144})
	s.initTemplates()
	waitForObserved(t, s, func() bool { return len(s.process.Advice()) > 0 })

	req := httptest.NewRequest("GET", "/api/service/advice", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleServiceAdvice(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, "our arithmetic") {
		t.Errorf("a value computed here is presented as the engine's own:\n%.500s", body)
	}
	if !strings.Contains(body, "0.85") {
		t.Errorf("the computed fraction is missing:\n%.500s", body)
	}
}

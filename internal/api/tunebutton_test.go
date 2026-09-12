package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// rowFor builds a model row the way the page does, for one registry entry.
func rowFor(t *testing.T, m *models.Model) modelRow {
	t.Helper()
	s := newGoldenServer(t, goldenEnvGeneric)
	if err := s.registry.Register(m); err != nil {
		t.Fatal(err)
	}
	for _, r := range s.modelRows() {
		if r.ID == m.ID {
			return r
		}
	}
	t.Fatalf("no row for %s", m.ID)
	return modelRow{}
}

// The Tune button appears for exactly the models tuning can help. Offering it
// more widely sends someone into a run that stops their server and changes
// nothing; offering it less widely hides the feature from the models it exists
// for.
func TestTuneButtonEligibility(t *testing.T) {
	hf := models.HFConfig{
		HiddenSize: 4096, IntermediateSize: 11008,
		NumAttentionHeads: 32, NumKeyValueHeads: 8, HeadDim: 128,
	}

	for _, tc := range []struct {
		name string
		q    models.QuantMeta
		want bool
	}{
		{"block-FP8", models.QuantMeta{Method: "fp8", WeightBlockSize: []int{128, 128}}, true},
		{"per-tensor FP8", models.QuantMeta{Method: "fp8"}, false},
		{"compressed-tensors block", models.QuantMeta{Method: "compressed-tensors", Bits: 8, WeightBlockSize: []int{128, 128}}, true},
		{"compressed-tensors channel", models.QuantMeta{Method: "compressed-tensors", Bits: 8}, false},
		{"awq", models.QuantMeta{Method: "awq", Bits: 4}, false},
		{"unquantized", models.QuantMeta{Method: "none"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			row := rowFor(t, &models.Model{
				ID: "test/" + tc.name, LocalPath: "/models/x",
				Quantization: tc.q, HFConfig: hf,
			})
			if row.Tunable != tc.want {
				t.Errorf("Tunable = %v, want %v", row.Tunable, tc.want)
			}
			if tc.want && row.TunableShapes == 0 {
				t.Error("a tunable model should report the shape count the run would cover")
			}
		})
	}
}

// A model with no derivable shapes is not tunable however it is quantized:
// a run would measure nothing.
func TestTuneButtonNeedsShapes(t *testing.T) {
	row := rowFor(t, &models.Model{
		ID: "test/no-dims", LocalPath: "/models/x",
		Quantization: models.QuantMeta{Method: "fp8", WeightBlockSize: []int{128, 128}},
		HFConfig:     models.HFConfig{}, // nothing to derive shapes from
	})
	if row.Tunable {
		t.Error("a model whose shapes cannot be derived must not offer tuning")
	}
}

// The button must actually reach the markup, and must not appear on cards that
// are not eligible.
func TestTuneButtonRendersOnlyWhenEligible(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	hf := models.HFConfig{HiddenSize: 4096, IntermediateSize: 11008, NumAttentionHeads: 32, NumKeyValueHeads: 8, HeadDim: 128}
	if err := s.registry.Register(&models.Model{
		ID: "a/block-fp8", LocalPath: "/models/a",
		Quantization: models.QuantMeta{Method: "fp8", WeightBlockSize: []int{128, 128}}, HFConfig: hf,
	}); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.Register(&models.Model{
		ID: "b/awq", LocalPath: "/models/b",
		Quantization: models.QuantMeta{Method: "awq", Bits: 4}, HFConfig: hf,
	}); err != nil {
		t.Fatal(err)
	}

	var buf strings.Builder
	s.renderPartial(&buf, "model_list", struct{ Rows []modelRow }{s.modelRows()})

	// html/template escapes "/" as "\/" inside a JS string, which is correct
	// -- it is what stops a model id containing "</script>" from breaking out
	// -- and evaluates back to "/" at runtime. Undo it so the assertions can
	// name models the way the registry does.
	out := strings.ReplaceAll(buf.String(), `\/`, "/")

	if strings.Count(out, "tuneModel(") != 1 {
		t.Errorf("expected exactly one Tune button, got %d", strings.Count(out, "tuneModel("))
	}
	if !strings.Contains(out, "tuneModel('a/block-fp8')") {
		t.Error("the block-FP8 model has no Tune button")
	}
	if strings.Contains(out, "tuneModel('b/awq')") {
		t.Error("the AWQ model should not offer tuning")
	}
}

// The handler the button calls has to exist on the page it is rendered into,
// and must warn before taking the server down.
func TestModelsPageCarriesTheTuneHandler(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := httptest.NewRecorder()
	s.handleModelsPage(w, httptest.NewRequest("GET", "/models", nil))
	body := w.Body.String()

	if !strings.Contains(body, "function tuneModel") {
		t.Fatal("the Tune button's handler is not on the page")
	}
	if !strings.Contains(body, "confirm(") || !strings.Contains(body, "stop the running vLLM server") {
		t.Error("starting a tuning run must warn that it stops the server")
	}
	if !strings.Contains(body, "/api/tuning/start") {
		t.Error("the handler should post to the same endpoint the Tuning page uses")
	}
}

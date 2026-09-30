package api

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

const (
	draftID  = "tcclaviger/Qwen3.8-27B-DFlash2-FP8"
	targetID = "tcclaviger/ThinkingCap-3.8-27B-PARO5"
)

// newDraftServer is a Server holding a DFlash draft, the 64-layer model it
// drafts for, and the golden fixtures -- which are other shapes, and so are
// what an unsuitable target looks like.
func newDraftServer(t *testing.T) (*Server, *models.Model, *models.Model) {
	t.Helper()
	s := newGoldenServer(t, goldenEnvGeneric)
	s.router = s.buildRouter()

	draftDir := filepath.Join(s.cfg.DataDir, "models", "tcclaviger", "Qwen3.8-27B-DFlash2-FP8")
	if err := os.MkdirAll(draftDir, 0o755); err != nil {
		t.Fatal(err)
	}
	draft := &models.Model{
		ID: draftID, DisplayName: "Qwen3.8-27B-DFlash2-FP8", LocalPath: draftDir, Enabled: true,
		HFConfig: models.HFConfig{
			Architectures: []string{"DFlash2DraftModel"}, HiddenSize: 5120, NumHiddenLayers: 5, VocabSize: 248320,
			Draft: &models.DraftMeta{Method: "dflash", BlockSize: 8, TargetLayers: 64},
		},
	}
	target := &models.Model{
		ID: targetID, DisplayName: "ThinkingCap-3.8-27B-PARO5", Enabled: true,
		LocalPath: filepath.Join(s.cfg.DataDir, "models", "tcclaviger", "ThinkingCap-3.8-27B-PARO5"),
		HFConfig: models.HFConfig{
			Architectures: []string{"Qwen3_5ForConditionalGeneration"}, HiddenSize: 5120, NumHiddenLayers: 64,
			AttentionLayers: 16, VocabSize: 248320, MaxPositionEmbeddings: 262144,
		},
		VLLMConfig: models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 262144},
	}
	for _, m := range []*models.Model{draft, target} {
		if err := s.registry.Register(m); err != nil {
			t.Fatal(err)
		}
	}
	return s, draft, target
}

func rowOn(t *testing.T, s *Server, id string) modelRow {
	t.Helper()
	for _, r := range s.modelRows() {
		if r.ID == id {
			return r
		}
	}
	t.Fatalf("no row for %s", id)
	return modelRow{}
}

// A draft started on its own loads for minutes and then fails. Every way of
// starting a model has to decline it before that.
func TestADraftCannotBeStartedOrActivated(t *testing.T) {
	s, draft, _ := newDraftServer(t)

	if err := s.startModel(draft); err == nil || !strings.Contains(err.Error(), "draft model") {
		t.Errorf("startModel on a draft returned %v, want a refusal", err)
	}
	if rec := htmxRequest(s, http.MethodPut, "/api/models/activate?id="+draftID); rec.Code != http.StatusConflict {
		t.Errorf("activating a draft returned %d, want 409", rec.Code)
	}
	if s.cfg.ActiveModel == draftID {
		t.Error("a draft was made the active model")
	}
	for _, m := range s.servable() {
		if m.ID == draftID {
			t.Error("a draft is listed among the models that can be served")
		}
	}
}

func TestADraftIsListedAsOneAndNamesItsUsers(t *testing.T) {
	s, draft, target := newDraftServer(t)

	row := rowOn(t, s, draftID)
	if !row.Draft || row.DraftMethod != "dflash" || len(row.UsedBy) != 0 {
		t.Errorf("draft row = %+v", row)
	}

	target.VLLMConfig.SpeculativeConfig = draft.SpeculativeConfigFor(0)
	if got := rowOn(t, s, draftID).UsedBy; len(got) != 1 || got[0] != target.DisplayName {
		t.Errorf("used by = %v, want the model whose config names it", got)
	}

	// The card: a badge, no radio and no Configure, and a Remove dialog that
	// says what removing it breaks.
	body := htmxRequest(s, http.MethodGet, "/api/models").Body.String()
	card := body[strings.Index(body, `id="model-card-`+safeID(draftID)+`"`):]
	card = card[:strings.Index(card, "</article>")]
	for _, want := range []string{">draft</mark>", "Drafts for <strong>ThinkingCap-3.8-27B-PARO5</strong>", "will not start once its files are deleted"} {
		if !strings.Contains(card, want) {
			t.Errorf("draft card is missing %q", want)
		}
	}
	for _, unwanted := range []string{`type="radio"`, "config-panel"} {
		if strings.Contains(card, unwanted) {
			t.Errorf("draft card offers %q", unwanted)
		}
	}
}

// The engine finds a missing draft after it has loaded the target's weights.
func TestAModelWhoseDraftIsGoneIsNotStarted(t *testing.T) {
	s, draft, target := newDraftServer(t)
	target.VLLMConfig.SpeculativeConfig = draft.SpeculativeConfigFor(0)
	if err := os.RemoveAll(draft.LocalPath); err != nil {
		t.Fatal(err)
	}

	err := s.startModel(target)
	if err == nil || !strings.Contains(err.Error(), "which is not there") {
		t.Fatalf("startModel returned %v, want a refusal naming the missing draft", err)
	}
	if !strings.Contains(s.draftWarning(target), "will not start") {
		t.Errorf("the config panel does not warn about it: %q", s.draftWarning(target))
	}

	// Neither of these names a directory, so neither is this tool's to check.
	for _, spec := range []string{
		`{"method": "mtp", "num_speculative_tokens": 3}`,
		`{"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7}`,
	} {
		if err := missingDraft(spec); err != nil {
			t.Errorf("%s was refused: %v", spec, err)
		}
	}
}

func TestTheDraftPickerOffersWhatFits(t *testing.T) {
	s, draft, target := newDraftServer(t)

	opts := s.draftOptions(target)
	if len(opts) != 1 || opts[0].Disabled || opts[0].Selected || opts[0].Value != draft.SpeculativeConfigFor(0) {
		t.Fatalf("options for the matching model = %+v", opts)
	}

	// A 64-layer draft beside a model of another width: listed, so its
	// absence is not a mystery, and not choosable.
	other, _ := s.registry.Get("TheBloke/Mixtral-8x7B-AWQ")
	if opts := s.draftOptions(other); len(opts) != 1 || !opts[0].Disabled || opts[0].Value != "" {
		t.Errorf("options for a model it does not fit = %+v", opts)
	}

	// The panel renders it, and choosing it is what fills the JSON in.
	body := htmxRequest(s, http.MethodGet, "/api/models/config-panel?id="+targetID).Body.String()
	for _, want := range []string{"Draft model", "Qwen3.8-27B-DFlash2-FP8 (dflash, 7 tokens)", "&#34;method&#34;: &#34;dflash&#34;"} {
		if !strings.Contains(body, want) {
			t.Errorf("config panel is missing %q", want)
		}
	}
}

// The selected option stands for what is saved. If it carried the default
// instead, a token count set by hand would be reset the next time anything
// made the picker fire.
func TestThePickerKeepsAHandEditedConfig(t *testing.T) {
	s, draft, target := newDraftServer(t)
	edited := strings.Replace(draft.SpeculativeConfigFor(0), `"num_speculative_tokens": 7`, `"num_speculative_tokens": 4`, 1)
	target.VLLMConfig.SpeculativeConfig = edited

	opts := s.draftOptions(target)
	if len(opts) != 1 || !opts[0].Selected || opts[0].Value != edited {
		t.Errorf("options = %+v, want the saved config selected as it stands", opts)
	}
	if w := s.draftWarning(target); w != "" {
		t.Errorf("a valid pairing drew a warning: %q", w)
	}
}

func TestThePanelWarnsAboutAPairingThatCannotWork(t *testing.T) {
	s, draft, target := newDraftServer(t)

	target.VLLMConfig.SpeculativeConfig = strings.Replace(draft.SpeculativeConfigFor(0),
		`"num_speculative_tokens": 7`, `"num_speculative_tokens": 12`, 1)
	if w := s.draftWarning(target); !strings.Contains(w, "at most 7 tokens") {
		t.Errorf("warning = %q, want the draft's limit", w)
	}

	other, _ := s.registry.Get("TheBloke/Mixtral-8x7B-AWQ")
	other.VLLMConfig.SpeculativeConfig = draft.SpeculativeConfigFor(0)
	if w := s.draftWarning(other); !strings.Contains(w, "does not look like a draft for this model") {
		t.Errorf("warning = %q, want the mismatch", w)
	}
}

func TestADraftHasNoConfigPanel(t *testing.T) {
	s, _, _ := newDraftServer(t)
	body := htmxRequest(s, http.MethodGet, "/api/models/config-panel?id="+draftID).Body.String()
	if !strings.Contains(body, "no launch config of its own") || strings.Contains(body, "<form") {
		t.Errorf("a draft was given a config panel:\n%.300s", body)
	}
}

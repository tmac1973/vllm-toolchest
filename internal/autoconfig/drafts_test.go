package autoconfig

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

type fakeHub struct {
	repos map[string]*huggingface.ModelDetail
	err   error
	asked int
}

func (h *fakeHub) GetModel(_ context.Context, id string) (*huggingface.ModelDetail, error) {
	h.asked++
	if h.err != nil {
		return nil, h.err
	}
	if d, ok := h.repos[id]; ok {
		return d, nil
	}
	return nil, errors.New("get model: HTTP 404")
}

var drafterRepo = &huggingface.ModelDetail{ID: "org/drafter", Files: []huggingface.ModelFile{
	{Filename: "model.safetensors", Size: 2 << 30, Category: "weights"},
	{Filename: "config.json", Size: 1000, Category: "config"},
}}

func noneInstalled(string) *models.Model { return nil }

func TestFindDraftSuggestions(t *testing.T) {
	ctx := context.Background()
	hub := &fakeHub{repos: map[string]*huggingface.ModelDetail{"org/drafter": drafterRepo}}

	got, err := FindDraftSuggestions(ctx, hub, thinkingCap(), "dflash", "org/drafter", noneInstalled)
	if err != nil || len(got) != 1 || got[0].Installed || !strings.HasPrefix(got[0].SizeLabel, "2.0 GB") {
		t.Fatalf("got %+v err %v", got, err)
	}
	if got, err := FindDraftSuggestions(ctx, hub, thinkingCap(), "dflash", "org/gone", noneInstalled); len(got) != 0 || err != nil {
		t.Errorf("a missing repository: %+v %v", got, err)
	}
	before := hub.asked
	if got, _ := FindDraftSuggestions(ctx, hub, thinkingCap(), "dflash", "", noneInstalled); len(got) != 0 || hub.asked != before {
		t.Error("with no repository named the Hub was asked anyway")
	}
	if _, err := FindDraftSuggestions(ctx, &fakeHub{err: errors.New("dial tcp: timeout")}, thinkingCap(), "dflash", "org/drafter", noneInstalled); err == nil {
		t.Error("a network failure was reported as a missing repository")
	}

	wrong := dflashDraft()
	wrong.HFConfig.Draft.TargetLayers = 48
	installed := func(repo string) *models.Model {
		if repo == "org/drafter" {
			return wrong
		}
		return nil
	}
	got, _ = FindDraftSuggestions(ctx, hub, thinkingCap(), "dflash", "org/drafter", installed)
	if len(got) != 1 || !got[0].Installed || !strings.Contains(got[0].Why, "48-layer") {
		t.Errorf("an installed mismatch: %+v", got)
	}
}

func TestRunSuggestsTheNamedDraft(t *testing.T) {
	card := "```\nvllm serve org/model --speculative-config '{\"method\": \"dflash\", \"model\": \"org/drafter\", \"num_speculative_tokens\": 7}'\n```\n"
	d, _ := runDeps(t, card)
	d.Drafts = nil
	d.Hub = &fakeHub{repos: map[string]*huggingface.ModelDetail{"org/drafter": drafterRepo}}
	d.Installed = noneInstalled
	res, err := Run(context.Background(), d, models.ContextMax)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Suggestions) != 1 || res.Suggestions[0].Repo != "org/drafter" {
		t.Fatalf("suggestions: %+v", res.Suggestions)
	}
	if !noteFor(res.Notes, "speculative_config", "downloaded below") {
		t.Errorf("notes: %+v", res.Notes)
	}

	d.Hub = &fakeHub{}
	res, _ = Run(context.Background(), d, models.ContextMax)
	if !noteFor(res.Notes, "speculative_config", "no such repository") {
		t.Errorf("a named repository the Hub lacks: %+v", res.Notes)
	}
}

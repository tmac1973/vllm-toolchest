package autoconfig

import (
	"context"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Hub is the part of the Hub client the draft search needs.
type Hub interface {
	GetModel(ctx context.Context, modelID string) (*huggingface.ModelDetail, error)
}

// DraftSuggestion is a draft model the card recommends that is not in use.
type DraftSuggestion struct {
	Repo      string
	SizeLabel string
	Method    string
	Why       string
	// Installed marks a copy already on disk that cannot be used; it is
	// shown, with the reason, and not offered for download.
	Installed bool
}

// FindDraftSuggestions looks up the draft repository the card names.
//
// Only a repository the card itself names is suggested. There is no search
// by name and no guess from the model's family: a drafter trained for a
// different checkpoint loads and then fails minutes in, and a wrong
// suggestion with a Download button on it is worse than none.
//
// A repository the Hub does not have returns no suggestion and no error. Any
// other failure is returned, so it is not reported as a missing repository.
func FindDraftSuggestions(ctx context.Context, hub Hub, target *models.Model,
	wantMethod, draftRepo string, installed func(repo string) *models.Model) ([]DraftSuggestion, error) {
	if hub == nil || draftRepo == "" {
		return nil, nil
	}
	if installed != nil {
		if m := installed(draftRepo); m != nil {
			why := models.DraftMismatch(target, m)
			switch {
			case why != "":
			case !m.IsDraft():
				why = "it is not recognised as a draft model"
			case m.HFConfig.Draft.Method != wantMethod:
				why = "it is a " + m.HFConfig.Draft.Method + " draft and the card asks for " + wantMethod
			default:
				return nil, nil // usable; validation would have paired it
			}
			return []DraftSuggestion{{Repo: draftRepo, Method: wantMethod, Installed: true, Why: why}}, nil
		}
	}
	detail, err := hub.GetModel(ctx, draftRepo)
	if err != nil {
		if strings.Contains(err.Error(), "HTTP 404") {
			return nil, nil
		}
		return nil, err
	}
	var size int64
	for _, f := range huggingface.DownloadableFiles(detail.Files) {
		size += f.Size
	}
	return []DraftSuggestion{{
		Repo: draftRepo, Method: wantMethod, SizeLabel: huggingface.FormatBytes(size),
		Why: "The card recommends it for speculative decoding. Download it, then run Autoconfigure again to pair it.",
	}}, nil
}

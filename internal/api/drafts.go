package api

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// A draft model is registered like any other -- it is downloaded, updated and
// removed the same way -- and is otherwise the odd one out: it cannot be
// served, and what matters about it is which models point at it. This file is
// where that relationship is read, from the one place it is stored: the path
// in a model's speculative config.

// servable is the registered models that can be started, in registry order.
// Everything that offers a model to launch, benchmark or tune lists these.
func (s *Server) servable() []*models.Model {
	all := s.registry.List()
	out := make([]*models.Model, 0, len(all))
	for _, m := range all {
		if !m.IsDraft() && !m.Helper {
			out = append(out, m)
		}
	}
	return out
}

// draftAt is the registered draft whose files are at path, or nil.
func (s *Server) draftAt(path string) *models.Model {
	if path == "" {
		return nil
	}
	for _, m := range s.registry.List() {
		if m.IsDraft() && filepath.Clean(m.LocalPath) == path {
			return m
		}
	}
	return nil
}

// draftUsers is the models whose speculative config names this draft.
func (s *Server) draftUsers(draft *models.Model) []*models.Model {
	var users []*models.Model
	want := filepath.Clean(draft.LocalPath)
	for _, m := range s.registry.List() {
		if ref, ok := models.ParseSpeculative(m.VLLMConfig.SpeculativeConfig); ok && ref.LocalDraft() == want {
			users = append(users, m)
		}
	}
	return users
}

// launchBlocker is why a model cannot be started as configured, or nil.
//
// Both refusals are things the engine would also refuse, but only after
// loading the target's weights: minutes later, as a traceback, with the GPUs
// held in the meantime.
func (s *Server) launchBlocker(m *models.Model) error {
	if m.IsDraft() {
		return fmt.Errorf("%s is a draft model: it drafts for another model and cannot be served on its own", displayNameOf(m))
	}
	if m.Helper {
		return fmt.Errorf("%s is the helper model: autoconfigure runs it, and it is not served on its own", displayNameOf(m))
	}
	return missingDraft(m.VLLMConfig.SpeculativeConfig)
}

// missingDraft reports a speculative config whose draft directory is not
// there. A Hub repo id is left alone: the engine fetches those itself.
func missingDraft(speculativeConfig string) error {
	if path := missingDraftPath(speculativeConfig); path != "" {
		return fmt.Errorf("the speculative config names a draft model at %s, which is not there; "+
			"download it again or clear the speculative config", path)
	}
	return nil
}

// missingDraftPath is the draft directory a speculative config names and the
// disk does not have, or "".
func missingDraftPath(speculativeConfig string) string {
	ref, ok := models.ParseSpeculative(speculativeConfig)
	if !ok || ref.LocalDraft() == "" {
		return ""
	}
	if _, err := os.Stat(ref.LocalDraft()); err != nil {
		return ref.LocalDraft()
	}
	return ""
}

// draftWarning is what to say about the draft a model's config names, or ""
// when there is nothing to say. It warns and never refuses, for the reason
// envBlockWarning gives: the panel autosaves, and the start is where a
// pairing that cannot work is stopped.
func (s *Server) draftWarning(m *models.Model) string {
	ref, ok := models.ParseSpeculative(m.VLLMConfig.SpeculativeConfig)
	if !ok || ref.LocalDraft() == "" {
		return ""
	}
	if path := missingDraftPath(m.VLLMConfig.SpeculativeConfig); path != "" {
		return fmt.Sprintf("The speculative config names a draft model at %s, which is not there. "+
			"This model will not start until it is downloaded again or the speculative config is cleared.", path)
	}
	draft := s.draftAt(ref.LocalDraft())
	if draft == nil {
		return ""
	}
	if why := models.DraftMismatch(m, draft); why != "" {
		return fmt.Sprintf("%s does not look like a draft for this model: %s.", displayNameOf(draft), why)
	}
	if max := draft.MaxDraftTokens(); max > 0 && ref.Tokens > max {
		return fmt.Sprintf("%s drafts at most %d tokens per step, and the speculative config asks for %d.",
			displayNameOf(draft), max, ref.Tokens)
	}
	return ""
}

// draftOption is one entry in the config panel's draft picker. Value is the
// whole speculative config that choosing it writes: the JSON field stays the
// single place the setting lives, and the picker is a way of filling it in.
type draftOption struct {
	Label    string
	Value    string
	Selected bool
	Disabled bool
}

// draftOptions lists the downloaded drafts for one model's picker, or nil
// when there are none -- in which case the picker is not shown at all.
func (s *Server) draftOptions(m *models.Model) []draftOption {
	ref, _ := models.ParseSpeculative(m.VLLMConfig.SpeculativeConfig)
	current := ref.LocalDraft()

	var opts []draftOption
	for _, d := range s.registry.List() {
		if !d.IsDraft() || d.Orphaned {
			continue
		}
		opt := draftOption{Label: displayNameOf(d)}
		switch why := models.DraftMismatch(m, d); {
		case d.HFConfig.Draft.Method == "":
			opt.Label += " — unrecognised kind; write its config by hand"
			opt.Disabled = true
		case why != "":
			opt.Label += " — not for this model: " + why
			opt.Disabled = true
		default:
			opt.Value = d.SpeculativeConfigFor(0)
			opt.Label += fmt.Sprintf(" (%s, %d tokens)", d.HFConfig.Draft.Method, d.MaxDraftTokens())
		}
		if current != "" && filepath.Clean(d.LocalPath) == current {
			// Whatever is saved now is what this option stands for, so that
			// a hand-edited token count is not silently reset by the option
			// merely being the selected one.
			opt.Selected = true
			opt.Disabled = false
			opt.Value = strings.TrimSpace(m.VLLMConfig.SpeculativeConfig)
		}
		opts = append(opts, opt)
	}
	return opts
}

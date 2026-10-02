package autoconfig

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"sort"
	"strings"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Deps is what a run needs from the server around it. Everything that
// touches the engine, the Hub or the registry comes in as a value or a
// function, so a run is tested with fakes.
type Deps struct {
	Model *models.Model
	// Image is the image this machine serves with, for checking a card that
	// names the one it needs.
	Image Image
	// Base is the live config when the run began.
	Base models.VLLMConfig
	// Previous returns an earlier reading of the card with this hash, when
	// there is one to reuse.
	Previous func(cardHash string) (json.RawMessage, bool)
	// Reread ignores Previous and asks the helper again.
	Reread bool

	Fetcher    Fetcher
	Flags      FlagChecker
	Backends   []string
	MachineEnv []string
	Drafts     []*models.Model

	// Plan fits a config to this machine, for the context class the run is
	// for.
	Plan func(base models.VLLMConfig, cardKVDtype string) models.FitPlan

	// Helper asks the helper model; nil when there is none to ask, and
	// NoHelperWhy then says why.
	Helper      CallFunc
	NoHelperWhy string
	CardChars   int

	// Hub looks up a draft the card names; Installed returns the registry
	// model with a repository's ID, or nil.
	Hub       Hub
	Installed func(repo string) *models.Model

	Progress func(string)
}

// Result is a finished run: everything the review shows, and what saving
// needs.
type Result struct {
	ModelID string
	// ImageWarning says the card needs an image this machine is not running.
	ImageWarning string
	Class        models.ContextClass
	Base         models.VLLMConfig
	Plan         models.FitPlan
	Rows         []Row
	// Notes are validation's and the run's own. The plan's notes stay on
	// Plan, so a re-plan at save replaces them rather than leaving stale
	// ones behind.
	Notes       []models.ProfileNote
	CardSources []string
	CardHash    string
	Advice      json.RawMessage
	// CardKVDtype is kept so a re-plan at save still honours the card.
	CardKVDtype string
	// AdviceFrom is "helper" when the helper read the card in this run,
	// "previous" when an earlier reading was reused, "" when the text was
	// not read.
	AdviceFrom string
	WantDraft  string
	DraftRepo  string
	// Suggestions are drafts the card names that are not in use.
	Suggestions []DraftSuggestion

	plan func(base models.VLLMConfig, cardKVDtype string) models.FitPlan
}

// Run proposes a configuration for one model.
func Run(ctx context.Context, d Deps, class models.ContextClass) (*Result, error) {
	progress := d.Progress
	if progress == nil {
		progress = func(string) {}
	}
	if d.Model == nil {
		return nil, fmt.Errorf("no model")
	}
	res := &Result{ModelID: d.Model.ID, Class: class, Base: d.Base, plan: d.Plan}
	runNote := func(reason string) {
		res.Notes = append(res.Notes, models.ProfileNote{Reason: reason, Origin: "default"})
	}

	progress("Reading the model card")
	var card Card
	if d.Fetcher != nil {
		chars := d.CardChars
		card = FetchCard(ctx, d.Fetcher, d.Model.ID, chars)
	}
	res.CardSources, res.CardHash = card.Sources, card.Hash
	commands := ExtractCommands(card.Raw)
	inline := InlineFlags(card.Raw)
	if card.Raw == "" {
		runNote("No model card was found, so these settings come from this machine alone.")
	}

	var adv Advice
	hasAdvice := false
	if card.Raw != "" {
		if prev, ok := previousAdvice(d, card.Hash); ok && json.Unmarshal(prev, &adv) == nil {
			progress("Using the card advice from the last run")
			hasAdvice, res.AdviceFrom, res.Advice = true, "previous", prev
		} else if d.Helper != nil {
			// The helper call reports its own progress: stopping, starting,
			// asking, restoring, in the order the engine work happens.
			a, err := Ask(ctx, d.Helper, d.Model, card, commands, d.CardChars)
			switch {
			case err != nil && ctx.Err() != nil:
				return nil, ctx.Err()
			case err != nil:
				runNote("The helper model could not read the card (" + err.Error() + "), so the card's text was not used. Its command still was.")
			default:
				adv, hasAdvice, res.AdviceFrom = a, true, "helper"
				res.Advice, _ = json.Marshal(a)
			}
		} else if d.NoHelperWhy != "" {
			runNote(d.NoHelperWhy)
		}
	}

	progress("Checking the answer")
	in := Inputs{
		Model: d.Model, Base: d.Base, Card: card, Commands: commands, Inline: inline,
		Advice: adv, HasAdvice: hasAdvice,
		Flags: d.Flags, Backends: d.Backends, MachineEnv: d.MachineEnv, Drafts: d.Drafts,
		Image: d.Image,
	}
	checked := Validate(in)
	res.Rows, res.CardKVDtype = checked.Rows, checked.CardKVDtype
	res.ImageWarning = checked.ImageWarning
	res.WantDraft, res.DraftRepo = checked.WantDraft, checked.DraftRepo
	res.Notes = append(res.Notes, checked.Notes...)
	if res.AdviceFrom == "helper" {
		in.HasAdvice, in.Advice = false, Advice{}
		without := Validate(in)
		if len(checked.Rows) == len(without.Rows) && len(checked.Notes) == len(without.Notes) {
			runNote("The helper found nothing in the card's text beyond its command.")
		}
	}

	if res.WantDraft != "" && d.Hub != nil {
		progress("Looking for the draft model the card recommends")
		sugg, err := FindDraftSuggestions(ctx, d.Hub, d.Model, res.WantDraft, res.DraftRepo, d.Installed)
		res.Suggestions = sugg
		res.Notes = finishDraftNote(res.Notes, res.DraftRepo, sugg, err)
	}

	progress("Checking what fits on this machine")
	// Planned on the config the card produces: a speculative config adds a
	// drafter to what the model needs, and an offload variable takes tens of
	// gigabytes off it.
	planBase, err := Apply(d.Base, res.Rows, nil)
	if err != nil {
		runNote("The card's settings could not be combined (" + err.Error() + "), so the fit was planned on the current config.")
		planBase = d.Base
	}
	if d.Plan != nil {
		res.Plan = d.Plan(planBase, res.CardKVDtype)
	}

	sortNotes(res.Notes)
	return res, nil
}

func previousAdvice(d Deps, hash string) (json.RawMessage, bool) {
	if d.Reread || d.Previous == nil || hash == "" {
		return nil, false
	}
	return d.Previous(hash)
}

// sortNotes puts notes about a field together, in field order, and general
// notes last, keeping each group's own order.
func sortNotes(notes []models.ProfileNote) {
	sort.SliceStable(notes, func(i, j int) bool {
		a, b := notes[i].Field, notes[j].Field
		if (a == "") != (b == "") {
			return b == ""
		}
		return a < b
	})
}

// Config is the proposal for a width choice and a set of ticked rows. When
// the ticks differ from the defaults the fit is planned again, since
// unticking a speculative config or an offload variable changes what the
// model needs; the plan used is returned so the caller can say when it
// differs from the one the review showed.
func (r *Result) Config(width string, ticked map[string]bool) (models.VLLMConfig, models.FitPlan, error) {
	if ticked == nil {
		ticked = DefaultTicks(r.Rows)
	}
	cfg, err := Apply(r.Base, r.Rows, ticked)
	if err != nil {
		return r.Base, r.Plan, err
	}
	plan := r.Plan
	if !maps.Equal(ticked, DefaultTicks(r.Rows)) && r.plan != nil {
		plan = r.plan(cfg, r.CardKVDtype)
	}
	if w := ChosenWidth(plan, width); w != nil {
		cfg.TensorParallelSize = w.Config.TensorParallelSize
		cfg.MaxModelLen = w.Config.MaxModelLen
		cfg.GPUMemoryUtilization = w.Config.GPUMemoryUtilization
		cfg.MaxNumSeqs = w.Config.MaxNumSeqs
		cfg.KVCacheDtype = w.Config.KVCacheDtype
		cfg.KVCacheMemory = 0
		if w.Offload {
			// The plan's own config has the flag on and any MTP config out;
			// see models.planOffload for why MTP cannot stay.
			cfg.ExtraFlags = models.CarryOffloadFlags(cfg.ExtraFlags, w.Config.ExtraFlags)
			if w.Config.SpeculativeConfig == "" {
				cfg.SpeculativeConfig = ""
			}
		}
	}
	return cfg, plan, nil
}

// ChosenWidth is the width plan a width choice selects, or nil when the plan
// is not known.
func ChosenWidth(plan models.FitPlan, width string) *models.WidthPlan {
	if !plan.Known {
		return nil
	}
	switch {
	case width == "narrow" && plan.Narrow != nil:
		return plan.Narrow
	case width == "offload" && plan.Offload != nil:
		return plan.Offload
	}
	return &plan.All
}

// Profile is the Autoconfig profile to save for a width choice and a set of
// ticked rows.
func (r *Result) Profile(width string, ticked map[string]bool, meta models.ProfileMeta) (models.ConfigProfile, models.FitPlan, error) {
	if ticked == nil {
		ticked = DefaultTicks(r.Rows)
	}
	cfg, plan, err := r.Config(width, ticked)
	if err != nil {
		return models.ConfigProfile{}, plan, err
	}
	if ChosenWidth(plan, width) == &plan.All {
		width = "all"
	}

	var notes []models.ProfileNote
	for _, row := range r.Rows {
		if !ticked[row.Key] {
			continue
		}
		n := models.ProfileNote{Field: row.Field, Reason: row.Reason, Origin: row.Origin}
		switch row.Kind {
		case RowFlag:
			n.Field, n.Reason = "extra_flags", row.Flag[0]+": "+row.Reason
		case RowEnv:
			n.Field, n.Reason = "env", row.Env+": "+row.Reason
		}
		notes = append(notes, n)
	}
	if w := ChosenWidth(plan, width); w != nil {
		notes = append(notes, w.Notes...)
	}
	notes = append(notes, plan.Notes...)
	notes = append(notes, r.Notes...)

	return models.ConfigProfile{
		Config:         cfg,
		Variant:        meta.Variant,
		VariantVersion: meta.VariantVersion,
		Source:         models.ProfileSourceAutoconfig,
		Notes:          notes,
		Autoconfig: &models.AutoconfigRecord{
			At:          time.Now().UTC(),
			Class:       r.Class,
			Width:       width,
			FirstGuess:  plan.FirstGuess,
			CardSources: r.CardSources,
			CardHash:    r.CardHash,
			Advice:      r.Advice,
		},
	}, plan, nil
}

// finishDraftNote ends the "draft is not installed" note with what can be
// done about it.
func finishDraftNote(notes []models.ProfileNote, repo string, sugg []DraftSuggestion, err error) []models.ProfileNote {
	var tail string
	switch {
	case err != nil:
		tail = " The draft the card names, " + repo + ", could not be checked (" + err.Error() + ")."
	case len(sugg) > 0 && !sugg[0].Installed:
		tail = " It can be downloaded below."
	case len(sugg) > 0:
		tail = " An installed copy, " + repo + ", cannot be used: " + sugg[0].Why + "."
	case repo != "":
		tail = " The card names " + repo + ", but the Hub has no such repository."
	default:
		tail = " The card does not name a repository for it."
	}
	for i, n := range notes {
		if n.Field == "speculative_config" && strings.Contains(n.Reason, "not installed") {
			notes[i].Reason += tail
		}
	}
	return notes
}

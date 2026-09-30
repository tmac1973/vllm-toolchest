package api

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// refinementContextStep is the least change worth proposing: a refinement
// that moves the context by less than this is noise in the arithmetic.
const refinementContextStep = 1024

// refinement is what a measured start says the context of an autoconfigured
// model could be.
type refinement struct {
	MeasuredWhen        string
	CurrentContext      string
	ProposedContext     string
	FullContextRequests int
	More                bool
	proposed            int
	plan                models.FitPlan
}

// refinementFor is the refinement on offer for m, or nil. It needs the
// Autoconfig profile active -- modified or not -- with the record of the
// context class asked for, a measurement that applies to the live config,
// and no failed start to deal with first.
func (s *Server) refinementFor(m *models.Model) *refinement {
	name, _ := s.registry.ActiveProfile(m.ID)
	if name != models.AutoconfigProfileName {
		return nil
	}
	p, ok := s.registry.Profile(m.ID, models.AutoconfigProfileName)
	if !ok || p.Autoconfig == nil {
		return nil
	}
	est, ok := models.MeasuredEstimate(m, s.engineIdentity())
	if !ok {
		return nil
	}
	if s.startFixFor(m) != nil {
		return nil
	}

	// Only the context moves: the width, the KV dtype and the batch are what
	// the measurement describes, and changing any of them would retire it.
	in := s.planInput(m, m.VLLMConfig, p.Autoconfig.Class, m.VLLMConfig.KVCacheDtype)
	in.FixedTP = max(1, m.VLLMConfig.TensorParallelSize)
	plan := models.PlanFit(in)
	if !plan.Known || !plan.All.Measured {
		return nil
	}
	current := m.VLLMConfig.MaxModelLen
	proposed := plan.All.ContextTokens
	if d := proposed - current; d > -refinementContextStep && d < refinementContextStep {
		return nil
	}
	return &refinement{
		MeasuredWhen:        est.MeasuredAt.Format("2 Jan 15:04"),
		CurrentContext:      groupThousands(current),
		ProposedContext:     groupThousands(proposed),
		FullContextRequests: plan.All.FullContextRequests,
		More:                proposed > current,
		proposed:            proposed,
		plan:                plan,
	}
}

// autoconfigRefineView is the refinement review.
type autoconfigRefineView struct {
	ModelID, SafeID string
	Refinement      *refinement
	Hardware        hardwareTableView
}

// handleRefinementReview shows the hardware table with the refined context.
func (s *Server) handleRefinementReview(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "That model is no longer in the registry."})
		return
	}
	v := autoconfigRefineView{ModelID: id, SafeID: safeID(id), Refinement: s.refinementFor(m)}
	if v.Refinement != nil {
		v.Hardware = hardwareTable(v.SafeID, m.VLLMConfig, v.Refinement.plan, nil)
	}
	respondHTML(w)
	s.renderPartial(w, "autoconfig_refine", v)
}

// handleApplyRefinement applies the refined context, recomputed here: the
// form carries no values.
func (s *Server) handleApplyRefinement(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "That model is no longer in the registry."})
		return
	}
	ref := s.refinementFor(m)
	if ref == nil {
		s.renderConfigPanel(w, m, panelBanner{Error: "Nothing to apply: the configured context already matches the measurement."})
		return
	}
	if s.registry.ReadOnly() != "" {
		s.renderConfigPanel(w, m, panelBanner{Error: "Not applied: the registry is read-only."})
		return
	}
	cfg, err := applyToConfig(m.VLLMConfig, "max_model_len", strconv.Itoa(ref.proposed))
	if err != nil {
		s.renderConfigPanel(w, m, panelBanner{Error: "Not applied: " + err.Error()})
		return
	}
	note := models.ProfileNote{Field: "max_model_len", Origin: "this machine",
		Reason: fmt.Sprintf("Context set to %d from a real start measured on %s.", ref.proposed, ref.MeasuredWhen)}
	s.applyToAutoconfigured(m, cfg, []models.ProfileNote{note}, func(rec *models.AutoconfigRecord) {
		rec.FirstGuess = false
	})
	m, _ = s.registry.Get(id)
	s.renderConfigPanel(w, m, panelBanner{OK: "Applied. It takes effect the next time this model starts."})
}

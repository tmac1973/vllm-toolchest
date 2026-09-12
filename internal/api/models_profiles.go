package api

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Named config profiles, driven from the config panel.
//
// Every handler here ends in renderConfigPanel, so the panel after a profile
// action is produced by the same code that produces it on open, and what the
// action has to say is a banner inside the panel rather than an error swapped
// over the form the operator was working in. For the same reason they answer
// 200 even when they refuse: htmx swaps only a 2xx response.

// panelBanner is what a profile action wants to say on the panel it re-renders.
type panelBanner struct{ OK, Warning, Error string }

func (s *Server) renderConfigPanel(w http.ResponseWriter, m *models.Model, b panelBanner) {
	v := s.newModelConfigView(m)
	v.Banner = b
	respondHTML(w)
	s.renderPartial(w, "model_config", v)
}

// profileLabel is how a profile reads in the restore picker. The date and the
// image are inline rather than in a tooltip, because a profile saved on another
// image is the one thing worth seeing before restoring it. The date is
// absolute: the config panel is a golden recording, and "2 days ago" is not
// reproducible.
func profileLabel(p models.ConfigProfile) string {
	label := p.Name + " — saved " + p.SavedAt.Format("2006-01-02")
	if p.Variant != "" {
		label += " on " + p.Variant
	}
	return label
}

// profileRequest resolves what every profile handler starts from: the model in
// the query string and the profile name in the form.
func (s *Server) profileRequest(w http.ResponseWriter, r *http.Request) (*models.Model, string, bool) {
	m, ok := s.registry.Get(r.URL.Query().Get("id"))
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return nil, "", false
	}
	r.ParseForm()
	return m, r.FormValue("name"), true
}

func (s *Server) handleSaveModelProfile(w http.ResponseWriter, r *http.Request) {
	m, name, ok := s.profileRequest(w, r)
	if !ok {
		return
	}

	// No backend check here. Save-as names the config already stored, which
	// passed handleUpdateModelConfig's check on its way in, and the image it
	// will eventually be restored onto is not knowable now. The check belongs
	// where a profile becomes the live config on a particular image: restore.
	replaced, err := s.registry.SaveProfile(m.ID, name, models.ProfileMeta{
		Variant: s.vllmEnv.Variant, VariantVersion: s.vllmEnv.VariantVersion,
	})
	switch {
	case errors.Is(err, models.ErrProfileName):
		s.renderConfigPanel(w, m, panelBanner{Error: "Type a name for these settings first."})
		return
	case err != nil:
		s.renderConfigPanel(w, m, panelBanner{Error: "Not saved: " + err.Error()})
		return
	}

	verb := "Saved"
	if replaced {
		verb = "Replaced"
	}
	s.renderConfigPanel(w, m, panelBanner{OK: verb + " profile " + models.NormalizeProfileName(name) + "."})
}

func (s *Server) handleApplyModelProfile(w http.ResponseWriter, r *http.Request) {
	m, name, ok := s.profileRequest(w, r)
	if !ok {
		return
	}
	p, ok := s.registry.Profile(m.ID, name)
	if !ok {
		s.renderConfigPanel(w, m, panelBanner{Error: "There is no profile named " + name + "."})
		return
	}

	// The check the config PUT makes, and with more reason: a profile is a
	// config written at another time, possibly on another image, which is
	// exactly the provenance backend_refs.go was written about. Refused rather
	// than restored with a warning. A restore replaces the live config with no
	// undo, and what it would buy is an engine that dies minutes into a load.
	d, known := s.vllmEnv.Descriptor()
	if err := validateNamedBackends(d, known, p.Config.SpeculativeConfig, p.Config.ExtraFlags); err != nil {
		s.renderConfigPanel(w, m, panelBanner{Error: "Profile " + p.Name + " was not restored: " + err.Error()})
		return
	}

	if _, err := s.registry.ApplyProfile(m.ID, p.Name); err != nil {
		s.renderConfigPanel(w, m, panelBanner{Error: "Not restored: " + err.Error()})
		return
	}

	b := panelBanner{OK: "Restored profile " + p.Name + ". It applies on the next restart."}
	// The picker's own backend is warned about, not refused. The picker keeps
	// a value this image does not offer as a labelled option, so the operator
	// can see it and change it; refusing would instead leave a profile that
	// can never be restored on this image, since a profile cannot be edited.
	if warn := unofferedBackendWarning(d, known, p.Config.AttentionBackend); warn != "" {
		b.Warning = warn
	} else if p.Variant != "" && p.Variant != s.vllmEnv.Variant {
		b.Warning = fmt.Sprintf("It was saved on the %s image, and this is %s. Check the settings below before starting.",
			p.Variant, s.vllmEnv.Variant)
	}
	s.renderConfigPanel(w, m, b)
}

func (s *Server) handleDeleteModelProfile(w http.ResponseWriter, r *http.Request) {
	m, name, ok := s.profileRequest(w, r)
	if !ok {
		return
	}
	// Checked first so the refusal is not reported as a missing profile. The
	// panel already says why the registry is read-only.
	if s.registry.ReadOnly() != "" {
		s.renderConfigPanel(w, m, panelBanner{Error: "Not deleted: the registry is read-only."})
		return
	}
	if !s.registry.DeleteProfile(m.ID, name) {
		s.renderConfigPanel(w, m, panelBanner{Error: "There is no profile named " + name + "."})
		return
	}
	s.renderConfigPanel(w, m, panelBanner{
		OK: "Deleted profile " + models.NormalizeProfileName(name) + ". The settings below are unchanged.",
	})
}

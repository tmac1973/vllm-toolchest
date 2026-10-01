package api

import (
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// startFix is what the config panel offers after a start of this model
// failed: the engine's own explanation, and the change that answers it.
type startFix struct {
	Why     string
	Changes []startFixChange
}

type startFixChange struct {
	Field, Label, Current, Proposed, Reason string
	// value is what applyToConfig is given.
	value string
	note  string
}

// startFixFor is the fix for m's last start, or nil when there is none: the
// last start was of another model, it did not fail, the engine said nothing
// that points at a change, or the configuration has changed since. The advice
// belongs to the last start and is cleared by the next one, so this needs no
// state of its own.
func (s *Server) startFixFor(m *models.Model) *startFix {
	if s.process == nil {
		return nil
	}
	st := s.process.GetStatus()
	if st.ModelID != m.ID || !(st.State == process.StateError || (st.State == process.StateStarting && st.StartFailed)) {
		return nil
	}
	// The notice describes the start that failed. Once the configuration
	// differs from what that start was given -- by this fix or by hand --
	// it is out of date.
	if !slices.Equal(process.BuildArgs(m.StartConfig()), st.Args) {
		return nil
	}

	c := m.VLLMConfig
	fix := &startFix{}
	byField := map[string]startFixChange{}
	for _, it := range s.process.Advice() {
		if it.Severity != advice.Error {
			continue
		}
		if fix.Why == "" && it.Line != "" {
			fix.Why = strings.TrimSpace(it.Line)
		}
		switch it.Field {
		case "max_model_len":
			if reported, err := strconv.Atoi(it.Suggested); err == nil && reported > 0 {
				n := reported / 1024 * 1024
				// A ceiling the engine reported beats halving, which is a
				// guess.
				byField["max_model_len"] = startFixChange{
					Field: "max_model_len", Label: "Context length",
					Current: strconv.Itoa(c.MaxModelLen), Proposed: strconv.Itoa(n), value: strconv.Itoa(n),
					Reason: "The engine reported room for this many tokens.",
					note:   fmt.Sprintf("Context lowered from %d to %d after a start failed: the engine reported room for %d tokens.", c.MaxModelLen, n, reported),
				}
			} else if _, have := byField["max_model_len"]; !have && c.MaxModelLen > 2048 {
				n := max(2048, c.MaxModelLen/2/1024*1024)
				byField["max_model_len"] = startFixChange{
					Field: "max_model_len", Label: "Context length",
					Current: strconv.Itoa(c.MaxModelLen), Proposed: strconv.Itoa(n), value: strconv.Itoa(n),
					Reason: "The engine ran out of memory without naming a figure; half the context is a first step down.",
					note:   fmt.Sprintf("Context halved from %d to %d after a start ran out of memory.", c.MaxModelLen, n),
				}
			}
		case "gpu_memory_utilization":
			if f, err := strconv.ParseFloat(it.Suggested, 64); err == nil && f > 0 && f <= 1 {
				byField["gpu_memory_utilization"] = startFixChange{
					Field: "gpu_memory_utilization", Label: "GPU memory utilization",
					Current: fmt.Sprintf("%.2f", c.GPUMemoryUtilization), Proposed: fmt.Sprintf("%.2f", f), value: it.Suggested,
					Reason: "The engine reported only this share of the card free.",
					note:   fmt.Sprintf("GPU memory utilization lowered from %.2f to %.2f after a start failed: the engine reported only %.2f of the card free.", c.GPUMemoryUtilization, f, f),
				}
			}
		case "speculative_config":
			if c.SpeculativeConfig == "" {
				continue
			}
			fix.Why = it.Message
			byField["speculative_config"] = startFixChange{
				Field: "speculative_config", Label: "Speculative config", Current: c.SpeculativeConfig, Proposed: "removed", value: "",
				Reason: "The engine failed building it, before the model loaded.",
				note:   "Speculative decoding removed after a start failed while the engine was setting it up.",
			}
		case "extra_flags":
			flag := strings.TrimSpace(it.Suggested)
			if !strings.HasPrefix(flag, "--") {
				continue
			}
			if !process.HasFlag(c.ExtraFlags, flag) {
				// From a mapped setting: nothing here can remove it safely.
				fix.Why += " " + flag + " comes from one of the model's settings rather than its extra flags; clear it in Configure."
				continue
			}
			byField["extra_flags:"+flag] = startFixChange{
				Field: "extra_flags", Label: "Extra flags", Current: flag, Proposed: "removed", value: flag,
				Reason: "This image's vLLM does not recognise it.",
				note:   "Removed " + flag + " after a start failed: this image's vLLM does not recognise it.",
			}
		}
	}
	for _, k := range sortedKeys(byField) {
		ch := byField[k]
		if ch.Current != ch.Proposed {
			fix.Changes = append(fix.Changes, ch)
		}
	}
	if len(fix.Changes) == 0 && fix.Why == "" {
		return nil
	}
	return fix
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// handleApplyStartFix applies the fix the panel showed, recomputed here: the
// form carries no values, so a stale page cannot apply something the engine
// no longer says.
func (s *Server) handleApplyStartFix(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "That model is no longer in the registry."})
		return
	}
	fix := s.startFixFor(m)
	if fix == nil || len(fix.Changes) == 0 {
		s.renderConfigPanel(w, m, panelBanner{Error: "Nothing to apply: the failed start this was about is no longer the latest."})
		return
	}
	if reason := s.registry.ReadOnly(); reason != "" {
		s.renderConfigPanel(w, m, panelBanner{Error: "Not applied: the registry is read-only."})
		return
	}

	cfg := m.VLLMConfig
	var notes []models.ProfileNote
	for _, ch := range fix.Changes {
		var err error
		field := ch.Field
		if field == "extra_flags" {
			field = "extra_flags_remove"
		}
		if field == "speculative_config" {
			cfg.SpeculativeConfig = "" // removed, not set to a value
			notes = append(notes, models.ProfileNote{Field: ch.Field, Reason: ch.note, Origin: "this machine"})
			continue
		}
		if cfg, err = applyToConfig(cfg, field, ch.value); err != nil {
			s.renderConfigPanel(w, m, panelBanner{Error: "Not applied: " + err.Error()})
			return
		}
		notes = append(notes, models.ProfileNote{Field: ch.Field, Reason: ch.note, Origin: "this machine"})
	}
	s.applyToAutoconfigured(m, cfg, notes, nil)
	m, _ = s.registry.Get(id)
	s.renderConfigPanel(w, m, panelBanner{OK: "Applied. Start the model again when you are ready."})
}

// applyToAutoconfigured writes a corrected config live and, when the model's
// Autoconfig profile is active and unmodified, into that profile too, with
// notes saying why. A hand-modified profile is left alone: copying the fix
// into it would also copy edits the operator has not saved there.
func (s *Server) applyToAutoconfigured(m *models.Model, cfg models.VLLMConfig, notes []models.ProfileNote, edit func(*models.AutoconfigRecord)) {
	name, modified := s.registry.ActiveProfile(m.ID)
	keepProfile := name == models.AutoconfigProfileName && !modified

	if err := s.registry.UpdateConfig(m.ID, cfg); err != nil {
		return
	}
	m.VLLMConfig = cfg
	m.VRAMEstimate = models.EstimateVRAM(m, s.configuredEnvPairs(m))
	s.registry.Register(m)

	if !keepProfile {
		return
	}
	p, ok := s.registry.Profile(m.ID, models.AutoconfigProfileName)
	if !ok {
		return
	}
	p.Config = cfg
	p.Notes = append(p.Notes, notes...)
	if edit != nil && p.Autoconfig != nil {
		edit(p.Autoconfig)
	}
	s.registry.SaveProfileFrom(m.ID, models.AutoconfigProfileName, p)
	// Saving a profile does not touch the active label; UpdateConfig kept
	// it, and the profile now matches the live config again.
}

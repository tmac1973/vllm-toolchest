package api

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// applyAdvice writes one suggested value into one model's config.
//
// This is the first thing in the project that changes a configuration from
// parsed output, so it is deliberately narrow: one named field, one value, one
// explicit click, and nothing inferred. The parser has twice produced advice
// that was confidently backwards -- once telling the operator to raise
// gpu_memory_utilization when the engine had asked them to lower it -- and a
// button that acts without being asked would have propagated both.
func (s *Server) handleApplyAdvice(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	modelID := r.FormValue("model_id")
	field := r.FormValue("field")
	value := r.FormValue("value")

	m, ok := s.registry.Get(modelID)
	if !ok {
		s.renderAdvice(w, adviceOutcome{Error: "That model is no longer in the registry."})
		return
	}
	// Checked before anything is changed, following the profile handlers: a
	// refusal reported after a mutation leaves this process disagreeing with
	// what is on disk.
	if s.registry.ReadOnly() != "" {
		s.renderAdvice(w, adviceOutcome{Error: "Not applied: the registry is read-only."})
		return
	}

	cfg, err := applyToConfig(m.VLLMConfig, field, value)
	if err != nil {
		s.renderAdvice(w, adviceOutcome{Error: "Not applied: " + err.Error()})
		return
	}

	if err := s.registry.UpdateConfig(modelID, cfg); err != nil {
		s.renderAdvice(w, adviceOutcome{Error: "Not applied: " + err.Error()})
		return
	}

	// Same two steps the config form takes after a save, so the estimate and
	// the panel cannot fall out of step with what was written.
	m.VLLMConfig = cfg
	m.VRAMEstimate = models.EstimateVRAM(m, s.configuredEnvPairs(m))
	s.registry.Register(m)

	s.renderAdvice(w, adviceOutcome{
		OK: fmt.Sprintf("Set %s to %s. It takes effect on the next restart.", field, value),
	})
}

// adviceOutcome is what an apply has to say for itself.
type adviceOutcome struct{ OK, Error string }

// renderAdvice replaces the panel with itself, carrying the outcome.
//
// The panel is its own target: it lives on the server page, which has no
// notice element to swap into, and the config panel it first tried to render
// exists only on the models page. Re-rendering here also means the remaining
// suggestions come back with their buttons intact.
func (s *Server) renderAdvice(w http.ResponseWriter, outcome adviceOutcome) {
	view := s.adviceSnapshot()
	view.OK, view.Error = outcome.OK, outcome.Error

	respondHTML(w)
	s.renderPartial(w, "service_advice", view)
}

// applyToConfig returns cfg with one field replaced, or an error naming why it
// refused.
//
// Only the fields an applicable rule can name are accepted. A field the caller
// invents is refused rather than ignored, because silently doing nothing to a
// configuration is indistinguishable from doing the wrong thing to it.
func applyToConfig(cfg models.VLLMConfig, field, value string) (models.VLLMConfig, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return cfg, fmt.Errorf("no value was given")
	}

	switch field {
	case "max_model_len":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return cfg, fmt.Errorf("%q is not a context length", value)
		}
		cfg.MaxModelLen = n

	case "kv_cache_memory":
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil || n <= 0 {
			return cfg, fmt.Errorf("%q is not a size in bytes", value)
		}
		cfg.KVCacheMemory = n

	case "gpu_memory_utilization":
		f, err := strconv.ParseFloat(value, 64)
		if err != nil || f <= 0 || f > 1 {
			return cfg, fmt.Errorf("%q is not a fraction between 0 and 1", value)
		}
		cfg.GPUMemoryUtilization = f

	default:
		return cfg, fmt.Errorf("%s is not a setting this can change", field)
	}
	return cfg, nil
}

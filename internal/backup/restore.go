package backup

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Parse is the structural gate: shape errors reject the whole file, while
// value errors (unknown env variables, a tensor-parallel size this machine
// can't satisfy) are per-item concerns handled during Apply. Shape errors are
// exactly: invalid JSON, an unsupported version, or a model config entry with
// no model_id. They are collected into one rejection, and nothing is applied
// if Parse fails.
func Parse(data []byte) (*File, error) {
	var f File
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("not a valid backup file: %w", err)
	}
	if f.Version != Version {
		return nil, fmt.Errorf("backup version %d is not supported by this build (want %d)", f.Version, Version)
	}
	var shape []string
	for i, mc := range f.ModelConfigs {
		if strings.TrimSpace(mc.ModelID) == "" {
			shape = append(shape, fmt.Sprintf("model_configs[%d]: model_id is required", i))
		}
	}
	if len(shape) > 0 {
		return nil, fmt.Errorf("malformed backup file: %s", strings.Join(shape, "; "))
	}
	return &f, nil
}

// Selections chooses which sections a restore applies. Server-side and
// authoritative regardless of what the client-side preview showed.
type Selections struct {
	Settings     bool
	RuntimeEnv   bool
	Radiance     bool
	ModelConfigs bool
}

// None reports an empty selection.
func (s Selections) None() bool {
	return !s.Settings && !s.RuntimeEnv && !s.Radiance && !s.ModelConfigs
}

// SkippedItem is one item that failed to apply, with the reason.
type SkippedItem struct {
	Item   string `json:"item"`
	Reason string `json:"reason"`
}

// MissingModel is a config whose model isn't installed here. Config carries
// the already-normalized launch config; Pending says whether it was held for
// auto-claim.
type MissingModel struct {
	ModelID string            `json:"model_id"`
	Config  models.VLLMConfig `json:"config"`
	Pending bool              `json:"pending"`
}

// Report itemizes what a restore did. Error is set only for whole-file
// refusals (structural rejection, busy, empty selection), rendered through the
// same partial so a failure is always visible.
type Report struct {
	Error               string         `json:"error,omitempty"`
	Applied             []string       `json:"applied,omitempty"`
	AppliedModelConfigs int            `json:"applied_model_configs"`
	Notes               []string       `json:"notes,omitempty"`
	Warnings            []string       `json:"warnings,omitempty"`
	Skipped             []SkippedItem  `json:"skipped,omitempty"`
	NotSelected         []string       `json:"not_selected,omitempty"`
	Missing             []MissingModel `json:"missing,omitempty"`
}

// Deps injects the collaborators Apply writes through, keeping the engine
// testable without the api package. Every mutation goes through an existing
// setter on the api side.
type Deps struct {
	// ApplySettings merges the non-nil preference fields and reports which
	// ones changed value.
	ApplySettings func(Settings) (changed []string, err error)
	// CurrentEnv returns the target's live runtime environment; the engine
	// needs it to build the never-deletes merge.
	CurrentEnv func() RuntimeEnv
	// ApplyEnv stores the engine-built, engine-validated merged set.
	ApplyEnv func(RuntimeEnv) error
	// ApplyRadiance stores the Radiance switches.
	ApplyRadiance func(config.RadianceConfig) error
	// InstalledModel reports whether a model with this ID is registered.
	InstalledModel func(modelID string) bool
	// ApplyModelConfig writes one model's launch config.
	ApplyModelConfig func(modelID string, cfg models.VLLMConfig) error
	// SavePending holds a missing model's config for auto-claim. Nil disables
	// pending, and the entry stays a skip.
	SavePending func(MissingModel) error
	// NumGPUs is how many GPUs this machine has, for normalizing tensor
	// parallel size. Zero means "unknown", which normalizes nothing.
	NumGPUs int
}

// Apply runs the selected sections item by item. It never deletes: settings
// merge field-wise, the environment merges per key, model configs upsert.
func Apply(f *File, sel Selections, deps Deps) Report {
	var rep Report

	note := func(section string, present bool) {
		if present {
			rep.NotSelected = append(rep.NotSelected, section)
		}
	}

	restartReminder := false

	// ── Settings ──
	if sel.Settings && f.Settings != nil {
		s := *f.Settings
		// Defensive: a present-but-empty secret must never blank the target's
		// credential. Assemble never emits these; a hand-edited file might.
		if s.HFToken != nil && *s.HFToken == "" {
			s.HFToken = nil
			rep.Warnings = append(rep.Warnings, "settings: empty hf_token ignored")
		}
		if s.APIKey != nil && *s.APIKey == "" {
			s.APIKey = nil
			rep.Warnings = append(rep.Warnings, "settings: empty api_key ignored")
		}
		changed, err := deps.ApplySettings(s)
		switch {
		case err != nil:
			rep.Skipped = append(rep.Skipped, SkippedItem{"settings", err.Error()})
		case len(changed) > 0:
			rep.Applied = append(rep.Applied, "settings: "+strings.Join(changed, ", "))
			restartReminder = true
		default:
			rep.Applied = append(rep.Applied, "settings: no changes")
		}
	} else if !sel.Settings {
		note("settings", f.Settings != nil)
	}

	// ── Runtime environment: per-key merge, honouring never-deletes ──
	if sel.RuntimeEnv && f.RuntimeEnv != nil {
		cur := deps.CurrentEnv()
		merged := RuntimeEnv{Curated: map[string]string{}, Extra: cur.Extra}
		for k, v := range cur.Curated {
			merged.Curated[k] = v
		}
		for k, v := range f.RuntimeEnv.Curated {
			merged.Curated[k] = v
		}
		if strings.TrimSpace(f.RuntimeEnv.Extra) != "" {
			merged.Extra = f.RuntimeEnv.Extra
		}
		set := config.EnvSet{Curated: merged.Curated, Extra: merged.Extra}
		if err := set.Validate(); err != nil {
			rep.Skipped = append(rep.Skipped, SkippedItem{"runtime env", err.Error()})
		} else if err := deps.ApplyEnv(merged); err != nil {
			rep.Skipped = append(rep.Skipped, SkippedItem{"runtime env", err.Error()})
		} else {
			rep.Applied = append(rep.Applied, fmt.Sprintf("runtime env: %d variables", len(merged.Curated)))
			rep.Warnings = append(rep.Warnings, set.Warnings()...)
			restartReminder = true
		}
	} else if !sel.RuntimeEnv {
		note("runtime env", f.RuntimeEnv != nil)
	}

	// ── Radiance switches ──
	if sel.Radiance && f.Radiance != nil {
		if err := deps.ApplyRadiance(*f.Radiance); err != nil {
			rep.Skipped = append(rep.Skipped, SkippedItem{"radiance", err.Error()})
		} else {
			rep.Applied = append(rep.Applied, "radiance switches")
			restartReminder = true
		}
	} else if !sel.Radiance {
		note("radiance", f.Radiance != nil)
	}

	// ── Model configs ──
	// Normalize first, then match, so a Missing entry carries a config that is
	// already valid for this machine and a later pending claim is a plain
	// attach rather than a second round of fixing up.
	if sel.ModelConfigs {
		for _, mc := range f.ModelConfigs {
			cfg := mc.Config

			// Tensor parallelism is the one field that cannot survive a move
			// between machines unexamined: TP must match the GPU count, and a
			// config asking for four GPUs on a two-GPU box does not fail on
			// restore, it fails minutes into a model load.
			if deps.NumGPUs > 0 && cfg.TensorParallelSize > deps.NumGPUs {
				rep.Warnings = append(rep.Warnings, fmt.Sprintf(
					"%s: tensor parallel size %d exceeds this machine's %d GPU(s) — reduced to %d",
					mc.ModelID, cfg.TensorParallelSize, deps.NumGPUs, deps.NumGPUs))
				cfg.TensorParallelSize = deps.NumGPUs
			}

			if !deps.InstalledModel(mc.ModelID) {
				missing := MissingModel{ModelID: mc.ModelID, Config: cfg}
				if deps.SavePending == nil {
					rep.Skipped = append(rep.Skipped, SkippedItem{mc.ModelID, "model not installed"})
				} else if err := deps.SavePending(missing); err != nil {
					rep.Skipped = append(rep.Skipped, SkippedItem{mc.ModelID, "pending save failed: " + err.Error()})
				} else {
					missing.Pending = true
				}
				rep.Missing = append(rep.Missing, missing)
				continue
			}

			if err := deps.ApplyModelConfig(mc.ModelID, cfg); err != nil {
				rep.Skipped = append(rep.Skipped, SkippedItem{mc.ModelID, err.Error()})
				continue
			}
			rep.AppliedModelConfigs++
			rep.Applied = append(rep.Applied, "model config: "+mc.ModelID)
		}
	} else {
		note("model configs", len(f.ModelConfigs) > 0)
	}

	if restartReminder {
		rep.Notes = append(rep.Notes, "these changes take effect the next time vLLM starts")
	}
	return rep
}

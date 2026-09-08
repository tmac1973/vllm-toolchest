package models

import (
	"log/slog"
	"sort"
	"time"
)

// PendingConfig is a model's vLLM launch config imported from a backup for a
// model that isn't installed here yet, held for auto-claim: when a model with
// the same ID registers — downloaded through any path, or found by a scan —
// the config attaches automatically.
//
// Identity is the HuggingFace repo ID and nothing else. vLLM serves a
// repository, not a file within one, so the repo ID is the whole identity —
// which is simpler than llama-toolchest's equivalent, where a config belongs
// to one quantization of one repo and the identity needs the filename too.
type PendingConfig struct {
	ModelID string     `json:"model_id"`
	Config  VLLMConfig `json:"config"`
	SavedAt time.Time  `json:"saved_at"`
}

// SetPendingConfig upserts a pending entry by model ID, so re-importing a
// backup refreshes the entry rather than duplicating it.
func (r *Registry) SetPendingConfig(p PendingConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := range r.pending {
		if r.pending[i].ModelID == p.ModelID {
			r.pending[i] = p
			return r.save()
		}
	}
	r.pending = append(r.pending, p)
	sort.Slice(r.pending, func(i, j int) bool { return r.pending[i].ModelID < r.pending[j].ModelID })
	return r.save()
}

// PendingConfigs returns a copy of the held entries, ordered by model ID.
func (r *Registry) PendingConfigs() []PendingConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]PendingConfig, len(r.pending))
	copy(out, r.pending)
	return out
}

// DiscardPendingConfig removes a held entry, reporting whether it existed.
func (r *Registry) DiscardPendingConfig(modelID string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, p := range r.pending {
		if p.ModelID == modelID {
			r.pending = append(r.pending[:i], r.pending[i+1:]...)
			r.save()
			return true
		}
	}
	return false
}

// claimPendingLocked attaches a held config to a model that has just been
// registered. Called from Register — which both the download path and the
// directory scan go through — with r.mu held.
//
// A just-arrived model has never been launched, so there is no running process
// whose configuration this could diverge from; the config simply becomes the
// model's own and is used at the next start.
func (r *Registry) claimPendingLocked(m *Model) {
	for i, p := range r.pending {
		if p.ModelID != m.ID {
			continue
		}
		m.VLLMConfig = p.Config
		r.pending = append(r.pending[:i], r.pending[i+1:]...)
		slog.Info("pending config claimed", "model", m.ID)
		return
	}
}

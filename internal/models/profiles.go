package models

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ConfigProfile is a named snapshot of one model's vLLM launch config, taken
// on demand and restored on demand. It is not a backup and not an undo step:
// nothing writes one except an operator typing a name, and nothing reads one
// except an operator picking it out of the list.
//
// Identity is the model ID plus the folded name. The scope is one model,
// deliberately: "long-ctx" means a different max_model_len on a 27B than on a
// 4B, and a global profile would have to either carry fields most models
// should not take or become a partial overlay — a merge policy, which is the
// thing this feature exists to avoid.
//
// Profiles hang off the registry envelope rather than off Model, for the
// reason pending configs do: Register replaces the whole *Model, and a
// re-download builds a fresh one, so anything stored on the old record would
// be lost exactly when a profile is most useful.
type ConfigProfile struct {
	ModelID string     `json:"model_id"`
	Name    string     `json:"name"`
	Config  VLLMConfig `json:"config"`
	SavedAt time.Time  `json:"saved_at"`

	// Variant and VariantVersion are the image this config was saved on. The
	// attention backend, speculative config and compilation config are only
	// meaningful against a particular vLLM build, so a profile carried onto
	// another image is worth flagging before it is restored.
	Variant        string `json:"variant,omitempty"`
	VariantVersion string `json:"variant_version,omitempty"`

	// Source says what wrote the profile: empty for one saved from the panel,
	// ProfileSourceAutoconfig for one autoconfigure proposed. Notes explain
	// its settings, and Autoconfig records what the proposal was built from.
	//
	// A profile saved from the panel carries none of these. Saving over the
	// Autoconfig profile by hand therefore drops them, which is right: the
	// config is no longer what autoconfigure proposed.
	Source     string            `json:"source,omitempty"`
	Notes      []ProfileNote     `json:"notes,omitempty"`
	Autoconfig *AutoconfigRecord `json:"autoconfig,omitempty"`
}

const (
	// ProfileSourceManual is a profile saved from the config panel.
	ProfileSourceManual = ""
	// ProfileSourceAutoconfig is one autoconfigure wrote.
	ProfileSourceAutoconfig = "autoconfig"
)

// AutoconfigProfileName is the one profile autoconfigure writes. A second run
// replaces it.
const AutoconfigProfileName = "Autoconfig"

// ProfileNote is one line of explanation attached to a profile.
type ProfileNote struct {
	// Field is the VLLMConfig JSON key the note is about, or "" for a note
	// about the proposal as a whole.
	Field  string `json:"field,omitempty"`
	Reason string `json:"reason"`
	// Origin is where the reason came from: "model card", "this machine",
	// "default" or "helper summary".
	Origin string `json:"origin"`
}

// AutoconfigRecord is what an Autoconfig profile was built from, kept so that
// a later refinement knows what it is refining towards and a later run can
// reuse a reading of a card that has not changed.
//
// These fields were added without a schema version bump, as VLLMConfig.Env
// and Model.Measured were: they are optional and an older build loads the
// file without them. The cost is that a build older than this one, rewriting
// models.json, drops them -- the Autoconfig profile survives as a plain
// profile with its config intact and its explanation gone.
type AutoconfigRecord struct {
	At    time.Time    `json:"at"`
	Class ContextClass `json:"class"`
	// Width is "all" or "narrow": which of the planner's two options was
	// saved.
	Width string `json:"width"`
	// FirstGuess marks hardware settings made before any measurement applied.
	FirstGuess  bool     `json:"first_guess,omitempty"`
	CardSources []string `json:"card_sources,omitempty"`
	CardHash    string   `json:"card_hash,omitempty"`
	// Advice is the helper's reading of the card, kept raw.
	Advice json.RawMessage `json:"advice,omitempty"`
}

// ProfileMeta is the provenance the registry cannot work out for itself.
type ProfileMeta struct{ Variant, VariantVersion string }

// MaxProfileNameLen bounds a name. Long enough for "mtp-8 + 128k, eager",
// short enough that the picker does not reflow the panel.
const MaxProfileNameLen = 64

var (
	ErrProfileName     = errors.New("a profile needs a name")
	ErrProfileNotFound = errors.New("no such profile")
)

// NormalizeProfileName is the stored display form: trimmed, with internal runs
// of whitespace collapsed to one space.
func NormalizeProfileName(name string) string {
	return strings.Join(strings.Fields(name), " ")
}

// profileKey folds a name for identity only, so saving "Long Ctx" over
// "long  ctx" replaces it instead of producing two entries a dropdown cannot
// tell apart.
func profileKey(name string) string {
	return strings.ToLower(NormalizeProfileName(name))
}

func (r *Registry) findProfileLocked(modelID, name string) int {
	key := profileKey(name)
	for i, p := range r.profiles {
		if p.ModelID == modelID && profileKey(p.Name) == key {
			return i
		}
	}
	return -1
}

// SaveProfile stores a model's live config under a name and makes it the
// model's active profile. An existing name is overwritten and reported by
// replaced: the caller says so afterwards rather than asking first.
func (r *Registry) SaveProfile(modelID, name string, meta ProfileMeta) (replaced bool, err error) {
	name, err = checkProfileName(name)
	if err != nil {
		return false, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return false, err
	}
	m, ok := r.models[modelID]
	if !ok {
		return false, fmt.Errorf("model not found: %s", modelID)
	}

	p := ConfigProfile{
		ModelID:        modelID,
		Name:           name,
		Config:         m.VLLMConfig,
		SavedAt:        time.Now().UTC(),
		Variant:        meta.Variant,
		VariantVersion: meta.VariantVersion,
	}
	replaced = r.putProfileLocked(p)
	m.ActiveProfile = name
	return replaced, r.save()
}

// SaveProfileFrom stores a profile whose config is not the live one -- what
// autoconfigure proposes -- under a name. The model's ID, the name and the
// time are set here, whatever p carried.
//
// Unlike SaveProfile it neither reads nor changes the live config, and it does
// not make the profile active: saving a proposal must not change what will be
// launched. ApplyProfile is what makes it live.
func (r *Registry) SaveProfileFrom(modelID, name string, p ConfigProfile) (replaced bool, err error) {
	name, err = checkProfileName(name)
	if err != nil {
		return false, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return false, err
	}
	if _, ok := r.models[modelID]; !ok {
		return false, fmt.Errorf("model not found: %s", modelID)
	}

	p.ModelID = modelID
	p.Name = name
	p.SavedAt = time.Now().UTC()
	replaced = r.putProfileLocked(p)
	return replaced, r.save()
}

// checkProfileName normalises a name and refuses one that is empty or too
// long.
func checkProfileName(name string) (string, error) {
	name = NormalizeProfileName(name)
	if name == "" {
		return "", ErrProfileName
	}
	if n := len([]rune(name)); n > MaxProfileNameLen {
		return "", fmt.Errorf("a profile name can be at most %d characters; this one is %d", MaxProfileNameLen, n)
	}
	return name, nil
}

// putProfileLocked replaces the profile with p's model and folded name, or
// inserts p in order, and reports whether it replaced one.
func (r *Registry) putProfileLocked(p ConfigProfile) (replaced bool) {
	if i := r.findProfileLocked(p.ModelID, p.Name); i >= 0 {
		r.profiles[i] = p
		return true
	}
	r.profiles = append(r.profiles, p)
	sort.Slice(r.profiles, func(i, j int) bool {
		a, b := r.profiles[i], r.profiles[j]
		if a.ModelID != b.ModelID {
			return a.ModelID < b.ModelID
		}
		return profileKey(a.Name) < profileKey(b.Name)
	})
	return false
}

// Profiles returns a copy of one model's profiles, ordered by name.
func (r *Registry) Profiles(modelID string) []ConfigProfile {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []ConfigProfile
	for _, p := range r.profiles {
		if p.ModelID == modelID {
			out = append(out, p)
		}
	}
	return out
}

// Profile returns one profile by name, matched the way SaveProfile matches.
func (r *Registry) Profile(modelID, name string) (ConfigProfile, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if i := r.findProfileLocked(modelID, name); i >= 0 {
		return r.profiles[i], true
	}
	return ConfigProfile{}, false
}

// ApplyProfile copies a saved profile over the live config, marks it active and
// recomputes the VRAM estimate. Validation is the caller's: the registry has no
// view of the image the config would be launched on.
func (r *Registry) ApplyProfile(modelID, name string) (VLLMConfig, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return VLLMConfig{}, err
	}
	m, ok := r.models[modelID]
	if !ok {
		return VLLMConfig{}, fmt.Errorf("model not found: %s", modelID)
	}
	i := r.findProfileLocked(modelID, name)
	if i < 0 {
		return VLLMConfig{}, ErrProfileNotFound
	}

	if m.VLLMConfig != r.profiles[i].Config {
		m.ConfigSource = "" // a profile's config is not what was seeded
	}
	m.VLLMConfig = r.profiles[i].Config
	m.ActiveProfile = r.profiles[i].Name
	m.VRAMEstimate = EstimateVRAM(m, m.OwnEnvPairs())
	return m.VLLMConfig, r.save()
}

// DeleteProfile removes one, reporting whether it existed. The live config is
// untouched even when the deleted profile was the active one — what is running,
// or about to be, does not change because a bookmark was dropped. The label
// goes, though: "from profile x" is not a claim to keep making once x is gone.
func (r *Registry) DeleteProfile(modelID, name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.writableLocked() != nil {
		return false
	}
	i := r.findProfileLocked(modelID, name)
	if i < 0 {
		return false
	}
	if m, ok := r.models[modelID]; ok && profileKey(m.ActiveProfile) == profileKey(r.profiles[i].Name) {
		m.ActiveProfile = ""
	}
	r.profiles = append(r.profiles[:i], r.profiles[i+1:]...)
	r.save()
	return true
}

// ActiveProfile reports which profile the live config came from and whether it
// still matches. modified is true once the config has been edited since, which
// the panel shows and a benchmark records: a run labelled "long-ctx" whose
// numbers came from something else is worse than an unlabelled one.
//
// The comparison is ==, which relies on every VLLMConfig field being a scalar.
// Adding a slice or map there stops this package compiling, which is the
// intended failure: the replacement comparison has to be written deliberately.
func (r *Registry) ActiveProfile(modelID string) (name string, modified bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[modelID]
	if !ok || m.ActiveProfile == "" {
		return "", false
	}
	i := r.findProfileLocked(modelID, m.ActiveProfile)
	if i < 0 {
		return "", false
	}
	return r.profiles[i].Name, r.profiles[i].Config != m.VLLMConfig
}

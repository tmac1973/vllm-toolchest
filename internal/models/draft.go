package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DraftMeta marks a checkpoint that is not a model in its own right: a
// drafter for speculative decoding, which proposes tokens for some other
// model to verify.
//
// It downloads, updates and takes up disk like any model, which is why it is
// registered as one. It cannot be served. Started on its own it fails several
// minutes into a load, so everything that offers to start, configure or
// benchmark a model has to know to leave it out.
type DraftMeta struct {
	// Method is the --speculative-config method this draft is used with, or
	// "" when the checkpoint is recognisably a draft but not of a kind whose
	// config this tool can write.
	Method string `json:"method,omitempty"`
	// BlockSize is how many positions the drafter predicts in one pass. One
	// of them is the token being verified, so it drafts BlockSize-1.
	BlockSize int `json:"block_size,omitempty"`
	// TargetLayers is the depth of the model it was trained to draft for.
	// With the hidden size and vocabulary, which the draft shares with its
	// target, it is what says whether a pairing can work.
	TargetLayers int `json:"target_layers,omitempty"`
}

// IsDraft reports whether the model is a drafter rather than something that
// can be served.
func (m *Model) IsDraft() bool {
	return m != nil && m.HFConfig.Draft != nil
}

// parseDraft reads a config.json for the marks of a draft model.
//
// DFlash is the only kind read in full, because it is the only kind seen: its
// config carries a dflash_config block. An architecture named "...DraftModel"
// is taken as a draft of an unknown kind, which is enough to keep it off the
// list of things that can be started.
func parseDraft(raw map[string]json.RawMessage, architectures []string) *DraftMeta {
	if block, ok := raw["dflash_config"]; ok {
		d := &DraftMeta{Method: "dflash"}
		var dflash map[string]json.RawMessage
		if json.Unmarshal(block, &dflash) == nil {
			jsonFieldFrom(dflash, &d.BlockSize, "block_size")
		}
		jsonFieldFrom(raw, &d.TargetLayers, "num_target_layers")
		return d
	}
	for _, arch := range architectures {
		if strings.HasSuffix(arch, "DraftModel") {
			return &DraftMeta{}
		}
	}
	return nil
}

// SpeculativeRef is the part of a --speculative-config this tool reads. The
// rest of the JSON is the engine's business and is passed through untouched.
type SpeculativeRef struct {
	Method string `json:"method"`
	// Model is the draft: a directory inside the container, or a Hub repo id
	// the engine fetches for itself. Empty for methods that draft with the
	// target's own head, such as MTP.
	Model  string `json:"model"`
	Tokens int    `json:"num_speculative_tokens"`
}

// ParseSpeculative reads a speculative config. ok is false for an empty or
// malformed one, which is not this function's to report: the engine says so
// at launch, and the panel it is typed into autosaves half-typed JSON.
func ParseSpeculative(raw string) (SpeculativeRef, bool) {
	var ref SpeculativeRef
	if strings.TrimSpace(raw) == "" || json.Unmarshal([]byte(raw), &ref) != nil {
		return SpeculativeRef{}, false
	}
	return ref, true
}

// LocalDraft is the directory a speculative config names as its draft, or ""
// when it names none or names a Hub repo.
func (r SpeculativeRef) LocalDraft() string {
	if filepath.IsAbs(r.Model) {
		return filepath.Clean(r.Model)
	}
	return ""
}

// MaxDraftTokens is the most tokens this draft can propose per step, or 0
// when that is not known.
func (m *Model) MaxDraftTokens() int {
	if !m.IsDraft() || m.HFConfig.Draft.BlockSize < 2 {
		return 0
	}
	return m.HFConfig.Draft.BlockSize - 1
}

// SpeculativeConfigFor is the --speculative-config that pairs a model with
// this draft, or "" when the draft is of a kind whose method is not known.
func (m *Model) SpeculativeConfigFor(tokens int) string {
	if !m.IsDraft() || m.HFConfig.Draft.Method == "" {
		return ""
	}
	if tokens <= 0 {
		tokens = m.MaxDraftTokens()
	}
	// Written by hand rather than marshalled so the keys come out in the
	// order every model card prints them, and the field reads the same
	// whether the picker or a person filled it in.
	path, _ := json.Marshal(m.LocalPath)
	return fmt.Sprintf(`{"method": %q, "model": %s, "num_speculative_tokens": %d}`,
		m.HFConfig.Draft.Method, path, tokens)
}

// DraftMismatch says why a draft cannot serve a target, or "" when nothing
// on record rules it out.
//
// A drafter reads the target's hidden states and writes into its vocabulary,
// so the widths have to agree, and it taps layers by index, so the depth does
// too. Only fields both configs state are compared: an unknown is not a
// mismatch.
func DraftMismatch(target, draft *Model) string {
	if !draft.IsDraft() || target == nil {
		return ""
	}
	t, d := target.HFConfig, draft.HFConfig
	switch {
	case t.HiddenSize > 0 && d.HiddenSize > 0 && t.HiddenSize != d.HiddenSize:
		return fmt.Sprintf("it was made for a hidden size of %d and this model's is %d", d.HiddenSize, t.HiddenSize)
	case t.NumHiddenLayers > 0 && d.Draft.TargetLayers > 0 && t.NumHiddenLayers != d.Draft.TargetLayers:
		return fmt.Sprintf("it was made for a %d-layer model and this one has %d", d.Draft.TargetLayers, t.NumHiddenLayers)
	case t.VocabSize > 0 && d.VocabSize > 0 && t.VocabSize != d.VocabSize:
		return fmt.Sprintf("its vocabulary is %d tokens and this model's is %d", d.VocabSize, t.VocabSize)
	}
	return ""
}

// draftWeightsGB is the size of the draft a speculative config names, read
// from its weight files. Zero when there is none or it is not on this disk.
//
// The files are the witness here as they are for the target: nothing about a
// draft's architecture is modelled, and the weights are nearly all of what it
// costs.
func draftWeightsGB(c VLLMConfig) float64 {
	ref, ok := ParseSpeculative(c.SpeculativeConfig)
	if !ok || ref.LocalDraft() == "" {
		return 0
	}
	files, _ := filepath.Glob(filepath.Join(ref.LocalDraft(), "*.safetensors"))
	var total int64
	for _, f := range files {
		if info, err := os.Stat(f); err == nil {
			total += info.Size()
		}
	}
	return BytesToGB(total)
}

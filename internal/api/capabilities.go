package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// Client auto-discovery.
//
// A client that knows nothing about this server can ask it what is being
// served and how to drive it, and configure itself from the answer: context
// to compact against, whether to send images or tools, how to switch thinking
// on and which effort levels exist, what sampling the model wants.
//
// The contract is llama-toolchest's, key for key, because the clients are the
// same ones -- Haruspex probes GET /api/service/status and then this endpoint
// and does not care which toolchest answers. The values differ where vLLM
// does; see buildCapabilities.

// CapabilitiesSchemaVersion is the version of the capabilities object. Clients
// branch on it, so it moves only on a breaking change to the shape.
const CapabilitiesSchemaVersion = 1

// discoveredModel is one entry of /api/service/loaded-models.
type discoveredModel struct {
	// ID is what a client sends as "model". vLLM is started with the repo id
	// as its served name, so this, PublicName and RegistryID are one string;
	// all three are present because llama-toolchest has three and clients
	// read whichever they were written against.
	ID           string         `json:"id"`
	Status       string         `json:"status"` // "loaded" or "loading"
	PublicName   string         `json:"public_name"`
	RegistryID   string         `json:"registry_id"`
	Capabilities map[string]any `json:"capabilities"`
}

// handleLoadedModels lists what /v1 will answer for, with everything a client
// needs to use it.
//
// That is one model or none. vLLM serves one model per process and nothing
// here loads another on demand, so listing the rest of the registry would
// advertise names that every request for them fails on -- the mistake
// handleV1Models used to make. A model still loading is listed as such: a
// client can show it and wait, which is better than finding nothing.
func (s *Server) handleLoadedModels(w http.ResponseWriter, r *http.Request) {
	out := struct {
		SchemaVersion int               `json:"schema_version"`
		Server        string            `json:"server"`
		Running       bool              `json:"running"`
		Models        []discoveredModel `json:"models"`
	}{SchemaVersion: CapabilitiesSchemaVersion, Server: "vllm-toolchest", Models: []discoveredModel{}}

	st := s.process.GetStatus()
	status := ""
	switch st.State {
	case process.StateRunning:
		status = "loaded"
		out.Running = true
	case process.StateStarting:
		if !st.StartFailed {
			status = "loading"
		}
	}
	if m, ok := s.registry.Get(st.ModelID); ok && status != "" {
		out.Models = append(out.Models, discoveredModel{
			ID:           m.ID,
			Status:       status,
			PublicName:   m.ID,
			RegistryID:   m.ID,
			Capabilities: buildCapabilities(m, st.Args),
		})
	}
	respondJSON(w, out)
}

// buildCapabilities describes a model as it is being served. args is the
// flag list the engine was launched with: the saved config may have been
// edited since, and a client has to be told what is running, not what would
// run after a restart.
//
// Every key is always present, with null or false for "known to be absent",
// so that a missing key can only mean a server too old to report it.
func buildCapabilities(m *models.Model, args []string) map[string]any {
	trained := m.HFConfig.MaxPositionEmbeddings
	served := flagInt(args, "--max-model-len")
	if served <= 0 {
		served = trained
	}

	// The scheduler's cap on requests in flight. When the flag was not passed
	// the engine chose its own, which is never lower than the saved figure,
	// so that is reported: a number the engine will certainly honour.
	parallel := flagInt(args, "--max-num-seqs")
	if parallel <= 0 {
		parallel = m.VLLMConfig.MaxNumSeqs
	}
	if parallel < 1 {
		parallel = 1
	}

	override := generationOverride(args)
	template, _ := flagValue(args, "--chat-template")

	return map[string]any{
		"schema_version": CapabilitiesSchemaVersion,

		// context_size is what is being served, and is what a client
		// compacts against; context_length is what the model was trained to,
		// for information.
		"context_size":   served,
		"context_length": trained,
		"parallel":       parallel,
		// Not divided by parallel. llama.cpp gives each slot a fixed share of
		// the context; vLLM draws every request from one KV pool, so each may
		// use the whole window, and one that does not fit yet waits for room
		// rather than failing.
		"context_per_request": served,

		"vision": m.Vision.IsVisionModel && !hasFlag(args, "--language-model-only"),
		// Whether tool calls will be parsed on this serve, which is a launch
		// flag -- not whether the model could make them.
		"tools":     hasFlag(args, "--enable-auto-tool-choice"),
		"embedding": false,

		"reasoning": models.DetectReasoning(m.LocalPath, template),
		"sampling":  buildSampling(m.GenDefaults, override),

		"max_output_tokens": maxOutputTokens(m.GenDefaults, override),
	}
}

// generationSettings is the sampling fields of an --override-generation-config.
type generationSettings struct {
	Temperature       *float64 `json:"temperature"`
	TopP              *float64 `json:"top_p"`
	TopK              *int     `json:"top_k"`
	MinP              *float64 `json:"min_p"`
	PresencePenalty   *float64 `json:"presence_penalty"`
	RepetitionPenalty *float64 `json:"repetition_penalty"`
	MaxTokens         *int     `json:"max_tokens"`
	MaxNewTokens      *int     `json:"max_new_tokens"`
}

// generationOverride reads the launch's --override-generation-config, or nil
// when there is none or it does not parse.
func generationOverride(args []string) *generationSettings {
	raw, ok := flagValue(args, "--override-generation-config")
	if !ok {
		return nil
	}
	var g generationSettings
	if json.Unmarshal([]byte(raw), &g) != nil {
		return nil
	}
	return &g
}

// buildSampling is the sampling the engine applies when a request names none:
// the launch override where it sets a value, the checkpoint's
// generation_config.json where it does not. A client that sends its own
// values replaces these, so it should be sending these.
//
// presets is empty. llama-toolchest publishes per-family thinking and
// non-thinking presets from a table it keeps; there is no such table here,
// and one default that is right beats a list that is guessed.
func buildSampling(card models.GenDefaults, override *generationSettings) map[string]any {
	pickF := func(over, base *float64) any {
		switch {
		case over != nil:
			return *over
		case base != nil:
			return *base
		}
		return nil
	}
	pickI := func(over, base *int) any {
		switch {
		case over != nil:
			return *over
		case base != nil:
			return *base
		}
		return nil
	}
	if override == nil {
		override = &generationSettings{}
	}

	def := map[string]any{
		"temperature":      pickF(override.Temperature, card.Temperature),
		"top_p":            pickF(override.TopP, card.TopP),
		"top_k":            pickI(override.TopK, card.TopK),
		"min_p":            pickF(override.MinP, nil),
		"presence_penalty": pickF(override.PresencePenalty, nil),
		"repeat_penalty":   pickF(override.RepetitionPenalty, card.RepetitionPenalty),
	}

	var source any
	switch {
	case override.Temperature != nil || override.TopP != nil || override.TopK != nil ||
		override.MinP != nil || override.PresencePenalty != nil || override.RepetitionPenalty != nil:
		source = "override-generation-config"
	case card.Temperature != nil || card.TopP != nil || card.TopK != nil || card.RepetitionPenalty != nil:
		source = "generation_config.json"
	}
	return map[string]any{
		"source":     source,
		"source_url": nil,
		"default":    def,
		"presets":    []any{},
	}
}

// maxOutputTokens is the completion length the engine uses when a request
// gives none, or nil when it imposes none.
func maxOutputTokens(card models.GenDefaults, override *generationSettings) any {
	if override != nil {
		switch {
		case override.MaxTokens != nil:
			return *override.MaxTokens
		case override.MaxNewTokens != nil:
			return *override.MaxNewTokens
		}
	}
	if card.MaxNewTokens != nil {
		return *card.MaxNewTokens
	}
	return nil
}

// flagValue finds a launch flag's value, written as `--flag value` or
// `--flag=value`.
func flagValue(args []string, name string) (string, bool) {
	for i, a := range args {
		if a == name && i+1 < len(args) {
			return args[i+1], true
		}
		if v, ok := strings.CutPrefix(a, name+"="); ok {
			return v, true
		}
	}
	return "", false
}

func flagInt(args []string, name string) int {
	v, ok := flagValue(args, name)
	if !ok {
		return 0
	}
	n, _ := strconv.Atoi(v)
	return n
}

func hasFlag(args []string, name string) bool {
	for _, a := range args {
		if a == name {
			return true
		}
	}
	return false
}

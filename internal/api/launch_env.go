package api

import (
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/variants"
)

// launchEnv builds the environment for a vLLM launch: the process manager's
// own requirements, then what the image's manifest says its stack needs, then
// the configured runtime environment, then the variant's feature knobs, then
// the model's own.
//
// Order is the contract. The command is run with the container environment
// plus these appended, and the last occurrence of a name wins, so each layer
// overrides the one before it:
//
//	container image → BuildEnv defaults → manifest image env → runtime env → knobs → model
//
// The manifest layer sits low because it carries the image author's defaults,
// not the operator's choices. The knobs go after the runtime environment
// because each one is a control the operator can see, and a free-text box
// silently overriding a visible switch is the worse failure. The Settings page
// warns when a knob's variable is typed into the extra-environment box for
// that reason.
//
// The model goes last because it is the most specific statement available: a
// variable like VLLM_PLE_CPU_OFFLOAD describes one checkpoint, and setting it
// machine-wide to satisfy that checkpoint imposes it on every other model on
// the host.
//
// It takes the model rather than a quantization method and an environment,
// because those must come from the same record — a caller that passed one and
// forgot the other would serve a model under an environment it was never
// configured with. Every launch path — start, restart, benchmark jobs, the
// context probe — must go through here for the same reason.
func (s *Server) launchEnv(m *models.Model) []string {
	quantMethod := ""
	if m != nil {
		quantMethod = m.Quantization.Method
	}
	return process.BuildEnv(quantMethod, s.configuredEnvPairs(m)...)
}

// configuredEnvPairs is the configured part of a launch environment, in the
// order launchEnv applies it: manifest image env, runtime env, feature knobs,
// then the model's own. It excludes the process manager's own defaults, which
// BuildEnv adds and which nobody configures.
//
// It exists so the effective-environment preview and the actual launch cannot
// disagree. They used to assemble the same layers separately, which was fine
// until a layer was added to one of them.
//
// A nil model means the machine-wide environment alone — what the Settings
// page shows, where no model is in view.
func (s *Server) configuredEnvPairs(m *models.Model) []string {
	pairs := append(s.variantImageEnv(), s.cfg.RuntimeEnvPairs()...)
	pairs = append(pairs, s.cfg.KnobEnv(s.vllmEnv.Variant)...)
	if m != nil {
		// Parsed by the same code as the machine-wide block, so a malformed
		// line is dropped here exactly as it is there, and the config panel
		// has already warned about it.
		pairs = append(pairs, config.EnvSet{Extra: m.VLLMConfig.Env}.Pairs()...)
	}
	return pairs
}

// variantImageEnv is the environment the running image's manifest says its
// stack needs — AITER routing, backend selection and the like.
//
// It is the lowest of the three layers launchEnv appends, so the runtime
// environment and the feature knobs both override it. That ordering is the
// point: these are the image author's defaults, not the operator's choices.
func (s *Server) variantImageEnv() []string {
	d, ok := variants.Get(s.vllmEnv.Variant)
	if !ok {
		return nil
	}
	// Copied rather than returned directly: callers append to it, and
	// appending to a slice backed by the embedded manifest would let one
	// launch's environment leak into the next.
	return append([]string(nil), d.ImageEnv...)
}

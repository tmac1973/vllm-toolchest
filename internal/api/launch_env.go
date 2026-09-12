package api

import (
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/variants"
)

// launchEnv builds the environment for a vLLM launch: the process manager's
// own requirements, then what the image's manifest says its stack needs, then
// the configured runtime environment, then the variant's feature knobs.
//
// Order is the contract. The command is run with the container environment
// plus these appended, and the last occurrence of a name wins, so each layer
// overrides the one before it:
//
//	container image → BuildEnv defaults → manifest image env → runtime env → knobs
//
// The manifest layer sits low because it carries the image author's defaults,
// not the operator's choices. The knobs go last because each one is a control
// the operator can see, and a free-text box silently overriding a visible
// switch is the worse failure. The Settings page warns when a knob's variable
// is typed into the extra-environment box for that reason.
//
// Every launch path — start, restart, benchmark jobs, the context probe —
// must go through here, or a model benchmarked under one environment gets
// served under another.
func (s *Server) launchEnv(quantMethod string) []string {
	extra := append(s.variantImageEnv(), s.cfg.RuntimeEnvPairs()...)
	extra = append(extra, s.cfg.KnobEnv(s.vllmEnv.Variant)...)
	return process.BuildEnv(quantMethod, extra...)
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

package api

import (
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// launchEnv builds the environment for a vLLM launch: the process manager's
// own requirements, then the configured runtime environment, then the
// Radiance switches.
//
// Order is the contract. The command is run with the container environment
// plus these appended, and the last occurrence of a name wins, so each layer
// overrides the one before it:
//
//	container image  →  BuildEnv defaults  →  runtime env  →  Radiance
//
// Radiance goes last because its Settings section is a dedicated tri-state
// control the user can see, and a free-text box silently overriding a visible
// switch is the worse failure. The Settings page warns when a RADIANCE_*
// variable is typed into the extra-environment box for that reason.
//
// Every launch path — start, restart, benchmark jobs, the context probe —
// must go through here, or a model benchmarked under one environment gets
// served under another.
func (s *Server) launchEnv(quantMethod string) []string {
	extra := append(s.cfg.RuntimeEnvPairs(), s.cfg.Radiance.Env()...)
	return process.BuildEnv(quantMethod, extra...)
}

package api

import (
	"os"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/config"
)

// envLine is one row of the effective-environment preview: the KEY=VALUE pair
// as it will be applied, and — when the container environment already defines
// the same name — the value being replaced.
//
// The direction matters and is the opposite of what the equivalent view in
// llama-toolchest reports. There the inherited environment wins and configured
// values are annotated as overridden. Here the vLLM process is launched with
// `append(os.Environ(), env...)`, and os/exec resolves a duplicate name to its
// last occurrence, so a value set in Settings replaces the image's. Reading
// this the wrong way round would tell someone their change had no effect when
// it is the only thing taking effect.
type envLine struct {
	Text string
	// Replaces is the container's own value for this name, empty when the
	// container does not set it.
	Replaces string
}

// effectiveEnvLines renders the configured runtime environment as the launch
// will apply it, annotating entries that replace a value the image exports.
//
// It reports the same two layers the launch does, in the same order, so the
// preview cannot disagree with what actually runs: the runtime environment
// first, then the Radiance switches. See launchEnv.
func (s *Server) effectiveEnvLines() []envLine {
	pairs := append(s.cfg.RuntimeEnvPairs(), s.cfg.Radiance.Env()...)

	// Later wins, exactly as os/exec resolves it, so a name set in both
	// layers appears once with the value that will actually apply.
	order := []string{}
	value := map[string]string{}
	for _, kv := range pairs {
		name, val := kv, ""
		if i := strings.IndexByte(kv, '='); i > 0 {
			name, val = kv[:i], kv[i+1:]
		}
		if _, seen := value[name]; !seen {
			order = append(order, name)
		}
		value[name] = val
	}

	out := make([]envLine, 0, len(order))
	for _, name := range order {
		line := envLine{Text: name + "=" + value[name]}
		if inherited, ok := os.LookupEnv(name); ok && inherited != value[name] {
			line.Replaces = name + "=" + inherited
		}
		out = append(out, line)
	}
	return out
}

// runtimeEnvView is the Settings page's view of the curated table: one row per
// option, carrying the value currently stored for it.
type runtimeEnvRow struct {
	config.RuntimeEnvOption
	Value string
}

// runtimeEnvRows pairs the curated options with their stored values.
func (s *Server) runtimeEnvRows() []runtimeEnvRow {
	opts := config.RuntimeEnvOptions()
	rows := make([]runtimeEnvRow, 0, len(opts))
	for _, o := range opts {
		rows = append(rows, runtimeEnvRow{RuntimeEnvOption: o, Value: s.cfg.RuntimeEnv[o.Name]})
	}
	return rows
}

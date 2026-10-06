package api

import (
	"slices"
	"strings"
	"testing"
)

// With a key configured, the engine is started requiring it, so its own
// published port is no way round the /v1 check. Without one, nothing changes.
func TestLaunchEnvCarriesTheAPIKey(t *testing.T) {
	s := settingsServer(t)
	if slices.ContainsFunc(s.launchEnv(nil), func(kv string) bool { return strings.HasPrefix(kv, "VLLM_API_KEY=") }) {
		t.Error("VLLM_API_KEY set with no key configured")
	}
	s.cfg.APIKey = "sk-test"
	env := s.launchEnv(nil)
	if env[len(env)-1] != "VLLM_API_KEY=sk-test" {
		t.Errorf("want VLLM_API_KEY last, so it wins; env ends %q", env[len(env)-1])
	}
}

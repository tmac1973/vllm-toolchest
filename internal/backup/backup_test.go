package backup

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

func testRegistry(t *testing.T, ids ...string) *models.Registry {
	t.Helper()
	dir := t.TempDir()
	reg := models.NewRegistry(dir, filepath.Join(dir, "models"))
	for i, id := range ids {
		if err := reg.Register(&models.Model{
			ID:         id,
			VLLMConfig: models.VLLMConfig{MaxModelLen: 4096 * (i + 1), TensorParallelSize: 1},
		}); err != nil {
			t.Fatal(err)
		}
	}
	return reg
}

func testConfig() *config.Config {
	return &config.Config{
		DataDir:       "/data",
		LogLevel:      "info",
		GPUMemoryUtil: 0.9,
		MaxNumSeqs:    16,
		Theme:         "graphite",
		HFToken:       "hf_secret",
		APIKey:        "key_secret",
		RuntimeEnv:    map[string]string{"VLLM_LOGGING_LEVEL": "DEBUG"},
	}
}

// A secret leaves the machine only on the explicit opt-in. The default export
// gets shared, pasted into issues and copied to other hosts.
func TestSecretsOnlyTravelOnOptIn(t *testing.T) {
	f := Assemble(testConfig(), testRegistry(t), "generic", nil, false)
	if f.Settings.HFToken != nil || f.Settings.APIKey != nil {
		t.Error("secrets present in a backup taken without the opt-in")
	}

	f = Assemble(testConfig(), testRegistry(t), "generic", nil, true)
	if f.Settings.HFToken == nil || *f.Settings.HFToken != "hf_secret" {
		t.Error("opted-in backup is missing the HF token")
	}
}

// An empty secret must not be emitted at all: absence is the only way the
// format says "no secret", and a present empty string would blank the
// target's credential on restore.
func TestEmptySecretsAreOmittedNotEmitted(t *testing.T) {
	cfg := testConfig()
	cfg.HFToken, cfg.APIKey = "", ""
	f := Assemble(cfg, testRegistry(t), "generic", nil, true)
	if f.Settings.HFToken != nil || f.Settings.APIKey != nil {
		t.Error("an empty secret was emitted as a present key")
	}
}

// The file describes where it came from, but a restore must never adopt the
// source's identity — a backup from another host would otherwise move this
// server's listening address.
func TestSourceIsNeverApplied(t *testing.T) {
	f := &File{
		Version: Version,
		Source:  SourceInfo{ListenAddr: ":9999", ExternalURL: "http://elsewhere", DataDir: "/other"},
		Settings: &Settings{
			Theme: ptr("cyberpunk"),
		},
	}
	applied := map[string]string{}
	Apply(f, Selections{Settings: true}, Deps{
		ApplySettings: func(in Settings) ([]string, error) {
			if in.Theme != nil {
				applied["theme"] = *in.Theme
			}
			return []string{"theme"}, nil
		},
	})
	// Settings has no field for any of them, which is the guarantee: the
	// source block is unreachable from Apply by construction, not by care.
	if applied["theme"] != "cyberpunk" {
		t.Errorf("expected the theme to be applied; got %v", applied)
	}
}

func TestModelConfigsAreSortedAndComplete(t *testing.T) {
	reg := testRegistry(t, "zeta/model", "alpha/model", "mid/model")
	f := Assemble(testConfig(), reg, "generic", nil, false)
	var got []string
	for _, mc := range f.ModelConfigs {
		got = append(got, mc.ModelID)
	}
	want := []string{"alpha/model", "mid/model", "zeta/model"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("got %v, want %v", got, want)
	}
}

// The Settings page's preview parses these exact keys in the browser. A
// renamed JSON tag would leave the preview silently reporting every section
// as absent.
func TestWireFormatKeys(t *testing.T) {
	f := Assemble(testConfig(), testRegistry(t, "a/b"), "generic", nil, false)
	data, err := f.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"version", "exported_at", "source", "settings", "runtime_env", "radiance", "model_configs"} {
		if _, ok := raw[key]; !ok {
			t.Errorf("missing top-level key %q", key)
		}
	}
}

func TestParseRejects(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"not json", "{{{", "not a valid backup file"},
		{"wrong version", `{"version": 2}`, "not supported"},
		{"empty model id", `{"version":1,"model_configs":[{"model_id":"  "}]}`, "model_id is required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse([]byte(tc.in))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("got %v, want something containing %q", err, tc.want)
			}
		})
	}
}

func TestParseAcceptsAnAssembledFile(t *testing.T) {
	data, err := Assemble(testConfig(), testRegistry(t, "a/b"), "generic", nil, true).Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(data); err != nil {
		t.Errorf("a file this package produced was rejected by its own parser: %v", err)
	}
}

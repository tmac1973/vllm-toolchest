package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

func configTestServer(t *testing.T, env vllmenv.Env) (*Server, *models.Model) {
	t.Helper()
	dir := t.TempDir()
	s := &Server{
		cfg:      &config.Config{DataDir: dir, VLLMPort: 8000},
		registry: models.NewRegistry(dir),
		vllmEnv:  env,
	}

	m := &models.Model{
		ID:        "org/model",
		LocalPath: dir,
		HFConfig:  models.HFConfig{MaxPositionEmbeddings: 8192},
		VLLMConfig: models.VLLMConfig{
			Dtype:                  "auto",
			MaxModelLen:            4096,
			LoadFormat:             "auto",
			KVCacheDtype:           "fp8",
			AttentionBackend:       "ROCM_AITER_UNIFIED_ATTN",
			ReasoningParser:        "qwen3",
			MambaCacheMode:         "align",
			SpeculativeConfig:      `{"method":"mtp","num_speculative_tokens":8}`,
			CompilationConfig:      `{"cudagraph_capture_sizes":[1,2,4]}`,
			KVCacheMemory:          15989735424,
			DisableAsyncScheduling: true,
			LanguageModelOnly:      true,
			ChatTemplate:           "/work/qwen3.8-enhanced.jinja",
		},
	}
	if err := s.registry.Register(m); err != nil {
		t.Fatal(err)
	}
	return s, m
}

// The config panel is built with hand-written Printf'd HTML, where a mismatched
// verb count silently renders "%!d(MISSING)" into the page instead of failing.
func TestModelConfigPanelRenders(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{
		Variant: vllmenv.VariantRadiance, Launcher: []string{"/opt/radiance_entrypoint.sh"},
	})

	w := httptest.NewRecorder()
	s.handleModelConfigPanel(w, httptest.NewRequest("GET", "/x?id="+url.QueryEscape(m.ID), nil))

	if w.Code != 200 {
		t.Fatalf("status = %d", w.Code)
	}
	body := w.Body.String()

	if strings.Contains(body, "%!") {
		for _, line := range strings.Split(body, "\n") {
			if strings.Contains(line, "%!") {
				t.Errorf("format verb mismatch: %s", strings.TrimSpace(line))
			}
		}
	}

	// Every new field must actually reach the form, otherwise saving the
	// config silently clears it.
	for _, want := range []string{
		`name="attention_backend"`, `name="reasoning_parser"`,
		`name="mamba_cache_mode"`, `name="speculative_config"`,
		`name="compilation_config"`, `name="kv_cache_memory"`,
		`name="disable_async_scheduling"`, `name="language_model_only"`,
		`name="chat_template"`, `name="tokenizer"`, `name="extra_flags"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("form is missing %s", want)
		}
	}

	// Stored values must round-trip into the rendered inputs.
	for _, want := range []string{
		"15989735424",
		"/work/qwen3.8-enhanced.jinja",
		`value="ROCM_AITER_UNIFIED_ATTN" selected`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("form did not render stored value %q", want)
		}
	}

	// The effective-command box must show the launcher that will really run.
	if !strings.Contains(body, "/opt/radiance_entrypoint.sh") {
		t.Error("effective command does not reflect the radiance launcher")
	}
	if !strings.Contains(body, "--speculative-config=") {
		t.Error("effective command is missing the speculative config")
	}
}

// A round trip through the update handler must not drop fields. This regressed
// before: the handler rebuilds VLLMConfig from scratch, so any field missing
// from the form is silently zeroed on every save.
func TestModelConfigUpdateRoundTrip(t *testing.T) {
	s, m := configTestServer(t, vllmenv.Env{Variant: vllmenv.VariantGeneric})

	form := url.Values{
		"dtype":                    {"auto"},
		"max_model_len":            {"4096"},
		"load_format":              {"auto"},
		"kv_cache_dtype":           {"fp8"},
		"attention_backend":        {"R4D"},
		"reasoning_parser":         {"qwen3"},
		"mamba_cache_mode":         {"align"},
		"speculative_config":       {`{"method":"mtp","num_speculative_tokens":8}`},
		"compilation_config":       {`{"cudagraph_capture_sizes":[1,2]}`},
		"kv_cache_memory":          {"15989735424"},
		"disable_async_scheduling": {"on"},
		"language_model_only":      {"on"},
		"chat_template":            {"/work/t.jinja"},
		"tokenizer":                {"/models/tok"},
		"extra_flags":              {"--swap-space 4"},
	}
	req := httptest.NewRequest("POST", "/x?id="+url.QueryEscape(m.ID), strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	s.handleUpdateModelConfig(httptest.NewRecorder(), req)

	got, ok := s.registry.Get(m.ID)
	if !ok {
		t.Fatal("model vanished from the registry")
	}
	c := got.VLLMConfig

	for _, tc := range []struct{ name, got, want string }{
		{"AttentionBackend", c.AttentionBackend, "R4D"},
		{"ReasoningParser", c.ReasoningParser, "qwen3"},
		{"MambaCacheMode", c.MambaCacheMode, "align"},
		{"SpeculativeConfig", c.SpeculativeConfig, `{"method":"mtp","num_speculative_tokens":8}`},
		{"CompilationConfig", c.CompilationConfig, `{"cudagraph_capture_sizes":[1,2]}`},
		{"ChatTemplate", c.ChatTemplate, "/work/t.jinja"},
		{"Tokenizer", c.Tokenizer, "/models/tok"},
		{"ExtraFlags", c.ExtraFlags, "--swap-space 4"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %q, want %q", tc.name, tc.got, tc.want)
		}
	}
	// 15,989,735,424 overflows int32; make sure it survives the form parse.
	if c.KVCacheMemory != 15989735424 {
		t.Errorf("KVCacheMemory = %d, want 15989735424", c.KVCacheMemory)
	}
	if !c.DisableAsyncScheduling || !c.LanguageModelOnly {
		t.Errorf("boolean flags lost: async=%v lmonly=%v",
			c.DisableAsyncScheduling, c.LanguageModelOnly)
	}
}

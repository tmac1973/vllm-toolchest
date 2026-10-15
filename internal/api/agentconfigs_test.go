package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// The base URL comes from the dialog's field when given, else the external
// URL, and always ends in /v1.
func TestAgentBaseURL(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.initTemplates()
	s.cfg.ExternalURL = "http://gpu-box:3000"
	for q, want := range map[string]string{
		"":                                   "http://gpu-box:3000/v1",
		"base=http://10.0.0.5:3000":          "http://10.0.0.5:3000/v1",
		"base=http://10.0.0.5:3000/v1/":      "http://10.0.0.5:3000/v1",
		"base=https://llm.example.com/v1":    "https://llm.example.com/v1",
		"base=https://llm.example.com/x?y=1": "https://llm.example.com/x/v1",
	} {
		got, err := s.agentBaseURL(httptest.NewRequest("GET", "/api/agent-configs?"+q, nil))
		if err != nil || got != want {
			t.Errorf("%q: got %q, %v; want %q", q, got, err, want)
		}
	}
	// No external URL: the address the page was reached on.
	s.cfg.ExternalURL = ""
	if got, _ := s.agentBaseURL(httptest.NewRequest("GET", "http://box.lan:3000/api/agent-configs", nil)); got != "http://box.lan:3000/v1" {
		t.Errorf("no external URL: got %q", got)
	}
	for _, q := range []string{"base=ftp://h/", "base=not a url", "base=http://"} {
		if _, err := s.agentBaseURL(httptest.NewRequest("GET", "/api/agent-configs?"+strings.ReplaceAll(q, " ", "%20"), nil)); err == nil {
			t.Errorf("%q accepted", q)
		}
	}
}

func TestIsLocalOnly(t *testing.T) {
	for base, want := range map[string]bool{
		"http://localhost:3000/v1":    true,
		"http://127.0.0.1:3000/v1":    true,
		"http://127.1.2.3:3000/v1":    true,
		"http://[::1]:3000/v1":        true,
		"http://192.168.1.50:3000/v1": false,
		"https://llm.example.com/v1":  false,
	} {
		if got := isLocalOnly(base); got != want {
			t.Errorf("isLocalOnly(%q) = %v", base, got)
		}
	}
}

func agentConfigsDialog(s *Server) string {
	req := httptest.NewRequest("GET", "/api/agent-configs", nil)
	req.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.handleAgentConfigs(w, req)
	return w.Body.String()
}

func downloadAgentConfig(s *Server, agent string) *httptest.ResponseRecorder {
	r := chi.NewRouter()
	r.Get("/api/agent-configs/{agent}", s.handleAgentConfigDownload)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("GET", "/api/agent-configs/"+agent+"?base=http://gpu-box:3000", nil))
	return w
}

// With nothing running there is nothing to list: the dialog says so and
// offers no downloads, and a download asks for a model to be started.
func TestAgentConfigsWithNothingServed(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.initTemplates()
	s.process = process.NewManager("127.0.0.1", 0, 0)
	out := agentConfigsDialog(s)
	if !strings.Contains(out, "isn&#39;t serving a model") || strings.Contains(out, "Download ") {
		t.Errorf("dialog with nothing served:\n%s", out)
	}
	w := downloadAgentConfig(s, "opencode")
	if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "isn't serving a model") {
		t.Errorf("download with nothing served: %d %s", w.Code, w.Body.String())
	}
	if w = downloadAgentConfig(s, "no-such-agent"); w.Code != http.StatusNotFound {
		t.Errorf("unknown agent: %d", w.Code)
	}
}

// startFake runs m under a fake engine that reports itself up and stays up.
func startFake(t *testing.T, s *Server, m *models.Model, args []string) {
	t.Helper()
	mgr := process.NewManager("127.0.0.1", 0, 0)
	s.process = mgr
	if err := s.registry.Register(m); err != nil {
		t.Fatal(err)
	}
	fake := testutil.WriteScript(t, "echo 'INFO:     Application startup complete.'\nsleep 60\n")
	mgr.SetLauncher(process.Launcher{Bin: fake})
	if err := mgr.Start(m.ID, "", args, nil); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mgr.Stop() })
	testutil.Eventually(t, 5*time.Second, func() bool { return mgr.GetStatus().State == process.StateRunning },
		"the fake engine never came up")
}

// The files name the running model with what it was launched with: the
// served context, not the saved config's, and tools, images and reasoning
// as discovery reports them.
func TestAgentConfigsDescribeTheRunningModel(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.initTemplates()
	m := thinkingCapModel(t)
	startFake(t, s, m, thinkingCapArgs)

	out := agentConfigsDialog(s)
	for _, want := range []string{m.ID, "131072 context, tools, images, reasoning", "Download vllm-toolchest.opencode.json"} {
		if !strings.Contains(out, want) {
			t.Errorf("dialog lacks %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "Tool calls are not being parsed") || strings.Contains(out, "requires an API key") {
		t.Errorf("dialog warns about what is fine:\n%s", out)
	}

	w := downloadAgentConfig(s, "opencode")
	if w.Code != http.StatusOK || !strings.Contains(w.Header().Get("Content-Disposition"), "vllm-toolchest.opencode.json") {
		t.Fatalf("download: %d %v", w.Code, w.Header())
	}
	var cfg struct {
		Model    string `json:"model"`
		Provider map[string]struct {
			Options map[string]string `json:"options"`
			Models  map[string]struct {
				ToolCall bool `json:"tool_call"`
				Limit    struct{ Context, Output int }
			} `json:"models"`
		} `json:"provider"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &cfg); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, w.Body.String())
	}
	p := cfg.Provider["vllm-toolchest"]
	got := p.Models[m.ID]
	if cfg.Model != "vllm-toolchest/"+m.ID || p.Options["baseURL"] != "http://gpu-box:3000/v1" ||
		p.Options["apiKey"] != "none" || !got.ToolCall || got.Limit.Context != 131072 || got.Limit.Output != 32768 {
		t.Errorf("opencode config:\n%s", w.Body.String())
	}
}

// Without --enable-auto-tool-choice the engine returns tool calls as text,
// which no coding agent can work with: the dialog says so. With a key set,
// the files name the variable and never the key.
func TestAgentConfigsWarnWithoutTools(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.initTemplates()
	s.cfg.APIKey = "sekrit"
	m := thinkingCapModel(t)
	startFake(t, s, m, []string{"--max-model-len", "32768", "--language-model-only"})

	out := agentConfigsDialog(s)
	for _, want := range []string{"32768 context, reasoning<", "Tool calls are not being parsed", "export VLLM_TOOLCHEST_API_KEY="} {
		if !strings.Contains(out, want) {
			t.Errorf("dialog lacks %q:\n%s", want, out)
		}
	}
	body := downloadAgentConfig(s, "crush").Body.String()
	if strings.Contains(body, "sekrit") || !strings.Contains(body, "${VLLM_TOOLCHEST_API_KEY}") {
		t.Errorf("crushrc with a key set:\n%s", body)
	}
}

// While autoconfigure has its helper running, the operator's model is
// stopped and will come back: a config naming the helper would be useless.
func TestAgentConfigsSkipTheHelper(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.initTemplates()
	helper := &models.Model{ID: models.HelperRepo, LocalPath: t.TempDir(), Enabled: true, Helper: true}
	startFake(t, s, helper, []string{"--max-model-len", "16384"})

	if out := agentConfigsDialog(s); !strings.Contains(out, "autoconfigure") || strings.Contains(out, "Download ") {
		t.Errorf("dialog with the helper running:\n%s", out)
	}
	if w := downloadAgentConfig(s, "pi"); w.Code != http.StatusConflict {
		t.Errorf("download with the helper running: %d", w.Code)
	}
}

// The key line names the variable but never the key: Settings keeps the key
// write-only, and the dialog must not be the place it leaks.
func TestAgentConfigsKeyExport(t *testing.T) {
	d := agentConfigsData{KeyEnv: "VLLM_TOOLCHEST_API_KEY"}
	if got := d.KeyExport(); got != "export VLLM_TOOLCHEST_API_KEY='your API key'" {
		t.Errorf("KeyExport = %q", got)
	}
}

// A model still starting is not answering yet, and the files would name a
// context its first start may still refine.
func TestAgentConfigsWaitForTheStart(t *testing.T) {
	s := newTestServer(t, "http://127.0.0.1:1")
	s.initTemplates()
	mgr := process.NewManager("127.0.0.1", 0, 0)
	s.process = mgr
	m := thinkingCapModel(t)
	if err := s.registry.Register(m); err != nil {
		t.Fatal(err)
	}
	mgr.SetLauncher(process.Launcher{Bin: testutil.WriteScript(t, "sleep 60\n")})
	if err := mgr.Start(m.ID, "", thinkingCapArgs, nil); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mgr.Stop() })

	if out := agentConfigsDialog(s); !strings.Contains(out, "still starting") || strings.Contains(out, "Download ") {
		t.Errorf("dialog while starting:\n%s", out)
	}
}

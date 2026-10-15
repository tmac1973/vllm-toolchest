package api

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/agentconfig"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// agentConfigsData feeds the agent_configs partial, the Server tab's
// "Agent Configs" dialog.
type agentConfigsData struct {
	// BaseURL is the endpoint the files will point at, as shown in the
	// dialog's URL field.
	BaseURL string
	// LocalOnly is true when BaseURL names this machine only, so an agent
	// elsewhere could not reach it.
	LocalOnly bool
	// APIKey is true when the server requires a key.
	APIKey bool
	KeyEnv string
	// Models are what the files will list: the one model being served, or
	// none, and then Unavailable says why and the dialog offers nothing.
	Models      []agentconfig.Model
	Unavailable string
	Agents      []agentConfigRow
}

// agentConfigRow is one agent in the dialog.
type agentConfigRow struct {
	agentconfig.Agent
	// DownloadURL carries the base URL, so the file matches the field.
	DownloadURL string
	// Steps lists where to save the file per OS, in a fixed order.
	Steps []agentConfigStep
}

// KeyExport is the line that sets the key's variable, with a placeholder:
// the key is write-only on the Settings page, and this dialog does not
// reveal it either.
func (d agentConfigsData) KeyExport() string {
	return "export " + d.KeyEnv + "='your API key'"
}

type agentConfigStep struct {
	OS   agentconfig.OS
	Path string
}

// agentInput assembles the generators' input from the model being served.
//
// vLLM serves one model per process and loads no other on demand, so that
// is the only model the files can usefully name: a request for any other
// fails. Its facts come from buildCapabilities, the same launch flags client
// auto-discovery reports, so the files and discovery cannot disagree. When
// there is nothing to list, the second result says why.
func (s *Server) agentInput(base string) (agentconfig.Input, string) {
	in := agentconfig.Input{
		BaseURL:     base,
		APIKey:      s.cfg.APIKey != "",
		GeneratedAt: time.Now(),
	}
	st := s.process.GetStatus()
	m, known := s.registry.Get(st.ModelID)
	switch {
	case st.State == process.StateStarting:
		return in, "The server is still starting. Open this again once it is running."
	case st.State != process.StateRunning || !known:
		return in, "The server isn't serving a model. Start one on this tab, then open this again."
	case m.Helper || m.ID == models.HelperRepo:
		// Autoconfigure has stopped the operator's model to run its own,
		// and will put it back; a config naming the helper would be useless.
		return in, "The server is running autoconfigure's helper model. Open this again once your model is back."
	}

	caps := buildCapabilities(m, st.Args)
	ctx, _ := caps["context_per_request"].(int)
	reasoning, _ := caps["reasoning"].(models.ReasoningCapability)
	tools, _ := caps["tools"].(bool)
	vision, _ := caps["vision"].(bool)
	in.Models = []agentconfig.Model{{
		ID:        m.ID,
		Context:   ctx,
		MaxOutput: agentconfig.MaxOutputFor(ctx),
		Tools:     tools,
		Vision:    vision,
		Reasoning: reasoning.Supported,
	}}
	return in, ""
}

// agentBaseURL is the endpoint the files point at: the dialog's URL field
// when given, else the configured external URL, else the address the
// browser reached this page on. A missing /v1 is added, so pasting the
// server's address works.
func (s *Server) agentBaseURL(r *http.Request) (string, error) {
	raw := strings.TrimSpace(r.URL.Query().Get("base"))
	if raw == "" {
		raw = s.cfg.ExternalURL
	}
	if raw == "" && r.Host != "" {
		scheme := "http"
		if r.TLS != nil {
			scheme = "https"
		}
		raw = scheme + "://" + r.Host
	}
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("not an http(s) URL: %q", raw)
	}
	u.RawQuery, u.Fragment = "", ""
	u.Path = strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(u.Path, "/v1") {
		u.Path += "/v1"
	}
	return u.String(), nil
}

// isLocalOnly reports a base URL only this machine can reach.
func isLocalOnly(base string) bool {
	u, err := url.Parse(base)
	if err != nil {
		return false
	}
	switch h := u.Hostname(); h {
	case "localhost", "127.0.0.1", "::1", "0.0.0.0":
		return true
	default:
		return strings.HasPrefix(h, "127.")
	}
}

func (s *Server) handleAgentConfigs(w http.ResponseWriter, r *http.Request) {
	base, err := s.agentBaseURL(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "agent_configs", newAgentConfigsData(s.agentInput(base)))
}

// newAgentConfigsData is the dialog for in, or for why there is nothing to
// offer when why is set.
func newAgentConfigsData(in agentconfig.Input, why string) agentConfigsData {
	data := agentConfigsData{
		BaseURL:     in.BaseURL,
		LocalOnly:   isLocalOnly(in.BaseURL),
		APIKey:      in.APIKey,
		KeyEnv:      agentconfig.KeyEnv,
		Models:      in.Models,
		Unavailable: why,
	}
	for _, a := range agentconfig.Agents {
		row := agentConfigRow{
			Agent:       a,
			DownloadURL: "/api/agent-configs/" + a.ID + "?base=" + url.QueryEscape(in.BaseURL),
		}
		for _, os := range []agentconfig.OS{agentconfig.Linux, agentconfig.MacOS, agentconfig.Windows} {
			if p := a.SavePath[os]; p != "" {
				row.Steps = append(row.Steps, agentConfigStep{OS: os, Path: p})
			}
		}
		data.Agents = append(data.Agents, row)
	}
	return data
}

func (s *Server) handleAgentConfigDownload(w http.ResponseWriter, r *http.Request) {
	a, ok := agentconfig.Find(chi.URLParam(r, "agent"))
	if !ok {
		http.Error(w, "unknown agent", http.StatusNotFound)
		return
	}
	base, err := s.agentBaseURL(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	in, why := s.agentInput(base)
	if len(in.Models) == 0 {
		http.Error(w, why, http.StatusConflict)
		return
	}
	body, err := a.Generate(in)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	ctype := "application/json"
	if !strings.HasSuffix(a.FileName, ".json") {
		ctype = "text/plain"
	}
	w.Header().Set("Content-Type", ctype+"; charset=utf-8")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", a.FileName))
	w.Header().Set("Cache-Control", "no-store")
	w.Write(body)
}

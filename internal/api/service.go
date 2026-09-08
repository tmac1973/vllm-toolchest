package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

func (s *Server) handleServiceStatus(w http.ResponseWriter, r *http.Request) {
	// Escapes string arguments: model IDs come from HuggingFace and
	// error text quotes whatever input produced it.
	hp := htmlPrinter(w)
	status := s.process.GetStatus()

	if !isHTMX(r) {
		respondJSON(w, status)
		return
	}

	respondHTML(w)
	var badge string
	switch status.State {
	case process.StateRunning:
		badge = `<ins>Running</ins>`
	case process.StateStarting:
		badge = `<mark>Starting...</mark>`
	case process.StateStopping:
		badge = `<mark>Stopping...</mark>`
	case process.StateError:
		badge = `<del>Error</del>`
	default:
		badge = `Stopped`
	}

	hp(`<div>
  <p>Status: %s</p>`, badge)

	if status.ModelID != "" {
		hp(`<p>Model: <strong>%s</strong></p>`, status.ModelID)
	}
	if status.Uptime != "" {
		hp(`<p>Uptime: %s (PID: %d)</p>`, status.Uptime, status.PID)
	}
	if status.Error != "" {
		hp(`<p><small><del>%s</del></small></p>`, status.Error)
	}
	fmt.Fprint(w, `</div>`)

	// Clear the stale action result message via OOB swap when state settles
	if status.State != process.StateStarting && status.State != process.StateStopping {
		fmt.Fprint(w, `<div id="service-action-result" hx-swap-oob="innerHTML"></div>`)
	}
}

func (s *Server) handleServiceStart(w http.ResponseWriter, r *http.Request) {
	// Escapes string arguments: model IDs come from HuggingFace and
	// error text quotes whatever input produced it.
	hp := htmlPrinter(w)
	var req struct {
		ModelID string `json:"model_id"`
	}

	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "json") {
		json.NewDecoder(r.Body).Decode(&req)
	} else {
		r.ParseForm()
		req.ModelID = r.FormValue("model_id")
	}

	if req.ModelID == "" {
		http.Error(w, "missing model_id", http.StatusBadRequest)
		return
	}

	m, ok := s.registry.Get(req.ModelID)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	modelPath := process.ResolveModelPath(m.LocalPath)

	startCfg := m.VLLMConfig.StartConfig()

	args := process.BuildArgs(startCfg)
	env := process.BuildEnv(m.Quantization.Method, s.cfg.Radiance.Env()...)

	if err := s.process.Start(m.ID, modelPath, args, env); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			hp(`<p><del>Failed to start: %s</del></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		hp(`<p><mark>Starting vLLM with %s...</mark></p>`, m.DisplayName)
		return
	}
	respondJSON(w, map[string]string{"status": "starting"})
}

func (s *Server) handleServiceStop(w http.ResponseWriter, r *http.Request) {
	// Escapes string arguments: model IDs come from HuggingFace and
	// error text quotes whatever input produced it.
	hp := htmlPrinter(w)
	if err := s.process.Stop(); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			hp(`<p><del>%s</del></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		fmt.Fprint(w, `<p>vLLM stopped.</p>`)
		return
	}
	respondJSON(w, map[string]string{"status": "stopped"})
}

func (s *Server) handleServiceRestart(w http.ResponseWriter, r *http.Request) {
	// Escapes string arguments: model IDs come from HuggingFace and
	// error text quotes whatever input produced it.
	hp := htmlPrinter(w)
	status := s.process.GetStatus()
	if status.ModelID == "" {
		http.Error(w, "no model was running", http.StatusBadRequest)
		return
	}

	m, ok := s.registry.Get(status.ModelID)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	modelPath := process.ResolveModelPath(m.LocalPath)
	startCfg := m.VLLMConfig.StartConfig()
	args := process.BuildArgs(startCfg)
	env := process.BuildEnv(m.Quantization.Method, s.cfg.Radiance.Env()...)

	if err := s.process.Restart(m.ID, modelPath, args, env); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			hp(`<p><del>%s</del></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		fmt.Fprint(w, `<p><mark>Restarting vLLM...</mark></p>`)
		return
	}
	respondJSON(w, map[string]string{"status": "restarting"})
}

func (s *Server) handleClearServiceLogs(w http.ResponseWriter, r *http.Request) {
	s.process.ClearLogs()
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleServiceLogs(w http.ResponseWriter, r *http.Request) {
	lines := s.process.RecentLogs(5000)

	if !isHTMX(r) {
		respondJSON(w, lines)
		return
	}

	respondHTML(w)
	if len(lines) == 0 {
		fmt.Fprint(w, "No logs yet.")
		return
	}
	for _, line := range lines {
		fmt.Fprintf(w, "%s\n", line)
	}
}

func (s *Server) handleServiceLogStream(w http.ResponseWriter, r *http.Request) {
	ch := s.process.SubscribeLogs()
	defer s.process.UnsubscribeLogs(ch)

	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	for {
		select {
		case line, ok := <-ch:
			if !ok {
				return
			}
			sse.SendLine(line)
		case <-r.Context().Done():
			return
		}
	}
}

func (s *Server) handleServiceHealth(w http.ResponseWriter, r *http.Request) {
	status := s.process.GetStatus()
	respondJSON(w, map[string]interface{}{
		"vllm_state": status.State,
		"model":      status.ModelID,
		"healthy":    status.State == process.StateRunning,
	})
}

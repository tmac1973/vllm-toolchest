package api

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

func (s *Server) handleServiceStatus(w http.ResponseWriter, r *http.Request) {
	status := s.process.GetStatus()

	if !isHTMX(r) {
		respondJSON(w, status)
		return
	}

	respondHTML(w)
	s.renderPartial(w, "service_status", struct {
		process.Status
		// Settled reports that the process has stopped moving between
		// states, which is when the leftover start/stop message is cleared.
		Settled bool
	}{
		Status:  status,
		Settled: status.State != process.StateStarting && status.State != process.StateStopping,
	})
}

func (s *Server) handleServiceStart(w http.ResponseWriter, r *http.Request) {
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

	// The Server page posts its picker's value; anything else (a script, the
	// auto-start path) gets the model the picker last saved.
	if req.ModelID == "" {
		req.ModelID = s.cfg.ActiveModel
	}
	if req.ModelID == "" {
		http.Error(w, "no model selected — pick one on the Server page", http.StatusBadRequest)
		return
	}

	m, ok := s.registry.Get(req.ModelID)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	if err := s.startModel(m); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "error_message", fmt.Sprintf("Failed to start: %s", err))
			return
		}
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "notice", fmt.Sprintf("Starting vLLM with %s...", m.DisplayName))
		return
	}
	respondJSON(w, map[string]string{"status": "starting"})
}

func (s *Server) handleServiceStop(w http.ResponseWriter, r *http.Request) {
	if err := s.process.Stop(); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "error_message", err.Error())
			return
		}
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "plain_message", "vLLM stopped.")
		return
	}
	respondJSON(w, map[string]string{"status": "stopped"})
}

func (s *Server) handleServiceRestart(w http.ResponseWriter, r *http.Request) {
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
	env := s.launchEnv(m.Quantization.Method)

	if err := s.process.Restart(m.ID, modelPath, args, env); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "error_message", err.Error())
			return
		}
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "notice", "Restarting vLLM...")
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

// startModel launches one registry model. Shared by the Server page's Start
// button and the auto-start path so the two cannot drift into launching the
// same model two different ways.
func (s *Server) startModel(m *models.Model) error {
	return s.process.Start(
		m.ID,
		process.ResolveModelPath(m.LocalPath),
		process.BuildArgs(m.VLLMConfig.StartConfig()),
		s.launchEnv(m.Quantization.Method),
	)
}

// AutoStart launches the active model if the setting is on, for the container
// startup path. It reports problems to the log and returns rather than
// retrying: the UI is already listening by the time this runs, so a failure
// here leaves an operator able to look at the log and press Start, which is
// better than a boot loop.
func (s *Server) AutoStart() {
	if !s.cfg.AutoStart {
		return
	}
	id := s.cfg.ActiveModel
	if id == "" {
		slog.Warn("auto-start is on but no model is active; nothing to start")
		return
	}
	m, ok := s.registry.Get(id)
	if !ok {
		slog.Warn("auto-start model is not in the registry", "model", id)
		return
	}
	slog.Info("auto-starting vLLM", "model", m.ID)
	if err := s.startModel(m); err != nil {
		slog.Error("auto-start failed", "model", m.ID, "error", err)
	}
}

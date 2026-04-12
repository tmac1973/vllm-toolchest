package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

func (s *Server) handleServiceStatus(w http.ResponseWriter, r *http.Request) {
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

	fmt.Fprintf(w, `<div>
  <p>Status: %s</p>`, badge)

	if status.ModelID != "" {
		fmt.Fprintf(w, `<p>Model: <strong>%s</strong></p>`, status.ModelID)
	}
	if status.Uptime != "" {
		fmt.Fprintf(w, `<p>Uptime: %s (PID: %d)</p>`, status.Uptime, status.PID)
	}
	if status.Error != "" {
		fmt.Fprintf(w, `<p><small><del>%s</del></small></p>`, status.Error)
	}
	fmt.Fprint(w, `</div>`)
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

	startCfg := process.VLLMStartConfig{
		Dtype:                m.VLLMConfig.Dtype,
		MaxModelLen:          m.VLLMConfig.MaxModelLen,
		TensorParallelSize:   m.VLLMConfig.TensorParallelSize,
		GPUMemoryUtilization: m.VLLMConfig.GPUMemoryUtilization,
		EnforceEager:         m.VLLMConfig.EnforceEager,
		TrustRemoteCode:      m.VLLMConfig.TrustRemoteCode,
		MaxNumSeqs:           m.VLLMConfig.MaxNumSeqs,
		Quantization:         m.VLLMConfig.Quantization,
		LoadFormat:           m.VLLMConfig.LoadFormat,
		EnablePrefixCaching:  m.VLLMConfig.EnablePrefixCaching,
		KVCacheDtype:         m.VLLMConfig.KVCacheDtype,
		EnableChunkedPrefill: m.VLLMConfig.EnableChunkedPrefill,
		MaxNumBatchedTokens:  m.VLLMConfig.MaxNumBatchedTokens,
		EnableAutoToolChoice: m.VLLMConfig.EnableAutoToolChoice,
		ToolCallParser:       m.VLLMConfig.ToolCallParser,
		Tokenizer:            m.VLLMConfig.Tokenizer,
		ChatTemplate:         m.VLLMConfig.ChatTemplate,
		ExtraFlags:           m.VLLMConfig.ExtraFlags,
	}

	args := process.BuildArgs(startCfg)
	env := process.BuildEnv(m.Quantization.Method)

	if err := s.process.Start(m.ID, modelPath, args, env); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><del>Failed to start: %s</del></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		fmt.Fprintf(w, `<p><mark>Starting vLLM with %s...</mark></p>`, m.DisplayName)
		return
	}
	respondJSON(w, map[string]string{"status": "starting"})
}

func (s *Server) handleServiceStop(w http.ResponseWriter, r *http.Request) {
	if err := s.process.Stop(); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><del>%s</del></p>`, err)
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
	startCfg := process.VLLMStartConfig{
		Dtype:                m.VLLMConfig.Dtype,
		MaxModelLen:          m.VLLMConfig.MaxModelLen,
		TensorParallelSize:   m.VLLMConfig.TensorParallelSize,
		GPUMemoryUtilization: m.VLLMConfig.GPUMemoryUtilization,
		EnforceEager:         m.VLLMConfig.EnforceEager,
		TrustRemoteCode:      m.VLLMConfig.TrustRemoteCode,
		MaxNumSeqs:           m.VLLMConfig.MaxNumSeqs,
		Quantization:         m.VLLMConfig.Quantization,
		LoadFormat:           m.VLLMConfig.LoadFormat,
		EnablePrefixCaching:  m.VLLMConfig.EnablePrefixCaching,
		KVCacheDtype:         m.VLLMConfig.KVCacheDtype,
		EnableAutoToolChoice: m.VLLMConfig.EnableAutoToolChoice,
		ToolCallParser:       m.VLLMConfig.ToolCallParser,
		ExtraFlags:           m.VLLMConfig.ExtraFlags,
	}
	args := process.BuildArgs(startCfg)
	env := process.BuildEnv(m.Quantization.Method)

	if err := s.process.Restart(m.ID, modelPath, args, env); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><del>%s</del></p>`, err)
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

func (s *Server) handleServiceLogs(w http.ResponseWriter, r *http.Request) {
	lines := s.process.RecentLogs(200)

	if !isHTMX(r) {
		respondJSON(w, lines)
		return
	}

	respondHTML(w)
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

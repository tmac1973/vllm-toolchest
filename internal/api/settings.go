package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// settingsResponse is the public-facing settings (sensitive fields masked).
type settingsResponse struct {
	ListenAddr         string  `json:"listen_addr"`
	DataDir            string  `json:"data_dir"`
	LogLevel           string  `json:"log_level"`
	ExternalURL        string  `json:"external_url"`
	HasAPIKey          bool    `json:"has_api_key"`
	HasHFToken         bool    `json:"has_hf_token"`
	VLLMPort           int     `json:"vllm_port"`
	VLLMHost           string  `json:"vllm_host"`
	GPUMemoryUtil      float64 `json:"gpu_memory_util"`
	MaxModelLen        int     `json:"max_model_len"`
	TensorParallelSize int     `json:"tensor_parallel_size"`
	EnforceEager       bool    `json:"enforce_eager"`
	EnablePrefixCache  bool    `json:"enable_prefix_cache"`
	MaxNumSeqs         int     `json:"max_num_seqs"`
	DefaultDtype       string  `json:"default_dtype"`
	AttentionBackend   string  `json:"attention_backend"`
	ToolUseEnabled     bool    `json:"tool_use_enabled"`
	DefaultToolParser  string  `json:"default_tool_parser"`
	PreferMarlin       bool    `json:"prefer_marlin"`
	DefaultKVCacheDtype string `json:"default_kv_cache_dtype"`
	AutoRestart        bool    `json:"auto_restart"`
	Theme              string  `json:"theme"`
}

func (s *Server) handleGetSettings(w http.ResponseWriter, r *http.Request) {
	c := s.cfg
	resp := settingsResponse{
		ListenAddr:         c.ListenAddr,
		DataDir:            c.DataDir,
		LogLevel:           c.LogLevel,
		ExternalURL:        c.ExternalURL,
		HasAPIKey:          c.APIKey != "",
		HasHFToken:         c.HFToken != "",
		VLLMPort:           c.VLLMPort,
		VLLMHost:           c.VLLMHost,
		GPUMemoryUtil:      c.GPUMemoryUtil,
		MaxModelLen:        c.MaxModelLen,
		TensorParallelSize: c.TensorParallelSize,
		EnforceEager:       c.EnforceEager,
		EnablePrefixCache:  c.EnablePrefixCache,
		MaxNumSeqs:         c.MaxNumSeqs,
		DefaultDtype:       c.DefaultDtype,
		AttentionBackend:   c.AttentionBackend,
		ToolUseEnabled:     c.ToolUseEnabled,
		DefaultToolParser:  c.DefaultToolParser,
		PreferMarlin:       c.PreferMarlin,
		DefaultKVCacheDtype: c.DefaultKVCacheDtype,
		AutoRestart:        c.AutoRestart,
		Theme:              c.Theme,
	}

	respondJSON(w, resp)
}

func (s *Server) handleUpdateSettings(w http.ResponseWriter, r *http.Request) {
	c := s.cfg
	contentType := r.Header.Get("Content-Type")

	if strings.Contains(contentType, "json") {
		var updates map[string]interface{}
		json.NewDecoder(r.Body).Decode(&updates)
		applyJSONUpdates(c, updates)
	} else {
		r.ParseForm()
		if v := r.FormValue("external_url"); v != "" {
			c.ExternalURL = v
		}
		if v := r.FormValue("api_key"); v != "" {
			c.APIKey = v
		} else if r.Form.Has("api_key") {
			c.APIKey = "" // explicitly cleared
		}
		if v := r.FormValue("hf_token"); v != "" {
			c.HFToken = v
			s.hfClient.SetToken(v)
			s.downloader.SetToken(v)
		} else if r.Form.Has("hf_token") {
			c.HFToken = ""
			s.hfClient.SetToken("")
			s.downloader.SetToken("")
		}
		if v := r.FormValue("gpu_memory_util"); v != "" {
			fmt.Sscanf(v, "%f", &c.GPUMemoryUtil)
		}
		if v := r.FormValue("max_num_seqs"); v != "" {
			fmt.Sscanf(v, "%d", &c.MaxNumSeqs)
		}
		if v := r.FormValue("default_dtype"); v != "" {
			c.DefaultDtype = v
		}
		if v := r.FormValue("attention_backend"); v != "" {
			c.AttentionBackend = v
		}
		c.ToolUseEnabled = r.FormValue("tool_use_enabled") == "on"
		if v := r.FormValue("default_tool_parser"); v != "" {
			c.DefaultToolParser = v
		}
		c.PreferMarlin = r.FormValue("prefer_marlin") == "on"
		if v := r.FormValue("default_kv_cache_dtype"); v != "" {
			c.DefaultKVCacheDtype = v
		}
		c.AutoRestart = r.FormValue("auto_restart") == "on"
		c.EnforceEager = r.FormValue("enforce_eager") == "on"
		c.EnablePrefixCache = r.FormValue("enable_prefix_cache") == "on"
		if v := r.FormValue("theme"); v != "" {
			c.Theme = v
		}
	}

	if err := c.Save(""); err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><del>Failed to save: %s</del></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		fmt.Fprint(w, `<p><ins>Settings saved.</ins></p>`)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleTestConnection(w http.ResponseWriter, r *http.Request) {
	status := s.process.GetStatus()
	health := map[string]interface{}{
		"vllm_running": status.State == "running",
		"vllm_state":   status.State,
	}

	if status.State == "running" {
		resp, err := http.Get(fmt.Sprintf("http://%s:%d/health", s.cfg.VLLMHost, s.cfg.VLLMPort))
		if err != nil {
			health["vllm_health"] = "unreachable"
			health["error"] = err.Error()
		} else {
			resp.Body.Close()
			health["vllm_health"] = "ok"
			health["vllm_status_code"] = resp.StatusCode
		}
	}

	if isHTMX(r) {
		respondHTML(w)
		if status.State == "running" {
			fmt.Fprint(w, `<p><ins>vLLM is running and healthy.</ins></p>`)
		} else {
			fmt.Fprintf(w, `<p>vLLM is %s.</p>`, status.State)
		}
		return
	}
	respondJSON(w, health)
}

func applyJSONUpdates(c interface{}, updates map[string]interface{}) {
	// Re-marshal updates and unmarshal onto config for simple field updates
	data, err := json.Marshal(updates)
	if err != nil {
		return
	}
	json.Unmarshal(data, c)
}

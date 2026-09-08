package api

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/backup"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// handleBackupExport serves the configuration backup as a JSON download.
//
// This is a same-origin UI route like the rest of /api — the API key guards
// /v1 only — so a secrets-bearing export has the same exposure as the settings
// API that can already read them back. Secrets are included only on the
// explicit ?secrets=1 opt-in.
func (s *Server) handleBackupExport(w http.ResponseWriter, r *http.Request) {
	includeSecrets := r.URL.Query().Get("secrets") == "1"

	var gpus []string
	for _, g := range s.monitor.Current().GPU {
		gpus = append(gpus, g.Name)
	}

	f := backup.Assemble(s.cfg, s.registry, s.vllmEnv.Variant, gpus, includeSecrets)
	data, err := f.Marshal()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=vllm-toolchest-backup-%s.json", time.Now().Format("2006-01-02")))
	w.Write(data)
}

// restoreFileLimit bounds the uploaded backup. Real backups are kilobytes, so
// 10 MB is generous.
const restoreFileLimit = 10 << 20

// handleRestore applies an uploaded backup with merge semantics and an
// itemized report.
//
// Response contract — failures must be visible, and htmx does not swap a
// non-2xx response: htmx callers always get 200 with the report partial, and a
// refusal is rendered as a report carrying only Error. Everything else gets
// real status codes with a JSON body.
func (s *Server) handleRestore(w http.ResponseWriter, r *http.Request) {
	fail := func(status int, msg string) {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "restore_report", backup.Report{Error: msg})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprintf(w, `{"error": %q}`, msg)
	}

	// A restore mid-benchmark would rewrite the configs a running cell is
	// reporting as fixed, which silently invalidates its numbers.
	if _, busy := s.benchSvc.ActiveJobID(); busy {
		fail(http.StatusConflict, "a benchmark job is running — cancel it before restoring")
		return
	}
	if _, busy := s.benchSvc.ActiveRunID(); busy {
		fail(http.StatusConflict, "a benchmark run is in progress — cancel it before restoring")
		return
	}

	if err := r.ParseMultipartForm(restoreFileLimit); err != nil {
		fail(http.StatusBadRequest, "invalid upload: "+err.Error())
		return
	}
	sel := backup.Selections{
		Settings:     r.FormValue("sec_settings") == "on",
		RuntimeEnv:   r.FormValue("sec_env") == "on",
		Radiance:     r.FormValue("sec_radiance") == "on",
		ModelConfigs: r.FormValue("sec_models") == "on",
	}
	if sel.None() {
		fail(http.StatusBadRequest, "select at least one section to restore")
		return
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		fail(http.StatusBadRequest, "no backup file in upload")
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, restoreFileLimit))
	if err != nil {
		fail(http.StatusBadRequest, "reading upload: "+err.Error())
		return
	}

	f, err := backup.Parse(data)
	if err != nil {
		fail(http.StatusBadRequest, err.Error())
		return
	}

	report := backup.Apply(f, sel, s.restoreDeps())

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "restore_report", report)
		return
	}
	respondJSON(w, report)
}

// restoreDeps wires the engine's collaborators to the live server. Every
// mutation goes through a setter that already exists for the UI's own use, so
// a restore cannot write state by a route nothing else uses.
func (s *Server) restoreDeps() backup.Deps {
	return backup.Deps{
		ApplySettings: func(in backup.Settings) ([]string, error) {
			var changed []string
			// set records a field only when the backup carries it and the
			// value actually differs, so the report says what a restore did
			// rather than what it looked at.
			setStr := func(name string, in *string, dst *string, allowEmpty bool) {
				if in == nil || (*in == "" && !allowEmpty) || *dst == *in {
					return
				}
				*dst = *in
				changed = append(changed, name)
			}
			setBool := func(name string, in *bool, dst *bool) {
				if in == nil || *dst == *in {
					return
				}
				*dst = *in
				changed = append(changed, name)
			}
			setInt := func(name string, in *int, dst *int) {
				if in == nil || *dst == *in {
					return
				}
				*dst = *in
				changed = append(changed, name)
			}

			c := s.cfg
			setStr("log_level", in.LogLevel, &c.LogLevel, false)
			if in.GPUMemoryUtil != nil && c.GPUMemoryUtil != *in.GPUMemoryUtil {
				c.GPUMemoryUtil = *in.GPUMemoryUtil
				changed = append(changed, "gpu_memory_util")
			}
			setInt("max_model_len", in.MaxModelLen, &c.MaxModelLen)
			setInt("tensor_parallel_size", in.TensorParallelSize, &c.TensorParallelSize)
			setInt("max_num_seqs", in.MaxNumSeqs, &c.MaxNumSeqs)
			setStr("default_dtype", in.DefaultDtype, &c.DefaultDtype, false)
			// "" is a real value for the attention backend — it means "let
			// vLLM choose" — so an empty one is applied rather than skipped.
			setStr("attention_backend", in.AttentionBackend, &c.AttentionBackend, true)
			setBool("enforce_eager", in.EnforceEager, &c.EnforceEager)
			setBool("enable_prefix_cache", in.EnablePrefixCache, &c.EnablePrefixCache)
			setBool("tool_use_enabled", in.ToolUseEnabled, &c.ToolUseEnabled)
			setStr("default_tool_parser", in.DefaultToolParser, &c.DefaultToolParser, true)
			setBool("prefer_marlin", in.PreferMarlin, &c.PreferMarlin)
			setStr("default_kv_cache_dtype", in.DefaultKVCacheDtype, &c.DefaultKVCacheDtype, false)
			setBool("auto_restart", in.AutoRestart, &c.AutoRestart)
			setBool("auto_start", in.AutoStart, &c.AutoStart)
			setStr("theme", in.Theme, &c.Theme, false)

			if in.HFToken != nil && c.HFToken != *in.HFToken {
				c.HFToken = *in.HFToken
				s.hfClient.SetToken(c.HFToken)
				s.downloader.SetToken(c.HFToken)
				changed = append(changed, "hf_token")
			}
			setStr("api_key", in.APIKey, &c.APIKey, false)

			if len(changed) == 0 {
				return nil, nil
			}
			return changed, c.Save("")
		},
		CurrentEnv: func() backup.RuntimeEnv {
			cur := make(map[string]string, len(s.cfg.RuntimeEnv))
			for k, v := range s.cfg.RuntimeEnv {
				cur[k] = v
			}
			return backup.RuntimeEnv{Curated: cur, Extra: s.cfg.RuntimeEnvExtra}
		},
		ApplyEnv: func(merged backup.RuntimeEnv) error {
			s.cfg.RuntimeEnv = merged.Curated
			s.cfg.RuntimeEnvExtra = merged.Extra
			return s.cfg.Save("")
		},
		ApplyRadiance: func(rad config.RadianceConfig) error {
			s.cfg.Radiance = rad
			return s.cfg.Save("")
		},
		InstalledModel: func(modelID string) bool {
			_, ok := s.registry.Get(modelID)
			return ok
		},
		ApplyModelConfig: func(modelID string, cfg models.VLLMConfig) error {
			return s.registry.UpdateConfig(modelID, cfg)
		},
		SavePending: func(m backup.MissingModel) error {
			return s.registry.SetPendingConfig(models.PendingConfig{
				ModelID: m.ModelID,
				Config:  m.Config,
				SavedAt: time.Now().UTC(),
			})
		},
		NumGPUs: len(s.monitor.Current().GPU),
	}
}

// handleDiscardPending drops a pending config the operator no longer wants
// waiting for its model.
func (s *Server) handleDiscardPending(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	modelID := r.FormValue("model_id")
	if !s.registry.DiscardPendingConfig(modelID) {
		http.Error(w, fmt.Sprintf("no pending config for %s", modelID), http.StatusNotFound)
		return
	}
	w.Header().Set("HX-Trigger", "modelsChanged")
	w.WriteHeader(http.StatusNoContent)
}

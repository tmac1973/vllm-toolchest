package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// handleBenchmarkForm renders the new-run form partial. The form lists
// all registered models and presets; the model selector is disabled when
// vLLM isn't currently serving (or is serving a different model).
func (s *Server) handleBenchmarkForm(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	status := s.process.GetStatus()
	loadedID := ""
	if status.State == process.StateRunning {
		loadedID = status.ModelID
	}

	registered := s.registry.List()
	if loadedID == "" {
		fmt.Fprint(w, `<article><em>Start a model on the <a href="/service">Service</a> page before running benchmarks.</em></article>`)
		return
	}

	// Find the loaded model's display name (the only one eligible).
	loadedName := loadedID
	for _, m := range registered {
		if m.ID == loadedID {
			loadedName = displayNameOf(m)
			break
		}
	}

	fmt.Fprintf(w, `<article>
  <header><strong>New benchmark</strong> &middot; <small>running against <code>%s</code></small></header>
  <form hx-post="/api/benchmarks/" hx-target="#bench-start-result" hx-swap="innerHTML">
    <input type="hidden" name="model_id" value="%s">
    <label>Preset
      <select name="preset" required>`,
		htmlEscape(loadedName), htmlEscape(loadedID))
	for _, p := range benchmark.Presets() {
		fmt.Fprintf(w, `<option value="%s">%s</option>`, htmlEscape(p.Name), htmlEscape(p.Label))
	}
	fmt.Fprint(w, `      </select>
    </label>
    <button type="submit">Start run</button>
  </form>
  <div id="bench-start-result"></div>
</article>`)
}

// handleListBenchmarks returns all benchmark runs, optionally filtered by
// model_id, preset, or job_id. Dual-mode (HTML/JSON) per HX-Request.
func (s *Server) handleListBenchmarks(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	modelFilter := q.Get("model_id")
	presetFilter := q.Get("preset")
	jobFilter := q.Get("job_id")

	all := s.bench.List()
	runs := make([]benchmark.BenchmarkRun, 0, len(all))
	for _, run := range all {
		if modelFilter != "" && run.ModelID != modelFilter {
			continue
		}
		if presetFilter != "" && run.Preset != presetFilter {
			continue
		}
		if jobFilter != "" && run.JobID != jobFilter {
			continue
		}
		runs = append(runs, run)
	}

	if !isHTMX(r) {
		respondJSON(w, map[string]any{
			"runs":  runs,
			"total": len(runs),
		})
		return
	}

	respondHTML(w)
	renderRunList(w, runs)
}

// handleGetBenchmark returns a single run by ID. Dual-mode.
func (s *Server) handleGetBenchmark(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	run, err := s.bench.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if !isHTMX(r) {
		respondJSON(w, run)
		return
	}
	respondHTML(w)
	renderRunDetail(w, run)
}

// startRunRequest is the POST /api/benchmarks body.
type startRunRequest struct {
	ModelID string                       `json:"model_id"`
	Preset  string                       `json:"preset"`
	// Overrides are accepted but ignored for the ad-hoc path in step 2;
	// the model's saved config wins. Job runs (step 4) will honor them.
	Overrides *benchmark.ConfigOverrides `json:"overrides,omitempty"`
}

// handleStartBenchmark begins a new ad-hoc benchmark run.
//
// Preconditions enforced here:
//   - the requested model must be currently loaded in vLLM
//   - the preset must exist
//   - no other run may be active
func (s *Server) handleStartBenchmark(w http.ResponseWriter, r *http.Request) {
	var req startRunRequest
	if r.Header.Get("Content-Type") == "application/x-www-form-urlencoded" || isHTMX(r) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, "invalid form", http.StatusBadRequest)
			return
		}
		req.ModelID = r.FormValue("model_id")
		req.Preset = r.FormValue("preset")
	} else {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
	}

	if req.ModelID == "" || req.Preset == "" {
		http.Error(w, "model_id and preset are required", http.StatusBadRequest)
		return
	}

	model, ok := s.registry.Get(req.ModelID)
	if !ok {
		http.Error(w, "model not registered: "+req.ModelID, http.StatusNotFound)
		return
	}

	// Preset must exist by exact name. GetPreset falls back to standard
	// silently, which would hide typos here.
	var presetFound bool
	for _, p := range benchmark.Presets() {
		if p.Name == req.Preset {
			presetFound = true
			break
		}
	}
	if !presetFound {
		http.Error(w, "unknown preset: "+req.Preset, http.StatusBadRequest)
		return
	}
	preset := benchmark.GetPreset(req.Preset)

	// Service must be running and serving this model.
	status := s.process.GetStatus()
	if status.State != process.StateRunning {
		http.Error(w, "vLLM is not running; start the service before benchmarking", http.StatusConflict)
		return
	}
	if status.ModelID != "" && status.ModelID != req.ModelID {
		http.Error(w,
			fmt.Sprintf("vLLM is currently serving %q; stop and start it with %q first", status.ModelID, req.ModelID),
			http.StatusConflict)
		return
	}

	// Discover the served model name via /v1/models. vLLM uses the file
	// path as the model identifier unless --served-model-name was set.
	servedName, err := s.discoverServedName(req.ModelID)
	if err != nil {
		http.Error(w, "could not discover vLLM served model name: "+err.Error(), http.StatusBadGateway)
		return
	}

	// Build the run record.
	run := benchmark.BenchmarkRun{
		ID:        newRunID(),
		JobID:     benchmark.AdhocJobID,
		CreatedAt: time.Now(),
		Status:    benchmark.StatusRunning,

		ModelID:   model.ID,
		ModelName: displayNameOf(model),
		Quant:     model.Quantization.Method,
		SizeGB:    float64(model.TotalSizeBytes) / (1024 * 1024 * 1024),

		Config: configSnapshotFromModel(model),

		GPUs: benchmark.GPUSnapshotsFromMetrics(s.monitor.Current()),

		Preset:       preset.Name,
		PromptTokens: preset.PromptTokens,
		GenTokens:    preset.GenTokens,
	}

	if err := s.bench.Save(run); err != nil {
		http.Error(w, "failed to save run: "+err.Error(), http.StatusInternalServerError)
		return
	}

	cfg := benchmark.RunnerConfig{
		Run:         run,
		Preset:      preset,
		VLLMURL:     fmt.Sprintf("http://%s:%d", s.cfg.VLLMHost, s.cfg.VLLMPort),
		ServedName:  servedName,
		MaxModelLen: model.VLLMConfig.MaxModelLen,
		HFRepoID:    model.ID,
		HFToken:     s.cfg.HFToken,
		HFHome:      s.cfg.DataDir + "/cache/huggingface",
	}

	if err := s.benchSvc.StartRun(cfg); err != nil {
		// Roll the run back to a failed state so the UI doesn't show a stale "running".
		run.Status = benchmark.StatusFailed
		run.Error = err.Error()
		_ = s.bench.Save(run)
		if errors.Is(err, benchmark.ErrRunAlreadyActive) {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if !isHTMX(r) {
		w.WriteHeader(http.StatusAccepted)
		respondJSON(w, map[string]any{
			"id":     run.ID,
			"status": run.Status,
		})
		return
	}
	respondHTML(w)
	w.WriteHeader(http.StatusAccepted)
	fmt.Fprintf(w, `<p>Started run <code>%s</code>. <a href="#" hx-get="/api/benchmarks/" hx-target="#bench-runs">Refresh list</a></p>`, htmlEscape(run.ID))
}

// handleCancelBenchmark cancels the in-flight run with the given id.
func (s *Server) handleCancelBenchmark(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if ok := s.benchSvc.CancelRun(id); !ok {
		http.Error(w, "no active run with that id", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

// handleDeleteBenchmark removes a single run from the store. Idempotent.
func (s *Server) handleDeleteBenchmark(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if active, ok := s.benchSvc.ActiveRunID(); ok && active == id {
		http.Error(w, "cannot delete a run while it is active; cancel first", http.StatusConflict)
		return
	}
	if err := s.bench.Delete(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleBenchmarkProgress streams SSE progress events for an in-flight run.
// Sends an immediate "snapshot" event with the latest known update, then
// every subsequent update until the run completes.
func (s *Server) handleBenchmarkProgress(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	sub, last, unsub := s.benchSvc.Subscribe(id)
	if sub == nil {
		// No active run for this id. Surface the final state if it exists so
		// the client can stop polling.
		if run, err := s.bench.Get(id); err == nil {
			respondJSON(w, map[string]any{
				"status": run.Status,
				"detail": run.ProgressDetail,
				"error":  run.Error,
			})
			return
		}
		http.Error(w, "no progress for that id", http.StatusNotFound)
		return
	}
	defer unsub()

	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if last != nil {
		payload, _ := json.Marshal(last)
		sse.SendEvent("progress", string(payload))
	}

	for {
		select {
		case update, ok := <-sub:
			if !ok {
				// Run finished — emit a final event with the run's terminal state.
				if run, err := s.bench.Get(id); err == nil {
					payload, _ := json.Marshal(map[string]any{
						"id":     run.ID,
						"status": run.Status,
						"error":  run.Error,
					})
					sse.SendEvent("complete", string(payload))
				}
				return
			}
			payload, _ := json.Marshal(update)
			sse.SendEvent("progress", string(payload))
		case <-r.Context().Done():
			return
		}
	}
}

// renderRunList lists runs in a simple table for the benchmarks page.
func renderRunList(w http.ResponseWriter, runs []benchmark.BenchmarkRun) {
	if len(runs) == 0 {
		fmt.Fprint(w, `<p style="opacity:0.7;">No benchmark runs yet. Load a model and start a run from the form above.</p>`)
		return
	}

	fmt.Fprint(w, `<table><thead><tr>
    <th>Model</th><th>Preset</th><th>Avg gen TPS</th><th>Avg TTFT</th><th>Status</th><th>When</th><th></th>
  </tr></thead><tbody>`)
	for _, run := range runs {
		var avgGen, avgTTFT string
		if run.Summary != nil {
			avgGen = fmt.Sprintf("%.1f t/s", run.Summary.AvgGenTokPerSec)
			avgTTFT = fmt.Sprintf("%.0f ms", run.Summary.AvgTTFTMs)
		} else {
			avgGen = "—"
			avgTTFT = "—"
		}
		when := run.CreatedAt.Format("Jan 2 15:04")
		actions := fmt.Sprintf(
			`<a href="#" hx-get="/api/benchmarks/%s" hx-target="#bench-detail-%s" hx-swap="innerHTML">details</a>`,
			run.ID, run.ID,
		)
		if run.Status == benchmark.StatusRunning {
			actions = fmt.Sprintf(
				`<a href="#" hx-post="/api/benchmarks/%s/cancel" hx-confirm="Cancel this run?">cancel</a>`,
				run.ID,
			)
		} else {
			actions += fmt.Sprintf(
				` &middot; <a href="#" hx-delete="/api/benchmarks/%s" hx-confirm="Delete this run?" hx-target="#bench-runs" hx-swap="none" hx-on::after-request="if(event.detail.successful) htmx.ajax('GET','/api/benchmarks/','#bench-runs')">delete</a>`,
				run.ID,
			)
		}
		fmt.Fprintf(w, `<tr>
      <td><strong>%s</strong><br><small style="opacity:0.6;">%s</small></td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
      <td><small>%s</small></td>
      <td><small>%s</small></td>
    </tr>
    <tr><td colspan="7"><div id="bench-detail-%s"></div></td></tr>`,
			htmlEscape(run.ModelName), htmlEscape(run.ModelID),
			run.Preset, avgGen, avgTTFT,
			statusBadge(run.Status),
			when,
			actions,
			run.ID,
		)
	}
	fmt.Fprint(w, `</tbody></table>`)
}

// renderRunDetail renders the expanded detail view of a single run.
func renderRunDetail(w http.ResponseWriter, run *benchmark.BenchmarkRun) {
	fmt.Fprintf(w, `<article style="margin:0.5rem 0;">
  <header><strong>%s</strong> &middot; <small>%s</small></header>`,
		htmlEscape(run.Preset), htmlEscape(run.CreatedAt.Format("2006-01-02 15:04:05")))

	if run.Error != "" {
		fmt.Fprintf(w, `<p><del>error:</del> %s</p>`, htmlEscape(run.Error))
	}

	if run.ProgressDetail != "" && run.Status == benchmark.StatusRunning {
		fmt.Fprintf(w, `<p><em>%s</em></p>`, htmlEscape(run.ProgressDetail))
	}

	// Config snapshot
	c := run.Config
	fmt.Fprintf(w, `<small>
  <strong>config</strong>:
  max_model_len=%d, tp=%d, gpu_mem_util=%.2f, dtype=%s, kv_cache=%s, eager=%v, quant=%s
</small>`,
		c.MaxModelLen, c.TensorParallelSize, c.GPUMemoryUtilization,
		c.Dtype, c.KVCacheDtype, c.EnforceEager, c.QuantMethod)

	// Per-test-point table
	if len(run.Results) > 0 {
		fmt.Fprint(w, `<table><thead><tr>
  <th>#</th><th>prompt tok</th><th>gen tok</th><th>TTFT (ms)</th><th>total (ms)</th><th>prompt tok/s</th><th>gen tok/s</th>
</tr></thead><tbody>`)
		for i, res := range run.Results {
			fmt.Fprintf(w, `<tr>
  <td>%d</td><td>%d</td><td>%d</td><td>%.0f</td><td>%.0f</td><td>%.1f</td><td>%.1f</td>
</tr>`,
				i+1, res.PromptTokens, res.GenTokens, res.TTFTMs, res.TotalMs,
				res.PromptTokPerSec, res.GenTokPerSec)
		}
		fmt.Fprint(w, `</tbody></table>`)
	}

	if run.Summary != nil {
		fmt.Fprintf(w, `<p><strong>summary</strong>: avg gen %.1f t/s (min %.1f, max %.1f), avg TTFT %.0f ms, avg prompt %.1f t/s</p>`,
			run.Summary.AvgGenTokPerSec, run.Summary.MinGenTokPerSec, run.Summary.MaxGenTokPerSec,
			run.Summary.AvgTTFTMs, run.Summary.AvgPromptTokPerSec)
	}

	if len(run.Warnings) > 0 {
		fmt.Fprint(w, `<small><strong>warnings</strong>:<ul>`)
		for _, w2 := range run.Warnings {
			fmt.Fprintf(w, `<li>%s</li>`, htmlEscape(w2))
		}
		fmt.Fprint(w, `</ul></small>`)
	}

	fmt.Fprint(w, `</article>`)
}

// configSnapshotFromModel extracts the 7 perf-relevant fields from a
// model's saved vLLM config.
func configSnapshotFromModel(m *models.Model) benchmark.ConfigSnapshot {
	v := m.VLLMConfig
	return benchmark.ConfigSnapshot{
		MaxModelLen:          v.MaxModelLen,
		TensorParallelSize:   v.TensorParallelSize,
		GPUMemoryUtilization: v.GPUMemoryUtilization,
		KVCacheDtype:         v.KVCacheDtype,
		EnforceEager:         v.EnforceEager,
		Dtype:                v.Dtype,
		QuantMethod:          m.Quantization.Method,
	}
}

// displayNameOf returns the model's display name, falling back to its ID.
func displayNameOf(m *models.Model) string {
	if m.DisplayName != "" {
		return m.DisplayName
	}
	return m.ID
}

// discoverServedName queries vLLM's /v1/models to find the identifier the
// server actually responds to. vLLM uses the local filesystem path by
// default; --served-model-name overrides that. We match by suffix on the
// model's HF repo id (e.g. "Hermes-3" matches "/data/models/.../Hermes-3").
func (s *Server) discoverServedName(modelID string) (string, error) {
	url := fmt.Sprintf("http://%s:%d/v1/models", s.cfg.VLLMHost, s.cfg.VLLMPort)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}

	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	if len(body.Data) == 0 {
		return "", fmt.Errorf("vLLM /v1/models returned no models")
	}

	// Prefer an exact match against the requested modelID.
	for _, m := range body.Data {
		if m.ID == modelID {
			return m.ID, nil
		}
	}
	// vLLM typically reports the model path; pick the first entry.
	return body.Data[0].ID, nil
}

// newRunID returns a short random identifier suitable for URLs.
func newRunID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func statusBadge(s string) string {
	switch s {
	case benchmark.StatusRunning:
		return `<mark>running</mark>`
	case benchmark.StatusCompleted:
		return `<ins>completed</ins>`
	case benchmark.StatusFailed:
		return `<del>failed</del>`
	default:
		return s
	}
}

func htmlEscape(s string) string {
	r := []byte{}
	for _, c := range []byte(s) {
		switch c {
		case '<':
			r = append(r, []byte("&lt;")...)
		case '>':
			r = append(r, []byte("&gt;")...)
		case '&':
			r = append(r, []byte("&amp;")...)
		case '"':
			r = append(r, []byte("&quot;")...)
		default:
			r = append(r, c)
		}
	}
	return string(r)
}

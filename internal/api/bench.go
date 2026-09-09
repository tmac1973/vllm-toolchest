package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// handleTimingsList returns running averages for every model that has
// passed the minimum-samples threshold.
func (s *Server) handleTimingsList(w http.ResponseWriter, r *http.Request) {
	avgs := s.bench.RunningAverages()
	if !isHTMX(r) {
		respondJSON(w, map[string]any{"averages": avgs, "total": len(avgs)})
		return
	}

	rows := make([]dashboardTiming, 0, len(avgs))
	for _, a := range avgs {
		rows = append(rows, dashboardTiming{
			ModelID:   a.ModelID,
			AvgGenTPS: a.AvgGenTPS,
			Count:     a.Count,
			LastSeen:  a.LastUpdated.Format("Jan 2 15:04"),
		})
	}

	respondHTML(w)
	s.renderPartial(w, "timings_list", rows)
}

// handleTimingsForModel returns recent timing samples plus the running
// average for one model. The model ID can contain slashes (HF repo ids
// like "owner/repo"), so we capture it via chi's `*` wildcard.
func (s *Server) handleTimingsForModel(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "*")
	samples := s.bench.RecentTimings(modelID, 100)
	avg, _ := s.bench.RunningAverage(modelID)
	respondJSON(w, map[string]any{
		"model_id": modelID,
		"average":  avg,
		"samples":  samples,
	})
}

// handleBenchmarkForm renders the new-run form partial. The form lists
// all registered models and presets; the model selector is disabled when
// vLLM isn't currently serving (or is serving a different model).
func (s *Server) handleBenchmarkForm(w http.ResponseWriter, r *http.Request) {
	// Only the model vLLM currently has loaded can be benchmarked ad-hoc:
	// there is one process, and swapping models is what batch jobs are for.
	loadedID := ""
	if status := s.process.GetStatus(); status.State == process.StateRunning {
		loadedID = status.ModelID
	}
	loadedName := loadedID
	for _, m := range s.registry.List() {
		if m.ID == loadedID {
			loadedName = displayNameOf(m)
			break
		}
	}

	respondHTML(w)
	s.renderPartial(w, "benchmark_form", struct {
		LoadedID   string
		LoadedName string
		Presets    []benchmark.Preset
	}{loadedID, loadedName, benchmark.Presets()})
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
	s.renderRunList(w, runs)
}

// handleBatchDeleteBenchmarks removes the selected runs.
//
// The cells that produced them keep pointing at ids that no longer resolve,
// which is deliberate: the cell records that an attempt happened, and rewriting
// history to hide a deleted result would be worse than a Detail button that
// says the run is gone.
func (s *Server) handleBatchDeleteBenchmarks(w http.ResponseWriter, r *http.Request) {
	var ids []string
	for _, id := range strings.Split(r.URL.Query().Get("ids"), ",") {
		if id = strings.TrimSpace(id); id != "" {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		http.Error(w, "no runs selected", http.StatusBadRequest)
		return
	}

	// A run still being measured would come back on the next save.
	if active, ok := s.benchSvc.ActiveRunID(); ok {
		for _, id := range ids {
			if id == active {
				http.Error(w, "that run is still going; cancel it first", http.StatusConflict)
				return
			}
		}
	}

	deleted, notFound, err := s.bench.BatchDelete(ids)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	respondJSON(w, map[string]int{"deleted": deleted, "not_found": notFound})
}

// handleCompareBenchmarks renders a comparison of the selected runs.
func (s *Server) handleCompareBenchmarks(w http.ResponseWriter, r *http.Request) {
	ids := strings.Split(r.URL.Query().Get("ids"), ",")
	var runs []benchmark.BenchmarkRun
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		run, err := s.bench.Get(id)
		if err != nil {
			http.Error(w, "run not found: "+id, http.StatusNotFound)
			return
		}
		runs = append(runs, *run)
	}
	if len(runs) < 2 {
		http.Error(w, "select at least two runs to compare", http.StatusBadRequest)
		return
	}

	if !isHTMX(r) {
		respondJSON(w, benchmark.BuildCompare(runs))
		return
	}
	respondHTML(w)
	s.renderPartial(w, "benchmark_compare", benchmark.BuildCompare(runs))
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
	s.renderRunDetail(w, run)
}

// startRunRequest is the POST /api/benchmarks body.
type startRunRequest struct {
	ModelID string `json:"model_id"`
	Preset  string `json:"preset"`
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
	s.renderPartial(w, "run_started", run.ID)
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

// runRow is one line of the benchmark run table.
type runRow struct {
	ID        string
	ModelName string
	ModelID   string
	Quant     string
	Preset    string
	PPTPS     string
	TGTPS     string
	TTFT      string
	Status    string
	Running   bool
	When      string
	// SweepText names the point of a sweep this run measured, so a run
	// listed on its own still says what it was measuring.
	SweepText string
	// Search is the lowercased haystack the filter box matches against.
	Search string
}

// runGroup collects one model's runs. The list groups because a history of a
// few hundred runs across a handful of models is unreadable flat, and the
// question being asked is almost always about one model at a time.
type runGroup struct {
	Name string
	Rows []runRow
}

func (s *Server) renderRunList(w http.ResponseWriter, runs []benchmark.BenchmarkRun) {
	byModel := map[string]*runGroup{}
	var order []string
	for _, run := range runs {
		row := runRow{
			ID:        run.ID,
			ModelName: run.ModelName,
			ModelID:   run.ModelID,
			Quant:     run.Quant,
			Preset:    run.Preset,
			PPTPS:     "\u2014",
			TGTPS:     "\u2014",
			TTFT:      "\u2014",
			Status:    run.Status,
			Running:   run.Status == benchmark.StatusRunning,
			When:      run.CreatedAt.Format("Jan 2 15:04"),
			SweepText: sweepValuesText(run.SweepValues),
		}
		if run.Summary != nil {
			row.PPTPS = fmt.Sprintf("%.0f", run.Summary.AvgPromptTokPerSec)
			row.TGTPS = fmt.Sprintf("%.1f", run.Summary.AvgGenTokPerSec)
			row.TTFT = fmt.Sprintf("%.0f ms", run.Summary.AvgTTFTMs)
		}
		row.Search = strings.ToLower(strings.Join(
			[]string{run.ModelName, run.ModelID, run.Quant, run.Preset, row.SweepText}, " "))

		name := run.ModelName
		if name == "" {
			name = run.ModelID
		}
		if name == "" {
			name = "(unknown)"
		}
		g, ok := byModel[name]
		if !ok {
			g = &runGroup{Name: name}
			byModel[name] = g
			order = append(order, name)
		}
		g.Rows = append(g.Rows, row)
	}
	sort.Slice(order, func(i, j int) bool {
		return strings.ToLower(order[i]) < strings.ToLower(order[j])
	})
	groups := make([]runGroup, 0, len(order))
	for _, name := range order {
		groups = append(groups, *byModel[name])
	}
	s.renderPartial(w, "run_list", groups)
}

// runResultRow is one test point within a run.
type runResultRow struct {
	N               int
	PromptTokens    int
	GenTokens       int
	TTFTMs          float64
	TotalMs         float64
	PromptTokPerSec float64
	GenTokPerSec    float64
}

// renderRunDetail renders the expanded detail view of a single run.
func (s *Server) renderRunDetail(w http.ResponseWriter, run *benchmark.BenchmarkRun) {
	c := run.Config
	view := struct {
		Preset     string
		CreatedAt  string
		Error      string
		Progress   string
		ConfigLine string
		Results    []runResultRow
		Summary    string
		Warnings   []string
	}{
		Preset:    run.Preset,
		CreatedAt: run.CreatedAt.Format("2006-01-02 15:04:05"),
		Error:     run.Error,
		ConfigLine: fmt.Sprintf(
			"max_model_len=%d, tp=%d, gpu_mem_util=%.2f, dtype=%s, kv_cache=%s, eager=%v, quant=%s",
			c.MaxModelLen, c.TensorParallelSize, c.GPUMemoryUtilization,
			c.Dtype, c.KVCacheDtype, c.EnforceEager, c.QuantMethod),
		Warnings: run.Warnings,
	}
	// Progress detail is only meaningful while the run is still moving.
	if run.Status == benchmark.StatusRunning {
		view.Progress = run.ProgressDetail
	}
	for i, res := range run.Results {
		view.Results = append(view.Results, runResultRow{
			N: i + 1, PromptTokens: res.PromptTokens, GenTokens: res.GenTokens,
			TTFTMs: res.TTFTMs, TotalMs: res.TotalMs,
			PromptTokPerSec: res.PromptTokPerSec, GenTokPerSec: res.GenTokPerSec,
		})
	}
	if run.Summary != nil {
		view.Summary = fmt.Sprintf(
			"avg gen %.1f t/s (min %.1f, max %.1f), avg TTFT %.0f ms, avg prompt %.1f t/s",
			run.Summary.AvgGenTokPerSec, run.Summary.MinGenTokPerSec, run.Summary.MaxGenTokPerSec,
			run.Summary.AvgTTFTMs, run.Summary.AvgPromptTokPerSec)
	}
	s.renderPartial(w, "run_detail", view)
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

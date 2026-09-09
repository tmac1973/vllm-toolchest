package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// ErrProbeAlreadyActive is returned when a probe is requested while one is
// in flight on the same Server. Only one probe at a time; vLLM is the
// shared resource.
var ErrProbeAlreadyActive = errors.New("a context probe is already in progress")

// probeManager tracks the in-flight probe. Lives on Server (added in
// Server construction).
type probeManager struct {
	mu     sync.Mutex
	store  *probeResultStore
	env    *probeEnv
	active *activeProbe
}

type activeProbe struct {
	id      string
	modelID string
	cancel  context.CancelFunc

	subMu sync.Mutex
	subs  map[chan benchmark.ProbeProgress]struct{}
	last  *benchmark.ProbeProgress
	done  chan struct{}
}

func newProbeManager(s *Server) *probeManager {
	return &probeManager{
		store: newProbeResultStore(),
		env:   newProbeEnv(s),
	}
}

// probeStartRequest is the POST /api/benchmarks/probe-context body.
type probeStartRequest struct {
	ModelID           string    `json:"model_id"`
	TPSize            int       `json:"tp_size"`
	UtilizationLevels []float64 `json:"utilization_levels,omitempty"`
	ConcurrencyLevels []int     `json:"concurrency_levels,omitempty"`
	MinContext        int       `json:"min_context,omitempty"`
	MaxContext        int       `json:"max_context,omitempty"`
}

// handleStartContextProbe begins a probe in the background. Returns 409
// when main vLLM is running (probe needs the GPU to itself) or when
// another probe is already active.
func (s *Server) handleStartContextProbe(w http.ResponseWriter, r *http.Request) {
	var req probeStartRequest
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "json") {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		r.ParseForm()
		req.ModelID = r.FormValue("model_id")
		req.TPSize, _ = strconv.Atoi(r.FormValue("tp_size"))
	}

	if req.ModelID == "" {
		http.Error(w, "model_id is required", http.StatusBadRequest)
		return
	}
	if _, ok := s.registry.Get(req.ModelID); !ok {
		http.Error(w, "model not registered: "+req.ModelID, http.StatusNotFound)
		return
	}

	// Main vLLM must be stopped — the probe spawns vLLM repeatedly and
	// needs the GPU's VRAM to itself.
	if state := s.process.GetStatus().State; state == process.StateRunning || state == process.StateStarting {
		http.Error(w, "stop the main vLLM process before probing", http.StatusConflict)
		return
	}

	s.probe.mu.Lock()
	if s.probe.active != nil {
		s.probe.mu.Unlock()
		http.Error(w, ErrProbeAlreadyActive.Error(), http.StatusConflict)
		return
	}

	probeID := newRunID()
	ctx, cancel := context.WithCancel(context.Background())
	ap := &activeProbe{
		id:      probeID,
		modelID: req.ModelID,
		cancel:  cancel,
		subs:    make(map[chan benchmark.ProbeProgress]struct{}),
		done:    make(chan struct{}),
	}
	s.probe.active = ap
	s.probe.mu.Unlock()

	progress := make(chan benchmark.ProbeProgress, 64)

	// Fan-out goroutine.
	go func() {
		for p := range progress {
			p := p
			ap.subMu.Lock()
			ap.last = &p
			for sub := range ap.subs {
				select {
				case sub <- p:
				default:
				}
			}
			ap.subMu.Unlock()
		}
		ap.subMu.Lock()
		for sub := range ap.subs {
			close(sub)
			delete(ap.subs, sub)
		}
		ap.subMu.Unlock()

		s.probe.mu.Lock()
		s.probe.active = nil
		s.probe.mu.Unlock()
		close(ap.done)
	}()

	// Probe goroutine.
	go func() {
		result, err := benchmark.RunProbe(ctx, s.probe.env, benchmark.ProbeConfig{
			ModelID:           req.ModelID,
			TPSize:            req.TPSize,
			UtilizationLevels: req.UtilizationLevels,
			ConcurrencyLevels: req.ConcurrencyLevels,
			MinContext:        req.MinContext,
			MaxContext:        req.MaxContext,
		}, progress)
		if err != nil && !errors.Is(err, context.Canceled) {
			// Probe failed; emit a final error event before exiting.
			// (Progress channel was already closed by RunProbe's defer.)
		}
		if result != nil {
			s.probe.store.Set(req.ModelID, result)
		}
	}()

	if !isHTMX(r) {
		w.WriteHeader(http.StatusAccepted)
		respondJSON(w, map[string]any{"id": probeID, "model_id": req.ModelID})
		return
	}
	respondHTML(w)
	w.WriteHeader(http.StatusAccepted)
	s.renderPartial(w, "probe_started", struct{ ProbeID, ModelID string }{probeID, req.ModelID})
}

// handleContextProbeProgress streams SSE progress events for an in-flight
// probe. Falls back to the latest stored result when no probe is active.
func (s *Server) handleContextProbeProgress(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	s.probe.mu.Lock()
	ap := s.probe.active
	s.probe.mu.Unlock()

	if ap == nil || ap.id != id {
		// Probe completed or didn't exist. Surface any stored result for context.
		modelID := r.URL.Query().Get("model_id")
		if modelID != "" {
			if result, ok := s.probe.store.Get(modelID); ok {
				respondJSON(w, result)
				return
			}
		}
		http.Error(w, "no active probe with that id", http.StatusNotFound)
		return
	}

	sub := make(chan benchmark.ProbeProgress, 32)
	ap.subMu.Lock()
	ap.subs[sub] = struct{}{}
	last := ap.last
	ap.subMu.Unlock()
	defer func() {
		ap.subMu.Lock()
		if _, ok := ap.subs[sub]; ok {
			delete(ap.subs, sub)
			close(sub)
		}
		ap.subMu.Unlock()
	}()

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
		case p, ok := <-sub:
			if !ok {
				// Probe finished — emit the final result if available.
				if result, found := s.probe.store.Get(ap.modelID); found {
					payload, _ := json.Marshal(result)
					sse.SendEvent("complete", string(payload))
				}
				return
			}
			payload, _ := json.Marshal(p)
			sse.SendEvent("progress", string(payload))
		case <-r.Context().Done():
			return
		}
	}
}

// handleGetProbeResult returns the latest probe result for a model.
func (s *Server) handleGetProbeResult(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "*")
	result, ok := s.probe.store.Get(modelID)
	if !ok {
		http.Error(w, "no probe result for "+modelID, http.StatusNotFound)
		return
	}
	if !isHTMX(r) {
		respondJSON(w, result)
		return
	}
	respondHTML(w)
	s.renderProbeResult(w, modelID, result)
}

// applyProbeRequest is the POST body for applying a probed value.
type applyProbeRequest struct {
	ModelID              string  `json:"model_id"`
	MaxModelLen          int     `json:"max_model_len"`
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization,omitempty"`
	MaxNumSeqs           int     `json:"max_num_seqs,omitempty"`
}

// handleApplyProbe writes a probed (max_model_len, gpu_mem_util, max_num_seqs)
// tuple onto the model's saved config in the registry.
func (s *Server) handleApplyProbe(w http.ResponseWriter, r *http.Request) {
	var req applyProbeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.ModelID == "" || req.MaxModelLen <= 0 {
		http.Error(w, "model_id and max_model_len (>0) are required", http.StatusBadRequest)
		return
	}
	m, ok := s.registry.Get(req.ModelID)
	if !ok {
		http.Error(w, "model not registered: "+req.ModelID, http.StatusNotFound)
		return
	}
	m.VLLMConfig.MaxModelLen = req.MaxModelLen
	if req.GPUMemoryUtilization > 0 {
		m.VLLMConfig.GPUMemoryUtilization = req.GPUMemoryUtilization
	}
	if req.MaxNumSeqs > 0 {
		m.VLLMConfig.MaxNumSeqs = req.MaxNumSeqs
	}
	if err := s.registry.Register(m); err != nil {
		http.Error(w, "save model: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// probeRow is one sweep point: the swept value, the largest context that
// survived at it, and the config write-back the Apply button posts.
type probeRow struct {
	Label      string
	MaxContext int
	// Vals is the hx-vals JSON. Built here rather than in the template so the
	// model ID is JSON-escaped by the JSON encoder, not by HTML escaping —
	// they are not the same thing inside an attribute that holds JSON.
	Vals string
}

func probeApplyVals(v map[string]any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// renderProbeResult lays out the probe result as two tables: the utilization
// sweep and the concurrency sweep, each row offering to apply its finding.
func (s *Server) renderProbeResult(w http.ResponseWriter, modelID string, result *benchmark.ProbeResult) {
	// Both result sets are maps, so they need sorting before display —
	// otherwise the rows come out in a different order on every poll.
	utilKeys := make([]string, 0, len(result.UtilizationResults))
	for k := range result.UtilizationResults {
		utilKeys = append(utilKeys, k)
	}
	sort.Slice(utilKeys, func(i, j int) bool {
		a, _ := strconv.ParseFloat(utilKeys[i], 64)
		b, _ := strconv.ParseFloat(utilKeys[j], 64)
		return a < b
	})

	byUtil := make([]probeRow, 0, len(utilKeys))
	for _, util := range utilKeys {
		maxCtx := result.UtilizationResults[util]
		utilF, _ := strconv.ParseFloat(util, 64)
		byUtil = append(byUtil, probeRow{
			Label:      util,
			MaxContext: maxCtx,
			Vals: probeApplyVals(map[string]any{
				"model_id":               modelID,
				"max_model_len":          maxCtx,
				"gpu_memory_utilization": utilF,
			}),
		})
	}

	concKeys := make([]int, 0, len(result.ConcurrencyResults))
	for k := range result.ConcurrencyResults {
		concKeys = append(concKeys, k)
	}
	sort.Ints(concKeys)

	byConc := make([]probeRow, 0, len(concKeys))
	for _, conc := range concKeys {
		maxCtx := result.ConcurrencyResults[conc]
		byConc = append(byConc, probeRow{
			Label:      strconv.Itoa(conc),
			MaxContext: maxCtx,
			Vals: probeApplyVals(map[string]any{
				"model_id":      modelID,
				"max_model_len": maxCtx,
				"max_num_seqs":  conc,
			}),
		})
	}

	s.renderPartial(w, "probe_result", struct {
		TPSize        int
		Timestamp     string
		ByUtilization []probeRow
		ByConcurrency []probeRow
	}{result.TPSize, result.Timestamp.Format("2006-01-02 15:04"), byUtil, byConc})
}

// handleProbeForm renders the probe form partial. Models are restricted
// to registered ones; main vLLM must be stopped before probing.
func (s *Server) handleProbeForm(w http.ResponseWriter, r *http.Request) {
	type modelChoice struct{ ID, Name string }
	var choices []modelChoice
	for _, m := range s.registry.List() {
		if !m.Enabled || m.Orphaned {
			continue
		}
		choices = append(choices, modelChoice{ID: m.ID, Name: displayNameOf(m)})
	}

	// Probing starts its own vLLM processes, so the main one has to be out of
	// the way first — otherwise the two fight over the same VRAM.
	state := s.process.GetStatus().State
	blocked := state == process.StateRunning || state == process.StateStarting

	respondHTML(w)
	s.renderPartial(w, "probe_form", struct {
		Blocked bool
		Models  []modelChoice
	}{blocked, choices})
}

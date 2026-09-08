package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

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
	mu        sync.Mutex
	store     *probeResultStore
	env       *probeEnv
	active    *activeProbe
}

type activeProbe struct {
	id       string
	modelID  string
	cancel   context.CancelFunc

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
	ModelID            string    `json:"model_id"`
	TPSize             int       `json:"tp_size"`
	UtilizationLevels  []float64 `json:"utilization_levels,omitempty"`
	ConcurrencyLevels  []int     `json:"concurrency_levels,omitempty"`
	MinContext         int       `json:"min_context,omitempty"`
	MaxContext         int       `json:"max_context,omitempty"`
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
	fmt.Fprintf(w, `<p>Started probe <code>%s</code> for <code>%s</code>. <a href="#" hx-get="/api/benchmarks/probe-context/%s/progress" hx-target="#probe-progress" hx-swap="innerHTML" hx-trigger="every 3s">Refresh progress</a></p>`,
		esc(probeID), esc(req.ModelID), esc(probeID))
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
	renderProbeResult(w, modelID, result)
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

// renderProbeResult lays out the probe result as a two-column table:
// utilization sweep on the left, concurrency sweep on the right, each
// with an "Apply" button.
func renderProbeResult(w http.ResponseWriter, modelID string, result *benchmark.ProbeResult) {
	fmt.Fprintf(w, `<article id="probe-progress">
  <header><strong>Context probe results</strong> &middot; <small>tp=%d</small> &middot; <small>%s</small></header>`,
		result.TPSize, result.Timestamp.Format("2006-01-02 15:04"))

	// Utilization table.
	fmt.Fprint(w, `<h4>By GPU memory utilization</h4>
<table><thead><tr><th>util</th><th>max context</th><th></th></tr></thead><tbody>`)
	for util, maxCtx := range result.UtilizationResults {
		utilF, _ := strconv.ParseFloat(util, 64)
		fmt.Fprintf(w, `<tr>
  <td>%s</td><td>%d</td>
  <td><button class="secondary outline" style="padding:0.1rem 0.5rem;"
       hx-post="/api/benchmarks/probe-context/apply"
       hx-vals='{"model_id":"%s","max_model_len":%d,"gpu_memory_utilization":%.2f}'
       hx-ext="json-enc">apply</button></td>
</tr>`, util, maxCtx, esc(modelID), maxCtx, utilF)
	}
	fmt.Fprint(w, `</tbody></table>`)

	// Concurrency table.
	fmt.Fprint(w, `<h4>By max_num_seqs (at highest probed utilization)</h4>
<table><thead><tr><th>max_num_seqs</th><th>max context</th><th></th></tr></thead><tbody>`)
	for conc, maxCtx := range result.ConcurrencyResults {
		fmt.Fprintf(w, `<tr>
  <td>%d</td><td>%d</td>
  <td><button class="secondary outline" style="padding:0.1rem 0.5rem;"
       hx-post="/api/benchmarks/probe-context/apply"
       hx-vals='{"model_id":"%s","max_model_len":%d,"max_num_seqs":%d}'
       hx-ext="json-enc">apply</button></td>
</tr>`, conc, maxCtx, esc(modelID), maxCtx, conc)
	}
	fmt.Fprint(w, `</tbody></table></article>`)
}

// handleProbeForm renders the probe form partial. Models are restricted
// to registered ones; main vLLM must be stopped before probing.
func (s *Server) handleProbeForm(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	if state := s.process.GetStatus().State; state == process.StateRunning || state == process.StateStarting {
		fmt.Fprint(w, `<article><em>Stop the main vLLM process before probing. <a href="/service">Service page →</a></em></article>`)
		return
	}

	fmt.Fprint(w, `<article>
  <header><strong>Probe maximum context length</strong></header>
  <p style="opacity:0.7;"><small>Binary-search the largest <code>--max-model-len</code> the model can serve without OOM, across several GPU-memory-utilization and max_num_seqs levels. Main vLLM must be stopped.</small></p>
  <form hx-post="/api/benchmarks/probe-context" hx-target="#probe-progress" hx-swap="innerHTML">
    <label>Model
      <select name="model_id" required>`)
	for _, m := range s.registry.List() {
		if !m.Enabled || m.Orphaned {
			continue
		}
		fmt.Fprintf(w, `<option value="%s">%s</option>`, esc(m.ID), esc(displayNameOf(m)))
	}
	fmt.Fprint(w, `      </select>
    </label>
    <label>Tensor parallel size
      <select name="tp_size"><option value="1" selected>1</option><option value="2">2</option></select>
    </label>
    <button type="submit">Start probe</button>
  </form>
  <div id="probe-progress"></div>
</article>`)
	// minimal 30-min note
	_ = time.Now
}

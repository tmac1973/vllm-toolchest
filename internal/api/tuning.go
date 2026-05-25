package api

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/tmac1973/vllm-toolchest/internal/tuning"
)

// shapeView is one row in the per-model shape table.
type shapeView struct {
	N      int  `json:"n"`
	K      int  `json:"k"`
	BlockN int  `json:"block_n"`
	BlockK int  `json:"block_k"`
	Tuned  bool `json:"tuned"`
}

// modelTuningView is what the /tuning page lists per model.
type modelTuningView struct {
	ID         string      `json:"id"`
	IsFP8      bool        `json:"is_fp8"`
	TPSize     int         `json:"tp_size"`
	Shapes     []shapeView `json:"shapes"`
	TunedCount int         `json:"tuned_count"`
	TotalCount int         `json:"total_count"`
}

type tuningPageData struct {
	Title      string
	Nav        string
	DeviceName string
	Models     []modelTuningView
	Active     *tuning.Job
}

func (s *Server) handleTuningPage(w http.ResponseWriter, r *http.Request) {
	views := s.buildTuningViews()
	data := tuningPageData{
		Title:      "Kernel Tuning",
		Nav:        "tuning",
		DeviceName: s.tuner.DeviceName(),
		Models:     views,
		Active:     s.tuner.ActiveJob(),
	}
	s.render(w, "tuning.html", data)
}

// buildTuningViews snapshots the registry through the lens of tuning status.
// Heavy lift is shape derivation, which is pure-Go and cheap.
func (s *Server) buildTuningViews() []modelTuningView {
	all := s.registry.List()
	out := make([]modelTuningView, 0, len(all))
	for _, m := range all {
		isFP8 := m.Quantization.Method == "fp8" ||
			m.Quantization.Method == "compressed-tensors" ||
			m.Quantization.Method == "compressed_tensors"
		tp := m.VLLMConfig.TensorParallelSize
		if tp < 1 {
			tp = 1
		}
		shapes := tuning.DeriveShapes(m.HFConfig, tp, 128, 128)

		row := modelTuningView{
			ID:         m.ID,
			IsFP8:      isFP8,
			TPSize:     tp,
			TotalCount: len(shapes),
		}
		row.Shapes = make([]shapeView, len(shapes))
		for i, sh := range shapes {
			tuned := s.tuner.IsTuned(sh, 128, 128)
			if tuned {
				row.TunedCount++
			}
			row.Shapes[i] = shapeView{N: sh.N, K: sh.K, BlockN: 128, BlockK: 128, Tuned: tuned}
		}
		out = append(out, row)
	}
	return out
}

// POST /api/tuning/start  — body {model_id: "..."}
func (s *Server) handleStartTuning(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ModelID string `json:"model_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	if body.ModelID == "" {
		http.Error(w, "model_id required", http.StatusBadRequest)
		return
	}
	m, ok := s.registry.Get(body.ModelID)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	tp := m.VLLMConfig.TensorParallelSize
	if tp < 1 {
		tp = 1
	}
	shapes := tuning.DeriveShapes(m.HFConfig, tp, 128, 128)
	if len(shapes) == 0 {
		http.Error(w, "no tunable shapes derived from model config", http.StatusBadRequest)
		return
	}
	job, err := s.tuner.StartJob(m.ID, shapes, tp, 128, 128)
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	writeJSON(w, http.StatusAccepted, job)
}

// POST /api/tuning/cancel
func (s *Server) handleCancelTuning(w http.ResponseWriter, r *http.Request) {
	s.tuner.Cancel()
	w.WriteHeader(http.StatusNoContent)
}

// GET /api/tuning/status
func (s *Server) handleTuningStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"device_name": s.tuner.DeviceName(),
		"active":      s.tuner.ActiveJob(),
		"models":      s.buildTuningViews(),
	})
}

// GET /api/tuning/logs — paginated dump of recent log lines.
func (s *Server) handleTuningLogs(w http.ResponseWriter, r *http.Request) {
	limit := 500
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 5000 {
			limit = n
		}
	}
	all := s.tuner.LogBuffer()
	if len(all) > limit {
		all = all[len(all)-limit:]
	}
	writeJSON(w, http.StatusOK, all)
}

// GET /api/tuning/log-stream — SSE of live log lines.
func (s *Server) handleTuningLogStream(w http.ResponseWriter, r *http.Request) {
	ch := s.tuner.Subscribe()
	defer s.tuner.Unsubscribe(ch)
	StreamLines(w, r.Context(), ch, "tuning job ended")
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

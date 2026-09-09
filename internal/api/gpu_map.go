package api

import (
	"fmt"
	"net/http"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// The GPU allocation map: one bar per card, showing how much of it is spoken
// for and by what.
//
// llama-toolchest can attribute each segment to a model, because its router
// holds several at once and assigns them to cards. vLLM runs one engine, and
// a tensor-parallel engine spreads one model evenly across its ranks — so the
// interesting split here is not model-vs-model but "the engine" against
// "everything else on this card", which on a workstation is a desktop session
// and is exactly what makes a load fail to fit.
type gpuMapSegment struct {
	Label    string
	WidthPct float64
	Color    string
}

type gpuMapBar struct {
	Index      int
	Name       string
	UsedGB     float64
	TotalGB    float64
	Overcommit bool
	Segments   []gpuMapSegment
}

type gpuMapLegend struct {
	Label string
	Color string
}

const (
	gpuMapEngineColor = "#5b7fbd"
	gpuMapOtherColor  = "#8a6d3b"
)

func (s *Server) handleGPUMap(w http.ResponseWriter, r *http.Request) {
	metrics := s.monitor.Current()
	if len(metrics.GPU) == 0 {
		respondHTML(w)
		s.renderPartial(w, "gpu_map", struct {
			Bars   []gpuMapBar
			Legend []gpuMapLegend
		}{})
		return
	}

	// What the engine is using per card, if one is up. vLLM does not report
	// this per rank, so it is derived: the model's estimate divided across the
	// tensor-parallel size it was started with. It is an attribution of
	// already-measured usage, not a second estimate — the bar's length always
	// comes from what the driver reports.
	var enginePerGPUGB float64
	engineName := ""
	if st := s.process.GetStatus(); st.State == process.StateRunning {
		if m, ok := s.registry.Get(st.ModelID); ok {
			tp := m.VLLMConfig.TensorParallelSize
			if tp < 1 {
				tp = 1
			}
			enginePerGPUGB = m.VRAMEstimate.TotalSingleGPUGB / float64(tp)
			engineName = displayNameOf(m)
		}
	}

	var bars []gpuMapBar
	anyEngine := false
	for _, g := range metrics.GPU {
		usedGB := float64(g.VRAMUsedMB) / 1024
		totalGB := float64(g.VRAMTotalMB) / 1024
		bar := gpuMapBar{
			Index:      g.Index,
			Name:       g.Name,
			UsedGB:     usedGB,
			TotalGB:    totalGB,
			Overcommit: totalGB > 0 && usedGB > totalGB,
		}

		if totalGB > 0 {
			engine := enginePerGPUGB
			if engine > usedGB {
				// The estimate overshot what the card actually reports; the
				// measurement is the thing that is true.
				engine = usedGB
			}
			if engine > 0 {
				anyEngine = true
				bar.Segments = append(bar.Segments, gpuMapSegment{
					Label:    engineName,
					WidthPct: engine / totalGB * 100,
					Color:    gpuMapEngineColor,
				})
			}
			if other := usedGB - engine; other > 0.05 {
				bar.Segments = append(bar.Segments, gpuMapSegment{
					Label:    "other",
					WidthPct: other / totalGB * 100,
					Color:    gpuMapOtherColor,
				})
			}
		}
		bars = append(bars, bar)
	}

	var legend []gpuMapLegend
	if anyEngine {
		legend = append(legend, gpuMapLegend{
			Label: fmt.Sprintf("%s (engine)", engineName),
			Color: gpuMapEngineColor,
		})
	}
	legend = append(legend, gpuMapLegend{Label: "other processes", Color: gpuMapOtherColor})

	respondHTML(w)
	s.renderPartial(w, "gpu_map", struct {
		Bars   []gpuMapBar
		Legend []gpuMapLegend
	}{bars, legend})
}

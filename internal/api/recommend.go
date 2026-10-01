package api

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/recommend"
)

// recommendState holds the feed's engine, made on first use.
type recommendState struct {
	once   sync.Once
	engine *recommend.Engine
}

// serverHub reaches the server's Hugging Face client at each call, so a
// token set in Settings after the engine was made is used.
type serverHub struct{ s *Server }

var errNoHub = errors.New("no Hugging Face client")

func (h serverHub) Candidates(ctx context.Context, q huggingface.CandidateQuery) ([]huggingface.ModelSearchResult, error) {
	if h.s.hfClient == nil {
		return nil, errNoHub
	}
	return h.s.hfClient.Candidates(ctx, q)
}

func (h serverHub) FetchConfigJSON(ctx context.Context, modelID, revision string) ([]byte, error) {
	if h.s.hfClient == nil {
		return nil, errNoHub
	}
	return h.s.hfClient.FetchConfigJSON(ctx, modelID, revision)
}

func (h serverHub) GetFiles(ctx context.Context, modelID, revision string) (string, []huggingface.ModelFile, error) {
	if h.s.hfClient == nil {
		return "", nil, errNoHub
	}
	return h.s.hfClient.GetFiles(ctx, modelID, revision)
}

func (s *Server) recommendEngine() *recommend.Engine {
	s.recommend.once.Do(func() {
		s.recommend.engine = recommend.NewEngine(serverHub{s}, s.cfg.DataDir)
	})
	return s.recommend.engine
}

// recommendProfile is this machine as the feed judges it, from live readings:
// the cards, the image and what its vLLM can load, the defaults a download
// is seeded with, and the host's RAM for expert offload.
func (s *Server) recommendProfile() recommend.Profile {
	inv := s.gpuInventory()
	inv.FreePerCardGB = 0 // judged as if this model were the one running
	desc, _ := s.vllmEnv.Descriptor()
	archs, known := s.SupportedArchs()
	var gpuName string
	var hostRAM float64
	if s.monitor != nil {
		cur := s.monitor.Current()
		if len(cur.GPU) > 0 {
			gpuName = cur.GPU[0].Name
		}
		hostRAM = float64(cur.Memory.TotalMB) / 1024
	}
	return recommend.NewProfile(inv, gpuName, s.cfg.GPUArch, desc, archs, known,
		models.PlanDefaults{GPUMemoryUtilization: s.cfg.GPUMemoryUtil, MaxNumSeqs: s.cfg.MaxNumSeqs}, hostRAM)
}

// recommendTimeout bounds a build: twenty Hub queries and forty finalists'
// configs and file listings.
const recommendTimeout = 3 * time.Minute

// handleRecommend serves the feed for one aim. It is always a 200: a failure
// is reported in "unavailable", so the page renders and the search below it
// keeps working.
func (s *Server) handleRecommend(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), recommendTimeout)
	defer cancel()
	respondJSON(w, s.recommendEngine().Result(ctx, s.recommendProfile(), r.URL.Query().Get("intent")))
}

// handleRecommendRefresh builds the pool again.
func (s *Server) handleRecommendRefresh(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), recommendTimeout)
	defer cancel()
	respondJSON(w, s.recommendEngine().Refresh(ctx, s.recommendProfile()))
}

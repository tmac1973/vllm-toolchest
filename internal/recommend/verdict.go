package recommend

import (
	"context"
	"errors"
	"sync"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// The reasons a finalist is unverified, first match wins.
const (
	reasonNoListing  = "size unknown — could not read the repository's file listing"
	reasonNoWeights  = "size unknown — the repository's file listing has no weight files"
	reasonNoConfig   = "config unavailable — could not fetch config.json"
	reasonUnreadable = "config unreadable — config.json is missing the fields the fit needs"
)

func reasonArch(arch string) string {
	return "architecture " + arch + " is not in this image's model registry"
}

// evaluate fetches and judges every finalist, fetchWorkers at a time.
func evaluate(ctx context.Context, hub Hub, dataDir string, p Profile, cands []Candidate) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, fetchWorkers)
	for i := range cands {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			judge(ctx, hub, dataDir, p, &cands[i])
		}()
	}
	wg.Wait()
}

// judge gives one finalist its verdict, and when it runs here, its plan.
//
// The plan is models.PlanFit -- the planner autoconfigure uses -- on the model
// as registration would describe it, at the most context it holds: so the
// width, context and requests a card states are what Autoconfigure would
// configure, offload included on an image that has it.
func judge(ctx context.Context, hub Hub, dataDir string, p Profile, c *Candidate) {
	dir := cacheDir(dataDir, *c)
	weights, known, err := fetchFinalist(ctx, hub, dir, *c)
	switch {
	case errors.Is(err, errNoConfig):
		c.Verdict, c.Reason = Unverified, reasonNoConfig
		return
	case errors.Is(err, errNoListing):
		c.Verdict, c.Reason = Unverified, reasonNoListing
		return
	case err != nil:
		c.Verdict, c.Reason = Unverified, reasonNoConfig
		return
	case !known:
		c.Verdict, c.Reason = Unverified, reasonNoWeights
		return
	}
	c.WeightBytes, c.WeightsKnown = weights, true
	c.WeightGB = float64(weights) / (1 << 30)

	m := models.Describe(c.ID, dir)
	m.TotalSizeBytes = weights
	cfg := m.HFConfig
	if cfg.NumHiddenLayers <= 0 || cfg.HiddenSize <= 0 {
		c.Verdict, c.Reason = Unverified, reasonUnreadable
		return
	}
	if cfg.Draft != nil {
		// A drafter proposes tokens for another model; it cannot be served
		// on its own, as registration already knows.
		c.Verdict = Dropped
		return
	}
	if len(cfg.Architectures) > 0 {
		c.Arch = cfg.Architectures[0] // the file vLLM itself reads
	}
	if p.ArchsKnown {
		c.ArchSupported = p.Archs[c.Arch]
		if !c.ArchSupported {
			c.Verdict, c.Reason = Unverified, reasonArch(c.Arch)
			return
		}
	}

	plan := models.PlanFit(models.PlanInput{
		Model: m, Base: m.VLLMConfig, Inventory: p.Inventory, Class: models.ContextMax,
		Defaults: p.Defaults,
		Estimate: func(vc models.VLLMConfig) models.VRAMEstimate {
			x := *m
			x.VLLMConfig = vc
			var env []string
			if p.Env != nil {
				env = p.Env(&x)
			}
			return models.EstimateVRAM(&x, env)
		},
		ExpertOffload: p.ExpertOffload, HostRAMGB: p.HostRAMGB,
	})
	if !plan.Known {
		c.Verdict = Dropped
		return
	}
	// The narrowest width that holds the most context, when it is fewer
	// cards; the all-cards plan otherwise. The context stated is what the
	// all-cards plan holds: the most this machine gives the model.
	w := plan.All
	if plan.Narrow != nil {
		w = *plan.Narrow
	}
	c.Verdict = Verified
	c.TP, c.Required, c.Available = w.TP, w.RequiredGB, w.AvailableGB
	c.Spare = max(0, w.AvailableGB-w.RequiredGB)
	c.AffordableTokens = plan.All.ContextTokens
	c.FullContextRequests = plan.All.FullContextRequests
	c.Offload = plan.All.Offload
	c.Config = w.Config
}

package api

import "github.com/tmac1973/vllm-toolchest/internal/models"

// planInput is what the hardware planner needs for one model on this host.
//
// base is the config to plan on, and every caller says which: autoconfigure
// passes the config the model card produced, refinement the live config, and
// seeding the model's default config.
//
// The inventory has free memory cleared. The planner asks what the model
// needs when it is the one running, and whatever else is on the cards now --
// usually the engine, serving something else -- will have been stopped by
// then.
func (s *Server) planInput(m *models.Model, base models.VLLMConfig, class models.ContextClass, cardKVDtype string) models.PlanInput {
	inv := s.gpuInventory()
	inv.FreePerCardGB = 0
	return models.PlanInput{
		Model:     m,
		Base:      base,
		Inventory: inv,
		Class:     class,
		Defaults: models.PlanDefaults{
			GPUMemoryUtilization: s.cfg.GPUMemoryUtil,
			MaxNumSeqs:           s.cfg.MaxNumSeqs,
		},
		CardKVDtype: cardKVDtype,
		Estimate:    s.estimateFor(m),
	}
}

// estimateFor returns what m needs under a candidate config, by the rule
// effectiveVRAM uses: a measurement when one applies to that config, the
// projection with the fully resolved environment otherwise.
func (s *Server) estimateFor(m *models.Model) func(models.VLLMConfig) models.VRAMEstimate {
	return func(c models.VLLMConfig) models.VRAMEstimate {
		cp := *m
		cp.VLLMConfig = c
		if est, ok := models.MeasuredEstimate(&cp, s.engineIdentity()); ok {
			return est
		}
		est := models.EstimateVRAM(&cp, s.configuredEnvPairs(&cp))
		est.Source = models.SourceProjected
		return est
	}
}

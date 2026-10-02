package recommend

import (
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// SeedConfig is base with the hardware settings a plan chose: the width, the
// context, the KV cache dtype, the memory fraction and the batch cap, with a
// pinned pool cleared -- as autoconfigure writes them. It works nothing out;
// the plan is models.PlanFit's, for the downloaded model on this machine.
//
// Nothing else is taken from the plan, except what makes an expert-offload
// plan startable: the flag that turns offload on, and leaving out an MTP
// config, which this image cannot start with offload. Every other field
// stays as registration defaulted it: seeding is not an optimizer, and
// pinning a field here would stop it tracking the image's defaults.
func SeedConfig(base, planned models.VLLMConfig) models.VLLMConfig {
	c := base
	c.TensorParallelSize = planned.TensorParallelSize
	c.MaxModelLen = planned.MaxModelLen
	c.KVCacheDtype = planned.KVCacheDtype
	c.GPUMemoryUtilization = planned.GPUMemoryUtilization
	c.MaxNumSeqs = planned.MaxNumSeqs
	c.KVCacheMemory = 0
	if process.HasFlag(planned.ExtraFlags, models.ExpertOffloadFlag) {
		c.ExtraFlags = process.SetFlag(c.ExtraFlags, []string{models.ExpertOffloadFlag})
		if ref, ok := models.ParseSpeculative(c.SpeculativeConfig); ok && strings.Contains(strings.ToLower(ref.Method), "mtp") {
			c.SpeculativeConfig = ""
		}
	}
	return c
}

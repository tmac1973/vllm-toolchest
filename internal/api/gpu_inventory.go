package api

import (
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
)

// gpuInventory is the hardware the fit calculation is judged against.
func (s *Server) gpuInventory() models.GPUInventory {
	if s.gpuInvOverride != nil {
		return *s.gpuInvOverride
	}
	if s.monitor == nil {
		// No monitor at all is the same answer as a monitor with nothing in
		// it: figures without a verdict.
		return models.GPUInventory{}
	}
	return gpuInventoryFrom(s.monitor.Current().GPU)
}

// gpuInventoryFrom reduces a monitor reading to how many cards this host has
// and how much memory the smallest of them carries.
//
// The smallest card wins because a tensor-parallel group is bounded by its
// weakest member. Every rank holds an equal shard, so one 16 GiB card beside
// three 32 GiB cards makes the group behave like four 16 GiB cards, and
// averaging would promise a fit that fails on exactly one rank.
//
// Known stays false when nothing has been read from the driver yet -- a fresh
// process, or a host with no GPU. Callers must then show the figures without a
// verdict, rather than falling back to an invented card size: inventing one is
// what had a four-card host judged against a single 32 GiB card.
func gpuInventoryFrom(gpus []monitor.GPUInfo) models.GPUInventory {
	var inv models.GPUInventory
	for _, g := range gpus {
		if g.VRAMTotalMB <= 0 {
			continue
		}
		gb := float64(g.VRAMTotalMB) / 1024
		if !inv.Known || gb < inv.PerCardGB {
			inv.PerCardGB = gb
		}
		inv.Count++
		inv.Known = true
	}
	return inv
}

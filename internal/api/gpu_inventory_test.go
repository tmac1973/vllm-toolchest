package api

import (
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/monitor"
)

func TestGPUInventoryFrom(t *testing.T) {
	for _, tc := range []struct {
		name       string
		gpus       []monitor.GPUInfo
		wantCount  int
		wantPerGB  float64
		wantFreeGB float64
		wantKnown  bool
	}{
		{
			name:      "no reading yet",
			wantKnown: false,
		},
		{
			name: "four uniform cards",
			gpus: []monitor.GPUInfo{
				{VRAMTotalMB: 32620}, {VRAMTotalMB: 32620},
				{VRAMTotalMB: 32620}, {VRAMTotalMB: 32620},
			},
			wantCount: 4, wantPerGB: 31.855, wantKnown: true,
		},
		{
			// A tensor-parallel group is bounded by its weakest member, so the
			// smallest card is the one that counts. Averaging here would
			// promise a fit that fails on exactly one rank.
			name: "mixed cards take the smallest",
			gpus: []monitor.GPUInfo{
				{VRAMTotalMB: 32768}, {VRAMTotalMB: 16384}, {VRAMTotalMB: 32768},
			},
			wantCount: 3, wantPerGB: 16, wantKnown: true,
		},
		{
			// A card the driver reports nothing for is not a card we can plan
			// against, and counting it as 0 GiB would bound every group at
			// zero.
			name: "cards with no reading are skipped",
			gpus: []monitor.GPUInfo{
				{VRAMTotalMB: 32768}, {VRAMTotalMB: 0},
			},
			wantCount: 1, wantPerGB: 32, wantKnown: true,
		},
		{
			// The case that let the estimator promise a fit vLLM refuses:
			// a leaked worker from a cancelled tuning job was holding 1.2 GiB,
			// and an estimate drawn from card size alone could not see it.
			name: "memory already in use counts against the budget",
			gpus: []monitor.GPUInfo{
				{VRAMTotalMB: 32620, VRAMUsedMB: 1270},
				{VRAMTotalMB: 32620, VRAMUsedMB: 993},
				{VRAMTotalMB: 32620, VRAMUsedMB: 61},
				{VRAMTotalMB: 32620, VRAMUsedMB: 61},
			},
			wantCount: 4, wantPerGB: 31.855, wantKnown: true,
			wantFreeGB: 30.615,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := gpuInventoryFrom(tc.gpus)
			if tc.wantFreeGB > 0 {
				if diff := got.FreePerCardGB - tc.wantFreeGB; diff > 0.01 || diff < -0.01 {
					t.Errorf("FreePerCardGB = %.3f, want %.3f", got.FreePerCardGB, tc.wantFreeGB)
				}
			}
			if got.Known != tc.wantKnown {
				t.Errorf("Known = %v, want %v", got.Known, tc.wantKnown)
			}
			if got.Count != tc.wantCount {
				t.Errorf("Count = %d, want %d", got.Count, tc.wantCount)
			}
			if diff := got.PerCardGB - tc.wantPerGB; diff > 0.01 || diff < -0.01 {
				t.Errorf("PerCardGB = %.3f, want %.3f", got.PerCardGB, tc.wantPerGB)
			}
		})
	}
}

// A server with no monitor at all must answer the same way as one with nothing
// in it: figures, no verdict. It renders the config panel, so a nil dereference
// here takes the page down.
func TestGPUInventoryToleratesNoMonitor(t *testing.T) {
	s := &Server{}
	if inv := s.gpuInventory(); inv.Known {
		t.Errorf("reported a known inventory with no monitor: %+v", inv)
	}
}

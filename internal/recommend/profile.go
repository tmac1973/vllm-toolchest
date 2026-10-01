// Package recommend ranks Hub models against this machine: which will run
// here, how well, and in what order for each of four aims. It decides which
// repositories to ask about and orders the answers; the fit itself is
// models.PlanFit, the planner autoconfigure uses, so a recommendation and a
// later Autoconfigure agree.
package recommend

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/variants"
)

// Profile is the machine a pool is judged against.
type Profile struct {
	Inventory   models.GPUInventory
	GPUName     string // e.g. "AMD Radeon AI PRO R9700"; may be ""
	GPUArch     string // e.g. "gfx1201"
	Variant     string // the image variant, e.g. "rdna4-clav"
	Accelerated []string
	Archs       map[string]bool // what the image's vLLM can load
	ArchsKnown  bool

	// What the planner needs beyond the cards: the machine-wide defaults a
	// download is seeded with, and the host RAM expert offload can use on an
	// image that has it.
	Defaults      models.PlanDefaults
	ExpertOffload bool
	HostRAMGB     float64
}

// NewProfile builds a profile, working out the accelerated formats from the
// GPU architecture less what the image lacks kernels for.
func NewProfile(inv models.GPUInventory, gpuName, gpuArch string, d variants.Descriptor,
	archs map[string]bool, archsKnown bool, defaults models.PlanDefaults, hostRAMGB float64) Profile {
	return Profile{
		Inventory: inv, GPUName: gpuName, GPUArch: gpuArch, Variant: d.ID,
		Accelerated: acceleratedFormats(gpuArch, d.NoAccel),
		Archs:       archs, ArchsKnown: archsKnown,
		Defaults: defaults, ExpertOffload: d.Has("expert_offload"), HostRAMGB: hostRAMGB,
	}
}

// Key identifies the judgement a pool rests on. A pool built for another key
// is never served: it is wrong in a way the user cannot see.
func (p Profile) Key() string {
	acc := slices.Clone(p.Accelerated)
	slices.Sort(acc)
	return strings.Join([]string{
		fmt.Sprint(p.Inventory.Count), fmt.Sprintf("%.1f", p.Inventory.PerCardGB), p.GPUArch, p.Variant,
		strings.Join(acc, ","), fmt.Sprint(p.ArchsKnown), fmt.Sprint(len(p.Archs)),
		fmt.Sprintf("%.2f", p.Defaults.GPUMemoryUtilization), fmt.Sprint(p.Defaults.MaxNumSeqs),
		fmt.Sprint(p.ExpertOffload), fmt.Sprintf("%.0f", p.HostRAMGB),
	}, "|")
}

// ProfileView is the profile as the endpoint reports it.
type ProfileView struct {
	GPUCount       int      `json:"gpu_count"`
	PerCardGB      float64  `json:"per_card_gb"`
	TotalVRAMGB    float64  `json:"total_vram_gb"`
	GPUName        string   `json:"gpu_name,omitempty"`
	GPUArch        string   `json:"gpu_arch"`
	Variant        string   `json:"variant"`
	Accelerated    []string `json:"accelerated"`
	ArchsKnown     bool     `json:"archs_known"`
	InventoryKnown bool     `json:"inventory_known"`
	ExpertOffload  bool     `json:"expert_offload,omitempty"`
}

// View is the profile flattened for the endpoint.
func (p Profile) View() ProfileView {
	acc := p.Accelerated
	if acc == nil {
		acc = []string{}
	}
	return ProfileView{
		GPUCount: p.Inventory.Count, PerCardGB: p.Inventory.PerCardGB,
		TotalVRAMGB: float64(p.Inventory.Count) * p.Inventory.PerCardGB,
		GPUName:     p.GPUName, GPUArch: p.GPUArch, Variant: p.Variant, Accelerated: acc,
		ArchsKnown: p.ArchsKnown, InventoryKnown: p.Inventory.Known, ExpertOffload: p.ExpertOffload,
	}
}

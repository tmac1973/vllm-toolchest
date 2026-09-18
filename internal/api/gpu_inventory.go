package api

import (
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
	"github.com/tmac1973/vllm-toolchest/internal/process"
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

	// The panel asks "could this be started", so memory the engine under
	// discussion is already holding must not count against it. While a model
	// is loaded it occupies most of every card, and subtracting that reported
	// "of 12.3 GB available" beside a model that had been serving for
	// twenty-four minutes -- a risk written into todo.md before the change
	// shipped, and shipped anyway.
	//
	// Giving the whole of the used figure back over-credits when something
	// else is also resident, and that is the direction to err: it restores
	// exactly the behaviour that held before free memory was consulted at all,
	// and the pre-flight refusal this was meant to catch is reported by the
	// engine itself as advice rather than guessed at here.
	engineUp := s.process != nil && s.process.GetStatus().State == process.StateRunning
	return gpuInventoryFrom(s.monitor.Current().GPU, engineUp)
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
// engineUp says the model under discussion is the thing occupying the cards,
// in which case what is resident is not competition for the budget and the
// card counts as empty. With the engine down, anything resident is somebody
// else's -- a leaked worker from a cancelled job, most often -- and does count.
func gpuInventoryFrom(gpus []monitor.GPUInfo, engineUp bool) models.GPUInventory {
	var inv models.GPUInventory
	for _, g := range gpus {
		if g.VRAMTotalMB <= 0 {
			continue
		}
		gb := float64(g.VRAMTotalMB) / 1024
		if !inv.Known || gb < inv.PerCardGB {
			inv.PerCardGB = gb
		}
		// vLLM checks the fraction it was asked for against memory that is
		// actually free, not against the card's size, and refuses to start
		// when they disagree:
		//
		//   Free memory on device cuda:0 (27.28/31.86 GiB) on startup is less
		//   than desired GPU memory utilization (0.97, 30.9 GiB)
		//
		// So an estimate drawn from the card's size alone can report a
		// comfortable fit for a configuration that cannot start. Anything
		// already resident -- a leaked worker from a cancelled job, another
		// user's process -- counts against the budget.
		free := float64(g.VRAMTotalMB-g.VRAMUsedMB) / 1024
		if free < 0 {
			free = 0
		}
		// engineUp means what is resident is mostly the engine we are being
		// asked about, so the card counts as empty. See gpuInventory.
		if engineUp {
			free = gb
		}
		if !inv.Known || free < inv.FreePerCardGB {
			inv.FreePerCardGB = free
		}
		inv.Count++
		inv.Known = true
	}
	return inv
}

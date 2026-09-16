package models

import (
	"strconv"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// Offload records what a launch configuration deliberately keeps outside GPU
// memory. It matters to the estimate because those bytes are in the checkpoint
// on disk and are never resident on a card, and counting them as VRAM is what
// made a model that serves happily on four 32 GiB cards report "Too large".
type Offload struct {
	// PLE is the n-gram / per-layer embedding table held in system RAM.
	PLE bool `json:"ple,omitempty"`
	// Experts is MoE expert weights streamed from host RAM on demand.
	Experts bool `json:"experts,omitempty"`
	// NVMe is offload backed by storage rather than RAM. Detected so the
	// estimate can say it does not model it, never to size it.
	NVMe bool `json:"nvme,omitempty"`

	// ExpertGB is the cache the operator sized explicitly, and ExpertSized
	// says whether they did. Expert offload without a size is the case that
	// forces a range rather than a figure: the engine picks the cache itself
	// and we cannot know what it picked without asking it.
	ExpertGB    float64 `json:"expert_gb,omitempty"`
	ExpertSized bool    `json:"expert_sized,omitempty"`
}

// Any reports whether anything at all is being kept off the GPUs.
func (o Offload) Any() bool { return o.PLE || o.Experts || o.NVMe }

// Sized reports whether every active offload has a known size. When it is
// false the estimate must show a band rather than a number.
//
// PLE never counts as sized. Its size is inferred from the tensors the
// structural formula cannot account for, which is a reading of a residual and
// not a measurement: it reconciles with the two checkpoints it was calibrated
// against, and it will misattribute any other unmodelled tensor to the
// embedding table. On a checkpoint with no such table the residual is zero, so
// treating it as certain would have the estimate announce an offload and then
// apply nothing to it -- confident, and silently wrong in whichever direction
// the residual happens to fall.
func (o Offload) Sized() bool {
	if o.NVMe || o.PLE {
		return false
	}
	if o.Experts && !o.ExpertSized {
		return false
	}
	return true
}

// DetectOffload reads the offload settings out of a resolved launch
// environment and the extra-flags string.
//
// It takes the environment already resolved rather than reaching for the
// model's own Env block, and that is the whole point: a variable set
// machine-wide, or by a variant knob, reaches the engine exactly as one typed
// into the model's box does. Reading only the model's block would have the
// estimate disagree with the command that actually runs -- the drift the
// layered environment was unified to prevent.
func DetectOffload(envPairs []string, extraFlags string) Offload {
	var o Offload

	// Last occurrence wins, matching os/exec: a later layer overriding an
	// earlier one must be able to turn offload back off, not only on.
	for _, p := range envPairs {
		k, v, ok := strings.Cut(p, "=")
		if !ok {
			continue
		}
		switch strings.TrimSpace(k) {
		case "VLLM_PLE_CPU_OFFLOAD":
			o.PLE = truthyEnv(v)
		case "VLLM_EXPERT_CPU_OFFLOAD":
			o.Experts = truthyEnv(v)
		}
	}

	flags := process.SplitFlags(extraFlags)
	for i := 0; i < len(flags); i++ {
		name, val, hasVal := strings.Cut(flags[i], "=")
		// vLLM accepts both "--flag=value" and "--flag value".
		if !hasVal && i+1 < len(flags) && !strings.HasPrefix(flags[i+1], "-") {
			val, hasVal = flags[i+1], true
			i++
		}

		switch name {
		case "--enable-expert-offload":
			o.Experts = true
		case "--expert-offload-mem", "--expert-cache-gb":
			o.Experts = true
			if gb, ok := parseOffloadGB(val, name); hasVal && ok {
				o.ExpertGB, o.ExpertSized = gb, true
			}
		case "--ple-nvme-offload":
			// Storage-backed, so nothing is resident on a card, but the
			// engine still stages through a buffer we do not model.
			o.PLE = true
			o.NVMe = true
		case "--enable-ple-cpu-offload":
			o.PLE = true
		}
	}

	return o
}

func truthyEnv(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// parseOffloadGB reads a memory size in the spellings these flags accept.
//
// A bare number is ambiguous, so the flag name settles it: anything named
// "-gb" is gigabytes, and elsewhere a value large enough to be a byte count
// is treated as one. Returns false rather than guessing when it cannot tell,
// which surfaces as a range in the estimate instead of a wrong figure.
func parseOffloadGB(v, flag string) (float64, bool) {
	s := strings.TrimSpace(strings.ToLower(v))
	if s == "" {
		return 0, false
	}

	mult := 0.0
	switch {
	case strings.HasSuffix(s, "gib"), strings.HasSuffix(s, "gb"), strings.HasSuffix(s, "g"):
		mult = 1
	case strings.HasSuffix(s, "mib"), strings.HasSuffix(s, "mb"), strings.HasSuffix(s, "m"):
		mult = 1.0 / 1024
	case strings.HasSuffix(s, "tib"), strings.HasSuffix(s, "tb"), strings.HasSuffix(s, "t"):
		mult = 1024
	}
	if mult > 0 {
		n, err := strconv.ParseFloat(strings.TrimRight(s, "gibmt"), 64)
		if err != nil {
			return 0, false
		}
		return n * mult, true
	}

	n, err := strconv.ParseFloat(s, 64)
	if err != nil || n <= 0 {
		return 0, false
	}
	if strings.HasSuffix(flag, "-gb") {
		return n, true
	}
	// Bare number on a byte-valued flag: only a byte count is plausible at
	// this magnitude, and only gigabytes are plausible below it.
	if n >= 1<<30 {
		return n / (1024 * 1024 * 1024), true
	}
	return n, true
}

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

	// ExpertHostCapGB is --expert-offload-mem: a ceiling on how much expert
	// weight may live in host RAM. It is not a statement of how much does --
	// a run configured with 46 here was measured moving 18.72 GB -- so it
	// bounds the optimistic end of the band and never sets a figure.
	ExpertHostCapGB float64 `json:"expert_host_cap_gb,omitempty"`
	ExpertCapSet    bool    `json:"expert_cap_set,omitempty"`

	// ExpertCacheGB is --expert-cache-gb: the staging cache for streamed
	// experts. It is resident on the card, so it adds to what a rank must
	// hold rather than subtracting from it.
	//
	// These two were once parsed into one field, which meant a command
	// carrying both had the second silently overwrite the first: a 46 GB
	// offload ceiling was recorded as a 5.5 GB one.
	ExpertCacheGB  float64 `json:"expert_cache_gb,omitempty"`
	ExpertCacheSet bool    `json:"expert_cache_set,omitempty"`
}

// Any reports whether anything at all is being kept off the GPUs.
func (o Offload) Any() bool { return o.PLE || o.Experts || o.NVMe }

// Bounded reports whether the host-resident share can be bounded at all.
//
// Nothing here is ever known exactly. The PLE table is estimated from the
// unaccounted-tensor residual, and expert offload is given only as a ceiling.
// What matters is whether the bounds are tight enough to judge, which is a
// question for the fit against a particular budget -- not something the flags
// can answer on their own. So this reports only the case where no bound exists
// at all: NVMe, whose staging behaviour is not modelled.
func (o Offload) Bounded() bool { return !o.NVMe }

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
		case "--expert-offload-mem":
			o.Experts = true
			if gb, ok := parseOffloadGB(val, name); hasVal && ok {
				o.ExpertHostCapGB, o.ExpertCapSet = gb, true
			}
		case "--expert-cache-gb":
			o.Experts = true
			if gb, ok := parseOffloadGB(val, name); hasVal && ok {
				o.ExpertCacheGB, o.ExpertCacheSet = gb, true
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

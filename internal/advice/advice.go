// Package advice reads vLLM's own output: what it measured on a start, and
// what it suggests changing when something is wrong.
//
// Both halves are thrown away today. The measurements are the more valuable:
// the VRAM estimator infers figures the engine reports exactly, and a
// calibration constant derived by reading a terminal is a constant nobody can
// check later. The advice is the more obvious: when a start fails, vLLM
// usually names the setting to change, and the user gets a stack trace.
//
// It imports nothing from this project on purpose. internal/process is
// deliberately thin -- it depends only on internal/ansi -- and the process
// layer is exactly where log lines arrive first.
package advice

import (
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Severity orders items for display. A string rather than an integer because
// it crosses into JSON and HTML templates, where a numeric severity renders as
// an unhelpful "2".
type Severity string

const (
	Info    Severity = "info"
	Warning Severity = "warning"
	Error   Severity = "error"
)

// Item is one thing the engine said worth acting on.
type Item struct {
	Severity Severity `json:"severity"`
	// Message is what to tell the user, in our words.
	Message string `json:"message"`
	// Field names the VLLMConfig setting implicated, or "" when the advice
	// does not point at one. It is the hook a later phase would act on; for
	// now it is what lets the panel put the advice beside the right control.
	Field string `json:"field,omitempty"`
	// Suggested is a value extracted from the engine's own wording, when it
	// offered one. Never inferred -- an empty string means vLLM did not say.
	Suggested string `json:"suggested,omitempty"`
	// Line is the source line, verbatim, so the user can see what was really
	// written rather than only our paraphrase of it.
	Line string `json:"line"`
}

// Measurements is what the engine reported about a start that actually
// happened. Zero values mean "not seen", which is why every field is a
// quantity vLLM only prints when it has one.
//
// Every field is a scalar and must stay one: Any compares the struct with ==,
// and a slice or map field would stop this package compiling.
type Measurements struct {
	// KVCacheGB is the pool vLLM claimed after placing the weights. The half
	// of the VRAM estimate that has never been checked against anything.
	KVCacheGB float64 `json:"kv_cache_gb,omitempty"`
	// WeightsPerRankGB is what one rank's weights actually took.
	WeightsPerRankGB float64 `json:"weights_per_rank_gb,omitempty"`
	LoadSeconds      float64 `json:"load_seconds,omitempty"`

	// MaxConcurrency is how many simultaneous requests at ConcurrencyTokens
	// the KV pool supports, as the engine computes it.
	MaxConcurrency    float64 `json:"max_concurrency,omitempty"`
	ConcurrencyTokens int     `json:"concurrency_tokens,omitempty"`

	GPUBlocks int `json:"gpu_blocks,omitempty"`
	CPUBlocks int `json:"cpu_blocks,omitempty"`

	// PLEOffloadGB is how much of the embedding table was pinned in host RAM,
	// and PLEOffloadFailed records the case this project has already hit: the
	// host's memlock limit refuses the lock and the offload silently does
	// nothing, while the estimate goes on assuming tens of gigabytes left the
	// cards.
	PLEOffloadGB     float64 `json:"ple_offload_gb,omitempty"`
	PLEOffloadFailed bool    `json:"ple_offload_failed,omitempty"`
	PLEOffloadWanted float64 `json:"ple_offload_wanted_gb,omitempty"`
}

// Any reports whether anything at all was captured.
func (m Measurements) Any() bool { return m != Measurements{} }

// Ready reports whether the line says the server finished starting.
//
// It lives here with the other line rules rather than as a loose pair of
// string comparisons in the process manager, so there is one place that knows
// what vLLM's output looks like.
func Ready(line string) bool {
	return strings.Contains(line, "Uvicorn running on") ||
		strings.Contains(line, "Application startup complete")
}

// Scan matches one log line, returning nil when it says nothing useful --
// which is almost every line, so that path is kept cheap: one lowercase copy
// and a substring test per rule before any expression is run.
func Scan(line string) *Item {
	if line == "" {
		return nil
	}
	lower := strings.ToLower(line)

	for _, r := range rules {
		if !strings.Contains(lower, r.hint) {
			continue
		}
		if item := r.match(line); item != nil {
			return item
		}
	}
	return nil
}

// ScanAll reads a whole transcript, for callers holding one rather than a
// stream: the context probe, and the post-mortem of a start that died.
//
// Repeats are dropped. A failing start prints the same complaint from every
// rank, and four copies of one message is not four problems.
func ScanAll(logs string) []Item {
	var out []Item
	seen := map[string]bool{}
	for _, line := range strings.Split(logs, "\n") {
		item := Scan(strings.TrimRight(line, "\r"))
		if item == nil {
			continue
		}
		key := string(item.Severity) + "\x00" + item.Message + "\x00" + item.Suggested
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, *item)
	}
	return out
}

// Observe folds whatever one line measures into m.
//
// Separate from Scan because the two accumulate differently: advice is a list
// that grows, measurements are fields that get filled in, and a single line
// can be both.
func Observe(m *Measurements, line string) {
	if m == nil || line == "" {
		return
	}
	lower := strings.ToLower(line)

	if strings.Contains(lower, "kv cache memory") {
		if v, ok := firstFloat(reKVCache, line); ok {
			m.KVCacheGB = v
		}
	}
	if strings.Contains(lower, "model loading took") {
		if v, ok := firstFloat(reModelLoad, line); ok {
			m.WeightsPerRankGB = v
		}
		if v, ok := firstFloat(reLoadSeconds, line); ok {
			m.LoadSeconds = v
		}
	}
	if strings.Contains(lower, "maximum concurrency") {
		if g := reConcurrency.FindStringSubmatch(line); g != nil {
			if n, err := strconv.Atoi(g[1]); err == nil {
				m.ConcurrencyTokens = n
			}
			if v, err := strconv.ParseFloat(g[2], 64); err == nil {
				m.MaxConcurrency = v
			}
		}
	}
	if strings.Contains(lower, "gpu blocks") {
		if g := reBlocks.FindStringSubmatch(line); g != nil {
			if n, err := strconv.Atoi(g[1]); err == nil {
				m.GPUBlocks = n
			}
			if n, err := strconv.Atoi(g[2]); err == nil {
				m.CPUBlocks = n
			}
		}
	}
	// One line carries both halves on a partial failure:
	//   PLE offload: locked 0.0 GiB, FAILED to lock 47.7 GiB
	if strings.Contains(lower, "locked") {
		if v, ok := firstFloat(reLocked, line); ok {
			m.PLEOffloadGB = v
		}
	}
	if strings.Contains(lower, "failed to lock") {
		m.PLEOffloadFailed = true
		if v, ok := firstFloat(reFailedLock, line); ok {
			m.PLEOffloadWanted = v
		}
	}
}

// OOM reports whether a transcript shows the engine running out of memory, and
// any max_model_len the engine suggested, taking the smallest such value as
// the most conservative.
//
// This is the entry point internal/benchmark's context probe uses. It lives
// here so there is one set of OOM patterns rather than two that drift.
func OOM(logs string) (oom bool, suggestedMaxLen int) {
	for _, re := range oomPatterns {
		if re.MatchString(logs) {
			oom = true
			break
		}
	}
	if !oom {
		return false, 0
	}
	for _, g := range reSuggestedMaxLen.FindAllStringSubmatch(logs, -1) {
		if n, err := strconv.Atoi(g[1]); err == nil && n > 0 {
			if suggestedMaxLen == 0 || n < suggestedMaxLen {
				suggestedMaxLen = n
			}
		}
	}
	return oom, suggestedMaxLen
}

type rule struct {
	// hint is a lowercase substring that must be present. It is the cheap
	// gate: nearly every log line fails it, and only then is an expression run.
	hint  string
	match func(line string) *Item
}

var rules = []rule{
	{
		// The engine states the exact ceiling it could not meet, which makes
		// this the one OOM case that suggests a value rather than a direction.
		hint: "is larger than the maximum number of tokens",
		match: func(line string) *Item {
			g := reSeqLenVsKV.FindStringSubmatch(line)
			if g == nil {
				return nil
			}
			return &Item{
				Severity:  Error,
				Message:   "The configured context is longer than the KV cache can hold. Lower it, or free VRAM for the cache.",
				Field:     "max_model_len",
				Suggested: g[2],
				Line:      line,
			}
		},
	},
	{
		// The pre-flight refusal, and the one that prompted this rule set
		// being rewritten: vLLM compares the fraction asked for against the
		// memory actually *free*, not the card's size. Anything else already
		// resident -- a leaked worker from a cancelled job, most likely --
		// makes an aggressive utilization unsatisfiable.
		hint: "free memory on device",
		match: func(line string) *Item {
			g := reFreeMemory.FindStringSubmatch(line)
			if g == nil {
				return nil
			}
			free, err1 := strconv.ParseFloat(g[1], 64)
			total, err2 := strconv.ParseFloat(g[2], 64)
			if err1 != nil || err2 != nil || total <= 0 {
				return nil
			}
			msg := "Another process is already holding memory on this GPU, so the engine could not reserve the fraction configured. Free it, or lower gpu_memory_utilization."
			item := &Item{
				Severity: Error,
				Message:  msg,
				Field:    "gpu_memory_utilization",
				Line:     line,
			}
			// Round down to a 0.05 step so every rank agrees on one number
			// and the panel shows a single suggestion rather than one per card.
			if step := math.Floor(free/total*20) / 20; step > 0 && step < 1 {
				item.Suggested = strconv.FormatFloat(step, 'f', 2, 64)
			}
			return item
		},
	},
	{
		// vLLM writes "GPU memory utilization" in prose and
		// "gpu_memory_utilization" in argument dumps, so the hint has to match
		// both. The first version gated on the underscore spelling and missed
		// the prose; the correction gated on the spaced spelling and missed the
		// dumps. Matching the one word common to both is the fix, and the
		// direction check below is what keeps it narrow.
		hint: "utilization",
		match: func(line string) *Item {
			lower := strings.ToLower(line)
			// Direction matters, and the first version of this rule assumed
			// one: it always advised raising the value. The engine asks for
			// the opposite at least as often, and telling someone to raise a
			// setting the engine just asked them to lower is worse than
			// staying quiet.
			switch {
			case strings.Contains(lower, "decrease"), strings.Contains(lower, "reduce"):
				return &Item{
					Severity: Error,
					Message:  "The engine asks for a lower gpu_memory_utilization -- it could not reserve the fraction configured.",
					Field:    "gpu_memory_utilization",
					Line:     line,
				}
			case strings.Contains(lower, "increas"):
				return &Item{
					Severity: Error,
					Message:  "The engine ran out of room and suggests raising gpu_memory_utilization. Check what else is on the cards first -- above about 0.95 there is nothing left to give.",
					Field:    "gpu_memory_utilization",
					Line:     line,
				}
			}
			return nil
		},
	},
	{
		hint: "worker",
		match: func(line string) *Item {
			lower := strings.ToLower(line)
			if !strings.Contains(lower, "failed to start") {
				return nil
			}
			return &Item{
				Severity: Error,
				Message:  "A worker process failed to start, so the engine never came up. The cause is in the lines above this one.",
				Line:     line,
			}
		},
	},
	{
		hint: "num_speculative_tokens",
		match: func(line string) *Item {
			if !strings.Contains(strings.ToLower(line), "acceptance rate") {
				return nil
			}
			return &Item{
				Severity: Warning,
				Message:  "More than one speculative token runs the draft layer repeatedly, which can lower the acceptance rate. Worth measuring against a lower count.",
				Field:    "speculative_config",
				Line:     line,
			}
		},
	},
	{
		hint: "data type to store kv cache",
		match: func(line string) *Item {
			return &Item{
				Severity: Info,
				Message:  "The KV cache is quantized, which halves its footprint and can cost a little accuracy without a proper scaling factor.",
				Field:    "kv_cache_dtype",
				Line:     line,
			}
		},
	},
	{
		hint: "cuda_visible_devices on rocm",
		match: func(line string) *Item {
			return &Item{
				Severity: Warning,
				Message:  "CUDA_VISIBLE_DEVICES is deprecated on ROCm. Set HIP_VISIBLE_DEVICES instead.",
				Field:    "env",
				Line:     line,
			}
		},
	},
	{
		// A flag the running image does not know. This project has been here:
		// a manifest bump that silently did nothing surfaced only as
		// "unrecognized arguments: --enable-expert-offload".
		hint: "unrecognized arguments",
		match: func(line string) *Item {
			g := reUnrecognized.FindStringSubmatch(line)
			if g == nil {
				return nil
			}
			return &Item{
				Severity:  Error,
				Message:   "The running image does not recognise " + g[1] + ". It may belong to a newer vLLM than the image carries.",
				Field:     "extra_flags",
				Suggested: g[1],
				Line:      line,
			}
		},
	},
	{
		hint: "out of memory",
		match: func(line string) *Item {
			return &Item{
				Severity: Error,
				Message:  "The GPU ran out of memory. Lower the context length, raise the tensor-parallel size, or offload more.",
				Field:    "max_model_len",
				Line:     line,
			}
		},
	},
	{
		hint: "failed to lock",
		match: func(line string) *Item {
			return &Item{
				Severity: Error,
				Message:  "Offload could not pin host memory, so the weights stayed on the GPUs. Raise the host's memlock limit -- rootless podman cannot raise it above the invoking user's hard limit.",
				Field:    "env",
				Line:     line,
			}
		},
	},
	{
		hint: "chunked prefill is enabled",
		match: func(line string) *Item {
			g := reChunkedPrefill.FindStringSubmatch(line)
			if g == nil {
				return nil
			}
			return &Item{
				Severity:  Info,
				Message:   "Chunked prefill is on, batching " + g[1] + " tokens per step. Activation memory scales with that, not with the context length.",
				Field:     "max_num_batched_tokens",
				Suggested: g[1],
				Line:      line,
			}
		},
	},
}

var (
	reKVCache     = regexp.MustCompile(`(?i)KV cache memory:?\s*([\d.]+)\s*GiB`)
	reModelLoad   = regexp.MustCompile(`(?i)model loading took\s*([\d.]+)\s*GiB`)
	reLoadSeconds = regexp.MustCompile(`(?i)and\s*([\d.]+)\s*seconds`)
	reConcurrency = regexp.MustCompile(`(?i)Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x`)
	reBlocks      = regexp.MustCompile(`(?i)GPU blocks:\s*([\d,]+),\s*CPU blocks:\s*([\d,]+)`)
	reLocked      = regexp.MustCompile(`(?i)locked\s*([\d.]+)\s*GiB`)
	reFailedLock  = regexp.MustCompile(`(?i)FAILED to lock\s*([\d.]+)\s*GiB`)
	reSeqLenVsKV  = regexp.MustCompile(`(?i)max seq len \((\d+)\).*?KV cache.*?\((\d+)\)`)
	// Free memory on device cuda:0 (27.28/31.86 GiB) on startup is less than
	// desired GPU memory utilization (0.97, 30.9 GiB).
	reFreeMemory     = regexp.MustCompile(`(?i)Free memory on device\s+\S+\s+\(([\d.]+)/([\d.]+)\s*GiB\)`)
	reUnrecognized   = regexp.MustCompile(`unrecognized arguments:\s*(\S+)`)
	reChunkedPrefill = regexp.MustCompile(`(?i)max_num_batched_tokens\s*[=:]\s*(\d+)`)

	// vLLM occasionally suggests "Try reducing max_model_len to N".
	reSuggestedMaxLen = regexp.MustCompile(`(?i)max[_ ]model[_ ]len.*?(\d{3,7})`)
)

// oomPatterns is the set moved from internal/benchmark, unchanged. The context
// probe's behaviour depends on exactly these matching exactly what they did.
var oomPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)torch\.cuda\.OutOfMemoryError`),
	regexp.MustCompile(`(?i)torch\.OutOfMemoryError`),
	regexp.MustCompile(`(?i)CUDA out of memory`),
	regexp.MustCompile(`(?i)HIP out of memory`),
	regexp.MustCompile(`(?i)RuntimeError:\s*out of memory`),
	regexp.MustCompile(`(?i)not enough memory`),
	regexp.MustCompile(`(?i)KV cache.*cannot fit`),
	regexp.MustCompile(`(?i)ValueError:\s*The model's max seq len .* is larger than the maximum`),
}

func firstFloat(re *regexp.Regexp, line string) (float64, bool) {
	g := re.FindStringSubmatch(line)
	if g == nil {
		return 0, false
	}
	v, err := strconv.ParseFloat(g[1], 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

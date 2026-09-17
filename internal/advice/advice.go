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

	// ConsumedGB is weights plus allocator overhead, PeakActivationGB the
	// working set at the configured batch, GraphPoolGB the captured graphs --
	// all per rank, and all reported together on one line.
	//
	// These are the three terms the structural estimate got wrong: activation
	// by a factor of twenty-three, the graph pool by two, and the allocator
	// overhead by not modelling it at all.
	ConsumedGB       float64 `json:"consumed_gb,omitempty"`
	NonTorchGB       float64 `json:"non_torch_gb,omitempty"`
	PeakActivationGB float64 `json:"peak_activation_gb,omitempty"`
	GraphPoolGB      float64 `json:"graph_pool_gb,omitempty"`

	// MaxConcurrency is how many simultaneous requests at ConcurrencyTokens
	// the KV pool supports, as the engine computes it.
	MaxConcurrency    float64 `json:"max_concurrency,omitempty"`
	ConcurrencyTokens int     `json:"concurrency_tokens,omitempty"`

	// KVCacheTokens is the whole pool's capacity, across every rank. With
	// KVCacheGB it gives the bytes a token really costs -- the figure that
	// turns "what if I halve the context" into arithmetic instead of a guess.
	KVCacheTokens int `json:"kv_cache_tokens,omitempty"`
	// KVCacheMemoryBytes is the value the engine offers for --kv-cache-memory
	// to reproduce this pool exactly. It hands over the config field's value
	// directly; nothing here has to derive it.
	KVCacheMemoryBytes int64 `json:"kv_cache_memory_bytes,omitempty"`

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

// KVBytesPerToken is what one token of context really costs across the whole
// pool, and 0 when the run did not report enough to say.
//
// tp is the tensor-parallel width the run used, because KVCacheGB is per rank
// while KVCacheTokens counts the pool as a whole.
//
// This is the most valuable thing a single start yields. Derived from the
// architecture it came out 2.61x low on a hybrid, because the recurrent layers
// hold state in the same pool at matched page sizes and the draft model brings
// its own -- none of which the attention-layer count can see. Measured, it is
// exact, and it is near enough a constant of the model and the KV dtype, so it
// re-answers the context-length question without another run.
func (m Measurements) KVBytesPerToken(tp int) float64 {
	if tp < 1 {
		tp = 1
	}
	if m.KVCacheGB <= 0 || m.KVCacheTokens <= 0 {
		return 0
	}
	return m.KVCacheGB * float64(tp) * (1024 * 1024 * 1024) / float64(m.KVCacheTokens)
}

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
		item := r.match(line)
		if item == nil {
			continue
		}
		// The PLE offload helper runs a small engine of its own, with its own
		// scheduler settings, and those are not the operator's configuration.
		// A live start reported "batching 2048 tokens per step" from the
		// helper beside the engine's real 8192, pointing at a config field
		// nobody had set to that. Its failures still matter; its housekeeping
		// does not.
		if item.Severity != Error && strings.Contains(line, "(PleOffloadWorker") {
			return nil
		}
		return item
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
			if n, ok := parseCount(g[1]); ok {
				m.ConcurrencyTokens = n
			}
			if v, err := strconv.ParseFloat(g[2], 64); err == nil {
				m.MaxConcurrency = v
			}
		}
	}
	// The pool's capacity. Written "GPU KV cache size: 651,081 tokens" -- an
	// earlier rule looked for "GPU blocks: N, CPU blocks: M", a line vLLM does
	// not print and nobody had checked.
	if strings.Contains(lower, "kv cache size") {
		if g := reKVCacheSize.FindStringSubmatch(line); g != nil {
			if n, ok := parseCount(g[1]); ok {
				m.KVCacheTokens = n
			}
		}
	}
	// One line, and it carries every term the structural estimate got wrong.
	if strings.Contains(lower, "for consumed memory") {
		if g := reActualUsage.FindStringSubmatch(line); g != nil {
			if v, err := strconv.ParseFloat(g[1], 64); err == nil {
				m.ConsumedGB = v
			}
			if v, err := strconv.ParseFloat(g[2], 64); err == nil {
				m.PeakActivationGB = v
			}
			if v, err := strconv.ParseFloat(g[3], 64); err == nil {
				m.GraphPoolGB = v
			}
		}
	}
	// The engine offers the exact value for --kv-cache-memory. Two appear on
	// the line: the first fits the requested budget, the second fills the card.
	// The conservative one is the one to keep.
	if strings.Contains(lower, "kv-cache-memory") {
		if g := reKVCacheMemory.FindStringSubmatch(line); g != nil {
			if n, err := strconv.ParseInt(g[1], 10, 64); err == nil {
				m.KVCacheMemoryBytes = n
			}
		}
	}
	if strings.Contains(lower, "non-torch") {
		if v, ok := firstFloat(reNonTorch, line); ok {
			m.NonTorchGB = v
		}
	}
	// Reported after capture, against the estimate the engine itself made.
	if strings.Contains(lower, "cuda graph pool memory") {
		if v, ok := firstFloat(reGraphPool, line); ok {
			m.GraphPoolGB = v
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
			// A successful start reports free memory too, in the same words up
			// to this clause. Without it the healthy case would raise an error
			// saying the engine could not start -- the same false positive
			// this rule set has now produced twice.
			if !strings.Contains(strings.ToLower(line), "less than desired") {
				return nil
			}
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
		// Informational, and it shipped in main as an error: the note contains
		// the word "increase", so every healthy start raised "the engine ran
		// out of room". It sits ahead of the direction rule so the specific
		// reading wins over the general one.
		// Gated on the clause that carries the number rather than on the
		// sentence that introduces it. vLLM prints both on one line today, and
		// a hint on the introduction alone would miss the value the moment
		// they are split -- which is a wording change away, not a redesign.
		hint: "--gpu-memory-utilization to",
		match: func(line string) *Item {
			item := &Item{
				Severity: Info,
				Message:  "Graph-capture memory is counted inside gpu_memory_utilization, so the configured fraction buys slightly less KV cache than it did before that accounting existed.",
				Field:    "gpu_memory_utilization",
				Line:     line,
			}
			if g := reProfilingEquiv.FindStringSubmatch(line); g != nil {
				item.Suggested = g[1]
			}
			return item
		},
	},
	{
		// The engine states the exact byte count that reproduces the pool it
		// just built. Nothing here has to derive it.
		hint: "--kv-cache-memory=",
		match: func(line string) *Item {
			g := reKVCacheMemory.FindStringSubmatch(line)
			if g == nil {
				return nil
			}
			return &Item{
				Severity:  Info,
				Message:   "The engine reports the exact KV pool it allocated. Pinning kv_cache_memory to it makes the split reproducible instead of dependent on what else is resident at startup.",
				Field:     "kv_cache_memory",
				Suggested: g[1],
				Line:      line,
			}
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
			// "try increasing" is the failure's phrasing. Bare "increase"
			// also appears in healthy informational notes, and matching it
			// turned every successful start into a reported error.
			case strings.Contains(lower, "try increas"):
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
	// The pool's capacity. An earlier rule looked for "GPU blocks: N, CPU
	// blocks: M", which vLLM does not print -- it was invented whole and
	// silently matched nothing.
	reKVCacheSize   = regexp.MustCompile(`(?i)GPU KV cache size:\s*([\d,]+)\s*tokens`)
	reNonTorch      = regexp.MustCompile(`(?i)non-torch\s*([\d.]+)\s*GiB`)
	reGraphPool     = regexp.MustCompile(`(?i)CUDA graph pool memory:\s*([\d.]+)\s*GiB\s*\(actual\)`)
	reKVCacheMemory = regexp.MustCompile(`--kv-cache-memory=(\d+)`)
	// Actual usage is 24.58 GiB for consumed memory (weights + non-torch),
	// 1.46 GiB for peak activation, and 0.49 GiB for CUDAGraph memory.
	reActualUsage = regexp.MustCompile(`(?i)Actual usage is\s*([\d.]+)\s*GiB for consumed memory.*?([\d.]+)\s*GiB for peak activation.*?([\d.]+)\s*GiB for CUDAGraph memory`)
	// The graph-accounting note, which is informational and not a failure.
	// The trailing \d matters: [\d.]+ alone swallows the sentence's full stop
	// and yields "0.9826." -- a value destined for a config field.
	reProfilingEquiv = regexp.MustCompile(`(?i)increase --gpu-memory-utilization to\s*(\d+(?:\.\d+)?)`)
	reLocked         = regexp.MustCompile(`(?i)locked\s*([\d.]+)\s*GiB`)
	reFailedLock     = regexp.MustCompile(`(?i)FAILED to lock\s*([\d.]+)\s*GiB`)
	reSeqLenVsKV     = regexp.MustCompile(`(?i)max seq len \((\d+)\).*?KV cache.*?\((\d+)\)`)
	// The refusal names the device:
	//   Free memory on device cuda:0 (27.28/31.86 GiB) on startup is less than
	//   desired GPU memory utilization (0.97, 30.9 GiB).
	// The post-start report does not:
	//   Free memory on device (31.23/31.86 GiB) on startup. Desired ...
	// Written against the refusal alone, this matched only half of them.
	reFreeMemory     = regexp.MustCompile(`(?i)Free memory on device\s+(?:cuda:\S+\s+)?\(([\d.]+)/([\d.]+)\s*GiB\)`)
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

// parseCount reads an integer that may carry thousands separators. vLLM writes
// "262,144 tokens" and "651,081 tokens", and Atoi rejects both -- which is why
// the concurrency figure silently came back as zero.
func parseCount(s string) (int, bool) {
	n, err := strconv.Atoi(strings.ReplaceAll(strings.TrimSpace(s), ",", ""))
	if err != nil {
		return 0, false
	}
	return n, true
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

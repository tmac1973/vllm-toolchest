package api

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
)

// The "This start" section of Engine Notes: what vLLM measured, said in words
// and in K, so nobody has to read the log and divide by hand.
//
// The question it was written to answer was "with 16 streams on a 512K
// context, how much can each one use?" The engine prints the pool's size and
// how many full-length requests fit; it never prints the per-stream share,
// which is the figure people actually plan with.

// summaryRow is one figure. Value is the figure itself and Note what it means,
// so the value column stays short enough to scan.
type summaryRow struct {
	Label string
	Value string
	Note  string
	// Ours marks a figure worked out here rather than printed by vLLM, as the
	// advice rows do.
	Ours bool
	// Warn tints the row: the figure is fine to read but worth acting on.
	Warn bool
}

type summarySection struct {
	Title string
	Rows  []summaryRow
}

// startSummary is the whole section, or nil when the engine has not yet said
// enough to fill any of it.
type startSummary struct {
	Sections []summarySection
	// Headline is the one-line version for the strip on a quiet start, where
	// "nothing to report" alone would hide the figure people come for.
	Headline string
}

// newStartSummary reads the run's measurements against the flags it was
// launched with. The flags rather than the stored config, because the stored
// config may have been edited since this start and the figures describe this
// start.
func newStartSummary(m advice.Measurements, args []string) *startSummary {
	// The flags alone are not a start: before the engine reports anything,
	// rows read from them would describe a run that may yet fail.
	if !m.Any() {
		return nil
	}
	var sum startSummary
	if s := contextSection(m, args, &sum.Headline); len(s.Rows) > 0 {
		sum.Sections = append(sum.Sections, s)
	}
	if s := memorySection(m, args); len(s.Rows) > 0 {
		sum.Sections = append(sum.Sections, s)
	}
	if s := engineSection(m, args); len(s.Rows) > 0 {
		sum.Sections = append(sum.Sections, s)
	}
	if len(sum.Sections) == 0 {
		return nil
	}
	return &sum
}

func contextSection(m advice.Measurements, args []string, headline *string) summarySection {
	sec := summarySection{Title: "Context"}
	if m.KVCacheTokens <= 0 {
		return sec
	}
	pool := m.KVCacheTokens

	poolNote := "Every running request draws from this, prompt and output alike."
	if m.Hybrid {
		poolNote += " This model also keeps recurrent state per request in the same pool, so each running request costs a little more than its tokens."
	}
	sec.Rows = append(sec.Rows, summaryRow{Label: "Shared pool", Value: tokensK(pool) + " tokens", Note: poolNote})

	maxLen := flagInt(args, "--max-model-len")
	if maxLen == 0 {
		maxLen = m.ConcurrencyTokens
	}
	if maxLen > 0 {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "Longest request",
			Value: tokensK(maxLen) + " tokens",
			Note:  "max_model_len: the most one request may use. Raising it does not make the pool bigger.",
		})
	}
	if m.MaxConcurrency > 0 && m.ConcurrencyTokens > 0 {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "Full-length requests at once",
			Value: strconv.FormatFloat(m.MaxConcurrency, 'f', -1, 64),
			Note:  fmt.Sprintf("How many %s-token requests fit in the pool side by side.", tokensK(m.ConcurrencyTokens)),
		})
	}

	seqs := flagInt(args, "--max-num-seqs")
	if seqs <= 0 {
		*headline = tokensK(pool) + " tokens of context"
		return sec
	}
	share := pool / seqs
	if maxLen > 0 && share > maxLen {
		share = maxLen
	}
	sec.Rows = append(sec.Rows, summaryRow{
		Label: fmt.Sprintf("Each of %d streams, all busy", seqs),
		Value: tokensK(share) + " tokens",
		Note:  "The pool split evenly at max_num_seqs. Space is handed out as streams grow, so one long session beside many short ones fits too.",
		Ours:  true,
	})
	*headline = fmt.Sprintf("%s tokens of context, %s each across %d streams", tokensK(pool), tokensK(share), seqs)

	if out := defaultMaxTokens(args); out > share {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "Default output limit",
			Value: tokensK(out) + " tokens",
			Note:  "Longer than a stream's even share. With every stream busy, a long reply can fill the pool; vLLM then pauses the newest request and runs it again later, which reads as a stall.",
			Ours:  true,
			Warn:  true,
		})
	}

	if hasFlag(args, "--enable-prefix-caching") && m.BlockTokens > 0 {
		note := "A shared prefix, such as a system prompt or an earlier turn, is reused only in whole blocks of this size."
		if m.Hybrid {
			note += " vLLM raised the block size so the attention pages line up with the model's recurrent-state pages."
		}
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "Prefix cache reuse",
			Value: "in " + tokensK(m.BlockTokens) + "-token blocks",
			Note:  note,
		})
	}
	return sec
}

func memorySection(m advice.Measurements, args []string) summarySection {
	sec := summarySection{Title: "Memory, per GPU"}
	if m.DeviceGB > 0 {
		v := gib(m.DeviceGB)
		note := ""
		if m.BudgetGB > 0 {
			note = fmt.Sprintf("vLLM may use %s of it (gpu_memory_utilization %s).", gib(m.BudgetGB), flagOr(args, "--gpu-memory-utilization", "default"))
		}
		sec.Rows = append(sec.Rows, summaryRow{Label: "Card", Value: v, Note: note})
	}
	if m.ConsumedGB > 0 {
		note := "Weights plus what the runtime holds outside PyTorch."
		if m.WeightsPerRankGB > 0 {
			note = fmt.Sprintf("Weights %s, the rest runtime overhead.", gib(m.WeightsPerRankGB))
		}
		sec.Rows = append(sec.Rows, summaryRow{Label: "Weights and runtime", Value: gib(m.ConsumedGB), Note: note})
	}
	if m.PeakActivationGB > 0 {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "Working memory",
			Value: gib(m.PeakActivationGB),
			Note:  "Peak activations for one step at max_num_batched_tokens " + flagOr(args, "--max-num-batched-tokens", "default") + ".",
		})
	}
	if m.GraphPoolGB > 0 {
		sec.Rows = append(sec.Rows, summaryRow{Label: "Captured graphs", Value: gib(m.GraphPoolGB)})
	}
	if m.KVCacheGB > 0 {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "KV cache",
			Value: gib(m.KVCacheGB),
			Note:  "What was left in the budget. This is the context pool above.",
		})
	}
	if m.PLEOffloadGB > 0 {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "In host RAM",
			Value: gib(m.PLEOffloadGB),
			Note:  "Embedding table pinned in system memory instead of on the cards.",
		})
	}
	if m.PLEOffloadFailed {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "Host RAM offload",
			Value: "failed",
			Note:  fmt.Sprintf("%s could not be pinned in system memory, so it stayed on the cards.", gib(m.PLEOffloadWanted)),
			Warn:  true,
		})
	}
	if len(sec.Rows) > 0 {
		// The ranks report separately and the figures here are whichever
		// arrived last, so they need not add up to the budget exactly.
		sec.Title += " (as one rank reported it)"
	}
	return sec
}

func engineSection(m advice.Measurements, args []string) summarySection {
	sec := summarySection{Title: "Engine"}
	if m.LoadSeconds > 0 {
		sec.Rows = append(sec.Rows, summaryRow{Label: "Loading weights", Value: duration(m.LoadSeconds)})
	}
	if m.InitSeconds > 0 {
		note := "Profiling, allocating the KV cache and warming up."
		if m.CompileSeconds > 0 {
			note = fmt.Sprintf("%s compiling, the rest profiling, allocating the KV cache and warming up.", duration(m.CompileSeconds))
		}
		sec.Rows = append(sec.Rows, summaryRow{Label: "Engine setup", Value: duration(m.InitSeconds), Note: note})
	}
	if m.ExpertsResident > 0 {
		sec.Rows = append(sec.Rows, summaryRow{
			Label: "Experts",
			Value: fmt.Sprintf("all %s on the GPUs", formatThousands(m.ExpertsResident)),
			Note:  "Expert offload switched itself off: every routed expert fits on each card.",
		})
	}
	// No note: the accuracy caveat for a quantized cache is already one of
	// the advice rows below.
	if v, ok := flagValue(args, "--kv-cache-dtype"); ok && v != "auto" {
		sec.Rows = append(sec.Rows, summaryRow{Label: "KV cache format", Value: v})
	}
	if v, ok := flagValue(args, "--speculative-config"); ok {
		var sc struct {
			Method string `json:"method"`
			N      int    `json:"num_speculative_tokens"`
		}
		if json.Unmarshal([]byte(v), &sc) == nil && sc.Method != "" {
			val := sc.Method
			if sc.N > 0 {
				val = fmt.Sprintf("%s, %d draft tokens", sc.Method, sc.N)
			}
			sec.Rows = append(sec.Rows, summaryRow{Label: "Speculative decoding", Value: val})
		}
	}
	if m.EngineVersion != "" {
		sec.Rows = append(sec.Rows, summaryRow{Label: "vLLM build", Value: m.EngineVersion})
	}
	return sec
}

// defaultMaxTokens is the completion limit a request gets when it names none,
// from the launch's --override-generation-config, or 0 when it sets none.
func defaultMaxTokens(args []string) int {
	g := generationOverride(args)
	switch {
	case g == nil:
		return 0
	case g.MaxTokens != nil:
		return *g.MaxTokens
	case g.MaxNewTokens != nil:
		return *g.MaxNewTokens
	}
	return 0
}

// tokensK writes a token count in K of 1024, the unit context lengths are
// named in: 524,288 is "512K", not "524K". Small counts stay exact.
func tokensK(n int) string {
	if n < 1024 {
		return strconv.Itoa(n)
	}
	k := float64(n) / 1024
	if k < 10 {
		return strconv.FormatFloat(math.Round(k*10)/10, 'f', -1, 64) + "K"
	}
	return formatThousands(int(math.Round(k))) + "K"
}

func gib(v float64) string {
	if v < 1 {
		return strconv.FormatFloat(v, 'f', 2, 64) + " GiB"
	}
	return strconv.FormatFloat(v, 'f', 1, 64) + " GiB"
}

func duration(sec float64) string {
	if sec < 60 {
		return strconv.FormatFloat(math.Round(sec*10)/10, 'f', -1, 64) + " s"
	}
	s := int(math.Round(sec))
	return fmt.Sprintf("%dm %02ds", s/60, s%60)
}

func formatThousands(n int) string {
	s := strconv.Itoa(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

func flagOr(args []string, name, fallback string) string {
	if v, ok := flagValue(args, name); ok {
		return v
	}
	return fallback
}

package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// LlamaBenchyReport is the top-level JSON object llama-benchy writes when
// invoked with --format json --save-result.
type LlamaBenchyReport struct {
	Version              string              `json:"version"`
	Timestamp            string              `json:"timestamp"`
	LatencyMode          string              `json:"latency_mode"`
	LatencyMs            float64             `json:"latency_ms"`
	Model                string              `json:"model"`
	PrefixCachingEnabled bool                `json:"prefix_caching_enabled"`
	MaxConcurrency       int                 `json:"max_concurrency"`
	Benchmarks           []LlamaBenchyResult `json:"benchmarks"`
}

// BenchyConfig captures the parameters needed to invoke llama-benchy.
type BenchyConfig struct {
	BaseURL         string
	APIKey          string
	ServedModelName string
	Tokenizer       string
	PromptSizes     []int
	GenSizes        []int
	Runs            int
	Concurrency     []int
	SaveResultPath  string

	// HFToken is forwarded as HF_TOKEN to lift HuggingFace's anonymous
	// rate limit. llama-benchy fetches the tokenizer on every invocation;
	// without a token, repeated runs hit 429s and the gpt2 fallback fires.
	HFToken string

	// HFHome is forwarded as HF_HOME so the tokenizer cache persists
	// across runs. Empty leaves the default (~/.cache/huggingface).
	HFHome string
}

// BuildBenchyArgs returns the argument vector passed to `uvx`. Pure
// function so the disclosure modal can render the same string the
// runner will execute. `--with sentencepiece tiktoken` pulls in the
// tokenizer backends needed for common HF formats (SentencePiece for
// LLaMA / Mistral, tiktoken for OpenAI-style). Without these, benchy
// silently falls back to GPT-2 and prompt-size accounting drifts.
func BuildBenchyArgs(c BenchyConfig) []string {
	args := []string{
		"--with", "sentencepiece",
		"--with", "tiktoken",
		"llama-benchy",
		"--base-url", c.BaseURL,
		"--api-key", c.APIKey,
		"--model", c.ServedModelName,
	}
	if c.Tokenizer != "" {
		args = append(args, "--tokenizer", c.Tokenizer)
	}
	for _, pp := range c.PromptSizes {
		args = append(args, "--pp", strconv.Itoa(pp))
	}
	for _, tg := range c.GenSizes {
		args = append(args, "--tg", strconv.Itoa(tg))
	}
	if c.Runs > 0 {
		args = append(args, "--runs", strconv.Itoa(c.Runs))
	}
	for _, cc := range c.Concurrency {
		args = append(args, "--concurrency", strconv.Itoa(cc))
	}
	args = append(args, "--format", "json", "--save-result", c.SaveResultPath)
	return args
}

// FormatBenchyCommand returns a shell-quoted, single-line representation
// of the uvx command for disclosure to the user. Used by the About modal
// and persisted on each run so the exact command that produced a result
// is forever reproducible.
func FormatBenchyCommand(c BenchyConfig) string {
	var b strings.Builder
	b.WriteString("uvx")
	for _, a := range BuildBenchyArgs(c) {
		b.WriteByte(' ')
		if strings.ContainsAny(a, " \t\"'\\$") {
			b.WriteString(strconv.Quote(a))
		} else {
			b.WriteString(a)
		}
	}
	return b.String()
}

// summarizeBenchy folds llama-benchy results into the BenchmarkSummary
// shape the run list view uses. Picks the concurrency=1 row if present
// (most directly comparable to internal API runs), otherwise the first.
func summarizeBenchy(results []LlamaBenchyResult) *BenchmarkSummary {
	if len(results) == 0 {
		return nil
	}
	pick := results[0]
	for _, r := range results {
		if r.Concurrency == 1 {
			pick = r
			break
		}
	}
	s := &BenchmarkSummary{}
	if pick.PPThroughput != nil {
		s.AvgPromptTokPerSec = pick.PPThroughput.Mean
	}
	if pick.TGThroughput != nil {
		s.AvgGenTokPerSec = pick.TGThroughput.Mean
		s.MinGenTokPerSec = pick.TGThroughput.Mean
		s.MaxGenTokPerSec = pick.TGThroughput.Mean
		for _, v := range pick.TGThroughput.Values {
			if v < s.MinGenTokPerSec {
				s.MinGenTokPerSec = v
			}
			if v > s.MaxGenTokPerSec {
				s.MaxGenTokPerSec = v
			}
		}
	}
	if pick.E2ETTFT != nil {
		s.AvgTTFTMs = pick.E2ETTFT.Mean
	} else if pick.TTFR != nil {
		s.AvgTTFTMs = pick.TTFR.Mean
	}
	return s
}

// runLlamaBenchy executes uvx llama-benchy and returns the parsed report.
// The command string is returned alongside the results so callers can
// store it on the BenchmarkRun for disclosure.
func runLlamaBenchy(ctx context.Context, c BenchyConfig) ([]LlamaBenchyResult, string, error) {
	if _, err := exec.LookPath("uvx"); err != nil {
		return nil, "", errors.New("uvx not found on PATH — install uv (https://astral.sh/uv) so llama-benchy can run inside the container")
	}

	if c.SaveResultPath == "" {
		f, err := os.CreateTemp("", "llama-benchy-*.json")
		if err != nil {
			return nil, "", fmt.Errorf("create result tempfile: %w", err)
		}
		c.SaveResultPath = f.Name()
		f.Close()
	}
	defer os.Remove(c.SaveResultPath)

	args := BuildBenchyArgs(c)
	cmdStr := FormatBenchyCommand(c)
	slog.Info("running llama-benchy", "command", cmdStr)

	cmd := exec.CommandContext(ctx, "uvx", args...)
	if c.HFToken != "" || c.HFHome != "" {
		env := os.Environ()
		if c.HFToken != "" {
			env = append(env, "HF_TOKEN="+c.HFToken)
		}
		if c.HFHome != "" {
			if err := os.MkdirAll(c.HFHome, 0o755); err != nil {
				return nil, cmdStr, fmt.Errorf("create HF_HOME dir %q: %w", c.HFHome, err)
			}
			env = append(env, "HF_HOME="+c.HFHome)
		}
		cmd.Env = env
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	start := time.Now()
	if err := cmd.Run(); err != nil {
		return nil, cmdStr, fmt.Errorf("llama-benchy exited with %w\nstderr: %s", err, stderr.String())
	}
	slog.Info("llama-benchy finished", "duration", time.Since(start))

	data, err := os.ReadFile(c.SaveResultPath)
	if err != nil {
		return nil, cmdStr, fmt.Errorf("read benchy result: %w", err)
	}
	var report LlamaBenchyReport
	if err := json.Unmarshal(data, &report); err != nil {
		return nil, cmdStr, fmt.Errorf("parse benchy result: %w", err)
	}
	if len(report.Benchmarks) == 0 {
		return nil, cmdStr, errors.New("llama-benchy produced no benchmark results")
	}
	return report.Benchmarks, cmdStr, nil
}

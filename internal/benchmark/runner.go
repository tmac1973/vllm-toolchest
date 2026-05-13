package benchmark

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// ProgressUpdate is one event in a run's lifecycle, fanned out to SSE
// subscribers and reflected in the run's ProgressDetail.
type ProgressUpdate struct {
	Stage  string `json:"stage"`  // "warmup" | "benchmark" | "done" | "error"
	Detail string `json:"detail"` // human-readable status line
	Pct    int    `json:"pct"`    // 0..100
}

// RunnerConfig holds everything the runner needs to execute one
// BenchmarkRun. The caller owns the BenchmarkRun and must Save it with
// StatusRunning before calling Run.
type RunnerConfig struct {
	Run        BenchmarkRun
	Preset     Preset
	VLLMURL    string // e.g. "http://127.0.0.1:8000"
	ServedName string // model identifier vLLM responds to ("model" field in /v1/chat/completions)
	MaxModelLen int   // for skip-rule when prompt_tokens+gen_tokens exceeds the model's context

	HFRepoID string // forwarded as --tokenizer to llama-benchy in Step 3
	HFToken  string
	HFHome   string
}

// Runner executes benchmarks. One Runner is shared across runs; concurrency
// control (single-run-at-a-time) lives in the Service layer.
type Runner struct {
	store *Store
}

// NewRunner creates a runner bound to a Store. The Store is updated as
// the run progresses (intermediate results, final summary).
func NewRunner(store *Store) *Runner {
	return &Runner{store: store}
}

// Run executes a benchmark. Sends progress updates to progress (closes on
// exit). The caller must have already persisted cfg.Run with
// StatusRunning so cancellation mid-run leaves a visible failed record.
func (r *Runner) Run(ctx context.Context, cfg RunnerConfig, progress chan<- ProgressUpdate) {
	run := cfg.Run
	startTime := time.Now()

	defer func() {
		run.DurationMs = time.Since(startTime).Milliseconds()
		if err := r.store.Save(run); err != nil {
			slog.Error("failed to save final run", "id", run.ID, "error", err)
		}
		if progress != nil {
			close(progress)
		}
	}()

	send := func(stage, detail string, pct int) {
		run.ProgressDetail = detail
		_ = r.store.Save(run)
		if progress != nil {
			select {
			case progress <- ProgressUpdate{Stage: stage, Detail: detail, Pct: pct}:
			default:
				// Drop updates if no subscriber is keeping up — progress
				// is best-effort. The run state on disk is canonical.
			}
		}
	}

	// Warmup with retry. vLLM may still be initializing kernels on first
	// request after a model load, so a few seconds of backoff is normal.
	send("warmup", "Warming up — sending small request to initialize kernels...", 5)
	var warmupErr error
	for attempt := 1; attempt <= 5; attempt++ {
		if ctx.Err() != nil {
			run.Status = StatusFailed
			run.Error = "cancelled during warmup"
			send("error", run.Error, 0)
			return
		}
		_, warmupErr = r.sendCompletionStream(ctx, cfg.VLLMURL, cfg.ServedName, 64, 16, 0)
		if warmupErr == nil {
			break
		}
		slog.Warn("benchmark warmup attempt failed, retrying", "attempt", attempt, "error", warmupErr)
		select {
		case <-ctx.Done():
			run.Status = StatusFailed
			run.Error = "cancelled during warmup"
			send("error", run.Error, 0)
			return
		case <-time.After(time.Duration(attempt*3) * time.Second):
		}
	}
	if warmupErr != nil {
		run.Status = StatusFailed
		run.Error = fmt.Sprintf("warmup failed after retries: %v", warmupErr)
		send("error", run.Error, 0)
		return
	}

	switch cfg.Preset.EffectiveSource() {
	case PresetSourceInternal:
		r.runInternal(ctx, &run, cfg, send)
	case PresetSourceBenchy:
		r.runBenchy(ctx, &run, cfg, send)
	default:
		run.Status = StatusFailed
		run.Error = fmt.Sprintf("unknown preset source: %q", cfg.Preset.Source)
		send("error", run.Error, 0)
		return
	}
}

// runBenchy shells out to `uvx llama-benchy` against the same /v1 endpoint
// vLLM is serving. The disclosed command and the parsed multi-concurrency
// report land on the run so the About modal can show what actually ran.
func (r *Runner) runBenchy(ctx context.Context, run *BenchmarkRun, cfg RunnerConfig, send func(stage, detail string, pct int)) {
	concurrency := cfg.Preset.Concurrency
	if len(concurrency) == 0 {
		concurrency = []int{1}
	}

	send("benchmark", "Running llama-benchy via uvx — output streams when finished...", 30)

	results, cmdStr, err := runLlamaBenchy(ctx, BenchyConfig{
		BaseURL:         cfg.VLLMURL + "/v1",
		APIKey:          "EMPTY",
		ServedModelName: cfg.ServedName,
		Tokenizer:       cfg.HFRepoID,
		PromptSizes:     cfg.Preset.PromptTokens,
		GenSizes:        []int{cfg.Preset.GenTokens},
		Runs:            cfg.Preset.Repetitions,
		Concurrency:     concurrency,
		HFToken:         cfg.HFToken,
		HFHome:          cfg.HFHome,
	})
	run.BenchyCommand = cmdStr
	if err != nil {
		run.Status = StatusFailed
		run.Error = err.Error()
		send("error", run.Error, 0)
		return
	}

	run.LlamaBenchy = results
	run.Summary = summarizeBenchy(results)
	run.Status = StatusCompleted
	send("done", "Benchmark complete", 100)
}

// runInternal executes the matrix of (prompt_tokens × repetitions) using
// streaming /v1/chat/completions requests against vLLM. Results are
// appended to run.Results as they complete; on cancel, partial results
// are preserved.
func (r *Runner) runInternal(ctx context.Context, run *BenchmarkRun, cfg RunnerConfig, send func(stage, detail string, pct int)) {
	// Apply skip rule: prompts that would exceed max_model_len are dropped
	// with a warning rather than erroring the whole run.
	promptLens := []int{}
	for _, pp := range cfg.Preset.PromptTokens {
		if cfg.MaxModelLen > 0 && pp+cfg.Preset.GenTokens > cfg.MaxModelLen {
			run.Warnings = append(run.Warnings,
				fmt.Sprintf("skipped prompt size %d: exceeds model max_model_len %d",
					pp, cfg.MaxModelLen))
			continue
		}
		promptLens = append(promptLens, pp)
	}
	if len(promptLens) == 0 {
		run.Status = StatusFailed
		run.Error = "no prompt sizes fit within the model's max_model_len"
		send("error", run.Error, 0)
		return
	}

	totalTests := len(promptLens) * cfg.Preset.Repetitions
	completed := 0
	var lastErr error

	for _, promptTokens := range promptLens {
		for rep := 1; rep <= cfg.Preset.Repetitions; rep++ {
			if ctx.Err() != nil {
				run.Status = StatusFailed
				run.Error = "cancelled"
				send("error", "Cancelled", 0)
				return
			}

			completed++
			pct := 10 + (completed*85)/totalTests
			send("benchmark",
				fmt.Sprintf("Testing %d prompt tokens × %d gen tokens (rep %d/%d)",
					promptTokens, cfg.Preset.GenTokens, rep, cfg.Preset.Repetitions),
				pct)

			result, err := r.runOneTest(ctx, cfg.VLLMURL, cfg.ServedName, promptTokens, cfg.Preset.GenTokens, rep)
			if err != nil {
				lastErr = err
				slog.Error("benchmark test failed", "prompt_tokens", promptTokens, "rep", rep, "error", err)
				continue
			}
			slog.Info("benchmark result",
				"prompt_tokens", result.PromptTokens, "gen_tokens", result.GenTokens,
				"ttft_ms", result.TTFTMs, "gen_tps", result.GenTokPerSec)
			run.Results = append(run.Results, *result)
			// Intermediate save so partial results survive a crash.
			_ = r.store.Save(*run)
		}
	}

	if len(run.Results) == 0 && lastErr != nil {
		run.Status = StatusFailed
		run.Error = fmt.Sprintf("all tests failed: %v", lastErr)
		send("error", run.Error, 0)
		return
	}

	run.Summary = ComputeSummary(run.Results)
	run.Status = StatusCompleted
	send("done", "Benchmark complete", 100)
}

// runOneTest sends a single streaming chat completion and parses its
// timing into a BenchmarkResult.
func (r *Runner) runOneTest(ctx context.Context, vllmURL, model string, promptTokens, genTokens, rep int) (*BenchmarkResult, error) {
	timings, err := r.sendCompletionStream(ctx, vllmURL, model, promptTokens, genTokens, rep)
	if err != nil {
		return nil, err
	}

	ttftMs := float64(timings.ttft.Milliseconds())
	totalMs := float64(timings.total.Milliseconds())
	genMs := totalMs - ttftMs
	if genMs < 1 {
		genMs = 1 // guard for tiny generations where TTFT ≈ total
	}

	var promptTPS, genTPS float64
	if ttftMs > 0 {
		promptTPS = float64(timings.promptTokens) / (ttftMs / 1000.0)
	}
	if timings.genTokens > 0 {
		genTPS = float64(timings.genTokens) / (genMs / 1000.0)
	}

	return &BenchmarkResult{
		PromptTokens:    timings.promptTokens,
		GenTokens:       timings.genTokens,
		Repetition:      rep,
		PromptTokPerSec: promptTPS,
		GenTokPerSec:    genTPS,
		TTFTMs:          ttftMs,
		TotalMs:         totalMs,
	}, nil
}

// streamTimings is the bag of measurements returned by a single streaming
// chat completion: time-to-first-chunk (TTFT), total wall time, and the
// token counts from the trailing usage chunk.
type streamTimings struct {
	ttft         time.Duration
	total        time.Duration
	promptTokens int
	genTokens    int
}

// sendCompletionStream issues a streaming chat completion and measures
// TTFT from the first SSE chunk. vLLM honors stream_options.include_usage
// and emits a final chunk with the usage block, which we use for accurate
// token counts.
func (r *Runner) sendCompletionStream(ctx context.Context, vllmURL, model string, promptTokens, genTokens, rep int) (*streamTimings, error) {
	prompt := buildPrompt(promptTokens, rep)
	reqBody, _ := json.Marshal(map[string]any{
		"model":       model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":  genTokens,
		"temperature": 0.0,
		"stream":      true,
		"stream_options": map[string]any{
			"include_usage": true,
		},
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, vllmURL+"/v1/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: 15 * time.Minute}
	startTime := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var (
		ttft         time.Duration
		firstChunk   = true
		usagePrompt  int
		usageGen     int
		sawAnyChunk  bool
	)

	scanner := bufio.NewScanner(resp.Body)
	// vLLM can emit large chunks (long sampled tokens, full usage blocks).
	scanner.Buffer(make([]byte, 1<<20), 1<<20)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		// First payload — record TTFT once we've seen real data.
		if firstChunk {
			ttft = time.Since(startTime)
			firstChunk = false
		}
		sawAnyChunk = true

		// The usage chunk arrives at the end with stream_options.include_usage.
		var chunk struct {
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err == nil && chunk.Usage != nil {
			usagePrompt = chunk.Usage.PromptTokens
			usageGen = chunk.Usage.CompletionTokens
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read stream: %w", err)
	}
	total := time.Since(startTime)

	if !sawAnyChunk {
		return nil, fmt.Errorf("no SSE chunks received from vLLM")
	}
	if usageGen == 0 {
		// vLLM should always emit a usage chunk when include_usage=true;
		// if it doesn't, surface that rather than reporting bogus numbers.
		return nil, fmt.Errorf("no usage in response; set stream_options.include_usage=true")
	}

	return &streamTimings{
		ttft:         ttft,
		total:        total,
		promptTokens: usagePrompt,
		genTokens:    usageGen,
	}, nil
}

// BenchPromptText is the deterministic prose passage repeated to fill
// prompts to a target token count. Identical to llama-toolchest so the
// engine-agnostic comparison via llama-benchy is truly apples-to-apples.
const BenchPromptText = `The history of computing is a story of human ingenuity and the relentless pursuit of automation. From the earliest mechanical calculators of the 17th century to the modern silicon chips that power our world, each generation has built upon the discoveries of the last. Charles Babbage conceived of the Analytical Engine in the 1830s, a mechanical general-purpose computer that, had it been built, would have contained many features of modern computers. Ada Lovelace, working with Babbage, wrote what is often considered the first computer program. The 20th century brought electronic computing into reality. Alan Turing formalized the concept of computation itself, while engineers at the University of Pennsylvania built ENIAC, one of the first electronic general-purpose computers. The invention of the transistor at Bell Labs in 1947 revolutionized electronics, leading to smaller, faster, and more reliable computers. The integrated circuit, developed independently by Jack Kilby and Robert Noyce, made it possible to place thousands and eventually billions of transistors on a single chip. This exponential growth in computing power, described by Moore's Law, has driven decades of innovation. Personal computers brought computing to the masses in the 1980s, the internet connected them in the 1990s, and smartphones made computing truly ubiquitous in the 2000s. Today, artificial intelligence and machine learning represent the latest frontier, with large language models demonstrating remarkable capabilities in understanding and generating human language. These models, trained on vast amounts of text data, can engage in conversation, write code, analyze documents, and assist with creative tasks. The computational requirements for training and running these models have driven advances in GPU computing, distributed systems, and specialized hardware accelerators. `

// BenchPromptPrefixTemplate is the per-repetition prefix that varies each
// prompt slightly to defeat vLLM's prefix cache.
const BenchPromptPrefixTemplate = "This is benchmark repetition number %d. Please analyze the following text carefully and provide a detailed response.\n\n"

// BenchPromptCharsPerToken is the chars-per-token approximation used when
// sizing prompts (English prose averages ~4 chars/token under most BPE
// tokenizers; actual counts come back from vLLM's usage field).
const BenchPromptCharsPerToken = 4

// buildPrompt constructs a prompt of approximately targetTokens by
// repeating the benchmark text. rep varies the prompt across runs.
func buildPrompt(targetTokens, rep int) string {
	targetChars := targetTokens * BenchPromptCharsPerToken
	var b strings.Builder
	b.WriteString(fmt.Sprintf(BenchPromptPrefixTemplate, rep))
	for b.Len() < targetChars {
		b.WriteString(BenchPromptText)
	}
	text := b.String()
	if len(text) > targetChars {
		text = text[:targetChars]
	}
	return text
}

package benchmark

import (
	"strings"
	"testing"
)

func TestBuildBenchyArgs(t *testing.T) {
	args := BuildBenchyArgs(BenchyConfig{
		BaseURL:         "http://localhost:8000/v1",
		APIKey:          "EMPTY",
		ServedModelName: "/data/models/test",
		Tokenizer:       "test/repo",
		PromptSizes:     []int{128, 512},
		GenSizes:        []int{64},
		Runs:            3,
		Concurrency:     []int{1, 4},
		SaveResultPath:  "/tmp/result.json",
	})

	// First two args must be the tokenizer backends so prompt sizing is correct.
	if args[0] != "--with" || args[1] != "sentencepiece" {
		t.Fatalf("expected --with sentencepiece first, got %v", args[:4])
	}

	joined := strings.Join(args, " ")
	for _, want := range []string{
		"llama-benchy",
		"--base-url http://localhost:8000/v1",
		"--api-key EMPTY",
		"--model /data/models/test",
		"--tokenizer test/repo",
		"--pp 128",
		"--pp 512",
		"--tg 64",
		"--runs 3",
		"--concurrency 1",
		"--concurrency 4",
		"--format json",
		"--save-result /tmp/result.json",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in args: %s", want, joined)
		}
	}
}

func TestBuildBenchyArgsOmitsEmpty(t *testing.T) {
	args := BuildBenchyArgs(BenchyConfig{
		BaseURL: "x", APIKey: "y", ServedModelName: "z",
		PromptSizes: []int{1}, GenSizes: []int{1},
		SaveResultPath: "/tmp/x",
	})
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--tokenizer") {
		t.Errorf("empty tokenizer should not produce --tokenizer arg; got %s", joined)
	}
	if strings.Contains(joined, "--runs") {
		t.Errorf("zero runs should not produce --runs arg; got %s", joined)
	}
}

func TestFormatBenchyCommandQuotesSpaces(t *testing.T) {
	cmd := FormatBenchyCommand(BenchyConfig{
		BaseURL: "x", APIKey: "y",
		ServedModelName: "model with spaces",
		PromptSizes:     []int{1}, GenSizes: []int{1}, SaveResultPath: "/tmp/x",
	})
	if !strings.HasPrefix(cmd, "uvx ") {
		t.Errorf("expected uvx prefix, got %s", cmd)
	}
	if !strings.Contains(cmd, `"model with spaces"`) {
		t.Errorf("expected quoted spaces in command, got %s", cmd)
	}
}

func TestSummarizeBenchyPicksConcurrencyOne(t *testing.T) {
	results := []LlamaBenchyResult{
		{
			Concurrency:  4,
			TGThroughput: &LlamaBenchyMetric{Mean: 100, Values: []float64{95, 105}},
			E2ETTFT:      &LlamaBenchyMetric{Mean: 200},
		},
		{
			Concurrency:  1,
			TGThroughput: &LlamaBenchyMetric{Mean: 50, Values: []float64{48, 52}},
			E2ETTFT:      &LlamaBenchyMetric{Mean: 80},
		},
	}
	summary := summarizeBenchy(results)
	if summary == nil {
		t.Fatal("expected non-nil summary")
	}
	if summary.AvgGenTokPerSec != 50 {
		t.Errorf("expected concurrency=1 row (50 t/s), got %f", summary.AvgGenTokPerSec)
	}
	if summary.AvgTTFTMs != 80 {
		t.Errorf("expected TTFT from concurrency=1 (80ms), got %f", summary.AvgTTFTMs)
	}
}

func TestSummarizeBenchyFallsBackToFirstResult(t *testing.T) {
	results := []LlamaBenchyResult{
		{Concurrency: 8, TGThroughput: &LlamaBenchyMetric{Mean: 200}},
	}
	summary := summarizeBenchy(results)
	if summary == nil || summary.AvgGenTokPerSec != 200 {
		t.Fatalf("expected fallback to first result (200), got %+v", summary)
	}
}

func TestSummarizeBenchyEmpty(t *testing.T) {
	if summarizeBenchy(nil) != nil {
		t.Error("expected nil summary for empty results")
	}
}

package api

import (
	"encoding/csv"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
)

func exportFixtureRuns() []benchmark.BenchmarkRun {
	created := time.Date(2026, 9, 7, 19, 31, 4, 0, time.UTC)
	return []benchmark.BenchmarkRun{
		{
			ID: "r1", JobID: "j1", CreatedAt: created, Status: "completed",
			ModelID: "unsloth/Qwen3.8-27B-FP8", ModelName: "Qwen3.8-27B-FP8",
			Quant: "fp8", SizeGB: 28.8, Preset: "internal-quick",
			SweepValues: map[string]string{"max_model_len": "8192"},
			Config: benchmark.ConfigSnapshot{
				MaxModelLen: 8192, TensorParallelSize: 2, GPUMemoryUtilization: 0.9,
				KVCacheDtype: "fp8", Dtype: "auto", MaxNumSeqs: 16,
			},
			GPUs: []benchmark.GPUSnapshot{{Index: 0, Name: "AMD Radeon RX 7900 XTX"}},
			Results: []benchmark.BenchmarkResult{
				{PromptTokens: 256, GenTokens: 128, Repetition: 1, PromptTokPerSec: 275.8, GenTokPerSec: 102.2, TTFTMs: 928, TotalMs: 2180},
				{PromptTokens: 1024, GenTokens: 128, Repetition: 1, PromptTokPerSec: 827.4, GenTokPerSec: 98.1, TTFTMs: 1104, TotalMs: 2410},
			},
			Summary: &benchmark.BenchmarkSummary{
				AvgGenTokPerSec: 100.15, MinGenTokPerSec: 98.1, MaxGenTokPerSec: 102.2,
				AvgPromptTokPerSec: 551.6, AvgTTFTMs: 1016,
			},
		},
		{
			// Failed: no summary, no results.
			ID: "r2", JobID: "j1", CreatedAt: created, Status: "failed",
			ModelID: "TheBloke/Mixtral-8x7B-AWQ", ModelName: "Mixtral-8x7B-AWQ",
			Quant: "awq", Preset: "internal-quick",
			Error:  "EngineCore initialization failed",
			Config: benchmark.ConfigSnapshot{MaxModelLen: 32768, TensorParallelSize: 1},
		},
	}
}

func parseCSV(t *testing.T, s string) [][]string {
	t.Helper()
	rows, err := csv.NewReader(strings.NewReader(s)).ReadAll()
	if err != nil {
		t.Fatalf("export is not valid CSV: %v", err)
	}
	return rows
}

func TestWriteCSVCellsHasARowPerTestPoint(t *testing.T) {
	var b strings.Builder
	cw := csv.NewWriter(&b)
	if err := writeCSVCells(cw, exportFixtureRuns()); err != nil {
		t.Fatal(err)
	}
	cw.Flush()

	rows := parseCSV(t, b.String())
	// Header plus the two test points of r1. r2 measured nothing, so it
	// contributes no cells.
	if len(rows) != 3 {
		t.Fatalf("got %d rows, want 3 (header + 2 test points)", len(rows))
	}
	header := rows[0]
	idx := map[string]int{}
	for i, h := range header {
		idx[h] = i
	}
	for _, want := range []string{"run_id", "sweep", "max_model_len", "prompt_tokens", "gen_tok_per_sec"} {
		if _, ok := idx[want]; !ok {
			t.Errorf("header is missing %q", want)
		}
	}
	if got := rows[1][idx["sweep"]]; got != "max_model_len=8192" {
		t.Errorf("sweep column = %q, want the swept value", got)
	}
	if got := rows[1][idx["prompt_tokens"]]; got != "256" {
		t.Errorf("first test point prompt_tokens = %q, want 256", got)
	}
	if got := rows[2][idx["prompt_tokens"]]; got != "1024" {
		t.Errorf("second test point prompt_tokens = %q, want 1024", got)
	}
}

// A run that failed still gets a summary row: knowing which configuration
// could not be measured is part of the result.
func TestWriteCSVSummaryKeepsFailedRuns(t *testing.T) {
	var b strings.Builder
	cw := csv.NewWriter(&b)
	if err := writeCSVSummary(cw, exportFixtureRuns()); err != nil {
		t.Fatal(err)
	}
	cw.Flush()

	rows := parseCSV(t, b.String())
	if len(rows) != 3 {
		t.Fatalf("got %d rows, want 3 (header + 2 runs)", len(rows))
	}
	idx := map[string]int{}
	for i, h := range rows[0] {
		idx[h] = i
	}
	if got := rows[1][idx["avg_gen_tok_per_sec"]]; got != "100.15" {
		t.Errorf("avg_gen_tok_per_sec = %q, want 100.15", got)
	}
	failed := rows[2]
	if failed[idx["run_id"]] != "r2" {
		t.Fatalf("second row is %q, want the failed run", failed[idx["run_id"]])
	}
	if got := failed[idx["avg_gen_tok_per_sec"]]; got != "" {
		t.Errorf("a failed run's average = %q, want empty rather than a zero that reads as a measurement", got)
	}
	if got := failed[idx["error"]]; got == "" {
		t.Error("a failed run should carry its error")
	}
}

// Both scopes share their identity columns, so a cells file and a summary file
// can be joined on the same keys.
func TestBothScopesShareTheirLeadingColumns(t *testing.T) {
	var cells, summary strings.Builder
	cw := csv.NewWriter(&cells)
	_ = writeCSVCells(cw, exportFixtureRuns())
	cw.Flush()
	sw := csv.NewWriter(&summary)
	_ = writeCSVSummary(sw, exportFixtureRuns())
	sw.Flush()

	c := parseCSV(t, cells.String())[0]
	s := parseCSV(t, summary.String())[0]
	for i := range runColumns {
		if c[i] != s[i] {
			t.Errorf("column %d differs: cells %q, summary %q", i, c[i], s[i])
		}
	}
}

// Runs recorded before a run carried its own conditions have no sweep values;
// the cell that produced them does.
func TestBackfillSweepValuesFromCells(t *testing.T) {
	job := &benchmark.BenchmarkJob{
		ID: "j1",
		Cells: []benchmark.JobCell{
			{BenchmarkRunID: "r1", SweepValues: map[string]string{"max_model_len": "8192"}},
			{BenchmarkRunID: "r2", SweepValues: map[string]string{"max_model_len": "32768"}},
		},
	}
	runs := []benchmark.BenchmarkRun{
		{ID: "r1"},
		{ID: "r2", SweepValues: map[string]string{"max_model_len": "65536"}},
	}

	got := backfillSweepValues(runs, job)
	if got[0].SweepValues["max_model_len"] != "8192" {
		t.Errorf("a run with no values should take the cell's: %v", got[0].SweepValues)
	}
	// A run that recorded its own conditions is the better source: it is what
	// actually ran, and the cell could have been retried since.
	if got[1].SweepValues["max_model_len"] != "65536" {
		t.Errorf("a run's own values must win over the cell's: %v", got[1].SweepValues)
	}
}

func TestExportBaseName(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"Quant compare", "quant-compare"},
		{"FP8 vs AWQ @ 32K", "fp8-vs-awq-32k"},
		{"  ", "job-j1"},
		{"///", "job-j1"},
		{strings.Repeat("a", 100), strings.Repeat("a", 60)},
	} {
		if got := exportBaseName(tc.in, "j1"); got != tc.want {
			t.Errorf("exportBaseName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

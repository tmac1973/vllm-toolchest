package api

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
)

// Benchmark results leave here for a spreadsheet or a notebook. Two CSV
// scopes, because the two questions are different:
//
//	cells   — one row per test point, for plotting how a run behaved across
//	          prompt sizes and repetitions.
//	summary — one row per run, for comparing runs against each other.
//
// The JSON export is the stored records verbatim, for anything the two shapes
// above do not answer.

// exportEnvelope wraps a JSON export with enough context to be read a year
// later without the UI that produced it.
type exportEnvelope struct {
	ExportedAt time.Time                `json:"exported_at"`
	Tool       string                   `json:"tool"`
	Version    string                   `json:"version,omitempty"`
	Job        *benchmark.BenchmarkJob  `json:"job,omitempty"`
	Runs       []benchmark.BenchmarkRun `json:"runs"`
}

// handleExportJob writes one job's results as CSV or JSON.
func (s *Server) handleExportJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	job, err := s.bench.GetJob(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	runs := backfillSweepValues(s.bench.RunsForJob(id), job)

	format := r.URL.Query().Get("format")
	if format == "" {
		format = "csv"
	}
	scope := r.URL.Query().Get("scope")
	if scope == "" {
		scope = "cells"
	}

	base := exportBaseName(job.Name, id)

	switch format {
	case "json":
		s.writeJSONExport(w, base+".json", exportEnvelope{
			ExportedAt: time.Now().UTC(),
			Tool:       "vllm-toolchest",
			Version:    s.versionLabel(),
			Job:        job,
			Runs:       runs,
		})
	case "csv":
		switch scope {
		case "cells":
			writeCSVExport(w, base+"-cells.csv", runs, writeCSVCells)
		case "summary":
			writeCSVExport(w, base+"-summary.csv", runs, writeCSVSummary)
		default:
			http.Error(w, fmt.Sprintf("unknown scope %q (want cells or summary)", scope), http.StatusBadRequest)
		}
	default:
		http.Error(w, fmt.Sprintf("unknown format %q (want csv or json)", format), http.StatusBadRequest)
	}
}

// backfillSweepValues fills in a run's sweep values from the cell that
// produced it. Runs recorded before a run carried its own conditions have
// none, and an export that silently drops the swept parameter is an export of
// numbers with no independent variable.
func backfillSweepValues(runs []benchmark.BenchmarkRun, job *benchmark.BenchmarkJob) []benchmark.BenchmarkRun {
	byRun := map[string]map[string]string{}
	for _, c := range job.Cells {
		if c.BenchmarkRunID != "" && len(c.SweepValues) > 0 {
			byRun[c.BenchmarkRunID] = c.SweepValues
		}
	}
	if len(byRun) == 0 {
		return runs
	}
	for i := range runs {
		if len(runs[i].SweepValues) == 0 {
			if v, ok := byRun[runs[i].ID]; ok {
				runs[i].SweepValues = v
			}
		}
	}
	return runs
}

func (s *Server) writeJSONExport(w http.ResponseWriter, filename string, env exportEnvelope) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", "attachment; filename="+strconv.Quote(filename))
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(env)
}

func writeCSVExport(w http.ResponseWriter, filename string, runs []benchmark.BenchmarkRun,
	write func(*csv.Writer, []benchmark.BenchmarkRun) error) {

	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", "attachment; filename="+strconv.Quote(filename))

	cw := csv.NewWriter(w)
	if err := write(cw, runs); err != nil {
		// The header is already sent, so there is no status left to change.
		// A trailing comment is the only honest signal available.
		fmt.Fprintf(w, "\n# export failed: %v\n", err)
		return
	}
	cw.Flush()
}

// runColumns are the identity and configuration columns both scopes share, so
// a cells file and a summary file line up on the same keys.
var runColumns = []string{
	"run_id", "job_id", "created_at", "status",
	"model_id", "model_name", "quant", "size_gb",
	"preset", "sweep",
	"max_model_len", "tensor_parallel_size", "gpu_memory_utilization",
	"max_num_seqs", "kv_cache_dtype", "enforce_eager", "dtype",
	"vllm_version", "gpus",
}

func runValues(run benchmark.BenchmarkRun) []string {
	c := run.Config
	return []string{
		run.ID, run.JobID, run.CreatedAt.UTC().Format(time.RFC3339), run.Status,
		run.ModelID, run.ModelName, run.Quant, formatFloat(run.SizeGB, 2),
		run.Preset, sweepValuesText(run.SweepValues),
		strconv.Itoa(c.MaxModelLen), strconv.Itoa(c.TensorParallelSize),
		formatFloat(c.GPUMemoryUtilization, 2),
		strconv.Itoa(c.MaxNumSeqs), c.KVCacheDtype, strconv.FormatBool(c.EnforceEager), c.Dtype,
		run.VLLMVersion, gpuNames(run.GPUs),
	}
}

func writeCSVCells(cw *csv.Writer, runs []benchmark.BenchmarkRun) error {
	header := append(append([]string{}, runColumns...),
		"prompt_tokens", "gen_tokens", "repetition",
		"prompt_tok_per_sec", "gen_tok_per_sec", "ttft_ms", "total_ms")
	if err := cw.Write(header); err != nil {
		return err
	}
	for _, run := range runs {
		base := runValues(run)
		for _, res := range run.Results {
			row := append(append([]string{}, base...),
				strconv.Itoa(res.PromptTokens), strconv.Itoa(res.GenTokens),
				strconv.Itoa(res.Repetition),
				formatFloat(res.PromptTokPerSec, 2), formatFloat(res.GenTokPerSec, 2),
				formatFloat(res.TTFTMs, 1), formatFloat(res.TotalMs, 1))
			if err := cw.Write(row); err != nil {
				return err
			}
		}
	}
	return cw.Error()
}

func writeCSVSummary(cw *csv.Writer, runs []benchmark.BenchmarkRun) error {
	header := append(append([]string{}, runColumns...),
		"avg_gen_tok_per_sec", "min_gen_tok_per_sec", "max_gen_tok_per_sec",
		"avg_prompt_tok_per_sec", "avg_ttft_ms", "test_points", "error")
	if err := cw.Write(header); err != nil {
		return err
	}
	for _, run := range runs {
		row := append(append([]string{}, runValues(run)...),
			"", "", "", "", "",
			strconv.Itoa(len(run.Results)), run.Error)
		// A run that failed still gets a row: knowing which configuration
		// could not be measured is part of the result.
		if s := run.Summary; s != nil {
			row[len(runColumns)+0] = formatFloat(s.AvgGenTokPerSec, 2)
			row[len(runColumns)+1] = formatFloat(s.MinGenTokPerSec, 2)
			row[len(runColumns)+2] = formatFloat(s.MaxGenTokPerSec, 2)
			row[len(runColumns)+3] = formatFloat(s.AvgPromptTokPerSec, 2)
			row[len(runColumns)+4] = formatFloat(s.AvgTTFTMs, 1)
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	return cw.Error()
}

// formatFloat writes a plain decimal — no exponent, no locale — because the
// destination is a spreadsheet.
func formatFloat(v float64, decimals int) string {
	return strconv.FormatFloat(v, 'f', decimals, 64)
}

func gpuNames(gpus []benchmark.GPUSnapshot) string {
	if len(gpus) == 0 {
		return ""
	}
	names := make([]string, 0, len(gpus))
	for _, g := range gpus {
		names = append(names, g.Name)
	}
	return strings.Join(names, " + ")
}

// exportBaseName turns a job name into something a filesystem will accept,
// falling back to the id when nothing usable survives.
func exportBaseName(name, id string) string {
	var b strings.Builder
	lastDash := true // suppress a leading dash
	for _, r := range strings.ToLower(name) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		default:
			if !lastDash {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		out = "job-" + id
	}
	const maxLen = 60
	if len(out) > maxLen {
		out = strings.Trim(out[:maxLen], "-")
	}
	return out
}

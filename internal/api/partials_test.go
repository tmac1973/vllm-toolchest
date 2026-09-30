package api

import (
	"bytes"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// presetChoice matches what the benchmark form template reads off a preset.
type presetChoice struct{ Name, Label string }

// The handler-level recordings in golden_test.go can only reach the states a
// local, deterministic Server can be put into: nothing running, nothing
// downloading, no results. That leaves the more interesting half of the
// markup — a live download's progress bar, a finished run's result table, a
// populated job — with no coverage at all.
//
// Now that those are partials, they can be rendered straight from view data,
// which is both deterministic (no uptime clock, no network) and a lot less
// setup than getting a Server into the matching state.
func TestGoldenPartials(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)

	cases := []struct {
		name    string
		partial string
		data    any
	}{
		{
			name:    "service_status_running",
			partial: "service_status",
			data: serviceStatusView{Status: process.Status{
				State: process.StateRunning, ModelID: "unsloth/Qwen3.8-27B-FP8", PID: 4242, Uptime: "13h59m45s",
			}, Settled: true},
		},
		{
			name:    "service_status_failed",
			partial: "service_status",
			data: serviceStatusView{Status: process.Status{
				State: process.StateError, ModelID: "unsloth/Qwen3.8-27B-FP8", Error: "engine core initialization failed",
			}, Settled: true},
		},
		{
			// A drafter for speculative decoding: listed because it is on
			// disk, with nothing on the card that would start or configure it.
			name:    "model_card_draft",
			partial: "model_card",
			data: modelRow{
				ID: "tcclaviger/Qwen3.8-27B-DFlash2-FP8", SafeID: "tcclaviger--Qwen3-8-27B-DFlash2-FP8",
				DisplayName: "Qwen3.8-27B-DFlash2-FP8",
				URL:         "https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8",
				Quant:       quantBadge{Method: "fp8"},
				SizeLabel:   "2.0 GB",
				Draft:       true, DraftMethod: "dflash",
				UsedBy:     []string{"ThinkingCap-3.8-27B-PARO5"},
				SearchText: "tcclaviger/qwen3.8-27b-dflash2-fp8 draft dflash",
			},
		},
		{
			// Past the startup timeout with the process alive. Not an error,
			// and it must not look like one: this is what a first start of a
			// large model shows for several minutes.
			name:    "service_status_overdue",
			partial: "service_status",
			data: serviceStatusView{Status: process.Status{
				State: process.StateStarting, ModelID: "unsloth/Qwen3.8-27B-FP8", PID: 4242, Uptime: "7m12s",
				Overdue: true,
				Notice: "Still starting after 5m0s, which is longer than the startup timeout. " +
					"Nothing has failed: the process is alive and being watched, and the log shows what it is doing. " +
					"A model's first start is often much slower than later ones. " +
					"This changes to Running when the engine answers.",
			}},
		},
		{
			// The engine gave up and its process did not exit.
			name:    "service_status_start_failed",
			partial: "service_status",
			data: serviceStatusView{Status: process.Status{
				State: process.StateStarting, ModelID: "unsloth/Qwen3.8-27B-FP8", PID: 4242, Uptime: "2m3s",
				StartFailed: true,
				Notice: "The engine reported that it failed to start, but its process has not exited. " +
					"The log says why. Stop clears it.",
			}},
		},
		{
			name:    "hf_results",
			partial: "hf_results",
			data: struct {
				Groups      []hfResultGroup
				QuantFilter string
			}{
				Groups: []hfResultGroup{{
					ID: "Qwen/Qwen3.8-27B", SafeID: "Qwen--Qwen3-8-27B",
					Author: "Qwen", Downloads: "1.2M", Likes: "3.4K", Gated: false,
					Variants: []hfResultVariant{
						{ID: "Qwen/Qwen3.8-27B", Format: "FP16", Color: "#555"},
						{ID: "unsloth/Qwen3.8-27B-FP8", Format: "FP8", Color: "#7a3db8"},
					},
				}, {
					// The search API often returns no author; the separator
					// has to go with it rather than leading the line.
					ID: "meta-llama/Llama-4-70B", SafeID: "meta-llama--Llama-4-70B",
					Author: "", Downloads: "980.0K", Likes: "2.1K", Gated: true,
					Variants: []hfResultVariant{
						{ID: "meta-llama/Llama-4-70B", Format: "FP16", Color: "#555"},
					},
				}},
			},
		},
		{
			name:    "hf_results_empty",
			partial: "hf_results",
			data: struct {
				Groups      []hfResultGroup
				QuantFilter string
			}{nil, "AWQ"},
		},
		{
			name:    "hf_model_detail",
			partial: "hf_model_detail",
			data: hfModelDetail{
				ID: "unsloth/Qwen3.8-27B-FP8", SafeID: "unsloth--Qwen3-8-27B-FP8",
				Architecture: "Qwen3MoeForCausalLM", QuantInfo: "FP8 8-bit",
				VRAMLabel: "~29.8 GB", SizeLabel: "28.8 GB", TotalBytes: 30_900_000_000,
				Files: []hfDetailFile{
					{Filename: "model-00001-of-00007.safetensors", SizeLabel: "4.6 GB", Category: "weight"},
					{Filename: "config.json", SizeLabel: "1.4 KB", Category: "config"},
				},
				AvailableBytes: 400_000_000_000, AvailableLabel: "372.5 GB",
				FreeLabel: "374.6 GB", MarginLabel: "2.0 GB", FitsOnDisk: true,
			},
		},
		{
			// Already in the registry: offering the button again would
			// silently re-fetch tens of gigabytes of something on disk.
			name:    "hf_model_detail_already_downloaded",
			partial: "hf_model_detail",
			data: hfModelDetail{
				ID: "unsloth/Qwen3.8-27B-FP8", SafeID: "unsloth--Qwen3-8-27B-FP8",
				Architecture: "Qwen3MoeForCausalLM", QuantInfo: "FP8 8-bit",
				VRAMLabel: "~29.8 GB", SizeLabel: "28.8 GB",
				AlreadyHave:    true,
				AvailableBytes: 400_000_000_000, AvailableLabel: "372.5 GB",
				FreeLabel: "374.6 GB", MarginLabel: "2.0 GB", FitsOnDisk: true,
			},
		},
		{
			// A stopped transfer left bytes behind, so the button continues
			// rather than implying a fresh start.
			name:    "hf_model_detail_resumable",
			partial: "hf_model_detail",
			data: hfModelDetail{
				ID: "unsloth/Qwen3.8-27B-FP8", SafeID: "unsloth--Qwen3-8-27B-FP8",
				Architecture: "Qwen3MoeForCausalLM", QuantInfo: "FP8 8-bit",
				VRAMLabel: "~29.8 GB", SizeLabel: "28.8 GB",
				Partial:        true,
				PartialLabel:   "12.1 GB",
				AvailableBytes: 400_000_000_000, AvailableLabel: "372.5 GB",
				FreeLabel: "374.6 GB", MarginLabel: "2.0 GB", FitsOnDisk: true,
			},
		},
		{
			// Bigger than the budget: refused with the numbers behind the
			// refusal, rather than failing partway through the transfer.
			name:    "hf_model_detail_wont_fit",
			partial: "hf_model_detail",
			data: hfModelDetail{
				ID: "meta-llama/Llama-4-70B", SafeID: "meta-llama--Llama-4-70B",
				Architecture: "Llama4ForCausalLM", QuantInfo: "FP16/BF16 (unquantized)",
				VRAMLabel: "~141.0 GB", SizeLabel: "140.0 GB",
				AvailableBytes: 8_000_000_000, AvailableLabel: "7.5 GB",
				FreeLabel: "9.5 GB", MarginLabel: "2.0 GB", FitsOnDisk: false,
			},
		},
		{
			// statfs gave no answer. Unknown must not gate anything: greying
			// out every button with no way to find out why is worse than
			// letting a download try.
			name:    "hf_model_detail_unknown_disk",
			partial: "hf_model_detail",
			data: hfModelDetail{
				ID: "unsloth/Qwen3.8-27B-FP8", SafeID: "unsloth--Qwen3-8-27B-FP8",
				Architecture: "Qwen3MoeForCausalLM", QuantInfo: "FP8 8-bit",
				VRAMLabel: "~29.8 GB", SizeLabel: "28.8 GB",
				AvailableBytes: -1, FitsOnDisk: true,
			},
		},
		{
			// Gated with no token, and too big for the first GPU: both
			// warnings, and the button disabled.
			name:    "hf_model_detail_blocked",
			partial: "hf_model_detail",
			data: hfModelDetail{
				ID: "meta-llama/Llama-4-70B", SafeID: "meta-llama--Llama-4-70B",
				Architecture: "Llama4ForCausalLM", QuantInfo: "FP16/BF16 (unquantized)",
				VRAMLabel: "~141.0 GB", SizeLabel: "140.0 GB",
				GatedWarning:   true,
				VRAMWarning:    "Estimated VRAM (141.0 GB) exceeds GPU memory (32 GB). Consider a quantized variant or TP=2.",
				Disabled:       true,
				AvailableBytes: 400_000_000_000, AvailableLabel: "372.5 GB",
				FreeLabel: "374.6 GB", MarginLabel: "2.0 GB", FitsOnDisk: true,
				Files: []hfDetailFile{{Filename: "model.safetensors", SizeLabel: "140.0 GB", Category: "weight"}},
			},
		},
		{
			name:    "download_progress_running",
			partial: "download_progress",
			data: downloadView{
				ID: "abc123", Status: "downloading", Percent: 42,
				DownloadedLabel: "12.1 GB", TotalLabel: "28.8 GB", SpeedLabel: "94.2 MB/s",
				CompletedFiles: 3, TotalFiles: 7,
			},
		},
		{
			name:    "download_progress_failed",
			partial: "download_progress",
			data:    downloadView{ID: "abc123", Status: "failed", Error: "unexpected EOF after 4.2 GB"},
		},
		{
			// One transfer running and one paused, which is the case the panel
			// exists for: a paused download that nothing showed would be disk
			// usage with no way to find or reclaim it.
			name:    "downloads_panel",
			partial: "downloads_panel",
			data: struct{ Rows []downloadRow }{[]downloadRow{
				{
					downloadView: downloadView{
						ID: "abc123", ModelID: "unsloth/Qwen3.8-27B-FP8", Status: "downloading",
						Percent: 42, DownloadedLabel: "12.1 GB", TotalLabel: "28.8 GB",
						SpeedLabel: "94.2 MB/s", CompletedFiles: 3, TotalFiles: 7,
					},
					Active: true,
				},
				{
					downloadView: downloadView{ModelID: "TheBloke/Mixtral-8x7B-AWQ"},
					OnDiskLabel:  "8.4 GB",
					PartFiles:    2,
				},
			}},
		},
		{
			// An update of a registered model, still hashing what it has:
			// the byte counts are disk reads, and the row has to say so.
			name:    "downloads_panel_update",
			partial: "downloads_panel",
			data: struct{ Rows []downloadRow }{[]downloadRow{{
				downloadView: downloadView{
					ID: "abc123", ModelID: "unsloth/Qwen3.8-27B-FP8", Status: "downloading",
					Percent: 12, DownloadedLabel: "3.4 GB", TotalLabel: "28.8 GB",
					SpeedLabel: "0 B/s", CompletedFiles: 1, TotalFiles: 7,
					VerifyingFiles: 3, Update: true,
				},
				Active: true,
			}}},
		},
		{
			// The case the panel exists for: a maintainer republished part of
			// a repo, and the rest of a copy made before files were tracked
			// can only be told apart by hashing.
			name:    "model_update",
			partial: "model_update",
			data: modelUpdateView{
				ID: "unsloth/Qwen3.8-27B-FP8", SafeID: "unsloth--Qwen3-8-27B-FP8",
				DisplayName:   "Qwen3.8-27B-FP8",
				Revision:      "90a83912b33232dd5f739720e90ac704ac21f07a",
				RevisionShort: "90a83912",
				CommitsURL:    "https://huggingface.co/unsloth/Qwen3.8-27B-FP8/commits/main",
				Rows: []modelUpdateRow{
					{Filename: "config.json", SizeLabel: "9.8 KB", State: "changed"},
					{Filename: "model-00001.safetensors", SizeLabel: "3.0 GB", State: "changed", Note: "1.2 GB already fetched"},
					{Filename: "extras/projector.safetensors", SizeLabel: "1.4 GB", State: "new"},
					{Filename: "model-00002.safetensors", SizeLabel: "5.1 GB", State: "to verify",
						Note: "same size as upstream; fetched only if its contents differ"},
				},
				CurrentCount: 3,
				FetchCount:   3, FetchLabel: "3.2 GB",
				VerifyCount: 1, VerifyLabel: "5.1 GB",
				Stale: []string{"model-mtp-old.safetensors"},
			},
		},
		{
			name:    "model_update_current",
			partial: "model_update",
			data: modelUpdateView{
				ID: "unsloth/Qwen3.8-27B-FP8", SafeID: "unsloth--Qwen3-8-27B-FP8",
				DisplayName:   "Qwen3.8-27B-FP8",
				Revision:      "90a83912b33232dd5f739720e90ac704ac21f07a",
				RevisionShort: "90a83912", SyncedShort: "90a83912",
				CommitsURL:   "https://huggingface.co/unsloth/Qwen3.8-27B-FP8/commits/main",
				CurrentCount: 7,
			},
		},
		{
			// Work to do and a reason it cannot start: the reason takes the
			// button's place rather than sitting beside one that will fail.
			name:    "model_update_blocked",
			partial: "model_update",
			data: modelUpdateView{
				ID: "unsloth/Qwen3.8-27B-FP8", SafeID: "unsloth--Qwen3-8-27B-FP8",
				DisplayName:   "Qwen3.8-27B-FP8",
				Revision:      "90a83912b33232dd5f739720e90ac704ac21f07a",
				RevisionShort: "90a83912", SyncedShort: "390783e4",
				Rows: []modelUpdateRow{
					{Filename: "model-00001.safetensors", SizeLabel: "3.0 GB", State: "changed"},
				},
				CurrentCount: 6,
				FetchCount:   1, FetchLabel: "3.0 GB",
				Blocked: "This model is being served. Stop the server before updating its files.",
			},
		},
		{
			// Nothing in flight and nothing half-finished: the panel renders
			// nothing at all rather than an empty card.
			name:    "downloads_panel_empty",
			partial: "downloads_panel",
			data:    struct{ Rows []downloadRow }{nil},
		},
		{
			name:    "benchmark_form_loaded",
			partial: "benchmark_form",
			data: struct {
				LoadedID   string
				LoadedName string
				Presets    []presetChoice
			}{
				LoadedID: "unsloth/Qwen3.8-27B-FP8", LoadedName: "Qwen3.8-27B-FP8",
				Presets: []presetChoice{
					{Name: "internal-quick", Label: "internal-quick — 1 rep, 256-token prompt"},
					{Name: "benchy-thorough", Label: "benchy-thorough — 3 reps"},
				},
			},
		},
		{
			// Built rather than hand-written: the template indexes Columns by
			// cell position, so a Cells/Columns mismatch is a render panic
			// and a fixture that hardcoded both could hide one.
			name:    "benchmark_compare",
			partial: "benchmark_compare",
			data: benchmark.BuildCompare([]benchmark.BenchmarkRun{
				{
					ID: "r1", ModelID: "unsloth/Qwen3.8-27B-FP8", ModelName: "Qwen3.8-27B-FP8",
					Quant: "fp8", Preset: "internal-standard",
					PromptTokens: []int{512}, GenTokens: 128, VLLMVersion: "0.1.dev1",
					Config:  benchmark.ConfigSnapshot{MaxModelLen: 8192, TensorParallelSize: 1, Dtype: "auto"},
					Summary: &benchmark.BenchmarkSummary{AvgGenTokPerSec: 46, AvgPromptTokPerSec: 980, AvgTTFTMs: 115},
				},
				{
					ID: "r2", ModelID: "unsloth/Qwen3.8-27B-FP8", ModelName: "Qwen3.8-27B-FP8",
					Quant: "fp8", Preset: "internal-standard",
					PromptTokens: []int{512}, GenTokens: 128, VLLMVersion: "0.1.dev1",
					Config:  benchmark.ConfigSnapshot{MaxModelLen: 65536, TensorParallelSize: 1, Dtype: "auto"},
					Summary: &benchmark.BenchmarkSummary{AvgGenTokPerSec: 31, AvgPromptTokPerSec: 1210, AvgTTFTMs: 155},
				},
				// A failed run has no numbers and still has to render.
				{
					ID: "r3", ModelID: "unsloth/Qwen3.8-27B-FP8", ModelName: "Qwen3.8-27B-FP8",
					Quant: "fp8", Preset: "internal-standard",
					Config: benchmark.ConfigSnapshot{MaxModelLen: 131072, TensorParallelSize: 1, Dtype: "auto"},
				},
			}),
		},
		{
			name:    "run_list",
			partial: "run_list",
			// Two groups, because grouping is the point of this list, and one
			// of them holds a run at a sweep point and a running one.
			data: runListView{RunCount: 3, Running: true, Groups: []runGroup{
				{
					Name: "Qwen3.8-27B-FP8",
					Rows: []runRow{
						{ID: "r1", ModelName: "Qwen3.8-27B-FP8", ModelID: "unsloth/Qwen3.8-27B-FP8",
							Quant: "fp8", Preset: "internal-thorough",
							PPTPS: "928", TGTPS: "102.2", TTFT: "928 ms",
							SweepText: "max_model_len=32768", Status: "completed",
							When: "Sep 7 19:31", Search: "qwen3.8-27b-fp8 fp8 internal-thorough"},
						{ID: "r3", ModelName: "Qwen3.8-27B-FP8", ModelID: "unsloth/Qwen3.8-27B-FP8",
							Quant: "fp8", Preset: "internal-quick",
							PPTPS: "—", TGTPS: "—", TTFT: "—",
							Status: "running", Running: true,
							When: "Sep 8 09:10", Search: "qwen3.8-27b-fp8 fp8 internal-quick"},
					},
				},
				{
					Name: "Mixtral-8x7B-AWQ",
					Rows: []runRow{
						{ID: "r2", ModelName: "Mixtral-8x7B-AWQ", ModelID: "TheBloke/Mixtral-8x7B-AWQ",
							Quant: "awq", Preset: "internal-quick",
							PPTPS: "612", TGTPS: "44.0", TTFT: "1204 ms",
							Status: "completed", When: "Sep 8 09:02",
							Search: "mixtral-8x7b-awq awq internal-quick"},
					},
				},
			}},
		},
		{
			name:    "run_detail",
			partial: "run_detail",
			data: struct {
				Preset     string
				CreatedAt  string
				Error      string
				Progress   string
				ConfigLine string
				Hardware   string
				VLLMVer    string
				PerSize    []benchmark.PerSizeStats
				Results    []runResultRow
				Summary    string
				Warnings   []string
			}{
				Preset: "internal-thorough", CreatedAt: "2026-09-07 19:31:04",
				ConfigLine: "max_model_len=32768, tp=2, gpu_mem_util=0.90, dtype=auto, kv_cache=fp8, eager=false, quant=fp8",
				Hardware:   "2 × AMD Radeon AI PRO R9700 (32 GB)",
				VLLMVer:    "0.1.dev1",
				// Two sizes, one with repetitions and one without, so both the
				// "± spread" and the bare-mean branches render.
				PerSize: benchmark.PerSize([]benchmark.BenchmarkResult{
					{PromptTokens: 256, PromptTokPerSec: 275.8, GenTokPerSec: 102.2, TTFTMs: 928},
					{PromptTokens: 256, PromptTokPerSec: 281.2, GenTokPerSec: 99.8, TTFTMs: 940},
					{PromptTokens: 1024, PromptTokPerSec: 927.5, GenTokPerSec: 98.1, TTFTMs: 1104},
				}),
				Results: []runResultRow{
					{N: 1, PromptTokens: 256, GenTokens: 128, TTFTMs: 928, TotalMs: 2180, PromptTokPerSec: 275.8, GenTokPerSec: 102.2},
					{N: 2, PromptTokens: 1024, GenTokens: 128, TTFTMs: 1104, TotalMs: 2410, PromptTokPerSec: 927.5, GenTokPerSec: 98.1},
				},
				Summary:  "avg gen 100.2 t/s (min 98.1, max 102.2), avg TTFT 1016 ms, avg prompt 601.7 t/s",
				Warnings: []string{"served model name was discovered, not configured"},
			},
		},
		{
			name:    "job_list",
			partial: "job_list",
			data: []jobRow{{
				ID: "j1", Name: "Quant compare", Description: "FP8 vs AWQ at 32K",
				Status: "completed", Done: 4, Total: 4,
				CreatedAt: "Sep 7 18:00", FinishedAt: "Sep 7 18:41",
			}, {
				ID: "j2", Name: "Long context sweep", Status: "running",
				Done: 1, Total: 6, Failed: 1, CreatedAt: "Sep 8 10:15",
			}, {
				// The catch-all pseudo-job counts runs, not cells.
				ID: "adhoc", Name: "Ad-Hoc Runs", Status: "completed",
				IsAdhoc: true, RunCount: 22, CreatedAt: "Apr 25 18:14",
			}},
		},
		{
			name:    "job_detail",
			partial: "job_detail",
			data: struct {
				jobRow
				Running      bool
				HasSweeps    bool
				ColSpan      int
				OverrideText string
				Rows         []jobCellRow
			}{
				jobRow: jobRow{
					ID: "j1", Name: "Quant compare", Status: "failed",
					Done: 1, Total: 2, Failed: 1, CreatedAt: "Sep 7 18:00",
				},
				ColSpan:      10,
				OverrideText: "max_model_len=32768 · tensor_parallel_size=2",
				Rows: []jobCellRow{
					{Idx: 0, ModelName: "Qwen3.8-27B-FP8", Quant: "fp8", Preset: "internal-quick",
						Status: "completed", TGTPS: "102.2", PPTPS: "928", TTFT: "928 ms",
						Attempt: 1, RunID: "r1"},
					{Idx: 1, ModelName: "Mixtral-8x7B-AWQ", Quant: "awq", Preset: "internal-quick",
						Status: "failed", Attempt: 2,
						TGTPS: "—", PPTPS: "—", TTFT: "—",
						Error:      "EngineCore initialization failed. See root cause above.\nTraceback (most recent call last):\n  File \"/opt/vllm/...\"",
						ErrorShort: "EngineCore initialization failed. See root cause abo…"},
				},
			},
		},
		{
			// A sweep grows a column, and the cells carry their swept values.
			name:    "job_detail_swept",
			partial: "job_detail",
			data: struct {
				jobRow
				Running      bool
				HasSweeps    bool
				ColSpan      int
				OverrideText string
				Rows         []jobCellRow
			}{
				jobRow:    jobRow{ID: "j3", Name: "Context sweep", Status: "running", Done: 1, Total: 3},
				Running:   true,
				HasSweeps: true,
				ColSpan:   11,
				Rows: []jobCellRow{
					{Idx: 0, ModelName: "Qwen3.8-27B-FP8", Quant: "fp8", Preset: "internal-quick",
						SweepText: "max_model_len=8192", Status: "completed",
						TGTPS: "110.4", PPTPS: "1204", TTFT: "412 ms", Attempt: 1, RunID: "r1"},
					{Idx: 1, ModelName: "Qwen3.8-27B-FP8", Quant: "fp8", Preset: "internal-quick",
						SweepText: "max_model_len=32768", Status: "running",
						TGTPS: "—", PPTPS: "—", TTFT: "—", Attempt: 1},
				},
			},
		},
		{
			name:    "probe_result",
			partial: "probe_result",
			data: struct {
				TPSize        int
				Timestamp     string
				ByUtilization []probeRow
				ByConcurrency []probeRow
			}{
				TPSize: 2, Timestamp: "2026-09-08 09:41",
				ByUtilization: []probeRow{
					{Label: "0.85", MaxContext: 40960, Vals: `{"gpu_memory_utilization":0.85,"max_model_len":40960,"model_id":"unsloth/Qwen3.8-27B-FP8"}`},
					{Label: "0.95", MaxContext: 65536, Vals: `{"gpu_memory_utilization":0.95,"max_model_len":65536,"model_id":"unsloth/Qwen3.8-27B-FP8"}`},
				},
				ByConcurrency: []probeRow{
					{Label: "1", MaxContext: 65536, Vals: `{"max_model_len":65536,"max_num_seqs":1,"model_id":"unsloth/Qwen3.8-27B-FP8"}`},
					{Label: "16", MaxContext: 32768, Vals: `{"max_model_len":32768,"max_num_seqs":16,"model_id":"unsloth/Qwen3.8-27B-FP8"}`},
				},
			},
		},
		{
			name:    "timings_list",
			partial: "timings_list",
			data: timingsView{
				MinSamples: 10,
				Rows: []dashboardTiming{
					{ModelID: "unsloth/Qwen3.8-27B-FP8", AvgGenTPS: 102.2, Count: 47, LastSeen: "Sep 8 10:58"},
				},
			},
		},
		{
			// A model still gathering samples must not render as "nothing
			// captured": that reads as broken when it is working.
			name:    "timings_list_pending",
			partial: "timings_list",
			data: timingsView{
				MinSamples: 10,
				Pending: []dashboardTiming{
					{ModelID: "unsloth/Qwen3.8-27B-FP8", Count: 6, LastSeen: "Sep 8 10:58"},
				},
			},
		},
		{
			name:    "timings_list_empty",
			partial: "timings_list",
			data:    timingsView{MinSamples: 10},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			s.renderPartial(&buf, tc.partial, tc.data)

			// renderPartial reports a missing template or a bad field as an
			// HTML comment rather than an error, so the comment is the signal.
			if bytes.Contains(buf.Bytes(), []byte("<!-- partial not found")) ||
				bytes.Contains(buf.Bytes(), []byte("<!-- render error")) {
				t.Fatalf("render failed: %s", buf.String())
			}
			assertGolden(t, "partial_"+tc.name, buf.String())
		})
	}
}

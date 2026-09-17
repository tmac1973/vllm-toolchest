package advice

import "testing"

// The lines here are real ones this project has hit, not invented shapes. Each
// case that names a checkpoint or a bug is one that cost somebody an afternoon.
func TestScan(t *testing.T) {
	for _, tc := range []struct {
		name      string
		line      string
		want      bool
		severity  Severity
		field     string
		suggested string
	}{
		{
			name: "ordinary log line says nothing",
			line: "INFO 09-16 11:03:17 [core.py:193] init engine (profile, create kv cache, warmup model) took 41.55 seconds",
		},
		{
			name: "empty",
			line: "",
		},
		{
			// The manifest-bump bug: the build ran on a stale base image and
			// the only symptom was this line.
			name:      "a flag the image does not know",
			line:      "vllm: error: unrecognized arguments: --enable-expert-offload --expert-offload-mem 46",
			want:      true,
			severity:  Error,
			field:     "extra_flags",
			suggested: "--enable-expert-offload",
		},
		{
			// The memlock failure on compute. Currently invisible to the
			// estimator, which goes on assuming the offload worked.
			name:     "offload could not pin host memory",
			line:     "PLE offload: locked 0.0 GiB, FAILED to lock 47.7 GiB",
			want:     true,
			severity: Error,
			field:    "env",
		},
		{
			name:      "context longer than the cache can hold",
			line:      "ValueError: The model's max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache (47328).",
			want:      true,
			severity:  Error,
			field:     "max_model_len",
			suggested: "47328",
		},
		{
			name:     "engine asks for more of the card",
			line:     "Try increasing gpu_memory_utilization when initializing the engine.",
			want:     true,
			severity: Error,
			field:    "gpu_memory_utilization",
		},
		{
			name:     "out of memory",
			line:     "torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB",
			want:     true,
			severity: Error,
			field:    "max_model_len",
		},
		{
			name:      "chunked prefill batch size",
			line:      "INFO Chunked prefill is enabled with max_num_batched_tokens=8192.",
			want:      true,
			severity:  Info,
			field:     "max_num_batched_tokens",
			suggested: "8192",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Scan(tc.line)
			if (got != nil) != tc.want {
				t.Fatalf("Scan(%q) = %v, want match=%v", tc.line, got, tc.want)
			}
			if got == nil {
				return
			}
			if got.Severity != tc.severity {
				t.Errorf("severity = %q, want %q", got.Severity, tc.severity)
			}
			if got.Field != tc.field {
				t.Errorf("field = %q, want %q", got.Field, tc.field)
			}
			if got.Suggested != tc.suggested {
				t.Errorf("suggested = %q, want %q", got.Suggested, tc.suggested)
			}
			if got.Line != tc.line {
				t.Error("the source line was not preserved verbatim")
			}
			if got.Message == "" {
				t.Error("matched but said nothing")
			}
		})
	}
}

// A failing start prints the same complaint from every rank. Four copies of
// one problem is not four problems.
func TestScanAllDeduplicates(t *testing.T) {
	logs := `INFO starting engine
(Worker_TP0) ERROR torch.cuda.OutOfMemoryError: CUDA out of memory
(Worker_TP1) ERROR torch.cuda.OutOfMemoryError: CUDA out of memory
(Worker_TP2) ERROR torch.cuda.OutOfMemoryError: CUDA out of memory
INFO shutting down`

	items := ScanAll(logs)
	if len(items) != 1 {
		t.Fatalf("got %d items, want 1 after deduplication: %+v", len(items), items)
	}
	if items[0].Severity != Error {
		t.Errorf("severity = %q, want error", items[0].Severity)
	}
}

// Every line in this fixture is copied verbatim from a real transcript, and
// must stay that way.
//
// An earlier version paired an invented cache size with a real token count,
// and the derived bytes-per-token came out at twice the truth -- computed
// confidently from two figures that never appeared together. That is the exact
// habit this package was reworked to end, and it had got inside the test
// written to prove the habit was over.
func TestObserve(t *testing.T) {
	var m Measurements
	if m.Any() {
		t.Error("a zero Measurements reports having seen something")
	}

	for _, line := range []string{
		"INFO [gpu_worker.py:701] Available KV cache memory: 4.87 GiB",
		"INFO [model_runner.py:415] Model loading took 19.07 GiB memory and 58.214395 seconds",
		// Real line and real numbers, thousands separators included -- Atoi
		// rejects those, which is how the concurrency figure came back zero.
		"INFO [kv_cache_utils.py:2393] GPU KV cache size: 651,081 tokens, Maximum concurrency for 262,144 tokens per request: 2.48x",
		"INFO [gpu_worker.py:935] Actual usage is 24.58 GiB for consumed memory (weights + non-torch), 1.46 GiB for peak activation, and 0.49 GiB for CUDAGraph memory.",
		"PLE offload: locked 38.8 GiB",
	} {
		Observe(&m, line)
	}

	if !m.Any() {
		t.Fatal("observed five measurements and reported none")
	}
	if m.KVCacheGB != 4.87 {
		t.Errorf("KVCacheGB = %v, want 4.87", m.KVCacheGB)
	}
	if m.WeightsPerRankGB != 19.07 {
		t.Errorf("WeightsPerRankGB = %v, want 19.07", m.WeightsPerRankGB)
	}
	if m.LoadSeconds != 58.214395 {
		t.Errorf("LoadSeconds = %v, want 58.214395", m.LoadSeconds)
	}
	if m.KVCacheTokens != 651081 {
		t.Errorf("KV pool = %d tokens, want 651081", m.KVCacheTokens)
	}
	if m.ConcurrencyTokens != 262144 || m.MaxConcurrency != 2.48 {
		t.Errorf("concurrency = %.2fx at %d tokens, want 2.48x at 262144",
			m.MaxConcurrency, m.ConcurrencyTokens)
	}
	if m.ConsumedGB != 24.58 || m.PeakActivationGB != 1.46 || m.GraphPoolGB != 0.49 {
		t.Errorf("consumed/activation/graphs = %.2f/%.2f/%.2f, want 24.58/1.46/0.49",
			m.ConsumedGB, m.PeakActivationGB, m.GraphPoolGB)
	}

	// The figure this whole approach turns on: derived from the architecture
	// it came out 12,288, and the engine's own allocation says 32,126.
	if b := m.KVBytesPerToken(4); b < 32000 || b > 32300 {
		t.Errorf("KV bytes/token = %.0f, want ~32126", b)
	}
	if m.PLEOffloadGB != 38.8 {
		t.Errorf("PLEOffloadGB = %v, want 38.8", m.PLEOffloadGB)
	}
	if m.PLEOffloadFailed {
		t.Error("a successful lock was recorded as a failure")
	}
}

// The partial-failure line carries both halves at once, and reading only the
// first number would record a successful 0.0 GiB offload.
func TestObservePartialOffloadFailure(t *testing.T) {
	var m Measurements
	Observe(&m, "PLE offload: locked 0.0 GiB, FAILED to lock 47.7 GiB")

	if !m.PLEOffloadFailed {
		t.Error("the failure to lock was not recorded")
	}
	if m.PLEOffloadWanted != 47.7 {
		t.Errorf("PLEOffloadWanted = %v, want 47.7", m.PLEOffloadWanted)
	}
	if m.PLEOffloadGB != 0 {
		t.Errorf("PLEOffloadGB = %v, want 0 -- nothing was actually pinned", m.PLEOffloadGB)
	}
}

// Moved wholesale from internal/benchmark. The context probe's behaviour
// depends on these matching exactly what they used to.
func TestOOM(t *testing.T) {
	for _, tc := range []struct {
		name string
		logs string
		want bool
	}{
		{"torch CUDA OOM", "RuntimeError: torch.cuda.OutOfMemoryError: blah", true},
		{"HIP OOM", "HIP out of memory. Tried to allocate 4 GiB.", true},
		{"explicit phrase", "CUDA out of memory", true},
		{"max seq len value error", "ValueError: The model's max seq len (8192) is larger than the maximum", true},
		{"benign info log", "INFO: vLLM is ready", false},
		{"empty", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got, _ := OOM(tc.logs); got != tc.want {
				t.Errorf("OOM(%q) = %v, want %v", tc.logs, got, tc.want)
			}
		})
	}
}

func TestOOMTakesTheSmallestSuggestion(t *testing.T) {
	_, suggested := OOM("CUDA out of memory. Try max_model_len to 16384. Or max_model_len 8192.")
	if suggested != 8192 {
		t.Errorf("suggested = %d, want the most conservative 8192", suggested)
	}
}

// Lines from a real failing start on the four-R9700 box, 2026-09-16. The first
// version of these rules matched one line out of eight and missed the actual
// failure, so the corpus here is the log itself rather than what the rules were
// written against.
func TestAgainstARealFailingStart(t *testing.T) {
	for _, tc := range []struct {
		name      string
		line      string
		want      bool
		severity  Severity
		field     string
		suggested string
	}{
		{
			// The failure. vLLM compares the fraction asked for against memory
			// actually free, not the card's size.
			name:      "pre-flight refusal states free and total",
			line:      `(Worker pid=44803) ERROR 09-16 20:50:38 [multiproc_executor.py:948] ValueError: Free memory on device cuda:0 (27.28/31.86 GiB) on startup is less than desired GPU memory utilization (0.97, 30.9 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.`,
			want:      true,
			severity:  Error,
			field:     "gpu_memory_utilization",
			suggested: "0.85",
		},
		{
			// The rule this replaced advised *raising* the value whatever the
			// engine said, which is the opposite of what this line asks for.
			name:     "the engine asking for less is not the engine asking for more",
			line:     "Decrease GPU memory utilization or reduce GPU memory used by other processes.",
			want:     true,
			severity: Error,
			field:    "gpu_memory_utilization",
		},
		{
			name:     "and the other direction still works",
			line:     "Try increasing gpu_memory_utilization when initializing the engine.",
			want:     true,
			severity: Error,
			field:    "gpu_memory_utilization",
		},
		{
			name:     "worker failed to start",
			line:     `(Worker pid=44803) ERROR 09-16 20:50:38 [multiproc_executor.py:948] WorkerProc failed to start.`,
			want:     true,
			severity: Error,
		},
		{
			name:     "speculative tokens above one",
			line:     `(APIServer pid=44393) WARNING 09-16 20:49:43 [speculative.py:1042] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer,which may result in lower acceptance rate`,
			want:     true,
			severity: Warning,
			field:    "speculative_config",
		},
		{
			name:     "quantized kv cache",
			line:     `(APIServer pid=44393) INFO 09-16 20:49:43 [cache.py:304] Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor`,
			want:     true,
			severity: Info,
			field:    "kv_cache_dtype",
		},
		{
			name:     "wrong visible-devices variable on ROCm",
			line:     `WARNING 09-16 20:49:52 [rocm.py:134] Using CUDA_VISIBLE_DEVICES on ROCm is deprecated and support will be removed in vLLM v0.26.0. Please use HIP_VISIBLE_DEVICES instead.`,
			want:     true,
			severity: Warning,
			field:    "env",
		},
		{
			// "entry.py" contains "try". An earlier version gated on that
			// substring and would have been one narrow hint away from
			// matching every traceback frame in the file.
			name: "a traceback frame is not advice",
			line: `(APIServer pid=44393)   File "/opt/vllm/lib/python3.14/site-packages/vllm/entrypoints/launchers/api_server/entry.py", line 176, in run_server`,
		},
		{
			name: "the banner is not advice",
			line: `(APIServer pid=44393) INFO 09-16 20:49:41 [api_utils.py:395] version 0.28.0  model /data/models/tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ  Clav Version 28.04.9`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Scan(tc.line)
			if (got != nil) != tc.want {
				t.Fatalf("match = %v, want %v\n  line: %.120s", got != nil, tc.want, tc.line)
			}
			if got == nil {
				return
			}
			if got.Severity != tc.severity {
				t.Errorf("severity = %q, want %q", got.Severity, tc.severity)
			}
			if got.Field != tc.field {
				t.Errorf("field = %q, want %q", got.Field, tc.field)
			}
			if tc.suggested != "" && got.Suggested != tc.suggested {
				t.Errorf("suggested = %q, want %q", got.Suggested, tc.suggested)
			}
		})
	}
}

// Every rank reports the refusal with its own free figure. Rounding the
// suggestion to a 0.05 step means they agree, so the panel shows one number
// rather than one per card.
func TestFreeMemorySuggestionAgreesAcrossRanks(t *testing.T) {
	logs := `ValueError: Free memory on device cuda:0 (27.28/31.86 GiB) on startup is less than desired GPU memory utilization (0.97, 30.9 GiB). Decrease GPU memory utilization.
ValueError: Free memory on device cuda:1 (27.54/31.86 GiB) on startup is less than desired GPU memory utilization (0.97, 30.9 GiB). Decrease GPU memory utilization.`

	items := ScanAll(logs)
	if len(items) != 1 {
		t.Fatalf("got %d items, want 1: %+v", len(items), items)
	}
	if items[0].Suggested != "0.85" {
		t.Errorf("suggested = %q, want 0.85", items[0].Suggested)
	}
}

// The successful start, which the previous rule set could not read at all: it
// matched one line in eight and mistook a healthy accounting note for a
// failure. A good start should yield measurements and almost no advice.
func TestSuccessfulStartYieldsMeasurementsNotAlarms(t *testing.T) {
	lines := []string{
		`(PleOffloadWorker pid=996) INFO [worker.py:222] PLE offload: locked 38.8 GiB of PLE weights in RAM`,
		`(Worker_TP2 pid=512) INFO [model_runner.py:415] Model loading took 19.07 GiB memory and 58.214395 seconds`,
		`(Worker_TP0 pid=510) INFO [gpu_worker.py:701] Available KV cache memory: 4.87 GiB`,
		`(EngineCore pid=384) INFO [kv_cache_utils.py:2393] GPU KV cache size: 651,081 tokens, Maximum concurrency for 262,144 tokens per request: 2.48x`,
		`(Worker_TP0 pid=510) INFO [gpu_worker.py:935] Free memory on device (31.23/31.86 GiB) on startup. Desired GPU memory utilization is (0.97, 30.9 GiB). Actual usage is 24.58 GiB for consumed memory (weights + non-torch), 1.46 GiB for peak activation, and 0.49 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with --kv-cache-memory=4536913634 (4.23 GiB) to fit into requested memory, or --kv-cache-memory=4887892992 (4.55 GiB) to fully utilize gpu memory.`,
		`(Worker_TP2 pid=512) INFO [gpu_worker.py:716] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9700 is equivalent to --gpu-memory-utilization=0.9574 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9826.`,
		`(Worker_TP1 pid=511) INFO [gpu_worker.py:872] CUDA graph pool memory: 0.49 GiB (actual), 0.4 GiB (estimated), difference: 0.09 GiB (18.6%).`,
		`(Worker_TP0 pid=510) INFO [mem_utils.py:363] memory after profile: torch reserved 22.7 GiB, torch allocated 21.59 GiB, non-torch 2.51 GiB (before load: non-torch 0.63 GiB)`,
	}

	var m Measurements
	for _, l := range lines {
		Observe(&m, l)
		// Nothing about a healthy start is an error. The rule set shipped one
		// anyway: the graph-accounting note contains the word "increase", so
		// every good start reported that the engine had run out of room.
		if it := Scan(l); it != nil && it.Severity == Error {
			t.Errorf("healthy line raised an error:\n  %.100s\n  -> %s", l, it.Message)
		}
	}

	// The three terms the structural estimate got wrong, all on one line.
	if m.ConsumedGB != 24.58 || m.PeakActivationGB != 1.46 || m.GraphPoolGB != 0.49 {
		t.Errorf("consumed/activation/graphs = %.2f/%.2f/%.2f, want 24.58/1.46/0.49",
			m.ConsumedGB, m.PeakActivationGB, m.GraphPoolGB)
	}
	if m.NonTorchGB != 2.51 {
		t.Errorf("non-torch = %v, want 2.51 -- the term that was not modelled at all", m.NonTorchGB)
	}
	// The point of the whole approach: derived from the architecture this was
	// 12,288, and the engine's own allocation says 32,126.
	if b := m.KVBytesPerToken(4); b < 32000 || b > 32300 {
		t.Errorf("KV bytes/token = %.0f, want ~32126", b)
	}
	// The engine hands over the config field's value directly.
	if m.KVCacheMemoryBytes != 4536913634 {
		t.Errorf("kv_cache_memory = %d, want the conservative 4536913634", m.KVCacheMemoryBytes)
	}
	if m.PLEOffloadGB != 38.8 || m.PLEOffloadFailed {
		t.Errorf("PLE offload = %v (failed=%v), want 38.8 and no failure", m.PLEOffloadGB, m.PLEOffloadFailed)
	}
}

// A value bound for a config field must be clean. [\d.]+ swallowed the
// sentence's full stop and produced "0.9826.".
func TestSuggestedValuesAreNotPunctuated(t *testing.T) {
	line := `To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9826.`
	it := Scan(line)
	if it == nil {
		t.Fatal("the graph-accounting note matched nothing")
	}
	if it.Severity != Info {
		t.Errorf("severity = %q, want info -- this is accounting, not a failure", it.Severity)
	}
	if it.Suggested != "0.9826" {
		t.Errorf("suggested = %q, want %q", it.Suggested, "0.9826")
	}
}

// The PLE offload helper runs a small engine of its own. A live start reported
// its scheduler's 2048 tokens alongside the real engine's 8192, pointing at a
// config field nobody had set to that value.
func TestTheOffloadHelpersHousekeepingIsNotAdvice(t *testing.T) {
	const engine = `(APIServer pid=109) INFO [scheduler.py:270] Chunked prefill is enabled with max_num_batched_tokens=8192.`
	const helper = `(PleOffloadWorker pid=996) INFO [scheduler.py:270] Chunked prefill is enabled with max_num_batched_tokens=2048.`

	it := Scan(engine)
	if it == nil || it.Suggested != "8192" {
		t.Fatalf("the engine's own setting was not reported: %+v", it)
	}
	if got := Scan(helper); got != nil {
		t.Errorf("the helper's scheduler was reported as advice about the model's config: %+v", got)
	}

	// Its failures are still its own, and still matter.
	const failure = `(PleOffloadWorker pid=996) ERROR torch.cuda.OutOfMemoryError: CUDA out of memory`
	if got := Scan(failure); got == nil || got.Severity != Error {
		t.Errorf("a real failure in the helper was suppressed: %+v", got)
	}

	// Measurements from the helper are still wanted -- the offload figure only
	// ever comes from it.
	var m Measurements
	Observe(&m, `(PleOffloadWorker pid=996) INFO [worker.py:222] PLE offload: locked 38.8 GiB of PLE weights in RAM`)
	if m.PLEOffloadGB != 38.8 {
		t.Errorf("PLE offload = %v, want 38.8 -- suppressing advice must not suppress measurement", m.PLEOffloadGB)
	}
}

func TestReady(t *testing.T) {
	for _, line := range []string{
		"INFO:     Uvicorn running on http://0.0.0.0:8000",
		"INFO:     Application startup complete.",
	} {
		if !Ready(line) {
			t.Errorf("Ready(%q) = false", line)
		}
	}
	if Ready("INFO: loading weights") {
		t.Error("an ordinary line was read as the ready signal")
	}
}

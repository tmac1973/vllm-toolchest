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

func TestObserve(t *testing.T) {
	var m Measurements
	if m.Any() {
		t.Error("a zero Measurements reports having seen something")
	}

	for _, line := range []string{
		"INFO [gpu_worker.py:298] Available KV cache memory: 10.34 GiB",
		"INFO [gpu_model_runner.py:2653] model loading took 19.07 GiB and 172.5 seconds",
		"INFO [kv_cache_utils.py:829] GPU blocks: 42301, CPU blocks: 0",
		"INFO [kv_cache_utils.py:833] Maximum concurrency for 262144 tokens per request: 2.48x",
		"PLE offload: locked 38.8 GiB",
	} {
		Observe(&m, line)
	}

	if !m.Any() {
		t.Fatal("observed five measurements and reported none")
	}
	if m.KVCacheGB != 10.34 {
		t.Errorf("KVCacheGB = %v, want 10.34", m.KVCacheGB)
	}
	if m.WeightsPerRankGB != 19.07 {
		t.Errorf("WeightsPerRankGB = %v, want 19.07", m.WeightsPerRankGB)
	}
	if m.LoadSeconds != 172.5 {
		t.Errorf("LoadSeconds = %v, want 172.5", m.LoadSeconds)
	}
	if m.GPUBlocks != 42301 || m.CPUBlocks != 0 {
		t.Errorf("blocks = %d/%d, want 42301/0", m.GPUBlocks, m.CPUBlocks)
	}
	if m.ConcurrencyTokens != 262144 || m.MaxConcurrency != 2.48 {
		t.Errorf("concurrency = %.2fx at %d tokens, want 2.48x at 262144",
			m.MaxConcurrency, m.ConcurrencyTokens)
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

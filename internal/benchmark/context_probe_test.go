package benchmark

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
)

// stubProbeEnv lets each test program its own spawn behavior. Records
// every attempt so we can assert exactly which contexts were tested.
type stubProbeEnv struct {
	hfMax    int
	attempts atomic.Int32
	respond  func(attempt ProbeAttempt) ProbeAttemptResult
	history  []ProbeAttempt
}

func (e *stubProbeEnv) ModelMaxPositionEmbeddings(modelID string) int { return e.hfMax }

func (e *stubProbeEnv) TrySpawn(ctx context.Context, modelID string, a ProbeAttempt) ProbeAttemptResult {
	e.attempts.Add(1)
	e.history = append(e.history, a)
	return e.respond(a)
}

// readyUntil returns a respond fn: every attempt with MaxModelLen <=
// maxOK succeeds; larger ones OOM. Useful for testing convergence.
func readyUntil(maxOK int) func(ProbeAttempt) ProbeAttemptResult {
	return func(a ProbeAttempt) ProbeAttemptResult {
		if a.MaxModelLen <= maxOK {
			return ProbeAttemptResult{Outcome: ProbeReady}
		}
		return ProbeAttemptResult{Outcome: ProbeOOM}
	}
}

func TestRunProbeConvergesToOOMBoundary(t *testing.T) {
	env := &stubProbeEnv{
		hfMax:   32768, // model ceiling
		respond: readyUntil(20480),
	}
	cfg := ProbeConfig{
		ModelID:           "test/m",
		TPSize:            1,
		UtilizationLevels: []float64{0.95}, // single utilization to keep test focused
		ConcurrencyLevels: []int{1},
		MinContext:        1024,
		MaxContext:        65536, // will be clamped to hfMax
	}
	progress := make(chan ProbeProgress, 256)
	go func() { for range progress {} }()

	result, err := RunProbe(context.Background(), env, cfg, progress)
	if err != nil {
		t.Fatal(err)
	}

	// Find the result for util=0.95.
	got, ok := result.UtilizationResults["0.95"]
	if !ok {
		t.Fatalf("expected util 0.95 in results, got %+v", result.UtilizationResults)
	}
	// Should converge to within probeContextGranularity (256) of 20480.
	if got < 20480-probeContextGranularity || got > 20480 {
		t.Errorf("expected ≈20480, got %d", got)
	}
}

func TestRunProbeRespectsHFCeiling(t *testing.T) {
	env := &stubProbeEnv{
		hfMax:   8192,
		respond: func(a ProbeAttempt) ProbeAttemptResult {
			// Always ready — would otherwise grow unbounded.
			return ProbeAttemptResult{Outcome: ProbeReady}
		},
	}
	cfg := ProbeConfig{
		ModelID:           "test/m",
		TPSize:            1,
		UtilizationLevels: []float64{0.95},
		ConcurrencyLevels: []int{1},
		MinContext:        1024,
		MaxContext:        131072,
	}
	progress := make(chan ProbeProgress, 256)
	go func() { for range progress {} }()
	result, _ := RunProbe(context.Background(), env, cfg, progress)

	got := result.UtilizationResults["0.95"]
	if got > 8192 {
		t.Errorf("probe should not exceed hfMax=8192; got %d", got)
	}
	// And every attempt must respect the ceiling too.
	for _, a := range env.history {
		if a.MaxModelLen > 8192 {
			t.Errorf("probed beyond hfMax: attempt %d > 8192", a.MaxModelLen)
		}
	}
}

func TestRunProbeUsesVLLMSuggestion(t *testing.T) {
	// vLLM suggests a max of 4096. The probe should narrow to that fast.
	env := &stubProbeEnv{
		hfMax: 32768,
		respond: func(a ProbeAttempt) ProbeAttemptResult {
			if a.MaxModelLen <= 4000 {
				return ProbeAttemptResult{Outcome: ProbeReady}
			}
			return ProbeAttemptResult{Outcome: ProbeOOM, SuggestedMaxLen: 4096}
		},
	}
	cfg := ProbeConfig{
		ModelID:           "test/m",
		TPSize:            1,
		UtilizationLevels: []float64{0.95},
		ConcurrencyLevels: []int{1},
		MinContext:        1024,
		MaxContext:        65536,
	}
	progress := make(chan ProbeProgress, 256)
	go func() { for range progress {} }()
	result, _ := RunProbe(context.Background(), env, cfg, progress)

	// Probe should hit the suggested 4096 boundary within ~5 attempts thanks
	// to the suggestion short-circuit.
	if env.attempts.Load() > 12 {
		t.Errorf("expected probe to converge in <12 attempts thanks to vLLM suggestion; got %d", env.attempts.Load())
	}
	got := result.UtilizationResults["0.95"]
	// Within 2 granularity steps of the true boundary (4000).
	if got < 4000-2*probeContextGranularity || got > 4000 {
		t.Errorf("expected within 2 granularity steps of 4000, got %d", got)
	}
}

func TestRunProbeMultipleUtilLevels(t *testing.T) {
	// Lower utilization → smaller max context. Three distinct levels.
	env := &stubProbeEnv{
		hfMax: 65536,
		respond: func(a ProbeAttempt) ProbeAttemptResult {
			limit := int(a.GPUMemoryUtilization * 30000) // arbitrary linear scale
			if a.MaxModelLen <= limit {
				return ProbeAttemptResult{Outcome: ProbeReady}
			}
			return ProbeAttemptResult{Outcome: ProbeOOM}
		},
	}
	cfg := ProbeConfig{
		ModelID:           "test/m",
		TPSize:            1,
		UtilizationLevels: []float64{0.98, 0.90, 0.80},
		ConcurrencyLevels: []int{1},
		MinContext:        1024,
		MaxContext:        65536,
	}
	progress := make(chan ProbeProgress, 1024)
	go func() { for range progress {} }()
	result, _ := RunProbe(context.Background(), env, cfg, progress)

	v98 := result.UtilizationResults["0.98"]
	v90 := result.UtilizationResults["0.90"]
	v80 := result.UtilizationResults["0.80"]

	// Each level should give a different max (higher util → higher max).
	if !(v98 > v90 && v90 > v80) {
		t.Errorf("expected v98 > v90 > v80; got %d, %d, %d", v98, v90, v80)
	}
}

func TestRunProbeConcurrencyDecreasesMaxContext(t *testing.T) {
	env := &stubProbeEnv{
		hfMax: 65536,
		respond: func(a ProbeAttempt) ProbeAttemptResult {
			// Higher max_num_seqs → less KV cache budget → smaller max context.
			ceil := 30000 / a.MaxNumSeqs
			if a.MaxModelLen <= ceil {
				return ProbeAttemptResult{Outcome: ProbeReady}
			}
			return ProbeAttemptResult{Outcome: ProbeOOM}
		},
	}
	cfg := ProbeConfig{
		ModelID:           "test/m",
		TPSize:            1,
		UtilizationLevels: []float64{0.95},
		ConcurrencyLevels: []int{1, 4, 16},
		MinContext:        1024,
		MaxContext:        65536,
	}
	progress := make(chan ProbeProgress, 1024)
	go func() { for range progress {} }()
	result, _ := RunProbe(context.Background(), env, cfg, progress)

	c1 := result.ConcurrencyResults[1]
	c4 := result.ConcurrencyResults[4]
	c16 := result.ConcurrencyResults[16]
	if !(c1 > c4 && c4 > c16) {
		t.Errorf("expected c1 > c4 > c16; got %d, %d, %d", c1, c4, c16)
	}
}

func TestDetectOOMPatterns(t *testing.T) {
	cases := []struct {
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
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := DetectOOM(tc.logs)
			if got != tc.want {
				t.Errorf("DetectOOM(%q): want %v, got %v", tc.logs, tc.want, got)
			}
		})
	}
}

func TestDetectOOMExtractsSuggestion(t *testing.T) {
	logs := "CUDA out of memory. Try reducing max_model_len to 4096 or lower."
	got, suggested := DetectOOM(logs)
	if !got {
		t.Fatal("expected OOM detected")
	}
	if suggested != 4096 {
		t.Errorf("expected suggested=4096, got %d", suggested)
	}
}

func TestDetectOOMPicksSmallestSuggestion(t *testing.T) {
	// vLLM sometimes prints multiple suggestions; we should take the smallest
	// for conservative narrowing.
	logs := "CUDA out of memory. Try max_model_len to 16384. Or max_model_len 8192."
	_, suggested := DetectOOM(logs)
	if suggested != 8192 {
		t.Errorf("expected smallest suggestion 8192, got %d", suggested)
	}
}

func TestClassifyLogs(t *testing.T) {
	if got := ClassifyLogs("torch.cuda.OutOfMemoryError"); got.Outcome != ProbeOOM {
		t.Errorf("expected OOM classification, got %s", got.Outcome)
	}
	if got := ClassifyLogs("INFO: some error happened"); got.Outcome != ProbeOther {
		t.Errorf("expected Other for error-but-not-OOM, got %s", got.Outcome)
	}
	if got := ClassifyLogs("normal log"); got.Outcome != ProbeOther {
		t.Errorf("expected Other for no-ready-no-error, got %s", got.Outcome)
	}
}

func TestRoundDown(t *testing.T) {
	if got := roundDown(1000, 256); got != 768 {
		t.Errorf("roundDown(1000, 256): want 768, got %d", got)
	}
	if got := roundDown(256, 256); got != 256 {
		t.Errorf("roundDown(256, 256): want 256, got %d", got)
	}
	if got := roundDown(255, 256); got != 0 {
		t.Errorf("roundDown(255, 256): want 0, got %d", got)
	}
}

func TestRunProbeRespectsContextCancellation(t *testing.T) {
	// Cancel from inside the respond fn so the cancellation is
	// deterministic with respect to the next ctx.Err() check.
	ctx, cancel := context.WithCancel(context.Background())
	env := &stubProbeEnv{
		hfMax: 65536,
		respond: func(a ProbeAttempt) ProbeAttemptResult {
			cancel() // cancel during the first spawn
			return ProbeAttemptResult{Outcome: ProbeReady}
		},
	}
	cfg := ProbeConfig{
		ModelID:           "test/m",
		UtilizationLevels: []float64{0.95, 0.90}, // 2 levels so the util-loop check fires on iter 2
		ConcurrencyLevels: []int{1, 4, 8},
		MinContext:        1024,
		MaxContext:        65536,
	}

	progress := make(chan ProbeProgress, 1024)
	go func() { for range progress {} }()
	result, err := RunProbe(ctx, env, cfg, progress)
	if err == nil {
		t.Fatal("expected context.Canceled error")
	}
	if result == nil {
		t.Fatal("expected partial result even on cancel")
	}
	hasWarning := false
	for _, w := range result.Warnings {
		if strings.Contains(w, "cancel") {
			hasWarning = true
			break
		}
	}
	if !hasWarning {
		t.Errorf("expected cancel warning in result, got %+v", result.Warnings)
	}
}

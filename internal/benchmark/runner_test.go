package benchmark

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeVLLM is a minimal SSE responder shaped like vLLM's
// /v1/chat/completions stream. The first chunk arrives after firstDelay,
// the rest after followDelay each, and a final usage chunk closes things out.
type fakeVLLM struct {
	firstDelay  time.Duration
	followDelay time.Duration
	numChunks   int
	usagePrompt int
	usageGen    int
}

func (f *fakeVLLM) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)

		emit := func(payload string) {
			fmt.Fprintf(w, "data: %s\n\n", payload)
			flusher.Flush()
		}

		time.Sleep(f.firstDelay)
		for i := 0; i < f.numChunks; i++ {
			if i > 0 {
				time.Sleep(f.followDelay)
			}
			emit(`{"choices":[{"delta":{"content":"tok"}}]}`)
		}
		// Final usage chunk.
		emit(fmt.Sprintf(
			`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
			f.usagePrompt, f.usageGen, f.usagePrompt+f.usageGen))
		emit("[DONE]")
	}
}

func TestSendCompletionStream(t *testing.T) {
	fv := &fakeVLLM{
		firstDelay:  50 * time.Millisecond,
		followDelay: 5 * time.Millisecond,
		numChunks:   10,
		usagePrompt: 128,
		usageGen:    32,
	}
	srv := httptest.NewServer(fv.handler())
	defer srv.Close()

	r := &Runner{}
	timings, err := r.sendCompletionStream(context.Background(), srv.URL, "test-model", 128, 32, 1)
	if err != nil {
		t.Fatal(err)
	}

	if timings.promptTokens != 128 || timings.genTokens != 32 {
		t.Fatalf("token counts: got prompt=%d gen=%d", timings.promptTokens, timings.genTokens)
	}
	// TTFT should be at least firstDelay (50ms) and well below total.
	if timings.ttft < 40*time.Millisecond {
		t.Errorf("TTFT suspiciously low: %v", timings.ttft)
	}
	if timings.ttft >= timings.total {
		t.Errorf("TTFT (%v) should be < total (%v)", timings.ttft, timings.total)
	}
}

func TestSendCompletionStreamNoUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		fmt.Fprintf(w, `data: {"choices":[{"delta":{"content":"x"}}]}`+"\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	r := &Runner{}
	_, err := r.sendCompletionStream(context.Background(), srv.URL, "test-model", 8, 4, 1)
	if err == nil || !strings.Contains(err.Error(), "no usage") {
		t.Fatalf("expected 'no usage' error, got %v", err)
	}
}

func TestSendCompletionStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not loaded", http.StatusBadRequest)
	}))
	defer srv.Close()

	r := &Runner{}
	_, err := r.sendCompletionStream(context.Background(), srv.URL, "x", 8, 4, 1)
	if err == nil || !strings.Contains(err.Error(), "HTTP 400") {
		t.Fatalf("expected HTTP 400, got %v", err)
	}
}

func TestBuildPromptApproximateLength(t *testing.T) {
	for _, target := range []int{128, 512, 2048} {
		got := buildPrompt(target, 1)
		expectChars := target * BenchPromptCharsPerToken
		if len(got) != expectChars {
			t.Errorf("buildPrompt(%d): expected %d chars, got %d", target, expectChars, len(got))
		}
		if !strings.HasPrefix(got, "This is benchmark repetition number 1.") {
			t.Errorf("buildPrompt(%d): missing repetition prefix; got first 50 chars: %q",
				target, got[:min(50, len(got))])
		}
	}
}

func TestBuildPromptVariesByRepetition(t *testing.T) {
	a := buildPrompt(512, 1)
	b := buildPrompt(512, 2)
	if a == b {
		t.Fatal("expected different prompts for different repetitions to defeat prefix caching")
	}
}

func TestRunnerInternalEndToEnd(t *testing.T) {
	fv := &fakeVLLM{
		firstDelay:  20 * time.Millisecond,
		followDelay: 2 * time.Millisecond,
		numChunks:   16,
		usagePrompt: 64,
		usageGen:    16,
	}
	srv := httptest.NewServer(fv.handler())
	defer srv.Close()

	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	store := NewStore(dir)
	runner := NewRunner(store)

	run := BenchmarkRun{
		ID:        "run-x",
		CreatedAt: time.Now(),
		Status:    StatusRunning,
		ModelID:   "test/model",
		Preset:    "internal-quick",
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	cfg := RunnerConfig{
		Run: run,
		Preset: Preset{
			Source:       PresetSourceInternal,
			PromptTokens: []int{64},
			GenTokens:    16,
			Repetitions:  2,
		},
		VLLMURL:     srv.URL,
		ServedName:  "test/model",
		MaxModelLen: 4096,
	}

	progress := make(chan ProgressUpdate, 32)
	done := make(chan struct{})
	go func() {
		runner.Run(context.Background(), cfg, progress)
		close(done)
	}()

	stages := map[string]int{}
	for u := range progress {
		stages[u.Stage]++
	}
	<-done

	if stages["warmup"] == 0 {
		t.Error("expected at least one warmup progress event")
	}
	if stages["done"] == 0 {
		t.Error("expected a done progress event")
	}

	final, err := store.Get("run-x")
	if err != nil {
		t.Fatal(err)
	}
	if final.Status != StatusCompleted {
		t.Fatalf("expected completed, got %s (error=%q)", final.Status, final.Error)
	}
	if len(final.Results) != 2 {
		t.Fatalf("expected 2 results (1 prompt × 2 reps), got %d", len(final.Results))
	}
	if final.Summary == nil || final.Summary.AvgGenTokPerSec <= 0 {
		t.Fatalf("expected non-zero summary, got %+v", final.Summary)
	}
}

func TestRunnerSkipsPromptOverMaxModelLen(t *testing.T) {
	fv := &fakeVLLM{
		firstDelay:  5 * time.Millisecond,
		followDelay: 1 * time.Millisecond,
		numChunks:   4,
		usagePrompt: 64, usageGen: 8,
	}
	srv := httptest.NewServer(fv.handler())
	defer srv.Close()

	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	store := NewStore(dir)
	runner := NewRunner(store)

	run := BenchmarkRun{ID: "skip-r", CreatedAt: time.Now(), Status: StatusRunning}
	store.Save(run)

	cfg := RunnerConfig{
		Run: run,
		Preset: Preset{
			Source:       PresetSourceInternal,
			PromptTokens: []int{64, 9999}, // second exceeds MaxModelLen
			GenTokens:    16,
			Repetitions:  1,
		},
		VLLMURL: srv.URL, ServedName: "m",
		MaxModelLen: 1024,
	}
	runner.Run(context.Background(), cfg, make(chan ProgressUpdate, 8))

	final, _ := store.Get("skip-r")
	if final.Status != StatusCompleted {
		t.Fatalf("expected completed (skipping should not fail run), got %s", final.Status)
	}
	if len(final.Warnings) == 0 || !strings.Contains(final.Warnings[0], "9999") {
		t.Fatalf("expected warning about skipped 9999 prompt, got %+v", final.Warnings)
	}
	if len(final.Results) != 1 {
		t.Fatalf("expected 1 result (only 64-token prompt ran), got %d", len(final.Results))
	}
}

func TestRunnerCancellation(t *testing.T) {
	// Slow fake — first chunk takes 500ms; we cancel before that.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		select {
		case <-r.Context().Done():
			return
		case <-time.After(500 * time.Millisecond):
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{}}]}\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	store := NewStore(dir)
	runner := NewRunner(store)
	store.Save(BenchmarkRun{ID: "cx", Status: StatusRunning, CreatedAt: time.Now()})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	runner.Run(ctx, RunnerConfig{
		Run: BenchmarkRun{ID: "cx", Status: StatusRunning},
		Preset: Preset{
			Source:       PresetSourceInternal,
			PromptTokens: []int{32}, GenTokens: 8, Repetitions: 1,
		},
		VLLMURL: srv.URL, ServedName: "m", MaxModelLen: 1024,
	}, make(chan ProgressUpdate, 8))

	final, _ := store.Get("cx")
	if final.Status != StatusFailed {
		t.Fatalf("expected failed (cancelled), got %s", final.Status)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

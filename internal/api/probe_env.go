package api

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// probeEnv adapts *Server to benchmark.ProbeEnv. It spawns vLLM on a
// secondary port for each probe attempt, captures stdout/stderr to a
// buffer for OOM detection, and polls /health to detect readiness.
//
// The probe spins up and tears down vLLM many times. Main vLLM must be
// stopped first (the handler enforces this).
type probeEnv struct {
	s         *Server
	probePort int
	probeHost string
}

// newProbeEnv returns a probeEnv that uses port (main_vllm_port + 1) so
// it never collides with the user-facing instance.
func newProbeEnv(s *Server) *probeEnv {
	return &probeEnv{
		s:         s,
		probePort: s.cfg.VLLMPort + 1,
		probeHost: "127.0.0.1",
	}
}

// ModelMaxPositionEmbeddings reads max_position_embeddings from the
// model's HF config. Used as the absolute ceiling for the probe so we
// don't spend attempts on context lengths the model architecture
// fundamentally can't support.
func (e *probeEnv) ModelMaxPositionEmbeddings(modelID string) int {
	m, ok := e.s.registry.Get(modelID)
	if !ok {
		return 0
	}
	return m.HFConfig.MaxPositionEmbeddings
}

// TrySpawn starts vLLM on the probe port with the given attempt
// parameters, waits for one of {ready, OOM, ctx-timeout}, stops the
// process, and returns the outcome.
//
// Logs are captured into a bounded buffer so OOM-detection regexes can
// scan them after the attempt ends.
func (e *probeEnv) TrySpawn(ctx context.Context, modelID string, attempt benchmark.ProbeAttempt) benchmark.ProbeAttemptResult {
	m, ok := e.s.registry.Get(modelID)
	if !ok {
		return benchmark.ProbeAttemptResult{Outcome: benchmark.ProbeOther, Detail: "model not registered"}
	}
	modelPath := process.ResolveModelPath(m.LocalPath)

	cfg := process.VLLMStartConfig{
		Dtype:                m.VLLMConfig.Dtype,
		MaxModelLen:          attempt.MaxModelLen,
		TensorParallelSize:   attempt.TensorParallelSize,
		GPUMemoryUtilization: attempt.GPUMemoryUtilization,
		EnforceEager:         m.VLLMConfig.EnforceEager,
		TrustRemoteCode:      m.VLLMConfig.TrustRemoteCode,
		MaxNumSeqs:           attempt.MaxNumSeqs,
		Quantization:         m.VLLMConfig.Quantization,
		LoadFormat:           m.VLLMConfig.LoadFormat,
		EnablePrefixCaching:  m.VLLMConfig.EnablePrefixCaching,
		KVCacheDtype:         m.VLLMConfig.KVCacheDtype,
		EnableChunkedPrefill: m.VLLMConfig.EnableChunkedPrefill,
	}
	args := append(
		[]string{"serve", modelPath,
			"--host", e.probeHost,
			"--port", fmt.Sprintf("%d", e.probePort),
		},
		process.BuildArgs(cfg)...,
	)
	env := process.BuildEnv(m.Quantization.Method)

	cmdCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "vllm", args...)
	cmd.Env = append(os.Environ(), env...)

	var logBuf safeBuffer
	cmd.Stdout = &logBuf
	cmd.Stderr = &logBuf

	if err := cmd.Start(); err != nil {
		return benchmark.ProbeAttemptResult{Outcome: benchmark.ProbeOther, Detail: "start vllm: " + err.Error()}
	}

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	// Make sure the spawned vLLM dies even if we return early.
	defer func() {
		cancel()
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			_ = cmd.Process.Kill()
			<-done
		}
		// Give VRAM a moment to free before the next attempt.
		time.Sleep(2 * time.Second)
	}()

	healthURL := fmt.Sprintf("http://%s:%d/health", e.probeHost, e.probePort)
	pollInterval := 2 * time.Second
	pollClient := &http.Client{Timeout: 2 * time.Second}

	for {
		select {
		case <-ctx.Done():
			return benchmark.ProbeAttemptResult{Outcome: benchmark.ProbeTimeout, Detail: "context deadline exceeded"}

		case err := <-done:
			// Process exited before we observed ready. Classify the
			// captured logs.
			logs := logBuf.String()
			res := benchmark.ClassifyLogs(logs)
			if err != nil && res.Detail == "" {
				res.Detail = "vllm exited: " + err.Error()
			}
			return res

		case <-time.After(pollInterval):
			resp, err := pollClient.Get(healthURL)
			if err != nil {
				// Not ready yet — check whether OOM is already visible.
				if oom, suggested := benchmark.DetectOOM(logBuf.String()); oom {
					return benchmark.ProbeAttemptResult{
						Outcome:         benchmark.ProbeOOM,
						SuggestedMaxLen: suggested,
						Detail:          "OOM detected before health came up",
					}
				}
				continue
			}
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return benchmark.ProbeAttemptResult{Outcome: benchmark.ProbeReady}
			}
		}
	}
}

// safeBuffer is a thread-safe bytes.Buffer wrapper. The vLLM stdout/stderr
// pipes are written from a goroutine inside exec.Cmd while we read the
// buffer from another goroutine doing OOM detection.
type safeBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (s *safeBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *safeBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Cap the snapshot at the last 64 KiB to keep regex scans cheap on
	// long-running probe attempts.
	const cap = 64 * 1024
	b := s.buf.Bytes()
	if len(b) > cap {
		return string(b[len(b)-cap:])
	}
	return string(b)
}

// Reset is unused but kept so the buffer can be repurposed by future
// long-lived probe sessions.
func (s *safeBuffer) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.buf.Reset()
}

// probeResultStore is a simple in-memory cache of probe results,
// keyed by model id. Probe results are advisory defaults — losing them
// on restart is fine; the user can re-run a probe. Persistent storage
// would require a registry schema bump which is out of scope here.
type probeResultStore struct {
	mu      sync.RWMutex
	results map[string]*benchmark.ProbeResult
}

func newProbeResultStore() *probeResultStore {
	return &probeResultStore{results: make(map[string]*benchmark.ProbeResult)}
}

func (s *probeResultStore) Get(modelID string) (*benchmark.ProbeResult, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	r, ok := s.results[modelID]
	return r, ok
}

func (s *probeResultStore) Set(modelID string, r *benchmark.ProbeResult) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.results[modelID] = r
}

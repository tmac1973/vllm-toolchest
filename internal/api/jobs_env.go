package api

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// jobEnv adapts *Server to benchmark.JobEnv. Created once at server
// construction and handed to the benchmark.Service so it can drive vLLM
// process swaps and inject registry/monitor state into runs.
type jobEnv struct {
	s *Server
}

func newJobEnv(s *Server) *jobEnv { return &jobEnv{s: s} }

// ResolveModel returns registry data for a model.
func (e *jobEnv) ResolveModel(modelID string) (benchmark.ModelInfo, error) {
	m, ok := e.s.registry.Get(modelID)
	if !ok {
		return benchmark.ModelInfo{}, fmt.Errorf("model not registered: %s", modelID)
	}
	return benchmark.ModelInfo{
		HFRepoID:    m.ID,
		Quant:       m.Quantization.Method,
		SizeGB:      float64(m.TotalSizeBytes) / (1024 * 1024 * 1024),
		DisplayName: displayNameOf(m),
		// ServedName is discovered lazily because vLLM may not be running
		// at the time ResolveModel is called. The job runner calls
		// EnsureModelLoaded first; we re-resolve the served name here so
		// the value is current.
		ServedName: e.resolveServedName(m),
		Config: benchmark.ConfigSnapshot{
			MaxModelLen:          m.VLLMConfig.MaxModelLen,
			TensorParallelSize:   m.VLLMConfig.TensorParallelSize,
			GPUMemoryUtilization: m.VLLMConfig.GPUMemoryUtilization,
			KVCacheDtype:         m.VLLMConfig.KVCacheDtype,
			EnforceEager:         m.VLLMConfig.EnforceEager,
			Dtype:                m.VLLMConfig.Dtype,
			QuantMethod:          m.Quantization.Method,
		},
	}, nil
}

// resolveServedName tries /v1/models first and falls back to the resolved
// filesystem path. The path-form is what vLLM reports unless the user
// passed --served-model-name (we don't expose that yet).
func (e *jobEnv) resolveServedName(m *models.Model) string {
	if name, err := e.s.discoverServedName(m.ID); err == nil {
		return name
	}
	return process.ResolveModelPath(m.LocalPath)
}

// CurrentLoadedModel returns the HF repo id of the model vLLM is serving,
// or "" when nothing is loaded.
func (e *jobEnv) CurrentLoadedModel() string {
	st := e.s.process.GetStatus()
	if st.State != process.StateRunning && st.State != process.StateStarting {
		return ""
	}
	return st.ModelID
}

// EnsureModelLoaded restarts vLLM with the target model+config if it
// isn't already serving it. Polls the manager's state until Running or
// the context is cancelled (or the manager goes to Error).
func (e *jobEnv) EnsureModelLoaded(ctx context.Context, modelID string, cfg benchmark.ConfigSnapshot) error {
	st := e.s.process.GetStatus()

	// Already serving the right model: nothing to do.
	if st.State == process.StateRunning && st.ModelID == modelID {
		return nil
	}

	m, ok := e.s.registry.Get(modelID)
	if !ok {
		return fmt.Errorf("model not registered: %s", modelID)
	}
	modelPath := process.ResolveModelPath(m.LocalPath)

	startCfg := vllmStartConfigFor(m, cfg)
	args := process.BuildArgs(startCfg)
	env := process.BuildEnv(m.Quantization.Method)

	// Restart handles the stop-if-running case for us.
	if err := e.s.process.Restart(modelID, modelPath, args, env); err != nil {
		return fmt.Errorf("restart vLLM: %w", err)
	}

	// Poll for readiness; the manager's waitForReady goroutine flips the
	// state to Running when /health responds. We re-check every 2s.
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		st := e.s.process.GetStatus()
		switch st.State {
		case process.StateRunning:
			return nil
		case process.StateError:
			return fmt.Errorf("vLLM entered error state: %s", st.Error)
		case process.StateStopped:
			return errors.New("vLLM stopped before becoming ready")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// vllmStartConfigFor builds the process-level start config from a model
// and the benchmark ConfigSnapshot (which carries overrides applied at
// the job level). Fields not in the snapshot fall through from the
// model's saved VLLMConfig.
func vllmStartConfigFor(m *models.Model, snap benchmark.ConfigSnapshot) process.VLLMStartConfig {
	v := m.VLLMConfig
	// snap fields win when set; treat zero values as "use saved".
	maxLen := v.MaxModelLen
	if snap.MaxModelLen > 0 {
		maxLen = snap.MaxModelLen
	}
	tp := v.TensorParallelSize
	if snap.TensorParallelSize > 0 {
		tp = snap.TensorParallelSize
	}
	gmu := v.GPUMemoryUtilization
	if snap.GPUMemoryUtilization > 0 {
		gmu = snap.GPUMemoryUtilization
	}
	kv := v.KVCacheDtype
	if snap.KVCacheDtype != "" {
		kv = snap.KVCacheDtype
	}
	dt := v.Dtype
	if snap.Dtype != "" {
		dt = snap.Dtype
	}
	eager := v.EnforceEager
	if snap.EnforceEager {
		eager = true
	}
	return process.VLLMStartConfig{
		Dtype:                dt,
		MaxModelLen:          maxLen,
		TensorParallelSize:   tp,
		GPUMemoryUtilization: gmu,
		EnforceEager:         eager,
		TrustRemoteCode:      v.TrustRemoteCode,
		MaxNumSeqs:           v.MaxNumSeqs,
		Quantization:         v.Quantization,
		LoadFormat:           v.LoadFormat,
		EnablePrefixCaching:  v.EnablePrefixCaching,
		KVCacheDtype:         kv,
		EnableChunkedPrefill: v.EnableChunkedPrefill,
		MaxNumBatchedTokens:  v.MaxNumBatchedTokens,
		EnableAutoToolChoice: v.EnableAutoToolChoice,
		ToolCallParser:       v.ToolCallParser,
		Tokenizer:            v.Tokenizer,
		ChatTemplate:         v.ChatTemplate,
		ExtraFlags:           v.ExtraFlags,
	}
}

// CurrentMetrics returns the latest GPU snapshot for the benchmark store.
func (e *jobEnv) CurrentMetrics() monitor.Metrics {
	return e.s.monitor.Current()
}

// VLLMURL returns the base URL the runner targets.
func (e *jobEnv) VLLMURL() string {
	return fmt.Sprintf("http://%s:%d", e.s.cfg.VLLMHost, e.s.cfg.VLLMPort)
}

// HFToken returns the HuggingFace token from config (empty when unset).
func (e *jobEnv) HFToken() string {
	return e.s.cfg.HFToken
}

// HFCacheDir returns the persistent tokenizer cache path forwarded to
// llama-benchy as HF_HOME.
func (e *jobEnv) HFCacheDir() string {
	return filepath.Join(e.s.cfg.DataDir, "cache", "huggingface")
}

// VLLMVersion is best-effort; we don't have a stable source for it yet.
// Returning "" lets the run save without failing.
func (e *jobEnv) VLLMVersion() string {
	// Future: parse process startup logs for the "vLLM <version>" line.
	return ""
}

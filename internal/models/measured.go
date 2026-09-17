package models

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
)

// RunMeasurement is what one successful start reported, together with enough
// of the configuration to know when it stops applying.
//
// It exists because every figure this project derived from first principles
// turned out wrong -- KV per token by a factor of 2.6, activation by 23, the
// allocator overhead by not being modelled at all -- while every figure taken
// from the engine was right. See plan/phase-14-measured-vram.md.
type RunMeasurement struct {
	At time.Time `json:"at"`
	// TP is the width the run used. Needed to read the per-rank figures, and
	// part of what invalidates them.
	TP int `json:"tp"`
	// ContextTokens is the max_model_len the run was started with, kept so the
	// KV figure can be re-scaled to a different one rather than reused blindly.
	ContextTokens int `json:"context_tokens,omitempty"`
	// Fingerprint identifies the configuration. A measurement taken under a
	// different one describes a different model, whatever the id says.
	Fingerprint string `json:"fingerprint"`

	Engine advice.Measurements `json:"engine"`
}

// Complete reports whether the run got far enough to be worth keeping. A start
// that died during weight loading has a weights figure and nothing else, which
// is not enough to base an estimate on.
func (r RunMeasurement) Complete() bool {
	return r.Engine.KVCacheGB > 0 && r.Engine.KVCacheTokens > 0 && r.Engine.ConsumedGB > 0
}

// KVBytesPerToken is what one token of context costs across the whole pool.
func (r RunMeasurement) KVBytesPerToken() float64 { return r.Engine.KVBytesPerToken(r.TP) }

// TotalRequiredGB is what this configuration would need in total, at a given
// context length, from measured figures alone.
//
// Only the KV term moves with the context. Everything else -- weights, the
// allocator's overhead, the working set, the captured graphs -- was measured
// per rank and is multiplied back out.
func (r RunMeasurement) TotalRequiredGB(contextTokens int) float64 {
	tp := r.TP
	if tp < 1 {
		tp = 1
	}
	e := r.Engine
	perRank := e.ConsumedGB + e.PeakActivationGB + e.GraphPoolGB
	total := perRank * float64(tp)

	if contextTokens <= 0 {
		contextTokens = r.ContextTokens
	}
	if b := r.KVBytesPerToken(); b > 0 && contextTokens > 0 {
		total += b * float64(contextTokens) / (1024 * 1024 * 1024)
	}
	return total
}

// MeasurementFingerprint identifies the configuration a measurement belongs to.
//
// What is in it is a deliberately conservative choice. A fingerprint that is
// too strict reports a measurement as stale more often than it needs to, which
// is visible and merely annoying. One that is too loose silently reuses a
// figure that no longer applies -- the exact failure this whole approach exists
// to escape. So anything that plausibly moves a measured number is included.
//
// MaxModelLen is deliberately absent: varying it is the point, and the KV term
// scales with it exactly. MaxNumBatchedTokens *is* included, because the
// activation figure is a peak over the batch and there is no measured slope to
// re-scale it with -- one run cannot establish one.
func MeasurementFingerprint(m *Model) string {
	if m == nil {
		return ""
	}
	c := m.VLLMConfig
	h := sha256.New()
	fmt.Fprintf(h, "tp=%d\x00kv=%s\x00eager=%t\x00batched=%d\x00",
		c.TensorParallelSize, c.KVCacheDtype, c.EnforceEager, c.MaxNumBatchedTokens)
	fmt.Fprintf(h, "spec=%s\x00compile=%s\x00", c.SpeculativeConfig, c.CompilationConfig)
	fmt.Fprintf(h, "env=%s\x00flags=%s\x00", c.Env, c.ExtraFlags)
	fmt.Fprintf(h, "quant=%s\x00bpp=%g\x00size=%d",
		m.Quantization.Method, m.Quantization.BytesPerParam, m.TotalSizeBytes)
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// Applies reports whether a measurement still describes this model's current
// configuration.
func (r RunMeasurement) Applies(m *Model) bool {
	return r.Complete() && r.Fingerprint != "" && r.Fingerprint == MeasurementFingerprint(m)
}

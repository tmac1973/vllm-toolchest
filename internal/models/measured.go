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

	// ImageVariant is the image this was measured in, stamped into every
	// build as VLLMCTL_IMAGE_VARIANT.
	//
	// Deliberately outside Fingerprint. The fingerprint identifies a
	// *configuration*, and is a pure function of the model record; this
	// identifies the engine underneath it, which no model record knows. Both
	// have to match for a measurement to apply, for different reasons.
	ImageVariant string `json:"image_variant,omitempty"`

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

// MeasuredEstimate builds an estimate from what a start actually reported,
// re-scaled to whatever context length is configured now.
//
// Reports false when the model has never run, when the run did not get far
// enough, or when the configuration has moved in a way that retires it -- in
// which case the caller falls back to the projected arithmetic and says so.
//
// Only the KV term is re-scaled. The weights, the allocator's overhead, the
// working set and the graph pool were all measured per rank and are carried
// across unchanged, because nothing in the configuration that would move them
// can change without retiring the measurement outright.
func MeasuredEstimate(m *Model, id EngineIdentity) (VRAMEstimate, bool) {
	if m == nil || m.Measured == nil || !m.Measured.Applies(m, id) {
		return VRAMEstimate{}, false
	}
	run := *m.Measured
	e := run.Engine

	ctx := m.VLLMConfig.MaxModelLen
	if ctx <= 0 {
		ctx = m.HFConfig.MaxPositionEmbeddings
	}

	est := VRAMEstimate{
		Source:     SourceMeasured,
		MeasuredAt: run.At,
		MeasuredTP: run.TP,

		ContextTokens: ctx,
		CheckpointGB:  float64(m.TotalSizeBytes) / (1024 * 1024 * 1024),

		// Per-rank figures multiplied back out. ConsumedGB is weights plus the
		// allocator's overhead, which the projected path never modelled at all.
		WeightsTotalGB:         e.ConsumedGB * float64(run.TP),
		GraphPoolGB:            e.GraphPoolGB * float64(run.TP),
		MeasuredGraphPerRankGB: e.GraphPoolGB,

		// The offload figure is measured too, rather than inferred from a
		// residual as the projected path has to.
		HostResidentGB:    e.PLEOffloadGB,
		HostResidentMinGB: e.PLEOffloadGB,

		Offload: DetectOffload(m.OwnEnvPairs(), m.VLLMConfig.ExtraFlags),
	}

	// Params are a property of the checkpoint, not of a run, so they come from
	// the structural count either way.
	if params := estimateParamCount(m.HFConfig); params > 0 {
		est.ParamCountBillion = float64(params) / 1e9
		est.ActiveParamBillion = float64(ActiveParamCount(m.HFConfig)) / 1e9
	}

	if b := run.KVBytesPerToken(); b > 0 {
		est.KVCachePerTokenB = int64(b)
		est.KVAtContextGB = b * float64(ctx) / (1024 * 1024 * 1024)
	}
	est.ActivationBaseGB = e.PeakActivationGB * float64(run.TP)
	// DeviceCacheGB is left at zero deliberately: an on-card expert cache is
	// already inside the measured ConsumedGB, and adding the configured figure
	// on top would count it twice.

	total := run.TotalRequiredGB(ctx)
	est.TotalRequiredGB = total
	est.TotalRequiredLowGB = total
	est.TotalRequiredHighGB = total

	// Kept so the panel can show the split, though the measured total does not
	// depend on them.
	est.DeviceWeightsGB = est.WeightsTotalGB
	est.DeviceWeightsHighGB = est.WeightsTotalGB
	est.StructuralGB = est.WeightsTotalGB

	return est, true
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

// EngineIdentity is which vLLM the machine is running, as distinct from how a
// model is configured.
//
// Two fields, because they become knowable at different moments. Variant is
// stamped into the image at build time and is therefore known from boot,
// before anything has been started; Version comes from the engine's own
// banner and is known only once something has. Switching images moves both;
// rebuilding a source-tracking variant moves only the second.
type EngineIdentity struct {
	Variant string
	Version string
}

// retires reports whether this identity describes a different engine from the
// one that took the measurement.
//
// Conservative on purpose: an unknown on either side retires nothing. A
// freshly built image has no version to compare until something has run in
// it, and discarding a measurement on that absence would throw away a figure
// that is very likely still good. Only a known mismatch counts -- which is
// also what keeps this from retiring everything the first time these fields
// appear on records written before they existed.
func (id EngineIdentity) retires(r RunMeasurement) bool {
	if id.Variant != "" && r.ImageVariant != "" && id.Variant != r.ImageVariant {
		return true
	}
	if id.Version != "" && r.Engine.EngineVersion != "" && id.Version != r.Engine.EngineVersion {
		return true
	}
	return false
}

// Applies reports whether a measurement still describes this model's current
// configuration, taken on the engine now installed.
//
// Two separate questions, and both have to answer yes. The fingerprint covers
// the configuration; the identity covers the engine, which no amount of
// configuration hashing can see. Passing a zero EngineIdentity asks the
// configuration question alone.
func (r RunMeasurement) Applies(m *Model, id EngineIdentity) bool {
	if !r.Complete() || r.Fingerprint == "" || r.Fingerprint != MeasurementFingerprint(m) {
		return false
	}
	return !id.retires(r)
}

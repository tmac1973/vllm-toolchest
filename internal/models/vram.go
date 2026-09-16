package models

import (
	"fmt"

	"github.com/tmac1973/vllm-toolchest/internal/config"
)

// VRAMEstimate is what a model needs, independent of the hardware it might run
// on. It is stored in the registry, so it holds only facts about the checkpoint
// and the configuration it would be launched with -- never a verdict.
//
// A verdict depends on the cards in the machine, and a stored verdict outlives
// them: the record that said "Too large" was written against an invented 32 GiB
// single card and then persisted, so it kept saying so on a host with four.
// Fit answers the hardware question, at render time, and is never written down.
type VRAMEstimate struct {
	ParamCountBillion float64 `json:"param_count_billion"`
	// ActiveParamBillion is what one token flows through on an MoE -- the
	// dense part plus only the experts the router picks. Zero on a dense
	// model, where the total already answers the question.
	ActiveParamBillion float64 `json:"active_param_billion,omitempty"`

	// StructuralGB is params x bytes-per-param. CheckpointGB is what the
	// weights actually occupy, which is the files on disk whenever we have
	// them. They are kept apart because the difference between them is
	// informative: it is the tensors the formula does not model.
	StructuralGB float64 `json:"structural_gb"`
	CheckpointGB float64 `json:"checkpoint_gb"`

	// HostResidentGB is the part of the checkpoint that offload keeps in
	// system RAM, and DeviceWeightsGB is what is left for the GPUs to hold.
	//
	// This replaces a "disk floor" that took the larger of the formula and the
	// file size and called it VRAM. The floor was reaching for something real
	// -- the formula undercounts, and the files cannot lie about their size --
	// but it counted host-resident bytes as GPU bytes, so a checkpoint whose
	// n-gram table sits in system RAM was judged as though it did not.
	// HostResidentGB is the most that could plausibly be host-resident and
	// HostResidentMinGB the least. They are equal only when every active
	// offload has a stated size.
	HostResidentGB    float64 `json:"host_resident_gb,omitempty"`
	HostResidentMinGB float64 `json:"host_resident_min_gb,omitempty"`
	// OffloadUnsized says an offload is active that cannot be bounded at all
	// -- in practice NVMe, whose staging behaviour is not modelled. Everything
	// else is bounded, loosely or tightly, and how tightly is a question for
	// the fit against a particular budget rather than for the flags.
	OffloadUnsized  bool    `json:"offload_unsized,omitempty"`
	DeviceWeightsGB float64 `json:"device_weights_gb"`
	// DeviceWeightsHighGB is the upper bound when offload is on but unsized:
	// what the cards hold if nothing moves off them after all. Equal to
	// DeviceWeightsGB when the figure is certain.
	DeviceWeightsHighGB float64 `json:"device_weights_high_gb,omitempty"`

	// KVCachePerTokenB is for the whole model, across every attention layer.
	KVCachePerTokenB int64 `json:"kv_cache_per_token_bytes"`

	// TotalRequiredGB is the headline: how much GPU memory it takes to load
	// this model and serve the context it is configured for, summed across
	// every card it will be split over.
	//
	// Totalled rather than per-card deliberately. What a single rank holds is
	// an implementation detail of the split; the question being asked is
	// whether this configuration can run at all, and that is answered by one
	// number moving as the configuration is edited -- context length, KV
	// dtype, offload, tensor-parallel width.
	//
	// It is computed from the configuration alone and never from the cards
	// present, so it means the same thing on any host. Comparing it to a
	// particular machine is a separate step.
	TotalRequiredGB     float64 `json:"total_required_gb"`
	TotalRequiredLowGB  float64 `json:"total_required_low_gb,omitempty"`
	TotalRequiredHighGB float64 `json:"total_required_high_gb,omitempty"`

	// The parts of that total, kept so the panel can show the working.
	WeightsTotalGB float64 `json:"weights_total_gb"`
	KVAtContextGB  float64 `json:"kv_at_context_gb"`
	GraphPoolGB    float64 `json:"graph_pool_gb,omitempty"`
	// ContextTokens is the context the KV figure was computed for.
	ContextTokens int `json:"context_tokens,omitempty"`

	// DeviceCacheGB is a buffer the engine stages on each card -- the expert
	// streaming cache. Unlike offload it adds to what a rank holds.
	DeviceCacheGB float64 `json:"device_cache_gb,omitempty"`

	// ActivationBaseGB is the working memory the model needs beyond its
	// weights, for the whole model at the configured batch size. Fit divides
	// it across the ranks. It lives here rather than in Fit because it depends
	// only on the shape and the configuration, neither of which is hardware.
	ActivationBaseGB float64 `json:"activation_base_gb"`

	Offload Offload `json:"offload,omitzero"`

	// ParamsUncertain marks a parameter count known to be understated rather
	// than merely approximate, with Caveat saying why. The checkpoint and
	// device figures remain usable: those come from the files on disk, which
	// are a measurement and not affected by what the formula cannot model.
	ParamsUncertain bool   `json:"params_uncertain,omitempty"`
	Caveat          string `json:"caveat,omitempty"`

	// Unknown withholds the figure entirely, and UnknownWhy says what is
	// missing. Showing a wrong number confidently is worse than showing none:
	// bytesPerParam already returns zero rather than inventing a value for a
	// quantization scheme it does not recognise, and this extends that posture
	// to the rest of the estimate.
	Unknown    bool   `json:"unknown,omitempty"`
	UnknownWhy string `json:"unknown_why,omitempty"`
}

// Ranged reports whether the device-weight figure is a band rather than a
// number, which happens when offload is active but its size is not knowable
// from the configuration alone.
func (e VRAMEstimate) Ranged() bool {
	return e.OffloadUnsized || e.DeviceWeightsHighGB > e.DeviceWeightsGB+0.05
}

// OwnEnvPairs is the model's own environment block, parsed by the same code
// that parses it for a launch, so the estimate cannot read it differently from
// the way the engine will receive it.
//
// It is the minimum a caller should pass to EstimateVRAM. The API layer passes
// the fully resolved environment instead, which also carries the machine-wide
// and variant-knob layers.
func (m *Model) OwnEnvPairs() []string {
	if m == nil {
		return nil
	}
	return config.EnvSet{Extra: m.VLLMConfig.Env}.Pairs()
}

// EstimateVRAM computes what a model needs, with no reference to the hardware
// available.
//
// envPairs is the resolved launch environment -- what the engine will actually
// be started with. Pass nil to use the model's own block alone; a caller that
// does so can only miss an offload setting, which overstates VRAM rather than
// understating it.
func EstimateVRAM(m *Model, envPairs []string) VRAMEstimate {
	est := VRAMEstimate{}
	if m == nil {
		est.Unknown = true
		est.UnknownWhy = "no model"
		return est
	}
	if envPairs == nil {
		envPairs = m.OwnEnvPairs()
	}

	cfg := m.HFConfig
	est.Offload = DetectOffload(envPairs, m.VLLMConfig.ExtraFlags)
	est.KVCachePerTokenB = kvCachePerToken(cfg, m.VLLMConfig)
	est.ActivationBaseGB = activationBaseGB(cfg, m.VLLMConfig)

	diskGB := float64(m.TotalSizeBytes) / (1024 * 1024 * 1024)
	params := estimateParamCount(cfg)
	bpp := m.Quantization.BytesPerParam

	if params > 0 {
		est.ParamCountBillion = float64(params) / 1e9
		est.ActiveParamBillion = float64(ActiveParamCount(cfg)) / 1e9
		if bpp > 0 {
			est.StructuralGB = float64(params) * bpp / (1024 * 1024 * 1024)
		}
	}

	// The checkpoint is what the weights occupy. The files on disk are the
	// better witness when we have them -- they include every tensor, modelled
	// or not -- and the formula stands in when we do not.
	switch {
	case diskGB > 0:
		est.CheckpointGB = diskGB
	case est.StructuralGB > 0:
		est.CheckpointGB = est.StructuralGB
	default:
		est.Unknown = true
		switch {
		case params == 0:
			est.UnknownWhy = "architecture not recognised"
		case bpp <= 0:
			est.UnknownWhy = "unrecognised quantization scheme"
		default:
			est.UnknownWhy = "no size on disk"
		}
		return est
	}

	if params == 0 && bpp > 0 {
		// Work the parameter count backwards so the panel still has a figure
		// to show, but do not pretend to know the architecture.
		est.ParamCountBillion = est.CheckpointGB / bpp
	}

	// An MoE whose expert shape is missing counts every expert layer as a
	// single dense MLP, so the parameter count reads an order of magnitude too
	// small. That discredits the parameter count -- not the checkpoint size,
	// which is measured from the files and does not care what the formula can
	// model.
	//
	// So this caveats rather than withholds. Blanking a figure we actually
	// have, because an architecture *name* hints at experts, would be the
	// worse answer: the disk size is the more reliable of the two numbers
	// here, and it is the one that decides whether the weights fit.
	if isMoE(cfg) && (cfg.NumExperts == 0 || cfg.MoEIntermediate == 0) {
		est.ParamsUncertain = true
		est.Caveat = "expert shape missing from config, so the parameter count is understated"
		if diskGB <= 0 {
			// Nothing measured to fall back on: now there is no figure.
			est.Unknown = true
			est.UnknownWhy = "mixture-of-experts shape missing from config"
		}
	}

	maxHost, minHost := hostResidentRange(est, cfg)
	est.HostResidentGB = maxHost
	est.HostResidentMinGB = minHost
	est.OffloadUnsized = est.Offload.Any() && !est.Offload.Bounded()
	// A cache staged on the card is not offload: it is one more thing a rank
	// must hold.
	est.DeviceCacheGB = est.Offload.ExpertCacheGB
	est.DeviceWeightsGB = est.CheckpointGB - maxHost
	est.DeviceWeightsHighGB = est.CheckpointGB - minHost
	if est.DeviceWeightsGB < 0 {
		est.DeviceWeightsGB = 0
	}
	if est.DeviceWeightsHighGB < est.DeviceWeightsGB {
		est.DeviceWeightsHighGB = est.DeviceWeightsGB
	}
	if est.Ranged() && est.Caveat == "" {
		est.Caveat = "offload size is estimated, so the GPU figure is a range"
	}

	if est.Offload.NVMe {
		est.UnknownWhy = "NVMe offload is not modelled"
	}

	addTotals(&est, cfg, m.VLLMConfig)
	return est
}

// Requirement is what one configuration costs in GPU memory, summed across the
// cards it is split over.
type Requirement struct {
	TP int

	WeightsGB     float64
	WeightsHighGB float64
	KVGB          float64
	GraphsGB      float64
	CacheGB       float64
	ActivationGB  float64

	TotalGB     float64
	TotalHighGB float64
}

// RequiredAt works out what this model needs at a given tensor-parallel width.
//
// Everything here scales with something the operator can change, which is the
// point: the figure is meant to move as the model is configured, so the effect
// of halving the context or enabling offload is visible before the engine is
// started rather than after it fails to start.
func RequiredAt(est VRAMEstimate, c VLLMConfig, tp int) Requirement {
	if tp < 1 {
		tp = 1
	}
	r := Requirement{
		TP:            tp,
		WeightsGB:     weightsTotalGB(est.DeviceWeightsGB, tp),
		WeightsHighGB: weightsTotalGB(est.DeviceWeightsHighGB, tp),
		KVGB:          est.KVAtContextGB,
		// Every rank captures its own graph ladder and stages its own expert
		// cache, so both scale with the width.
		CacheGB:      est.DeviceCacheGB * float64(tp),
		ActivationGB: est.ActivationBaseGB,
	}
	if !c.EnforceEager {
		r.GraphsGB = graphPoolPerGPUGB * float64(tp)
	}

	fixed := r.KVGB + r.GraphsGB + r.CacheGB + r.ActivationGB
	r.TotalGB = r.WeightsGB + fixed
	r.TotalHighGB = r.WeightsHighGB + fixed
	return r
}

// addTotals fills in the headline figure for the width the model is set to.
func addTotals(est *VRAMEstimate, cfg HFConfig, c VLLMConfig) {
	// The context actually served. An unset max_model_len means the
	// checkpoint's own maximum, which is what vLLM will use.
	ctx := c.MaxModelLen
	if ctx <= 0 {
		ctx = cfg.MaxPositionEmbeddings
	}
	est.ContextTokens = ctx
	if ctx > 0 && est.KVCachePerTokenB > 0 {
		est.KVAtContextGB = float64(est.KVCachePerTokenB) * float64(ctx) / (1024 * 1024 * 1024)
	}

	r := RequiredAt(*est, c, c.TensorParallelSize)
	est.WeightsTotalGB = (r.WeightsGB + r.WeightsHighGB) / 2
	est.GraphPoolGB = r.GraphsGB
	est.TotalRequiredLowGB = r.TotalGB
	est.TotalRequiredHighGB = r.TotalHighGB
	est.TotalRequiredGB = (r.TotalGB + r.TotalHighGB) / 2
}

// weightsTotalGB is what all the cards together hold of the weights.
//
// The replication surcharge is a cost of splitting, so a single rank does not
// pay it: with nothing sharded there is nothing replicated.
func weightsTotalGB(deviceGB float64, tp int) float64 {
	if tp < 2 {
		return deviceGB
	}
	return deviceGB * replicationOverhead
}

// hostResidentRange bounds what offload keeps off the GPUs: maxHost is the
// most that could plausibly be host-resident, minHost the least.
//
// Naming them by direction is deliberate. They were once (low, high) meaning
// host-resident amounts, where "low" produced the *larger* device figure --
// and that inversion is precisely how a zero-sized PLE table slipped through
// as a confident number twice.
//
// The PLE table is never sized to a figure. The unaccounted-tensor residual --
// what the checkpoint holds beyond what the structural formula explains --
// bounds it, and is used only as the optimistic end of a band. It reconciles
// with the two checkpoints measured on this machine, 38.8 and 47.68 GiB, but
// that is inference from two points: it will read any other unmodelled tensor
// as embedding table, and on a checkpoint with no such table it is zero, which
// says nothing about whether the offload did anything. So the pessimistic end
// always assumes nothing moves at all, and no flat verdict is ever drawn from
// the optimistic one.
func hostResidentRange(est VRAMEstimate, cfg HFConfig) (maxHost, minHost float64) {
	if !est.Offload.Any() {
		return 0, 0
	}

	if est.Offload.PLE {
		if residual := est.CheckpointGB - est.StructuralGB; est.StructuralGB > 0 && residual > 0 {
			table := residual * pleShareOfResidual
			maxHost += table * (1 + pleBandWidth)
			minHost += table * (1 - pleBandWidth)
		}
		// A residual at or below zero means the method found nothing to
		// measure. It says nothing about whether the offload did anything, so
		// the band is left open below rather than closed at the residual.
	}

	if est.Offload.Experts {
		// Capping at the idle share matters: the active path is resident by
		// definition, so no ceiling can push weights off the cards that every
		// token needs.
		idle := idleExpertGB(est, cfg)
		ceiling := idle
		if est.Offload.ExpertCapSet {
			ceiling = min(ceiling, est.Offload.ExpertHostCapGB)
		}

		// Centre on what offload was observed to move, not on what it was
		// permitted to move. The ceiling still bounds the optimistic end --
		// nothing can exceed what the operator allowed -- but it no longer
		// sets it.
		expected := min(idle*expertOffloadShare, ceiling)
		maxHost += min(expected*(1+expertBandWidth), ceiling)
		minHost += expected * (1 - expertBandWidth)
	}

	if maxHost > est.CheckpointGB {
		maxHost = est.CheckpointGB
	}
	if minHost > maxHost {
		minHost = maxHost
	}
	return maxHost, minHost
}

// idleExpertGB is the weight held in experts a given token does not route to,
// which is the most expert offload can ever move off the cards. Returns 0 when
// the shape is unknown, leaving the caller with a band.
//
// It is an upper bound and not a prediction: the resident working set depends
// on how the router spreads across a batch, which no static figure can know.
func idleExpertGB(est VRAMEstimate, cfg HFConfig) float64 {
	if !isMoE(cfg) || est.StructuralGB <= 0 {
		return 0
	}
	total := estimateParamCount(cfg)
	active := ActiveParamCount(cfg)
	if total <= 0 || active <= 0 || active >= total {
		return 0
	}
	// Offload holds the experts a token does not route to; the resident set is
	// the active path. This is the steady-state share, not the peak.
	idle := float64(total-active) / float64(total)
	return est.StructuralGB * idle
}

// pleShareOfResidual is how much of the unaccounted-tensor residual is
// actually the offloadable embedding table.
//
// It is not all of it. The residual is every tensor the structural formula
// does not model, which also includes the MTP draft weights and the vision
// tower -- both of which stay on the GPU. Taking the whole residual as
// host-resident therefore understates what the cards hold, and the two
// checkpoints measured on compute say by how much, consistently:
//
//	checkpoint   residual   table measured   ratio
//	GPTQ         43.6       38.8             0.890
//	MXFP4        52.2       47.7             0.913
//
// The band is narrow because those two agree to about a point. It widens the
// moment a third checkpoint disagrees, and should.
const (
	pleShareOfResidual = 0.90
	pleBandWidth       = 0.05
)

// expertOffloadShare is how much of the idle expert weight actually leaves the
// card when expert offload is on.
//
// The configured ceiling is not the answer: a run with --expert-offload-mem 46
// was measured moving 18.72 GiB, which is 30% of that checkpoint's idle expert
// weight. Using the ceiling as the optimistic bound is what put a 124B model at
// 2.3 GiB per card -- it assumed the maximum permitted offload and the maximum
// plausible table at the same time, which nothing observed does.
//
// One measurement, hence the wide band. Unlike the PLE share this is fitted
// rather than corroborated, and it stays wide until a second offload
// configuration has been measured.
const (
	expertOffloadShare = 0.30
	expertBandWidth    = 0.50
)

func isMoE(cfg HFConfig) bool {
	if cfg.NumExperts > 0 {
		return true
	}
	switch cfg.ModelType {
	case "mixtral", "qwen3_moe", "qwen3_next", "deepseek_v2", "deepseek_v3":
		return true
	}
	return false
}

// kvCachePerToken is the KV cache one token costs across the whole model.
//
// Only full-attention layers hold a KV cache. On a hybrid the rest keep a
// fixed-size recurrent state that does not grow with context, so counting
// every layer overstates this badly -- fourfold on a model that is 16
// attention layers out of 64.
func kvCachePerToken(cfg HFConfig, c VLLMConfig) int64 {
	headDim := cfg.HeadDim
	if headDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		headDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}

	kvDtypeBytes := 2 // FP16 default
	switch c.KVCacheDtype {
	case "fp8", "fp8_e5m2", "fp8_e4m3":
		kvDtypeBytes = 1
	}

	kvHeads := cfg.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = cfg.NumAttentionHeads
	}

	kvLayers := cfg.AttentionLayers
	if kvLayers <= 0 {
		kvLayers = cfg.NumHiddenLayers // unknown, or a dense model
	}

	if kvLayers <= 0 || kvHeads <= 0 || headDim <= 0 {
		return 0
	}
	return int64(2 * kvLayers * kvHeads * headDim * kvDtypeBytes)
}

// activationBaseGB is the working memory the model needs beyond its weights: a
// handful of hidden-sized buffers over the batch, plus the logits.
//
// It replaces a fixed ladder that returned one of five constants by parameter
// count and capped at 2.0 GB. That was wrong in both directions -- far too
// large for a small model with a small batch, far too small for a large batch
// on any model -- because activation scales with the batch and the hidden
// size, not with how many parameters are sitting still.
func activationBaseGB(cfg HFConfig, c VLLMConfig) float64 {
	h := float64(cfg.HiddenSize)
	if h <= 0 {
		return 0
	}

	// vLLM batches a chunk of tokens per step, not the whole context window,
	// so the window is the wrong fallback: it charged a 262k-context model
	// six gigabytes of activation for a batch it will never form in one step.
	// vLLM's own default is 2048 with chunked prefill, capped by the window
	// when that is smaller.
	tokens := c.MaxNumBatchedTokens
	if tokens <= 0 {
		tokens = 2048
		if c.MaxModelLen > 0 && c.MaxModelLen < tokens {
			tokens = c.MaxModelLen
		}
	}

	seqs := c.MaxNumSeqs
	if seqs <= 0 {
		seqs = 256
	}

	// Roughly the live set through one decoder layer: the residual stream, the
	// two MLP projections and the attention workspace, at activation
	// precision. Layers are processed one at a time, so this does not scale
	// with depth.
	const liveBuffers = 6
	const actBytes = 2

	act := float64(tokens) * h * actBytes * liveBuffers
	logits := float64(seqs) * float64(cfg.VocabSize) * 4

	return (act + logits) / (1024 * 1024 * 1024)
}

// BytesToGB converts bytes to GB for display.
func BytesToGB(b int64) float64 {
	return float64(b) / (1024 * 1024 * 1024)
}

// FormatParamCount formats parameter count for display.
func FormatParamCount(billions float64) string {
	if billions >= 1 {
		return fmt.Sprintf("%.1fB", billions)
	}
	return fmt.Sprintf("%.0fM", billions*1000)
}

func estimateParamCount(cfg HFConfig) int64 {
	if cfg.HiddenSize == 0 || cfg.NumHiddenLayers == 0 {
		return 0
	}

	h := int64(cfg.HiddenSize)
	l := int64(cfg.NumHiddenLayers)
	v := int64(cfg.VocabSize)
	inter := int64(cfg.IntermediateSize)
	kvHeads := int64(cfg.NumKeyValueHeads)
	heads := int64(cfg.NumAttentionHeads)
	headDim := int64(cfg.HeadDim)

	if inter == 0 {
		inter = 4 * h
	}
	if kvHeads == 0 {
		kvHeads = heads
	}
	if headDim == 0 && heads > 0 {
		headDim = h / heads
	}

	// Embeddings
	embedding := v * h
	outputEmbedding := int64(0)
	if !cfg.TieWordEmbeddings {
		outputEmbedding = v * h
	}

	// Per-layer attention. Billed to every layer, including the recurrent
	// layers of a hybrid, whose own projections are of a similar size. On the
	// hybrid checked against a published figure -- Qwen3-Next-80B, 36 of 48
	// layers recurrent -- the whole attention term is barely 1% of the model,
	// so the simplification is well inside the error of everything else here.
	qProj := h * heads * headDim
	kProj := h * kvHeads * headDim
	vProj := h * kvHeads * headDim
	oProj := heads * headDim * h
	attn := qProj + kProj + vProj + oProj

	// Per-layer norms
	norms := 2 * h

	// Dense MLP: gate, up, down.
	denseMLP := 3 * h * inter

	// Mixture-of-experts layers replace that one MLP with NumExperts of them,
	// each at moe_intermediate_size rather than intermediate_size, plus a
	// router and -- on the architectures that have one -- a shared expert
	// every token also passes through.
	//
	// Modelling these as a single dense MLP was the original defect: 512
	// experts of width 512 counted as one 5120-wide MLP undercounts the layer
	// about fiftyfold, and the undercount was then masked by falling back to
	// the checkpoint's size on disk.
	experts := int64(cfg.NumExperts)
	moeInter := int64(cfg.MoEIntermediate)
	moeMLP := int64(0)
	if experts > 0 && moeInter > 0 {
		moeMLP = experts*3*h*moeInter + h*experts // experts + router gate
		if shared := int64(cfg.SharedExpertInter); shared > 0 {
			moeMLP += 3 * h * shared
		}
	}

	// Leading layers that keep an ordinary MLP before the MoE layers begin.
	denseCount := int64(cfg.DenseLayers)
	if denseCount < 0 || denseCount > l {
		denseCount = 0
	}
	if moeMLP == 0 {
		denseCount = l // no usable MoE shape: every layer counts as dense
	}
	moeCount := l - denseCount

	layerAttnNorms := (attn + norms) * l
	finalNorm := h

	return embedding + outputEmbedding + layerAttnNorms +
		denseCount*denseMLP + moeCount*moeMLP + finalNorm
}

// ActiveParamCount is what a single token actually flows through: the dense
// part plus only the experts the router selects. Reported alongside the total
// because the two differ by an order of magnitude on an MoE, and the active
// figure is what predicts compute while the total predicts memory.
//
// Returns 0 when the shape is unknown or the model is dense, where the total
// already answers the question.
func ActiveParamCount(cfg HFConfig) int64 {
	if cfg.NumExperts <= 0 || cfg.NumExpertsPerTok <= 0 || cfg.MoEIntermediate <= 0 {
		return 0
	}
	total := estimateParamCount(cfg)
	if total == 0 {
		return 0
	}

	h := int64(cfg.HiddenSize)
	l := int64(cfg.NumHiddenLayers)
	denseCount := int64(cfg.DenseLayers)
	if denseCount < 0 || denseCount > l {
		denseCount = 0
	}
	moeCount := l - denseCount

	inactive := int64(cfg.NumExperts-cfg.NumExpertsPerTok) * 3 * h * int64(cfg.MoEIntermediate)
	if inactive < 0 {
		return total
	}
	return total - moeCount*inactive
}

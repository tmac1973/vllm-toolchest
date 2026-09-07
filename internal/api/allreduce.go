package api

import "fmt"

// The radiance image's peer-to-peer all-reduce takes a message only up to this
// size; above it, vLLM falls back to RCCL with no error and no log line, at
// roughly 2.3x the cost. The value is _MAX_BYTES in radiance_allreduce.py
// (49152 * 1024).
const r4dAllReduceMaxBytes = 49152 * 1024

// r4dOnlySupportsTP is the tensor-parallel size the shipped kernel library
// handles. libr4d v0.5.0 exposes ar_oneshot_2rank_exact and
// ar_oneshot_2rank_wht6, both constrained to world_size == 2; at any other TP
// the custom all-reduce does not attach and RCCL is used instead.
const r4dOnlySupportsTP = 2

// r4dBatchedTokenCeiling returns the largest --max-num-batched-tokens that
// still keeps a prefill chunk's all-reduce inside the custom kernel.
//
// The message a chunk sends is max_num_batched_tokens * hidden_size * 2 bytes
// (bf16), so the ceiling falls out of the cap. Returns 0 when hidden size is
// unknown.
func r4dBatchedTokenCeiling(hiddenSize int) int {
	if hiddenSize <= 0 {
		return 0
	}
	return r4dAllReduceMaxBytes / (hiddenSize * 2)
}

// batchedTokenAdvice renders the guidance shown under the batched-token field.
//
// This exists because getting it wrong is silent: exceed the ceiling and
// everything still works, just with the tensor-parallel all-reduce quietly
// running on RCCL instead of the kernel the image was built for.
//
// Returns the advice and whether it is a warning (the current setting is
// already over the ceiling).
func batchedTokenAdvice(radiance bool, hiddenSize, tpSize, configured int) (string, bool) {
	if !radiance {
		return "", false
	}
	if tpSize <= 1 {
		return "No tensor-parallel all-reduce at TP=1, so no ceiling applies here.", false
	}
	if tpSize != r4dOnlySupportsTP {
		return fmt.Sprintf(
			"The R4D all-reduce is TP=%d only in this image (libr4d ships 2-rank kernels), "+
				"so at TP=%d RCCL is used and no ceiling applies. Raising this is free here, "+
				"and worth it if vLLM warns that speculative decoding is eating the budget.",
			r4dOnlySupportsTP, tpSize), false
	}

	ceiling := r4dBatchedTokenCeiling(hiddenSize)
	if ceiling == 0 {
		return "", false
	}
	if configured > ceiling {
		return fmt.Sprintf(
			"Over the R4D all-reduce ceiling of %d tokens for this model (hidden size %d): "+
				"a prefill chunk's message exceeds %d MiB and silently falls back to RCCL, "+
				"about 2.3x slower. Nothing will report this at run time.",
			ceiling, hiddenSize, r4dAllReduceMaxBytes/(1024*1024)), true
	}
	return fmt.Sprintf(
		"Keep at or below %d for this model (hidden size %d): above that a prefill chunk's "+
			"all-reduce exceeds %d MiB and silently falls back to RCCL, about 2.3x slower.",
		ceiling, hiddenSize, r4dAllReduceMaxBytes/(1024*1024)), false
}

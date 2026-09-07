package api

// backendOption is one entry in an attention-backend picker.
// Fields are exported so html/template can read them.
type backendOption struct{ Val, Label string }

// attentionBackendOptions lists the backends offered in the UI.
//
// The empty value is first and is the default: vLLM's own selection logic is
// generally right, and naming a backend it did not choose is how you get a
// startup abort rather than a speedup. The radiance-only entries are appended
// when running that image, since they do not exist elsewhere.
func attentionBackendOptions(radiance bool) []backendOption {
	opts := []backendOption{
		{"", "(auto — let vLLM choose)"},
		{"TRITON_ATTN", "TRITON_ATTN"},
		{"TRITON_FLASH_ATTN", "TRITON_FLASH_ATTN"},
		{"FLASH_ATTN", "FLASH_ATTN"},
		{"ROCM_FLASH", "ROCM_FLASH"},
		{"ROCM_AITER_UNIFIED_ATTN", "ROCM_AITER_UNIFIED_ATTN"},
		{"FLASHINFER", "FLASHINFER (CUDA)"},
		{"XFORMERS", "XFORMERS"},
	}
	if radiance {
		// R4D refuses to load on any shape it was not compiled for -- head_dim
		// 256, paged block 16, 6 query heads per KV head, causal, bf16/fp8 KV
		// -- and says so at startup rather than falling back silently.
		opts = append(opts, backendOption{"R4D", "R4D — radiance gfx1201 kernels (head_dim 256, GQA 6)"})
	}
	return opts
}

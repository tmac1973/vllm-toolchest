package api

import "github.com/tmac1973/vllm-toolchest/variants"

// backendOption is one entry in an attention-backend picker.
// Fields are exported so html/template can read them.
type backendOption struct{ Val, Label string }

// Backends that work on any vendor. The empty value is first and is the
// default: vLLM's own selection logic is generally right, and naming a backend
// it did not choose is how you get a startup abort rather than a speedup.
var commonBackends = []backendOption{
	{"", "(auto — let vLLM choose)"},
	{"TRITON_ATTN", "TRITON_ATTN"},
	{"TRITON_FLASH_ATTN", "TRITON_FLASH_ATTN"},
}

// The rest are vendor-specific, and offering one the stack does not have is
// the failure this picker exists to avoid: it aborts at startup, minutes into
// a load, rather than falling back. The list used to be flat, which meant a
// ROCm image offered FLASHINFER and an NVIDIA one offered ROCM_FLASH.
var vendorBackends = map[string][]backendOption{
	"amd": {
		{"ROCM_FLASH", "ROCM_FLASH"},
		{"ROCM_AITER_UNIFIED_ATTN", "ROCM_AITER_UNIFIED_ATTN"},
	},
	"nvidia": {
		{"FLASH_ATTN", "FLASH_ATTN"},
		{"FLASHINFER", "FLASHINFER"},
		{"XFORMERS", "XFORMERS"},
	},
}

// attentionBackendOptions lists the backends offered for one image variant:
// the common set, its vendor's set, then anything its manifest adds.
//
// A variant no manifest describes gets the common set only. That is the safe
// reading — we cannot know what a stack we have no description of supports,
// and an option that aborts the engine is worse than an option that is absent.
func attentionBackendOptions(d variants.Descriptor, known bool) []backendOption {
	opts := append([]backendOption(nil), commonBackends...)
	if !known {
		return opts
	}
	opts = append(opts, vendorBackends[d.Vendor]...)
	for _, b := range d.AttentionBackends {
		opts = append(opts, backendOption{b.Value, b.Label})
	}
	return opts
}

package api

import "github.com/tmac1973/vllm-toolchest/variants"

// backendOption is one entry in an attention-backend picker.
// Fields are exported so html/template can read them.
type backendOption struct{ Val, Label string }

// autoBackend is always offered and is always the default: vLLM's own
// selection logic is generally right, and naming a backend it did not choose
// is how you get a startup abort rather than a speedup.
var autoBackend = backendOption{"", "(auto — let vLLM choose)"}

// attentionBackendOptions lists the backends offered for one image variant.
//
// The list comes from the variant's manifest and nowhere else. It used to be
// hardcoded here, first as one flat list and then split by vendor, and both
// were wrong in the same way: which backends exist is a property of the vLLM
// build inside a particular image, not of the vendor and not of this project.
//
// Two ways that bit. A ROCm image was offered FLASHINFER, and an NVIDIA one
// ROCM_FLASH. Worse, the names drifted: ROCM_FLASH, TRITON_FLASH_ATTN and
// XFORMERS were all offered long after they stopped existing upstream, so
// picking one aborted the engine minutes into a load with an unhelpful error.
// A manifest sits next to the vLLM version it pins, which is the only place
// that list is knowable.
//
// A variant that declares none gets the auto option alone. That is deliberate:
// an empty picker is useless, but a picker full of backends the stack does not
// have is worse than useless. Populating a variant's list needs one real run —
// the engine log prints the candidates it considered, as
// "out of potential backends: [...]".
func attentionBackendOptions(d variants.Descriptor, known bool) []backendOption {
	opts := []backendOption{autoBackend}
	if !known {
		return opts
	}
	for _, b := range d.AttentionBackends {
		opts = append(opts, backendOption{b.Value, b.Label})
	}
	return opts
}

// backendOptionsFor is attentionBackendOptions with one addition: a value
// already configured but not offered by this image is kept as an option.
//
// Without that, rebuilding onto a variant with a different list would render
// the picker with nothing selected, and the next save would quietly clear a
// setting the operator chose deliberately. Showing it, labelled, is the
// difference between "this image does not have that" and losing it silently.
func backendOptionsFor(d variants.Descriptor, known bool, configured string) []backendOption {
	opts := attentionBackendOptions(d, known)
	if configured == "" {
		return opts
	}
	for _, o := range opts {
		if o.Val == configured {
			return opts
		}
	}
	return append(opts, backendOption{
		configured,
		configured + " — configured here, but not one this image offers",
	})
}

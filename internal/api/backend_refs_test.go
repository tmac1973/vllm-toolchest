package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// The case this was written for: a config saved on radiance, carried onto an
// image built with VLLM_ROCM_USE_AITER=0, killing all four workers with
// "No module named 'aiter'" from four stack traces down.
func TestSpeculativeBackendIsValidated(t *testing.T) {
	d, ok := variants.Get("rocm") // declares ROCM_ATTN and TRITON_ATTN
	if !ok {
		t.Fatal("rocm manifest missing")
	}

	spec := `{"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN"}`
	err := validateNamedBackends(d, true, spec, "")
	if err == nil {
		t.Fatal("a backend this image does not have was accepted")
	}
	for _, want := range []string{"ROCM_AITER_UNIFIED_ATTN", "speculative", "ROCM_ATTN"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the message should mention %q; got %q", want, err)
		}
	}

	// One the image does declare is fine, as is omitting the key.
	if err := validateNamedBackends(d, true, `{"method":"mtp","attention_backend":"TRITON_ATTN"}`, ""); err != nil {
		t.Errorf("a declared backend was rejected: %v", err)
	}
	if err := validateNamedBackends(d, true, `{"method":"mtp","num_speculative_tokens":8}`, ""); err != nil {
		t.Errorf("omitting the key should be fine: %v", err)
	}
}

// The extra-flags box is a plain string that lands on the command line, so it
// can name a backend and bypass the picker entirely.
func TestExtraFlagsBackendIsValidated(t *testing.T) {
	d, _ := variants.Get("rocm")

	for _, flags := range []string{
		"--attention-backend FLASHINFER",
		"--attention-backend=FLASHINFER",
		"--max-num-seqs 8 --attention-backend FLASHINFER --enforce-eager",
	} {
		if err := validateNamedBackends(d, true, "", flags); err == nil {
			t.Errorf("%q was accepted", flags)
		}
	}

	for _, flags := range []string{
		"--attention-backend ROCM_ATTN",
		"--max-num-seqs 8",
		"",
		"--attention-backend", // truncated: nothing to check
	} {
		if err := validateNamedBackends(d, true, "", flags); err != nil {
			t.Errorf("%q was rejected: %v", flags, err)
		}
	}
}

// A variant that declares no backends cannot tell a wrong name from one it has
// no record of. Refusing on a guess would block configs that work.
func TestUndeclaredVariantDoesNotValidate(t *testing.T) {
	spec := `{"attention_backend":"ANYTHING_AT_ALL"}`

	if err := validateNamedBackends(variants.Descriptor{}, false, spec, ""); err != nil {
		t.Errorf("no manifest should mean no opinion: %v", err)
	}
	d, _ := variants.Get("rocm-source") // declares none
	if err := validateNamedBackends(d, true, spec, ""); err != nil {
		t.Errorf("a variant with no declared backends should not validate: %v", err)
	}
}

// Malformed JSON is the user's problem to see from vLLM, not a reason for this
// check to throw its own error or silently skip the flags alongside it.
func TestMalformedSpeculativeJSONIsNotFatalHere(t *testing.T) {
	d, _ := variants.Get("rocm")

	if err := validateNamedBackends(d, true, `{not json`, ""); err != nil {
		t.Errorf("malformed JSON should pass this check: %v", err)
	}
	// ...and the extra flags are still examined.
	if err := validateNamedBackends(d, true, `{not json`, "--attention-backend FLASHINFER"); err == nil {
		t.Error("a bad backend in the flags was missed because the JSON did not parse")
	}
}

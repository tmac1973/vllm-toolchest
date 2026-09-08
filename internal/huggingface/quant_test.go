package huggingface

import "testing"

// The repo name describes the recipe an author started from, not the format on
// disk. In a sample of 450 search results, 18 repos with "AWQ" in the name were
// compressed-tensors — so quantization_config has to win over both tags and
// name, or the badge and the filter both lie.
func TestDetectQuantFormatPrefersConfigOverNameAndTags(t *testing.T) {
	cfg := func(method string) *ModelConfigMeta {
		return &ModelConfigMeta{QuantizationConfig: &QuantConfig{QuantMethod: method}}
	}

	for _, tc := range []struct {
		name    string
		modelID string
		tags    []string
		cfg     *ModelConfigMeta
		want    string
	}{
		{
			name:    "AWQ in the name, compressed-tensors on disk",
			modelID: "philbert440/Qwen3.8-27B-W4A16-AWQ",
			tags:    []string{"safetensors", "compressed-tensors", "w4a16", "gptq"},
			cfg:     cfg("compressed-tensors"),
			want:    FormatCompressedTensor,
		},
		{
			name:    "AWQ in the name, quark on disk",
			modelID: "amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16",
			cfg:     cfg("quark"),
			want:    FormatQuark,
		},
		{
			name:    "genuinely AWQ",
			modelID: "TheBloke/Mixtral-8x7B-AWQ",
			cfg:     cfg("awq"),
			want:    FormatAWQ,
		},
		{name: "fbgemm counts as FP8", cfg: cfg("fbgemm_fp8"), want: FormatFP8},
		{name: "mxfp4_16 folds into MXFP4", cfg: cfg("mxfp4_16"), want: FormatMXFP4},
		{name: "auto-round", cfg: cfg("auto-round"), want: FormatAutoRound},
		{
			// Not in the switch: better to show the method than to drop it.
			name: "a method added after this was written",
			cfg:  cfg("some_new_scheme"),
			want: "SOME_NEW_SCHEME",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := DetectQuantFormat(tc.modelID, tc.tags, tc.cfg); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// bitsandbytes states its width with load_in_*bit rather than bits.
func TestDetectQuantFormatBitsAndBytesWidth(t *testing.T) {
	bnb := func(q QuantConfig) *ModelConfigMeta {
		q.QuantMethod = "bitsandbytes"
		return &ModelConfigMeta{QuantizationConfig: &q}
	}
	for _, tc := range []struct {
		name string
		cfg  *ModelConfigMeta
		want string
	}{
		{"load_in_4bit", bnb(QuantConfig{LoadIn4Bit: true}), FormatBnB4},
		{"load_in_8bit", bnb(QuantConfig{LoadIn8Bit: true}), FormatBnB8},
		{"bits: 4", bnb(QuantConfig{Bits: 4}), FormatBnB4},
		{"neither stated", bnb(QuantConfig{}), FormatBnB},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := DetectQuantFormat("x/y", nil, tc.cfg); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// A config with no quantization_config is a positive statement of full
// precision. Silence — no config published at all — is not, and must stay
// distinguishable from it.
func TestDetectQuantFormatUnquantizedVersusUnknown(t *testing.T) {
	empty := &ModelConfigMeta{}

	if got := DetectQuantFormat("meta-llama/Llama-4-70B", []string{"safetensors"}, empty); got != FormatFP16 {
		t.Errorf("config without quantization_config = %q, want %q", got, FormatFP16)
	}
	if got := DetectQuantFormat("someone/private-model", nil, nil); got != QuantUnknown {
		t.Errorf("no config at all = %q, want unknown", got)
	}
	// Even with no config, a marker is still worth reporting.
	if got := DetectQuantFormat("unsloth/mistral-7b-bnb-4bit", nil, nil); got != FormatBnB4 {
		t.Errorf("name marker without config = %q, want %q", got, FormatBnB4)
	}
	// A published config that lacks quantization_config but whose name says
	// otherwise: trust the marker rather than assert full precision.
	if got := DetectQuantFormat("someone/model-mxfp4", nil, empty); got != FormatMXFP4 {
		t.Errorf("name marker with empty config = %q, want %q", got, FormatMXFP4)
	}
}

func TestFormatFromTagsPrefersContainerOverRecipe(t *testing.T) {
	// Real tag set from syvai/Qwen3.8-27B-DFlash2-W4A16, which is
	// compressed-tensors despite also being tagged gptq.
	tags := []string{"safetensors", "qwen3", "compressed-tensors", "w4a16", "gptq", "vllm"}
	if got := formatFromTags(tags); got != FormatCompressedTensor {
		t.Errorf("got %q, want %q", got, FormatCompressedTensor)
	}
}

func TestMatchesQuantFilter(t *testing.T) {
	for _, tc := range []struct {
		format, filter string
		want           bool
	}{
		// Everything passes the empty filter, including unknown.
		{FormatCompressedTensor, "", true},
		{QuantUnknown, "", true},

		// The bug this replaced: unknown formats were bucketed as FP16, so a
		// search for unquantized models returned pages of 4-bit ones.
		{QuantUnknown, "unquantized", false},
		{FormatCompressedTensor, "unquantized", false},
		{FormatFP16, "unquantized", true},

		{FormatFP8, "fp8", true},
		{FormatAWQ, "awq", true},
		{FormatGPTQ, "gptq", true},
		{FormatCompressedTensor, "compressed-tensors", true},
		{FormatAWQ, "compressed-tensors", false},

		// Families share a bucket.
		{FormatMXFP4, "fp4", true},
		{FormatNVFP4, "fp4", true},
		{FormatFP8, "fp4", false},
		{FormatBnB4, "bnb", true},
		{FormatBnB8, "bnb", true},
		{FormatBnB, "bnb", true},

		// "other" is everything quantized with no bucket of its own — which
		// is what keeps a newly-invented format reachable.
		{FormatQuark, "other", true},
		{FormatAutoRound, "other", true},
		{"SOME_NEW_SCHEME", "other", true},
		{FormatAWQ, "other", false},
		{FormatFP16, "other", false},
		{QuantUnknown, "other", false},
	} {
		if got := MatchesQuantFilter(tc.format, tc.filter); got != tc.want {
			t.Errorf("MatchesQuantFilter(%q, %q) = %v, want %v",
				tc.format, tc.filter, got, tc.want)
		}
	}
}

// Every bucket the UI offers must be one the matcher knows, or picking it
// would silently return everything.
func TestQuantFilterOptionsAreAllHandled(t *testing.T) {
	for _, opt := range QuantFilterOptions() {
		if opt.Value == "" {
			continue
		}
		// A handled filter rejects at least one format; the fall-through for
		// an unrecognised value accepts them all.
		rejectedSomething := false
		for _, f := range []string{
			FormatFP16, FormatFP8, FormatAWQ, FormatGPTQ, FormatCompressedTensor,
			FormatMXFP4, FormatNVFP4, FormatBnB4, FormatQuark, QuantUnknown,
		} {
			if !MatchesQuantFilter(f, opt.Value) {
				rejectedSomething = true
				break
			}
		}
		if !rejectedSomething {
			t.Errorf("filter %q matches every format — the matcher does not handle it", opt.Value)
		}
	}
}

func TestNormalizeBaseNameGroupsQuantVariants(t *testing.T) {
	base := normalizeBaseName("org/Qwen3.8-27B")
	for _, variant := range []string{
		"org/Qwen3.8-27B-AWQ",
		"org/Qwen3.8-27B-FP8",
		"org/Qwen3.8-27B-W4A16",
		"org/Qwen3.8-27B-MXFP4",
		"org/Qwen3.8-27B-int4",
		"org/Qwen3.8-27B-Quark",
	} {
		if got := normalizeBaseName(variant); got != base {
			t.Errorf("normalizeBaseName(%q) = %q, want %q", variant, got, base)
		}
	}
}

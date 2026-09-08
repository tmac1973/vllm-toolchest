package huggingface

import (
	"path"
	"strings"
)

// Canonical quantization format labels.
//
// These are what the badge shows and what the filter matches on, so they are
// spelled once here rather than as literals scattered across the API layer.
const (
	QuantUnknown           = "" // nothing to go on: the repo published no config
	FormatFP16             = "FP16"
	FormatAWQ              = "AWQ"
	FormatGPTQ             = "GPTQ"
	FormatFP8              = "FP8"
	FormatMXFP4            = "MXFP4"
	FormatNVFP4            = "NVFP4"
	FormatCompressedTensor = "compressed-tensors"
	FormatBnB4             = "BnB-4bit"
	FormatBnB8             = "BnB-8bit"
	FormatBnB              = "BitsAndBytes"
	FormatQuark            = "Quark"
	FormatAutoRound        = "AutoRound"
	FormatModelOpt         = "ModelOpt"
	FormatGGUF             = "GGUF"
)

// QuantConfig is the quantization_config block of a model's config.json, as
// much of it as identifies the format.
type QuantConfig struct {
	QuantMethod string `json:"quant_method"`
	Bits        int    `json:"bits"`
	GroupSize   int    `json:"group_size"`
	// Format is compressed-tensors' storage layout ("pack-quantized",
	// "float-quantized", …).
	Format string `json:"format"`
	// bitsandbytes says which width it used with these rather than Bits.
	LoadIn4Bit bool `json:"load_in_4bit"`
	LoadIn8Bit bool `json:"load_in_8bit"`
}

// DetectQuantFormat names a repo's quantization format.
//
// The order matters. quantization_config is what vLLM itself reads, so it is
// authoritative and comes first; tags are curated but inconsistent; the repo
// name is a last resort and is frequently wrong. On a sample of 450 search
// results, 18 repos with "AWQ" in the name were compressed-tensors — the name
// describes the recipe the author started from, not the format on disk.
//
// Returns QuantUnknown only when there is nothing to go on at all. A repo that
// published a config with no quantization_config is unquantized, and says so
// as FormatFP16 — the two are different claims and the caller can tell them
// apart.
func DetectQuantFormat(modelID string, tags []string, cfg *ModelConfigMeta) string {
	if cfg != nil {
		if q := cfg.QuantizationConfig; q != nil {
			if f := formatFromMethod(q); f != QuantUnknown {
				return f
			}
		}
		// A config with no quantization_config is a positive statement that
		// the weights are stored at full precision.
		if f := formatFromTags(tags); f != QuantUnknown {
			return f
		}
		if f := formatFromName(modelID); f != QuantUnknown {
			return f
		}
		return FormatFP16
	}

	// No config published. Guess, but never claim full precision on silence.
	if f := formatFromTags(tags); f != QuantUnknown {
		return f
	}
	return formatFromName(modelID)
}

// formatFromMethod maps a quantization_config to a canonical label. An
// unrecognised method is passed through uppercased rather than dropped, so a
// format that appears after this was written still shows something true.
func formatFromMethod(q *QuantConfig) string {
	switch strings.ToLower(strings.TrimSpace(q.QuantMethod)) {
	case "":
		return QuantUnknown
	case "awq":
		return FormatAWQ
	case "gptq":
		return FormatGPTQ
	case "fp8", "fbgemm_fp8":
		return FormatFP8
	case "compressed-tensors", "compressed_tensors":
		return FormatCompressedTensor
	case "mxfp4", "mxfp4_16":
		return FormatMXFP4
	case "nvfp4":
		return FormatNVFP4
	case "bitsandbytes":
		switch {
		case q.LoadIn4Bit || q.Bits == 4:
			return FormatBnB4
		case q.LoadIn8Bit || q.Bits == 8:
			return FormatBnB8
		}
		return FormatBnB
	case "quark":
		return FormatQuark
	case "auto-round", "auto_round", "autoround":
		return FormatAutoRound
	case "modelopt", "modelopt_fp8":
		return FormatModelOpt
	case "gguf":
		return FormatGGUF
	default:
		return strings.ToUpper(q.QuantMethod)
	}
}

// formatFromTags reads HuggingFace's curated tags. Ordered most specific
// first: a compressed-tensors repo is commonly tagged "gptq" and "w4a16" too,
// describing the recipe rather than the container.
func formatFromTags(tags []string) string {
	has := make(map[string]bool, len(tags))
	for _, t := range tags {
		has[strings.ToLower(strings.TrimSpace(t))] = true
	}
	for _, c := range []struct {
		tag    string
		format string
	}{
		{"compressed-tensors", FormatCompressedTensor},
		{"mxfp4", FormatMXFP4},
		{"nvfp4", FormatNVFP4},
		{"auto-round", FormatAutoRound},
		{"quark", FormatQuark},
		{"bitsandbytes", FormatBnB},
		{"awq", FormatAWQ},
		{"gptq", FormatGPTQ},
		{"fp8", FormatFP8},
		{"gguf", FormatGGUF},
	} {
		if has[c.tag] {
			return c.format
		}
	}
	return QuantUnknown
}

// formatFromName reads the repo name. Least reliable of the three, and only
// consulted when nothing better exists.
func formatFromName(modelID string) string {
	name := strings.ToLower(path.Base(modelID))
	for _, c := range []struct {
		marker string
		format string
	}{
		{"-nvfp4", FormatNVFP4},
		{"-mxfp4", FormatMXFP4},
		{"-bnb-4bit", FormatBnB4},
		{"-bnb-8bit", FormatBnB8},
		{"-awq", FormatAWQ},
		{"-gptq", FormatGPTQ},
		{"-gguf", FormatGGUF},
		{"-fp8", FormatFP8},
		// w4a16 / w8a8 and friends are compressed-tensors' naming convention.
		{"w4a16", FormatCompressedTensor},
		{"w8a8", FormatCompressedTensor},
		{"w8a16", FormatCompressedTensor},
	} {
		if strings.Contains(name, c.marker) {
			return c.format
		}
	}
	return QuantUnknown
}

// QuantFilterOption is one entry of the search page's format filter.
type QuantFilterOption struct {
	Value string
	Label string
}

// QuantFilterOptions are the buckets the search page offers. Several formats
// share a bucket where the distinction does not change what an operator would
// pick — the FP4 family, the two bitsandbytes widths — and "other" catches
// everything quantized that has no bucket of its own, including formats that
// postdate this list.
func QuantFilterOptions() []QuantFilterOption {
	return []QuantFilterOption{
		{"", "All formats"},
		{"unquantized", "FP16/BF16 (unquantized)"},
		{"fp8", "FP8"},
		{"awq", "AWQ"},
		{"gptq", "GPTQ"},
		{"compressed-tensors", "compressed-tensors (W4A16, W8A8)"},
		{"fp4", "MXFP4 / NVFP4"},
		{"bnb", "BitsAndBytes"},
		{"other", "Other quantized"},
	}
}

// MatchesQuantFilter reports whether a detected format belongs in a bucket.
//
// An unknown format matches only "All formats": it is not evidence of full
// precision, and quietly listing it under FP16 is how a page of 4-bit models
// came back from a search for unquantized ones.
func MatchesQuantFilter(format, filter string) bool {
	if filter == "" {
		return true
	}
	switch filter {
	case "unquantized":
		return format == FormatFP16
	case "fp8":
		return format == FormatFP8
	case "awq":
		return format == FormatAWQ
	case "gptq":
		return format == FormatGPTQ
	case "compressed-tensors":
		return format == FormatCompressedTensor
	case "fp4":
		return format == FormatMXFP4 || format == FormatNVFP4
	case "bnb":
		return format == FormatBnB4 || format == FormatBnB8 || format == FormatBnB
	case "other":
		switch format {
		case QuantUnknown, FormatFP16, FormatFP8, FormatAWQ, FormatGPTQ,
			FormatCompressedTensor, FormatMXFP4, FormatNVFP4,
			FormatBnB4, FormatBnB8, FormatBnB:
			return false
		}
		return true
	}
	// An unrecognised filter value would otherwise silently match nothing.
	return true
}

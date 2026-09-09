package benchmark

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// A sweep runs the same model and preset at several values of one engine
// parameter, so the numbers either side of a change are measured rather than
// guessed at.
//
// Every axis here is an engine-launch parameter, which means a value change
// costs a reload — vLLM cannot alter its context length or tensor-parallel
// size in place. That is why the runner groups cells by (model, sweep values)
// rather than by model alone: a job sweeping three context lengths across two
// models is six loads, and doing them in a different order would be more.
//
// This is deliberately not a port of llama-toolchest's sweep.go, most of which
// encodes speculative-decoding modes that have no vLLM counterpart.

// SweepAxis is one parameter and the values to run it at.
type SweepAxis struct {
	Field  string   `json:"field"`
	Values []string `json:"values"`
}

// SweepField describes a sweepable parameter: how to show it, and how to turn
// one of its string values into a config override.
type SweepField struct {
	Name    string
	Label   string
	Help    string
	Example string
	// Choices, when set, are the suggested values offered in the UI. The
	// field still accepts anything parse accepts.
	Choices []string

	parse func(string) (string, error)
	apply func(*ConfigOverrides, string) error
}

// Parse normalizes one raw value, or explains why it cannot.
func (f SweepField) Parse(raw string) (string, error) { return f.parse(raw) }

// Apply writes one value onto a set of overrides.
func (f SweepField) Apply(o *ConfigOverrides, value string) error { return f.apply(o, value) }

func parseSweepInt(min, max int) func(string) (string, error) {
	return func(raw string) (string, error) {
		n, err := strconv.Atoi(strings.TrimSpace(raw))
		if err != nil {
			return "", fmt.Errorf("%q is not a whole number", raw)
		}
		if n < min || n > max {
			return "", fmt.Errorf("%d is outside %d–%d", n, min, max)
		}
		return strconv.Itoa(n), nil
	}
}

func parseSweepFloat(min, max float64) func(string) (string, error) {
	return func(raw string) (string, error) {
		v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
		if err != nil {
			return "", fmt.Errorf("%q is not a number", raw)
		}
		if v < min || v > max {
			return "", fmt.Errorf("%g is outside %g–%g", v, min, max)
		}
		return strconv.FormatFloat(v, 'g', -1, 64), nil
	}
}

func parseSweepChoice(allowed ...string) func(string) (string, error) {
	return func(raw string) (string, error) {
		v := strings.TrimSpace(raw)
		for _, a := range allowed {
			if v == a {
				return v, nil
			}
		}
		return "", fmt.Errorf("%q is not one of %s", raw, strings.Join(allowed, ", "))
	}
}

// sweepFields is the registry, in the order the form shows them.
var sweepFields = []SweepField{
	{
		Name:    "max_model_len",
		Label:   "Context length",
		Help:    "Maps to --max-model-len. The KV cache is pre-allocated at this size, so it is usually the parameter that decides whether a model fits.",
		Example: "8192, 32768, 131072",
		Choices: []string{"4096", "8192", "16384", "32768", "65536", "131072"},
		parse:   parseSweepInt(256, 8_388_608),
		apply: func(o *ConfigOverrides, v string) error {
			n, err := strconv.Atoi(v)
			if err != nil {
				return err
			}
			o.MaxModelLen = &n
			return nil
		},
	},
	{
		Name:    "tensor_parallel_size",
		Label:   "Tensor parallel size",
		Help:    "How many GPUs the model is split across. Must divide the model's attention head count.",
		Example: "1, 2, 4",
		Choices: []string{"1", "2", "4", "8"},
		parse:   parseSweepInt(1, 16),
		apply: func(o *ConfigOverrides, v string) error {
			n, err := strconv.Atoi(v)
			if err != nil {
				return err
			}
			o.TensorParallelSize = &n
			return nil
		},
	},
	{
		Name:    "gpu_memory_utilization",
		Label:   "GPU memory utilization",
		Help:    "Fraction of each card vLLM may use. Higher leaves more room for KV cache and less for everything else.",
		Example: "0.85, 0.90, 0.95",
		Choices: []string{"0.80", "0.85", "0.90", "0.95"},
		parse:   parseSweepFloat(0.1, 0.99),
		apply: func(o *ConfigOverrides, v string) error {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return err
			}
			o.GPUMemoryUtilization = &f
			return nil
		},
	},
	{
		Name:    "max_num_seqs",
		Label:   "Max concurrent sequences",
		Help:    "How many requests the engine will batch. Raising it trades latency for throughput.",
		Example: "1, 16, 64",
		Choices: []string{"1", "4", "16", "32", "64", "128", "256"},
		parse:   parseSweepInt(1, 4096),
		apply: func(o *ConfigOverrides, v string) error {
			n, err := strconv.Atoi(v)
			if err != nil {
				return err
			}
			o.MaxNumSeqs = &n
			return nil
		},
	},
	{
		Name:    "kv_cache_dtype",
		Label:   "KV cache dtype",
		Help:    "FP8 halves the KV cache, which buys context length at some cost in quality.",
		Example: "auto, fp8",
		Choices: []string{"auto", "fp8", "fp8_e5m2", "fp8_e4m3"},
		parse:   parseSweepChoice("auto", "fp8", "fp8_e5m2", "fp8_e4m3"),
		apply: func(o *ConfigOverrides, v string) error {
			o.KVCacheDtype = &v
			return nil
		},
	},
	{
		Name:    "enforce_eager",
		Label:   "Enforce eager mode",
		Help:    "Skips graph capture. Slower steady-state, faster startup — worth measuring rather than assuming.",
		Example: "true, false",
		Choices: []string{"false", "true"},
		parse:   parseSweepChoice("true", "false"),
		apply: func(o *ConfigOverrides, v string) error {
			b := v == "true"
			o.EnforceEager = &b
			return nil
		},
	},
}

// SweepFields returns the sweepable parameters, in form order.
func SweepFields() []SweepField { return sweepFields }

// LookupSweepField finds a field by name.
func LookupSweepField(name string) (SweepField, bool) {
	for _, f := range sweepFields {
		if f.Name == name {
			return f, true
		}
	}
	return SweepField{}, false
}

// ParseSweepValues splits and normalizes a comma-separated value list. Order
// is preserved and duplicates are dropped, so "8192, 8192" is one point rather
// than the same reload twice.
func ParseSweepValues(field SweepField, raw string) ([]string, error) {
	var out []string
	seen := map[string]bool{}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := field.Parse(part)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", field.Label, err)
		}
		if seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out, nil
}

// MaxSweepCombinations caps the matrix. Every combination is a separate engine
// load, so one that would take days is worth refusing rather than starting.
//
// Exported because the form shows the count as it is built and disables submit
// past the limit: finding out at submission that a job is too big is worse
// than watching the number climb towards the cap.
const MaxSweepCombinations = 64

// ValidateSweeps checks a set of axes before a job is persisted, so a bad
// value is reported at submission rather than discovered several loads in.
func ValidateSweeps(axes []SweepAxis) error {
	seen := map[string]bool{}
	total := 1
	for _, a := range axes {
		f, ok := LookupSweepField(a.Field)
		if !ok {
			return fmt.Errorf("unknown sweep parameter %q", a.Field)
		}
		if seen[a.Field] {
			return fmt.Errorf("%s is swept twice", f.Label)
		}
		seen[a.Field] = true
		if len(a.Values) == 0 {
			return fmt.Errorf("%s has no values", f.Label)
		}
		for _, v := range a.Values {
			if _, err := f.Parse(v); err != nil {
				return err
			}
		}
		total *= len(a.Values)
	}
	if total > MaxSweepCombinations {
		return fmt.Errorf("that is %d combinations of sweep values; %d is the limit, since each one reloads the engine",
			total, MaxSweepCombinations)
	}
	return nil
}

// SweepCombinations expands the axes into every combination, as ordered maps
// of field to value. Returns a single empty combination when there are no
// axes, so callers can treat "no sweep" as one pass without a special case.
func SweepCombinations(axes []SweepAxis) []map[string]string {
	combos := []map[string]string{{}}
	for _, a := range axes {
		var next []map[string]string
		for _, base := range combos {
			for _, v := range a.Values {
				c := make(map[string]string, len(base)+1)
				for k, bv := range base {
					c[k] = bv
				}
				c[a.Field] = v
				next = append(next, c)
			}
		}
		combos = next
	}
	return combos
}

// ApplySweep overlays a cell's swept values on top of the job's base
// overrides, leaving the base untouched.
func ApplySweep(base *ConfigOverrides, values map[string]string) (*ConfigOverrides, error) {
	out := ConfigOverrides{}
	if base != nil {
		out = *base
	}
	// Sorted, so an error message names fields in a stable order.
	for _, k := range sortedKeys(values) {
		f, ok := LookupSweepField(k)
		if !ok {
			return nil, fmt.Errorf("unknown sweep parameter %q", k)
		}
		if err := f.Apply(&out, values[k]); err != nil {
			return nil, err
		}
	}
	return &out, nil
}

// SweepKey identifies a combination for grouping. Cells sharing a key share an
// engine configuration and so can be measured on one load.
func SweepKey(values map[string]string) string {
	if len(values) == 0 {
		return ""
	}
	parts := make([]string, 0, len(values))
	for _, k := range sortedKeys(values) {
		parts = append(parts, k+"="+values[k])
	}
	return strings.Join(parts, ",")
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

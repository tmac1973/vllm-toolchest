package benchmark

// Preset source values. PresetSourceInternal drives the in-Go HTTP loop;
// PresetSourceBenchy shells out to `uvx llama-benchy`. Empty defaults to
// internal so older preset definitions stay valid.
const (
	PresetSourceInternal = "internal"
	PresetSourceBenchy   = "benchy"
)

// Preset defines a benchmark parameter set.
type Preset struct {
	Name         string
	Label        string
	Description  string
	Source       string
	PromptTokens []int
	GenTokens    int
	Repetitions  int
	Concurrency  []int // benchy only; defaults to [1] when empty
}

// EffectiveSource returns the dispatch key, defaulting empty → internal.
func (p Preset) EffectiveSource() string {
	if p.Source == "" {
		return PresetSourceInternal
	}
	return p.Source
}

// Presets returns the available benchmark presets. Internal presets use
// the streaming Go HTTP loop (per-test-point progress, TTFT from first
// SSE chunk). Benchy presets shell out to llama-benchy for engine-agnostic
// comparison.
func Presets() []Preset {
	return []Preset{
		{
			Name:         "internal-quick",
			Label:        "internal-quick — 1 rep, 512-token prompt (~15s)",
			Description:  "Single streaming chat completion at 512-token prompt / 128 gen tokens. Sanity check after model load.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{512}, GenTokens: 128, Repetitions: 1,
		},
		{
			Name:         "internal-standard",
			Label:        "internal-standard — 3 reps × 3 prompt sizes (~2 min)",
			Description:  "Three streaming chat completions at 128 / 512 / 2048-token prompts (128 gen tokens). Captures TTFT and gen-TPS scaling across short contexts.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{128, 512, 2048}, GenTokens: 128, Repetitions: 3,
		},
		{
			Name:         "internal-thorough",
			Label:        "internal-thorough — 5 reps × 4 prompt sizes up to 8K (~8 min)",
			Description:  "Five repetitions at 128 / 512 / 2048 / 8192-token prompts with 256 generated tokens. Stresses long-context prefill.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{128, 512, 2048, 8192}, GenTokens: 256, Repetitions: 5,
		},
		{
			Name:         "internal-long-ctx",
			Label:        "internal-long-ctx — 1 rep, 32K prompt / 512 gen",
			Description:  "Single 32768-token prompt with 512 generated tokens. Stresses KV cache, paged attention, and KV-cache dtype on long contexts.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{32768}, GenTokens: 512, Repetitions: 1,
		},
		{
			Name:         "benchy-quick",
			Label:        "benchy-quick — 1 rep, 512 prompt / 32 gen via llama-benchy (~30s)",
			Description:  "Single-shot llama-benchy run. Smoke test for the engine-agnostic comparison path.",
			Source:       PresetSourceBenchy,
			PromptTokens: []int{512}, GenTokens: 32, Repetitions: 1, Concurrency: []int{1},
		},
		{
			Name:         "benchy-standard",
			Label:        "benchy-standard — 3 reps, 2048 prompt / 128 gen via llama-benchy (~2 min)",
			Description:  "Three-run llama-benchy benchmark at 2048-token prompts. Comparable to llama-toolchest's benchy-standard.",
			Source:       PresetSourceBenchy,
			PromptTokens: []int{2048}, GenTokens: 128, Repetitions: 3, Concurrency: []int{1},
		},
		{
			Name:         "benchy-concurrency",
			Label:        "benchy-concurrency — 3 reps × {1,2,4,8} concurrent via llama-benchy (~5 min)",
			Description:  "Stress vLLM's continuous batching by sweeping concurrency. Highlights where prefill saturates vs. throughput scales.",
			Source:       PresetSourceBenchy,
			PromptTokens: []int{1024}, GenTokens: 128, Repetitions: 3, Concurrency: []int{1, 2, 4, 8},
		},
	}
}

// GetPreset returns a preset by name, falling back to internal-standard
// when the name doesn't match. Unknown presets are tolerated rather than
// erroring so persisted runs from older versions still render.
func GetPreset(name string) Preset {
	for _, p := range Presets() {
		if p.Name == name {
			return p
		}
	}
	return Presets()[1] // internal-standard
}

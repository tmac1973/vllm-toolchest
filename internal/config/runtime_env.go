package config

import (
	"fmt"
	"sort"
	"strings"
)

// RuntimeEnvOption is a curated environment variable that measurably affects
// the vLLM process. Deliberately a fixed list rather than a free-form map:
// vLLM defines nearly 300 environment variables, most of them for tests,
// distributed serving topologies or hardware this tool never runs on, and
// offering all of them as a menu would be worse than offering none. Anything
// outside this list goes through the free-form Extra entry, which warns about
// known-harmful variables instead of refusing them.
//
// The environment applies to the vLLM process as a whole. vLLM serves one
// model per process, so "process-wide" and "per model" coincide here — but
// these values are stored globally and apply to whichever model is launched
// next, not to a particular model.
type RuntimeEnvOption struct {
	Name   string
	Label  string
	Help   string
	Values []string // allowed values; empty means free text
	// Platforms the variable affects; empty means all. Display only — there
	// is no platform filter, because the ROCm images are what this tool
	// manages and a one-value dropdown is noise.
	Platforms []string
	// Example fills the placeholder for free-text options.
	Example string
}

// RuntimeEnvOptions is the curated set.
//
// Every name, default and value set here was read out of the vLLM and PyTorch
// actually installed in the image rather than from documentation, because the
// two disagree. Two variables the original plan called for are absent for that
// reason: VLLM_ATTENTION_BACKEND and VLLM_USE_V1 no longer exist in vLLM —
// the backend is a launch flag (--attention-backend, set per model in the
// model config panel) and V0 is gone. Offering either would be a control that
// does nothing.
//
// Also deliberately absent: VLLM_USE_TRITON_AWQ, which the image already
// exports as 1 for every model, and the RADIANCE_* switches, which have their
// own Settings section with tri-state semantics this table cannot express.
// Both remain reachable through Extra environment, with a warning.
func RuntimeEnvOptions() []RuntimeEnvOption {
	onOff := []string{"", "0", "1"}
	return []RuntimeEnvOption{
		{
			Name:  "VLLM_LOGGING_LEVEL",
			Label: "vLLM log level",
			Help: "How much vLLM writes to its log. INFO is the default. DEBUG adds engine " +
				"configuration, the resolved attention backend, KV cache block counts and " +
				"CUDA-graph capture detail — the figures worth checking a VRAM estimate " +
				"against — at the cost of a much noisier log.",
			Values: []string{"", "DEBUG", "INFO", "WARNING", "ERROR"},
		},
		{
			Name:  "VLLM_DISABLE_COMPILE_CACHE",
			Label: "Disable torch.compile cache",
			Help: "The image ships this set to 1, so every start recompiles from scratch. " +
				"Setting it to 0 lets vLLM reuse compiled artifacts between restarts, which " +
				"is the single largest startup saving available — at the risk of a stale " +
				"artifact surviving a driver or vLLM upgrade. Clear " +
				"/data/cache/vllm if a start begins failing oddly after an upgrade.",
			Values: onOff,
		},
		{
			Name:      "VLLM_ROCM_USE_AITER",
			Label:     "AITER kernels (master switch)",
			Help:      "The main toggle for every AITER operation. vLLM defaults it off, so the AITER kernels compiled into this image are unused until this is 1. The individual AITER switches below have no effect while this is off — they are gated by it. Worth measuring on a benchmark before keeping either way.",
			Values:    onOff,
			Platforms: []string{"rocm"},
		},
		{
			Name:      "VLLM_ROCM_USE_AITER_MHA",
			Label:     "AITER multi-head attention",
			Help:      "AITER's attention kernels. On by default, but only once the master AITER switch above is enabled. Turn off to isolate an attention-kernel problem while leaving the rest of AITER on.",
			Values:    onOff,
			Platforms: []string{"rocm"},
		},
		{
			Name:      "VLLM_ROCM_USE_SKINNY_GEMM",
			Label:     "Skinny GEMM kernels",
			Help:      "Specialised kernels for the tall-thin matrix shapes that dominate single-stream token generation. On by default. Note the radiance image has its own skinny-GEMM switch in the Radiance section, which is a different implementation.",
			Values:    onOff,
			Platforms: []string{"rocm"},
		},
		{
			Name:      "ROCBLAS_USE_HIPBLASLT",
			Label:     "rocBLAS via hipBLASLt",
			Help:      "Routes rocBLAS calls through hipBLASLt where possible. The image ships this set to 1. Most quantized matmuls never reach rocBLAS, so the effect is smaller than it sounds — measure before changing it.",
			Values:    onOff,
			Platforms: []string{"rocm"},
		},
		{
			Name:      "TORCH_BLAS_PREFER_HIPBLASLT",
			Label:     "PyTorch BLAS via hipBLASLt",
			Help:      "The same preference one layer up, applied to PyTorch's own BLAS calls rather than rocBLAS's. Set both or neither; setting only one is a common way to end up measuring nothing.",
			Values:    onOff,
			Platforms: []string{"rocm"},
		},
		{
			Name:  "PYTORCH_HIP_ALLOC_CONF",
			Label: "HIP allocator options",
			Help: "Comma-separated allocator settings. expandable_segments:True is the useful " +
				"one: it lets the caching allocator grow a segment instead of reserving " +
				"fixed blocks, which reduces fragmentation on long-context and " +
				"variable-batch workloads. Note vLLM pre-allocates the KV cache from " +
				"gpu_memory_utilization, so this affects the space around the cache, not " +
				"the cache itself.",
			Example:   "expandable_segments:True",
			Platforms: []string{"rocm"},
		},
		{
			Name:      "NCCL_P2P_DISABLE",
			Label:     "Disable GPU peer-to-peer",
			Help:      "Forces collective operations through host memory instead of direct GPU-to-GPU transfers. Only matters at tensor parallel size 2 or more. A troubleshooting option for hangs and garbage output on consumer boards or with IOMMU enabled — it costs real bandwidth when P2P was working.",
			Values:    onOff,
			Platforms: []string{"rocm"},
		},
		{
			Name:      "NCCL_DEBUG",
			Label:     "Collective communication log",
			Help:      "RCCL reads the NCCL_* names on ROCm. WARN reports problems only; INFO prints the topology and algorithm chosen for each collective, which is how to confirm whether an all-reduce is taking the P2P path. Only meaningful at tensor parallel size 2 or more.",
			Values:    []string{"", "WARN", "INFO", "TRACE"},
			Platforms: []string{"rocm"},
		},
		{
			Name:  "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
			Label: "Allow context beyond trained length",
			Help: "Lets --max-model-len exceed the length in the model's config. vLLM refuses " +
				"by default. Output quality past the trained length degrades in ways that " +
				"are not obvious from a benchmark — a model will happily produce fluent " +
				"nonsense — so treat any measurement taken with this on as suspect.",
			Values: onOff,
		},
	}
}

// PlatformsLabel renders the option's platforms for display. Empty means the
// variable applies everywhere, which needs no annotation.
func (o RuntimeEnvOption) PlatformsLabel() string {
	if len(o.Platforms) == 0 {
		return ""
	}
	return strings.Join(o.Platforms, ", ")
}

// EnvSet is one scope's worth of environment configuration: values for the
// curated options plus a free-form block of KEY=VALUE lines.
type EnvSet struct {
	Curated map[string]string
	Extra   string
}

// envPair is one parsed KEY=VALUE entry.
type envPair struct {
	Name  string
	Value string
}

// parseExtraEnv parses the free-form block: one KEY=VALUE per line, blank
// lines and #-comments ignored. Malformed lines are returned separately so
// validation can point at them.
func parseExtraEnv(raw string) (pairs []envPair, malformed []string) {
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq <= 0 || !validEnvName(line[:eq]) {
			malformed = append(malformed, line)
			continue
		}
		pairs = append(pairs, envPair{Name: line[:eq], Value: line[eq+1:]})
	}
	return pairs, malformed
}

func validEnvName(name string) bool {
	for i, r := range name {
		switch {
		case r == '_', r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
			if i == 0 {
				return false
			}
		default:
			return false
		}
	}
	return name != ""
}

// Validate checks the curated values against their allowed sets and the
// free-form block for well-formed KEY=VALUE lines. Unknown variable names in
// Extra are accepted by design — the free-form entry exists for variables
// outside the curated list; known-harmful ones warn (see Warnings) but never
// block.
func (e EnvSet) Validate() error {
	allowed := make(map[string]RuntimeEnvOption, len(RuntimeEnvOptions()))
	for _, o := range RuntimeEnvOptions() {
		allowed[o.Name] = o
	}

	names := make([]string, 0, len(e.Curated))
	for k := range e.Curated {
		names = append(names, k)
	}
	sort.Strings(names)

	for _, name := range names {
		opt, ok := allowed[name]
		if !ok {
			return fmt.Errorf("unknown runtime environment variable %q", name)
		}
		v := strings.TrimSpace(e.Curated[name])
		if v == "" || len(opt.Values) == 0 {
			continue
		}
		valid := false
		for _, allowedValue := range opt.Values {
			if v == allowedValue {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("%s: %q is not one of %s", opt.Label, v, strings.Join(opt.Values[1:], ", "))
		}
	}

	if _, malformed := parseExtraEnv(e.Extra); len(malformed) > 0 {
		return fmt.Errorf("extra environment: not KEY=VALUE: %q", malformed[0])
	}
	return nil
}

// riskyEnvVars are variables that are legal to set but conflict with a
// mechanism this tool relies on. They warn — never block — because each has a
// legitimate use in the right situation.
//
// Precedence is what makes these worth warning about: the vLLM process is
// launched with the container environment plus these values appended, and the
// later entry wins, so anything set here overrides what the image exports.
var riskyEnvVars = map[string]string{
	"HIP_VISIBLE_DEVICES":  "hides GPUs from vLLM, so tensor parallel size and the GPU indices shown on the dashboard stop agreeing with what the engine sees",
	"ROCR_VISIBLE_DEVICES": "hides GPUs from vLLM, so tensor parallel size and the GPU indices shown on the dashboard stop agreeing with what the engine sees",
	"CUDA_VISIBLE_DEVICES": "hides GPUs from vLLM, so tensor parallel size and the GPU indices shown on the dashboard stop agreeing with what the engine sees",
	"HSA_OVERRIDE_GFX_VERSION": "only needed for GPUs ROCm doesn't support natively; on a supported card it selects kernels built for a different architecture, " +
		"and it also changes the device name vLLM uses to look up tuned kernel configs, so the Tuning tab's output stops being found",
	"HF_HOME":                      "moves the HuggingFace cache this tool downloads into and scans; models already downloaded become invisible to the Models page",
	"HF_HUB_CACHE":                 "moves the HuggingFace cache this tool downloads into and scans; models already downloaded become invisible to the Models page",
	"VLLM_TARGET_DEVICE":           "selects the backend vLLM is compiled for; it is read at build time, so setting it at runtime cannot change anything",
	"VLLM_WORKER_MULTIPROC_METHOD": "the process manager sets this to spawn so worker output is captured into the log panel; changing it loses child-process logs",
	"PYTHONUNBUFFERED":             "the process manager sets this so errors reach the log panel as they happen rather than at exit",
}

// radiancePrefix marks the variables owned by the Radiance settings section.
const radiancePrefix = "RADIANCE_"

// Warnings returns one message per known-risky variable present in the set,
// plus one per RADIANCE_* variable, which the Radiance section owns. The
// variables still apply as described — this is information, not enforcement.
func (e EnvSet) Warnings() []string {
	var out []string
	seen := map[string]bool{}
	note := func(name string) {
		if seen[name] {
			return
		}
		if reason, ok := riskyEnvVars[name]; ok {
			seen[name] = true
			out = append(out, fmt.Sprintf("%s: %s", name, reason))
			return
		}
		if strings.HasPrefix(name, radiancePrefix) {
			seen[name] = true
			out = append(out, fmt.Sprintf(
				"%s: the Radiance section owns this variable and is applied after this one, so a value set there wins", name))
		}
	}
	curated := make([]string, 0, len(e.Curated))
	for k, v := range e.Curated {
		if strings.TrimSpace(v) != "" {
			curated = append(curated, k)
		}
	}
	sort.Strings(curated)
	for _, k := range curated {
		note(k)
	}
	pairs, _ := parseExtraEnv(e.Extra)
	for _, p := range pairs {
		note(p.Name)
	}
	return out
}

// Pairs renders the set as KEY=VALUE strings, skipping blank curated values so
// an unset option means "don't touch the environment" rather than "set it to
// empty". Free-form entries override curated ones with the same name. Sorted
// so a launch command is stable.
func (e EnvSet) Pairs() []string {
	merged := map[string]string{}
	for k, v := range e.Curated {
		if strings.TrimSpace(v) != "" {
			merged[k] = v
		}
	}
	pairs, _ := parseExtraEnv(e.Extra)
	for _, p := range pairs {
		merged[p.Name] = p.Value
	}
	if len(merged) == 0 {
		return nil
	}
	out := make([]string, 0, len(merged))
	for k, v := range merged {
		out = append(out, k+"="+v)
	}
	sort.Strings(out)
	return out
}

// EnvSet returns the global runtime environment as the reusable component
// type.
func (c *Config) EnvSet() EnvSet {
	return EnvSet{Curated: c.RuntimeEnv, Extra: c.RuntimeEnvExtra}
}

// RuntimeEnvPairs renders the configured variables as KEY=VALUE strings for
// the vLLM launch.
func (c *Config) RuntimeEnvPairs() []string {
	return c.EnvSet().Pairs()
}

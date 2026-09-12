package api

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmac1973/vllm-toolchest/variants"
)

// A model's launch config names an attention backend in three places, and only
// one of them is the picker:
//
//	attention_backend                        the main model (a select)
//	speculative_config.attention_backend     the MTP drafter (free-text JSON)
//	extra_flags: --attention-backend X       an escape hatch (free text)
//
// The picker was made safe by building it from the running variant's manifest.
// The other two were not, and the failure is identical: a value carried over
// from a different image names a backend this one does not have, and the engine
// dies on the first load. It is worse from the drafter, because the main model
// has already loaded by then and the error arrives four stack traces deep.
//
// Found in the wild: a config written on radiance kept
// speculative_config.attention_backend=ROCM_AITER_UNIFIED_ATTN, and on an image
// built with VLLM_ROCM_USE_AITER=0 every worker died with
// "ModuleNotFoundError: No module named 'aiter'".

// backendRef is one attention backend named somewhere in a launch config.
type backendRef struct {
	Value string // the backend name
	Where string // human-readable location, for the message
}

// namedBackends finds every attention backend a config refers to, other than
// the picker's own field. Best-effort by design: malformed JSON in the
// speculative block is somebody else's error to report, not a reason to skip
// the check on the parts that did parse.
func namedBackends(specJSON, extraFlags string) []backendRef {
	var out []backendRef

	if s := strings.TrimSpace(specJSON); s != "" {
		var spec map[string]any
		if json.Unmarshal([]byte(s), &spec) == nil {
			if v, ok := spec["attention_backend"].(string); ok && v != "" {
				out = append(out, backendRef{v, "the speculative config's attention_backend"})
			}
		}
	}

	// --attention-backend X, --attention-backend=X, and the same with the
	// single-dash spelling vLLM also accepts.
	fields := strings.Fields(extraFlags)
	for i, f := range fields {
		name, val, hasEq := strings.Cut(f, "=")
		if name != "--attention-backend" && name != "-attention-backend" {
			continue
		}
		if !hasEq {
			if i+1 >= len(fields) {
				continue
			}
			val = fields[i+1]
		}
		if val != "" {
			out = append(out, backendRef{val, "--attention-backend in the extra flags"})
		}
	}
	return out
}

// validateNamedBackends rejects a backend this image's variant does not
// declare. It is deliberately silent when the running variant has no manifest
// or declares no backends: we cannot then tell a wrong name from one we simply
// have no record of, and refusing a config on a guess would be worse than
// letting the engine report it.
func validateNamedBackends(d variants.Descriptor, known bool, specJSON, extraFlags string) error {
	if !known || len(d.AttentionBackends) == 0 {
		return nil
	}
	allowed := map[string]bool{}
	names := make([]string, 0, len(d.AttentionBackends))
	for _, b := range d.AttentionBackends {
		allowed[b.Value] = true
		names = append(names, b.Value)
	}

	for _, ref := range namedBackends(specJSON, extraFlags) {
		if allowed[ref.Value] {
			continue
		}
		return fmt.Errorf(
			"%s names %s, which the %s image does not have — it would abort the engine on load. "+
				"Remove the key to let vLLM choose, or use one of: %s",
			ref.Where, ref.Value, d.ID, strings.Join(names, ", "))
	}
	return nil
}

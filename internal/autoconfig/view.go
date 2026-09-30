package autoconfig

import (
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// Label is how a row is named in the review: the setting, the flag or the
// variable.
func (r Row) Label() string {
	switch r.Kind {
	case RowFlag:
		return r.Flag[0]
	case RowEnv:
		k, _, _ := strings.Cut(r.Env, "=")
		return k
	}
	return r.Field
}

// Proposed is the value the row would set, as text.
func (r Row) Proposed() string {
	switch r.Kind {
	case RowFlag:
		if len(r.Flag) > 1 {
			return r.Flag[1]
		}
		return "on"
	case RowEnv:
		_, v, _ := strings.Cut(r.Env, "=")
		return v
	}
	return r.Value
}

// Current is what the config has now for the row's setting, "" when it has
// nothing.
func (r Row) Current(c models.VLLMConfig) string {
	switch r.Kind {
	case RowFlag:
		args := process.SplitFlags(c.ExtraFlags)
		for i, a := range args {
			name, value, inline := strings.Cut(a, "=")
			if name != r.Flag[0] {
				continue
			}
			switch {
			case inline:
				return value
			case i+1 < len(args) && !strings.HasPrefix(args[i+1], "-"):
				return args[i+1]
			}
			return "on"
		}
		return ""
	case RowEnv:
		key, _, _ := strings.Cut(r.Env, "=")
		for _, l := range strings.Split(c.Env, "\n") {
			if k, v, ok := strings.Cut(strings.TrimSpace(l), "="); ok && k == key {
				return v
			}
		}
		return ""
	}
	v, _ := fieldText(c, r.Field)
	if v == "false" || v == "0" {
		return ""
	}
	return v
}

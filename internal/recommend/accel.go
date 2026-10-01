package recommend

import (
	"slices"
	"strings"
)

// acceleratedByArch is what each GPU architecture runs natively rather than
// by unpacking to bf16 first. Anything unlisted is "not accelerated": claiming
// speed that is not there is worse than leaving out speed that is.
var acceleratedByArch = map[string][]string{
	"gfx1200": {"fp8"}, "gfx1201": {"fp8"},
	"gfx942": {"fp8"}, "gfx950": {"fp8"},
	"gfx1100": nil, "gfx1101": nil, "gfx1102": nil,
	"sm_89": {"fp8"}, "sm_90": {"fp8"}, "sm_100": {"fp8"}, "sm_120": {"fp8"},
	"sm_80": nil, "sm_86": nil,
}

// acceleratedFormats is an architecture's accelerated formats, lowercase,
// less those the image has no kernels for.
func acceleratedFormats(arch string, noAccel []string) []string {
	var out []string
	for _, f := range acceleratedByArch[strings.ToLower(arch)] {
		if !slices.ContainsFunc(noAccel, func(n string) bool { return strings.EqualFold(n, f) }) {
			out = append(out, f)
		}
	}
	return out
}

// isAccelerated reports a format badge among the accelerated formats.
func isAccelerated(format string, accelerated []string) bool {
	return slices.ContainsFunc(accelerated, func(a string) bool { return strings.EqualFold(a, format) })
}

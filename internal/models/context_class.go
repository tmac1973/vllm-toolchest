package models

// ContextClass is the context length an operator asks autoconfigure for, in
// the four sizes the start panel offers. It is a class rather than a number
// because the answer depends on the machine: "Long" is a request, and what the
// planner grants is what fits.
//
// It lives here, rather than with the planner, because an Autoconfig profile
// records the class it was built for, and a later refinement aims at it.
type ContextClass string

const (
	ContextShort  ContextClass = "short"  // 8,192 tokens
	ContextMedium ContextClass = "medium" // 32,768
	ContextLong   ContextClass = "long"   // 131,072
	ContextMax    ContextClass = "max"    // the model's own maximum
)

// Tokens is the target for a class, or 0 for ContextMax, whose target is the
// model's own and has to be read from its config.
func (c ContextClass) Tokens() int {
	switch c {
	case ContextShort:
		return 8192
	case ContextLong:
		return 131072
	case ContextMax:
		return 0
	default:
		return 32768
	}
}

// ParseContextClass reads a class from a form value. Anything unrecognised is
// medium, the panel's default, so a malformed request still asks for something
// sensible rather than for nothing.
func ParseContextClass(s string) ContextClass {
	switch c := ContextClass(s); c {
	case ContextShort, ContextMedium, ContextLong, ContextMax:
		return c
	}
	return ContextMedium
}

package process

import "strings"

// JoinFlags is the write side of SplitFlags: it quotes an argument that
// would not survive being split -- one with whitespace, a quote, a brace or a
// backslash -- so that SplitFlags(JoinFlags(a)) is a.
func JoinFlags(args []string) string {
	out := make([]string, 0, len(args))
	for _, a := range args {
		out = append(out, quoteArg(a))
	}
	return strings.Join(out, " ")
}

func quoteArg(a string) string {
	if a != "" && !strings.ContainsAny(a, " \t\n\r'\"\\{}[]") {
		return a
	}
	if !strings.Contains(a, "'") {
		return "'" + a + "'"
	}
	// Double quotes, escaping what SplitFlags unescapes inside them.
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`)
	return `"` + r.Replace(a) + `"`
}

// SetFlag puts flag -- a name and, optionally, its value -- into an extra-flags
// string, replacing an existing occurrence of the same name and its value,
// or appending. Everything else in extra is kept as it was split.
func SetFlag(extra string, flag []string) string {
	if len(flag) == 0 {
		return extra
	}
	args, found := replaceFlag(SplitFlags(extra), flag[0], flag)
	if !found {
		args = append(args, flag...)
	}
	return JoinFlags(args)
}

// RemoveFlag removes a flag and its value from an extra-flags string.
func RemoveFlag(extra, name string) string {
	args, _ := replaceFlag(SplitFlags(extra), name, nil)
	return JoinFlags(args)
}

// HasFlag reports whether an extra-flags string contains the flag.
func HasFlag(extra, name string) bool {
	_, found := replaceFlag(SplitFlags(extra), name, nil)
	return found
}

// replaceFlag swaps every occurrence of name (with its value, when it has
// one) for with. A value is the next argument when it does not begin with
// "-", or what follows "=".
func replaceFlag(args []string, name string, with []string) (out []string, found bool) {
	for i := 0; i < len(args); i++ {
		a := args[i]
		n, _, inline := strings.Cut(a, "=")
		if n != name {
			out = append(out, a)
			continue
		}
		if !inline && i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			i++
		}
		if !found {
			out = append(out, with...)
		}
		found = true
	}
	return out, found
}

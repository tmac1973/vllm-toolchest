package api

import (
	"fmt"
	"html"
	"io"
)

// The HTML fragments in this package are assembled with Printf rather than
// templates, and the values going into them are not trustworthy: model IDs and
// display names come from HuggingFace, and every per-model config field is
// whatever the operator typed.
//
// Interpolating those raw is how a JSON speculative config
//
//	{"method":"mtp","num_speculative_tokens":8}
//
// pasted into a text box came back as just `{` -- its first double quote closed
// the value="..." attribute it was being written into, and the browser threw
// away the rest. The same hole lets a model named `x"><script>...` run script
// in the page.
//
// So escaping is the default here, and HTML has to be opted into.

// safeHTML marks a string that is already valid HTML and must not be escaped
// again -- a badge, an attribute fragment like " checked". Converting to this
// type is the explicit claim that the value is trusted.
type safeHTML string

// htmlPrinter returns a Printf-like function that HTML-escapes every string
// argument before interpolating it. Values of type safeHTML pass through
// untouched.
//
// Escaping the arguments rather than the format string is what makes this
// safe by default: the format is a literal in the source, the arguments are
// the part that came from outside.
func htmlPrinter(w io.Writer) func(string, ...any) {
	return func(format string, a ...any) {
		for i, v := range a {
			switch t := v.(type) {
			case safeHTML:
				a[i] = string(t)
			case string:
				a[i] = html.EscapeString(t)
			case error:
				// Error text routinely quotes the input that caused it, so
				// it carries the same risk as the input itself.
				a[i] = html.EscapeString(t.Error())
			case fmt.Stringer:
				a[i] = html.EscapeString(t.String())
			}
		}
		fmt.Fprintf(w, format, a...)
	}
}

// esc escapes a single value for use in HTML text or a quoted attribute.
// Prefer htmlPrinter; this is for the few places that build a string first.
func esc(s string) string { return html.EscapeString(s) }

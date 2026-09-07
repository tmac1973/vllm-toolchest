package api

import (
	"errors"
	"strings"
	"testing"
)

func TestHTMLPrinterEscapesByType(t *testing.T) {
	for _, tc := range []struct {
		name   string
		format string
		arg    any
		want   string
	}{
		{
			// The reported bug: JSON in a value="..." attribute.
			name:   "string in an attribute",
			format: `<input value="%s">`,
			arg:    `{"method":"mtp"}`,
			want:   `<input value="{&#34;method&#34;:&#34;mtp&#34;}">`,
		},
		{
			name:   "attribute break-out is neutralised",
			format: `<a title="%s">x</a>`,
			arg:    `"><script>alert(1)</script>`,
			want:   `<a title="&#34;&gt;&lt;script&gt;alert(1)&lt;/script&gt;">x</a>`,
		},
		{
			// The old hand-rolled escaper missed this one, and single-quoted
			// attributes exist in these templates.
			name:   "single quote is escaped",
			format: `<input placeholder='%s'>`,
			arg:    `it's`,
			want:   `<input placeholder='it&#39;s'>`,
		},
		{
			name:   "error text is escaped too",
			format: `<p>%s</p>`,
			arg:    errors.New(`bad model <script>`),
			want:   `<p>bad model &lt;script&gt;</p>`,
		},
		{
			// safeHTML is the explicit opt-out for trusted fragments.
			name:   "safeHTML passes through",
			format: `<span>%s</span>`,
			arg:    safeHTML(`<del>missing</del>`),
			want:   `<span><del>missing</del></span>`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var b strings.Builder
			htmlPrinter(&b)(tc.format, tc.arg)
			if got := b.String(); got != tc.want {
				t.Errorf("\n got %s\nwant %s", got, tc.want)
			}
		})
	}
}

// Non-string arguments must still format normally.
func TestHTMLPrinterLeavesNonStringsAlone(t *testing.T) {
	var b strings.Builder
	htmlPrinter(&b)(`<td>%d</td><td>%.2f</td><td>%v</td>`, 42, 3.14159, true)
	if got, want := b.String(), `<td>42</td><td>3.14</td><td>true</td>`; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestEscHandlesAllFiveEntities(t *testing.T) {
	got := esc(`<>&"'`)
	want := "&lt;&gt;&amp;&#34;&#39;"
	if got != want {
		t.Errorf("esc() = %s, want %s", got, want)
	}
}

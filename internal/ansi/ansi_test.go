package ansi

import "testing"

func TestStrip(t *testing.T) {
	const esc = "\x1b"

	for _, tc := range []struct {
		name string
		in   string
		want string
	}{
		{"plain text is untouched", "INFO starting vLLM", "INFO starting vLLM"},
		{"empty", "", ""},

		// The line the user actually saw.
		{
			name: "radiance banner heading",
			in:   esc + "[38;5;39;1m───[ GPUs ]──────────────" + esc + "[0m",
			want: "───[ GPUs ]──────────────",
		},
		// Square brackets are literal banner content, not escapes: only a
		// bracket immediately after ESC starts a sequence.
		{
			name: "literal brackets survive",
			in:   esc + "[32m✓" + esc + "[0m  [0] AMD Radeon R9700  gfx1201",
			want: "✓  [0] AMD Radeon R9700  gfx1201",
		},
		{"simple colour", esc + "[31mred" + esc + "[0m", "red"},
		{"bare reset", esc + "[0m", ""},

		// tqdm repaints with erase-line and cursor moves.
		{"erase line", "progress" + esc + "[K", "progress"},
		{"cursor up", esc + "[2Amoved", "moved"},

		// OSC, both terminators.
		{"osc with BEL", esc + "]0;title\x07text", "text"},
		{"osc with ST", esc + "]8;;http://x" + esc + "\\link", "link"},

		{"two-byte escape", esc + "(Btext", "text"},

		// Degenerate input must not panic or eat the line.
		{"trailing ESC", "text" + esc, "text"},
		{"unterminated CSI", "text" + esc + "[38;5", "text"},
		{"unterminated OSC", "text" + esc + "]0;no-terminator", "text"},

		// UTF-8 either side of a sequence must survive intact.
		{
			name: "utf8 preserved around escapes",
			in:   "└─" + esc + "[1m日本語 ✓ ─" + esc + "[0m─┘",
			want: "└─日本語 ✓ ──┘",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := Strip(tc.in); got != tc.want {
				t.Errorf("Strip(%q)\n got %q\nwant %q", tc.in, got, tc.want)
			}
		})
	}
}

// A line with no escapes must come back as the identical string, since that is
// the case that runs on essentially every log line.
func TestStripFastPathReturnsInput(t *testing.T) {
	in := "INFO 09-07 10:12:03 [core.py:512] Waiting for init message"
	if got := Strip(in); got != in {
		t.Errorf("fast path altered the line: %q", got)
	}
}

func TestStripIsIdempotent(t *testing.T) {
	in := "\x1b[38;5;39;1m───[ GPUs ]───\x1b[0m"
	once := Strip(in)
	if twice := Strip(once); twice != once {
		t.Errorf("second pass changed the result: %q -> %q", once, twice)
	}
}

func BenchmarkStripClean(b *testing.B) {
	line := "INFO 09-07 10:12:03 [core.py:512] Waiting for init message"
	for i := 0; i < b.N; i++ {
		_ = Strip(line)
	}
}

// Package ansi removes terminal control sequences from captured output.
//
// The processes vllmctl supervises assume they are attached to a terminal and
// colour their output accordingly -- the radiance startup banner draws boxes in
// 256-colour SGR, vLLM highlights warnings, tqdm repaints progress bars. None
// of that survives a trip through the log buffer to a browser: the escapes are
// not rendered, they are displayed, so a banner line arrives looking like
//
//	ESC[38;5;39;1m───[ GPUs ]───────ESC[0m
//
// Stripping at ingest fixes more than the display. The same lines are matched
// against readiness and out-of-memory patterns, and are handed to the clipboard
// by the log panel's copy button; an embedded escape breaks a match and a copied
// log full of them is unreadable.
package ansi

import "strings"

// Strip removes ANSI escape sequences from s.
//
// Handles the three forms these processes actually emit: CSI (colour, cursor
// movement, line erase), OSC (window titles, hyperlinks), and bare two-byte
// escapes. Text is copied through byte-wise, so UTF-8 -- including the box
// drawing characters the banner is made of -- passes through untouched.
func Strip(s string) string {
	// The overwhelming majority of log lines contain no escapes at all, and
	// this runs on every line of every stream.
	if !strings.ContainsRune(s, 0x1b) {
		return s
	}

	var b strings.Builder
	b.Grow(len(s))

	for i := 0; i < len(s); {
		if s[i] != 0x1b {
			b.WriteByte(s[i])
			i++
			continue
		}
		// A trailing ESC has no sequence to skip; drop it.
		if i+1 >= len(s) {
			break
		}

		switch s[i+1] {
		case '[':
			// CSI: parameter and intermediate bytes (0x20-0x3F), then a
			// final byte (0x40-0x7E) that ends the sequence.
			j := i + 2
			for j < len(s) && s[j] >= 0x20 && s[j] <= 0x3f {
				j++
			}
			if j < len(s) && s[j] >= 0x40 && s[j] <= 0x7e {
				j++
			}
			i = j
		case ']':
			// OSC: runs until BEL or a string terminator (ESC \).
			j := i + 2
			for j < len(s) {
				if s[j] == 0x07 {
					j++
					break
				}
				if s[j] == 0x1b && j+1 < len(s) && s[j+1] == '\\' {
					j += 2
					break
				}
				j++
			}
			i = j
		default:
			// Everything else: zero or more intermediate bytes (0x20-0x2F)
			// followed by one final byte. Not always two bytes total --
			// charset selection is ESC ( B, three bytes -- and assuming it
			// was left the final byte behind as stray text.
			j := i + 1
			for j < len(s) && s[j] >= 0x20 && s[j] <= 0x2f {
				j++
			}
			if j < len(s) {
				j++
			}
			i = j
		}
	}

	return b.String()
}

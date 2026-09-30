package autoconfig

import (
	"html"
	"regexp"
	"slices"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// Command is one serve command found in a model card.
type Command struct {
	// Raw is the command as written, continuation lines joined.
	Raw string
	// Model is the positional model argument, for information: this tool
	// always serves the model's own files.
	Model string
	// Args are the arguments after the model.
	Args []string
	// Env is KEY=VALUE, in the order the card gives them.
	Env []string
	// Docker marks a command taken from a docker run, whose image's
	// entrypoint is the server.
	Docker bool
	// Partial marks a fragment whose model is a placeholder ("vllm serve
	// ... --flag"): it documents one option, and is not a command to run.
	Partial bool
}

// CommandSettings are the facts in a command, sorted by whether this tool has
// a field for them. Which become proposals is decided elsewhere.
type CommandSettings struct {
	Start process.VLLMStartConfig
	// Rest holds each flag this tool has no field for, with its value.
	Rest [][]string
	Env  []string
}

// Settings sorts a command's arguments into known fields and the rest.
func (c Command) Settings() CommandSettings {
	start, rest := process.ParseArgs(c.Args)
	return CommandSettings{Start: start, Rest: groupFlags(rest), Env: c.Env}
}

// groupFlags pairs each flag with the value that followed it.
func groupFlags(args []string) [][]string {
	var out [][]string
	for _, a := range args {
		if strings.HasPrefix(a, "-") || len(out) == 0 {
			out = append(out, []string{a})
			continue
		}
		out[len(out)-1] = append(out[len(out)-1], a)
	}
	return out
}

var (
	placeholder = regexp.MustCompile(`<[^<>\n]+>`)
	fenceLine   = regexp.MustCompile("^\\s*(```|~~~)")
	preBlock    = regexp.MustCompile(`(?is)<pre\b[^>]*>(.*?)</pre>`)
	envAssign   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=`)
)

// codeBlocks returns the text of every markdown fence and HTML <pre> region
// in a card, tags inside them removed and entities unescaped.
func codeBlocks(raw string) []string {
	raw = strings.ReplaceAll(raw, "\r\n", "\n")
	var blocks []string
	for _, m := range preBlock.FindAllStringSubmatch(raw, -1) {
		blocks = append(blocks, html.UnescapeString(htmlTag.ReplaceAllString(m[1], "")))
	}
	var cur []string
	in := false
	for _, l := range strings.Split(raw, "\n") {
		if fenceLine.MatchString(l) {
			if in {
				blocks = append(blocks, strings.Join(cur, "\n"))
				cur = nil
			}
			in = !in
			continue
		}
		if in {
			cur = append(cur, l)
		}
	}
	return blocks
}

// closePlaceholders removes the spaces inside a placeholder such as
// "<nvme path>", which a card writes where the reader fills in their own
// path. Split on its space, it would read as two arguments and end the docker
// options early.
func closePlaceholders(line string) string {
	return placeholder.ReplaceAllStringFunc(line, func(p string) string {
		return strings.ReplaceAll(p, " ", "_")
	})
}

// logicalLines joins backslash continuations and drops comment lines, which
// in a card's code block separate one command from the next.
func logicalLines(block string) []string {
	var out []string
	var cur strings.Builder
	for _, l := range strings.Split(block, "\n") {
		t := strings.TrimSpace(l)
		if strings.HasPrefix(t, "#") {
			continue
		}
		if strings.HasSuffix(t, "\\") {
			cur.WriteString(strings.TrimSuffix(t, "\\"))
			cur.WriteByte(' ')
			continue
		}
		cur.WriteString(t)
		if s := strings.TrimSpace(cur.String()); s != "" {
			out = append(out, s)
		}
		cur.Reset()
	}
	if s := strings.TrimSpace(cur.String()); s != "" {
		out = append(out, s)
	}
	return out
}

// shellEnd cuts a command at the first token that ends it.
func shellEnd(tokens []string) []string {
	for i, t := range tokens {
		switch t {
		case "&", "&&", "|", "||", ";", ">", ">>", "2>&1":
			return tokens[:i]
		}
	}
	return tokens
}

// ExtractCommands finds every serve command in a card, in order of first
// appearance, without repeats.
//
// Two forms are recognised. A direct command contains "vllm serve" or the
// OpenAI api_server module, optionally after KEY=VALUE assignments. A docker
// command is a `docker run` (or `podman run`) whose arguments after the
// image carry at least two flags this tool recognises: an image whose
// entrypoint is the server is how the cards that matter most are written,
// with no "vllm" in the command at all.
func ExtractCommands(raw string) []Command {
	var out []Command
	for _, block := range codeBlocks(raw) {
		var exported []string
		for _, line := range logicalLines(block) {
			tokens := shellEnd(process.SplitFlags(closePlaceholders(line)))
			if len(tokens) == 0 {
				continue
			}
			if tokens[0] == "export" {
				for _, t := range tokens[1:] {
					if envAssign.MatchString(t) {
						exported = append(exported, t)
					}
				}
				continue
			}
			// Docker first: "docker run image vllm serve m" is a docker
			// command whose environment comes from its -e options.
			cmd, ok := dockerCommand(tokens)
			if !ok {
				cmd, ok = directCommand(tokens)
			}
			if !ok {
				continue
			}
			cmd.Raw = line
			if !cmd.Docker {
				cmd.Env = append(append([]string{}, exported...), cmd.Env...)
			}
			if !containsCommand(out, cmd) {
				out = append(out, cmd)
			}
		}
	}
	return out
}

func containsCommand(cmds []Command, c Command) bool {
	for _, x := range cmds {
		if slices.Equal(x.Args, c.Args) && slices.Equal(x.Env, c.Env) && x.Model == c.Model {
			return true
		}
	}
	return false
}

func directCommand(tokens []string) (Command, bool) {
	var cmd Command
	for i, t := range tokens {
		switch {
		case t == "vllm" && i+1 < len(tokens) && tokens[i+1] == "serve":
			collectEnv(&cmd, tokens[:i])
			cmd.Model, cmd.Args = modelAndArgs(tokens[i+2:])
			cmd.Partial = isPlaceholder(cmd.Model)
			return cmd, true
		case strings.HasSuffix(t, "vllm.entrypoints.openai.api_server"):
			collectEnv(&cmd, tokens[:i])
			cmd.Args = tokens[i+1:]
			if j := slices.Index(cmd.Args, "--model"); j >= 0 && j+1 < len(cmd.Args) {
				cmd.Model = cmd.Args[j+1]
			}
			cmd.Partial = isPlaceholder(cmd.Model)
			return cmd, true
		}
	}
	return Command{}, false
}

func collectEnv(cmd *Command, tokens []string) {
	for _, t := range tokens {
		if envAssign.MatchString(t) {
			cmd.Env = append(cmd.Env, t)
		}
	}
}

// modelAndArgs splits the positional model off the front of serve arguments.
func modelAndArgs(tokens []string) (string, []string) {
	if len(tokens) > 0 && !strings.HasPrefix(tokens[0], "-") {
		return tokens[0], tokens[1:]
	}
	return "", tokens
}

// dockerValueless are the docker run options that take no value.
var dockerValueless = map[string]bool{
	"--rm": true, "-i": true, "-t": true, "-it": true, "-ti": true, "-d": true,
	"--privileged": true, "--init": true, "--read-only": true, "-P": true,
	"--interactive": true, "--tty": true, "--detach": true,
}

func dockerCommand(tokens []string) (Command, bool) {
	if len(tokens) < 3 || (tokens[0] != "docker" && tokens[0] != "podman") || tokens[1] != "run" {
		return Command{}, false
	}
	cmd := Command{Docker: true}
	i := 2
	for i < len(tokens) {
		t := tokens[i]
		if !strings.HasPrefix(t, "-") {
			break // the image
		}
		name, value, inline := strings.Cut(t, "=")
		takes := !inline && !dockerValueless[name]
		if (name == "-e" || name == "--env") && (inline || i+1 < len(tokens)) {
			if !inline {
				value = tokens[i+1]
			}
			if envAssign.MatchString(value) {
				cmd.Env = append(cmd.Env, value)
			}
		}
		i++
		if takes && i < len(tokens) {
			i++
		}
	}
	if i >= len(tokens) {
		return Command{}, false
	}
	after := tokens[i+1:] // everything after the image
	if len(after) >= 2 && after[0] == "vllm" && after[1] == "serve" {
		after = after[2:]
	}
	cmd.Model, cmd.Args = modelAndArgs(after)
	if knownFlagCount(cmd.Args) < 2 {
		return Command{}, false
	}
	cmd.Partial = isPlaceholder(cmd.Model)
	return cmd, true
}

func knownFlagCount(args []string) int {
	known := process.KnownServeFlags()
	n := 0
	for _, a := range args {
		name, _, _ := strings.Cut(a, "=")
		if slices.Contains(known, name) {
			n++
		}
	}
	return n
}

// isPlaceholder reports a model argument that stands for something else:
// "...", or "<path>" with no path in it.
func isPlaceholder(model string) bool {
	m := strings.TrimSpace(model)
	switch {
	case m == "":
		return false
	case strings.Trim(m, ".…") == "":
		return true
	case strings.HasPrefix(m, "<") && strings.HasSuffix(m, ">") && !strings.Contains(m, "/"):
		return true
	}
	return false
}

var (
	inlineBacktick = regexp.MustCompile("`([^`\n]+)`")
	inlineCodeTag  = regexp.MustCompile(`(?is)<code\b[^>]*>(.*?)</code>`)
)

// InlineFlags finds serve flags mentioned outside any command, in inline code
// such as "add `--speculative-config '{…}'`". Each is returned with its value.
//
// Only flags this tool recognises are collected. A code span like `--help`,
// or a flag for another server, is far more often an aside than a
// recommendation, and only a flag that maps to a setting is worth a row.
func InlineFlags(raw string) [][]string {
	raw = strings.ReplaceAll(raw, "\r\n", "\n")
	raw = preBlock.ReplaceAllString(raw, "")
	raw = removeFences(raw)

	var spans []string
	for _, m := range inlineBacktick.FindAllStringSubmatch(raw, -1) {
		spans = append(spans, m[1])
	}
	for _, m := range inlineCodeTag.FindAllStringSubmatch(raw, -1) {
		spans = append(spans, html.UnescapeString(htmlTag.ReplaceAllString(m[1], "")))
	}

	known := process.KnownServeFlags()
	var out [][]string
	for _, span := range spans {
		tokens := process.SplitFlags(span)
		if len(tokens) == 0 || !strings.HasPrefix(tokens[0], "--") {
			continue
		}
		name, value, inline := strings.Cut(tokens[0], "=")
		if !slices.Contains(known, name) {
			continue
		}
		flag := []string{name}
		switch {
		case inline:
			flag = append(flag, value)
		case len(tokens) > 1 && !strings.HasPrefix(tokens[1], "-"):
			flag = append(flag, tokens[1])
		}
		if !slices.ContainsFunc(out, func(f []string) bool { return slices.Equal(f, flag) }) {
			out = append(out, flag)
		}
	}
	return out
}

func removeFences(raw string) string {
	var keep []string
	in := false
	for _, l := range strings.Split(raw, "\n") {
		if fenceLine.MatchString(l) {
			in = !in
			continue
		}
		if !in {
			keep = append(keep, l)
		}
	}
	return strings.Join(keep, "\n")
}

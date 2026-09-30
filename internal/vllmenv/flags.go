package vllmenv

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"time"
)

// ProbeServeFlags asks the installed vLLM which flags `serve` accepts, by
// running its help.
//
// Autoconfigure takes the whole serve command from a model card, and a card
// written for a newer or different image can name a flag this one does not
// have. The only other way to find out is a start that dies on "unrecognized
// arguments", one flag at a time, minutes into a load.
//
// Newer vLLM prints a grouped summary for --help and the full listing only
// for --help=all; the summary says so, and the listing is asked for then.
//
// It runs the way ProbeDeviceName does -- its own process group, killed as a
// group on timeout -- because a Python import that starts children can hold
// the output pipe open past the deadline.
func (e Env) ProbeServeFlags(timeout time.Duration) ([]string, error) {
	out, err := e.serveHelp(timeout, "--help")
	if err != nil {
		return nil, err
	}
	if strings.Contains(out, "--help=") {
		if out, err = e.serveHelp(timeout, "--help=all"); err != nil {
			return nil, err
		}
	}
	flags := parseServeFlags(out)
	if !looksLikeServeHelp(flags) {
		return nil, fmt.Errorf("the serve help listed %d flags, which is not a help listing: %s",
			len(flags), firstLine(out))
	}
	return flags, nil
}

func (e Env) serveHelp(timeout time.Duration, arg string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	bin, args := e.ServeCommand(arg, nil)
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 5 * time.Second

	// argparse writes help to stdout; some launchers wrap it and write to
	// stderr. Both are read.
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return "", fmt.Errorf("vllm serve %s did not finish within %s", arg, timeout)
	}
	if err != nil {
		return "", fmt.Errorf("vllm serve %s: %v: %s", arg, err, firstLine(string(out)))
	}
	return string(out), nil
}

var flagToken = regexp.MustCompile(`--[a-z][a-z0-9-]*`)

// parseServeFlags reads every flag out of a serve help listing, sorted and
// without repeats.
//
// Reading starts at the "usage:" line when there is one. Images print their
// own start-up lines first -- the radiance image announces its kernels, and a
// JIT build may echo compiler flags -- and those are not serve flags. Negated
// forms need no special handling: argparse prints a boolean pair as
// "--x, --no-x", and both are found.
func parseServeFlags(help string) []string {
	if i := strings.Index(help, "usage:"); i >= 0 {
		help = help[i:]
	}
	seen := map[string]bool{}
	var flags []string
	for _, f := range flagToken.FindAllString(help, -1) {
		f = strings.TrimRight(f, "-")
		if !seen[f] {
			seen[f] = true
			flags = append(flags, f)
		}
	}
	sort.Strings(flags)
	return flags
}

// looksLikeServeHelp refuses anything that is not plainly a serve listing:
// a flag list trusted when it is not one would untick every card flag.
func looksLikeServeHelp(flags []string) bool {
	if len(flags) < 20 {
		return false
	}
	var hasLen, hasTP bool
	for _, f := range flags {
		hasLen = hasLen || f == "--max-model-len"
		hasTP = hasTP || f == "--tensor-parallel-size"
	}
	return hasLen && hasTP
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 200 {
		s = s[:200]
	}
	return s
}

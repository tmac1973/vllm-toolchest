package autoconfig

import (
	"fmt"
	"regexp"
	"strings"
)

// Image is the vLLM image this machine serves with: its variant, the name it
// goes by, and the image it is built from. All empty when not known.
type Image struct {
	ID        string
	Label     string
	BaseImage string
}

// vllmImage is an image reference whose repository is a vLLM build:
// tcclaviger/vllm:latest, vllm/vllm-openai, rocm/vllm-dev.
var vllmImage = regexp.MustCompile(`(?i)\b(?:docker\.io/)?([a-z0-9][a-z0-9_.-]*/[a-z0-9_.-]*vllm[a-z0-9_.-]*)(?::[a-z0-9_.-]+)?`)

// requires is the wording of a card that says an image is needed, not just
// used in an example.
var requires = regexp.MustCompile(`(?i)\b(requires?|required|only|must|needs?)\b`)

// imageRequirement reads a card's statement that the model needs a
// particular vLLM image, and checks it against the image running here.
//
// tcclaviger/gemma-4-31B-it-MXFP416-MTP's card says it "Requires
// tcclaviger/vllm:latest -- the only build with the mxfp4_16 kernels; no other
// vLLM build loads these weights". On that image it serves; on any other it
// fails minutes into a start. Only a sentence that names a vLLM image and
// says it is needed counts: cards show `docker run vllm/vllm-openai` as an
// example often, and that is not a requirement.
func (v *validator) imageRequirement() {
	text := v.in.Card.Plain()
	for _, sentence := range splitSentences(text) {
		if !requires.MatchString(sentence) {
			continue
		}
		m := vllmImage.FindStringSubmatch(sentence)
		if m == nil {
			continue
		}
		wanted := strings.ToLower(m[1])
		quote := clip(readable(sentence), maxQuote)
		running := imageRepo(v.in.Image.BaseImage)
		switch {
		case running == "":
			v.note("", originMachine, fmt.Sprintf("The card says the model needs the %s image: “%s” Which image this machine serves with is not known, so that was not checked.", wanted, quote))
		case running == wanted:
			v.note("", originMachine, fmt.Sprintf("The card says the model needs the %s image, which is the one this machine serves with.", wanted))
		default:
			v.out.ImageWarning = fmt.Sprintf("The card says the model needs the %s image: “%s” This machine serves with %s (%s), so it may not load here.",
				wanted, quote, v.in.Image.Label, v.in.Image.BaseImage)
		}
		return
	}
}

var mdLink = regexp.MustCompile(`\[([^\]]*)\]\([^)]*\)`)

// readable is a line of markdown as a reader sees it: links as their text,
// without emphasis, code ticks or a quote marker.
func readable(s string) string {
	s = mdLink.ReplaceAllString(s, "$1")
	s = strings.NewReplacer("**", "", "`", "", "__", "").Replace(s)
	return strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(s), "> -*"))
}

// imageRepo is an image reference's repository, lower case, without the
// registry or tag: docker.io/tcclaviger/vllm:latest is tcclaviger/vllm.
func imageRepo(ref string) string {
	if m := vllmImage.FindStringSubmatch(ref); m != nil {
		return strings.ToLower(m[1])
	}
	ref = strings.ToLower(strings.TrimPrefix(ref, "docker.io/"))
	if i := strings.LastIndex(ref, ":"); i > strings.LastIndex(ref, "/") {
		ref = ref[:i]
	}
	return ref
}

var sentenceStop = regexp.MustCompile(`[.!?]\s+`)

// splitSentences breaks text at line ends and sentence stops.
func splitSentences(text string) []string {
	var out []string
	for _, line := range strings.Split(text, "\n") {
		for _, s := range sentenceStop.Split(line, -1) {
			if s = strings.TrimSpace(s); s != "" {
				out = append(out, s)
			}
		}
	}
	return out
}

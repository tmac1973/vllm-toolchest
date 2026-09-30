// Package autoconfig proposes a configuration for a model: the model card
// says how its publisher serves it, code decides what fits this machine, and
// a helper model reads what only a reader can. Nothing here changes a
// config; it produces proposals for an operator to review.
package autoconfig

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"html"
	"regexp"
	"strings"
)

// Fetcher reads model cards from the Hub.
type Fetcher interface {
	ModelCard(ctx context.Context, modelID string) (string, error)
	BaseModel(ctx context.Context, modelID string) string
}

// Card is a model's documentation, gathered from its own repository and the
// one it was made from.
type Card struct {
	// Text is trimmed to the sections about running the model, entities
	// unescaped: what the helper model reads.
	Text string
	// Raw is the cards as written, joined, HTML and all: what commands are
	// extracted from. Quotes are checked against Plain(), its readable form.
	Raw string
	// Sources are the repositories read, the model's own first.
	Sources []string
	// Hash identifies Raw, so a later run can tell whether the card changed.
	Hash string
}

// DefaultCardChars is the card budget when the caller has no better one.
const DefaultCardChars = 24000

// FetchCard reads the model's own card and, when it names a different base
// model, that one's too. The model's own card comes first: a quantizer's card
// is the one that says how to serve the quantized files.
//
// A missing card is not an error. The result is simply empty, and what
// follows proceeds on the machine alone.
func FetchCard(ctx context.Context, f Fetcher, modelID string, maxChars int) Card {
	if maxChars <= 0 {
		maxChars = DefaultCardChars
	}
	if !strings.Contains(modelID, "/") {
		return Card{} // not a Hub repository
	}
	repos := []string{modelID}
	if base := f.BaseModel(ctx, modelID); base != "" && !strings.EqualFold(base, modelID) {
		repos = append(repos, base)
	}

	var card Card
	var raws, texts []string
	for _, repo := range repos {
		body, err := f.ModelCard(ctx, repo)
		if err != nil || strings.TrimSpace(body) == "" {
			continue
		}
		card.Sources = append(card.Sources, repo)
		raws = append(raws, body)
		if text := TrimCard(body); text != "" {
			texts = append(texts, "## From the model card of "+repo+"\n\n"+text)
		}
	}
	if len(raws) == 0 {
		return Card{}
	}
	card.Raw = strings.Join(raws, "\n\n")
	card.Text = capText(strings.Join(texts, "\n\n"), maxChars)
	sum := sha256.Sum256([]byte(card.Raw))
	card.Hash = hex.EncodeToString(sum[:])[:16]
	return card
}

// Plain is the card as a reader sees it: tags removed, entities unescaped.
// It is what a quoted sentence is looked for in.
func (c Card) Plain() string {
	return plainText(c.Raw)
}

// cardKeywords select the sections worth reading. A section is kept when its
// heading or its text mentions any of them.
var cardKeywords = []string{
	"recommend", "sampling", "temperature", "top_p", "top-p", "top_k", "top-k", "min_p", "min-p",
	"presence", "repetition", "vllm", "serve", "docker", "context", "max-model-len",
	"thinking", "reasoning", "parser", "tool", "speculative", "mtp", "draft", "dflash", "eagle",
	"kv-cache", "tensor-parallel", "offload", "environment", "best practice", "usage",
	"quickstart", "deploy",
}

var (
	frontMatter = regexp.MustCompile(`(?s)\A---\n.*?\n---\n`)
	htmlComment = regexp.MustCompile(`(?s)<!--.*?-->`)
	blockTag    = regexp.MustCompile(`(?i)</?(pre|p|li|tr|br|div|h[1-6]|ul|ol|table)\b[^>]*>`)
	htmlTag     = regexp.MustCompile(`</?[a-zA-Z][^>]*>`)
	imageLink   = regexp.MustCompile(`!\[[^\]]*\]\([^)]*\)`)
	linkOnly    = regexp.MustCompile(`^\s*(\[[^\]]*\]\([^)]*\)\s*)+$`)
	blankRuns   = regexp.MustCompile(`\n{3,}`)
	headingLine = regexp.MustCompile(`^#{1,6}\s`)
)

// plainText strips a card to what a reader sees. Block tags become line
// breaks first, so an HTML card does not collapse into one line.
func plainText(md string) string {
	md = strings.ReplaceAll(md, "\r\n", "\n")
	md = htmlComment.ReplaceAllString(md, "")
	md = blockTag.ReplaceAllString(md, "\n")
	md = htmlTag.ReplaceAllString(md, "")
	return html.UnescapeString(md)
}

// TrimCard strips what a reader does not need -- front matter, HTML, images,
// badge rows -- and keeps only the sections about running the model, in
// their order.
func TrimCard(md string) string {
	md = strings.ReplaceAll(md, "\r\n", "\n")
	md = frontMatter.ReplaceAllString(md, "")
	md = imageLink.ReplaceAllString(md, "")
	md = plainText(md)

	var lines []string
	for _, l := range strings.Split(md, "\n") {
		if linkOnly.MatchString(l) {
			continue
		}
		lines = append(lines, strings.TrimRight(l, " \t"))
	}

	var sections [][]string
	cur := []string{}
	for _, l := range lines {
		if headingLine.MatchString(l) && len(cur) > 0 {
			sections = append(sections, cur)
			cur = []string{}
		}
		cur = append(cur, l)
	}
	sections = append(sections, cur)

	var kept []string
	for _, sec := range sections {
		text := strings.TrimSpace(strings.Join(sec, "\n"))
		if text != "" && mentionsKeyword(text) {
			kept = append(kept, text)
		}
	}
	return strings.TrimSpace(blankRuns.ReplaceAllString(strings.Join(kept, "\n\n"), "\n\n"))
}

func mentionsKeyword(text string) bool {
	lower := strings.ToLower(text)
	for _, k := range cardKeywords {
		if strings.Contains(lower, k) {
			return true
		}
	}
	return false
}

// capText cuts text at the last paragraph break within max characters.
func capText(text string, max int) string {
	if len(text) <= max {
		return text
	}
	cut := text[:max]
	if i := strings.LastIndex(cut, "\n\n"); i > max/2 {
		cut = cut[:i]
	}
	return cut + "\n\n[The rest of the model card was left out for length.]"
}

// CardCharsForContext is the card budget for a helper model with a context of
// ctx tokens: what is left after the answer and the instructions, at a
// conservative three characters per token.
func CardCharsForContext(ctx int) int {
	const answerTokens, instructionTokens, charsPerToken = 2048, 1000, 3
	n := (ctx - answerTokens - instructionTokens) * charsPerToken
	if n < 4000 {
		return 4000
	}
	return min(n, 48000)
}

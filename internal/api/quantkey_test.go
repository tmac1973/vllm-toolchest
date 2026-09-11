package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
)

// The Download page explains what each format badge means. The badges come
// from huggingface.DetectQuantFormat, so a format added there without a line
// here leaves the operator with a label and no way to find out what it is —
// which is how someone picks a 4-bit weight-only quant expecting it to be
// faster, and then wonders why their card is slow.
func TestEveryQuantBadgeIsExplained(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := httptest.NewRecorder()
	s.handleModelsBrowsePage(w, httptest.NewRequest("GET", "/models/browse", nil))
	if w.Code != 200 {
		t.Fatalf("status = %d", w.Code)
	}
	// Scoped to the help section itself. The labels also appear in the
	// filter dropdown, so searching the whole page would pass whether or not
	// anything explains them.
	page := quantKeySection(t, w.Body.String())

	// Every canonical label the badge can show.
	for _, label := range []string{
		huggingface.FormatFP16,
		huggingface.FormatAWQ,
		huggingface.FormatGPTQ,
		huggingface.FormatFP8,
		huggingface.FormatMXFP4,
		huggingface.FormatNVFP4,
		huggingface.FormatCompressedTensor,
		huggingface.FormatBnB4,
		huggingface.FormatBnB8,
		huggingface.FormatBnB,
		huggingface.FormatQuark,
		huggingface.FormatAutoRound,
		huggingface.FormatModelOpt,
		huggingface.FormatGGUF,
	} {
		if !strings.Contains(page, label) {
			t.Errorf("format %q can appear as a badge but the page never explains it", label)
		}
	}
}

// The distinction the whole section exists to make. Losing it in an edit would
// leave a table of formats that does not say which ones buy speed.
func TestQuantKeySeparatesFitFromSpeed(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	w := httptest.NewRecorder()
	s.handleModelsBrowsePage(w, httptest.NewRequest("GET", "/models/browse", nil))
	page := quantKeySection(t, w.Body.String())

	for _, phrase := range []string{
		"Weight-only", // what AWQ/GPTQ actually do
		"will not be", // "a 4-bit version of it will not be faster"
		"fp8, where the hardware has it",
	} {
		if !strings.Contains(page, phrase) {
			t.Errorf("the fit-versus-speed explanation is missing %q", phrase)
		}
	}
}

// quantKeySection returns just the <details> block explaining the badges.
func quantKeySection(t *testing.T, page string) string {
	t.Helper()
	const open = `<details class="quant-key">`
	i := strings.Index(page, open)
	if i < 0 {
		t.Fatal("the format-badge help section is not on the page")
	}
	j := strings.Index(page[i:], "</details>")
	if j < 0 {
		t.Fatal("the help section is not closed")
	}
	return page[i : i+j]
}

package api

import (
	"bytes"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/recommend"
)

var feedNow = time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

func feedResult(archsKnown bool) recommend.Result {
	return recommend.Result{
		Profile: recommend.ProfileView{GPUCount: 4, PerCardGB: 32, TotalVRAMGB: 128, GPUName: "AMD Radeon AI PRO R9700",
			GPUArch: "gfx1201", Variant: "rdna4-clav", Accelerated: []string{"fp8"}, ArchsKnown: archsKnown, InventoryKnown: true},
		Intent: "fastest", GeneratedAt: feedNow.Add(-3 * time.Minute),
		Verified: []recommend.Candidate{
			{ID: "Qwen/Qwen3.5-35B-A3B-FP8", Format: "FP8", WeightGB: 34.9, TP: 2, Required: 41.2, Available: 57.3,
				AffordableTokens: 262144, FullContextRequests: 6, Accelerated: true, Arch: "Qwen3_5MoeForConditionalGeneration", ArchSupported: archsKnown},
			{ID: "org/Model-AWQ", Format: "AWQ", WeightGB: 18.1, TP: 1, Required: 24, Available: 28.7,
				AffordableTokens: 32768, FullContextRequests: 1, Arch: "LlamaForCausalLM", ArchSupported: archsKnown, Gated: true},
		},
		Unverified: []recommend.Candidate{
			{ID: "org/Odd", Reason: "config unreadable — config.json is missing the fields the fit needs"},
			{ID: "org/New", Reason: "architecture NewForCausalLM is not in this image's model registry"},
		},
	}
}

func renderFeed(t *testing.T, r recommend.Result) string {
	t.Helper()
	s := newGoldenServer(t, goldenEnvGeneric)
	var buf bytes.Buffer
	s.renderPartial(&buf, "recommend_feed", newRecommendFeedView(r, feedNow))
	return buf.String()
}

func TestTheFeedRenders(t *testing.T) {
	out := renderFeed(t, feedResult(true))
	for _, want := range []string{
		"4× AMD Radeon AI PRO R9700 · 128 GB · gfx1201 · rdna4-clav",
		"fits at TP=2 · 41 GB of 57 · accelerated on gfx1201 · Qwen3_5MoeForConditionalGeneration supported by this image",
		"fits at TP=1 · 24 GB of 29 · not accelerated on gfx1201 · LlamaForCausalLM supported by this image",
		"holds 262,144 tokens, 6 full-length requests at once",
		"holds 32,768 tokens, one full-length request at a time",
		"[gated]", "updated 3 minutes ago",
		"2 could not be fully checked", "config unreadable", "NewForCausalLM is not in this image",
		`aria-pressed="true"`, `hx-get="/api/recommend?intent=fastest"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Count(out, `aria-pressed="true"`) != 1 {
		t.Error("more than one chip pressed")
	}

	// With the registry unread there is no architecture clause, and no
	// separator left hanging where it was.
	out = renderFeed(t, feedResult(false))
	if strings.Contains(out, "supported by this image") || strings.Contains(out, "gfx1201 · <") || strings.Contains(out, "gfx1201 ·\n") {
		t.Errorf("an architecture clause without a check:\n%s", out)
	}
	if !strings.Contains(out, "accelerated on gfx1201</small>") {
		t.Error("the acceleration clause did not end the sentence")
	}
}

func TestTheFeedsEmptyStates(t *testing.T) {
	r := feedResult(true)
	r.Verified, r.Unverified = nil, nil
	r.Unavailable = "the GPUs have not been read yet"
	out := renderFeed(t, r)
	if !strings.Contains(out, "No recommendations right now: the GPUs have not been read yet") || strings.Contains(out, `class="recommend-card"`) || strings.Contains(out, `class="recommend-chip"`) {
		t.Errorf("unavailable:\n%s", out)
	}

	r.Unavailable = ""
	out = renderFeed(t, r)
	if !strings.Contains(out, "Nothing among the models looked at fits this machine") || strings.Contains(out, `class="recommend-card"`) {
		t.Errorf("empty:\n%s", out)
	}

	r.Stale = true
	if out := renderFeed(t, r); !strings.Contains(out, "more than six hours old") {
		t.Error("a stale list did not say so")
	}
}

// The page keeps its search box whatever the feed says: the feed is a
// section above it, loaded on its own.
func TestTheBrowsePageKeepsItsSearch(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.router = s.buildRouter()
	rec := httptest.NewRecorder()
	s.router.ServeHTTP(rec, httptest.NewRequest("GET", "/models/browse", nil))
	body := rec.Body.String()
	if !strings.Contains(body, `id="recommend-feed"`) || !strings.Contains(body, `id="hf-search-input"`) {
		t.Errorf("browse page: feed %v, search %v", strings.Contains(body, "recommend-feed"), strings.Contains(body, "hf-search-input"))
	}
	if strings.Index(body, `id="recommend-feed"`) > strings.Index(body, `id="hf-search-input"`) {
		t.Error("the feed is not above the search")
	}
}

// Past the first eight the cards fold away, so the search box below stays
// within reach: compute's pool verified 35.
func TestTheFeedFoldsALongList(t *testing.T) {
	r := feedResult(true)
	for i := 0; i < 8; i++ {
		c := r.Verified[0]
		c.ID = fmt.Sprintf("org/extra-%d", i)
		r.Verified = append(r.Verified, c)
	}
	v := newRecommendFeedView(r, feedNow)
	if len(v.Cards) != 8 || len(v.More) != 2 {
		t.Fatalf("shown %d, folded %d", len(v.Cards), len(v.More))
	}
	if out := renderFeed(t, r); !strings.Contains(out, "Show 2 more") || strings.Count(out, `class="recommend-card"`) != 10 {
		t.Error("the fold did not render")
	}
}

// The image's own publisher's models are a section above the orders; their
// own formats are made for this image's kernels, not "not accelerated".
func TestTheFeaturedSection(t *testing.T) {
	r := feedResult(true)
	r.FeaturedBy = []string{"tcclaviger"}
	for i := 0; i < 5; i++ {
		r.Featured = append(r.Featured, recommend.Candidate{ID: fmt.Sprintf("tcclaviger/M%d-MXFP416", i), Format: "Other", WeightGB: 16, TP: 2,
			Required: 30, Available: 57, AffordableTokens: 262144, FullContextRequests: 3, Featured: true, Arch: "Qwen3_5ForConditionalGeneration"})
	}
	out := renderFeed(t, r)
	if !strings.Contains(out, "Made for this image") || !strings.Contains(out, "by tcclaviger") ||
		!strings.Contains(out, "made for this image&#39;s kernels") || strings.Contains(out, "M0-MXFP416</strong>\n      <a href=\"https://huggingface.co/tcclaviger/M0-MXFP416\" target=\"_blank\" rel=\"noopener\" class=\"recommend-hub\" title=\"View on HuggingFace\">&#8599;</a>\n      <br><small>fits at TP=2 · 30 GB of 57 · not accelerated") {
		t.Errorf("featured section:\n%s", out)
	}
	if strings.Index(out, "Made for this image") > strings.Index(out, `class="recommend-chip"`) {
		t.Error("the featured section is not above the orders")
	}
	if !strings.Contains(out, "Show 1 more") {
		t.Error("the fifth featured model did not fold")
	}
	if out := renderFeed(t, feedResult(true)); strings.Contains(out, "Made for this image") {
		t.Error("a section with nothing in it")
	}
}

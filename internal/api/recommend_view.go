package api

import (
	"fmt"
	"strings"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/recommend"
)

// recommendFeedView is the feed as the page shows it: every sentence made
// here, so the template only lays it out.
type recommendFeedView struct {
	Profile string // "4× AMD Radeon AI PRO R9700 · 128 GB · gfx1201 · rdna4-clav"
	Intent  string
	Chips   []recommendChip
	// Featured are the image's own publishers' models that fit, in a section
	// of their own above the orders; FeaturedBy names the publishers.
	Featured, FeaturedMore []recommendCard
	FeaturedBy             string
	Cards                  []recommendCard // the first few, shown
	More                   []recommendCard // the rest, folded
	Unverified             []recommendUnverified
	Unavailable            string
	Empty                  bool // built, and nothing in it fits
	Age                    string
	Stale                  bool
}

type recommendChip struct {
	Intent, Label string
	Pressed       bool
}

type recommendCard struct {
	ID, SafeID, Format, Color, Weights string
	// Fit is the sentence that says why it is here: the width, the memory,
	// whether the format is accelerated, and the architecture when it was
	// checked. Context is what it holds here and how many at once.
	Fit, Context string
	Gated        bool
	Offload      bool
}

type recommendUnverified struct {
	ID, Reason string
	Gated      bool
}

var recommendChips = []recommendChip{
	{Intent: recommend.IntentQuality, Label: "Best quality"},
	{Intent: recommend.IntentFastest, Label: "Fastest"},
	{Intent: recommend.IntentContext, Label: "Longest context"},
	{Intent: recommend.IntentNewest, Label: "Newest"},
}

func newRecommendFeedView(r recommend.Result, now time.Time) recommendFeedView {
	p := r.Profile
	v := recommendFeedView{Intent: r.Intent, Unavailable: r.Unavailable, Stale: r.Stale}
	parts := []string{fmt.Sprintf("%d×", p.GPUCount)}
	if p.GPUName != "" {
		parts[0] += " " + p.GPUName
	}
	parts = append(parts, fmt.Sprintf("%.0f GB", p.TotalVRAMGB))
	for _, s := range []string{p.GPUArch, p.Variant} {
		if s != "" {
			parts = append(parts, s)
		}
	}
	v.Profile = strings.Join(parts, " · ")

	for _, c := range recommendChips {
		c.Pressed = c.Intent == r.Intent
		v.Chips = append(v.Chips, c)
	}
	if !r.GeneratedAt.IsZero() {
		v.Age = "updated " + ago(now.Sub(r.GeneratedAt))
	}

	for _, c := range r.Verified {
		v.Cards = append(v.Cards, newRecommendCard(c, p))
	}
	for _, c := range r.Featured {
		v.Featured = append(v.Featured, newRecommendCard(c, p))
	}
	if len(v.Featured) > featuredShown {
		v.Featured, v.FeaturedMore = v.Featured[:featuredShown], v.Featured[featuredShown:]
	}
	v.FeaturedBy = strings.Join(r.FeaturedBy, ", ")
	for _, c := range r.Unverified {
		v.Unverified = append(v.Unverified, recommendUnverified{ID: c.ID, Reason: c.Reason, Gated: c.Gated})
	}
	if len(v.Cards) > recommendShown {
		v.Cards, v.More = v.Cards[:recommendShown], v.Cards[recommendShown:]
	}
	v.Empty = r.Unavailable == "" && len(r.Verified) == 0
	return v
}

// recommendShown is how many cards show before the rest fold away. The pool
// verified 35 on compute, and all of them pushed the search box a page and a
// half down: the feed is above search, not instead of it.
const recommendShown = 8

// featuredShown is how many of the image's own publishers' cards show before
// the rest fold.
const featuredShown = 4

// newRecommendCard is one model's card.
func newRecommendCard(c recommend.Candidate, p recommend.ProfileView) recommendCard {
	clauses := []string{
		fmt.Sprintf("fits at TP=%d", c.TP),
		fmt.Sprintf("%.0f GB of %.0f", c.Required, c.Available),
	}
	switch {
	case c.Accelerated:
		clauses = append(clauses, "accelerated on "+p.GPUArch)
	case c.Featured:
		// The image's own formats run on its own kernels: "not accelerated"
		// would be the wrong thing to say of them.
		clauses = append(clauses, "made for this image's kernels")
	default:
		clauses = append(clauses, "not accelerated on "+p.GPUArch)
	}
	if p.ArchsKnown && c.Arch != "" {
		clauses = append(clauses, c.Arch+" supported by this image")
	}
	ctx := groupThousands(c.AffordableTokens) + " tokens"
	switch {
	case c.Offload:
		ctx += " · experts in system RAM, one request at a time — slower generation"
	case c.FullContextRequests == 1:
		ctx += ", one full-length request at a time"
	case c.FullContextRequests > 1:
		ctx += fmt.Sprintf(", %d full-length requests at once", c.FullContextRequests)
	}
	return recommendCard{
		ID: c.ID, SafeID: safeID(c.ID), Format: c.Format, Color: quantBadgeColor(c.Format),
		Weights: fmt.Sprintf("%.1f GB", c.WeightGB),
		Fit:     strings.Join(clauses, " · "), Context: "holds " + ctx,
		Gated: c.Gated, Offload: c.Offload,
	}
}

// ago is a duration as a reader says it.
func ago(d time.Duration) string {
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		if m := int(d.Minutes()); m != 1 {
			return fmt.Sprintf("%d minutes ago", m)
		}
		return "1 minute ago"
	case d < 48*time.Hour:
		if h := int(d.Hours()); h != 1 {
			return fmt.Sprintf("%d hours ago", h)
		}
		return "1 hour ago"
	}
	return fmt.Sprintf("%d days ago", int(d.Hours()/24))
}

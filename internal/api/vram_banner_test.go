package api

import (
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// A measured figure and a projected one must not look alike. The projected
// arithmetic was wrong about a real model by a third, so a reader who cannot
// tell which they are looking at is worse off than one shown nothing.
func TestTheBannerSaysWhereItsNumbersCameFrom(t *testing.T) {
	measured := models.VRAMEstimate{
		Source:          models.SourceMeasured,
		MeasuredAt:      time.Date(2026, 9, 17, 19, 15, 0, 0, time.UTC),
		MeasuredTP:      4,
		TotalRequiredGB: 113.9,
		WeightsTotalGB:  98.3,
		KVAtContextGB:   7.8,
		ContextTokens:   262144,
	}
	b := newVRAMBanner(measured, models.VRAMFit{})
	if !b.Measured {
		t.Error("a measured estimate does not report itself as measured")
	}
	if b.MeasuredWhen == "" {
		t.Error("a measured estimate carries no date, so its age cannot be judged")
	}
	if b.MeasuredTP != 4 {
		t.Errorf("measured width = %d, want 4", b.MeasuredTP)
	}

	projected := models.VRAMEstimate{
		Source:          models.SourceProjected,
		TotalRequiredGB: 83.3,
		WeightsTotalGB:  76.3,
	}
	p := newVRAMBanner(projected, models.VRAMFit{})
	if p.Measured {
		t.Error("a projection claims to be measured")
	}
	if p.MeasuredWhen != "" {
		t.Error("a projection carries a measurement date")
	}
}

// Exactly one row comes from a real start; the rest project it onto widths
// nothing has run at.
func TestOnlyTheMeasuredRowIsMarked(t *testing.T) {
	est := models.VRAMEstimate{Source: models.SourceMeasured, MeasuredTP: 4, TotalRequiredGB: 113.9}
	fit := models.VRAMFit{
		Known: true,
		Options: []models.TPOption{
			{TP: 1, RequiredGB: 120},
			{TP: 2, RequiredGB: 117},
			{TP: 4, RequiredGB: 113.9, Measured: true},
		},
	}

	rows := newVRAMTPRows(est, fit)
	if len(rows) != 3 {
		t.Fatalf("got %d rows, want 3", len(rows))
	}

	marked := 0
	for _, r := range rows {
		if r.Measured {
			marked++
			if r.TP != 4 {
				t.Errorf("TP=%d is marked measured; the run was at TP=4", r.TP)
			}
		}
	}
	if marked != 1 {
		t.Errorf("%d rows marked measured, want exactly 1", marked)
	}
}

package api

import (
	"strings"
	"testing"
)

func TestR4DBatchedTokenCeiling(t *testing.T) {
	for _, tc := range []struct {
		name       string
		hiddenSize int
		want       int
	}{
		// Qwen3.8-27B. 48 MiB / (5120 * 2) = 4915.2, so 4096 is safe and
		// vLLM's own default of 8192 would not be.
		{"qwen3.8-27b", 5120, 4915},
		{"narrower model allows more tokens", 2560, 9830},
		{"wider model allows fewer", 8192, 3072},
		{"unknown hidden size", 0, 0},
		{"negative is not a crash", -1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := r4dBatchedTokenCeiling(tc.hiddenSize); got != tc.want {
				t.Errorf("ceiling(%d) = %d, want %d", tc.hiddenSize, got, tc.want)
			}
		})
	}
}

func TestBatchedTokenAdvice(t *testing.T) {
	for _, tc := range []struct {
		name       string
		radiance   bool
		hidden     int
		tp         int
		configured int
		wantWarn   bool
		wantSubstr string
		wantEmpty  bool
	}{
		{
			name: "generic image says nothing",
			// The cap is a property of radiance's kernel, not of vLLM.
			radiance: false, hidden: 5120, tp: 2, configured: 8192,
			wantEmpty: true,
		},
		{
			name:     "at TP=2 a safe value is advised, not warned",
			radiance: true, hidden: 5120, tp: 2, configured: 4096,
			wantWarn: false, wantSubstr: "4915",
		},
		{
			name: "at TP=2 an over-ceiling value warns",
			// This is the silent case: it runs, just 2.3x slower on RCCL.
			radiance: true, hidden: 5120, tp: 2, configured: 8192,
			wantWarn: true, wantSubstr: "silently falls back",
		},
		{
			name:     "exactly at the ceiling is fine",
			radiance: true, hidden: 5120, tp: 2, configured: 4915,
			wantWarn: false,
		},
		{
			name: "at TP=4 there is no ceiling to hit",
			// libr4d ships 2-rank kernels only, so RCCL is used regardless.
			radiance: true, hidden: 5120, tp: 4, configured: 16384,
			wantWarn: false, wantSubstr: "no ceiling applies",
		},
		{
			name:     "TP=1 has no all-reduce at all",
			radiance: true, hidden: 5120, tp: 1, configured: 99999,
			wantWarn: false, wantSubstr: "No tensor-parallel all-reduce",
		},
		{
			name:     "unknown hidden size cannot advise",
			radiance: true, hidden: 0, tp: 2, configured: 4096,
			wantEmpty: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			advice, warn := batchedTokenAdvice(tc.radiance, tc.hidden, tc.tp, tc.configured)

			if tc.wantEmpty {
				if advice != "" {
					t.Errorf("expected no advice, got %q", advice)
				}
				return
			}
			if advice == "" {
				t.Fatal("expected advice, got none")
			}
			if warn != tc.wantWarn {
				t.Errorf("warn = %v, want %v (advice: %s)", warn, tc.wantWarn, advice)
			}
			if tc.wantSubstr != "" && !strings.Contains(advice, tc.wantSubstr) {
				t.Errorf("advice missing %q:\n%s", tc.wantSubstr, advice)
			}
		})
	}
}

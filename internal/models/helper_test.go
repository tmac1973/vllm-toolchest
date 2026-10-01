package models

import (
	"strings"
	"testing"
)

func TestHelperConfigIsFixed(t *testing.T) {
	c := HelperConfig(0)
	if !c.EnforceEager || c.MaxNumSeqs != 1 || c.MaxModelLen != HelperContext || c.TensorParallelSize != 1 {
		t.Errorf("helper config: %+v", c)
	}
	if c.GPUMemoryUtilization != 0.90 {
		t.Errorf("an unset utilization gave %.2f, want 0.90", c.GPUMemoryUtilization)
	}
	if c.ExtraFlags != "" || c.Env != "" || c.EnableAutoToolChoice || c.SpeculativeConfig != "" {
		t.Error("the helper config carries settings it must not")
	}
	if HelperConfig(0.8).GPUMemoryUtilization != 0.8 || HelperUtil(1.5) != 0.90 {
		t.Error("utilization not taken, or not bounded")
	}
}

func TestHelperFits(t *testing.T) {
	if ok, why := HelperFits(GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}, 0.9); !ok {
		t.Errorf("a 32 GB card: %s", why)
	}
	ok, why := HelperFits(GPUInventory{Count: 1, PerCardGB: 8, Known: true}, 0.9)
	if ok {
		t.Error("an 8 GB card was said to hold an 8 GB model and its cache")
	}
	if !strings.Contains(why, "smallest card offers 7.2") {
		t.Errorf("why = %q", why)
	}
	if ok, _ := HelperFits(GPUInventory{}, 0.9); !ok {
		t.Error("an unread inventory refused the helper")
	}
	// Unset utilization is not "fits in nothing".
	if ok, _ := HelperFits(GPUInventory{Count: 1, PerCardGB: 24, Known: true}, 0); !ok {
		t.Error("a 24 GB card refused the helper at the default utilization")
	}
}

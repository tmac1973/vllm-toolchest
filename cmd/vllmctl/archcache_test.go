package main

import (
	"os"
	"slices"
	"testing"
)

func TestArchCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	archs := []string{"Gemma4ForCausalLM", "Qwen4ExpForConditionalGeneration"}
	if err := writeCachedArchs(dir, "rdna4-clav", "64c80d8c6659833a", archs); err != nil {
		t.Fatal(err)
	}
	if got := readCachedArchs(dir, "rdna4-clav", "64c80d8c6659833a"); !slices.Equal(got, archs) {
		t.Errorf("got %v", got)
	}
}

// A list written for another variant, or before the registry changed, is a
// miss: that is when the supported architectures can differ.
func TestArchCacheIsInvalidatedByTheImage(t *testing.T) {
	dir := t.TempDir()
	writeCachedArchs(dir, "rdna4-clav", "64c80d8c6659833a", []string{"A"})
	for _, c := range []struct{ variant, fp string }{
		{"radiance", "64c80d8c6659833a"},
		{"rdna4-clav", "0000000000000000"},
		{"rdna4-clav", ""}, // a registry that could not be read is never a hit
	} {
		if got := readCachedArchs(dir, c.variant, c.fp); got != nil {
			t.Errorf("%s %q: got %v", c.variant, c.fp, got)
		}
	}
}

func TestArchCacheMissingOrEmpty(t *testing.T) {
	dir := t.TempDir()
	if got := readCachedArchs(dir, "rdna4-clav", "x"); got != nil {
		t.Errorf("missing file: %v", got)
	}
	os.WriteFile(archCache(dir), []byte("rdna4-clav x\n"), 0o644)
	if got := readCachedArchs(dir, "rdna4-clav", "x"); got != nil {
		t.Errorf("a key with no names: %v", got)
	}
}

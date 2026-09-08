package main

import (
	"os"
	"path/filepath"
	"testing"
)

func cacheDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestDeviceNameCacheRoundTrip(t *testing.T) {
	dir := cacheDir(t)
	if err := writeCachedDeviceName(dir, "gfx1100", "AMD_Radeon_RX_7900_XTX"); err != nil {
		t.Fatal(err)
	}
	if got := readCachedDeviceName(dir, "gfx1100"); got != "AMD_Radeon_RX_7900_XTX" {
		t.Errorf("got %q", got)
	}
}

// The data directory outlives the image. A name resolved under one build is
// not evidence about another, and a stale one is silent: tuned kernel configs
// are keyed by it, so the wrong name writes JSON nothing reads.
func TestDeviceNameCacheIsInvalidatedByAnArchChange(t *testing.T) {
	dir := cacheDir(t)
	writeCachedDeviceName(dir, "gfx1201", "AMD-gfx1201")

	if got := readCachedDeviceName(dir, "gfx1100"); got != "" {
		t.Errorf("a cache from another architecture was accepted: %q", got)
	}
	if got := readCachedDeviceName(dir, "gfx1201"); got != "AMD-gfx1201" {
		t.Errorf("the cache should still be good for its own architecture; got %q", got)
	}
}

// Caches written before the architecture was recorded are bare names. They
// predate the patch-gating fix and are exactly the ones that must not be
// trusted, so one re-probe is the correct cost.
func TestBareDeviceNameCacheIsTreatedAsStale(t *testing.T) {
	dir := cacheDir(t)
	if err := os.WriteFile(filepath.Join(dir, "config", "device-name"),
		[]byte("AMD-gfx1201\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readCachedDeviceName(dir, "gfx1201"); got != "" {
		t.Errorf("a pre-versioning cache was accepted: %q", got)
	}
}

func TestMissingOrEmptyDeviceNameCache(t *testing.T) {
	dir := cacheDir(t)
	if got := readCachedDeviceName(dir, "gfx1100"); got != "" {
		t.Errorf("no cache file should read as empty; got %q", got)
	}
	os.WriteFile(filepath.Join(dir, "config", "device-name"), []byte("gfx1100 \n"), 0o644)
	if got := readCachedDeviceName(dir, "gfx1100"); got != "" {
		t.Errorf("an arch with no name is not a usable cache; got %q", got)
	}
}

func TestArchFromDeviceName(t *testing.T) {
	cases := []struct {
		name     string
		wantArch string
		wantOK   bool
	}{
		// The patched form encodes the target and can be checked.
		{"AMD-gfx1201", "gfx1201", true},
		{"AMD-gfx1100", "gfx1100", true},
		{"AMD-gfx90a", "gfx90a", true},

		// Marketing names say nothing about the architecture. Guessing from
		// them would discard a perfectly good name — the radiance image
		// reports one of these.
		{"AMD_Radeon_R9700", "", false},
		{"AMD_Radeon_RX_7900_XTX", "", false},
		{"AMD Instinct MI300X", "", false},
		{"", "", false},

		// Malformed near-misses must not parse into something checkable.
		{"AMD-gfx", "", false},
		{"AMD-gfx12 01", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			arch, ok := archFromDeviceName(tc.name)
			if ok != tc.wantOK || arch != tc.wantArch {
				t.Errorf("archFromDeviceName(%q) = (%q, %v), want (%q, %v)",
					tc.name, arch, ok, tc.wantArch, tc.wantOK)
			}
		})
	}
}

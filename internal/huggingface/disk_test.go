package huggingface

import (
	"path/filepath"
	"testing"
)

func TestFreeBytesAtReportsSomething(t *testing.T) {
	got := freeBytesAt(t.TempDir())
	if got <= 0 {
		t.Fatalf("free bytes on a temp dir = %d, want a positive figure", got)
	}
}

// The models directory does not exist until the first download, and a budget
// that reads -1 there would gate every button on a fresh install.
func TestFreeBytesAtWalksUpToAnExistingParent(t *testing.T) {
	deep := filepath.Join(t.TempDir(), "models", "owner", "name")
	got := freeBytesAt(deep)
	if got <= 0 {
		t.Errorf("free bytes below a directory that does not exist yet = %d, want a positive figure", got)
	}
}

// -1 is "unknown", and has to stay distinguishable from "full": a failed
// statfs must not read as a full disk and refuse everything.
func TestFreeBytesAtUnknownOnGarbage(t *testing.T) {
	if got := freeBytesAt(""); got != -1 {
		t.Errorf("freeBytesAt(\"\") = %d, want -1", got)
	}
}

func TestAvailableForDownloadReservesTheSafetyMargin(t *testing.T) {
	d := NewDownloader(t.TempDir(), "")

	free := d.FreeBytes()
	if free < 0 {
		t.Skip("no free-space figure on this filesystem")
	}
	avail := d.AvailableForDownload()

	if avail > free-DiskSafetyMarginBytes {
		t.Errorf("available %d exceeds free %d minus the %d margin",
			avail, free, DiskSafetyMarginBytes)
	}
	if avail < 0 {
		t.Errorf("available = %d; a known free figure must clamp to 0, not go negative", avail)
	}
}

// Bytes an in-flight download has yet to write are not bytes a new one can
// have, or two downloads started together both "fit" and neither does.
func TestPendingBytesCountsWhatIsStillOwed(t *testing.T) {
	d := NewDownloader(t.TempDir(), "")
	if got := d.PendingBytes(); got != 0 {
		t.Fatalf("PendingBytes with nothing running = %d, want 0", got)
	}

	d.active["x--y"] = &download{
		id:         "x--y",
		modelID:    "x/y",
		status:     "downloading",
		totalBytes: 1000,
		files: []fileState{
			{filename: "a", size: 600, downloaded: 250},
			{filename: "b", size: 400, downloaded: 0},
		},
	}
	if got, want := d.PendingBytes(), int64(750); got != want {
		t.Errorf("PendingBytes = %d, want %d", got, want)
	}

	// A settled download has stopped reserving anything.
	d.active["x--y"].status = "cancelled"
	if got := d.PendingBytes(); got != 0 {
		t.Errorf("PendingBytes after the download stopped = %d, want 0", got)
	}
}

// A download that overshot its declared total must not push the budget up.
func TestPendingBytesIgnoresOvershoot(t *testing.T) {
	d := NewDownloader(t.TempDir(), "")
	d.active["x--y"] = &download{
		status:     "downloading",
		totalBytes: 1000,
		files:      []fileState{{size: 1000, downloaded: 1200}},
	}
	if got := d.PendingBytes(); got != 0 {
		t.Errorf("PendingBytes = %d, want 0 (never negative)", got)
	}
}

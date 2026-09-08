package huggingface

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Cancelling a download used to delete its directory, so a paused transfer
// left nothing to find. Now it leaves .part files, and this is what makes them
// visible — without it they would be disk usage with no way to reclaim it.
func TestListIncomplete(t *testing.T) {
	dir := t.TempDir()
	d := NewDownloader(dir, "")
	models := filepath.Join(dir, "models")

	// Two partial files for one model.
	writeFile(t, filepath.Join(models, "unsloth", "Qwen3.8-27B-FP8", "model-00001.safetensors.part"), 1000)
	writeFile(t, filepath.Join(models, "unsloth", "Qwen3.8-27B-FP8", "model-00002.safetensors.part"), 2000)
	// A finished file alongside them must not change the count.
	writeFile(t, filepath.Join(models, "unsloth", "Qwen3.8-27B-FP8", "config.json"), 50)
	// One partial for another model, nested a directory deeper.
	writeFile(t, filepath.Join(models, "TheBloke", "Mixtral-8x7B-AWQ", "shards", "w.safetensors.part"), 500)
	// A fully-downloaded model has nothing partial and is not our business.
	writeFile(t, filepath.Join(models, "meta-llama", "Llama-4-70B", "model.safetensors"), 900)

	got := d.ListIncomplete()
	if len(got) != 2 {
		t.Fatalf("got %d incomplete, want 2: %+v", len(got), got)
	}

	// Sorted by model id, so the order is fixed.
	if got[0].ModelID != "TheBloke/Mixtral-8x7B-AWQ" {
		t.Errorf("got[0].ModelID = %q", got[0].ModelID)
	}
	if got[0].PartFiles != 1 || got[0].OnDisk != 500 {
		t.Errorf("got[0] = %d files / %d bytes, want 1 / 500", got[0].PartFiles, got[0].OnDisk)
	}
	if got[1].ModelID != "unsloth/Qwen3.8-27B-FP8" {
		t.Errorf("got[1].ModelID = %q", got[1].ModelID)
	}
	if got[1].PartFiles != 2 || got[1].OnDisk != 3000 {
		t.Errorf("got[1] = %d files / %d bytes, want 2 / 3000", got[1].PartFiles, got[1].OnDisk)
	}
}

func TestListIncompleteEmpty(t *testing.T) {
	d := NewDownloader(t.TempDir(), "")
	if got := d.ListIncomplete(); len(got) != 0 {
		t.Errorf("got %+v, want none", got)
	}
}

func TestDiscardRemovesPartialsAndEmptyOwner(t *testing.T) {
	dir := t.TempDir()
	d := NewDownloader(dir, "")
	models := filepath.Join(dir, "models")
	writeFile(t, filepath.Join(models, "unsloth", "Qwen3.8-27B-FP8", "model.safetensors.part"), 1000)

	if err := d.Discard("unsloth/Qwen3.8-27B-FP8"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "unsloth", "Qwen3.8-27B-FP8")); !os.IsNotExist(err) {
		t.Error("model directory survived the discard")
	}
	// Its owner held nothing else, so it should not be left behind either.
	if _, err := os.Stat(filepath.Join(models, "unsloth")); !os.IsNotExist(err) {
		t.Error("empty owner directory was left behind")
	}
}

// An owner with another model still under it must survive.
func TestDiscardKeepsOwnerWithOtherModels(t *testing.T) {
	dir := t.TempDir()
	d := NewDownloader(dir, "")
	models := filepath.Join(dir, "models")
	writeFile(t, filepath.Join(models, "unsloth", "A", "model.safetensors.part"), 10)
	writeFile(t, filepath.Join(models, "unsloth", "B", "model.safetensors"), 10)

	if err := d.Discard("unsloth/A"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "unsloth", "B", "model.safetensors")); err != nil {
		t.Errorf("the other model under this owner was removed: %v", err)
	}
}

// The guard against a path that would take out the whole data directory.
func TestDiscardRefusesEmptyModelID(t *testing.T) {
	d := NewDownloader(t.TempDir(), "")
	if err := d.Discard(""); err == nil {
		t.Error("Discard(\"\") should refuse")
	}
}

// modelDir falls back to joining whatever it is handed, so an id that is not
// owner/name resolves to the models root — and discarding that would delete
// every model on the box.
func TestDiscardRefusesNonModelIDs(t *testing.T) {
	dir := t.TempDir()
	d := NewDownloader(dir, "")
	models := filepath.Join(dir, "models")
	writeFile(t, filepath.Join(models, "unsloth", "A", "model.safetensors"), 10)

	for _, id := range []string{"", "/", "noslash", "..", "../..", "owner/", "/name", "a/b/c"} {
		if err := d.Discard(id); err == nil {
			t.Errorf("Discard(%q) was allowed", id)
		}
	}
	if _, err := os.Stat(filepath.Join(models, "unsloth", "A", "model.safetensors")); err != nil {
		t.Fatalf("an unrelated model was removed: %v", err)
	}
}

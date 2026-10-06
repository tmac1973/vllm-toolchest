package models

import (
	"os"
	"path/filepath"
	"testing"
)

// writeConfig writes body as a model directory's config.json and returns the
// directory.
func writeConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

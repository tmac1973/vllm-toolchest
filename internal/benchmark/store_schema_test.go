package benchmark

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeBenchmarksFile(t *testing.T, dir, body string) string {
	t.Helper()
	path := filepath.Join(dir, "config", benchmarksFileName)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

const v1Benchmarks = `{"version": 1, "jobs": [], "runs": [{"id": "r1", "status": "completed"}]}`

// Unlike the model registry, history cannot be rebuilt by rescanning, so a
// file this build cannot account for must never be written over. Checked on
// the bytes: an error returned after the file was rewritten would pass a
// weaker test.
func TestStoreNeverOverwritesAFileItCouldNotRead(t *testing.T) {
	for _, tc := range []struct{ name, body, reason string }{
		{"newer schema", strings.Replace(v1Benchmarks, `"version": 1`, `"version": 99`, 1), "schema version 99"},
		{"truncated", v1Benchmarks[:len(v1Benchmarks)/2], "could not be parsed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := writeBenchmarksFile(t, dir, tc.body)
			s := NewStore(dir)

			if reason := s.ReadOnly(); !strings.Contains(reason, tc.reason) {
				t.Fatalf("ReadOnly() = %q, want it to mention %q", reason, tc.reason)
			}
			if err := s.Save(BenchmarkRun{ID: "new"}); err == nil || !strings.Contains(err.Error(), path) {
				t.Errorf("Save: err = %v, want a refusal naming %s", err, path)
			}
			if got, _ := os.ReadFile(path); !bytes.Equal(got, []byte(tc.body)) {
				t.Errorf("the file was rewritten:\n got %s\nwant %s", got, tc.body)
			}
		})
	}
}

// A version-1 file is the one every existing install has. It must load, and
// be rewritten as the current version with its history intact.
func TestStoreUpgradesAVersionOneFile(t *testing.T) {
	dir := t.TempDir()
	path := writeBenchmarksFile(t, dir, v1Benchmarks)
	s := NewStore(dir)

	if reason := s.ReadOnly(); reason != "" {
		t.Fatalf("a version-1 file should be writable, got read-only: %s", reason)
	}
	if err := s.Save(BenchmarkRun{ID: "r2", Config: ConfigSnapshot{ProfileName: "mtp"}}); err != nil {
		t.Fatal(err)
	}

	var bf benchmarkFile
	data, _ := os.ReadFile(path)
	if err := json.Unmarshal(data, &bf); err != nil {
		t.Fatal(err)
	}
	if bf.Version != schemaVersion {
		t.Errorf("version = %d, want %d", bf.Version, schemaVersion)
	}
	if len(bf.Runs) != 2 {
		t.Errorf("history lost on rewrite: %d runs, want 2", len(bf.Runs))
	}
}

func TestStoreNewerSchemaStillListsRuns(t *testing.T) {
	dir := t.TempDir()
	writeBenchmarksFile(t, dir, strings.Replace(v1Benchmarks, `"version": 1`, `"version": 99`, 1))

	if runs := NewStore(dir).List(); len(runs) != 1 {
		t.Errorf("runs from a newer file should still be listed, got %d", len(runs))
	}
}

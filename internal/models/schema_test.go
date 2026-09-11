package models

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeRegistryFile puts a hand-written models.json where NewRegistry will
// look for it, and returns its path.
func writeRegistryFile(t *testing.T, dir, body string) string {
	t.Helper()
	path := filepath.Join(dir, "config", "models.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

const v2File = `{
  "models": {"org/model": {"id": "org/model", "vllm_config": {"max_model_len": 4096}}},
  "schema_version": 2,
  "pending_configs": [{"model_id": "org/waiting", "config": {"max_model_len": 8192}}]
}`

// The file this build already writes must load and rewrite without loss.
func TestCurrentSchemaLoadsAndRewrites(t *testing.T) {
	dir := t.TempDir()
	path := writeRegistryFile(t, dir, v2File)

	reg := NewRegistry(dir, filepath.Join(dir, "models"))
	if reason := reg.ReadOnly(); reason != "" {
		t.Fatalf("a current file should be writable, got read-only: %s", reason)
	}
	if err := reg.UpdateConfig("org/model", VLLMConfig{MaxModelLen: 2048}); err != nil {
		t.Fatal(err)
	}

	var rf registryFile
	data, _ := os.ReadFile(path)
	if err := json.Unmarshal(data, &rf); err != nil {
		t.Fatal(err)
	}
	if rf.SchemaVersion != schemaVersion {
		t.Errorf("schema_version = %d, want %d", rf.SchemaVersion, schemaVersion)
	}
	if len(rf.PendingConfigs) != 1 || rf.PendingConfigs[0].ModelID != "org/waiting" {
		t.Errorf("pending configs lost on rewrite: %+v", rf.PendingConfigs)
	}
}

// A file with no version was written before the field existed, not by a newer
// build, and must stay writable.
func TestMissingSchemaVersionIsCurrent(t *testing.T) {
	dir := t.TempDir()
	writeRegistryFile(t, dir, `{"models": {"org/model": {"id": "org/model"}}}`)

	reg := NewRegistry(dir, filepath.Join(dir, "models"))
	if reason := reg.ReadOnly(); reason != "" {
		t.Errorf("an unversioned file should be writable, got read-only: %s", reason)
	}
}

// The guarantee this whole gate exists for: a file this build cannot fully
// account for is never written over. Checked on the bytes, not on the error —
// an error returned after the file was rewritten would pass a weaker test.
func TestUnreadableRegistryIsNeverOverwritten(t *testing.T) {
	for _, tc := range []struct {
		name, body, reason string
	}{
		{"newer schema", strings.Replace(v2File, `"schema_version": 2`, `"schema_version": 99`, 1), "schema version 99"},
		{"truncated", v2File[:len(v2File)/2], "could not be parsed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := writeRegistryFile(t, dir, tc.body)
			reg := NewRegistry(dir, filepath.Join(dir, "models"))

			if reason := reg.ReadOnly(); !strings.Contains(reason, tc.reason) {
				t.Fatalf("ReadOnly() = %q, want it to mention %q", reason, tc.reason)
			}

			for name, err := range map[string]error{
				"UpdateConfig":     reg.UpdateConfig("org/model", VLLMConfig{MaxModelLen: 1}),
				"SetEnabled":       reg.SetEnabled("org/model", false),
				"Register":         reg.Register(&Model{ID: "org/new"}),
				"Delete":           reg.Delete("org/model", false),
				"SetPendingConfig": reg.SetPendingConfig(PendingConfig{ModelID: "org/other"}),
			} {
				if err == nil || !strings.Contains(err.Error(), path) {
					t.Errorf("%s: err = %v, want a refusal naming %s", name, err, path)
				}
			}
			if reg.DiscardPendingConfig("org/waiting") {
				t.Error("DiscardPendingConfig reported success on a read-only registry")
			}
			reg.Maintenance()

			got, _ := os.ReadFile(path)
			if !bytes.Equal(got, []byte(tc.body)) {
				t.Errorf("the file was rewritten:\n got %s\nwant %s", got, tc.body)
			}
		})
	}
}

// What parsed is still shown: a read-only registry that also looked empty
// would send the operator hunting for models that were never lost.
func TestNewerSchemaStillListsModels(t *testing.T) {
	dir := t.TempDir()
	writeRegistryFile(t, dir, strings.Replace(v2File, `"schema_version": 2`, `"schema_version": 3`, 1))

	reg := NewRegistry(dir, filepath.Join(dir, "models"))
	if _, ok := reg.Get("org/model"); !ok {
		t.Error("models from a newer file should still be listed")
	}
}

// A refused mutation must not have happened in memory either. Otherwise the
// next start launches a config the panel reported as not saved.
func TestRefusedUpdateLeavesMemoryAlone(t *testing.T) {
	dir := t.TempDir()
	writeRegistryFile(t, dir, strings.Replace(v2File, `"schema_version": 2`, `"schema_version": 99`, 1))
	reg := NewRegistry(dir, filepath.Join(dir, "models"))

	reg.UpdateConfig("org/model", VLLMConfig{MaxModelLen: 1})
	if m, _ := reg.Get("org/model"); m.VLLMConfig.MaxModelLen != 4096 {
		t.Errorf("MaxModelLen = %d after a refused update, want 4096", m.VLLMConfig.MaxModelLen)
	}
}

// Delete removes files before it writes the registry, so the refusal has to
// come first or a read-only registry still deletes tens of gigabytes.
func TestReadOnlyDeleteKeepsModelFiles(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "org", "model")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	weights := filepath.Join(modelDir, "model.safetensors")
	if err := os.WriteFile(weights, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeRegistryFile(t, dir, `{"schema_version": 99, "models": {"org/model":
		{"id": "org/model", "local_path": "`+modelDir+`"}}}`)

	reg := NewRegistry(dir, filepath.Join(dir, "models"))
	if err := reg.Delete("org/model", true); err == nil {
		t.Fatal("Delete succeeded on a read-only registry")
	}
	if _, err := os.Stat(weights); err != nil {
		t.Errorf("the model's files were deleted: %v", err)
	}
}

func TestSaveLeavesNoTempFile(t *testing.T) {
	reg, dir := pendingRegistry(t)
	if err := reg.Register(&Model{ID: "org/model"}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "config", "models.json.tmp")); !os.IsNotExist(err) {
		t.Errorf("models.json.tmp left behind: %v", err)
	}
}

package fsutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteJSONAtomicCreatesTheDirectoryAndLeavesNoTemp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "state.json")
	if err := WriteJSONAtomic(path, map[string]int{"a": 1}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]int
	if err := json.Unmarshal(data, &got); err != nil || got["a"] != 1 {
		t.Fatalf("read back %q: %v", data, err)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("temp file left behind: %v", err)
	}
}

// A value that cannot be marshalled must not touch the existing file.
func TestWriteJSONAtomicKeepsTheOldFileOnFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte(`{"old":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteJSONAtomic(path, map[string]any{"bad": make(chan int)}); err == nil {
		t.Fatal("marshalling a channel succeeded")
	}
	if data, _ := os.ReadFile(path); string(data) != `{"old":true}` {
		t.Errorf("file = %q, want the old contents", data)
	}
}

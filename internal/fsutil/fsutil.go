// Package fsutil holds the file writes the stores share.
package fsutil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// WriteJSONAtomic writes v as indented JSON to path, creating its directory.
// It writes a temporary file and renames it over path: os.WriteFile
// truncates first, so a crash or a full disk mid-write would leave a
// half-file the next load cannot parse, where this leaves the old one.
func WriteJSONAtomic(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// RefuseWrite is the error a store returns when it loaded path but will not
// write it back. reason completes "it ..." -- "could not be parsed (...)",
// "is schema version 4, and this build writes 3".
func RefuseWrite(path, reason string) error {
	return fmt.Errorf("refusing to write %s: it %s — move it aside or fix it, then restart", path, reason)
}

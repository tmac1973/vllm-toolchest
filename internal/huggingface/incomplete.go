package huggingface

import (
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
)

// Incomplete is a download that stopped partway and still has bytes on disk.
type Incomplete struct {
	ModelID   string `json:"model_id"`
	OnDisk    int64  `json:"on_disk_bytes"`
	PartFiles int    `json:"part_files"`
}

// ListIncomplete finds every model directory holding .part files.
//
// Cancelling or failing a download now leaves those in place so it can be
// resumed, which means something has to be able to see them — otherwise a
// abandoned 30GB transfer is invisible disk usage nobody can find or reclaim.
//
// A directory with no .part files is a finished download and is not reported,
// whether or not the registry knows about it; that is the model scan's job.
func (d *Downloader) ListIncomplete() []Incomplete {
	root := filepath.Join(d.dataDir, "models")

	// owner/name is two levels below root, matching modelDir().
	byModel := map[string]*Incomplete{}
	_ = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() || !strings.HasSuffix(entry.Name(), ".part") {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return nil
		}
		parts := strings.Split(filepath.ToSlash(rel), "/")
		if len(parts) < 3 {
			// Not owner/name/file — not something modelDir() would have
			// written, so not ours to describe.
			return nil
		}
		modelID := parts[0] + "/" + parts[1]

		rec := byModel[modelID]
		if rec == nil {
			rec = &Incomplete{ModelID: modelID}
			byModel[modelID] = rec
		}
		rec.PartFiles++
		if info, statErr := entry.Info(); statErr == nil {
			rec.OnDisk += info.Size()
		}
		return nil
	})

	out := make([]Incomplete, 0, len(byModel))
	for _, rec := range byModel {
		out = append(out, *rec)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ModelID < out[j].ModelID })
	return out
}

// ActiveModelIDs is the set of models with a download still running, so the
// caller can tell a paused transfer from one still in flight.
func (d *Downloader) ActiveModelIDs() map[string]bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := make(map[string]bool, len(d.active))
	for _, dl := range d.active {
		dl.mu.Lock()
		running := dl.status == "downloading"
		modelID := dl.modelID
		dl.mu.Unlock()
		if running {
			out[modelID] = true
		}
	}
	return out
}

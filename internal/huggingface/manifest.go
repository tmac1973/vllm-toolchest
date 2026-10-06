package huggingface

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/fsutil"
)

// manifestName is the record, kept beside a model's files, of what each of
// them is. It lives in the model directory rather than the registry so it
// travels with the files: moving the models directory, or removing a model
// from the registry and scanning it back in, must not lose it, because
// rebuilding it means reading every byte of the checkpoint.
const manifestName = ".hf-manifest.json"

const (
	identitySHA256 = "sha256:"
	identityGit    = "git:"
)

// Manifest says which upstream file each local file is a copy of.
//
// Without it there is nothing to compare a repo's current listing against, and
// the only way to pick up a maintainer's changes is to delete the model and
// fetch all of it again.
type Manifest struct {
	ModelID string `json:"model_id"`
	// Revision is the commit the directory was last brought fully in line
	// with. Empty until a transfer has finished.
	Revision  string    `json:"revision,omitempty"`
	UpdatedAt time.Time `json:"updated_at"`
	// Files is keyed by path within the model directory.
	Files map[string]ManifestFile `json:"files"`
	// Parts describes the .part files: which upstream file each is a download
	// of, and whether it is finished. A finished one is a replacement waiting
	// for the rest of its update, see Downloader.run.
	Parts map[string]ManifestPart `json:"parts,omitempty"`
}

type ManifestFile struct {
	Size     int64  `json:"size"`
	Identity string `json:"identity"`
}

type ManifestPart struct {
	Identity string `json:"identity"`
	// Verified is set once the whole file has arrived and its hash matched.
	Verified bool `json:"verified,omitempty"`
}

func newManifest(modelID string) *Manifest {
	return &Manifest{
		ModelID: modelID,
		Files:   map[string]ManifestFile{},
		Parts:   map[string]ManifestPart{},
	}
}

// loadManifest reads a model directory's manifest. A missing or unreadable one
// comes back empty rather than as an error: it only ever saves work, so the
// right response to losing it is to do that work again.
func loadManifest(modelDir, modelID string) *Manifest {
	m := newManifest(modelID)
	data, err := os.ReadFile(filepath.Join(modelDir, manifestName))
	if err != nil {
		return m
	}
	var onDisk Manifest
	if json.Unmarshal(data, &onDisk) != nil {
		return m
	}
	if onDisk.Files != nil {
		m.Files = onDisk.Files
	}
	if onDisk.Parts != nil {
		m.Parts = onDisk.Parts
	}
	m.Revision = onDisk.Revision
	m.UpdatedAt = onDisk.UpdatedAt
	return m
}

func (m *Manifest) save(modelDir string) error {
	m.UpdatedAt = time.Now().UTC()
	return fsutil.WriteJSONAtomic(filepath.Join(modelDir, manifestName), m)
}

// FileState is where one upstream file stands against the copy on disk.
type FileState string

const (
	// FileCurrent is known to be the upstream file.
	FileCurrent FileState = "current"
	// FileNew is not on disk at all.
	FileNew FileState = "new"
	// FileChanged is on disk, and known to differ.
	FileChanged FileState = "changed"
	// FileUnverified is on disk at the right size but has never been compared,
	// which is every file of a model downloaded before manifests existed. Only
	// hashing it can say whether it is current.
	FileUnverified FileState = "unverified"
)

// PlannedFile is one upstream file and what a transfer would do about it.
type PlannedFile struct {
	ModelFile
	State FileState `json:"state"`
	// PartBytes is how much of it an earlier, interrupted transfer already
	// fetched.
	PartBytes int64 `json:"part_bytes,omitempty"`
}

// Plan is what bringing a model directory in line with a revision involves.
// Building one reads no file contents, so it is cheap enough to show before
// anything is committed to.
type Plan struct {
	ModelID  string `json:"model_id"`
	Revision string `json:"revision"`
	// LocalRevision is the commit the directory was last fully synced to, or
	// "" when that was never recorded.
	LocalRevision string        `json:"local_revision,omitempty"`
	Files         []PlannedFile `json:"files"`
	// Stale lists files an earlier transfer fetched that upstream has since
	// dropped. They are reported, and removed only on request: they may be
	// the only copy of something the operator still wants.
	Stale []string `json:"stale,omitempty"`
}

// Pending is the files that are not known to be current.
func (p Plan) Pending() []PlannedFile {
	var out []PlannedFile
	for _, f := range p.Files {
		if f.State != FileCurrent {
			out = append(out, f)
		}
	}
	return out
}

// FetchBytes is what is certain to be downloaded: new and changed files, less
// what is already in their .part files. Unverified files are not counted,
// since whether they need fetching is what verifying them finds out.
func (p Plan) FetchBytes() int64 {
	var n int64
	for _, f := range p.Files {
		if f.State != FileNew && f.State != FileChanged {
			continue
		}
		if rest := f.Size - f.PartBytes; rest > 0 {
			n += rest
		}
	}
	return n
}

// Plan compares a model's directory against an upstream file listing.
func (d *Downloader) Plan(modelID, revision string, files []ModelFile) Plan {
	modelDir := d.modelDir(modelID)
	return buildPlan(modelDir, loadManifest(modelDir, modelID), modelID, revision, files)
}

func buildPlan(modelDir string, m *Manifest, modelID, revision string, files []ModelFile) Plan {
	plan := Plan{
		ModelID:       modelID,
		Revision:      revision,
		LocalRevision: m.Revision,
	}
	upstream := make(map[string]bool, len(files))
	for _, f := range files {
		upstream[f.Filename] = true
		plan.Files = append(plan.Files, classify(modelDir, m, f))
	}
	for name := range m.Files {
		if upstream[name] {
			continue
		}
		if _, err := os.Stat(filepath.Join(modelDir, name)); err == nil {
			plan.Stale = append(plan.Stale, name)
		}
	}
	sort.Strings(plan.Stale)
	return plan
}

func classify(modelDir string, m *Manifest, f ModelFile) PlannedFile {
	p := PlannedFile{ModelFile: f}
	finalPath := filepath.Join(modelDir, f.Filename)
	if info, err := os.Stat(finalPath + ".part"); err == nil {
		p.PartBytes = info.Size()
	}

	info, err := os.Stat(finalPath)
	if err != nil {
		p.State = FileNew
		return p
	}

	id := f.Identity()
	if id == "" {
		// Nothing to compare against, so fall back to what downloads did
		// before identities were known: a non-empty file is taken as done.
		if info.Size() > 0 {
			p.State = FileCurrent
		} else {
			p.State = FileNew
		}
		return p
	}

	// The record only counts while it still describes the file on disk. A
	// size that has moved means something else wrote the file since.
	if rec, ok := m.Files[f.Filename]; ok && rec.Size == info.Size() {
		if rec.Identity == id {
			p.State = FileCurrent
		} else {
			p.State = FileChanged
		}
		return p
	}

	if info.Size() != f.Size {
		p.State = FileChanged
		return p
	}
	p.State = FileUnverified
	return p
}

// newIdentityHash returns the hash that produces identities of the same kind
// as id, for a file of the given size, or nil when id is not one this package
// can check.
func newIdentityHash(id string, size int64) hash.Hash {
	switch {
	case strings.HasPrefix(id, identitySHA256):
		return sha256.New()
	case strings.HasPrefix(id, identityGit):
		// A git blob id covers a header naming the length, then the content.
		h := sha1.New()
		fmt.Fprintf(h, "blob %d\x00", size)
		return h
	}
	return nil
}

// identityOf finishes a hash from newIdentityHash into an identity of the same
// kind as like.
func identityOf(like string, h hash.Hash) string {
	sum := hex.EncodeToString(h.Sum(nil))
	if strings.HasPrefix(like, identityGit) {
		return identityGit + sum
	}
	return identitySHA256 + sum
}

// hashFile computes the identity of the file at path, of the same kind as
// like. progress, when not nil, is called with the bytes read so far; the
// checkpoints this runs over are tens of gigabytes, and minutes of silence
// reads as a hang.
func hashFile(ctx context.Context, path, like string, progress func(done int64)) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return "", err
	}
	h := newIdentityHash(like, info.Size())
	if h == nil {
		return "", fmt.Errorf("unrecognised file identity %q", like)
	}

	buf := make([]byte, 1024*1024)
	var done int64
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, readErr := file.Read(buf)
		if n > 0 {
			h.Write(buf[:n])
			done += int64(n)
			if progress != nil {
				progress(done)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", readErr
		}
	}
	return identityOf(like, h), nil
}

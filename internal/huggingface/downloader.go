package huggingface

import (
	"context"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// CompletionFunc is called when a model download finishes successfully.
type CompletionFunc func(downloadID, modelID, modelDir string)

type Downloader struct {
	dataDir string
	// modelsDir is where model files are written. Separate from dataDir
	// because the Settings page can point it somewhere else entirely — a
	// second disk, usually — while registry state stays under dataDir.
	modelsDir     string
	token         string
	maxConcurrent int
	onComplete    CompletionFunc
	// baseURL is the Hub's origin. Only ever the real one outside tests.
	baseURL string
	// httpClient fetches the files. See newDownloadClient.
	httpClient *http.Client

	mu     sync.Mutex
	active map[string]*download
}

func NewDownloader(dataDir, modelsDir, token string) *Downloader {
	return &Downloader{
		dataDir:       dataDir,
		modelsDir:     modelsDir,
		token:         token,
		maxConcurrent: 3,
		baseURL:       baseURL,
		httpClient:    newDownloadClient(),
		active:        make(map[string]*download),
	}
}

// newDownloadClient has no overall Timeout, unlike the API client: a weight
// shard can take an hour to arrive, and a Timeout covers reading the body.
// What it does bound is waiting for the Hub to answer at all, so a request
// to a hung server fails instead of sitting in the queue forever.
func newDownloadClient() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = 60 * time.Second
	return &http.Client{Transport: t}
}

// SetBaseURL points the downloader at a different origin, for tests.
func (d *Downloader) SetBaseURL(u string) {
	d.baseURL = strings.TrimRight(u, "/")
}

func (d *Downloader) SetOnComplete(fn CompletionFunc) {
	d.onComplete = fn
}

func (d *Downloader) SetToken(token string) {
	d.token = token
}

// DownloadProgress represents current download state.
type DownloadProgress struct {
	ID              string `json:"id"`
	ModelID         string `json:"model_id"`
	TotalBytes      int64  `json:"total_bytes"`
	BytesDownloaded int64  `json:"bytes_downloaded"`
	SpeedBPS        int64  `json:"speed_bps"`
	Status          string `json:"status"` // downloading, complete, failed, cancelled
	Error           string `json:"error,omitempty"`
	ActiveFiles     int    `json:"active_files"`
	CompletedFiles  int    `json:"completed_files"`
	TotalFiles      int    `json:"total_files"`
	// VerifyingFiles is how many files are being hashed rather than fetched.
	// While that is all a transfer is doing, its byte counts are disk reads.
	VerifyingFiles int `json:"verifying_files,omitempty"`
	// Update marks a transfer into a model that was already registered.
	Update bool           `json:"update,omitempty"`
	Files  []FileProgress `json:"files,omitempty"`
}

type FileProgress struct {
	Filename   string `json:"filename"`
	Size       int64  `json:"size"`
	Downloaded int64  `json:"downloaded"`
	Status     string `json:"status"` // pending, verifying, downloading, staged, complete, failed
}

type download struct {
	id      string
	modelID string
	update  bool
	cancel  context.CancelFunc

	mu         sync.Mutex
	files      []fileState
	totalBytes int64
	// fetched is what this run has pulled over the network. The rate is
	// worked out from it rather than from the files' progress, which also
	// counts bytes that were already on disk and bytes that were only hashed.
	fetched   int64
	status    string
	errMsg    string
	startedAt time.Time
}

type fileState struct {
	filename   string
	size       int64
	downloaded int64
	status     string
}

func (dl *download) getProgress() DownloadProgress {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	var downloaded int64
	var completed, active, verifying int
	files := make([]FileProgress, len(dl.files))
	for i, f := range dl.files {
		downloaded += f.downloaded
		switch f.status {
		case "complete", "staged":
			completed++
		case "downloading":
			active++
		case "verifying":
			verifying++
		}
		files[i] = FileProgress{
			Filename:   f.filename,
			Size:       f.size,
			Downloaded: f.downloaded,
			Status:     f.status,
		}
	}

	elapsed := time.Since(dl.startedAt).Seconds()
	speed := int64(0)
	if elapsed > 0 {
		speed = int64(float64(dl.fetched) / elapsed)
	}

	return DownloadProgress{
		ID:              dl.id,
		ModelID:         dl.modelID,
		TotalBytes:      dl.totalBytes,
		BytesDownloaded: downloaded,
		SpeedBPS:        speed,
		Status:          dl.status,
		Error:           dl.errMsg,
		ActiveFiles:     active,
		CompletedFiles:  completed,
		TotalFiles:      len(dl.files),
		VerifyingFiles:  verifying,
		Update:          dl.update,
		Files:           files,
	}
}

// Request describes one transfer: bring a model's directory in line with a
// file listing.
type Request struct {
	ModelID string
	// Revision is the commit Files was listed at, and the one they are
	// fetched from. Empty follows main, and records no revision.
	Revision string
	// Files is everything the directory should hold, not only what is
	// missing: which of them need fetching is worked out against the disk.
	Files []ModelFile
	// RemoveStale deletes files an earlier transfer fetched that are no
	// longer in Files.
	RemoveStale bool
	// Update marks a transfer into a model that is already registered.
	Update bool
}

// Start begins a transfer. Returns the download ID.
//
// A first download, a resumed one and an update of a model that is already
// complete are the same operation: compare Files with the directory, fetch
// what differs. Only the files with work to do are tracked, so the progress
// of a two-file update is out of two and not out of forty.
func (d *Downloader) Start(req Request) (string, error) {
	modelID := req.ModelID
	if err := CheckModelID(modelID); err != nil {
		return "", err
	}
	id := DownloadID(modelID)

	for _, f := range req.Files {
		// The names come from the Hub and are joined onto the model
		// directory; one that climbs out of it has no business being written.
		if !filepath.IsLocal(f.Filename) {
			return "", fmt.Errorf("refusing file outside the model directory: %q", f.Filename)
		}
	}

	d.mu.Lock()
	if existing, exists := d.active[id]; exists {
		existing.mu.Lock()
		running := existing.status == "downloading"
		existing.mu.Unlock()
		if running {
			d.mu.Unlock()
			return id, nil // already downloading
		}
		// A settled entry lingers for 30s so a late progress poll can read its
		// final state. Resuming inside that window has to replace it, or the
		// resume silently becomes a no-op that returns the dead download's id.
		delete(d.active, id)
	}

	ctx, cancel := context.WithCancel(context.Background())

	modelDir := d.modelDir(modelID)
	manifest := loadManifest(modelDir, modelID)
	plan := buildPlan(modelDir, manifest, modelID, req.Revision, req.Files)
	pending := plan.Pending()

	var totalBytes int64
	states := make([]fileState, len(pending))
	for i, f := range pending {
		totalBytes += f.Size
		states[i] = fileState{
			filename: f.Filename,
			size:     f.Size,
			status:   "pending",
		}
		// What an interrupted transfer left behind counts as done, so a
		// resumed bar starts where the last one stopped.
		if f.State != FileUnverified && f.PartBytes <= f.Size {
			states[i].downloaded = f.PartBytes
		}
	}

	dl := &download{
		id:         id,
		modelID:    modelID,
		update:     req.Update,
		cancel:     cancel,
		files:      states,
		totalBytes: totalBytes,
		status:     "downloading",
		startedAt:  time.Now(),
	}
	d.active[id] = dl
	d.mu.Unlock()

	t := &transfer{
		d:        d,
		dl:       dl,
		modelID:  modelID,
		modelDir: modelDir,
		revision: req.Revision,
		manifest: manifest,
	}
	go d.run(ctx, t, plan, req.RemoveStale)
	return id, nil
}

func (d *Downloader) run(ctx context.Context, t *transfer, plan Plan, removeStale bool) {
	dl := t.dl
	defer d.cleanup(dl.id, dl)

	os.MkdirAll(t.modelDir, 0o755)

	// Separate config files (small, download first) from weight files (large, concurrent)
	var configFiles, weightFiles []PlannedFile
	for _, f := range plan.Pending() {
		if f.Category == "weight" {
			weightFiles = append(weightFiles, f)
		} else {
			configFiles = append(configFiles, f)
		}
	}

	var firstErr error

	// Config files first (sequential)
	for _, f := range configFiles {
		if firstErr = t.bring(ctx, f); firstErr != nil {
			break
		}
	}

	// Weight files (concurrent)
	if firstErr == nil {
		sem := make(chan struct{}, d.maxConcurrent)
		var wg sync.WaitGroup
		var errOnce sync.Once

	weights:
		for _, f := range weightFiles {
			select {
			case <-ctx.Done():
				break weights
			case sem <- struct{}{}:
			}

			wg.Add(1)
			go func(f PlannedFile) {
				defer wg.Done()
				defer func() { <-sem }()
				if err := t.bring(ctx, f); err != nil {
					errOnce.Do(func() { firstErr = err })
				}
			}(f)
		}
		wg.Wait()
	}

	// Nothing on disk changes meaning until every file has arrived: the
	// replacements are still .part files, and this is where they go in.
	if firstErr == nil && ctx.Err() == nil {
		firstErr = t.commit(plan, removeStale)
	}

	dl.mu.Lock()
	switch {
	case ctx.Err() != nil:
		dl.status = "cancelled"
	case firstErr != nil:
		dl.status = "failed"
		dl.errMsg = firstErr.Error()
	default:
		dl.status = "complete"
	}
	status := dl.status
	dl.mu.Unlock()

	if status == "complete" && d.onComplete != nil {
		d.onComplete(dl.id, t.modelID, t.modelDir)
	}
}

// Discard removes a model's partially-downloaded files.
//
// This is the only path that deletes them. Cancelling and failing used to call
// it implicitly, which threw away every byte of a 30GB transfer on a network
// blip — and made the Range-resume support in downloadFile unreachable, since
// there was never a .part file left to resume from.
func (d *Downloader) Discard(modelID string) error {
	dir, err := d.checkedModelDir(modelID)
	if err != nil {
		return err
	}
	if err := os.RemoveAll(dir); err != nil {
		return err
	}
	// Model dirs are owner/name, so removing the last model by an owner leaves
	// an empty owner directory behind. Remove returns an error for a non-empty
	// one, which is exactly the check wanted here.
	_ = os.Remove(filepath.Dir(dir))
	return nil
}

// DiscardParts removes a model's .part files and nothing else.
//
// It is what Discard has to become once the model is complete and registered:
// an interrupted update leaves partial replacements beside weights that are
// still good, and clearing the first must not take the second with it.
func (d *Downloader) DiscardParts(modelID string) error {
	dir, err := d.checkedModelDir(modelID)
	if err != nil {
		return err
	}
	var firstErr error
	_ = filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() || !strings.HasSuffix(entry.Name(), ".part") {
			return nil
		}
		if rmErr := os.Remove(path); rmErr != nil && firstErr == nil {
			firstErr = rmErr
		}
		return nil
	})
	if firstErr != nil {
		return firstErr
	}
	// The manifest's notes about those files describe nothing now.
	if m := loadManifest(dir, modelID); len(m.Parts) > 0 {
		m.Parts = map[string]ManifestPart{}
		return m.save(dir)
	}
	return nil
}

// CheckModelID refuses anything but the owner/name shape a Hub repository
// has. A model's directory is the id joined onto the models root, so an id
// that is empty, has one part or three, or climbs with "." or ".." would
// read, write -- and, on delete, remove -- somewhere other than its own
// directory: "acme/.." is the models root itself.
func CheckModelID(modelID string) error {
	owner, name, ok := strings.Cut(modelID, "/")
	if !ok || owner == "" || name == "" ||
		strings.Contains(name, "/") || strings.ContainsRune(modelID, '\\') ||
		owner == "." || owner == ".." || name == "." || name == ".." {
		return fmt.Errorf("not a model id: %q", modelID)
	}
	return nil
}

// checkedModelDir is modelDir for callers that delete things.
func (d *Downloader) checkedModelDir(modelID string) (string, error) {
	if err := CheckModelID(modelID); err != nil {
		return "", err
	}

	dir := d.modelDir(modelID)
	if dir == "" || dir == d.dataDir || dir == d.modelsDir {
		return "", fmt.Errorf("refusing to remove %q", dir)
	}
	return dir, nil
}

// transfer is one run's view of the model directory it is filling.
type transfer struct {
	d        *Downloader
	dl       *download
	modelID  string
	modelDir string
	revision string

	// mu guards manifest, which the concurrent weight downloads all write.
	mu       sync.Mutex
	manifest *Manifest
}

// record applies a change to the manifest and writes it out, so that what has
// been established so far survives the run being interrupted.
func (t *transfer) record(change func(m *Manifest)) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	change(t.manifest)
	if err := t.manifest.save(t.modelDir); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
}

func (t *transfer) part(filename string) (ManifestPart, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	rec, ok := t.manifest.Parts[filename]
	return rec, ok
}

// bring makes one file current: by establishing that it already is, or by
// fetching it.
func (t *transfer) bring(ctx context.Context, f PlannedFile) error {
	if f.State == FileUnverified {
		id := f.Identity()
		t.d.updateFileState(t.dl, f.Filename, 0, "verifying")

		lastBroadcast := time.Now()
		got, err := hashFile(ctx, filepath.Join(t.modelDir, f.Filename), id, func(done int64) {
			t.dl.setDownloaded(f.Filename, done)
			if time.Since(lastBroadcast) > 500*time.Millisecond {
				lastBroadcast = time.Now()
			}
		})
		if err != nil {
			t.d.updateFileState(t.dl, f.Filename, 0, "failed")
			return fmt.Errorf("verify %s: %w", f.Filename, err)
		}
		if got == id {
			if err := t.record(func(m *Manifest) {
				m.Files[f.Filename] = ManifestFile{Size: f.Size, Identity: id}
			}); err != nil {
				return err
			}
			t.d.updateFileState(t.dl, f.Filename, f.Size, "complete")
			return nil
		}
		// Same name and same size as upstream, different bytes.
		t.d.updateFileState(t.dl, f.Filename, 0, "pending")
	}
	return t.fetch(ctx, f)
}

// fetch downloads one file to its .part, checks it, and puts it in place.
func (t *transfer) fetch(ctx context.Context, f PlannedFile) error {
	d, dl := t.d, t.dl
	finalPath := filepath.Join(t.modelDir, f.Filename)
	partPath := finalPath + ".part"
	id := f.Identity()

	// Ensure subdirectories exist
	if dir := filepath.Dir(finalPath); dir != t.modelDir {
		os.MkdirAll(dir, 0o755)
	}

	_, statErr := os.Stat(finalPath)
	replacing := statErr == nil

	// Check existing partial
	var existingSize int64
	if info, err := os.Stat(partPath); err == nil {
		existingSize = info.Size()
	}
	if id != "" && existingSize > 0 {
		rec, known := t.part(f.Filename)
		switch {
		case (known && rec.Identity != id) || existingSize > f.Size:
			// A download of some other version of this file: upstream moved
			// on while the transfer was paused. Appending to it would build
			// a file that is neither.
			_ = os.Remove(partPath)
			existingSize = 0
		case known && rec.Verified && existingSize == f.Size:
			return t.place(f, replacing)
		}
	}
	if id != "" {
		if rec, known := t.part(f.Filename); !known || rec.Identity != id || rec.Verified {
			if err := t.record(func(m *Manifest) {
				m.Parts[f.Filename] = ManifestPart{Identity: id}
			}); err != nil {
				return err
			}
		}
	}

	// streamed is the hash of the file as it arrived. It stays nil when some
	// of the file was already on disk, and the check then has to read it back.
	var streamed hash.Hash

	// A .part that is already the full length needs checking, not fetching:
	// asking for the bytes after its end is an error.
	if id != "" && existingSize == f.Size && f.Size > 0 {
		d.updateFileState(dl, f.Filename, existingSize, "verifying")
	} else {
		// Checked here as well as before the transfer starts, because how
		// much an update fetches is not known until its files are verified.
		if free := d.FreeBytes(); free >= 0 && free-DiskSafetyMarginBytes < f.Size-existingSize {
			d.updateFileState(dl, f.Filename, existingSize, "failed")
			return fmt.Errorf("not enough disk space for %s: needs %s, and %s is free after the %s safety margin",
				f.Filename, FormatBytes(f.Size-existingSize),
				FormatBytes(max(free-DiskSafetyMarginBytes, 0)), FormatBytes(DiskSafetyMarginBytes))
		}

		downloadURL := fileURL(d.baseURL, t.modelID, t.revision, f.Filename)
		req, err := http.NewRequestWithContext(ctx, "GET", downloadURL, nil)
		if err != nil {
			return err
		}
		if d.token != "" {
			req.Header.Set("Authorization", "Bearer "+d.token)
		}
		if existingSize > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
		}

		resp, err := d.httpClient.Do(req)
		if err != nil {
			d.updateFileState(dl, f.Filename, existingSize, "failed")
			return fmt.Errorf("download %s: %w", f.Filename, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == 403 {
			d.updateFileState(dl, f.Filename, 0, "failed")
			return fmt.Errorf("access denied for %s — is this a gated model? Accept the license at https://huggingface.co/%s", f.Filename, t.modelID)
		}
		if resp.StatusCode != 200 && resp.StatusCode != 206 {
			d.updateFileState(dl, f.Filename, 0, "failed")
			return fmt.Errorf("download %s: HTTP %d", f.Filename, resp.StatusCode)
		}

		// Open file for writing (append if resuming)
		flags := os.O_CREATE | os.O_WRONLY
		if resp.StatusCode == 206 {
			flags |= os.O_APPEND
		} else {
			flags |= os.O_TRUNC
			existingSize = 0
			streamed = newIdentityHash(id, f.Size)
		}
		file, err := os.OpenFile(partPath, flags, 0o644)
		if err != nil {
			return fmt.Errorf("open %s: %w", partPath, err)
		}
		defer file.Close()

		d.updateFileState(dl, f.Filename, existingSize, "downloading")

		// Stream data with progress updates
		buf := make([]byte, 256*1024) // 256KB buffer
		downloaded := existingSize
		lastBroadcast := time.Now()

		for {
			n, readErr := resp.Body.Read(buf)
			if n > 0 {
				if _, writeErr := file.Write(buf[:n]); writeErr != nil {
					return fmt.Errorf("write %s: %w", f.Filename, writeErr)
				}
				if streamed != nil {
					streamed.Write(buf[:n])
				}
				downloaded += int64(n)

				dl.mu.Lock()
				dl.fetched += int64(n)
				dl.mu.Unlock()
				dl.setDownloaded(f.Filename, downloaded)

				if time.Since(lastBroadcast) > 500*time.Millisecond {
					lastBroadcast = time.Now()
				}
			}
			if readErr != nil {
				if readErr == io.EOF {
					break
				}
				return fmt.Errorf("read %s: %w", f.Filename, readErr)
			}
		}
		if err := file.Close(); err != nil {
			return fmt.Errorf("write %s: %w", f.Filename, err)
		}
	}

	if id != "" {
		var got string
		if streamed != nil {
			got = identityOf(id, streamed)
		} else {
			var err error
			if got, err = hashFile(ctx, partPath, id, nil); err != nil {
				return fmt.Errorf("verify %s: %w", f.Filename, err)
			}
		}
		if got != id {
			// Kept, it would be resumed from and fail the same way forever.
			_ = os.Remove(partPath)
			d.updateFileState(dl, f.Filename, 0, "failed")
			if err := t.record(func(m *Manifest) { delete(m.Parts, f.Filename) }); err != nil {
				return err
			}
			return fmt.Errorf("%s did not match its checksum after download; the partial file was discarded, try again", f.Filename)
		}
		if err := t.record(func(m *Manifest) {
			m.Parts[f.Filename] = ManifestPart{Identity: id, Verified: true}
		}); err != nil {
			return err
		}
	}

	return t.place(f, replacing)
}

// place puts a finished .part where it belongs — unless it replaces a file
// the model already has.
//
// A replacement waits as a .part until the whole transfer has succeeded. An
// update that swapped files in as they arrived would, on failing halfway,
// leave shards from two different snapshots behind one index: a model that
// loaded before the update was attempted and does not after. Files that are
// new to the directory displace nothing, so they go straight in.
func (t *transfer) place(f PlannedFile, replacing bool) error {
	id := f.Identity()
	if replacing && id != "" {
		t.d.updateFileState(t.dl, f.Filename, f.Size, "staged")
		return nil
	}

	finalPath := filepath.Join(t.modelDir, f.Filename)
	if err := os.Rename(finalPath+".part", finalPath); err != nil {
		return fmt.Errorf("rename %s: %w", f.Filename, err)
	}
	if id != "" {
		if err := t.record(func(m *Manifest) {
			m.Files[f.Filename] = ManifestFile{Size: f.Size, Identity: id}
			delete(m.Parts, f.Filename)
		}); err != nil {
			return err
		}
	}

	size := f.Size
	if info, err := os.Stat(finalPath); err == nil {
		size = info.Size()
	}
	t.d.updateFileState(t.dl, f.Filename, size, "complete")
	return nil
}

// commit finishes a transfer whose every file has arrived: the staged
// replacements are renamed over the files they replace, and the manifest is
// stamped with the revision the directory now matches.
func (t *transfer) commit(plan Plan, removeStale bool) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	m := t.manifest

	upstream := make(map[string]ModelFile, len(plan.Files))
	for _, f := range plan.Files {
		upstream[f.Filename] = f.ModelFile
	}

	names := make([]string, 0, len(m.Parts))
	for name := range m.Parts {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if !filepath.IsLocal(name) {
			delete(m.Parts, name)
			continue
		}
		finalPath := filepath.Join(t.modelDir, name)
		f, wanted := upstream[name]
		if !wanted {
			// A replacement for a file upstream has since dropped.
			_ = os.Remove(finalPath + ".part")
			delete(m.Parts, name)
			continue
		}
		if !m.Parts[name].Verified {
			continue
		}
		if err := os.Rename(finalPath+".part", finalPath); err != nil {
			return fmt.Errorf("rename %s: %w", name, err)
		}
		m.Files[name] = ManifestFile{Size: f.Size, Identity: f.Identity()}
		delete(m.Parts, name)
		t.d.updateFileState(t.dl, name, f.Size, "complete")
	}

	for _, name := range plan.Stale {
		if !removeStale || !filepath.IsLocal(name) {
			continue
		}
		if err := os.Remove(filepath.Join(t.modelDir, name)); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove %s: %w", name, err)
		}
		delete(m.Files, name)
		// A directory that only held stale files goes with them. Remove
		// fails on one that is not empty, which is the check wanted here.
		for dir := filepath.Dir(name); dir != "."; dir = filepath.Dir(dir) {
			if os.Remove(filepath.Join(t.modelDir, dir)) != nil {
				break
			}
		}
	}
	// Records of files that are gone from both ends describe nothing.
	for name := range m.Files {
		if _, wanted := upstream[name]; wanted {
			continue
		}
		if _, err := os.Stat(filepath.Join(t.modelDir, name)); os.IsNotExist(err) {
			delete(m.Files, name)
		}
	}

	if t.revision != "" {
		m.Revision = t.revision
	}
	if err := m.save(t.modelDir); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
}

func (dl *download) setDownloaded(filename string, downloaded int64) {
	dl.mu.Lock()
	for i := range dl.files {
		if dl.files[i].filename == filename {
			dl.files[i].downloaded = downloaded
			break
		}
	}
	dl.mu.Unlock()
}

func (d *Downloader) updateFileState(dl *download, filename string, downloaded int64, status string) {
	dl.mu.Lock()
	for i := range dl.files {
		if dl.files[i].filename == filename {
			dl.files[i].downloaded = downloaded
			dl.files[i].status = status
			break
		}
	}
	dl.mu.Unlock()
}

func (d *Downloader) cleanup(downloadID string, dl *download) {
	// Remove from active after a delay so a late progress poll can read the final state
	time.AfterFunc(30*time.Second, func() {
		d.mu.Lock()
		// Only if it is still this one: a transfer restarted inside the
		// window has taken the slot, and must not be reaped in its place.
		if d.active[downloadID] == dl {
			delete(d.active, downloadID)
		}
		d.mu.Unlock()
	})
}

// DownloadID is the id a model's transfer is tracked under. There is one per
// model, so it is derived from the model rather than handed out.
func DownloadID(modelID string) string {
	return strings.ReplaceAll(modelID, "/", "--")
}

// ModelDir is where a model's files are written.
func (d *Downloader) ModelDir(modelID string) string {
	return d.modelDir(modelID)
}

func (d *Downloader) modelDir(modelID string) string {
	parts := strings.SplitN(modelID, "/", 2)
	if len(parts) == 2 {
		return filepath.Join(d.modelsDir, parts[0], parts[1])
	}
	return filepath.Join(d.modelsDir, modelID)
}

// Cancel stops an in-progress download.
func (d *Downloader) Cancel(downloadID string) error {
	d.mu.Lock()
	dl, ok := d.active[downloadID]
	d.mu.Unlock()
	if !ok {
		return fmt.Errorf("no active download: %s", downloadID)
	}
	dl.cancel()
	return nil
}

// GetProgress returns progress for a specific download, or nil if not found.
func (d *Downloader) GetProgress(downloadID string) *DownloadProgress {
	d.mu.Lock()
	dl, ok := d.active[downloadID]
	d.mu.Unlock()
	if !ok {
		return nil
	}
	p := dl.getProgress()
	return &p
}

// ActiveDownloads returns progress for all in-progress downloads.
func (d *Downloader) ActiveDownloads() []DownloadProgress {
	d.mu.Lock()
	defer d.mu.Unlock()

	var result []DownloadProgress
	for _, dl := range d.active {
		result = append(result, dl.getProgress())
	}
	return result
}

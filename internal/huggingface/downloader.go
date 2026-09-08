package huggingface

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// CompletionFunc is called when a model download finishes successfully.
type CompletionFunc func(downloadID, modelID, modelDir string)

type Downloader struct {
	dataDir       string
	token         string
	maxConcurrent int
	onComplete    CompletionFunc

	mu     sync.Mutex
	active map[string]*download
}

func NewDownloader(dataDir, token string) *Downloader {
	return &Downloader{
		dataDir:       dataDir,
		token:         token,
		maxConcurrent: 3,
		active:        make(map[string]*download),
	}
}

func (d *Downloader) SetOnComplete(fn CompletionFunc) {
	d.onComplete = fn
}

func (d *Downloader) SetToken(token string) {
	d.token = token
}

// DownloadProgress represents current download state.
type DownloadProgress struct {
	ID              string         `json:"id"`
	ModelID         string         `json:"model_id"`
	TotalBytes      int64          `json:"total_bytes"`
	BytesDownloaded int64          `json:"bytes_downloaded"`
	SpeedBPS        int64          `json:"speed_bps"`
	Status          string         `json:"status"` // downloading, complete, failed, cancelled
	Error           string         `json:"error,omitempty"`
	ActiveFiles     int            `json:"active_files"`
	CompletedFiles  int            `json:"completed_files"`
	TotalFiles      int            `json:"total_files"`
	Files           []FileProgress `json:"files,omitempty"`
}

type FileProgress struct {
	Filename   string `json:"filename"`
	Size       int64  `json:"size"`
	Downloaded int64  `json:"downloaded"`
	Status     string `json:"status"` // pending, downloading, complete, failed
}

type download struct {
	id      string
	modelID string
	cancel  context.CancelFunc

	mu         sync.Mutex
	files      []fileState
	totalBytes int64
	status     string
	errMsg     string
	startedAt  time.Time

	subMu sync.Mutex
	subs  map[chan DownloadProgress]struct{}
}

type fileState struct {
	filename   string
	size       int64
	downloaded int64
	status     string
}

func (dl *download) broadcast() {
	progress := dl.getProgress()
	dl.subMu.Lock()
	for ch := range dl.subs {
		select {
		case ch <- progress:
		default:
		}
	}
	dl.subMu.Unlock()
}

func (dl *download) getProgress() DownloadProgress {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	var downloaded int64
	var completed, active int
	files := make([]FileProgress, len(dl.files))
	for i, f := range dl.files {
		downloaded += f.downloaded
		switch f.status {
		case "complete":
			completed++
		case "downloading":
			active++
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
		speed = int64(float64(downloaded) / elapsed)
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
		Files:           files,
	}
}

// Start begins downloading a model. Returns the download ID.
func (d *Downloader) Start(modelID string, files []ModelFile) (string, error) {
	id := strings.ReplaceAll(modelID, "/", "--")

	d.mu.Lock()
	if existing, exists := d.active[id]; exists {
		existing.mu.Lock()
		running := existing.status == "downloading"
		existing.mu.Unlock()
		if running {
			d.mu.Unlock()
			return id, nil // already downloading
		}
		// A settled entry lingers for 30s so late subscribers can read its
		// final state. Resuming inside that window has to replace it, or the
		// resume silently becomes a no-op that returns the dead download's id.
		delete(d.active, id)
	}

	ctx, cancel := context.WithCancel(context.Background())

	var totalBytes int64
	states := make([]fileState, len(files))
	for i, f := range files {
		totalBytes += f.Size
		states[i] = fileState{
			filename: f.Filename,
			size:     f.Size,
			status:   "pending",
		}
	}

	dl := &download{
		id:         id,
		modelID:    modelID,
		cancel:     cancel,
		files:      states,
		totalBytes: totalBytes,
		status:     "downloading",
		startedAt:  time.Now(),
		subs:       make(map[chan DownloadProgress]struct{}),
	}
	d.active[id] = dl
	d.mu.Unlock()

	go d.run(ctx, id, modelID, files, dl)
	return id, nil
}

func (d *Downloader) run(ctx context.Context, downloadID, modelID string, files []ModelFile, dl *download) {
	defer d.cleanup(downloadID, dl)

	modelDir := d.modelDir(modelID)
	os.MkdirAll(modelDir, 0o755)

	// Separate config files (small, download first) from weight files (large, concurrent)
	var configFiles, weightFiles []ModelFile
	for _, f := range files {
		if f.Category == "weight" {
			weightFiles = append(weightFiles, f)
		} else {
			configFiles = append(configFiles, f)
		}
	}

	// Config files first (sequential)
	for _, f := range configFiles {
		if err := d.downloadFile(ctx, modelID, modelDir, f, dl); err != nil {
			dl.mu.Lock()
			dl.status = "failed"
			dl.errMsg = err.Error()
			dl.mu.Unlock()
			dl.broadcast()
			return
		}
	}

	// Weight files (concurrent)
	sem := make(chan struct{}, d.maxConcurrent)
	var wg sync.WaitGroup
	var firstErr error
	var errOnce sync.Once

	for _, f := range weightFiles {
		select {
		case <-ctx.Done():
			dl.mu.Lock()
			dl.status = "cancelled"
			dl.mu.Unlock()
			dl.broadcast()
			return
		case sem <- struct{}{}:
		}

		wg.Add(1)
		go func(f ModelFile) {
			defer wg.Done()
			defer func() { <-sem }()
			if err := d.downloadFile(ctx, modelID, modelDir, f, dl); err != nil {
				errOnce.Do(func() { firstErr = err })
			}
		}(f)
	}
	wg.Wait()

	if firstErr != nil {
		dl.mu.Lock()
		dl.status = "failed"
		dl.errMsg = firstErr.Error()
		dl.mu.Unlock()
		dl.broadcast()
		return
	}

	dl.mu.Lock()
	dl.status = "complete"
	dl.mu.Unlock()
	dl.broadcast()

	if d.onComplete != nil {
		d.onComplete(downloadID, modelID, modelDir)
	}
}

// Discard removes a model's partially-downloaded files.
//
// This is the only path that deletes them. Cancelling and failing used to call
// it implicitly, which threw away every byte of a 30GB transfer on a network
// blip — and made the Range-resume support in downloadFile unreachable, since
// there was never a .part file left to resume from.
func (d *Downloader) Discard(modelID string) error {
	// modelDir falls back to joining whatever it is given, so an empty or
	// traversing id resolves to the models root — and this would then delete
	// every model on the box. Require the owner/name shape it actually writes.
	owner, name, ok := strings.Cut(modelID, "/")
	if !ok || owner == "" || name == "" ||
		strings.Contains(owner, "/") || strings.Contains(name, "/") ||
		owner == "." || owner == ".." || name == "." || name == ".." {
		return fmt.Errorf("not a model id: %q", modelID)
	}

	dir := d.modelDir(modelID)
	root := filepath.Join(d.dataDir, "models")
	if dir == "" || dir == d.dataDir || dir == root {
		return fmt.Errorf("refusing to remove %q", dir)
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

func (d *Downloader) downloadFile(ctx context.Context, modelID, modelDir string, f ModelFile, dl *download) error {
	finalPath := filepath.Join(modelDir, f.Filename)
	partPath := finalPath + ".part"

	// Ensure subdirectories exist
	if dir := filepath.Dir(finalPath); dir != modelDir {
		os.MkdirAll(dir, 0o755)
	}

	// Skip if already complete
	if info, err := os.Stat(finalPath); err == nil && info.Size() > 0 {
		d.updateFileState(dl, f.Filename, f.Size, "complete")
		return nil
	}

	// Check existing partial
	var existingSize int64
	if info, err := os.Stat(partPath); err == nil {
		existingSize = info.Size()
	}

	downloadURL := DownloadURL(modelID, f.Filename)
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

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		d.updateFileState(dl, f.Filename, existingSize, "failed")
		return fmt.Errorf("download %s: %w", f.Filename, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 403 {
		d.updateFileState(dl, f.Filename, 0, "failed")
		return fmt.Errorf("access denied for %s — is this a gated model? Accept the license at https://huggingface.co/%s", f.Filename, modelID)
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
			downloaded += int64(n)

			dl.mu.Lock()
			for i := range dl.files {
				if dl.files[i].filename == f.Filename {
					dl.files[i].downloaded = downloaded
					break
				}
			}
			dl.mu.Unlock()

			if time.Since(lastBroadcast) > 500*time.Millisecond {
				dl.broadcast()
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

	// Rename .part to final
	file.Close()
	if err := os.Rename(partPath, finalPath); err != nil {
		return fmt.Errorf("rename %s: %w", f.Filename, err)
	}

	d.updateFileState(dl, f.Filename, downloaded, "complete")
	dl.broadcast()
	return nil
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
	// Close all subscriber channels
	dl.subMu.Lock()
	for ch := range dl.subs {
		close(ch)
		delete(dl.subs, ch)
	}
	dl.subMu.Unlock()

	// Remove from active after a delay so late subscribers can read final state
	time.AfterFunc(30*time.Second, func() {
		d.mu.Lock()
		delete(d.active, downloadID)
		d.mu.Unlock()
	})
}

func (d *Downloader) modelDir(modelID string) string {
	parts := strings.SplitN(modelID, "/", 2)
	if len(parts) == 2 {
		return filepath.Join(d.dataDir, "models", parts[0], parts[1])
	}
	return filepath.Join(d.dataDir, "models", modelID)
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

// Subscribe returns a channel receiving progress updates for a download.
func (d *Downloader) Subscribe(downloadID string) (chan DownloadProgress, error) {
	d.mu.Lock()
	dl, ok := d.active[downloadID]
	d.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no active download: %s", downloadID)
	}

	ch := make(chan DownloadProgress, 8)
	dl.subMu.Lock()
	dl.subs[ch] = struct{}{}
	dl.subMu.Unlock()

	// Send current state immediately
	ch <- dl.getProgress()
	return ch, nil
}

// Unsubscribe removes a progress subscriber.
func (d *Downloader) Unsubscribe(downloadID string, ch chan DownloadProgress) {
	d.mu.Lock()
	dl, ok := d.active[downloadID]
	d.mu.Unlock()
	if !ok {
		return
	}
	dl.subMu.Lock()
	delete(dl.subs, ch)
	dl.subMu.Unlock()
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

package huggingface

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const testModel = "acme/widget"

// fakeHub serves a repo's files the way the Hub's resolve endpoint does, and
// counts what was asked for — the point of an update being what it does not
// fetch.
type fakeHub struct {
	t   *testing.T
	srv *httptest.Server

	mu      sync.Mutex
	content map[string]string
	broken  map[string]bool
	hits    map[string]int
	ranges  map[string]string
}

func newFakeHub(t *testing.T) *fakeHub {
	h := &fakeHub{
		t:       t,
		content: map[string]string{},
		broken:  map[string]bool{},
		hits:    map[string]int{},
		ranges:  map[string]string{},
	}
	h.srv = httptest.NewServer(http.HandlerFunc(h.serve))
	t.Cleanup(h.srv.Close)
	return h
}

func (h *fakeHub) serve(w http.ResponseWriter, r *http.Request) {
	// /<owner>/<name>/resolve/<revision>/<path…>
	_, rest, ok := strings.Cut(r.URL.Path, "/resolve/")
	if !ok {
		http.NotFound(w, r)
		return
	}
	_, name, _ := strings.Cut(rest, "/")

	h.mu.Lock()
	h.hits[name]++
	h.ranges[name] = r.Header.Get("Range")
	body, found := h.content[name]
	broken := h.broken[name]
	h.mu.Unlock()

	switch {
	case broken:
		http.Error(w, "upstream is having a day", http.StatusInternalServerError)
	case !found:
		http.NotFound(w, r)
	default:
		var from int
		if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-", &from); err == nil && from <= len(body) {
			w.WriteHeader(http.StatusPartialContent)
			w.Write([]byte(body[from:]))
			return
		}
		w.Write([]byte(body))
	}
}

func (h *fakeHub) set(name, body string) {
	h.mu.Lock()
	h.content[name] = body
	h.mu.Unlock()
}

func (h *fakeHub) setBroken(name string, broken bool) {
	h.mu.Lock()
	h.broken[name] = broken
	h.mu.Unlock()
}

func (h *fakeHub) fetched(name string) int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.hits[name]
}

func (h *fakeHub) resetHits() {
	h.mu.Lock()
	h.hits = map[string]int{}
	h.mu.Unlock()
}

// files lists the hub's current content as the tree API would: weights as LFS
// objects, everything else as plain blobs.
func (h *fakeHub) files() []ModelFile {
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []ModelFile
	for name, body := range h.content {
		cat, required := categorizeFile(name)
		f := ModelFile{Filename: name, Size: int64(len(body)), Category: cat, IsRequired: required}
		if cat == "weight" {
			sum := sha256.Sum256([]byte(body))
			f.SHA256 = hex.EncodeToString(sum[:])
		} else {
			hash := sha1.New()
			fmt.Fprintf(hash, "blob %d\x00%s", len(body), body)
			f.OID = hex.EncodeToString(hash.Sum(nil))
		}
		out = append(out, f)
	}
	return out
}

func newTestDownloader(t *testing.T, h *fakeHub) (*Downloader, string) {
	t.Helper()
	dir := t.TempDir()
	d := NewDownloader(dir, filepath.Join(dir, "models"), "")
	d.SetBaseURL(h.srv.URL)
	return d, d.ModelDir(testModel)
}

// transferTo runs one transfer to the end and returns how it settled.
func transferTo(t *testing.T, d *Downloader, req Request) DownloadProgress {
	t.Helper()
	req.ModelID = testModel
	id, err := d.Start(req)
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		p := d.GetProgress(id)
		if p == nil {
			t.Fatal("transfer disappeared")
		}
		if p.Status != "downloading" {
			return *p
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("transfer did not settle")
	return DownloadProgress{}
}

func mustComplete(t *testing.T, p DownloadProgress) {
	t.Helper()
	if p.Status != "complete" {
		t.Fatalf("transfer ended %s: %s", p.Status, p.Error)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func seedHub(h *fakeHub) {
	h.set("config.json", `{"v":1}`)
	h.set("model-00000.safetensors", "shard zero, first cut")
	h.set("model-00001.safetensors", "shard one, first cut")
}

const (
	rev1 = "1111111111111111111111111111111111111111"
	rev2 = "2222222222222222222222222222222222222222"
)

func TestDownloadRecordsWhatItFetched(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)

	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))

	if got := readFile(t, filepath.Join(dir, "model-00001.safetensors")); got != "shard one, first cut" {
		t.Errorf("shard content = %q", got)
	}
	m := loadManifest(dir, testModel)
	if m.Revision != rev1 {
		t.Errorf("manifest revision = %q, want %q", m.Revision, rev1)
	}
	if len(m.Files) != 3 || len(m.Parts) != 0 {
		t.Errorf("manifest has %d files and %d parts, want 3 and 0", len(m.Files), len(m.Parts))
	}

	// With the manifest in place, a second look finds nothing to do and needs
	// no file contents to say so.
	if pending := d.Plan(testModel, rev1, h.files()).Pending(); len(pending) != 0 {
		t.Errorf("after a complete download, %d files are still pending: %+v", len(pending), pending)
	}
}

// The case this exists for: a maintainer republishes some of a repo's files,
// and only those are fetched.
func TestUpdateFetchesOnlyWhatChanged(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)
	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))

	h.set("model-00001.safetensors", "shard one, second cut!")
	h.set("extras/projector.safetensors", "a new file in a new directory")
	h.resetHits()

	plan := d.Plan(testModel, rev2, h.files())
	states := map[string]FileState{}
	for _, f := range plan.Files {
		states[f.Filename] = f.State
	}
	want := map[string]FileState{
		"config.json":                  FileCurrent,
		"model-00000.safetensors":      FileCurrent,
		"model-00001.safetensors":      FileChanged,
		"extras/projector.safetensors": FileNew,
	}
	for name, state := range want {
		if states[name] != state {
			t.Errorf("%s planned as %q, want %q", name, states[name], state)
		}
	}

	p := transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true})
	mustComplete(t, p)
	if p.TotalFiles != 2 {
		t.Errorf("transfer tracked %d files, want the 2 with work to do", p.TotalFiles)
	}

	if n := h.fetched("model-00000.safetensors"); n != 0 {
		t.Errorf("unchanged shard was fetched %d times", n)
	}
	if n := h.fetched("config.json"); n != 0 {
		t.Errorf("unchanged config was fetched %d times", n)
	}
	if got := readFile(t, filepath.Join(dir, "model-00001.safetensors")); got != "shard one, second cut!" {
		t.Errorf("changed shard = %q", got)
	}
	if got := readFile(t, filepath.Join(dir, "extras", "projector.safetensors")); got != "a new file in a new directory" {
		t.Errorf("new file = %q", got)
	}
	if m := loadManifest(dir, testModel); m.Revision != rev2 {
		t.Errorf("manifest revision = %q, want %q", m.Revision, rev2)
	}
}

// A model downloaded before manifests existed has nothing on record. Its
// files are hashed once, which is a disk read and not a download.
func TestUpdateOfAnUntrackedModelHashesInsteadOfFetching(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)
	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))
	if err := os.Remove(filepath.Join(dir, manifestName)); err != nil {
		t.Fatal(err)
	}

	// Repacked in place: same name, same length, different bytes. Size alone
	// would call this one current.
	h.set("model-00000.safetensors", "shard ZERO, first cut")
	h.resetHits()

	for _, f := range d.Plan(testModel, rev2, h.files()).Files {
		if f.State != FileUnverified {
			t.Errorf("%s planned as %q before hashing, want %q", f.Filename, f.State, FileUnverified)
		}
	}

	mustComplete(t, transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true}))

	if n := h.fetched("model-00001.safetensors"); n != 0 {
		t.Errorf("a shard that matched its hash was fetched %d times", n)
	}
	if n := h.fetched("model-00000.safetensors"); n != 1 {
		t.Errorf("the repacked shard was fetched %d times, want 1", n)
	}
	if got := readFile(t, filepath.Join(dir, "model-00000.safetensors")); got != "shard ZERO, first cut" {
		t.Errorf("repacked shard = %q", got)
	}
	if pending := d.Plan(testModel, rev2, h.files()).Pending(); len(pending) != 0 {
		t.Errorf("%d files still pending after verifying", len(pending))
	}
}

// An update that dies halfway must leave the model it found. Replacements
// wait as .part files until all of them have arrived.
func TestFailedUpdateLeavesTheOldFilesInPlace(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)
	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))

	h.set("model-00000.safetensors", "shard zero, second cut")
	h.set("model-00001.safetensors", "shard one, second cut")
	h.setBroken("model-00001.safetensors", true)

	p := transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true})
	if p.Status != "failed" {
		t.Fatalf("transfer ended %s, want failed", p.Status)
	}
	if got := readFile(t, filepath.Join(dir, "model-00000.safetensors")); got != "shard zero, first cut" {
		t.Errorf("a shard was replaced by an update that did not finish: %q", got)
	}
	if m := loadManifest(dir, testModel); m.Revision != rev1 {
		t.Errorf("manifest revision moved to %q on a failed update", m.Revision)
	}
	if inc := d.ListIncomplete(); len(inc) != 1 || inc[0].ModelID != testModel {
		t.Errorf("the interrupted update is not listed as resumable: %+v", inc)
	}

	// Retrying picks up the shard that did arrive rather than fetching it again.
	h.setBroken("model-00001.safetensors", false)
	h.resetHits()
	mustComplete(t, transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true}))

	if n := h.fetched("model-00000.safetensors"); n != 0 {
		t.Errorf("the shard already staged was fetched %d more times", n)
	}
	for name, want := range map[string]string{
		"model-00000.safetensors": "shard zero, second cut",
		"model-00001.safetensors": "shard one, second cut",
	} {
		if got := readFile(t, filepath.Join(dir, name)); got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
	if inc := d.ListIncomplete(); len(inc) != 0 {
		t.Errorf(".part files left behind after the update finished: %+v", inc)
	}
}

func TestDownloadThatFailsItsChecksumIsDiscarded(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)

	files := h.files()
	// What is served is not what the listing promised.
	h.set("model-00001.safetensors", "shard one, tampered!")

	p := transferTo(t, d, Request{Revision: rev1, Files: files})
	if p.Status != "failed" || !strings.Contains(p.Error, "checksum") {
		t.Fatalf("transfer ended %s (%q), want a checksum failure", p.Status, p.Error)
	}
	for _, name := range []string{"model-00001.safetensors", "model-00001.safetensors.part"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Errorf("%s was kept after failing its checksum", name)
		}
	}
}

// A .part from before upstream moved on is a download of a different file.
// Resuming it would append new bytes to old ones.
func TestPartFromAnOlderVersionIsNotResumed(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)
	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))

	// An update to a second cut gets as far as a partial file…
	h.set("model-00001.safetensors", "shard one, second cut")
	oldID := ""
	for _, f := range h.files() {
		if f.Filename == "model-00001.safetensors" {
			oldID = f.Identity()
		}
	}
	part := filepath.Join(dir, "model-00001.safetensors.part")
	if err := os.WriteFile(part, []byte("shard one, sec"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := loadManifest(dir, testModel)
	m.Parts["model-00001.safetensors"] = ManifestPart{Identity: oldID}
	if err := m.save(dir); err != nil {
		t.Fatal(err)
	}

	// …and upstream publishes a third before it is resumed.
	h.set("model-00001.safetensors", "shard one, third cut, longer")
	mustComplete(t, transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true}))

	if got := h.ranges["model-00001.safetensors"]; got != "" {
		t.Errorf("resumed a stale partial file with Range %q", got)
	}
	if got := readFile(t, filepath.Join(dir, "model-00001.safetensors")); got != "shard one, third cut, longer" {
		t.Errorf("shard = %q", got)
	}
}

func TestInterruptedDownloadResumesFromItsPartFile(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)

	var id string
	for _, f := range h.files() {
		if f.Filename == "model-00001.safetensors" {
			id = f.Identity()
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model-00001.safetensors.part"), []byte("shard one,"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := newManifest(testModel)
	m.Parts["model-00001.safetensors"] = ManifestPart{Identity: id}
	if err := m.save(dir); err != nil {
		t.Fatal(err)
	}

	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))

	if got := h.ranges["model-00001.safetensors"]; got != "bytes=10-" {
		t.Errorf("Range = %q, want the download to continue from byte 10", got)
	}
	if got := readFile(t, filepath.Join(dir, "model-00001.safetensors")); got != "shard one, first cut" {
		t.Errorf("resumed shard = %q", got)
	}
}

func TestStaleFilesAreReportedAndRemovedOnlyOnRequest(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	h.set("extras/old-head.safetensors", "superseded")
	d, dir := newTestDownloader(t, h)
	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))

	// Something the operator put there is never upstream's to remove.
	writeFile(t, filepath.Join(dir, "notes.local"), 10)

	h.mu.Lock()
	delete(h.content, "extras/old-head.safetensors")
	h.mu.Unlock()

	plan := d.Plan(testModel, rev2, h.files())
	if len(plan.Stale) != 1 || plan.Stale[0] != "extras/old-head.safetensors" {
		t.Fatalf("stale = %v, want only the file upstream dropped", plan.Stale)
	}

	stale := filepath.Join(dir, "extras", "old-head.safetensors")
	mustComplete(t, transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true}))
	if _, err := os.Stat(stale); err != nil {
		t.Errorf("a stale file was removed without being asked: %v", err)
	}

	mustComplete(t, transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true, RemoveStale: true}))
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Errorf("the stale file is still there after asking for its removal")
	}
	if _, err := os.Stat(filepath.Join(dir, "extras")); !os.IsNotExist(err) {
		t.Errorf("the directory that only held the stale file was left behind")
	}
	if _, err := os.Stat(filepath.Join(dir, "notes.local")); err != nil {
		t.Errorf("a file this never downloaded was removed: %v", err)
	}
}

func TestDiscardPartsKeepsTheWeights(t *testing.T) {
	h := newFakeHub(t)
	seedHub(h)
	d, dir := newTestDownloader(t, h)
	mustComplete(t, transferTo(t, d, Request{Revision: rev1, Files: h.files()}))

	h.set("model-00000.safetensors", "shard zero, second cut")
	h.set("model-00001.safetensors", "shard one, second cut")
	h.setBroken("model-00001.safetensors", true)
	transferTo(t, d, Request{Revision: rev2, Files: h.files(), Update: true})

	if err := d.DiscardParts(testModel); err != nil {
		t.Fatal(err)
	}
	if inc := d.ListIncomplete(); len(inc) != 0 {
		t.Errorf(".part files survived: %+v", inc)
	}
	if got := readFile(t, filepath.Join(dir, "model-00000.safetensors")); got != "shard zero, first cut" {
		t.Errorf("discarding an update's partial files touched the weights: %q", got)
	}
	if m := loadManifest(dir, testModel); len(m.Parts) != 0 || len(m.Files) != 3 {
		t.Errorf("manifest has %d parts and %d files, want 0 and 3", len(m.Parts), len(m.Files))
	}
}

func TestStartRefusesAFileOutsideTheModelDirectory(t *testing.T) {
	h := newFakeHub(t)
	d, _ := newTestDownloader(t, h)
	_, err := d.Start(Request{ModelID: testModel, Files: []ModelFile{{Filename: "../../escape.safetensors", Size: 1}}})
	if err == nil {
		t.Fatal("a path climbing out of the model directory was accepted")
	}
}

func TestGetFilesReadsIdentitiesAtOneRevision(t *testing.T) {
	var treePath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/models/"+testModel:
			fmt.Fprintf(w, `{"id":%q,"sha":%q}`, testModel, rev1)
		case strings.HasPrefix(r.URL.Path, "/api/models/"+testModel+"/tree/"):
			treePath = r.URL.Path
			fmt.Fprint(w, `[
				{"type":"directory","oid":"d0","size":0,"path":"extras"},
				{"type":"file","oid":"b10b","size":12,"path":"config.json"},
				{"type":"file","oid":"9017e4","size":135,"path":"README.md"},
				{"type":"file","oid":"9017e5","size":4096,"lfs":{"oid":"abc123","size":4096,"pointerSize":135},"path":"model.safetensors"},
				{"type":"file","oid":"9017e6","size":4096,"lfs":{"oid":"def456","size":4096,"pointerSize":135},"path":"model.gguf"}
			]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	c := NewClient("")
	c.SetBaseURL(srv.URL)
	revision, files, err := c.GetFiles(context.Background(), testModel, "")
	if err != nil {
		t.Fatal(err)
	}
	if revision != rev1 {
		t.Errorf("revision = %q, want %q", revision, rev1)
	}
	// Listed at the commit, not at "main": main can move between the listing
	// and the download.
	if !strings.HasSuffix(treePath, "/tree/"+rev1) {
		t.Errorf("tree was read at %q, want it pinned to %s", treePath, rev1)
	}

	got := map[string]string{}
	for _, f := range files {
		got[f.Filename] = f.Identity()
	}
	want := map[string]string{
		"config.json":       "git:b10b",
		"model.safetensors": "sha256:abc123",
	}
	if len(got) != len(want) {
		t.Errorf("files = %v, want %v", got, want)
	}
	for name, id := range want {
		if got[name] != id {
			t.Errorf("%s identity = %q, want %q", name, got[name], id)
		}
	}
}

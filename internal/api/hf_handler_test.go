package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

const hfhRev = "3333333333333333333333333333333333333333"

// hfhHub is a fake Hub that answers for any repo id: the same small repo is
// behind every one of them. Answering for anything is deliberate. Whether an
// id is acceptable is the handler's call, not the Hub's, so a fake that 404s
// strange ids would hide a handler that never checks.
type hfhHub struct {
	srv *httptest.Server

	mu sync.Mutex
	// failSearch makes the search endpoint answer 500.
	failSearch bool
	// hold, when set, keeps the weight file from being served until closed.
	hold     chan struct{}
	arrivals chan string
}

var hfhFiles = map[string]string{
	"config.json":       `{"architectures": ["WidgetForCausalLM"], "max_position_embeddings": 4096}`,
	"model.safetensors": "weights weights weights",
}

func newHFHHub(t *testing.T) *hfhHub {
	t.Helper()
	h := &hfhHub{arrivals: make(chan string, 16)}
	h.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Path
		switch {
		case p == "/api/models":
			h.mu.Lock()
			fail := h.failSearch
			h.mu.Unlock()
			if fail {
				http.Error(w, "hub is down", http.StatusInternalServerError)
				return
			}
			fmt.Fprint(w, `[{"id": "acme/widget-7b", "author": "acme", "downloads": 1234, "likes": 5, "tags": ["safetensors"]}]`)
		case strings.HasPrefix(p, "/api/models/") && strings.Contains(p, "/tree/"):
			var tree []map[string]any
			for name, body := range hfhFiles {
				tree = append(tree, map[string]any{"type": "file", "path": name, "size": len(body)})
			}
			json.NewEncoder(w).Encode(tree)
		case strings.HasPrefix(p, "/api/models/"):
			id := strings.TrimPrefix(p, "/api/models/")
			fmt.Fprintf(w, `{"id": %q, "author": "acme", "sha": %q}`, id, hfhRev)
		case strings.Contains(p, "/resolve/"):
			name := p[strings.LastIndex(p, "/")+1:]
			h.mu.Lock()
			gate := h.hold
			h.mu.Unlock()
			if gate != nil && name == "model.safetensors" {
				h.arrivals <- name
				select {
				case <-gate:
				case <-r.Context().Done():
					return
				}
			}
			body, ok := hfhFiles[name]
			if !ok {
				http.NotFound(w, r)
				return
			}
			fmt.Fprint(w, body)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(h.srv.Close)
	return h
}

// hfhServer is a Server whose HF client and downloader talk to hub, and whose
// finished downloads register the way the real server's do.
func hfhServer(t *testing.T, hub *hfhHub) *Server {
	t.Helper()
	s := bkpServer(t)
	s.hfClient.SetBaseURL(hub.srv.URL)
	s.downloader.SetBaseURL(hub.srv.URL)
	s.downloader.SetOnComplete(recordTransfer(s.registry))
	// Nothing in flight may outlive the temp dir it writes into.
	t.Cleanup(func() {
		for id := range s.downloader.ActiveModelIDs() {
			s.downloader.Cancel(huggingface.DownloadID(id))
		}
		testutil.Eventually(t, 5*time.Second, func() bool {
			return len(s.downloader.ActiveModelIDs()) == 0
		}, "downloads still running at cleanup")
	})
	return s
}

func hfhDo(s *Server, method, target string, htmx bool) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, target, nil)
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	rec := httptest.NewRecorder()
	s.Router().ServeHTTP(rec, req)
	return rec
}

func TestHFSearchReturnsTheHubsResults(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))

	rec := hfhDo(s, http.MethodGet, "/api/hf/search?q=widget", false)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var results []huggingface.ModelSearchResult
	if err := json.Unmarshal(rec.Body.Bytes(), &results); err != nil {
		t.Fatal(err)
	}
	// Both library queries return the same repo; it is listed once.
	if len(results) != 1 || results[0].ID != "acme/widget-7b" {
		t.Errorf("results = %+v", results)
	}

	rec = hfhDo(s, http.MethodGet, "/api/hf/search?q=widget", true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "acme/widget-7b") {
		t.Errorf("htmx search: status %d, body %s", rec.Code, rec.Body)
	}
}

// An empty box is a prompt, not a query for everything on the Hub.
func TestHFSearchWithNoQueryDoesNotAskTheHub(t *testing.T) {
	hub := newHFHHub(t)
	hub.failSearch = true // any call to the Hub would show as an error
	s := hfhServer(t, hub)

	rec := hfhDo(s, http.MethodGet, "/api/hf/search", false)
	if rec.Code != http.StatusOK || strings.TrimSpace(rec.Body.String()) != "[]" {
		t.Errorf("plain: status %d, body %s", rec.Code, rec.Body)
	}
	rec = hfhDo(s, http.MethodGet, "/api/hf/search", true)
	if !strings.Contains(rec.Body.String(), "Enter a search query") {
		t.Errorf("htmx: %s", rec.Body)
	}
}

// A Hub failure is a bad gateway to an API caller, and a visible notice in the
// UI, where a 502 would not be swapped in at all.
func TestHFSearchReportsAHubFailure(t *testing.T) {
	hub := newHFHHub(t)
	hub.failSearch = true
	s := hfhServer(t, hub)

	if rec := hfhDo(s, http.MethodGet, "/api/hf/search?q=widget", false); rec.Code != http.StatusBadGateway {
		t.Errorf("plain: status %d, want 502", rec.Code)
	}
	rec := hfhDo(s, http.MethodGet, "/api/hf/search?q=widget", true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Search error") {
		t.Errorf("htmx: status %d, body %s", rec.Code, rec.Body)
	}
}

func TestHFModelDetailDescribesTheRepo(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))

	if rec := hfhDo(s, http.MethodGet, "/api/hf/model", false); rec.Code != http.StatusBadRequest {
		t.Errorf("no id: status %d, want 400", rec.Code)
	}

	rec := hfhDo(s, http.MethodGet, "/api/hf/model?id=acme/widget", false)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var detail huggingface.ModelDetail
	if err := json.Unmarshal(rec.Body.Bytes(), &detail); err != nil {
		t.Fatal(err)
	}
	if detail.ID != "acme/widget" || detail.Architecture != "WidgetForCausalLM" {
		t.Errorf("detail = %+v", detail)
	}
	if want := int64(len(hfhFiles["config.json"]) + len(hfhFiles["model.safetensors"])); detail.TotalSize != want {
		t.Errorf("total size = %d, want %d", detail.TotalSize, want)
	}

	rec = hfhDo(s, http.MethodGet, "/api/hf/model?id=acme/widget", true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "WidgetForCausalLM") {
		t.Errorf("htmx: status %d, body %s", rec.Code, rec.Body)
	}
}

// A download fetches the repo into the models directory and, once complete,
// registers it.
func TestHFDownloadFetchesAndRegistersTheModel(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))

	rec := hfhDo(s, http.MethodPost, "/api/hf/download?model_id=acme/widget", false)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var resp map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["download_id"] != "acme--widget" {
		t.Errorf("download_id = %q", resp["download_id"])
	}

	testutil.Eventually(t, 5*time.Second, func() bool {
		_, ok := s.registry.Get("acme/widget")
		return ok
	}, "the download never registered the model")
	dir := s.downloader.ModelDir("acme/widget")
	for name, body := range hfhFiles {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || string(got) != body {
			t.Errorf("%s on disk = %q, %v", name, got, err)
		}
	}
}

// model_id may come as a query parameter, a form field or JSON; without one
// there is nothing to fetch.
func TestHFDownloadRefusesAMissingModelID(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))

	if rec := hfhDo(s, http.MethodPost, "/api/hf/download", false); rec.Code != http.StatusBadRequest {
		t.Errorf("plain: status %d, want 400", rec.Code)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/hf/download", strings.NewReader(`{"model_id": ""}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Router().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("empty JSON id: status %d, want 400", rec.Code)
	}
	// The UI gets the reason in a 200 it will actually swap in.
	rec = hfhDo(s, http.MethodPost, "/api/hf/download", true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Missing model_id") {
		t.Errorf("htmx: status %d, body %s", rec.Code, rec.Body)
	}
}

// A repo id becomes a path under the models directory. One that is not the
// owner/name shape must not start a transfer, and above all one that climbs
// out of the models directory must not write there.
func TestHFDownloadRefusesAMalformedModelID(t *testing.T) {

	for _, id := range []string{"acme/../../escape", "../escape", "noslash", "a/b/c", "acme/.."} {
		t.Run(id, func(t *testing.T) {
			s := hfhServer(t, newHFHHub(t))
			rec := hfhDo(s, http.MethodPost, "/api/hf/download?model_id="+url.QueryEscape(id), false)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("status %d, want 400: %s", rec.Code, rec.Body)
			}
			// Give a wrongly started transfer the chance to write.
			time.Sleep(200 * time.Millisecond)
			if _, err := os.Stat(filepath.Join(s.cfg.DataDir, "escape")); err == nil {
				t.Errorf("%q wrote outside the models directory", id)
			}
			if _, ok := s.registry.Get(id); ok {
				t.Errorf("%q was registered", id)
			}
		})
	}
}

// Cancelling stops the transfer and leaves no registry entry behind for a
// model that never finished arriving.
func TestHFDownloadCancelStopsATransferInFlight(t *testing.T) {
	hub := newHFHHub(t)
	hub.hold = make(chan struct{})
	t.Cleanup(func() { close(hub.hold) })
	s := hfhServer(t, hub)

	if rec := hfhDo(s, http.MethodPost, "/api/hf/download?model_id=acme/widget", false); rec.Code != http.StatusOK {
		t.Fatalf("start: status %d: %s", rec.Code, rec.Body)
	}
	select {
	case <-hub.arrivals:
	case <-time.After(5 * time.Second):
		t.Fatal("the weight file was never requested")
	}

	rec := hfhDo(s, http.MethodDelete, "/api/hf/download/acme--widget", false)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("cancel: status %d: %s", rec.Code, rec.Body)
	}
	testutil.Eventually(t, 5*time.Second, func() bool {
		return !s.downloader.ActiveModelIDs()["acme/widget"]
	}, "the download was still running after cancel")
	if p := s.downloader.GetProgress("acme--widget"); p != nil && p.Status != "cancelled" {
		t.Errorf("status after cancel = %q", p.Status)
	}
	if _, ok := s.registry.Get("acme/widget"); ok {
		t.Error("a cancelled download left a registry entry")
	}
}

func TestHFDownloadCancelOfAnUnknownDownloadIsNotFound(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))
	if rec := hfhDo(s, http.MethodDelete, "/api/hf/download/acme--nothing", false); rec.Code != http.StatusNotFound {
		t.Errorf("status %d, want 404", rec.Code)
	}
}

// hfhWriteFiles writes name->body files under dir.
func hfhWriteFiles(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// A stalled download nobody will resume is disk space to reclaim, and the
// whole directory goes: nothing in it is a model yet.
func TestDiscardingAnIncompleteDownloadRemovesItsDirectory(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))
	dir := s.downloader.ModelDir("acme/stalled")
	hfhWriteFiles(t, dir, map[string]string{"config.json": "{}", "model.safetensors.part": "half"})

	rec := hfhDo(s, http.MethodDelete, "/api/hf/incomplete?model_id=acme/stalled", false)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("directory still there: %v", err)
	}
}

// For a registered model the partial files are an interrupted update, and the
// weights beside them are the ones being served. Only the partials go.
func TestDiscardingAnIncompleteUpdateKeepsTheRegisteredWeights(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))
	dir := s.downloader.ModelDir("acme/widget")
	hfhWriteFiles(t, dir, map[string]string{"model.safetensors": "good weights", "model.safetensors.part": "half"})
	if err := s.registry.Register(&models.Model{ID: "acme/widget", LocalPath: dir}); err != nil {
		t.Fatal(err)
	}

	rec := hfhDo(s, http.MethodDelete, "/api/hf/incomplete?model_id=acme/widget", false)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors.part")); !os.IsNotExist(err) {
		t.Errorf("partial file still there: %v", err)
	}
	if got, err := os.ReadFile(filepath.Join(dir, "model.safetensors")); err != nil || string(got) != "good weights" {
		t.Errorf("registered weights = %q, %v", got, err)
	}
}

// The id is joined onto the models directory and the result deleted. An empty
// or traversing id would resolve to the models root, or above it.
func TestDiscardingIncompleteRefusesAnIDThatIsNotAModel(t *testing.T) {
	s := hfhServer(t, newHFHHub(t))
	survivor := s.downloader.ModelDir("acme/survivor")
	hfhWriteFiles(t, survivor, map[string]string{"model.safetensors.part": "half"})

	if rec := hfhDo(s, http.MethodDelete, "/api/hf/incomplete", false); rec.Code != http.StatusBadRequest {
		t.Errorf("no id: status %d, want 400", rec.Code)
	}
	for _, id := range []string{"..", "acme/..", "../..", "acme", "acme/survivor/.."} {
		rec := hfhDo(s, http.MethodDelete, "/api/hf/incomplete?model_id="+url.QueryEscape(id), false)
		if rec.Code < 400 {
			t.Errorf("%q: status %d, want a refusal", id, rec.Code)
		}
	}
	if _, err := os.Stat(filepath.Join(survivor, "model.safetensors.part")); err != nil {
		t.Errorf("another model's files were removed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(s.cfg.DataDir, "config", "vllmctl.yaml")); err != nil {
		t.Errorf("the data directory was touched: %v", err)
	}
}

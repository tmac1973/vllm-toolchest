package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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

const (
	updModel = "acme/widget"
	updRev   = "2222222222222222222222222222222222222222"
)

// updateHub is a Hub with one repo in it: enough of the API for a listing,
// and the files themselves.
type updateHub struct {
	srv *httptest.Server

	mu      sync.Mutex
	content map[string]string
	hits    map[string]int
	// hold, when set, is closed to let a request for that file proceed.
	hold     map[string]chan struct{}
	arrivals chan string
}

func newUpdateHub(t *testing.T, content map[string]string) *updateHub {
	h := &updateHub{
		content:  content,
		hits:     map[string]int{},
		hold:     map[string]chan struct{}{},
		arrivals: make(chan string, 16),
	}
	h.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		switch {
		case r.URL.Path == "/api/models/"+updModel:
			fmt.Fprintf(w, `{"id":%q,"sha":%q}`, updModel, updRev)
		case strings.HasPrefix(r.URL.Path, "/api/models/"+updModel+"/tree/"):
			type lfs struct {
				OID string `json:"oid"`
			}
			type entry struct {
				Type string `json:"type"`
				Path string `json:"path"`
				Size int    `json:"size"`
				LFS  *lfs   `json:"lfs"`
			}
			var tree []entry
			for name, body := range h.content {
				sum := sha256.Sum256([]byte(body))
				tree = append(tree, entry{"file", name, len(body), &lfs{hex.EncodeToString(sum[:])}})
			}
			json.NewEncoder(w).Encode(tree)
		case strings.HasPrefix(r.URL.Path, "/"+updModel+"/resolve/"+updRev+"/"):
			name := strings.TrimPrefix(r.URL.Path, "/"+updModel+"/resolve/"+updRev+"/")
			h.hits[name]++
			if gate := h.hold[name]; gate != nil {
				h.arrivals <- name
				h.mu.Unlock()
				select {
				case <-gate:
				case <-r.Context().Done():
				}
				h.mu.Lock()
			}
			fmt.Fprint(w, h.content[name])
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(h.srv.Close)
	return h
}

// newUpdateServer is a Server holding one registered model whose files are on
// disk as first cuts, in front of a Hub that has since republished one shard.
func newUpdateServer(t *testing.T) (*Server, *updateHub, string) {
	t.Helper()
	s := newGoldenServer(t, goldenEnvGeneric)
	modelsDir := filepath.Join(s.cfg.DataDir, "models")

	hub := newUpdateHub(t, map[string]string{
		"config.json":             `{"v":1}`,
		"model-00000.safetensors": "shard zero, first cut",
		"model-00001.safetensors": "shard one, second cut!",
	})
	s.hfClient = huggingface.NewClient("")
	s.hfClient.SetBaseURL(hub.srv.URL)
	s.downloader = huggingface.NewDownloader(s.cfg.DataDir, modelsDir, "")
	s.downloader.SetBaseURL(hub.srv.URL)
	s.downloader.SetOnComplete(recordTransfer(s.registry))
	s.router = s.buildRouter()

	dir := s.downloader.ModelDir(updModel)
	for name, body := range map[string]string{
		"config.json":             `{"v":1}`,
		"model-00000.safetensors": "shard zero, first cut",
		"model-00001.safetensors": "shard one, first cut",
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.registry.Register(&models.Model{
		ID: updModel, DisplayName: "widget", LocalPath: dir, Enabled: true,
		// Stands for everything an operator sets by hand and an update has
		// no business resetting.
		VLLMConfig: models.VLLMConfig{TensorParallelSize: 4, ExtraFlags: "--hand-tuned"},
	}); err != nil {
		t.Fatal(err)
	}
	return s, hub, dir
}

func htmxRequest(s *Server, method, target string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, target, nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.Router().ServeHTTP(rec, req)
	return rec
}

func TestUpdateCheckReportsWhatChangedWithoutFetchingIt(t *testing.T) {
	s, hub, _ := newUpdateServer(t)

	body := htmxRequest(s, http.MethodGet, "/api/models/update-check?id="+updModel).Body.String()

	// The republished shard is a different length, so it is known to have
	// changed; the other two are the right length and have never been hashed.
	for _, want := range []string{
		"<strong>1</strong> file to download",
		"<strong>2</strong> to verify",
		"model-00001.safetensors",
		"revision=" + updRev,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("update panel is missing %q:\n%s", want, body)
		}
	}
	if len(hub.hits) != 0 {
		t.Errorf("checking for updates downloaded files: %v", hub.hits)
	}
}

func TestUpdateFetchesTheChangedShardAndKeepsTheConfig(t *testing.T) {
	s, hub, dir := newUpdateServer(t)

	rec := htmxRequest(s, http.MethodPost, "/api/models/update?id="+updModel+"&revision="+updRev)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Update started") {
		t.Fatalf("update did not start: %d %s", rec.Code, rec.Body.String())
	}
	var p *huggingface.DownloadProgress
	testutil.Eventually(t, 10*time.Second, func() bool {
		p = s.downloader.GetProgress(huggingface.DownloadID(updModel))
		return p != nil && p.Status != "downloading"
	}, "transfer did not settle")
	if p.Status != "complete" {
		t.Fatalf("transfer ended %s: %s", p.Status, p.Error)
	}

	got, err := os.ReadFile(filepath.Join(dir, "model-00001.safetensors"))
	if err != nil || string(got) != "shard one, second cut!" {
		t.Errorf("changed shard = %q (%v)", got, err)
	}
	if hub.hits["model-00000.safetensors"] != 0 || hub.hits["config.json"] != 0 {
		t.Errorf("files that had not changed were downloaded: %v", hub.hits)
	}

	m, ok := s.registry.Get(updModel)
	if !ok {
		t.Fatal("the model is no longer registered")
	}
	if m.VLLMConfig.TensorParallelSize != 4 || m.VLLMConfig.ExtraFlags != "--hand-tuned" {
		t.Errorf("the update reset the launch config: %+v", m.VLLMConfig)
	}
	if want := int64(len(`{"v":1}` + "shard zero, first cut" + "shard one, second cut!")); m.TotalSizeBytes < want {
		t.Errorf("size was not re-read after the update: %d, want at least %d", m.TotalSizeBytes, want)
	}

	// Everything is on record now, so the next check has nothing to hash.
	body := htmxRequest(s, http.MethodGet, "/api/models/update-check?id="+updModel).Body.String()
	if !strings.Contains(body, "Up to date.") {
		t.Errorf("a freshly updated model is not reported as up to date:\n%s", body)
	}
}

// Pausing a first download drops the half-made registry entry a disk scan may
// have created for it. An update's entry is a whole model, and pausing a fetch
// of one shard must not unregister it.
func TestPausingAnUpdateDoesNotUnregisterTheModel(t *testing.T) {
	s, hub, dir := newUpdateServer(t)
	gate := make(chan struct{})
	hub.mu.Lock()
	hub.hold["model-00001.safetensors"] = gate
	hub.mu.Unlock()
	defer close(gate)

	htmxRequest(s, http.MethodPost, "/api/models/update?id="+updModel+"&revision="+updRev)
	select {
	case <-hub.arrivals:
	case <-time.After(10 * time.Second):
		t.Fatal("the update never asked for the changed shard")
	}

	rec := htmxRequest(s, http.MethodDelete, "/api/hf/download/"+huggingface.DownloadID(updModel))
	if rec.Code != http.StatusOK {
		t.Fatalf("pause returned %d", rec.Code)
	}
	if _, ok := s.registry.Get(updModel); !ok {
		t.Error("pausing an update removed the model from the registry")
	}
	got, _ := os.ReadFile(filepath.Join(dir, "model-00001.safetensors"))
	if string(got) != "shard one, first cut" {
		t.Errorf("a paused update changed the shard in place: %q", got)
	}
}

func TestUpdateRefusesAModelOutsideTheModelsDirectory(t *testing.T) {
	s, _, _ := newUpdateServer(t)
	m, _ := s.registry.Get(updModel)
	m.LocalPath = "/srv/elsewhere/widget"

	rec := htmxRequest(s, http.MethodPost, "/api/models/update?id="+updModel+"&revision="+updRev)
	if !strings.Contains(rec.Body.String(), "outside the models directory") {
		t.Errorf("expected a refusal, got: %s", rec.Body.String())
	}
	if len(s.downloader.ActiveDownloads()) != 0 {
		t.Error("a transfer was started anyway")
	}
}

func TestUpdateRejectsARevisionThatIsNotACommit(t *testing.T) {
	s, _, _ := newUpdateServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/models/update?id="+updModel+"&revision=main/../../x", nil)
	rec := httptest.NewRecorder()
	s.Router().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

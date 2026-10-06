package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/backup"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// bkpServer is a Server with everything the backup routes touch: a config
// loaded from disk (a hand-built one has no path and every save fails), a
// registry, the benchmark service the busy check asks, and the HF clients a
// restored token is handed to. Requests go through the real router.
func bkpServer(t *testing.T) *Server {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "config", "vllmctl.yaml")
	if err := os.WriteFile(path, []byte("data_dir: "+dir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	// Load reads the environment too; a token on the machine running the
	// tests must not leak into what they assert.
	cfg.HFToken, cfg.APIKey = "", ""

	modelsDir := filepath.Join(dir, "models")
	s := &Server{
		cfg:        cfg,
		registry:   models.NewRegistry(dir, modelsDir),
		monitor:    monitor.New(0),
		process:    process.NewManager("127.0.0.1", 8000, 0),
		hfClient:   huggingface.NewClient(""),
		downloader: huggingface.NewDownloader(dir, modelsDir, ""),
	}
	s.bench = benchmark.NewStore(dir)
	s.benchSvc = benchmark.NewService(s.bench)
	s.initTemplates()
	s.router = s.buildRouter()
	return s
}

// bkpExport fetches the backup as a plain caller would.
func bkpExport(t *testing.T, s *Server, query string) []byte {
	t.Helper()
	rec := httptest.NewRecorder()
	s.Router().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/backup"+query, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("export: status %d: %s", rec.Code, rec.Body)
	}
	if cd := rec.Header().Get("Content-Disposition"); !strings.Contains(cd, "attachment") {
		t.Errorf("export should download as a file, Content-Disposition = %q", cd)
	}
	return rec.Body.Bytes()
}

// bkpRestore posts a multipart restore. file nil leaves the file part out;
// sections are the sec_* checkboxes that are ticked.
func bkpRestore(t *testing.T, s *Server, file []byte, htmx bool, sections ...string) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for _, sec := range sections {
		if err := mw.WriteField(sec, "on"); err != nil {
			t.Fatal(err)
		}
	}
	if file != nil {
		fw, err := mw.CreateFormFile("file", "backup.json")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fw.Write(file); err != nil {
			t.Fatal(err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/restore", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	rec := httptest.NewRecorder()
	s.Router().ServeHTTP(rec, req)
	return rec
}

var bkpAllSections = []string{"sec_settings", "sec_env", "sec_knobs", "sec_models"}

// bkpJSONError decodes the {"error": ...} body a plain caller gets.
func bkpJSONError(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("Content-Type = %q, want JSON", ct)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("body is not JSON: %v: %s", err, rec.Body)
	}
	if body.Error == "" {
		t.Errorf("no error in body: %s", rec.Body)
	}
	return body.Error
}

// A backup is a file people mail around and leave in Downloads. The secrets
// have to be an explicit choice, or every export leaks the token.
func TestABackupCarriesSecretsOnlyWhenAskedTo(t *testing.T) {
	s := bkpServer(t)
	s.cfg.HFToken = "hf_secret_token"
	s.cfg.APIKey = "sk-secret-key"

	plain := bkpExport(t, s, "")
	for _, secret := range []string{"hf_secret_token", "sk-secret-key", "hf_token", "api_key"} {
		if bytes.Contains(plain, []byte(secret)) {
			t.Errorf("an export without ?secrets=1 contains %q", secret)
		}
	}

	f, err := backup.Parse(bkpExport(t, s, "?secrets=1"))
	if err != nil {
		t.Fatal(err)
	}
	if f.Settings == nil || f.Settings.HFToken == nil || *f.Settings.HFToken != "hf_secret_token" {
		t.Errorf("?secrets=1 should carry the HF token, got settings %+v", f.Settings)
	}
	if f.Settings == nil || f.Settings.APIKey == nil || *f.Settings.APIKey != "sk-secret-key" {
		t.Errorf("?secrets=1 should carry the API key, got settings %+v", f.Settings)
	}
}

// The whole point of the feature: what one server exports, another restores
// to the same state, persisted rather than only in memory.
func TestARestoreRoundTripsAnExport(t *testing.T) {
	src := bkpServer(t)
	src.cfg.MaxNumSeqs = 7
	src.cfg.LogLevel = "debug"
	src.cfg.HFToken = "hf_round_trip"
	src.cfg.APIKey = "sk-round-trip"
	src.cfg.RuntimeEnvExtra = "BKP_ROUND_TRIP=1"
	installedCfg := models.VLLMConfig{MaxModelLen: 8192, MaxNumSeqs: 4, Dtype: "bfloat16"}
	missingCfg := models.VLLMConfig{MaxModelLen: 4096}
	for id, c := range map[string]models.VLLMConfig{"acme/installed": installedCfg, "acme/elsewhere": missingCfg} {
		if err := src.registry.Register(&models.Model{ID: id, VLLMConfig: c}); err != nil {
			t.Fatal(err)
		}
	}
	file := bkpExport(t, src, "?secrets=1")

	dst := bkpServer(t)
	// Installed here too, with a config the restore should replace; the other
	// model is not, and its config should wait as pending.
	if err := dst.registry.Register(&models.Model{ID: "acme/installed", VLLMConfig: models.VLLMConfig{MaxModelLen: 2048}}); err != nil {
		t.Fatal(err)
	}

	rec := bkpRestore(t, dst, file, false, bkpAllSections...)
	if rec.Code != http.StatusOK {
		t.Fatalf("restore: status %d: %s", rec.Code, rec.Body)
	}
	var report backup.Report
	if err := json.Unmarshal(rec.Body.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Error != "" || len(report.Skipped) > 0 {
		t.Fatalf("restore reported problems: %+v", report)
	}
	if report.AppliedModelConfigs != 1 {
		t.Errorf("applied model configs = %d, want 1", report.AppliedModelConfigs)
	}

	// Read back from disk: a restore that only changed memory is lost at the
	// next restart.
	saved, err := config.Load(filepath.Join(dst.cfg.DataDir, "config", "vllmctl.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if saved.MaxNumSeqs != 7 || saved.LogLevel != "debug" {
		t.Errorf("settings not persisted: max_num_seqs=%d log_level=%q", saved.MaxNumSeqs, saved.LogLevel)
	}
	if saved.HFToken != "hf_round_trip" || saved.APIKey != "sk-round-trip" {
		t.Errorf("secrets not persisted: hf_token=%q api_key=%q", saved.HFToken, saved.APIKey)
	}
	if saved.RuntimeEnvExtra != "BKP_ROUND_TRIP=1" {
		t.Errorf("runtime env not persisted: %q", saved.RuntimeEnvExtra)
	}

	reloaded := models.NewRegistry(dst.cfg.DataDir, filepath.Join(dst.cfg.DataDir, "models"))
	if m, ok := reloaded.Get("acme/installed"); !ok || m.VLLMConfig != installedCfg {
		t.Errorf("installed model's config = %+v, want %+v", m, installedCfg)
	}
	pending := reloaded.PendingConfigs()
	if len(pending) != 1 || pending[0].ModelID != "acme/elsewhere" || pending[0].Config != missingCfg {
		t.Errorf("pending configs = %+v, want acme/elsewhere waiting", pending)
	}
}

// Parse refuses anything that is not a backup before a single field is
// applied. Two shapes: not JSON at all, and JSON from a version this build
// cannot read.
func TestARestoreRefusesAFileThatIsNotABackup(t *testing.T) {
	for name, file := range map[string]string{
		"not json":      "this is not a backup",
		"wrong version": `{"version": 1, "settings": {"max_num_seqs": 99}}`,
	} {
		t.Run(name, func(t *testing.T) {
			s := bkpServer(t)
			before := s.cfg.MaxNumSeqs
			rec := bkpRestore(t, s, []byte(file), false, bkpAllSections...)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body)
			}
			bkpJSONError(t, rec)
			if s.cfg.MaxNumSeqs != before {
				t.Errorf("a refused file changed max_num_seqs to %d", s.cfg.MaxNumSeqs)
			}
		})
	}
}

// Without a file there is nothing to restore, and saying so beats a report
// that reads as "applied nothing".
func TestARestoreRefusesAnUploadWithNoFile(t *testing.T) {
	s := bkpServer(t)
	rec := bkpRestore(t, s, nil, false, bkpAllSections...)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body)
	}
	if msg := bkpJSONError(t, rec); !strings.Contains(msg, "no backup file") {
		t.Errorf("error = %q", msg)
	}
}

// An unticked form is a mistake, not a request to do nothing.
func TestARestoreRefusesAnEmptySelection(t *testing.T) {
	s := bkpServer(t)
	rec := bkpRestore(t, s, bkpExport(t, s, ""), false)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body)
	}
	if msg := bkpJSONError(t, rec); !strings.Contains(msg, "select at least one section") {
		t.Errorf("error = %q", msg)
	}
}

// restoreFileLimit is what the handler reads of the upload. A file larger
// than that is not a backup, whatever its first ten megabytes look like.
func TestARestoreRefusesAnOversizedFile(t *testing.T) {
	s := bkpServer(t)
	before := s.cfg.MaxNumSeqs

	// Garbage past the limit: refused for its size.
	junk := bytes.Repeat([]byte("x"), restoreFileLimit+1)
	rec := bkpRestore(t, s, junk, false, bkpAllSections...)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized junk: status %d, want 413", rec.Code)
	}
	bkpJSONError(t, rec)

	// A real backup padded past the limit must be refused, not cut at the
	// limit and its valid prefix applied.
	t.Run("valid prefix", func(t *testing.T) {
		f := []byte(`{"version": 2, "settings": {"max_num_seqs": 99}}`)
		padded := append(f, bytes.Repeat([]byte(" "), restoreFileLimit)...)
		rec := bkpRestore(t, s, padded, false, bkpAllSections...)
		if rec.Code != http.StatusRequestEntityTooLarge {
			t.Errorf("status %d, want 413", rec.Code)
		}
		if s.cfg.MaxNumSeqs != before {
			t.Errorf("an oversized file was applied: max_num_seqs=%d", s.cfg.MaxNumSeqs)
		}
	})
}

// htmx does not swap a non-2xx response, so a refusal sent as 400 to the UI
// would vanish: the button clicks and nothing appears. The UI gets 200 and the
// report partial with the reason in it.
func TestARestoreRefusalReachesAnHtmxCallerAsA200Report(t *testing.T) {
	s := bkpServer(t)
	rec := bkpRestore(t, s, []byte("not json"), true, bkpAllSections...)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200 for htmx", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/html") {
		t.Errorf("Content-Type = %q, want HTML", ct)
	}
	if !strings.Contains(rec.Body.String(), "not a valid backup file") {
		t.Errorf("the report should carry the refusal: %s", rec.Body)
	}

	// And a success renders the itemized report rather than JSON.
	rec = bkpRestore(t, s, bkpExport(t, s, ""), true, "sec_settings")
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Applied") {
		t.Errorf("htmx success: status %d, body %s", rec.Code, rec.Body)
	}
}

// A restore mid-benchmark rewrites the configs a running cell reports as
// fixed, which quietly invalidates its numbers.
func TestARestoreRefusesWhileABenchmarkRunIsActive(t *testing.T) {
	s := bkpServer(t)

	// An engine that never answers keeps the run active until cancelled.
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	t.Cleanup(engine.Close)

	run := benchmark.BenchmarkRun{ID: "bkp-run", Status: benchmark.StatusRunning, CreatedAt: time.Now()}
	if err := s.bench.Save(run); err != nil {
		t.Fatal(err)
	}
	if err := s.benchSvc.StartRun(benchmark.RunnerConfig{
		Run: run,
		Preset: benchmark.Preset{Source: benchmark.PresetSourceInternal,
			PromptTokens: []int{32}, GenTokens: 8, Repetitions: 1},
		VLLMURL: engine.URL, ServedName: "m", MaxModelLen: 4096,
	}); err != nil {
		t.Fatal(err)
	}
	// The run writes into the temp dir until it has wound down; removing the
	// dir under it fails the cleanup.
	t.Cleanup(func() {
		s.benchSvc.CancelRun("bkp-run")
		testutil.Eventually(t, 5*time.Second, func() bool {
			_, active := s.benchSvc.ActiveRunID()
			return !active
		}, "the benchmark run never wound down")
		time.Sleep(50 * time.Millisecond)
	})

	before := s.cfg.MaxNumSeqs
	file := []byte(fmt.Sprintf(`{"version": %d, "settings": {"max_num_seqs": %d}}`, backup.Version, before+5))
	rec := bkpRestore(t, s, file, false, bkpAllSections...)
	if rec.Code != http.StatusConflict {
		t.Fatalf("status %d, want 409: %s", rec.Code, rec.Body)
	}
	if msg := bkpJSONError(t, rec); !strings.Contains(msg, "benchmark run") {
		t.Errorf("error = %q", msg)
	}
	if s.cfg.MaxNumSeqs != before {
		t.Errorf("a refused restore changed max_num_seqs to %d", s.cfg.MaxNumSeqs)
	}
}

// Discarding a pending config removes it for good, and tells the models page
// to refresh.
func TestDiscardingAPendingConfigRemovesIt(t *testing.T) {
	s := bkpServer(t)
	if err := s.registry.SetPendingConfig(models.PendingConfig{ModelID: "acme/waiting", SavedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}

	rec := bkpDiscard(s, "acme/waiting")
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status %d, want 204: %s", rec.Code, rec.Body)
	}
	if rec.Header().Get("HX-Trigger") != "modelsChanged" {
		t.Errorf("HX-Trigger = %q", rec.Header().Get("HX-Trigger"))
	}
	reloaded := models.NewRegistry(s.cfg.DataDir, filepath.Join(s.cfg.DataDir, "models"))
	if p := reloaded.PendingConfigs(); len(p) != 0 {
		t.Errorf("pending configs after discard: %+v", p)
	}

	// Again: there is nothing left to discard.
	if rec := bkpDiscard(s, "acme/waiting"); rec.Code != http.StatusNotFound {
		t.Errorf("second discard: status %d, want 404", rec.Code)
	}
}

// A registry that will not write must say so, not report the entry missing.
func TestDiscardingAPendingConfigOnAReadOnlyRegistryIsAConflict(t *testing.T) {
	s := bkpServer(t)
	bkpWriteNewerRegistry(t, s.cfg.DataDir, "{}", `"pending_configs": [{"model_id": "acme/waiting"}]`)
	s.registry = models.NewRegistry(s.cfg.DataDir, filepath.Join(s.cfg.DataDir, "models"))
	if s.registry.ReadOnly() == "" {
		t.Fatal("setup: registry should be read-only")
	}

	rec := bkpDiscard(s, "acme/waiting")
	if rec.Code != http.StatusConflict {
		t.Fatalf("status %d, want 409: %s", rec.Code, rec.Body)
	}
	if len(s.registry.PendingConfigs()) != 1 {
		t.Error("a read-only registry discarded a pending config")
	}
}

func bkpDiscard(s *Server, modelID string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/api/pending-configs/discard",
		strings.NewReader("model_id="+modelID))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	s.Router().ServeHTTP(rec, req)
	return rec
}

// bkpWriteNewerRegistry writes a models.json from a schema version this build
// does not know, which makes the registry loaded from it read-only. modelsJSON
// is the "models" object, and extra further top-level JSON members, or "".
func bkpWriteNewerRegistry(t *testing.T, dataDir, modelsJSON, extra string) {
	t.Helper()
	body := `{"schema_version": 999, "models": ` + modelsJSON
	if extra != "" {
		body += ", " + extra
	}
	body += "}"
	path := filepath.Join(dataDir, "config", "models.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

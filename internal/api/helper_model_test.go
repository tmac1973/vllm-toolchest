package api

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

func TestTheHelperIsClaimedOnlyWhenAskedFor(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	dir := t.TempDir()

	// Downloaded from the search page: an ordinary model.
	s.onTransferComplete("x", models.HelperRepo, dir)
	if m, ok := s.registry.Get(models.HelperRepo); !ok || m.Helper {
		t.Fatalf("an unrequested download: registered=%v helper=%v", ok, ok && m.Helper)
	}
	if s.helperModel() != nil {
		t.Error("an ordinary model was taken for the helper")
	}

	// Asked for from Settings: claimed.
	s.wantHelper.Store(true)
	s.onTransferComplete("x", models.HelperRepo, dir)
	if h := s.helperModel(); h == nil || h.ID != models.HelperRepo {
		t.Fatal("the requested download was not claimed as the helper")
	}
	if s.wantHelper.Load() {
		t.Error("the request was not cleared")
	}
}

func TestDownloadHelperClaimsAnInstalledCopy(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.registry.Register(&models.Model{ID: models.HelperRepo, LocalPath: t.TempDir(), TotalSizeBytes: 8_044_936_192})

	rec := httptest.NewRecorder()
	s.handleDownloadHelper(rec, httptest.NewRequest("POST", "/api/settings/helper/download", nil))
	if s.helperModel() == nil {
		t.Fatal("the installed copy was not claimed")
	}
	if !strings.Contains(rec.Body.String(), "now the helper model") {
		t.Errorf("panel: %s", rec.Body.String())
	}
}

func TestTheHelperIsNotAModelToServe(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.registry.Register(&models.Model{ID: models.HelperRepo, LocalPath: t.TempDir(), Helper: true})

	for _, m := range s.servable() {
		if m.ID == models.HelperRepo {
			t.Error("the helper is servable")
		}
	}
	if slices.ContainsFunc(s.modelRows(), func(r modelRow) bool { return r.ID == models.HelperRepo }) {
		t.Error("the helper has a card on the Models page")
	}
	h, _ := s.registry.Get(models.HelperRepo)
	if err := s.launchBlocker(h); err == nil {
		t.Error("the helper can be started as a model")
	}
	rec := httptest.NewRecorder()
	s.handleActivateModel(rec, httptest.NewRequest("PUT", "/api/models/activate?id="+models.HelperRepo, nil))
	if rec.Code != http.StatusConflict {
		t.Errorf("activating the helper answered %d", rec.Code)
	}
}

func TestRemoveHelper(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.registry.Register(&models.Model{ID: models.HelperRepo, LocalPath: t.TempDir(), Helper: true, TotalSizeBytes: 1 << 30})
	rec := httptest.NewRecorder()
	s.handleRemoveHelper(rec, httptest.NewRequest("DELETE", "/api/settings/helper", nil))
	if _, ok := s.registry.Get(models.HelperRepo); ok {
		t.Error("the helper is still registered")
	}
	if !strings.Contains(rec.Body.String(), "freeing 1.0 GB") {
		t.Errorf("panel: %s", rec.Body.String())
	}
}

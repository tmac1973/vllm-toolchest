package api

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// The feed never takes the page down: without a Hub it is a 200 that says
// why it is empty, with the machine it would have judged against.
func TestRecommendEndpointWithoutTheHub(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.hfClient = nil
	s.gpuInvOverride = &models.GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}
	s.router = s.buildRouter()

	rec := httptest.NewRecorder()
	s.router.ServeHTTP(rec, httptest.NewRequest("GET", "/api/recommend?intent=nonsense", nil))
	var out struct {
		Profile struct {
			GPUCount    int  `json:"gpu_count"`
			InventoryOK bool `json:"inventory_known"`
		} `json:"profile"`
		Intent      string `json:"intent"`
		Unavailable string `json:"unavailable"`
		Verified    []any  `json:"verified"`
	}
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &out) != nil {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
	if out.Intent != "quality" || out.Unavailable == "" || out.Verified == nil || len(out.Verified) != 0 ||
		out.Profile.GPUCount != 4 || !out.Profile.InventoryOK {
		t.Errorf("%s", rec.Body.String())
	}

	rec = httptest.NewRecorder()
	s.router.ServeHTTP(rec, httptest.NewRequest("POST", "/api/recommend/refresh", nil))
	if rec.Code != 200 {
		t.Errorf("refresh: %d", rec.Code)
	}
}

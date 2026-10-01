package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// refineServer has the measured 27B registered with an Autoconfig profile
// applied, built for class, at the live context given.
func refineServer(t *testing.T, class models.ContextClass, context int, util float64) (*Server, *models.Model) {
	t.Helper()
	s := newGoldenServer(t, goldenEnvGeneric)
	s.gpuInvOverride = &models.GPUInventory{Count: 4, PerCardGB: 31.86, Known: true}
	s.router = s.buildRouter()
	m := measuredAPIModel()
	m.VLLMConfig.MaxModelLen = context
	m.VLLMConfig.GPUMemoryUtilization = util
	s.registry.Register(m)
	s.registry.SaveProfileFrom(m.ID, models.AutoconfigProfileName, models.ConfigProfile{
		Config: m.VLLMConfig, Source: models.ProfileSourceAutoconfig,
		Autoconfig: &models.AutoconfigRecord{Class: class, Width: "all", FirstGuess: true},
	})
	s.registry.ApplyProfile(m.ID, models.AutoconfigProfileName)
	m, _ = s.registry.Get(m.ID)
	return s, m
}

func TestRefineToMoreContext(t *testing.T) {
	s, m := refineServer(t, models.ContextMax, 131072, 0.92)
	ref := s.refinementFor(m)
	if ref == nil || !ref.More || ref.proposed != 262144 {
		t.Fatalf("refinement = %+v", ref)
	}

	panel := httptest.NewRecorder()
	s.router.ServeHTTP(panel, httptest.NewRequest("GET", "/api/models/config-panel?id="+url.QueryEscape(m.ID), nil))
	if !strings.Contains(panel.Body.String(), "Review refined settings") {
		t.Error("the config panel does not offer the refinement")
	}
	review := httptest.NewRecorder()
	s.router.ServeHTTP(review, httptest.NewRequest("GET", "/api/models/autoconfig/refine?id="+url.QueryEscape(m.ID), nil))
	if !strings.Contains(review.Body.String(), "262,144") || !strings.Contains(review.Body.String(), "autoconfig-table") {
		t.Errorf("review:\n%s", review.Body.String())
	}

	rec := post(s, "/api/models/autoconfig/refine?id="+url.QueryEscape(m.ID), url.Values{})
	if !strings.Contains(rec.Body.String(), "Applied.") {
		t.Fatalf("response: %s", rec.Body.String())
	}
	m, _ = s.registry.Get(m.ID)
	if m.VLLMConfig.MaxModelLen != 262144 {
		t.Errorf("context = %d", m.VLLMConfig.MaxModelLen)
	}
	if _, ok := models.MeasuredEstimate(m, s.engineIdentity()); !ok {
		t.Error("changing the context retired the measurement")
	}
	p, _ := s.registry.Profile(m.ID, models.AutoconfigProfileName)
	if p.Autoconfig.FirstGuess || p.Config.MaxModelLen != 262144 || len(p.Notes) == 0 {
		t.Errorf("profile not updated: %+v record %+v", p, p.Autoconfig)
	}
	if s.refinementFor(m) != nil {
		t.Error("the notice outlived the change")
	}
	if name, modified := s.registry.ActiveProfile(m.ID); name != models.AutoconfigProfileName || modified {
		t.Errorf("active = %q modified = %v", name, modified)
	}
}

func TestRefineToLessContext(t *testing.T) {
	// At 0.4 of the card, the measured pool holds less than the full context:
	// 12.74 GiB a card less the 11.67 consumed leaves about 96,000 tokens.
	// (At 0.5 it once did too, while the graphs and working set were charged
	// against the pool; the engine charges only consumed memory, and at 0.5
	// the full context fits.)
	s, m := refineServer(t, models.ContextMax, 262144, 0.4)
	ref := s.refinementFor(m)
	if ref == nil || ref.More || ref.proposed >= 262144 {
		t.Fatalf("refinement = %+v", ref)
	}
}

func TestNoRefinement(t *testing.T) {
	// Medium asked for, and configured: the measurement allows far more,
	// but more than was asked for is not offered.
	s, m := refineServer(t, models.ContextMedium, 32768, 0.92)
	if ref := s.refinementFor(m); ref != nil {
		t.Errorf("a Medium model was offered %+v", ref)
	}

	// Not autoconfigured.
	s, m = refineServer(t, models.ContextMax, 131072, 0.92)
	s.registry.SaveProfile(m.ID, "mine", models.ProfileMeta{})
	if s.refinementFor(m) != nil {
		t.Error("a model whose active profile is not Autoconfig was offered a refinement")
	}

	// The measurement no longer applies: the width changed.
	s, m = refineServer(t, models.ContextMax, 131072, 0.92)
	m.VLLMConfig.TensorParallelSize = 2
	if s.refinementFor(m) != nil {
		t.Error("a refinement was offered on a measurement that does not apply")
	}

	// A stale page posting when nothing is on offer.
	s, m = refineServer(t, models.ContextMedium, 32768, 0.92)
	rec := post(s, "/api/models/autoconfig/refine?id="+url.QueryEscape(m.ID), url.Values{})
	if !strings.Contains(rec.Body.String(), "Nothing to apply") {
		t.Errorf("response: %s", rec.Body.String())
	}
}

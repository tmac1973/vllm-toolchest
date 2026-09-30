package api

import (
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// A model that has run at TP=4, with the figures its start reported.
func measuredAPIModel() *models.Model {
	m := &models.Model{
		ID:             "org/measured",
		TotalSizeBytes: 25_247_995_844,
		HFConfig: models.HFConfig{
			NumHiddenLayers: 64, HiddenSize: 5120, IntermediateSize: 17408,
			NumAttentionHeads: 24, NumKeyValueHeads: 4, HeadDim: 256,
			VocabSize: 248320, AttentionLayers: 16, MaxPositionEmbeddings: 262144,
		},
		VLLMConfig: models.VLLMConfig{
			TensorParallelSize: 4, MaxModelLen: 262144, KVCacheDtype: "fp8",
			GPUMemoryUtilization: 0.92, MaxNumSeqs: 8, MaxNumBatchedTokens: 8192,
		},
	}
	m.Measured = &models.RunMeasurement{
		At: time.Date(2026, 9, 30, 15, 12, 0, 0, time.UTC), TP: 4, ContextTokens: 262144,
		Fingerprint: models.MeasurementFingerprint(m),
		Engine: advice.Measurements{
			KVCacheGB: 17.64, KVCacheTokens: 1583801, WeightsPerRankGB: 7.05,
			ConsumedGB: 11.67, PeakActivationGB: 1.27, GraphPoolGB: 1.21,
		},
	}
	return m
}

func TestPlanInputEstimatesByTheSameRuleAsThePanel(t *testing.T) {
	s := newGoldenServer(t, goldenEnvGeneric)
	s.gpuInvOverride = &models.GPUInventory{Count: 4, PerCardGB: 31.86, FreePerCardGB: 2, Known: true}

	m := measuredAPIModel()
	in := s.planInput(m, m.VLLMConfig, models.ContextMax, "")
	if in.Inventory.FreePerCardGB != 0 {
		t.Error("free memory was left in the inventory; the plan is for the model when it is running")
	}
	if in.Defaults.GPUMemoryUtilization != 0.90 || in.Defaults.MaxNumSeqs != 16 {
		t.Errorf("defaults = %+v, want the machine-wide 0.90 and 16", in.Defaults)
	}
	if in.Base != m.VLLMConfig {
		t.Error("the base config was not the one passed")
	}

	if est := in.Estimate(m.VLLMConfig); est.Source != models.SourceMeasured {
		t.Errorf("the configuration that ran was estimated as %q, want measured", est.Source)
	}
	other := m.VLLMConfig
	other.TensorParallelSize = 2
	if est := in.Estimate(other); est.Source != models.SourceProjected {
		t.Errorf("another width was estimated as %q, want projected", est.Source)
	}

	// Refinement at the width that ran plans on the measurement.
	in.FixedTP = 4
	if p := models.PlanFit(in); !p.Known || !p.All.Measured {
		t.Errorf("fixed-width plan: known=%v measured=%v (%s)", p.Known, p.All.Measured, p.Why)
	}
}

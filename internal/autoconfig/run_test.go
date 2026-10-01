package autoconfig

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// planRecorder is a Plan that records what it was asked to plan.
type planRecorder struct {
	bases []models.VLLMConfig
	kvs   []string
}

func (p *planRecorder) plan(base models.VLLMConfig, kv string) models.FitPlan {
	p.bases = append(p.bases, base)
	p.kvs = append(p.kvs, kv)
	all := models.WidthPlan{TP: 4, ContextTokens: 262144, Config: base}
	all.Config.TensorParallelSize, all.Config.MaxModelLen, all.Config.KVCacheDtype = 4, 262144, kv
	all.Config.GPUMemoryUtilization, all.Config.MaxNumSeqs, all.Config.KVCacheMemory = 0.9, 16, 0
	narrow := all
	narrow.TP, narrow.Config.TensorParallelSize = 2, 2
	all.Notes = []models.ProfileNote{{Field: "tensor_parallel_size", Reason: "all four", Origin: "this machine"}}
	narrow.Notes = []models.ProfileNote{{Field: "tensor_parallel_size", Reason: "two", Origin: "this machine"}}
	return models.FitPlan{Known: true, All: all, Narrow: &narrow, FirstGuess: true}
}

func runDeps(t *testing.T, card string) (Deps, *planRecorder) {
	t.Helper()
	m := thinkingCap()
	f := &fakeFetcher{cards: map[string]string{m.ID: card}}
	p := &planRecorder{}
	return Deps{
		Model: m, Base: m.VLLMConfig, Fetcher: f, Drafts: []*models.Model{dflashDraft()},
		MachineEnv: machineEnv, Plan: p.plan, NoHelperWhy: "No helper model is installed.",
	}, p
}

func TestRunWithoutAHelper(t *testing.T) {
	d, p := runDeps(t, readCard(t, "thinkingcap-27b-paro5.md"))
	var lines []string
	d.Progress = func(s string) { lines = append(lines, s) }
	res, err := Run(context.Background(), d, models.ContextMax)
	if err != nil {
		t.Fatal(err)
	}
	if res.AdviceFrom != "" || len(res.Advice) != 0 {
		t.Errorf("advice without a helper: %q", res.AdviceFrom)
	}
	if rowByKey(res.Rows, "field:speculative_config") == nil {
		t.Error("the command's settings were not proposed")
	}
	if !noteFor(res.Notes, "", "No helper model") {
		t.Error("no note saying the text was not read")
	}
	if want := []string{"Reading the model card", "Checking the answer", "Checking what fits on this machine"}; strings.Join(lines, "|") != strings.Join(want, "|") {
		t.Errorf("progress = %q", lines)
	}
	// Planned on the card's config, with the card's KV dtype.
	if len(p.bases) != 1 || p.bases[0].SpeculativeConfig == "" || !strings.Contains(p.bases[0].Env, "OMP_NUM_THREADS=8") || p.kvs[0] != "fp8" {
		t.Errorf("planned on %+v with %q", p.bases, p.kvs)
	}
	if res.CardHash == "" || len(res.CardSources) != 1 {
		t.Errorf("card provenance: %q %q", res.CardHash, res.CardSources)
	}
}

func TestRunWithAHelper(t *testing.T) {
	d, _ := runDeps(t, readCard(t, "flash-next-mxfp4.md"))
	d.Model = &models.Model{ID: "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"}
	d.Fetcher = &fakeFetcher{cards: map[string]string{d.Model.ID: readCard(t, "flash-next-mxfp4.md")}}
	calls := 0
	d.Helper = func(_ context.Context, _ string, _ map[string]any, _, _ string, out any) error {
		calls++
		return json.Unmarshal([]byte(`{"command_index": 2, "other_notes": ["Use the image's latest tag."]}`), out)
	}
	res, err := Run(context.Background(), d, models.ContextMax)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || res.AdviceFrom != "helper" || len(res.Advice) == 0 {
		t.Errorf("calls=%d from=%q", calls, res.AdviceFrom)
	}
	if r := rowByKey(res.Rows, "flag:--enable-expert-offload"); r == nil {
		t.Error("the helper's choice of the second command was not followed")
	}

	// An earlier reading of the same card is reused without asking.
	d.Previous = func(hash string) (json.RawMessage, bool) {
		if hash == res.CardHash {
			return res.Advice, true
		}
		return nil, false
	}
	again, _ := Run(context.Background(), d, models.ContextMax)
	if calls != 1 || again.AdviceFrom != "previous" {
		t.Errorf("calls=%d from=%q; an unchanged card was read again", calls, again.AdviceFrom)
	}
	d.Reread = true
	Run(context.Background(), d, models.ContextMax)
	if calls != 2 {
		t.Error("Reread did not ask again")
	}
}

func TestRunWhenTheHelperFails(t *testing.T) {
	d, _ := runDeps(t, readCard(t, "thinkingcap-27b-paro5.md"))
	d.Helper = func(context.Context, string, map[string]any, string, string, any) error {
		return errors.New("A benchmark is running and using the GPUs.")
	}
	res, err := Run(context.Background(), d, models.ContextMax)
	if err != nil {
		t.Fatal(err)
	}
	if !noteFor(res.Notes, "", "A benchmark is running") || rowByKey(res.Rows, "field:reasoning_parser") == nil {
		t.Error("a failed helper should leave the command's rows and say why the text was not read")
	}

	ctx, cancel := context.WithCancel(context.Background())
	d.Helper = func(context.Context, string, map[string]any, string, string, any) error {
		cancel()
		return context.Canceled
	}
	if _, err := Run(ctx, d, models.ContextMax); !errors.Is(err, context.Canceled) {
		t.Errorf("a cancelled run returned %v", err)
	}
}

func TestRunWithAnAnswerThatAddsNothing(t *testing.T) {
	d, _ := runDeps(t, readCard(t, "thinkingcap-27b-paro5.md"))
	d.Helper = func(_ context.Context, _ string, _ map[string]any, _, _ string, out any) error {
		return json.Unmarshal([]byte(`{}`), out)
	}
	res, _ := Run(context.Background(), d, models.ContextMax)
	if !noteFor(res.Notes, "", "found nothing") {
		t.Error("an empty reading was not noted")
	}
}

func TestRunWithNoCard(t *testing.T) {
	d, p := runDeps(t, "")
	res, err := Run(context.Background(), d, models.ContextMedium)
	if err != nil || len(res.Rows) != 0 || !noteFor(res.Notes, "", "No model card") || len(p.bases) != 1 {
		t.Errorf("err=%v rows=%d notes=%+v", err, len(res.Rows), res.Notes)
	}
}

func TestResultConfigAndProfile(t *testing.T) {
	d, p := runDeps(t, readCard(t, "thinkingcap-27b-paro5.md"))
	res, _ := Run(context.Background(), d, models.ContextMax)

	all, _, _ := res.Config("all", nil)
	narrow, _, _ := res.Config("narrow", nil)
	if all.TensorParallelSize != 4 || narrow.TensorParallelSize != 2 {
		t.Errorf("widths: %d / %d", all.TensorParallelSize, narrow.TensorParallelSize)
	}
	all.TensorParallelSize = narrow.TensorParallelSize
	if all != narrow {
		t.Error("the two widths differ in more than the hardware fields")
	}
	if len(p.bases) != 1 {
		t.Errorf("default ticks re-planned %d times", len(p.bases)-1)
	}

	ticked := DefaultTicks(res.Rows)
	ticked["field:speculative_config"] = false
	cfg, _, _ := res.Config("all", ticked)
	if cfg.SpeculativeConfig != "" {
		t.Error("an unticked row was applied")
	}
	if len(p.bases) != 2 || p.bases[1].SpeculativeConfig != "" || p.kvs[1] != "fp8" {
		t.Errorf("changed ticks were not re-planned on the reduced config with the card's dtype: %+v %q", p.bases, p.kvs)
	}

	prof, _, err := res.Profile("narrow", nil, models.ProfileMeta{Variant: "rdna4-clav"})
	if err != nil {
		t.Fatal(err)
	}
	r := prof.Autoconfig
	if prof.Source != models.ProfileSourceAutoconfig || r == nil || r.Class != models.ContextMax ||
		r.Width != "narrow" || !r.FirstGuess || r.CardHash != res.CardHash || prof.Variant != "rdna4-clav" {
		t.Errorf("profile: %+v record %+v", prof, r)
	}
	if prof.Config.TensorParallelSize != 2 {
		t.Error("the narrow config was not saved")
	}
	var flagNote, envNote, widthNote bool
	for _, n := range prof.Notes {
		flagNote = flagNote || (n.Field == "extra_flags" && strings.HasPrefix(n.Reason, "--override-generation-config"))
		envNote = envNote || (n.Field == "env" && strings.HasPrefix(n.Reason, "OMP_NUM_THREADS=8"))
		widthNote = widthNote || (n.Field == "tensor_parallel_size" && n.Reason == "two")
	}
	if !flagNote || !envNote || !widthNote {
		t.Errorf("notes: flag=%v env=%v narrow width=%v", flagNote, envNote, widthNote)
	}
}

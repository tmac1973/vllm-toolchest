package benchmark

import (
	"strings"
	"testing"
)

func TestParseSweepValuesNormalizesAndDedupes(t *testing.T) {
	f, ok := LookupSweepField("max_model_len")
	if !ok {
		t.Fatal("max_model_len is not registered")
	}

	got, err := ParseSweepValues(f, " 8192 ,32768, 8192 ,,65536 ")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"8192", "32768", "65536"}
	if len(got) != len(want) {
		t.Fatalf("got %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %q, want %q (order is the order they run in)", i, got[i], want[i])
		}
	}
}

func TestParseSweepValuesRejectsBadInput(t *testing.T) {
	intField, _ := LookupSweepField("max_model_len")
	if _, err := ParseSweepValues(intField, "8192,notanumber"); err == nil {
		t.Error("a non-numeric context length should be refused")
	}
	if _, err := ParseSweepValues(intField, "0"); err == nil {
		t.Error("a context length below the minimum should be refused")
	}

	choice, _ := LookupSweepField("kv_cache_dtype")
	if _, err := ParseSweepValues(choice, "auto,fp16"); err == nil {
		t.Error("a KV cache dtype vLLM does not accept should be refused")
	}
	if _, err := ParseSweepValues(choice, "auto,fp8"); err != nil {
		t.Errorf("auto,fp8 should be accepted: %v", err)
	}
}

func TestSweepCombinationsIsTheCartesianProduct(t *testing.T) {
	axes := []SweepAxis{
		{Field: "max_model_len", Values: []string{"8192", "32768"}},
		{Field: "kv_cache_dtype", Values: []string{"auto", "fp8"}},
	}
	combos := SweepCombinations(axes)
	if len(combos) != 4 {
		t.Fatalf("got %d combinations, want 4", len(combos))
	}
	seen := map[string]bool{}
	for _, c := range combos {
		seen[SweepKey(c)] = true
	}
	for _, want := range []string{
		"kv_cache_dtype=auto,max_model_len=8192",
		"kv_cache_dtype=fp8,max_model_len=8192",
		"kv_cache_dtype=auto,max_model_len=32768",
		"kv_cache_dtype=fp8,max_model_len=32768",
	} {
		if !seen[want] {
			t.Errorf("missing combination %q", want)
		}
	}
}

// No axes is one pass, not zero — otherwise every caller needs a special case
// for the un-swept job.
func TestSweepCombinationsWithNoAxes(t *testing.T) {
	combos := SweepCombinations(nil)
	if len(combos) != 1 || len(combos[0]) != 0 {
		t.Errorf("got %v, want one empty combination", combos)
	}
}

// Every axis is an engine-launch parameter, so a combination is a reload. The
// cells have to be ordered so each configuration loads once.
func TestExpandCellsGroupsBySweepBeforePreset(t *testing.T) {
	cells := ExpandCells(
		[]string{"m1", "m2"},
		[]string{"p1", "p2"},
		[]SweepAxis{{Field: "max_model_len", Values: []string{"8192", "32768"}}},
	)
	if len(cells) != 8 {
		t.Fatalf("got %d cells, want 8 (2 models × 2 presets × 2 values)", len(cells))
	}

	// Walk the cells and count how many times the engine configuration
	// changes. Eight cells over four configurations should be four loads.
	loads := 0
	prev := ""
	for _, c := range cells {
		key := c.ModelID + "|" + SweepKey(c.SweepValues)
		if key != prev {
			loads++
			prev = key
		}
	}
	if loads != 4 {
		t.Errorf("cell order implies %d engine loads, want 4 — presets must vary inside a sweep value, not outside it", loads)
	}
}

func TestApplySweepOverlaysWithoutMutatingTheBase(t *testing.T) {
	baseLen := 4096
	base := &ConfigOverrides{MaxModelLen: &baseLen, Dtype: strPtr("bfloat16")}

	out, err := ApplySweep(base, map[string]string{"max_model_len": "32768", "max_num_seqs": "64"})
	if err != nil {
		t.Fatal(err)
	}
	if out.MaxModelLen == nil || *out.MaxModelLen != 32768 {
		t.Errorf("swept value did not win over the base")
	}
	if out.MaxNumSeqs == nil || *out.MaxNumSeqs != 64 {
		t.Errorf("max_num_seqs was not applied")
	}
	if out.Dtype == nil || *out.Dtype != "bfloat16" {
		t.Errorf("a base override the sweep does not touch must survive")
	}
	// The base belongs to the job and is reused for every cell.
	if *base.MaxModelLen != 4096 {
		t.Errorf("ApplySweep mutated the job's base overrides: %d", *base.MaxModelLen)
	}
}

func TestValidateSweeps(t *testing.T) {
	if err := ValidateSweeps(nil); err != nil {
		t.Errorf("no sweeps is valid: %v", err)
	}
	if err := ValidateSweeps([]SweepAxis{{Field: "nonsense", Values: []string{"1"}}}); err == nil {
		t.Error("an unknown parameter should be refused")
	}
	if err := ValidateSweeps([]SweepAxis{{Field: "max_model_len"}}); err == nil {
		t.Error("an axis with no values should be refused")
	}
	if err := ValidateSweeps([]SweepAxis{
		{Field: "max_model_len", Values: []string{"8192"}},
		{Field: "max_model_len", Values: []string{"32768"}},
	}); err == nil {
		t.Error("sweeping one parameter twice should be refused")
	}

	// Each combination is an engine load, so a matrix that would run for days
	// is refused at submission rather than discovered hours in.
	big := []SweepAxis{
		{Field: "max_model_len", Values: []string{"1024", "2048", "4096", "8192"}},
		{Field: "max_num_seqs", Values: []string{"1", "4", "16", "64"}},
		{Field: "gpu_memory_utilization", Values: []string{"0.8", "0.85", "0.9", "0.95"}},
		{Field: "tensor_parallel_size", Values: []string{"1", "2", "4", "8"}},
	}
	err := ValidateSweeps(big)
	if err == nil {
		t.Fatal("256 combinations should be refused")
	}
	if !strings.Contains(err.Error(), "256") {
		t.Errorf("the refusal should say how many it would be: %v", err)
	}
}

func strPtr(s string) *string { return &s }

package api

import (
	"slices"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// A max_num_seqs sweep used to record the swept value on each run while
// launching every cell at the model's saved one.
func TestSweptMaxNumSeqsReachesTheLaunch(t *testing.T) {
	m := &models.Model{ID: "org/model", VLLMConfig: models.VLLMConfig{
		MaxModelLen: 4096, TensorParallelSize: 1, MaxNumSeqs: 32,
	}}

	cell8 := vllmStartConfigFor(m, benchmark.ConfigSnapshot{MaxNumSeqs: 8})
	cell64 := vllmStartConfigFor(m, benchmark.ConfigSnapshot{MaxNumSeqs: 64})
	if cell8.MaxNumSeqs != 8 || cell64.MaxNumSeqs != 64 {
		t.Fatalf("MaxNumSeqs = %d and %d, want the swept 8 and 64", cell8.MaxNumSeqs, cell64.MaxNumSeqs)
	}

	// EnsureModelLoaded decides whether to restart by comparing arguments, so
	// two cells that differ here must differ there too — otherwise the second
	// is reported as already loaded and measured against the first's engine.
	if slices.Equal(process.BuildArgs(cell8), process.BuildArgs(cell64)) {
		t.Error("cells differing in max_num_seqs build identical arguments")
	}

	// Zero still means "use the model's saved setting".
	if got := vllmStartConfigFor(m, benchmark.ConfigSnapshot{}).MaxNumSeqs; got != 32 {
		t.Errorf("an unset override gave MaxNumSeqs = %d, want the saved 32", got)
	}
}

package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The config.json of tcclaviger/Qwen3.8-27B-DFlash2-FP8, cut down to the
// fields that are read.
const dflashConfig = `{
  "architectures": ["DFlash2DraftModel"],
  "dflash_config": {"block_size": 8, "mask_token_id": 248070, "target_layer_ids": [5, 19, 33, 47, 61]},
  "hidden_size": 5120,
  "model_type": "qwen3",
  "num_hidden_layers": 5,
  "num_target_layers": 64,
  "vocab_size": 248320
}`

func writeModelDir(t *testing.T, config string, weights int64) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if weights > 0 {
		// Sparse: only the size is read, and the real thing is gigabytes.
		path := filepath.Join(dir, "model.safetensors")
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Truncate(path, weights); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestADFlashCheckpointIsRecognisedAsADraft(t *testing.T) {
	cfg := ParseHFConfig(writeModelDir(t, dflashConfig, 0))

	if cfg.Draft == nil {
		t.Fatal("a checkpoint with a dflash_config was not recognised as a draft")
	}
	if cfg.Draft.Method != "dflash" || cfg.Draft.BlockSize != 8 || cfg.Draft.TargetLayers != 64 {
		t.Errorf("draft = %+v, want method dflash, block size 8, 64 target layers", *cfg.Draft)
	}
}

// A draft of a kind this tool cannot write a config for is still a draft: it
// must stay off the list of things that can be started.
func TestAnUnknownKindOfDraftIsStillADraft(t *testing.T) {
	cfg := ParseHFConfig(writeModelDir(t, `{"architectures": ["SomeNewDraftModel"], "hidden_size": 1024}`, 0))
	if cfg.Draft == nil || cfg.Draft.Method != "" {
		t.Errorf("draft = %+v, want a draft with no known method", cfg.Draft)
	}
}

func TestAnOrdinaryModelIsNotADraft(t *testing.T) {
	cfg := ParseHFConfig(writeModelDir(t,
		`{"architectures": ["Qwen3_5ForConditionalGeneration"], "hidden_size": 5120, "num_hidden_layers": 64}`, 0))
	if cfg.Draft != nil {
		t.Errorf("an ordinary model was marked as a draft: %+v", *cfg.Draft)
	}
}

func TestSpeculativeConfigIsReadForItsDraft(t *testing.T) {
	ref, ok := ParseSpeculative(`{"method": "dflash", "model": "/data/models/a/b/", "num_speculative_tokens": 7, "disable_padded_drafter_batch": true}`)
	if !ok || ref.Method != "dflash" || ref.Tokens != 7 || ref.LocalDraft() != "/data/models/a/b" {
		t.Errorf("ref = %+v (ok %v)", ref, ok)
	}

	// MTP drafts with the target's own head, and a Hub id is fetched by the
	// engine: neither names a directory this tool should look for.
	for _, raw := range []string{
		`{"method": "mtp", "num_speculative_tokens": 3}`,
		`{"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7}`,
	} {
		if ref, _ := ParseSpeculative(raw); ref.LocalDraft() != "" {
			t.Errorf("%s names local draft %q, want none", raw, ref.LocalDraft())
		}
	}

	// Half-typed JSON is what an autosaving panel sends. Not an error here.
	for _, raw := range []string{"", "   ", `{"method": "dfl`} {
		if _, ok := ParseSpeculative(raw); ok {
			t.Errorf("%q was read as a speculative config", raw)
		}
	}
}

func draftModel(path string) *Model {
	return &Model{ID: "acme/draft", LocalPath: path, HFConfig: HFConfig{
		HiddenSize: 5120, VocabSize: 248320,
		Draft: &DraftMeta{Method: "dflash", BlockSize: 8, TargetLayers: 64},
	}}
}

func TestADraftWritesItsOwnSpeculativeConfig(t *testing.T) {
	d := draftModel("/data/models/acme/draft")

	got := d.SpeculativeConfigFor(0)
	want := `{"method": "dflash", "model": "/data/models/acme/draft", "num_speculative_tokens": 7}`
	if got != want {
		t.Errorf("config = %s\n  want   %s", got, want)
	}
	// And what it writes is what is read back.
	if ref, ok := ParseSpeculative(got); !ok || ref.LocalDraft() != d.LocalPath || ref.Tokens != d.MaxDraftTokens() {
		t.Errorf("round trip gave %+v", ref)
	}
	if got := (&Model{}).SpeculativeConfigFor(0); got != "" {
		t.Errorf("a model that is not a draft wrote a speculative config: %s", got)
	}
}

func TestDraftMismatchComparesOnlyWhatBothState(t *testing.T) {
	d := draftModel("/x")
	target := func(hidden, layers, vocab int) *Model {
		return &Model{HFConfig: HFConfig{HiddenSize: hidden, NumHiddenLayers: layers, VocabSize: vocab}}
	}

	if why := DraftMismatch(target(5120, 64, 248320), d); why != "" {
		t.Errorf("a matching pair was refused: %s", why)
	}
	for name, tc := range map[string]struct {
		target *Model
		want   string
	}{
		"a narrower model":  {target(2560, 64, 248320), "hidden size"},
		"a shallower model": {target(5120, 48, 248320), "48"},
		"another tokenizer": {target(5120, 64, 151936), "vocabulary"},
	} {
		if why := DraftMismatch(tc.target, d); !strings.Contains(why, tc.want) {
			t.Errorf("%s: mismatch = %q, want it to mention %q", name, why, tc.want)
		}
	}
	// A config that does not state a field has not contradicted anything.
	if why := DraftMismatch(target(0, 0, 0), d); why != "" {
		t.Errorf("an unknown shape was treated as a mismatch: %s", why)
	}
}

// The draft is loaded beside the target and is in none of the target's own
// figures, so an estimate that leaves it out is short by its whole size.
func TestTheEstimateCountsTheDraftModel(t *testing.T) {
	const gib = 1024 * 1024 * 1024
	draftDir := writeModelDir(t, dflashConfig, 2*gib)

	m := &Model{
		ID:             "acme/target",
		TotalSizeBytes: 20 * gib,
		HFConfig:       HFConfig{HiddenSize: 5120, NumHiddenLayers: 64, MaxPositionEmbeddings: 8192},
		Quantization:   QuantMeta{BytesPerParam: 1},
		VLLMConfig:     VLLMConfig{TensorParallelSize: 1, MaxModelLen: 8192},
	}
	without := EstimateVRAM(m, nil)

	m.VLLMConfig.SpeculativeConfig = draftModel(draftDir).SpeculativeConfigFor(0)
	with := EstimateVRAM(m, nil)

	if with.DraftGB < 1.99 || with.DraftGB > 2.01 {
		t.Errorf("draft = %.2f GB, want the 2 GB on disk", with.DraftGB)
	}
	if got := with.TotalRequiredGB - without.TotalRequiredGB; got < 1.99 || got > 2.01 {
		t.Errorf("the draft moved the total by %.2f GB, want 2", got)
	}

	// MTP has no separate checkpoint, and a draft that is not on this disk
	// has no size to count.
	for _, spec := range []string{
		`{"method": "mtp", "num_speculative_tokens": 3}`,
		`{"method": "dflash", "model": "/nowhere/at/all", "num_speculative_tokens": 7}`,
	} {
		m.VLLMConfig.SpeculativeConfig = spec
		if est := EstimateVRAM(m, nil); est.DraftGB != 0 {
			t.Errorf("%s counted %.2f GB of draft", spec, est.DraftGB)
		}
	}
}

// A draft downloaded before drafts were recognised is on record as an
// ordinary model. The next scan has to correct that, or it keeps its radio
// button and its Start.
func TestAnExistingRecordIsReclassifiedOnTheNextScan(t *testing.T) {
	dataDir := t.TempDir()
	modelsDir := filepath.Join(dataDir, "models")
	dir := filepath.Join(modelsDir, "acme", "draft")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(dflashConfig), 0o644); err != nil {
		t.Fatal(err)
	}

	reg := NewRegistry(dataDir, modelsDir)
	if err := reg.Register(&Model{
		ID: "acme/draft", LocalPath: dir, Enabled: true,
		// As the parser before this one left it.
		HFConfig: HFConfig{HiddenSize: 5120, NumHiddenLayers: 5, AttentionLayers: 5, MetaVersion: 1},
	}); err != nil {
		t.Fatal(err)
	}

	reg.Maintenance()

	m, _ := reg.Get("acme/draft")
	if !m.IsDraft() {
		t.Error("a draft registered by an older build is still an ordinary model after a scan")
	}
}

package huggingface

import (
	"slices"
	"testing"
)

// Mistral's repo holds its weights twice. With a shard index beside it, the
// consolidated copy is left; on its own, it is the weights and is kept.
func TestDownloadableFilesLeavesTheConsolidatedCopy(t *testing.T) {
	names := func(fs []ModelFile) []string {
		var out []string
		for _, f := range fs {
			out = append(out, f.Filename)
		}
		return out
	}
	both := []ModelFile{
		{Filename: "config.json", Category: "config"}, {Filename: "params.json", Category: "other"},
		{Filename: "tekken.json", Category: "other"}, {Filename: "consolidated.safetensors", Category: "weight"},
		{Filename: "model.safetensors.index.json", Category: "config"},
		{Filename: "model-00001-of-00002.safetensors", Category: "weight"}, {Filename: "model-00002-of-00002.safetensors", Category: "weight"},
	}
	got := names(DownloadableFiles(both))
	if slices.Contains(got, "consolidated.safetensors") || !slices.Contains(got, "model-00002-of-00002.safetensors") || !slices.Contains(got, "tekken.json") {
		t.Errorf("both layouts: %v", got)
	}
	alone := []ModelFile{{Filename: "params.json", Category: "other"}, {Filename: "consolidated.safetensors", Category: "weight"}}
	if got := names(DownloadableFiles(alone)); !slices.Contains(got, "consolidated.safetensors") {
		t.Errorf("consolidated alone: %v", got)
	}
}

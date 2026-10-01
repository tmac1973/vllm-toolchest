package huggingface

import (
	"slices"
	"testing"
)

// A repo with its weights in both layouts gets one. Mistral's own repo has no
// Hugging Face tokenizer or processor beside its shards, so its consolidated
// copy is the one that serves; a repo whose shards are complete keeps them.
func TestDownloadableFilesTakesOneLayout(t *testing.T) {
	names := func(fs []ModelFile) []string {
		var out []string
		for _, f := range fs {
			out = append(out, f.Filename)
		}
		return out
	}
	mistral := []ModelFile{
		{Filename: "config.json", Category: "config"}, {Filename: "params.json", Category: "other"},
		{Filename: "tekken.json", Category: "other"}, {Filename: "consolidated.safetensors", Category: "weight"},
		{Filename: "model.safetensors.index.json", Category: "config"},
		{Filename: "model-00001-of-00002.safetensors", Category: "weight"}, {Filename: "model-00002-of-00002.safetensors", Category: "weight"},
	}
	got := names(DownloadableFiles(mistral))
	if !slices.Contains(got, "consolidated.safetensors") || slices.Contains(got, "model-00001-of-00002.safetensors") ||
		slices.Contains(got, "model.safetensors.index.json") || !slices.Contains(got, "tekken.json") || !slices.Contains(got, "params.json") {
		t.Errorf("Mistral layout: %v", got)
	}

	complete := append(slices.Clone(mistral), ModelFile{Filename: "tokenizer.json", Category: "tokenizer"})
	got = names(DownloadableFiles(complete))
	if slices.Contains(got, "consolidated.safetensors") || !slices.Contains(got, "model-00002-of-00002.safetensors") {
		t.Errorf("complete shards: %v", got)
	}

	alone := []ModelFile{{Filename: "params.json", Category: "other"}, {Filename: "consolidated.safetensors", Category: "weight"}}
	if got := names(DownloadableFiles(alone)); !slices.Contains(got, "consolidated.safetensors") {
		t.Errorf("consolidated alone: %v", got)
	}
}

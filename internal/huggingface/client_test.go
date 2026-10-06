package huggingface

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Sizes are binary (1 KB = 1024 B) and shown to one decimal, which is what
// the download and disk-space panels print beside each other.
func TestFormatBytesPicksTheLargestFittingUnit(t *testing.T) {
	cases := []struct {
		in   int64
		want string
	}{
		{0, "0 B"},
		{1023, "1023 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{1024*1024 - 1, "1024.0 KB"}, // rounds up in the display, but is still under a MB
		{1024 * 1024, "1.0 MB"},
		{16 * 1024 * 1024 * 1024, "16.0 GB"},
		// About what an 8B model in BF16 takes on disk.
		{16_381_517_312, "15.3 GB"},
		// Nothing above GB: a 1 TiB checkpoint reads in GB.
		{1 << 40, "1024.0 GB"},
		// A negative size only arises from a subtraction gone wrong; it
		// must still print rather than panic.
		{-5, "-5 B"},
	}
	for _, c := range cases {
		if got := FormatBytes(c.in); got != c.want {
			t.Errorf("FormatBytes(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

func ids(g ModelGroup) []string {
	out := make([]string, len(g.Variants))
	for i, v := range g.Variants {
		out[i] = v.ID
	}
	return out
}

// A publisher's own quantizations are variants of its model. A third party's
// repack of the same model is not -- it is a different publisher's artifact
// and must not be offered as if Qwen had shipped it.
func TestGroupResultsGroupsAPublishersQuantsAndKeepsRepacksApart(t *testing.T) {
	results := []ModelSearchResult{
		{ID: "Qwen/Qwen3-8B", Author: "Qwen"},
		{ID: "unsloth/Qwen3-8B-GGUF", Author: "unsloth"},
		{ID: "Qwen/Qwen3-8B-FP8", Author: "Qwen"},
		// The Hub sometimes leaves author empty; it comes from the ID then.
		{ID: "Qwen/Qwen3-8B-AWQ"},
		{ID: "Qwen/Qwen3-30B-A3B", Author: "Qwen"},
	}
	got := GroupResults(results)
	if len(got) != 3 {
		t.Fatalf("got %d groups, want 3: %+v", len(got), got)
	}
	// Groups appear in order of first result, which is the Hub's download
	// order, so the most popular stays on top.
	wantNames := []string{"Qwen/qwen3-8b", "unsloth/qwen3-8b", "Qwen/qwen3-30b-a3b"}
	for i, w := range wantNames {
		if got[i].BaseName != w {
			t.Errorf("group %d = %q, want %q", i, got[i].BaseName, w)
		}
	}
	if want := []string{"Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8", "Qwen/Qwen3-8B-AWQ"}; !reflect.DeepEqual(ids(got[0]), want) {
		t.Errorf("Qwen3-8B variants = %v, want %v", ids(got[0]), want)
	}
	if want := []string{"unsloth/Qwen3-8B-GGUF"}; !reflect.DeepEqual(ids(got[1]), want) {
		t.Errorf("unsloth variants = %v, want %v", ids(got[1]), want)
	}
}

func TestGroupResultsOfNothingIsEmpty(t *testing.T) {
	if got := GroupResults(nil); len(got) != 0 {
		t.Errorf("got %+v", got)
	}
}

// Mistral ships one consolidated file (or numbered parts) beside, or instead
// of, the sharded HF layout. Downloading both doubles the transfer, so the
// name test has to catch every spelling and nothing else.
func TestIsConsolidatedWeightsMatchesMistralsLayoutOnly(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"consolidated.safetensors", true},
		{"Consolidated.SafeTensors", true},
		{"consolidated-00001-of-00002.safetensors", true},
		{"subdir/consolidated.safetensors", true},
		{"model-00001-of-00004.safetensors", false},
		{"model.safetensors", false},
		{"consolidated.00.pth", false},
		{"consolidated.safetensors.index.json", false},
		// The directory is not the file: only the base name counts.
		{"consolidated/model.safetensors", false},
	}
	for _, c := range cases {
		if got := IsConsolidatedWeights(c.name); got != c.want {
			t.Errorf("IsConsolidatedWeights(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

// The Hub's gated field is false for open repos and a string for gated ones,
// and has been seen as true. A gated repo downloaded without a token fails
// with a 401 partway through, so every gated form must read as gated.
func TestGatedFieldReadsEveryFormTheHubSends(t *testing.T) {
	cases := []struct {
		json  string
		want  GatedField
		gated bool
	}{
		{`false`, "", false},
		{`"auto"`, "auto", true},
		{`"manual"`, "manual", true},
		{`true`, "auto", true},
		{`null`, "", false},
		{`"false"`, "false", false},
		// An unexpected type is treated as not gated rather than failing
		// the whole search response.
		{`42`, "", false},
	}
	for _, c := range cases {
		var r ModelSearchResult
		if err := json.Unmarshal([]byte(`{"id":"org/m","gated":`+c.json+`}`), &r); err != nil {
			t.Errorf("gated %s: %v", c.json, err)
			continue
		}
		if r.Gated != c.want || r.Gated.IsGated() != c.gated {
			t.Errorf("gated %s: got %q (IsGated %v), want %q (%v)", c.json, r.Gated, r.Gated.IsGated(), c.want, c.gated)
		}
	}

	// Absent is the same as false.
	var r ModelSearchResult
	if err := json.Unmarshal([]byte(`{"id":"org/m"}`), &r); err != nil || r.Gated.IsGated() {
		t.Errorf("absent gated field: %q, %v", r.Gated, err)
	}
}

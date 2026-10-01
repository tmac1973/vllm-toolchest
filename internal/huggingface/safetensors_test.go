package huggingface

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// With any expand[] the Hub returns only what was expanded, so the search
// names every field a result uses; and it reads the sizes back.
func TestSearchExpandsEveryFieldItUses(t *testing.T) {
	var query string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		query = r.URL.RawQuery
		json.NewEncoder(w).Encode([]map[string]any{{
			"id": "org/m", "downloads": 9, "tags": []string{"fp8"},
			"config":      map[string]any{"quantization_config": map[string]any{"quant_method": "fp8"}},
			"safetensors": map[string]any{"parameters": map[string]int64{"F8_E4M3": 1000}, "total": 1000},
		}})
	}))
	defer srv.Close()
	c := NewClient("")
	c.SetBaseURL(srv.URL)
	got, err := c.Search(context.Background(), "m", "")
	if err != nil || len(got) != 1 {
		t.Fatalf("results %v, err %v", got, err)
	}
	for _, f := range []string{"config", "safetensors", "tags", "downloads", "author", "gated"} {
		if !strings.Contains(query, "expand%5B%5D="+f) && !strings.Contains(query, "expand[]="+f) {
			t.Errorf("the search does not expand %s: %s", f, query)
		}
	}
	if got[0].Safetensors == nil || got[0].Safetensors.Parameters["F8_E4M3"] != 1000 || got[0].QuantFormat == "" {
		t.Errorf("result: %+v", got[0])
	}
}

// The detail view's sizes come from the file tree, the only exact source for
// every format, through the download's own filter: a repository holding its
// weights twice, as Mistral's does, is counted once. A tree with no weight
// file is "not known", not zero.
func TestGetModelSizes(t *testing.T) {
	for _, withWeights := range []bool{true, false} {
		var metaQuery string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch {
			case r.URL.Path == "/api/models/org/m":
				metaQuery = r.URL.RawQuery
				json.NewEncoder(w).Encode(map[string]any{"id": "org/m", "author": "org", "sha": "abc", "tags": []string{},
					"safetensors": map[string]any{"parameters": map[string]int64{"I32": 8000}, "total": 8000}})
			case strings.HasPrefix(r.URL.Path, "/api/models/org/m/tree/"):
				tree := []map[string]any{{"type": "file", "path": "config.json", "size": 10}, {"type": "file", "path": "params.json", "size": 5}}
				if withWeights {
					tree = append(tree,
						map[string]any{"type": "file", "path": "consolidated.safetensors", "size": 3000},
						map[string]any{"type": "file", "path": "model.safetensors.index.json", "size": 20},
						map[string]any{"type": "file", "path": "model-00001-of-00002.safetensors", "size": 1500},
						map[string]any{"type": "file", "path": "model-00002-of-00002.safetensors", "size": 1500},
						map[string]any{"type": "file", "path": "tokenizer.json", "size": 100})
				}
				json.NewEncoder(w).Encode(tree)
			default:
				http.NotFound(w, r)
			}
		}))
		c := NewClient("")
		c.SetBaseURL(srv.URL)
		d, err := c.GetModel(context.Background(), "org/m")
		srv.Close()
		if err != nil {
			t.Fatal(err)
		}
		if d.Revision != "abc" || !strings.Contains(metaQuery, "sha") || d.Safetensors == nil {
			t.Errorf("meta query %q, revision %q", metaQuery, d.Revision)
		}
		// HF shards with a tokenizer: the shards are kept, consolidated left.
		if withWeights && (!d.WeightsKnown || d.WeightsBytes != 3000 || d.TotalSize != 3000+10+5+20+100) {
			t.Errorf("with weights: weights %d known %v, download %d", d.WeightsBytes, d.WeightsKnown, d.TotalSize)
		}
		if !withWeights && (d.WeightsKnown || d.WeightsBytes != 0) {
			t.Errorf("without weights: %d %v", d.WeightsBytes, d.WeightsKnown)
		}
	}
}

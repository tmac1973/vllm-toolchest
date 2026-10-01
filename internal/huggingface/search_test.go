package huggingface

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

// A repo published for vLLM may be tagged with the vllm library alone, as
// mistralai/Mistral-Small-3.2-24B-Instruct-2506 is. Both libraries are asked,
// and the answers merged by downloads, each repo once.
func TestSearchAsksBothLibraries(t *testing.T) {
	answers := map[string][]ModelSearchResult{
		"transformers": {{ID: "org/popular", Downloads: 900}, {ID: "org/both", Downloads: 50}},
		"vllm":         {{ID: "mistralai/Mistral-Small-3.2-24B-Instruct-2506", Downloads: 250}, {ID: "org/both", Downloads: 50}},
	}
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lib := r.URL.Query()["filter"][0]
		asked = append(asked, lib)
		json.NewEncoder(w).Encode(answers[lib])
	}))
	defer srv.Close()
	c := NewClient("")
	c.SetBaseURL(srv.URL)

	got, err := c.Search(context.Background(), "mistral", "")
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, r := range got {
		ids = append(ids, r.ID)
	}
	want := []string{"org/popular", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "org/both"}
	if !slices.Equal(ids, want) {
		t.Errorf("results %v, want %v", ids, want)
	}
	slices.Sort(asked)
	if !slices.Equal(asked, []string{"transformers", "vllm"}) {
		t.Errorf("asked %v", asked)
	}
}

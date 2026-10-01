package huggingface

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func cardHub(t *testing.T, seenAuth *string) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*seenAuth = r.Header.Get("Authorization")
		switch r.URL.Path {
		case "/org/quant/raw/main/README.md":
			w.Write([]byte("# Quant\nvllm serve org/quant"))
		case "/api/models/org/quant":
			w.Write([]byte(`{"cardData": {"base_model": "org/base"}}`))
		case "/api/models/org/listed":
			w.Write([]byte(`{"cardData": {"base_model": ["org/first", "org/second"]}}`))
		case "/api/models/org/none":
			w.Write([]byte(`{"cardData": {}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	c := NewClient("secret")
	c.SetBaseURL(srv.URL)
	return c
}

func TestModelCard(t *testing.T) {
	var auth string
	c := cardHub(t, &auth)
	ctx := context.Background()

	card, err := c.ModelCard(ctx, "org/quant")
	if err != nil || card != "# Quant\nvllm serve org/quant" {
		t.Errorf("card = %q, err = %v", card, err)
	}
	if auth != "Bearer secret" {
		t.Errorf("Authorization = %q; gated repositories need the token", auth)
	}
	if card, err := c.ModelCard(ctx, "org/missing"); card != "" || err != nil {
		t.Errorf("a missing card: %q, %v; want empty and no error", card, err)
	}
}

func TestBaseModel(t *testing.T) {
	var auth string
	c := cardHub(t, &auth)
	ctx := context.Background()
	for id, want := range map[string]string{
		"org/quant": "org/base", "org/listed": "org/first", "org/none": "", "org/missing": "",
	} {
		if got := c.BaseModel(ctx, id); got != want {
			t.Errorf("BaseModel(%s) = %q, want %q", id, got, want)
		}
	}
}

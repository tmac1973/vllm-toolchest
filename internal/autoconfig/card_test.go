package autoconfig

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type fakeFetcher struct {
	cards map[string]string
	base  map[string]string
	asked []string
}

func (f *fakeFetcher) ModelCard(_ context.Context, id string) (string, error) {
	f.asked = append(f.asked, id)
	if id == "org/broken" {
		return "", errors.New("HTTP 500")
	}
	return f.cards[id], nil
}

func (f *fakeFetcher) BaseModel(_ context.Context, id string) string { return f.base[id] }

func TestFetchCardReadsTheBaseModelToo(t *testing.T) {
	f := &fakeFetcher{
		cards: map[string]string{
			"org/quant": "# Quant\n\n## Usage\n\nvllm serve org/quant --kv-cache-dtype fp8\n",
			"org/base":  "# Base\n\n## Citation\n\nplease cite\n\n## Best practice\n\ntemperature 0.6\n",
		},
		base: map[string]string{"org/quant": "org/base"},
	}
	c := FetchCard(context.Background(), f, "org/quant", 0)
	if len(c.Sources) != 2 || c.Sources[0] != "org/quant" {
		t.Fatalf("sources = %q", c.Sources)
	}
	if strings.Index(c.Text, "org/quant") > strings.Index(c.Text, "org/base") {
		t.Error("the model's own card is not first")
	}
	if strings.Contains(c.Text, "please cite") || !strings.Contains(c.Text, "temperature 0.6") {
		t.Errorf("trimming kept the wrong sections:\n%s", c.Text)
	}
	if c.Hash == "" {
		t.Error("no hash")
	}

	before := c.Hash
	f.cards["org/base"] += "\nmore\n"
	if FetchCard(context.Background(), f, "org/quant", 0).Hash == before {
		t.Error("the hash did not change with the base card")
	}
}

func TestFetchCardWithoutACard(t *testing.T) {
	f := &fakeFetcher{base: map[string]string{"org/quant": "org/quant"}}
	if c := FetchCard(context.Background(), f, "org/quant", 0); c.Raw != "" || c.Hash != "" {
		t.Errorf("an empty repository produced a card: %+v", c)
	}
	if len(f.asked) != 1 {
		t.Errorf("asked for %q; a base naming itself must not be read twice", f.asked)
	}
	if c := FetchCard(context.Background(), f, "local-model", 0); c.Raw != "" || len(f.asked) != 1 {
		t.Error("a model with no Hub repository was looked up")
	}
	if c := FetchCard(context.Background(), f, "org/broken", 0); c.Raw != "" {
		t.Error("a failed fetch produced a card")
	}
}

// An HTML card keeps its lines, and its entities read as characters.
func TestTrimCardOnAnHTMLCard(t *testing.T) {
	text := TrimCard(readCard(t, "thinkingcap-27b-paro5.md"))
	if !strings.Contains(text, "-v <path>/ThinkingCap-3.8-27B-PARO5:/app/models") {
		t.Errorf("the docker command was not kept line by line with its entities unescaped:\n%s", text)
	}
	if strings.Contains(text, "&lt;") || strings.Contains(text, "<pre>") {
		t.Error("markup survived")
	}
}

func TestCapTextAndBudget(t *testing.T) {
	long := strings.Repeat("para one.\n\n", 50)
	cut := capText(long, 100)
	if len(cut) > 200 || !strings.HasSuffix(cut, "left out for length.]") {
		t.Errorf("cut = %q", cut)
	}
	if CardCharsForContext(16384) != int(float64(16384-2648)*2.5) || CardCharsForContext(1000) != 4000 || CardCharsForContext(1<<20) != 48000 {
		t.Error("card budget")
	}
}

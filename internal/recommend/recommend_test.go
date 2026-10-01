package recommend

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/variants"
)

// fakeHub answers from fixed results, counting fetches.
type fakeHub struct {
	mu        sync.Mutex
	byTag     map[string][]huggingface.ModelSearchResult
	configs   map[string]string
	files     map[string][]huggingface.ModelFile
	err       error
	cfgFetch  int
	treeFetch int
}

func (h *fakeHub) Candidates(_ context.Context, q huggingface.CandidateQuery) ([]huggingface.ModelSearchResult, error) {
	if h.err != nil {
		return nil, h.err
	}
	var out []huggingface.ModelSearchResult
	for _, r := range h.byTag[q.Tag] {
		r.QuantFormat = huggingface.DetectQuantFormat(r.ID, r.Tags, r.Config)
		out = append(out, r)
	}
	return out, nil
}

func (h *fakeHub) FetchConfigJSON(_ context.Context, id, _ string) ([]byte, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.cfgFetch++
	c, ok := h.configs[id]
	if !ok {
		return nil, errors.New("404")
	}
	return []byte(c), nil
}

func (h *fakeHub) GetFiles(_ context.Context, id, rev string) (string, []huggingface.ModelFile, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.treeFetch++
	return rev, h.files[id], nil
}

func dense(arch string, layers, hidden, heads, kv, maxPos int) string {
	return fmt.Sprintf(`{"architectures":[%q],"model_type":"x","hidden_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":128,"max_position_embeddings":%d,"vocab_size":150000,"intermediate_size":%d}`,
		arch, hidden, layers, heads, kv, maxPos, hidden*3)
}

var fp8Cfg = &huggingface.ModelConfigMeta{QuantizationConfig: &huggingface.QuantConfig{QuantMethod: "fp8"}}
var awqCfg = &huggingface.ModelConfigMeta{QuantizationConfig: &huggingface.QuantConfig{QuantMethod: "awq", Bits: 4}}

func result(id string, cfg *huggingface.ModelConfigMeta, dl int, params map[string]int64, modified string) huggingface.ModelSearchResult {
	return huggingface.ModelSearchResult{ID: id, Downloads: dl, Config: cfg, LastModified: modified, SHA: "abc",
		Safetensors: &huggingface.Safetensors{Parameters: params}, PipelineTag: "text-generation"}
}

func weights(gb float64) []huggingface.ModelFile {
	return []huggingface.ModelFile{{Filename: "model.safetensors", Size: int64(gb * (1 << 30)), Category: "weight"}, {Filename: "config.json", Size: 10, Category: "config"}}
}

func fourCards(archs map[string]bool) Profile {
	d, _ := variants.Get("rdna4-clav")
	d.Capabilities = slices.DeleteFunc(slices.Clone(d.Capabilities), func(c string) bool { return c == "expert_offload" })
	return NewProfile(models.GPUInventory{Count: 4, PerCardGB: 32, Known: true}, "R9700", "gfx1201", d, archs, archs != nil,
		models.PlanDefaults{GPUMemoryUtilization: 0.90, MaxNumSeqs: 16}, 188)
}

// A small market: an FP8 model, an AWQ one, a bf16 one, one too big for the
// cards, and one for each way of being unverified.
func market() *fakeHub {
	return &fakeHub{
		byTag: map[string][]huggingface.ModelSearchResult{
			"fp8": {
				result("Qwen/Big-FP8", fp8Cfg, 9000, map[string]int64{"F8_E4M3": 32e9}, "2026-09-01T00:00:00Z"),
				result("org/NoConfig-FP8", fp8Cfg, 500, map[string]int64{"F8_E4M3": 8e9}, "2026-09-02T00:00:00Z"),
				result("org/NoWeights-FP8", fp8Cfg, 400, map[string]int64{"F8_E4M3": 8e9}, "2026-09-03T00:00:00Z"),
				result("org/Odd-FP8", fp8Cfg, 300, map[string]int64{"F8_E4M3": 8e9}, "2026-09-04T00:00:00Z"),
				result("org/Huge-FP8", fp8Cfg, 8000, map[string]int64{"F8_E4M3": 400e9}, "2026-09-05T00:00:00Z"),
			},
			"awq": {result("org/Small-AWQ", awqCfg, 700, map[string]int64{"I32": 8e9, "F16": 1e9}, "2026-09-20T00:00:00Z")},
			"": {
				result("Qwen/Dense-BF16", &huggingface.ModelConfigMeta{Architectures: []string{"Qwen3ForCausalLM"}}, 6000, map[string]int64{"BF16": 8e9}, "2026-08-01T00:00:00Z"),
				result("Qwen/Big-FP8", fp8Cfg, 9000, map[string]int64{"F8_E4M3": 32e9}, "2026-09-01T00:00:00Z"), // the FP8 one again
			},
		},
		configs: map[string]string{
			"Qwen/Big-FP8":      dense("Qwen3ForCausalLM", 64, 5120, 64, 8, 131072),
			"org/NoWeights-FP8": dense("Qwen3ForCausalLM", 36, 4096, 32, 8, 40960),
			"org/Odd-FP8":       `{"architectures":["OddForCausalLM"]}`,
			"org/Small-AWQ":     dense("LlamaForCausalLM", 32, 4096, 32, 8, 131072),
			"Qwen/Dense-BF16":   dense("Qwen3ForCausalLM", 36, 4096, 32, 8, 40960),
		},
		files: map[string][]huggingface.ModelFile{
			"Qwen/Big-FP8": weights(32), "org/NoWeights-FP8": nil, "org/Odd-FP8": weights(8),
			"org/Small-AWQ": weights(5), "Qwen/Dense-BF16": weights(16), "org/NoConfig-FP8": weights(8),
		},
	}
}

func TestAcceleration(t *testing.T) {
	if got := acceleratedFormats("gfx1201", nil); !slices.Equal(got, []string{"fp8"}) {
		t.Errorf("gfx1201: %v", got)
	}
	if got := acceleratedFormats("gfx1100", nil); got != nil {
		t.Errorf("gfx1100: %v", got)
	}
	if got := acceleratedFormats("gfx9999", nil); got != nil {
		t.Errorf("unknown: %v", got)
	}
	if got := acceleratedFormats("gfx1201", []string{"fp8"}); got != nil {
		t.Errorf("vetoed: %v", got)
	}
	if !isAccelerated("FP8", []string{"fp8"}) {
		t.Error("the badge and the format compare case-insensitively")
	}
}

// Nothing accelerated still leaves the 4-bit and unquantized buckets.
func TestBucketsWithNothingAccelerated(t *testing.T) {
	tags := bucketTags(Profile{})
	if !slices.Contains(tags, "awq") || !slices.Contains(tags, "gptq") || !slices.Contains(tags, unquantizedTag) || slices.Contains(tags, "fp8") {
		t.Errorf("tags %q", tags)
	}
}

// The Hub's counts are parameters under the packing dtype for GPTQ: 4 bits
// each is the right least, and never more than the truth.
func TestLeastWeight(t *testing.T) {
	gptq := &huggingface.Safetensors{Parameters: map[string]int64{"I32": 32212254720, "BF16": 3739563184}}
	if gb, _ := leastWeightGB(gptq); gb < 21 || gb > 22.1 {
		t.Errorf("GPTQ-Int4 least %.1f GB, real 22.7", gb)
	}
	if _, ok := leastWeightGB(nil); ok {
		t.Error("a size without counts")
	}
}

func TestCoarseFilter(t *testing.T) {
	p := fourCards(nil)
	pool := []huggingface.ModelSearchResult{
		result("a/over", fp8Cfg, 1, map[string]int64{"F8_E4M3": 140 << 30}, ""),
		result("a/under", fp8Cfg, 1, map[string]int64{"F8_E4M3": 120 << 30}, ""),
		{ID: "a/unknown"},
		result("a/big-4bit", awqCfg, 1, map[string]int64{"I32": 320 << 30}, ""), // 160 GB at 4 bits
	}
	for i := range pool {
		pool[i].QuantFormat = huggingface.DetectQuantFormat(pool[i].ID, nil, pool[i].Config)
	}
	ids := func(rs []huggingface.ModelSearchResult) (out []string) {
		for _, r := range rs {
			out = append(out, r.ID)
		}
		return
	}
	if got := ids(coarseFilter(pool, p)); !slices.Equal(got, []string{"a/under", "a/unknown"}) {
		t.Errorf("survivors %v", got)
	}
	// With expert offload, a 4-bit model may use the RAM too; FP8 may not.
	p.ExpertOffload = true
	if got := ids(coarseFilter(pool, p)); !slices.Equal(got, []string{"a/under", "a/unknown", "a/big-4bit"}) {
		t.Errorf("with offload %v", got)
	}
}

func TestServable(t *testing.T) {
	ok := result("x/chat", nil, 100, map[string]int64{"BF16": 1e9}, "")
	emb := ok
	emb.PipelineTag = "feature-extraction"
	tiny := result("trl-internal-testing/tiny", nil, 1e6, map[string]int64{"F32": 1e6}, "")
	fresh := result("someone/new", nil, 3, map[string]int64{"BF16": 1e9}, "")
	trusted := result("Qwen/new", nil, 3, map[string]int64{"BF16": 1e9}, "")
	for r, want := range map[*huggingface.ModelSearchResult]bool{&ok: true, &emb: false, &tiny: false, &fresh: false, &trusted: true} {
		if servable(*r) != want {
			t.Errorf("%s: %v", r.ID, !want)
		}
	}
}

func TestWeightOnly(t *testing.T) {
	for q, want := range map[*huggingface.QuantConfig]bool{
		{QuantMethod: "awq"}: true,
		{QuantMethod: "compressed-tensors", Format: "pack-quantized"}:  true,
		{QuantMethod: "compressed-tensors", Format: "float-quantized"}: false,
		{QuantMethod: "fp8"}: false,
		nil:                  false,
	} {
		if weightOnly(q) != want {
			t.Errorf("%+v: %v", q, !want)
		}
	}
	if !isFloat8(&huggingface.QuantConfig{QuantMethod: "compressed-tensors", Format: "float-quantized"}) {
		t.Error("compressed-tensors FP8 is FP8")
	}
}

func TestTheEngine(t *testing.T) {
	hub := market()
	archs := map[string]bool{"Qwen3ForCausalLM": true, "OddForCausalLM": true}
	e := NewEngine(hub, t.TempDir())
	r := e.Result(context.Background(), fourCards(archs), "quality")
	if r.Unavailable != "" {
		t.Fatal(r.Unavailable)
	}
	reasons := map[string]string{}
	for _, c := range r.Unverified {
		reasons[c.ID] = c.Reason
	}
	want := map[string]string{
		"org/NoConfig-FP8":  reasonNoConfig,
		"org/NoWeights-FP8": reasonNoWeights,
		"org/Odd-FP8":       reasonUnreadable,
		"org/Small-AWQ":     reasonArch("LlamaForCausalLM"),
	}
	for id, reason := range want {
		if reasons[id] != reason {
			t.Errorf("%s: %q, want %q", id, reasons[id], reason)
		}
	}
	var verified []string
	for _, c := range r.Verified {
		verified = append(verified, c.ID)
		if c.TP == 0 || c.AffordableTokens == 0 || !c.ArchSupported {
			t.Errorf("%s: %+v", c.ID, c)
		}
	}
	slices.Sort(verified)
	if !slices.Equal(verified, []string{"Qwen/Big-FP8", "Qwen/Dense-BF16"}) {
		t.Errorf("verified %v", verified)
	}
	// The bf16 one came once, from the unquantized bucket; the FP8 one is not
	// counted twice. The 400 GB one never reached the fit.
	for _, c := range append(r.Verified, r.Unverified...) {
		if c.ID == "org/Huge-FP8" {
			t.Error("a model beyond every card was judged")
		}
	}
	if r.Verified[0].ID != "Qwen/Big-FP8" {
		t.Errorf("quality leads with %s", r.Verified[0].ID)
	}

	// With the registry unread, an architecture nobody listed is not held
	// against a model: a fresh install does not show an empty feed.
	e = NewEngine(market(), t.TempDir())
	r = e.Result(context.Background(), fourCards(nil), "quality")
	found := false
	for _, c := range r.Verified {
		found = found || c.ID == "org/Small-AWQ"
	}
	if !found {
		t.Error("an unknown registry held an architecture against a model")
	}
}

// A weight-only model never leads Fastest, and Context follows what each
// holds here, not what it advertises.
func TestObjectives(t *testing.T) {
	cands := []Candidate{
		{ID: "a/awq", WeightOnly: true, TP: 1, Available: 32, Spare: 20, Downloads: 9, AffordableTokens: 8192},
		{ID: "a/fp8", Accelerated: true, TP: 2, Available: 64, Spare: 30, Downloads: 5, AffordableTokens: 140000, ParamsB: 32},
		{ID: "a/bf16", TP: 1, Available: 32, Spare: 10, Downloads: 1, AffordableTokens: 40960, ParamsB: 8},
	}
	o := orders(cands, []int{0, 1, 2}, 4)
	if cands[o[IntentFastest][0]].WeightOnly {
		t.Error("a weight-only model leads Fastest")
	}
	if cands[o[IntentContext][0]].ID != "a/fp8" || cands[o[IntentQuality][0]].ID != "a/fp8" {
		t.Errorf("context %s, quality %s", cands[o[IntentContext][0]].ID, cands[o[IntentQuality][0]].ID)
	}
}

// A config is fetched once per revision; a new lastModified is a new fetch.
func TestConfigCache(t *testing.T) {
	dir := t.TempDir()
	hub := market()
	c := Candidate{ID: "Qwen/Big-FP8", SHA: "abc", LastModified: time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)}
	if !strings.Contains(cacheDir(dir, c), "Qwen%2FBig-FP8@2026-09-01T12:00:00Z") {
		t.Errorf("cache dir %s", cacheDir(dir, c))
	}
	fetchFinalist(context.Background(), hub, cacheDir(dir, c), c)
	fetchFinalist(context.Background(), hub, cacheDir(dir, c), c)
	if hub.cfgFetch != 1 || hub.treeFetch != 1 {
		t.Errorf("fetched config %d, tree %d times", hub.cfgFetch, hub.treeFetch)
	}
	c.LastModified = c.LastModified.Add(time.Hour)
	fetchFinalist(context.Background(), hub, cacheDir(dir, c), c)
	if hub.cfgFetch != 2 {
		t.Error("a new revision was not fetched")
	}
}

func TestStaleness(t *testing.T) {
	hub := market()
	e := NewEngine(hub, t.TempDir())
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	e.now = func() time.Time { return now }
	p := fourCards(nil)
	if r := e.Result(context.Background(), p, "quality"); r.Stale || len(r.Verified) == 0 {
		t.Fatalf("first: stale=%v verified=%d", r.Stale, len(r.Verified))
	}
	built := e.pool
	now = now.Add(7 * time.Hour)
	if r := e.Result(context.Background(), p, "newest"); !r.Stale || e.pool != built {
		t.Error("an old pool was rebuilt, or not marked stale")
	}
	p.Inventory.Count = 2
	if e.Result(context.Background(), p, "quality"); e.pool == built {
		t.Error("a pool for other cards was served")
	}
	if r := e.Result(context.Background(), p, "nonsense"); r.Intent != IntentQuality {
		t.Errorf("intent %q", r.Intent)
	}
}

func TestProfileKey(t *testing.T) {
	a := fourCards(nil)
	b := a
	a.Accelerated, b.Accelerated = []string{"fp8", "int8"}, []string{"int8", "fp8"}
	if a.Key() != b.Key() {
		t.Error("order of accelerated formats changed the key")
	}
	for _, change := range []func(*Profile){
		func(p *Profile) { p.Inventory.Count = 2 },
		func(p *Profile) { p.Inventory.PerCardGB = 16 },
		func(p *Profile) { p.GPUArch = "gfx1100" },
		func(p *Profile) { p.Variant = "radiance" },
		func(p *Profile) { p.ArchsKnown = !p.ArchsKnown },
	} {
		c := a
		change(&c)
		if c.Key() == a.Key() {
			t.Errorf("a change kept the key: %s", c.Key())
		}
	}
}

func TestUnavailable(t *testing.T) {
	hub := market()
	hub.err = errors.New("network down")
	if r := NewEngine(hub, t.TempDir()).Result(context.Background(), fourCards(nil), ""); !strings.Contains(r.Unavailable, "network down") || len(r.Verified) != 0 {
		t.Errorf("network: %+v", r)
	}
	p := fourCards(nil)
	p.Inventory.Known = false
	if r := NewEngine(market(), t.TempDir()).Result(context.Background(), p, ""); r.Unavailable != "the GPUs have not been read yet" {
		t.Errorf("no inventory: %q", r.Unavailable)
	}
}

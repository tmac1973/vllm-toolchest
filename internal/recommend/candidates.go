package recommend

import (
	"context"
	_ "embed"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Hub is what the engine asks of the Hugging Face client. An interface so
// the tests can answer from recorded payloads.
type Hub interface {
	Candidates(ctx context.Context, q huggingface.CandidateQuery) ([]huggingface.ModelSearchResult, error)
	FetchConfigJSON(ctx context.Context, modelID, revision string) ([]byte, error)
	GetFiles(ctx context.Context, modelID, revision string) (string, []huggingface.ModelFile, error)
}

// Verdict is which list a finalist ends in.
type Verdict string

const (
	Verified   Verdict = "verified"
	Unverified Verdict = "unverified"
	Dropped    Verdict = "dropped"
)

// Candidate is one repository, through as many stages as it reached.
type Candidate struct {
	ID           string    `json:"id"`
	Author       string    `json:"author"`
	Format       string    `json:"format"`
	Arch         string    `json:"arch,omitempty"`
	Gated        bool      `json:"gated"`
	Downloads    int       `json:"downloads"`
	LastModified time.Time `json:"last_modified"`

	WeightGB float64 `json:"weight_gb,omitempty"`
	ParamsB  float64 `json:"params_b,omitempty"`

	// From the plan at the recommended width.
	TP                  int     `json:"tp,omitempty"`
	Required            float64 `json:"required_gb,omitempty"`
	Available           float64 `json:"available_gb,omitempty"`
	Spare               float64 `json:"spare_gb,omitempty"`
	AffordableTokens    int     `json:"context_tokens,omitempty"`
	FullContextRequests int     `json:"full_context_requests,omitempty"`
	// Offload is a model that runs only with its experts in system RAM.
	Offload bool `json:"offload,omitempty"`
	// Featured is a model by one of the image's own publishers.
	Featured bool `json:"featured,omitempty"`

	Reason        string `json:"reason,omitempty"`
	Accelerated   bool   `json:"accelerated"`
	ArchSupported bool   `json:"arch_supported"`

	// Working state, never sent.
	SHA          string                   `json:"-"`
	WeightBytes  int64                    `json:"-"`
	WeightsKnown bool                     `json:"-"`
	WeightOnly   bool                     `json:"-"`
	Safetensors  *huggingface.Safetensors `json:"-"`
	Config       models.VLLMConfig        `json:"-"`
	Verdict      Verdict                  `json:"-"`
	coarse       float64
	bucket       string
}

const (
	finalists      = 40
	poolPerQuery   = 50
	fetchWorkers   = 8
	unquantizedTag = ""
)

//go:embed publishers.conf
var publishersConf string

// trustedPublishers are the authors whose checkpoints rank a little higher.
// Presence is a boost only; absence is never a penalty.
var trustedPublishers = func() map[string]bool {
	m := map[string]bool{}
	for _, line := range strings.Split(publishersConf, "\n") {
		if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "#") {
			m[strings.ToLower(line)] = true
		}
	}
	return m
}()

// bucketTags are the pool's queries: every accelerated format, the 4-bit
// weight-only formats, which are how a model that would not otherwise fit
// runs at all, and the unquantized, which on a large host is often the best
// answer and has no tag of its own.
func bucketTags(p Profile) []string {
	var tags []string
	for _, f := range p.Accelerated {
		tags = append(tags, huggingface.QuantFilterTags(f)...)
	}
	for _, b := range []string{"awq", "gptq", "compressed-tensors"} {
		tags = append(tags, huggingface.QuantFilterTags(b)...)
	}
	tags = append(tags, unquantizedTag)
	out := []string{}
	for _, t := range tags {
		if !slices.Contains(out, t) {
			out = append(out, t)
		}
	}
	return out
}

// fetchPool runs every query -- each bucket, in both libraries a servable
// repo is tagged with, by downloads and by recency -- and merges them, first
// occurrence winning. It fails only when every query does.
func fetchPool(ctx context.Context, hub Hub, p Profile) ([]huggingface.ModelSearchResult, error) {
	var queries []huggingface.CandidateQuery
	for _, tag := range bucketTags(p) {
		for _, lib := range []string{"transformers", "vllm"} {
			for _, sort := range []string{"downloads", "lastModified"} {
				queries = append(queries, huggingface.CandidateQuery{Library: lib, Tag: tag, Sort: sort, Limit: poolPerQuery})
			}
		}
	}
	results := make([][]huggingface.ModelSearchResult, len(queries))
	errs := make([]error, len(queries))
	var wg sync.WaitGroup
	sem := make(chan struct{}, 4)
	for i, q := range queries {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			batch, err := hub.Candidates(ctx, q)
			if err == nil && q.Tag == unquantizedTag {
				// A sieve, for the one bucket no tag expresses.
				batch = slices.DeleteFunc(batch, func(r huggingface.ModelSearchResult) bool {
					return r.QuantFormat != huggingface.FormatFP16
				})
			}
			results[i], errs[i] = batch, err
		}()
	}
	wg.Wait()

	seen := map[string]bool{}
	var merged []huggingface.ModelSearchResult
	var firstErr error
	failed := 0
	for i := range queries {
		if errs[i] != nil {
			failed++
			if firstErr == nil {
				firstErr = errs[i]
			}
			continue
		}
		for _, r := range results[i] {
			if !seen[r.ID] {
				seen[r.ID] = true
				merged = append(merged, r)
			}
		}
	}
	if failed == len(queries) {
		return nil, firstErr
	}
	return merged, nil
}

// leastWeightGB is the least a repository's weights can weigh, from the Hub's
// counts: floats at their width, integers at 4 bits. For a packed format the
// Hub recognises the counts are parameters under the packing dtype, and for
// one it does not they are stored elements, so integer dtypes may be either
// -- 4 bits never overstates them. ok is false without counts.
func leastWeightGB(s *huggingface.Safetensors) (float64, bool) {
	if s == nil || len(s.Parameters) == 0 {
		return 0, false
	}
	var bits int64
	for dtype, n := range s.Parameters {
		switch {
		case strings.HasPrefix(dtype, "F64"):
			bits += n * 64
		case strings.HasPrefix(dtype, "F32"):
			bits += n * 32
		case strings.HasPrefix(dtype, "F16"), strings.HasPrefix(dtype, "BF16"):
			bits += n * 16
		case strings.HasPrefix(dtype, "F8"):
			bits += n * 8
		default: // integer containers, BOOL, and anything unknown
			bits += n * 4
		}
	}
	return float64(bits) / 8 / (1 << 30), true
}

// coarseFilter drops what cannot run here under any split: weights at their
// least beyond every card. Unknown sizes stay. On an image that can hold a
// MoE's experts in RAM, a 4-bit model's bound is the cards and the RAM too --
// this tier cannot see experts, and the image offloads only 4-bit ones.
// Without that limit, 700 GB models took half the finalists' places on
// compute, to be dropped one by one after their configs were fetched.
func coarseFilter(pool []huggingface.ModelSearchResult, p Profile) []huggingface.ModelSearchResult {
	cards := float64(p.Inventory.Count) * p.Inventory.PerCardGB
	ram := max(0, models.OffloadHostRAMGB(p.HostRAMGB, p.Inventory.Count))
	return slices.DeleteFunc(slices.Clone(pool), func(r huggingface.ModelSearchResult) bool {
		gb, ok := leastWeightGB(r.Safetensors)
		if !ok {
			return false
		}
		bound := cards
		if p.ExpertOffload && bucketOf(r.QuantFormat, p.Accelerated) == bucketFourBit {
			bound += ram
		}
		return gb > bound
	})
}

// The buckets a candidate can fall in, each given places among the
// finalists.
const (
	bucketAccelerated = "accelerated"
	bucketFourBit     = "4-bit"
	bucketUnquantized = "unquantized"
	bucketOther       = "other"
)

// bucketPlaces is how many finalists each bucket is sure of; places a bucket
// cannot fill go to the best of the rest. Without them, on gfx1201, all forty
// were FP8: acceleration is a quarter of the coarse score and FP8 repos are
// popular, and a bf16 model that is the best answer on a large host never
// reached the fit.
var bucketPlaces = map[string]int{bucketAccelerated: 16, bucketFourBit: 12, bucketUnquantized: 12}

func bucketOf(format string, accelerated []string) string {
	switch {
	case isAccelerated(format, accelerated):
		return bucketAccelerated
	case format == huggingface.FormatFP16:
		return bucketUnquantized
	}
	switch strings.ToUpper(format) {
	case "AWQ", "GPTQ", "COMPRESSED-TENSORS", "MXFP4", "NVFP4", "BNB-4BIT":
		return bucketFourBit
	}
	return bucketOther
}

// notGenerative are the Hub tasks of models that are not served to chat or
// complete: embeddings, rerankers, classifiers, speech recognition. Found on
// compute's first live build, where an embedding model was second under
// Fastest.
var notGenerative = map[string]bool{
	"feature-extraction": true, "sentence-similarity": true, "text-ranking": true,
	"text-classification": true, "token-classification": true, "zero-shot-classification": true,
	"fill-mask": true, "automatic-speech-recognition": true, "audio-classification": true,
	"text-to-speech": true, "image-classification": true,
}

// minDownloads keeps out repositories nobody has used, unless their author is
// a trusted publisher: the queries by recency bring in uploads minutes old,
// and one led Newest on compute's first live build.
const minDownloads = 50

// minWeightGB is the least a real model weighs. Test fixtures are smaller and,
// downloaded by every CI run, popular: trl-internal-testing's tiny Qwen2 was
// second under Fastest on compute's first live build. The smallest real model
// in that pool, Qwen3-0.6B, is 1.4 GB.
const minWeightGB = 0.25

// servable reports a result worth a place in the pool.
func servable(r huggingface.ModelSearchResult) bool {
	if notGenerative[r.PipelineTag] {
		return false
	}
	if gb, ok := leastWeightGB(r.Safetensors); ok && gb < minWeightGB {
		return false
	}
	author, _, _ := strings.Cut(r.ID, "/")
	return r.Downloads >= minDownloads || trustedPublishers[strings.ToLower(author)]
}

// fetchFeatured is the image's own publishers' repositories, by downloads.
// Their downloads are not held against them, nor their formats: they are
// the image's own. Only what is not generative, a test fixture, or beyond the
// cards and the RAM together is left out before the fit.
func fetchFeatured(ctx context.Context, hub Hub, p Profile) []huggingface.ModelSearchResult {
	var out []huggingface.ModelSearchResult
	ram := max(0, models.OffloadHostRAMGB(p.HostRAMGB, p.Inventory.Count))
	bound := float64(p.Inventory.Count)*p.Inventory.PerCardGB + ram
	for _, author := range p.Featured {
		batch, err := hub.Candidates(ctx, huggingface.CandidateQuery{Author: author, Sort: "downloads", Limit: 100})
		if err != nil {
			continue
		}
		for _, r := range batch {
			if notGenerative[r.PipelineTag] {
				continue
			}
			if gb, ok := leastWeightGB(r.Safetensors); ok && (gb < minWeightGB || gb > bound) {
				continue
			}
			out = append(out, r)
		}
	}
	return out
}

// toCandidate is a search result as a candidate, from the cheap tier.
func toCandidate(r huggingface.ModelSearchResult, p Profile) Candidate {
	c := Candidate{
		ID: r.ID, Author: r.Author, Format: r.QuantFormat, Gated: r.Gated.IsGated(),
		Downloads: r.Downloads, SHA: r.SHA, Safetensors: r.Safetensors,
	}
	if c.Author == "" {
		c.Author, _, _ = strings.Cut(r.ID, "/")
	}
	if t, err := time.Parse(time.RFC3339, r.LastModified); err == nil {
		c.LastModified = t
	}
	if r.Config != nil {
		if len(r.Config.Architectures) > 0 {
			c.Arch = r.Config.Architectures[0]
		}
		c.WeightOnly = weightOnly(r.Config.QuantizationConfig)
	}
	if r.Safetensors != nil && r.Safetensors.Total > 0 {
		c.ParamsB = float64(r.Safetensors.Total) / 1e9
	}
	c.Accelerated = isAccelerated(c.Format, p.Accelerated)
	// FP8 stored as compressed-tensors is FP8: the badge says only the
	// container, the config says float-quantized.
	if r.Config != nil && isFloat8(r.Config.QuantizationConfig) && isAccelerated("fp8", p.Accelerated) {
		c.Accelerated = true
	}
	c.bucket = bucketOf(c.Format, p.Accelerated)
	if c.Accelerated {
		c.bucket = bucketAccelerated
	}
	return c
}

// isFloat8 reports compressed-tensors' 8-bit float layout.
func isFloat8(q *huggingface.QuantConfig) bool {
	return q != nil && strings.EqualFold(q.QuantMethod, "compressed-tensors") && q.Format == "float-quantized"
}

// weightOnly reports weights that unpack to bf16 before the matmul: the
// format buys memory, not speed.
func weightOnly(q *huggingface.QuantConfig) bool {
	if q == nil {
		return false
	}
	switch strings.ToLower(q.QuantMethod) {
	case "awq", "gptq":
		return true
	case "compressed-tensors":
		return q.Format == "pack-quantized"
	}
	return q.LoadIn4Bit || q.LoadIn8Bit
}

// coarseRank orders the pool by a crude score -- popularity, recency,
// acceleration, publisher -- and keeps the finalists, the only ones whose
// configs are fetched.
func coarseRank(cands []Candidate) []Candidate {
	if len(cands) == 0 {
		return cands
	}
	dl := percentiles(cands, func(c Candidate) float64 { return float64(c.Downloads) })
	rec := percentiles(cands, func(c Candidate) float64 { return float64(c.LastModified.Unix()) })
	for i := range cands {
		c := &cands[i]
		c.coarse = 0.40*dl[i] + 0.25*rec[i] + 0.25*b2f(c.Accelerated) + 0.10*b2f(trustedPublishers[strings.ToLower(c.Author)])
	}
	slices.SortStableFunc(cands, func(a, b Candidate) int {
		switch {
		case a.coarse > b.coarse:
			return -1
		case a.coarse < b.coarse:
			return 1
		}
		return strings.Compare(a.ID, b.ID)
	})

	// Each bucket's sure places first, best of each; then the rest by score.
	taken := make([]bool, len(cands))
	used := map[string]int{}
	var out []Candidate
	for i, c := range cands {
		if used[c.bucket] < bucketPlaces[c.bucket] {
			used[c.bucket]++
			taken[i] = true
			out = append(out, c)
		}
	}
	for i, c := range cands {
		if len(out) >= finalists {
			break
		}
		if !taken[i] {
			out = append(out, c)
		}
	}
	slices.SortStableFunc(out, func(a, b Candidate) int {
		switch {
		case a.coarse > b.coarse:
			return -1
		case a.coarse < b.coarse:
			return 1
		}
		return strings.Compare(a.ID, b.ID)
	})
	return out[:min(finalists, len(out))]
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// percentiles is each candidate's place in the set by value, 0 to 1, ties
// sharing their mean place.
func percentiles(cands []Candidate, value func(Candidate) float64) []float64 {
	n := len(cands)
	out := make([]float64, n)
	if n <= 1 {
		for i := range out {
			out[i] = 1
		}
		return out
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(a, b int) int {
		va, vb := value(cands[a]), value(cands[b])
		switch {
		case va < vb:
			return -1
		case va > vb:
			return 1
		}
		return 0
	})
	for i := 0; i < n; {
		j := i
		for j+1 < n && value(cands[idx[j+1]]) == value(cands[idx[i]]) {
			j++
		}
		place := float64(i+j) / 2 / float64(n-1)
		for k := i; k <= j; k++ {
			out[idx[k]] = place
		}
		i = j + 1
	}
	return out
}

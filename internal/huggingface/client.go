package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"
	"time"
)

const baseURL = "https://huggingface.co"

type Client struct {
	httpClient *http.Client
	token      string
	// base is the Hub's origin. Only ever the real one outside tests.
	base string
}

func NewClient(token string) *Client {
	return &Client{
		httpClient: &http.Client{Timeout: 30 * time.Second},
		token:      token,
		base:       baseURL,
	}
}

// SetBaseURL points the client at a different origin, for tests.
func (c *Client) SetBaseURL(u string) {
	c.base = strings.TrimRight(u, "/")
}

func (c *Client) SetToken(token string) {
	c.token = token
}

// ModelSearchResult is a single model from a HF Hub search.
type ModelSearchResult struct {
	ID           string     `json:"id"`
	Author       string     `json:"author"`
	Downloads    int        `json:"downloads"`
	Likes        int        `json:"likes"`
	Tags         []string   `json:"tags"`
	License      string     `json:"license,omitempty"`
	Private      bool       `json:"private"`
	Gated        GatedField `json:"gated"`
	LastModified string     `json:"lastModified,omitempty"`
	QuantFormat  string     `json:"quant_format,omitempty"`

	// Config is the repo's config.json as the Hub reports it, requested with
	// &config=true. It carries quantization_config, which is what vLLM reads
	// and therefore the only trustworthy answer to "what format is this?".
	Config *ModelConfigMeta `json:"config,omitempty"`
	// Safetensors is the exact size, by dtype; nil when the repository
	// publishes no safetensors metadata.
	Safetensors *Safetensors `json:"safetensors,omitempty"`
	// SHA is the commit the result describes, so a later read of its files
	// sees the same snapshot.
	SHA string `json:"sha,omitempty"`
	// PipelineTag is the Hub's task for the repository: text-generation,
	// image-text-to-text, feature-extraction and so on.
	PipelineTag string `json:"pipeline_tag,omitempty"`
}

// ModelConfigMeta is the part of a repo's config.json the search needs.
type ModelConfigMeta struct {
	QuantizationConfig *QuantConfig `json:"quantization_config,omitempty"`
	// Architectures and ModelType are what the Hub reports from config.json:
	// enough to name a model's architecture before its config is fetched.
	Architectures []string `json:"architectures,omitempty"`
	ModelType     string   `json:"model_type,omitempty"`
}

// GatedField handles HF's gated field which can be bool (false) or string ("auto"/"manual").
type GatedField string

func (g *GatedField) UnmarshalJSON(data []byte) error {
	// Try string first
	var s string
	if json.Unmarshal(data, &s) == nil {
		*g = GatedField(s)
		return nil
	}
	// Try bool (false = not gated)
	var b bool
	if json.Unmarshal(data, &b) == nil {
		if b {
			*g = "auto"
		} else {
			*g = ""
		}
		return nil
	}
	*g = ""
	return nil
}

func (g GatedField) IsGated() bool {
	return g != "" && g != "false"
}

// ModelGroup groups search results by normalized base model name.
type ModelGroup struct {
	BaseName string
	Variants []ModelSearchResult
}

// Search queries HuggingFace for models compatible with vLLM.
//
// quantFilter is one of QuantFilterOptions' values, or "". When it names tags,
// they go into the query so the Hub does the filtering. That is the difference
// between a filter and a sieve: without it the caller can only narrow whatever
// 50 repos a broad query returned by download count, and searching "qwen" for
// MXFP4 found nothing — not because there are none, but because none of them
// are popular enough to reach that page.
//
// We don't filter by pipeline_tag because models may be tagged as
// text-generation, image-text-to-text, or other tags that vLLM supports.
// Instead we filter by library -- see searchLibraries -- and sort by
// downloads.
func (c *Client) Search(ctx context.Context, query, quantFilter string) ([]ModelSearchResult, error) {
	tags := QuantFilterTags(quantFilter)
	if len(tags) == 0 {
		return c.search(ctx, query, "")
	}

	// The Hub ANDs repeated filter parameters, so a bucket covering more than
	// one tag needs one request per tag, merged. First occurrence wins, which
	// keeps each request's download ordering.
	seen := make(map[string]bool)
	var merged []ModelSearchResult
	var firstErr error
	for _, tag := range tags {
		batch, err := c.search(ctx, query, tag)
		if err != nil {
			// One tag failing should not empty the whole bucket.
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		for _, r := range batch {
			if seen[r.ID] {
				continue
			}
			seen[r.ID] = true
			merged = append(merged, r)
		}
	}
	if len(merged) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return merged, nil
}

// searchLibraries are the Hub libraries a servable repo is tagged with. A repo
// published for vLLM may carry only "vllm": mistralai/Mistral-Small-3.2-24B-
// Instruct-2506 does, and searching transformers alone never found it. The
// Hub ANDs repeated filters, so each is its own request, merged.
var searchLibraries = []string{"transformers", "vllm"}

// search runs one Hub query per library, optionally narrowed to a tag, and
// merges them by downloads.
func (c *Client) search(ctx context.Context, query, tag string) ([]ModelSearchResult, error) {
	seen := map[string]bool{}
	var raw []ModelSearchResult
	var firstErr error
	for _, lib := range searchLibraries {
		batch, err := c.searchLibrary(ctx, query, lib, tag)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		for _, r := range batch {
			if !seen[r.ID] {
				seen[r.ID] = true
				raw = append(raw, r)
			}
		}
	}
	if len(raw) == 0 && firstErr != nil {
		return nil, firstErr
	}
	sort.SliceStable(raw, func(i, j int) bool { return raw[i].Downloads > raw[j].Downloads })

	// Filter out GGUF-only repos (we serve safetensors/AWQ/GPTQ, not GGUF)
	var results []ModelSearchResult
	for _, r := range raw {
		if isGGUFOnly(r.ID, r.Tags) {
			continue
		}
		r.QuantFormat = DetectQuantFormat(r.ID, r.Tags, r.Config)
		results = append(results, r)
	}

	return results, nil
}

// searchExpand asks for each field a search result uses. With any expand[]
// the Hub returns only the fields expanded -- config=true is then ignored,
// and tags and downloads go with it -- so asking for safetensors means asking
// for everything else by name too. config is what DetectQuantFormat reads.
const searchExpand = "&expand[]=author&expand[]=downloads&expand[]=likes&expand[]=tags&expand[]=gated" +
	"&expand[]=private&expand[]=lastModified&expand[]=createdAt&expand[]=library_name&expand[]=pipeline_tag" +
	"&expand[]=config&expand[]=safetensors"

// CandidateQuery is one query for the recommendation feed's pool: one
// library, one quantization tag or none, one order.
type CandidateQuery struct {
	// Author, when set, asks for one publisher's repositories in any library,
	// in place of a library and tag.
	Author  string
	Library string // "transformers" or "vllm"
	Tag     string // a Hub quant tag; empty for the unquantized bucket
	Sort    string // "downloads" or "lastModified"
	Limit   int
}

// Candidates runs one pool query. The policy -- which queries, how many,
// what to keep -- is the caller's; this asks and tidies as Search does: no
// GGUF-only repositories, and each with its format badge.
func (c *Client) Candidates(ctx context.Context, q CandidateQuery) ([]ModelSearchResult, error) {
	if q.Limit <= 0 {
		q.Limit = 50
	}
	if q.Sort == "" {
		q.Sort = "downloads"
	}
	if q.Library == "" {
		q.Library = "transformers"
	}
	u := fmt.Sprintf("%s/models?filter=%s&sort=%s&direction=-1&limit=%d%s&expand[]=sha",
		c.apiURL(), url.QueryEscape(q.Library), url.QueryEscape(q.Sort), q.Limit, searchExpand)
	if q.Author != "" {
		u = fmt.Sprintf("%s/models?author=%s&sort=%s&direction=-1&limit=%d%s&expand[]=sha",
			c.apiURL(), url.QueryEscape(q.Author), url.QueryEscape(q.Sort), q.Limit, searchExpand)
	}
	if q.Tag != "" {
		u += "&filter=" + url.QueryEscape(q.Tag)
	}
	var raw []ModelSearchResult
	if err := c.getJSON(ctx, u, &raw); err != nil {
		return nil, fmt.Errorf("candidates: %w", err)
	}
	var out []ModelSearchResult
	for _, r := range raw {
		if isGGUFOnly(r.ID, r.Tags) {
			continue
		}
		r.QuantFormat = DetectQuantFormat(r.ID, r.Tags, r.Config)
		out = append(out, r)
	}
	return out, nil
}

// FetchConfigJSON is a repository's config.json at a revision, as bytes, for
// a caller that parses it with the same code as a local one.
func (c *Client) FetchConfigJSON(ctx context.Context, modelID, revision string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fileURL(c.base, modelID, revision, "config.json"), nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("config.json for %s: HTTP %d", modelID, resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 8<<20))
}

// searchLibrary runs one Hub query for repos of one library.
func (c *Client) searchLibrary(ctx context.Context, query, library, tag string) ([]ModelSearchResult, error) {
	// config=true returns each repo's config.json inline. Without it the only
	// clues to a model's format are its tags and its name, and the name is
	// wrong often enough to matter: "…-AWQ-W4A16" repos are usually
	// compressed-tensors, and were being labelled AWQ.
	u := fmt.Sprintf("%s/models?search=%s&filter=%s&sort=downloads&direction=-1&limit=50%s",
		c.apiURL(), url.QueryEscape(query), url.QueryEscape(library), searchExpand)
	if tag != "" {
		u += "&filter=" + url.QueryEscape(tag)
	}
	var raw []ModelSearchResult
	if err := c.getJSON(ctx, u, &raw); err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	return raw, nil
}

// GroupResults groups search results by author + normalized base model name.
// Only repos from the same author are grouped together, so third-party
// repacks don't appear as variants of the original model.
func GroupResults(results []ModelSearchResult) []ModelGroup {
	groups := make(map[string]*ModelGroup)
	var order []string

	for _, r := range results {
		author := r.Author
		if author == "" {
			// Extract author from model ID (e.g., "Qwen/Qwen3.5-9B" -> "Qwen")
			if parts := strings.SplitN(r.ID, "/", 2); len(parts) == 2 {
				author = parts[0]
			}
		}
		key := author + "/" + normalizeBaseName(r.ID)
		if g, ok := groups[key]; ok {
			g.Variants = append(g.Variants, r)
		} else {
			groups[key] = &ModelGroup{
				BaseName: key,
				Variants: []ModelSearchResult{r},
			}
			order = append(order, key)
		}
	}

	out := make([]ModelGroup, 0, len(order))
	for _, key := range order {
		out = append(out, *groups[key])
	}
	return out
}

// ModelDetail holds detailed info about a specific model repo.
type ModelDetail struct {
	ID string `json:"id"`
	// Revision is the commit the file list was read at. Downloading from it
	// rather than from main is what keeps a repo that changes mid-transfer
	// from handing over half of one snapshot and half of the next.
	Revision     string            `json:"revision,omitempty"`
	Author       string            `json:"author"`
	Tags         []string          `json:"tags"`
	Gated        GatedField        `json:"gated"`
	Files        []ModelFile       `json:"files"`
	TotalSize    int64             `json:"total_size"`
	Architecture string            `json:"architecture,omitempty"`
	QuantFormat  string            `json:"quant_format,omitempty"`
	QuantConfig  *QuantizationInfo `json:"quant_config,omitempty"`
	// WeightsBytes is what the weights occupy, exactly: the weight files a
	// download takes, by their sizes in the file tree. WeightsKnown is false
	// when the tree lists no weight file.
	WeightsBytes int64        `json:"weights_bytes,omitempty"`
	WeightsKnown bool         `json:"weights_known"`
	Safetensors  *Safetensors `json:"safetensors,omitempty"`
	MaxContext   int          `json:"max_context,omitempty"`
	IsGGUFRepo   bool         `json:"is_gguf_repo"`
}

type ModelFile struct {
	Filename   string `json:"filename"`
	Size       int64  `json:"size"`
	Category   string `json:"category"`
	IsRequired bool   `json:"is_required"`
	// OID is the git blob id and SHA256 the LFS object id, as the Hub's tree
	// listing gives them. A file has one or the other: weights are LFS, small
	// text files are plain blobs.
	OID    string `json:"oid,omitempty"`
	SHA256 string `json:"sha256,omitempty"`
}

// Identity names the exact content of a file, or "" when the Hub did not say.
// Two files with the same identity are byte-for-byte the same, which a name
// and a size cannot promise: repacked shards routinely keep both.
func (f ModelFile) Identity() string {
	switch {
	case f.SHA256 != "":
		return identitySHA256 + f.SHA256
	case f.OID != "":
		return identityGit + f.OID
	}
	return ""
}

type QuantizationInfo struct {
	Method    string `json:"method"`
	Bits      int    `json:"bits"`
	GroupSize int    `json:"group_size"`
}

// GetModel fetches model details including file listing and config.
func (c *Client) GetModel(ctx context.Context, modelID string) (*ModelDetail, error) {
	detail := &ModelDetail{ID: modelID}

	// Fetch model metadata
	var meta struct {
		ID          string       `json:"id"`
		Author      string       `json:"author"`
		Tags        []string     `json:"tags"`
		Gated       GatedField   `json:"gated"`
		SHA         string       `json:"sha"`
		Safetensors *Safetensors `json:"safetensors"`
	}
	// Each field by name: with any expand[] the Hub returns only those.
	metaURL := fmt.Sprintf("%s/models/%s?expand[]=author&expand[]=tags&expand[]=gated&expand[]=sha&expand[]=safetensors",
		c.apiURL(), modelID)
	if err := c.getJSON(ctx, metaURL, &meta); err != nil {
		return nil, fmt.Errorf("get model: %w", err)
	}
	detail.Author = meta.Author
	detail.Tags = meta.Tags
	detail.Gated = meta.Gated
	detail.Revision = meta.SHA
	detail.Safetensors = meta.Safetensors

	// Fetch file tree
	files, err := c.listFiles(ctx, modelID, meta.SHA)
	if err != nil {
		return nil, fmt.Errorf("get tree: %w", err)
	}
	detail.Files = files

	var hasGGUF, hasSafetensors bool
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			hasGGUF = true
		}
		if strings.HasSuffix(strings.ToLower(f.Filename), ".safetensors") {
			hasSafetensors = true
		}
	}

	detail.IsGGUFRepo = hasGGUF && !hasSafetensors

	// What a download takes, and of it the weights. Through the same filter a
	// download uses: summing the tree counted both copies of a repo that holds
	// its weights twice, as Mistral's do -- 96 GB for a 48 GB download.
	for _, f := range DownloadableFiles(detail.Files) {
		detail.TotalSize += f.Size
		if f.Category == "weight" {
			detail.WeightsBytes += f.Size
			detail.WeightsKnown = true
		}
	}

	// Fetch config.json for architecture info
	cfg, err := c.fetchConfig(ctx, modelID)
	if err == nil && cfg != nil {
		if len(cfg.Architectures) > 0 {
			detail.Architecture = cfg.Architectures[0]
		}
		detail.MaxContext = cfg.MaxPositionEmbeddings
		if cfg.QuantizationConfig != nil {
			detail.QuantConfig = &QuantizationInfo{
				Method:    cfg.QuantizationConfig.QuantMethod,
				Bits:      cfg.QuantizationConfig.Bits,
				GroupSize: cfg.QuantizationConfig.GroupSize,
			}
		}
		// Same mapping as the search row, so a model does not change format
		// between the list and the panel that opens under it.
		detail.QuantFormat = DetectQuantFormat(modelID, detail.Tags,
			&ModelConfigMeta{QuantizationConfig: cfg.QuantizationConfig})
	} else {
		detail.QuantFormat = DetectQuantFormat(modelID, detail.Tags, nil)
	}

	return detail, nil
}

// GetFiles lists the files worth downloading from a repo, and the commit they
// were listed at. An empty revision means wherever main points now; passing
// one back in re-reads exactly the listing an earlier call saw.
func (c *Client) GetFiles(ctx context.Context, modelID, revision string) (string, []ModelFile, error) {
	if revision == "" {
		var meta struct {
			SHA string `json:"sha"`
		}
		metaURL := fmt.Sprintf("%s/models/%s", c.apiURL(), modelID)
		if err := c.getJSON(ctx, metaURL, &meta); err != nil {
			return "", nil, fmt.Errorf("get model: %w", err)
		}
		revision = meta.SHA
	}
	files, err := c.listFiles(ctx, modelID, revision)
	if err != nil {
		return "", nil, fmt.Errorf("get tree: %w", err)
	}
	return revision, DownloadableFiles(files), nil
}

// listFiles is a repo's tree at revision, less the files categorizeFile says
// nothing here wants.
func (c *Client) listFiles(ctx context.Context, modelID, revision string) ([]ModelFile, error) {
	tree, err := c.fetchTree(ctx, modelID, revision)
	if err != nil {
		return nil, err
	}
	var files []ModelFile
	for _, entry := range tree {
		cat, required := categorizeFile(entry.Path)
		if cat == "skip" {
			continue
		}
		f := ModelFile{
			Filename:   entry.Path,
			Size:       entry.Size,
			Category:   cat,
			IsRequired: required,
		}
		// An LFS file's blob id is the id of its pointer, which says nothing
		// about the weights; the LFS id is the hash of the file itself.
		if entry.LFS != nil && entry.LFS.OID != "" {
			f.SHA256 = entry.LFS.OID
		} else {
			f.OID = entry.OID
		}
		files = append(files, f)
	}
	return files, nil
}

// DownloadableFiles is the subset of a repo's files a download fetches: no
// GGUF, which vLLM does not serve; no .bin weights when the same weights are
// there as safetensors; and, for a repo with its weights in both Mistral's
// consolidated layout and Hugging Face shards, only one of the two.
//
// Which one depends on which layout is complete. mistralai/Mistral-Small-3.2-
// 24B-Instruct-2506 carries both, 48 GB each, but its shards come with no
// tokenizer and no image processor config: loaded as Hugging Face weights it
// failed to start, "Can't load image processor". Its Mistral layout --
// consolidated.safetensors, params.json, tekken.json -- is the one it is
// served from, with --config_format and --load_format mistral. So the
// consolidated copy is kept when the repo has params.json and no Hugging Face
// tokenizer, and the shards otherwise.
func DownloadableFiles(files []ModelFile) []ModelFile {
	hasSafetensors, hasShardIndex, hasConsolidated := false, false, false
	hasParams, hasHFTokenizer := false, false
	for _, f := range files {
		lower := strings.ToLower(f.Filename)
		base := path.Base(lower)
		if strings.HasSuffix(lower, ".safetensors") {
			hasSafetensors = true
		}
		switch base {
		case "model.safetensors.index.json":
			hasShardIndex = true
		case "params.json":
			hasParams = true
		case "tokenizer.json", "tokenizer.model":
			hasHFTokenizer = true
		}
		if IsConsolidatedWeights(f.Filename) {
			hasConsolidated = true
		}
	}
	bothLayouts := hasShardIndex && hasConsolidated
	mistralLayout := bothLayouts && hasParams && !hasHFTokenizer

	var out []ModelFile
	for _, f := range files {
		lower := strings.ToLower(f.Filename)
		if f.Category == "skip" {
			continue
		}
		if hasSafetensors && f.Category == "weight" && strings.HasSuffix(lower, ".bin") {
			continue
		}
		if strings.HasSuffix(lower, ".gguf") {
			continue
		}
		switch {
		case bothLayouts && !mistralLayout && IsConsolidatedWeights(f.Filename):
			continue
		case mistralLayout && (isShard(f.Filename) || path.Base(lower) == "model.safetensors.index.json"):
			continue
		}
		out = append(out, f)
	}
	return out
}

// isShard reports a Hugging Face weight shard: a .safetensors file that is not
// Mistral's consolidated layout.
func isShard(name string) bool {
	return strings.HasSuffix(strings.ToLower(name), ".safetensors") && !IsConsolidatedWeights(name)
}

// IsConsolidatedWeights reports Mistral's single-file weight layout,
// consolidated.safetensors or its numbered parts.
func IsConsolidatedWeights(name string) bool {
	base := path.Base(strings.ToLower(name))
	return strings.HasPrefix(base, "consolidated") && strings.HasSuffix(base, ".safetensors")
}

type treeEntry struct {
	Type string `json:"type"` // "file" or "directory"
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"`
	LFS  *struct {
		OID string `json:"oid"`
	} `json:"lfs"`
}

func (c *Client) fetchTree(ctx context.Context, modelID, revision string) ([]treeEntry, error) {
	if revision == "" {
		revision = "main"
	}
	var all []treeEntry
	cursor := ""
	for {
		u := fmt.Sprintf("%s/models/%s/tree/%s?recursive=true", c.apiURL(), modelID, url.PathEscape(revision))
		if cursor != "" {
			u += "&cursor=" + url.QueryEscape(cursor)
		}
		var entries []treeEntry
		if err := c.getJSON(ctx, u, &entries); err != nil {
			return nil, err
		}
		for _, e := range entries {
			if e.Type == "file" {
				all = append(all, e)
			}
		}
		if len(entries) < 1000 {
			break
		}
		cursor = entries[len(entries)-1].Path
	}
	return all, nil
}

type modelConfig struct {
	Architectures         []string     `json:"architectures"`
	ModelType             string       `json:"model_type"`
	HiddenSize            int          `json:"hidden_size"`
	NumHiddenLayers       int          `json:"num_hidden_layers"`
	NumAttentionHeads     int          `json:"num_attention_heads"`
	NumKeyValueHeads      int          `json:"num_key_value_heads"`
	IntermediateSize      int          `json:"intermediate_size"`
	VocabSize             int          `json:"vocab_size"`
	MaxPositionEmbeddings int          `json:"max_position_embeddings"`
	QuantizationConfig    *QuantConfig `json:"quantization_config,omitempty"`
}

func (c *Client) fetchConfig(ctx context.Context, modelID string) (*modelConfig, error) {
	u := fmt.Sprintf("%s/%s/resolve/main/config.json", c.base, modelID)
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("config.json: HTTP %d", resp.StatusCode)
	}

	var cfg modelConfig
	if err := json.NewDecoder(resp.Body).Decode(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (c *Client) getJSON(ctx context.Context, u string, v any) error {
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 403 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("access denied (HTTP 403) — is this a gated model? %s", strings.TrimSpace(string(body)))
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

// ModelCard returns a repository's model card, its README.md, as written.
// A repository with no card returns "" and no error: most of what reads a card
// can proceed without one.
func (c *Client) ModelCard(ctx context.Context, modelID string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("%s/%s/raw/main/README.md", c.base, modelID), nil)
	if err != nil {
		return "", err
	}
	c.setAuth(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return "", nil
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("model card for %s: HTTP %d", modelID, resp.StatusCode)
	}
	// A card is text; a few hundred kilobytes is already an unusually long one.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// BaseModel returns the repository a model says it was made from -- the
// base_model in its card's metadata -- or "" when it names none or the Hub
// cannot be asked. The field is a string in some repositories and a list in
// others; the first entry of a list is taken.
func (c *Client) BaseModel(ctx context.Context, modelID string) string {
	var meta struct {
		CardData struct {
			BaseModel json.RawMessage `json:"base_model"`
		} `json:"cardData"`
	}
	u := fmt.Sprintf("%s/models/%s?expand[]=cardData", c.apiURL(), modelID)
	if err := c.getJSON(ctx, u, &meta); err != nil {
		return ""
	}
	raw := meta.CardData.BaseModel
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return one
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil && len(many) > 0 {
		return many[0]
	}
	return ""
}

func (c *Client) setAuth(req *http.Request) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
}

func (c *Client) apiURL() string {
	return c.base + "/api"
}

// DownloadURL returns the direct download URL for a file in a model repo, as
// it stood at revision. An empty revision follows main.
func DownloadURL(modelID, revision, filename string) string {
	return fileURL(baseURL, modelID, revision, filename)
}

func fileURL(base, modelID, revision, filename string) string {
	if revision == "" {
		revision = "main"
	}
	// Escaped a segment at a time: the slashes are the path, but a space or a
	// '#' in a file name is not.
	segments := strings.Split(filename, "/")
	for i, seg := range segments {
		segments[i] = url.PathEscape(seg)
	}
	return fmt.Sprintf("%s/%s/resolve/%s/%s", base, modelID, url.PathEscape(revision), strings.Join(segments, "/"))
}

// isGGUFOnly returns true if the model is a GGUF-only repo.
func isGGUFOnly(modelID string, tags []string) bool {
	lower := strings.ToLower(modelID)
	if strings.HasSuffix(lower, "-gguf") || strings.HasSuffix(lower, "_gguf") {
		return true
	}
	for _, t := range tags {
		if strings.ToLower(t) == "gguf" {
			// Has gguf tag -- check if it also has safetensors (mixed repo)
			for _, t2 := range tags {
				if strings.ToLower(t2) == "safetensors" {
					return false // mixed repo, keep it
				}
			}
			return true // gguf-only
		}
	}
	return false
}

// normalizeBaseName strips quantization suffixes for grouping.
func normalizeBaseName(modelID string) string {
	name := path.Base(modelID)
	suffixes := []string{
		"-AWQ", "-awq", "-GPTQ", "-gptq", "-GGUF", "-gguf",
		"-FP8", "-fp8", "-bnb-4bit", "-bnb-8bit", "-4bit", "-8bit",
		"-Marlin", "-marlin",
		"-MXFP4", "-mxfp4", "-NVFP4", "-nvfp4",
		"-W4A16", "-w4a16", "-W8A8", "-w8a8", "-W8A16", "-w8a16",
		"-INT4", "-int4", "-INT8", "-int8",
		"-Quark", "-quark", "-AutoRound", "-autoround", "-auto-round",
	}
	for _, s := range suffixes {
		name = strings.TrimSuffix(name, s)
	}
	return strings.ToLower(name)
}

// categorizeFile determines the category and whether a file must be downloaded.
func categorizeFile(filename string) (category string, required bool) {
	lower := strings.ToLower(filename)
	base := path.Base(lower)

	switch {
	case strings.HasSuffix(lower, ".md"),
		strings.HasSuffix(lower, ".txt"),
		base == ".gitattributes",
		strings.Contains(lower, "training"),
		strings.Contains(lower, "optimizer"),
		strings.HasSuffix(lower, ".h5"),
		strings.HasSuffix(lower, ".ot"),
		strings.HasSuffix(lower, ".msgpack"),
		strings.HasPrefix(base, "."):
		return "skip", false
	}

	switch {
	case strings.HasSuffix(lower, ".safetensors"):
		return "weight", true
	case strings.HasSuffix(lower, ".gguf"):
		return "weight", true
	case base == "pytorch_model.bin" || strings.HasPrefix(base, "pytorch_model-"):
		return "weight", true
	case strings.HasSuffix(lower, ".bin") && strings.Contains(lower, "model"):
		return "weight", true
	}

	switch base {
	case "config.json":
		return "config", true
	case "tokenizer.json":
		return "tokenizer", true
	case "tokenizer_config.json":
		return "tokenizer", true
	case "generation_config.json":
		return "config", true
	case "special_tokens_map.json":
		return "tokenizer", true
	case "tokenizer.model":
		return "tokenizer", true
	case "model.safetensors.index.json":
		return "config", true
	case "preprocessor_config.json":
		return "config", false
	case "chat_template.json":
		return "config", false
	case "added_tokens.json":
		return "tokenizer", false
	}

	return "other", false
}

// FormatBytes formats a byte count as a human-readable string.
func FormatBytes(b int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case b >= GB:
		return fmt.Sprintf("%.1f GB", float64(b)/float64(GB))
	case b >= MB:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(MB))
	case b >= KB:
		return fmt.Sprintf("%.1f KB", float64(b)/float64(KB))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
)

const baseURL = "https://huggingface.co"
const apiURL = baseURL + "/api"

type Client struct {
	httpClient *http.Client
	token      string
}

func NewClient(token string) *Client {
	return &Client{
		httpClient: &http.Client{Timeout: 30 * time.Second},
		token:      token,
	}
}

func (c *Client) SetToken(token string) {
	c.token = token
}

// ModelSearchResult is a single model from a HF Hub search.
type ModelSearchResult struct {
	ID           string   `json:"id"`
	Author       string   `json:"author"`
	Downloads    int      `json:"downloads"`
	Likes        int      `json:"likes"`
	Tags         []string `json:"tags"`
	License      string   `json:"license,omitempty"`
	Private      bool     `json:"private"`
	Gated        string   `json:"gated,omitempty"`
	LastModified string   `json:"lastModified,omitempty"`
	QuantFormat  string   `json:"quant_format,omitempty"`
}

// ModelGroup groups search results by normalized base model name.
type ModelGroup struct {
	BaseName string
	Variants []ModelSearchResult
}

// Search queries HuggingFace for text-generation models.
func (c *Client) Search(ctx context.Context, query string) ([]ModelSearchResult, error) {
	u := fmt.Sprintf("%s/models?search=%s&filter=text-generation&sort=downloads&direction=-1&limit=50",
		apiURL, url.QueryEscape(query))

	var results []ModelSearchResult
	if err := c.getJSON(ctx, u, &results); err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	for i := range results {
		results[i].QuantFormat = detectQuantFormat(results[i].ID, results[i].Tags)
	}

	return results, nil
}

// GroupResults groups search results by normalized base model name.
func GroupResults(results []ModelSearchResult) []ModelGroup {
	groups := make(map[string]*ModelGroup)
	var order []string

	for _, r := range results {
		base := normalizeBaseName(r.ID)
		if g, ok := groups[base]; ok {
			g.Variants = append(g.Variants, r)
		} else {
			groups[base] = &ModelGroup{
				BaseName: base,
				Variants: []ModelSearchResult{r},
			}
			order = append(order, base)
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
	ID             string           `json:"id"`
	Author         string           `json:"author"`
	Tags           []string         `json:"tags"`
	Gated          string           `json:"gated,omitempty"`
	Files          []ModelFile      `json:"files"`
	TotalSize      int64            `json:"total_size"`
	Architecture   string           `json:"architecture,omitempty"`
	QuantFormat    string           `json:"quant_format,omitempty"`
	QuantConfig    *QuantizationInfo `json:"quant_config,omitempty"`
	ParameterCount int64            `json:"parameter_count,omitempty"`
	VRAMEstGB      float64          `json:"vram_est_gb,omitempty"`
	MaxContext     int              `json:"max_context,omitempty"`
	IsGGUFRepo     bool            `json:"is_gguf_repo"`
}

type ModelFile struct {
	Filename   string `json:"filename"`
	Size       int64  `json:"size"`
	Category   string `json:"category"`
	IsRequired bool   `json:"is_required"`
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
		ID     string   `json:"id"`
		Author string   `json:"author"`
		Tags   []string `json:"tags"`
		Gated  string   `json:"gated"`
	}
	metaURL := fmt.Sprintf("%s/models/%s", apiURL, modelID)
	if err := c.getJSON(ctx, metaURL, &meta); err != nil {
		return nil, fmt.Errorf("get model: %w", err)
	}
	detail.Author = meta.Author
	detail.Tags = meta.Tags
	detail.Gated = meta.Gated

	// Fetch file tree
	tree, err := c.fetchTree(ctx, modelID)
	if err != nil {
		return nil, fmt.Errorf("get tree: %w", err)
	}

	var hasGGUF, hasSafetensors bool
	for _, entry := range tree {
		cat, required := categorizeFile(entry.Path)
		if cat == "skip" {
			continue
		}
		if strings.HasSuffix(strings.ToLower(entry.Path), ".gguf") {
			hasGGUF = true
		}
		if strings.HasSuffix(strings.ToLower(entry.Path), ".safetensors") {
			hasSafetensors = true
		}
		detail.Files = append(detail.Files, ModelFile{
			Filename:   entry.Path,
			Size:       entry.Size,
			Category:   cat,
			IsRequired: required,
		})
	}

	detail.IsGGUFRepo = hasGGUF && !hasSafetensors

	// Compute total size of downloadable files
	for _, f := range detail.Files {
		// For mixed repos, skip .bin weights if safetensors exist
		if hasSafetensors && f.Category == "weight" &&
			strings.HasSuffix(strings.ToLower(f.Filename), ".bin") {
			continue
		}
		detail.TotalSize += f.Size
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
			detail.QuantFormat = strings.ToUpper(cfg.QuantizationConfig.QuantMethod)
		}
		detail.ParameterCount = estimateParamCount(cfg)
	}

	if detail.QuantFormat == "" {
		detail.QuantFormat = detectQuantFormat(modelID, detail.Tags)
	}

	// VRAM estimation
	if detail.ParameterCount > 0 {
		bits := quantBits(detail.QuantFormat)
		detail.VRAMEstGB = estimateVRAM(detail.ParameterCount, bits, 4096)
	} else if detail.TotalSize > 0 {
		detail.VRAMEstGB = float64(detail.TotalSize) * 1.1 / (1024 * 1024 * 1024)
	}

	return detail, nil
}

type treeEntry struct {
	Type string `json:"type"` // "file" or "directory"
	Path string `json:"path"`
	Size int64  `json:"size"`
}

func (c *Client) fetchTree(ctx context.Context, modelID string) ([]treeEntry, error) {
	var all []treeEntry
	cursor := ""
	for {
		u := fmt.Sprintf("%s/models/%s/tree/main?recursive=true", apiURL, modelID)
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
	QuantizationConfig    *quantConfig `json:"quantization_config,omitempty"`
}

type quantConfig struct {
	QuantMethod string `json:"quant_method"`
	Bits        int    `json:"bits"`
	GroupSize   int    `json:"group_size"`
}

func (c *Client) fetchConfig(ctx context.Context, modelID string) (*modelConfig, error) {
	u := fmt.Sprintf("%s/%s/resolve/main/config.json", baseURL, modelID)
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
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("access denied (HTTP 403) — is this a gated model? %s", string(body))
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

func (c *Client) setAuth(req *http.Request) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
}

// DownloadURL returns the direct download URL for a file in a model repo.
func DownloadURL(modelID, filename string) string {
	return fmt.Sprintf("%s/%s/resolve/main/%s", baseURL, modelID, filename)
}

// detectQuantFormat detects quantization format from model ID and tags.
func detectQuantFormat(modelID string, tags []string) string {
	// Check tags first
	for _, t := range tags {
		switch strings.ToLower(t) {
		case "gptq":
			return "GPTQ"
		case "awq":
			return "AWQ"
		case "gguf":
			return "GGUF"
		}
	}
	// Check model ID suffix
	name := strings.ToLower(path.Base(modelID))
	switch {
	case strings.Contains(name, "-awq"):
		return "AWQ"
	case strings.Contains(name, "-gptq"):
		return "GPTQ"
	case strings.Contains(name, "-gguf"):
		return "GGUF"
	case strings.Contains(name, "-fp8"):
		return "FP8"
	case strings.Contains(name, "-bnb-4bit"):
		return "BnB-4bit"
	case strings.Contains(name, "-bnb-8bit"):
		return "BnB-8bit"
	}
	return ""
}

// normalizeBaseName strips quantization suffixes for grouping.
func normalizeBaseName(modelID string) string {
	name := path.Base(modelID)
	suffixes := []string{
		"-AWQ", "-awq", "-GPTQ", "-gptq", "-GGUF", "-gguf",
		"-FP8", "-fp8", "-bnb-4bit", "-bnb-8bit", "-4bit", "-8bit",
		"-Marlin", "-marlin",
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

// estimateParamCount estimates parameter count from model config.
func estimateParamCount(cfg *modelConfig) int64 {
	if cfg.HiddenSize == 0 || cfg.NumHiddenLayers == 0 {
		return 0
	}
	h := int64(cfg.HiddenSize)
	l := int64(cfg.NumHiddenLayers)
	v := int64(cfg.VocabSize)
	inter := int64(cfg.IntermediateSize)
	if inter == 0 {
		inter = 4 * h // default MLP ratio
	}

	// Rough formula: embedding + (attn + mlp) * layers + lm_head
	embedding := v * h
	attnPerLayer := 4 * h * h // Q, K, V, O projections
	mlpPerLayer := 3 * h * inter // gate, up, down
	lmHead := v * h

	return embedding + (attnPerLayer+mlpPerLayer)*l + lmHead
}

// quantBits returns approximate bits-per-parameter for a quant format.
func quantBits(format string) int {
	switch strings.ToUpper(format) {
	case "AWQ", "GPTQ":
		return 4
	case "FP8":
		return 8
	case "BNB-4BIT":
		return 4
	case "BNB-8BIT":
		return 8
	case "GGUF":
		return 5 // average estimate
	default:
		return 16 // FP16/BF16
	}
}

// estimateVRAM estimates VRAM in GB from parameter count and quant bits.
func estimateVRAM(paramCount int64, bits int, contextLen int) float64 {
	if paramCount == 0 || bits == 0 {
		return 0
	}
	weightBytes := float64(paramCount) * float64(bits) / 8.0
	overhead := 1.2
	if contextLen > 8192 {
		overhead = 1.3
	}
	return weightBytes * overhead / (1024 * 1024 * 1024)
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

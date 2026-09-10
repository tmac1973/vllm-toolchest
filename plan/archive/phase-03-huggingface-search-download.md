# Phase 3: HuggingFace Model Search & Download

## Goal

Build the HuggingFace Hub integration: search for models, detect quantization variants, download model files with concurrent transfers and resumable progress, and register downloaded models in the local inventory. The browser UI provides a search page with quantization-aware results, file detail panels, and SSE-driven download progress.

---

## 1. HuggingFace Hub API Integration

### Package: `internal/huggingface/`

Two files, same structure as llama-toolchest:
- `client.go` — API client for HF Hub (search, model details, file listing)
- `downloader.go` — concurrent resumable file downloads with progress fan-out

### What changes from llama-toolchest's HF client

llama-toolchest's client is GGUF-centric: it searches with `filter=gguf`, only shows `.gguf` files, groups GGUF shards, and estimates VRAM from GGUF file size. vllm-toolchest needs to handle the full range of model formats that vLLM supports.

| Aspect | llama-toolchest | vllm-toolchest |
|--------|-----------------|----------------|
| Search filter | `filter=gguf` | `filter=text-generation` (broader) |
| File filter | Only `.gguf` files | safetensors, bin, gguf, config files |
| Size source | Tree API for GGUF | Tree API for all files |
| VRAM estimate | `file_size * 1.1` | Parameter count from config.json + quant bits |
| Shard grouping | GGUF shard pattern (`-00001-of-00005.gguf`) | safetensors shard pattern (`model-00001-of-00005.safetensors`) |
| Quant detection | Parse from GGUF filename | Parse from repo name, config.json `quantization_config`, file inspection |
| Download unit | Single GGUF file (or shard set) | Entire model directory (many files) |

### Client struct

```go
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
```

Same as llama-toolchest.

---

## 2. Search Endpoint

### `Client.Search(ctx, query) ([]ModelSearchResult, error)`

```
GET https://huggingface.co/api/models?search={query}&filter=text-generation&sort=downloads&direction=-1&limit=50
```

#### Why `filter=text-generation` instead of GGUF

vLLM serves models in many formats — safetensors (FP16, BF16), AWQ, GPTQ, FP8, GGUF, BitsAndBytes-quantized. Filtering by `gguf` would miss the majority of usable models. The `text-generation` pipeline tag captures all causal LMs.

#### Search result struct

```go
type ModelSearchResult struct {
    ID           string   `json:"id"`            // e.g. "meta-llama/Llama-3.1-8B-Instruct"
    Author       string   `json:"author"`        // e.g. "meta-llama"
    Downloads    int      `json:"downloads"`
    Likes        int      `json:"likes"`
    Tags         []string `json:"tags"`           // pipeline tags, library tags, quant tags
    License      string   `json:"license,omitempty"`
    Private      bool     `json:"private"`
    Gated        string   `json:"gated,omitempty"` // "auto", "manual", or "" (not gated)
    ModelIndex   string   `json:"model_index,omitempty"` // architecture info from model card
    LastModified string   `json:"lastModified,omitempty"`
}
```

New fields vs llama-toolchest: `Gated` (important for Llama, Mistral, etc.), `Private`, `LastModified`.

#### Detecting quantization info from search results

The HF API `tags` array contains useful metadata:
- `"gguf"` — model has GGUF files
- `"text-generation"` — causal language model
- `"gptq"` — GPTQ quantized
- `"awq"` — AWQ quantized
- `"transformers"` — standard HF Transformers format

Parse tags to pre-classify quant format in search results:

```go
func detectQuantFromTags(tags []string) string {
    for _, t := range tags {
        switch strings.ToLower(t) {
        case "gptq":  return "GPTQ"
        case "awq":   return "AWQ"
        case "gguf":  return "GGUF"
        }
    }
    return "" // unquantized or unknown
}
```

Also detect from model ID patterns:
- `*-AWQ` → AWQ
- `*-GPTQ*` → GPTQ
- `*-GGUF` → GGUF
- `*-fp8` → FP8
- `*-bnb-4bit` → BitsAndBytes 4-bit

---

## 3. Quantization Variant Detection

### The problem

A base model like `meta-llama/Llama-3.1-8B-Instruct` has many quantized versions from different authors:
- `meta-llama/Llama-3.1-8B-Instruct` (original FP16/BF16)
- `TechxGenus/Llama-3.1-8B-Instruct-AWQ`
- `TechxGenus/Llama-3.1-8B-Instruct-GPTQ`
- `bartowski/Llama-3.1-8B-Instruct-GGUF`
- `neuralmagic/Llama-3.1-8B-Instruct-FP8`
- `unsloth/Llama-3.1-8B-Instruct-bnb-4bit`

The search page should group these so the user sees "Llama-3.1-8B-Instruct" once with variant options underneath.

### Approach: Client-side grouping after search

After fetching search results, group by a normalized base model name:

```go
func normalizeBaseName(modelID string) string {
    name := path.Base(modelID) // strip org prefix
    // Strip common quant suffixes
    suffixes := []string{
        "-AWQ", "-awq",
        "-GPTQ", "-gptq",
        "-GGUF", "-gguf",
        "-FP8", "-fp8",
        "-bnb-4bit", "-bnb-8bit",
        "-4bit", "-8bit",
        "-Marlin",
    }
    for _, s := range suffixes {
        name = strings.TrimSuffix(name, s)
    }
    return strings.ToLower(name)
}
```

Group results into:

```go
type ModelGroup struct {
    BaseName string              // normalized name for grouping
    Variants []ModelSearchResult // all variants (original + quantized)
}
```

**Decision**: Do grouping on the server side in the Go handler, not in the HF client. The client returns flat results; the handler groups them for display. This keeps the client simple and testable.

**Edge case**: Some repos don't follow naming conventions (e.g., `TheBloke/Llama-2-7B-Chat-GPTQ` vs `meta-llama/Llama-2-7b-chat-hf`). The normalization won't always group perfectly. Accept this — it's a best-effort UX improvement, not a requirement for correctness.

---

## 4. Model Detail Fetching

### `Client.GetModel(ctx, modelID) (*ModelDetail, error)`

```
GET https://huggingface.co/api/models/{modelID}
```

#### Response parsing

The HF API returns a large JSON object. Parse selectively:

```go
type ModelDetail struct {
    ID           string            `json:"id"`
    Author       string            `json:"author"`
    Tags         []string          `json:"tags"`
    Files        []ModelFile       `json:"files"`        // populated from siblings + tree
    TotalSize    int64             `json:"total_size"`   // sum of all downloadable files
    Architecture string            `json:"architecture"` // from config.json
    QuantFormat  string            `json:"quant_format"` // detected: AWQ, GPTQ, FP8, GGUF, BnB, ""
    QuantConfig  *QuantizationInfo `json:"quant_config,omitempty"`
    ParameterCount int64           `json:"parameter_count,omitempty"` // from safetensors metadata or config
    VRAMEstGB    float64           `json:"vram_est_gb"`  // estimated VRAM needed
    Gated        string            `json:"gated"`
    CardData     *CardData         `json:"card_data,omitempty"`
}

type ModelFile struct {
    Filename  string `json:"filename"`
    Size      int64  `json:"size"`
    Category  string `json:"category"` // "weight", "config", "tokenizer", "other"
    IsRequired bool  `json:"is_required"` // must download for model to work
}

type QuantizationInfo struct {
    Method    string `json:"method"`     // "awq", "gptq", "fp8", "bnb", "marlin"
    Bits      int    `json:"bits"`       // 4, 8, etc.
    GroupSize int    `json:"group_size"` // 128, -1, etc.
    Dataset   string `json:"dataset"`    // calibration dataset
}

type CardData struct {
    License      string   `json:"license"`
    PipelineTag  string   `json:"pipeline_tag"`
    BaseModel    string   `json:"base_model"`
    Datasets     []string `json:"datasets"`
}
```

#### Architecture detection from config.json

Fetch `config.json` from the repo to determine model architecture:

```
GET https://huggingface.co/api/models/{modelID}/tree/main
→ find config.json in file list
GET https://huggingface.co/{modelID}/resolve/main/config.json
→ parse "architectures" field
```

```go
type modelConfig struct {
    Architectures    []string         `json:"architectures"`     // e.g. ["LlamaForCausalLM"]
    ModelType        string           `json:"model_type"`        // e.g. "llama"
    HiddenSize       int              `json:"hidden_size"`
    NumHiddenLayers  int              `json:"num_hidden_layers"`
    NumAttentionHeads int             `json:"num_attention_heads"`
    NumKeyValueHeads  int             `json:"num_key_value_heads"`
    MaxPositionEmbeddings int         `json:"max_position_embeddings"`
    QuantizationConfig *quantConfig   `json:"quantization_config,omitempty"`
}

type quantConfig struct {
    QuantMethod string `json:"quant_method"` // "awq", "gptq", "fp8"
    Bits        int    `json:"bits"`
    GroupSize   int    `json:"group_size"`
    Dataset     string `json:"dataset"`
}
```

The `quantization_config` field in `config.json` is the authoritative source for quant format:
- Present with `"quant_method": "awq"` → AWQ model
- Present with `"quant_method": "gptq"` → GPTQ model
- Present with `"quant_method": "fp8"` → FP8 model
- Absent → unquantized (FP16/BF16) or GGUF (check file extensions)

#### Quantization format detection priority

1. `config.json` → `quantization_config.quant_method` (most reliable)
2. File extensions: all `.gguf` → GGUF format
3. Tags on the model (from search result)
4. Model ID suffix heuristic

#### VRAM estimation

More accurate than llama-toolchest's `file_size * 1.1`:

```go
func estimateVRAM(paramCount int64, quantBits int, contextLen int) float64 {
    if paramCount == 0 || quantBits == 0 {
        return 0 // unknown
    }
    // Model weights
    weightBytes := float64(paramCount) * float64(quantBits) / 8.0
    // KV cache estimate (rough: 2 * n_layers * 2 * hidden_size * context_len * 2 bytes for FP16)
    // Simplified: ~10-20% overhead on top of weights for typical context lengths
    overhead := 1.2
    if contextLen > 8192 {
        overhead = 1.3 // more KV cache overhead for long context
    }
    return weightBytes * overhead / (1024 * 1024 * 1024)
}
```

Fallback for when param count is unknown: `total_weight_file_size * 1.1` (same as llama-toolchest).

#### File listing from tree API

```
GET https://huggingface.co/api/models/{modelID}/tree/main?recursive=true
```

Returns all files in the repo with paths and sizes. Categorize each file:

```go
func categorizeFile(filename string) (category string, required bool) {
    lower := strings.ToLower(filename)
    base := path.Base(lower)

    // Skip files
    switch {
    case strings.HasSuffix(lower, ".md"),
         strings.HasSuffix(lower, ".txt"),
         base == ".gitattributes",
         strings.Contains(lower, "training"),
         strings.Contains(lower, "optimizer"),
         strings.HasPrefix(base, "."):
        return "skip", false
    }

    // Weight files
    switch {
    case strings.HasSuffix(lower, ".safetensors"):
        return "weight", true
    case strings.HasSuffix(lower, ".bin") && strings.Contains(lower, "model"):
        return "weight", true
    case strings.HasSuffix(lower, ".gguf"):
        return "weight", true
    }

    // Config files
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
    case "preprocessor_config.json":
        return "config", false // only needed for vision models
    case "chat_template.json":
        return "config", false // optional, vLLM can use tokenizer_config.json
    case "added_tokens.json":
        return "tokenizer", false
    }

    return "other", false
}
```

#### Files to download vs skip

**Always download (required)**:
- `*.safetensors` — model weight shards (primary format for vLLM)
- `*.bin` (pytorch model weights) — fallback if no safetensors
- `*.gguf` — for GGUF models
- `config.json` — model architecture configuration
- `tokenizer.json` — fast tokenizer
- `tokenizer_config.json` — tokenizer settings, chat template
- `generation_config.json` — default generation parameters
- `special_tokens_map.json` — token mappings

**Download if present (optional)**:
- `tokenizer.model` — SentencePiece tokenizer (Llama 1/2, some others)
- `preprocessor_config.json` — vision model image preprocessor
- `chat_template.json` — standalone Jinja chat template
- `added_tokens.json` — additional token definitions

**Never download**:
- `*.md` — README, documentation
- `*.txt` — text files (often license, requirements)
- `.gitattributes` — Git LFS pointers metadata
- `*optimizer*`, `*training*` — training artifacts
- `*.h5`, `*.ot` — alternative weight formats vLLM doesn't use

**Decision for GGUF models**: If the repo contains only GGUF files, download only the selected GGUF file(s) plus any config/tokenizer files. If the repo contains both safetensors and GGUF, prefer safetensors (vLLM's native format) unless the user explicitly selects GGUF.

**Decision for weight format priority**: If a repo has both `.safetensors` and `.bin` weights (common during format transition), download only `.safetensors`. Check by listing files first — if any `.safetensors` exist, skip all `.bin` weight files.

---

## 5. Gated Model Handling

### Detection

The HF API returns `"gated": "auto"` or `"gated": "manual"` for gated models. The `ModelSearchResult` and `ModelDetail` structs carry this field.

### Access requirements

- Gated models require a HuggingFace token (`HF_TOKEN`)
- The user must have accepted the model's license on huggingface.co
- Without a token: search results show the model, but downloading returns HTTP 403
- With a token but no license acceptance: HTTP 403 with a specific error message

### UI indicators

- Search results: show a lock icon or `[gated]` badge next to gated models
- Model detail panel: if `HF_TOKEN` is not configured, show a warning with a link to Settings
- Download button: disabled with tooltip "Requires HF token" if token is missing
- On 403 error during download: display "Access denied — have you accepted the license at huggingface.co/{modelID}?"

### Implementation

The `Client` sets the auth header on all requests when a token is present:

```go
func (c *Client) setAuth(req *http.Request) {
    if c.token != "" {
        req.Header.Set("Authorization", "Bearer "+c.token)
    }
}
```

Copied from llama-toolchest verbatim.

---

## 6. Downloader Architecture

### Package: `internal/huggingface/downloader.go`

### Key differences from llama-toolchest's downloader

| Aspect | llama-toolchest | vllm-toolchest |
|--------|-----------------|----------------|
| Download unit | Single GGUF file (+ shards) | Multiple independent files (safetensors shards, configs) |
| Concurrency | Sequential shard download | Concurrent file downloads (configurable parallelism) |
| Storage path | `{datadir}/models/{org}--{repo}/{filename}` | `{datadir}/models/{org}/{repo}/{filename}` |
| ID format | `{org}--{repo}--{filename}` | `{org}--{repo}` (whole model) |
| Resume | Per-file `.part` suffix | Per-file `.part` suffix (same) |
| Completion | Register single model file | Register model directory |

### Downloader struct

```go
type Downloader struct {
    dataDir       string
    token         string
    maxConcurrent int // default 3 — concurrent file downloads within a model
    onComplete    CompletionFunc

    mu     sync.Mutex
    active map[string]*download
}

type CompletionFunc func(downloadID, modelID, modelDir string)

func NewDownloader(dataDir, token string) *Downloader {
    return &Downloader{
        dataDir:       dataDir,
        token:         token,
        maxConcurrent: 3,
        active:        make(map[string]*download),
    }
}
```

### Download tracking

```go
type download struct {
    cancel     context.CancelFunc
    files      []fileDownload
    mu         sync.Mutex
    totalBytes int64 // sum of all file sizes
    downloaded int64 // sum of all bytes downloaded so far
    status     string // "downloading", "complete", "failed", "cancelled"
    error      string

    // Fan-out to multiple subscribers
    subMu sync.Mutex
    subs  map[chan DownloadProgress]struct{}
}

type fileDownload struct {
    Filename    string
    Size        int64
    Downloaded  int64
    Status      string // "pending", "downloading", "complete", "failed"
}

type DownloadProgress struct {
    ID              string         `json:"id"`
    ModelID         string         `json:"model_id"`
    TotalBytes      int64          `json:"total_bytes"`
    BytesDownloaded int64          `json:"bytes_downloaded"`
    SpeedBPS        int64          `json:"speed_bps"`
    Status          string         `json:"status"`
    Error           string         `json:"error,omitempty"`
    Files           []FileProgress `json:"files,omitempty"`
    ActiveFiles     int            `json:"active_files"` // how many files downloading concurrently
    CompletedFiles  int            `json:"completed_files"`
    TotalFiles      int            `json:"total_files"`
}

type FileProgress struct {
    Filename    string `json:"filename"`
    Size        int64  `json:"size"`
    Downloaded  int64  `json:"downloaded"`
    Status      string `json:"status"`
}
```

### Concurrent file downloads

```go
func (d *Downloader) run(ctx context.Context, downloadID, modelID string, files []ModelFile, dl *download) {
    defer d.cleanup(downloadID)

    modelDir := d.modelDir(modelID) // /data/models/{org}/{repo}/
    os.MkdirAll(modelDir, 0o755)

    // Separate weight files (large, download concurrently) from config files (small, download first)
    var configFiles, weightFiles []ModelFile
    for _, f := range files {
        if f.Category == "weight" {
            weightFiles = append(weightFiles, f)
        } else {
            configFiles = append(configFiles, f)
        }
    }

    // Download config files first (small, sequential, needed for model detection)
    for _, f := range configFiles {
        if err := d.downloadFile(ctx, downloadID, modelID, f, dl); err != nil {
            dl.setFailed(err)
            return
        }
    }

    // Download weight files concurrently
    sem := make(chan struct{}, d.maxConcurrent)
    var wg sync.WaitGroup
    var firstErr error
    var errOnce sync.Once

    for _, f := range weightFiles {
        select {
        case <-ctx.Done():
            dl.setCancelled()
            return
        case sem <- struct{}{}:
        }

        wg.Add(1)
        go func(f ModelFile) {
            defer wg.Done()
            defer func() { <-sem }()
            if err := d.downloadFile(ctx, downloadID, modelID, f, dl); err != nil {
                errOnce.Do(func() { firstErr = err })
            }
        }(f)
    }
    wg.Wait()

    if firstErr != nil {
        dl.setFailed(firstErr)
        return
    }

    dl.setComplete()
    if d.onComplete != nil {
        d.onComplete(downloadID, modelID, modelDir)
    }
}
```

### Why concurrent downloads for weight files

safetensors models shard weights into multiple independent files (e.g., `model-00001-of-00008.safetensors` through `model-00008-of-00008.safetensors`). These shards have no dependencies on each other — they can be downloaded in parallel.

With 3 concurrent downloads on a typical internet connection, total download time is reduced by ~2-3x compared to sequential. The HuggingFace CDN handles concurrent connections well.

Config files are downloaded first and sequentially because:
1. They're small (< 1MB total)
2. `config.json` is needed to register the model in the inventory immediately after download completes
3. No meaningful parallelism benefit

### Resumable downloads via HTTP Range

Identical pattern to llama-toolchest:

```go
func (d *Downloader) downloadFile(ctx context.Context, ...) error {
    partPath := filepath.Join(modelDir, filename + ".part")
    finalPath := filepath.Join(modelDir, filename)

    // Skip if already downloaded
    if _, err := os.Stat(finalPath); err == nil {
        return nil
    }

    // Check existing partial
    var existingSize int64
    if info, err := os.Stat(partPath); err == nil {
        existingSize = info.Size()
    }

    req, _ := http.NewRequestWithContext(ctx, "GET", downloadURL, nil)
    if d.token != "" {
        req.Header.Set("Authorization", "Bearer " + d.token)
    }
    if existingSize > 0 {
        req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
    }

    resp, err := http.DefaultClient.Do(req)
    // ... handle 200 OK (full) or 206 Partial Content (resume)
    // ... stream to .part file, track progress
    // ... rename .part to final on completion
}
```

HuggingFace CDN supports HTTP Range headers — interrupted downloads resume from where they left off. The `.part` suffix prevents partial files from being mistaken for complete ones.

### Progress tracking and SSE fan-out

Same fan-out pattern as llama-toolchest. Each `download` has a subscriber map:

```go
func (dl *download) broadcast(progress DownloadProgress) {
    dl.subMu.Lock()
    for ch := range dl.subs {
        select {
        case ch <- progress:
        default: // drop if slow
        }
    }
    dl.subMu.Unlock()
}
```

Progress updates are sent every 500ms (same as llama-toolchest) to avoid flooding the SSE connection.

The aggregate progress combines all files:

```go
func (dl *download) aggregateProgress() DownloadProgress {
    dl.mu.Lock()
    defer dl.mu.Unlock()

    var totalDownloaded int64
    var completed, active int
    for _, f := range dl.files {
        totalDownloaded += f.Downloaded
        switch f.Status {
        case "complete": completed++
        case "downloading": active++
        }
    }

    return DownloadProgress{
        TotalBytes:      dl.totalBytes,
        BytesDownloaded: totalDownloaded,
        Status:          dl.status,
        ActiveFiles:     active,
        CompletedFiles:  completed,
        TotalFiles:      len(dl.files),
    }
}
```

### Cancel support

```go
func (d *Downloader) Cancel(downloadID string) error {
    d.mu.Lock()
    dl, ok := d.active[downloadID]
    d.mu.Unlock()
    if !ok {
        return fmt.Errorf("no active download: %s", downloadID)
    }
    dl.cancel() // cancels the context, all in-flight HTTP requests abort
    return nil
}
```

Context cancellation propagates to all concurrent `downloadFile` goroutines via `http.NewRequestWithContext(ctx, ...)`. In-flight HTTP responses close, partial files remain on disk for resume.

### Storage layout

```
/data/models/
  meta-llama/
    Llama-3.1-8B-Instruct/
      config.json
      tokenizer.json
      tokenizer_config.json
      generation_config.json
      special_tokens_map.json
      model-00001-of-00004.safetensors
      model-00002-of-00004.safetensors
      model-00003-of-00004.safetensors
      model-00004-of-00004.safetensors
      model.safetensors.index.json
  TechxGenus/
    Llama-3.1-8B-Instruct-AWQ/
      config.json
      tokenizer.json
      ...
      model.safetensors
  bartowski/
    Llama-3.1-8B-Instruct-GGUF/
      Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

This mirrors the HuggingFace directory structure. vLLM loads models by directory path (e.g., `--model /data/models/meta-llama/Llama-3.1-8B-Instruct`), so the standard HF layout is the natural choice.

**Difference from llama-toolchest**: llama-toolchest flattens `org/repo` to `org--repo` (single directory level) because llama.cpp loads individual files. vLLM loads directories, so we preserve the hierarchy.

### Edge case: Safetensors index file

Models with multiple safetensors shards include a `model.safetensors.index.json` file that maps tensor names to shard files. This file MUST be downloaded — vLLM uses it to load shards efficiently. Categorize it as `"config", true`.

### Edge case: GGUF models with multiple quant variants

Some GGUF repos (e.g., bartowski uploads) contain many GGUF files at different quantization levels in the same repo. For these:
- Show each GGUF file as a separate downloadable item (like llama-toolchest)
- The user selects which quant level to download
- Only download the selected GGUF file(s), not all of them
- Still download config.json and tokenizer files if present (vLLM can use them with GGUF)

This requires the UI to distinguish between "download entire model" (safetensors) and "download selected file" (GGUF) modes.

---

## 7. API Endpoints

### Route group

```go
r.Route("/api/hf", func(r chi.Router) {
    r.Get("/search", s.handleHFSearch)             // search models
    r.Get("/model", s.handleHFModel)               // model detail + files
    r.Post("/download", s.handleHFDownload)         // start download
    r.Get("/downloads", s.handleHFActiveDownloads)  // list active downloads
    r.Get("/download/{id}/progress", s.handleHFDownloadProgress) // SSE progress stream
    r.Delete("/download/{id}", s.handleHFDownloadCancel)         // cancel download
})
```

Same route structure as llama-toolchest.

### `GET /api/hf/search?q={query}`

**Handler**: `handleHFSearch`

Request: `?q=llama+3.1+8b`

Response (htmx): HTML partial `hf_results` with search results grouped by base model
Response (JSON): `[]ModelSearchResult`

The handler calls `s.hfClient.Search()` then groups results by normalized base name. Passes both the groups and the flat list to the template.

### `GET /api/hf/model?id={modelID}`

**Handler**: `handleHFModel`

Request: `?id=meta-llama/Llama-3.1-8B-Instruct`

Response (htmx): HTML partial `hf_files` with file list, quant info, VRAM estimate
Response (JSON): `ModelDetail`

This fetches:
1. Model metadata from `/api/models/{id}`
2. File tree from `/api/models/{id}/tree/main?recursive=true`
3. `config.json` content (for architecture, quant config) — fetch inline via `/resolve/main/config.json`

### `POST /api/hf/download`

**Handler**: `handleHFDownload`

Request body (JSON): `{"model_id": "meta-llama/Llama-3.1-8B-Instruct", "files": ["config.json", "tokenizer.json", "model-00001-of-00004.safetensors", ...]}`
Request body (form): `model_id=...&files=...` (for htmx form submission)

The `files` field is optional — if omitted, download all required files (auto-detected). If provided, download only the listed files (for GGUF selective download).

Response (htmx): HTML partial `download_progress` with SSE-connected progress div
Response (JSON): `{"download_id": "meta-llama--Llama-3.1-8B-Instruct"}`

### `GET /api/hf/download/{id}/progress`

**Handler**: `handleHFDownloadProgress`

SSE stream. Events:

```
event: progress
data: <HTML progress bar with file counts and speed>

event: progress
data: <updated HTML>

event: done
data: <completion message HTML>
```

For htmx consumers, the progress event data is HTML (same approach as llama-toolchest):

```html
<progress value="45" max="100"></progress>
<small>3.2 / 7.1 GB (85.3 MB/s) — 45% — 4/8 files</small>
```

For JSON consumers, the SSE data is the `DownloadProgress` struct as JSON.

**Decision**: Send HTML in progress events (matching llama-toolchest pattern). This avoids client-side JavaScript for rendering progress — htmx's SSE extension swaps the HTML directly.

### `GET /api/hf/downloads`

**Handler**: `handleHFActiveDownloads`

Returns all currently active downloads with their latest progress. Used by the browse page to show an "Active Downloads" section.

### `DELETE /api/hf/download/{id}`

**Handler**: `handleHFDownloadCancel`

Cancels an in-progress download. Returns 204 No Content.

---

## 8. Browse Page UI

### Template: `web/templates/models_browse.html`

### Layout

```
┌──────────────────────────────────────────────────┐
│  Search HuggingFace Models                        │
│  [________________________] [Search]              │
│                                                    │
│  ┌─ Active Downloads ──────────────────────────┐  │
│  │ meta-llama/Llama-3.1-8B-Instruct            │  │
│  │ ████████████░░░░░░ 45% — 3.2/7.1 GB — 4/8  │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌─ Search Results ────────────────────────────┐  │
│  │                                              │  │
│  │  Llama-3.1-8B-Instruct                      │  │
│  │  meta-llama · 2.1M downloads · 1.5K likes   │  │
│  │  Variants: [FP16] [AWQ] [GPTQ] [GGUF] [FP8]│  │
│  │  [View Details]                              │  │
│  │                                              │  │
│  │  Qwen2.5-7B-Instruct                        │  │
│  │  Qwen · 800K downloads · 900 likes          │  │
│  │  Variants: [FP16] [AWQ] [GPTQ]              │  │
│  │  [View Details]                              │  │
│  │                                              │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌─ Model Detail Panel ───────────────────────┐   │
│  │ meta-llama/Llama-3.1-8B-Instruct-AWQ       │   │
│  │ Architecture: LlamaForCausalLM              │   │
│  │ Quantization: AWQ 4-bit (group_size=128)    │   │
│  │ Parameters: ~8B                             │   │
│  │ Est. VRAM: ~5.2 GB                          │   │
│  │ Total download: 4.7 GB                      │   │
│  │                                             │   │
│  │ Files:                                      │   │
│  │  ✓ config.json               1.2 KB         │   │
│  │  ✓ tokenizer.json           2.1 MB          │   │
│  │  ✓ tokenizer_config.json      812 B         │   │
│  │  ✓ model.safetensors        4.7 GB          │   │
│  │                                             │   │
│  │ [Download Model]                            │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### Search bar with debounced input

```html
<input type="search"
       name="q"
       placeholder="Search models (e.g., llama 3.1 8b, qwen2.5, mistral...)"
       hx-get="/api/hf/search"
       hx-trigger="input changed delay:400ms, search"
       hx-target="#search-results"
       hx-indicator="#search-spinner">
<span id="search-spinner" class="htmx-indicator">Searching...</span>
```

The `delay:400ms` provides debouncing — doesn't fire until 400ms after the user stops typing. `search` event fires on Enter key.

### Results list partial (`partials/hf_results.html`)

Each search result group shows:
- **Model name** (base name, linked)
- **Author** and download/like counts
- **Variant badges**: colored badges for each available format
  - FP16/BF16: default color
  - AWQ: green badge
  - GPTQ: blue badge
  - GGUF: orange badge
  - FP8: purple badge
  - BnB: yellow badge
- **Gated indicator**: lock icon if `Gated != ""`
- **View Details button**: `hx-get="/api/hf/model?id={modelID}" hx-target="#model-detail"`

Clicking a variant badge loads that specific variant's details.

### Model detail panel partial (`partials/hf_files.html`)

Shows after clicking a specific model variant:
- Model ID (full `org/repo`)
- Architecture (from config.json)
- Quantization details (method, bits, group size)
- Parameter count (if available)
- Estimated VRAM requirement
- Total download size

**File list table**:
```html
<table>
  <thead><tr><th>File</th><th>Size</th><th>Required</th></tr></thead>
  <tbody>
    {{range .Files}}
    {{if ne .Category "skip"}}
    <tr>
      <td>{{.Filename}}</td>
      <td>{{.Size | formatBytes}}</td>
      <td>{{if .IsRequired}}✓{{else}}○{{end}}</td>
    </tr>
    {{end}}
    {{end}}
  </tbody>
  <tfoot><tr><td colspan="3"><strong>Total: {{.TotalSize | formatBytes}}</strong></td></tr></tfoot>
</table>
```

**Download button**:
```html
<button hx-post="/api/hf/download"
        hx-vals='{"model_id": "{{.ID}}"}'
        hx-target="#download-status-{{.ID | safeID}}"
        hx-swap="innerHTML">
    Download Model ({{.TotalSize | formatBytes}})
</button>
<div id="download-status-{{.ID | safeID}}"></div>
```

For GGUF repos with multiple quant variants, show a dropdown or radio buttons to select which GGUF file to download:

```html
{{if .IsGGUFRepo}}
<select name="gguf_file">
  {{range .GGUFFiles}}
  <option value="{{.Filename}}">{{.Quant}} — {{.Size | formatBytes}} — Est. {{.VRAMEstGB | printf "%.1f"}} GB VRAM</option>
  {{end}}
</select>
{{end}}
```

### Download progress partial (`partials/download_progress.html`)

Injected after clicking Download. Contains an SSE-connected div:

```html
<div hx-ext="sse"
     sse-connect="/api/hf/download/{{.DownloadID}}/progress"
     sse-swap="progress">
    <progress value="0" max="100"></progress>
    <small>Starting download...</small>
</div>
```

The SSE `progress` events replace the inner HTML of this div with updated progress bars.

On completion, the `done` event replaces the div with:
```html
<p>Download complete! <a href="/models">View in Models →</a></p>
```

### Active downloads section

At the top of the browse page, shows all in-progress downloads:

```html
<div id="active-downloads"
     hx-get="/api/hf/downloads"
     hx-trigger="load, every 3s"
     hx-swap="innerHTML">
</div>
```

Each active download shows model name, progress bar, speed, and a cancel button.

---

## 9. What to Copy from llama-toolchest vs Build New

### Copy and adapt

| Component | From | Changes |
|-----------|------|---------|
| `internal/huggingface/downloader.go` | llama-toolchest | Concurrent multi-file download, directory-based storage, new progress struct |
| `internal/api/hf.go` | llama-toolchest | Adapt handlers for new client API, new response formats |
| `web/templates/models_browse.html` | llama-toolchest | New layout with variant grouping, format badges |
| `web/templates/partials/hf_results.html` | llama-toolchest | Redesign for multi-format results |
| `web/templates/partials/hf_files.html` | llama-toolchest | Redesign for directory-based file listing |
| `web/templates/partials/download_progress.html` | llama-toolchest | Minor changes (file count in progress) |

### Build new

| Component | Purpose |
|-----------|---------|
| `internal/huggingface/client.go` | Completely rewritten — no GGUF filter, config.json fetching, quant detection |
| Quant detection logic | `detectQuantFromTags()`, `normalizeBaseName()`, config.json parsing |
| Model grouping in handler | Group search results by base model name |
| GGUF file selector UI | Radio/dropdown for GGUF repos with multiple quants |
| `categorizeFile()` | File categorization for safetensors/config/tokenizer/skip |
| VRAM estimation | Parameter-count-based estimation from config.json |

---

## 10. Edge Cases

### Large models exceeding VRAM

If estimated VRAM exceeds the GPU's total VRAM, show a warning:
```html
<mark>This model requires ~24 GB VRAM. Your GPU has 16 GB.
Consider using a quantized variant (AWQ, GPTQ) or enabling KV cache quantization.</mark>
```

The VRAM estimate is compared against `s.monitor.Current().GPU[0].VRAMTotalMB`.

### Repos with no weight files

Some HF repos are dataset repos, adapter-only repos, or empty. If no weight files are found, show "No downloadable model weights found in this repository."

### Duplicate downloads

If the user tries to download a model that's already downloading, `Downloader.Start()` returns an error. The handler returns the existing download ID so the UI can connect to the in-progress SSE stream.

If the model is already fully downloaded (files exist on disk), the download completes instantly (each `downloadFile` skips existing final files).

### Network interruption during concurrent download

If one of the concurrent file downloads fails:
- The error propagates via `errOnce`
- The download context is NOT cancelled (other files continue)
- The download is marked as failed
- Retrying later resumes all files from their `.part` state
- Only the failed file needs to re-download from its last position

**Decision**: Don't cancel other concurrent downloads on single file failure. The partial progress is preserved via `.part` files, and the cost of re-downloading completed files is zero (they're skipped).

### HuggingFace API rate limiting

HF API has rate limits (varies by authentication status):
- Unauthenticated: ~100 requests/hour for search
- Authenticated: higher limits

If a 429 is received, the client should respect the `Retry-After` header. For Phase 3, log the rate limit error and show it in the UI. Don't implement automatic retry — the user can retry manually.

### Model with thousands of files

Some repos (especially multimodal models) have hundreds of files. The tree API may paginate. Handle pagination:

```go
func (c *Client) fetchTree(ctx context.Context, modelID string) ([]TreeEntry, error) {
    var allEntries []TreeEntry
    cursor := ""
    for {
        u := fmt.Sprintf("%s/models/%s/tree/main?recursive=true", baseURL, modelID)
        if cursor != "" {
            u += "&cursor=" + cursor
        }
        // ... fetch and decode
        allEntries = append(allEntries, entries...)
        if len(entries) < 1000 { // no more pages
            break
        }
        cursor = entries[len(entries)-1].Path
    }
    return allEntries, nil
}
```

### GGUF models served by vLLM

vLLM can serve GGUF files directly (added in vLLM 0.4+). For GGUF downloads:
- The model path passed to vLLM is the GGUF file path, not a directory
- Config and tokenizer files are optional (GGUF embeds metadata)
- But if tokenizer files are present, vLLM uses them (better tokenization)
- Download tokenizer files alongside GGUF if available in the repo

### Quantization format compatibility matrix

Not all quant formats work on all hardware. For RDNA4 / gfx1201:

| Format | vLLM Support | RDNA4 Notes |
|--------|-------------|-------------|
| FP16/BF16 | Native | Works, but uses 2x VRAM vs 8-bit |
| AWQ | Via Triton (`VLLM_USE_TRITON_AWQ=1`) | Needs Triton AWQ kernels, enabled in Phase 1 env |
| GPTQ | Via AutoGPTQ or Marlin | AutoGPTQ works; Marlin may need ROCm-specific kernels |
| FP8 | Native in vLLM | gfx1201 has native FP8 instructions |
| GGUF | Via llama.cpp integration in vLLM | Works for supported quant levels (Q4_K_M, Q5_K_M, etc.) |
| BitsAndBytes 4-bit | Via bitsandbytes-rocm | Needs the ROCm fork installed (Phase 1) |
| BitsAndBytes 8-bit | Via bitsandbytes-rocm | Same |
| Marlin | ROCm Marlin kernels | May not support gfx1201 yet — test at runtime |

Show this compatibility info in the model detail panel when relevant (e.g., warn about Marlin on gfx1201).

---

## 11. Validation Criteria

Phase 3 is complete when:

1. Search bar returns results from HuggingFace within 2 seconds
2. Results are grouped by base model with quant variant badges
3. Clicking a variant shows the file detail panel with architecture, quant info, VRAM estimate, and file list
4. Downloading a safetensors model:
   - Config files download first
   - Weight shards download concurrently (verify with `netstat` or download speed)
   - Progress bar updates in real-time via SSE
   - File count shows (e.g., "4/8 files")
   - `.part` files visible during download, renamed on completion
5. Downloading a GGUF model:
   - Dropdown shows available quant variants
   - Only the selected GGUF file is downloaded
6. Cancelling a download:
   - Button immediately cancels
   - Partial `.part` files preserved
   - Restarting the same download resumes from `.part` state
7. Gated model:
   - Shows lock indicator in search results
   - Without HF token: shows warning, download button disabled
   - With HF token but no access: shows "accept license" message on 403
8. Active downloads section shows all in-progress downloads with progress
9. Multiple browser tabs can watch the same download progress independently
10. Downloaded models appear in `/data/models/{org}/{repo}/` with expected file layout

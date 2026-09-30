# Phase 07 — The helper model, and asking it for a form

**Depends on:** nothing · **Enables:** reading a card's prose (phase 09)

## Goal

Install, own and remove the one helper model, and be able to ask an engine for
an answer constrained to a JSON schema. The helper is
`Qwen/Qwen3-4B-Instruct-2507`: 4.02B parameters, unquantized BF16, a plain
dense architecture, Apache-2.0 and ungated -- chosen so that every image
variant can load it. It is the app's model, not the operator's: it has fixed
settings, it is not listed among the models that can be served, and the only
things to do with it are download it and remove it, both from Settings.

## Files touched

- `internal/models/registry.go` — `Model.Helper bool`
  (`json:"helper,omitempty"`); `Registry.SetHelper`.
- `internal/models/helper.go` — new. `HelperRepo`, `HelperServedName`,
  `HelperContext`, `HelperUtil`, `HelperConfig`, `HelperFits`.
- `internal/models/helper_test.go` — new.
- `internal/llmcall/llmcall.go` — new package. `Client`, `Message`, `JSON`.
- `internal/llmcall/llmcall_test.go` — new, with an `httptest` engine.
- `internal/api/helper_model.go` — new. `helperModel`, `helperWanted`,
  the three Settings handlers, the claim on download.
- `internal/api/helper_model_test.go` — new.
- `internal/api/server.go` — routes; claim the helper in the download
  completion hook.
- `internal/api/drafts.go` — `servable()` excludes the helper;
  `launchBlocker` refuses to start it.
- `internal/api/models.go` — `modelRows()` excludes the helper;
  `handleActivateModel` refuses it.
- `web/templates/settings.html` — a "Helper model" section.
- `web/templates/partials/helper_model_panel.html` — new.

## Steps

1. `internal/models/helper.go`:

   ```go
   const HelperRepo = "Qwen/Qwen3-4B-Instruct-2507"
   const HelperServedName = "vllmctl-helper"
   const HelperContext = 16384
   ```

   `func HelperConfig(gpuMemoryUtil float64) VLLMConfig` returns fixed
   settings: `TensorParallelSize 1`, `MaxModelLen HelperContext`,
   `EnforceEager true` (no graph pool, and nothing to compile),
   `MaxNumSeqs 1`, `MaxNumBatchedTokens 2048`, `KVCacheDtype "auto"`,
   `Dtype "auto"`, `LoadFormat "auto"`, the given utilization, and everything
   else zero: no tools, no parsers, no extra flags, no env. It ignores
   whatever is stored on the helper's registry record, so a config edited by
   hand cannot break the feature.

2. `func HelperUtil(configured float64) float64` returns `configured` when it
   is in (0, 1] and 0.90 otherwise -- the same default the process manager
   and the planner apply. Every caller of `HelperConfig` and `HelperFits`
   passes `HelperUtil(cfg.GPUMemoryUtil)`, so an unset setting can never make
   the helper look as though it fits in nothing.

   `func HelperFits(inv GPUInventory, util float64) (ok bool, why string)`.
   It must answer before the helper is downloaded, so it does not read a
   registry record. It builds a `Model` from constants: the helper's shape as
   published in its `config.json` (checked 2026-09-30: `Qwen3ForCausalLM`, 36
   layers, hidden 2560, intermediate 9728, 32 attention heads, 8 KV heads,
   head dim 128, vocabulary 151,936, tied embeddings), `TotalSizeBytes` of
   8,044,936,192 (4,022,468,096 BF16 parameters), `BytesPerParam` 2, and
   `HelperConfig(util)`. It takes `EstimateVRAM` of that,
   `RequiredAt(est, cfg, 1).TotalHighGB`, and compares with
   `inv.PerCardGB * util` -- the card's size, not its free memory, because the
   engine will have been stopped. When it does not fit, `why` names the
   figures: "the helper needs about 9.4 GB and the smallest card offers 7.2".
   An unknown inventory returns `true`: refusing on no information would
   disable the feature on a host whose cards have simply not been read yet.

3. `Model.Helper` marks the record. `Registry.SetHelper(id string, on bool) error`
   is a narrow mutator in the style of `SetEnabled`. No schema bump, by the
   same precedent as phase 01.

4. `internal/api/helper_model.go`:
   - `func (s *Server) helperModel() *models.Model` — the registry model with
     `Helper` set and not orphaned, or nil.
   - `s.helperWanted` — an atomic bool set when the Settings panel asks for
     the download.
   - Claim: wrap the existing `recordTransfer(reg)` completion function so
     that after it registers the model, a download of `HelperRepo` while
     `helperWanted` is set calls `SetHelper(HelperRepo, true)` and clears the
     flag. A `HelperRepo` downloaded by the operator from the search page,
     with the flag unset, stays an ordinary model.

5. Exclude the helper from `servable()` and from `modelRows()`, so it has no
   card on the Models page and cannot be configured or benchmarked.
   `handleActivateModel` refuses it as it refuses a draft ("the helper model
   is not served on its own"), and `launchBlocker` refuses to start it through
   the ordinary Start path. It still appears in storage figures.

6. Settings handlers, outside the autosaving form as Backup & Restore is:
   - `GET /api/settings/helper` renders `helper_model_panel`.
   - `POST /api/settings/helper/download` — if `HelperRepo` is already
     installed as an ordinary model, mark it with `SetHelper` and re-render.
     Otherwise set `helperWanted` and call
     `s.startTransfer(ctx, models.HelperRepo, "", false)`, then re-render with
     `HX-Trigger: downloadsChanged`.
   - `DELETE /api/settings/helper` — `s.registry.Delete(id, true)` and
     re-render with the space freed.
   The panel's state is the first of these that holds: downloading (the
   existing download progress partial, polling every 2 s); installed (name,
   size on disk, Remove); not installed (what it is for, its size of about
   8 GB, a Download button). Independently of the state, when `HelperFits`
   says no, the panel adds its sentence and that autoconfigure will work from
   the machine and the card's command alone -- above the Download button, or
   beside Remove if it was installed anyway.

7. `internal/llmcall`:

   ```go
   type Message struct{ Role, Content string }
   type Client struct{ HTTP *http.Client } // Timeout 5 minutes when nil

   // JSON asks the engine at baseURL for an answer constrained to schema and
   // decodes it into out.
   func (c *Client) JSON(ctx context.Context, baseURL, model, schemaName string,
       schema map[string]any, msgs []Message, out any) error
   ```

   It POSTs `baseURL + "/v1/chat/completions"` with `model`, `messages`,
   `temperature: 0`, `max_tokens: 2048`, `stream: false` and
   `response_format: {"type": "json_schema", "json_schema": {"name":
   schemaName, "schema": schema, "strict": true}}`. No Authorization header:
   the engine is never started with an API key. It reads
   `choices[0].message.content` and unmarshals it into `out`.

8. Two recoveries, each tried once:
   - an HTTP 400 whose body mentions `response_format` or `json_schema` is
     retried with the older spelling, `guided_json: schema` in place of
     `response_format`, for the images pinned to vLLM 0.23;
   - content that is not valid JSON for `out` is retried with one more user
     message: "Answer again with only the JSON object."
   Any other failure is returned with the status and the first 300 characters
   of the body.

9. `finish_reason: "length"` is an error ("the answer was cut off"), not a
   partial result.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
go test ./internal/api -run 'TestGoldenPartials' -update   # records helper_model_panel
```

## Test plan

- `HelperConfig` is eager, one sequence, 16,384 context, no flags; it is the
  same whatever the helper's stored config says.
- `HelperFits` on a 31.86 GiB card is true; on an 8 GiB card it is false and
  the sentence carries both figures; on an unknown inventory it is true. It
  gives the same answer with no helper installed.
- A download of `HelperRepo` with the flag set is marked as the helper; the
  same download with the flag unset is not; the Download handler on an
  already-installed copy marks it without transferring anything.
- The helper is absent from `servable()` and from the model list fragment, and
  `PUT /api/models/activate` for its ID is refused.
- Remove deletes the files and the record.
- `helper_model_panel` golden in its three states, and in the not-installed
  state with the too-large sentence.
- `llmcall.JSON` against a fake engine: a valid answer decodes; a 400 naming
  `response_format` is retried with `guided_json` and succeeds; invalid JSON
  is retried once and then returned as an error; `finish_reason: length` is an
  error; a cancelled context returns promptly; the request body carries the
  schema and `temperature: 0`.

## Commit

```
feat(helper): install the app's helper model and ask an engine for a JSON form
```

## Rollback

Revert the commit. A helper already downloaded becomes an ordinary model on
the Models page, because the older build ignores the `helper` field; remove it
there, or leave it. Nothing loads the helper until phase 09.

# Phase 17 — The recommendation engine

**Depends on:** phase 15 (exact sizes), phase 16 (architecture registry) ·
**Enables:** phase 18's feed and phase 19's config handoff.

## Goal

Build `internal/recommend`: the machine profile, the candidate pool, the
staged ranking and the four objectives. This phase has no UI. It ends with a
JSON endpoint that returns the ranked result, which is what phase 18 renders
and what makes this phase verifiable on its own.

The staging is forced by what the Hub returns. Verified while planning:
`expand[]=safetensors` gives exact parameter counts for every result, and
`expand[]=config` gives `architectures` and `model_type` — but **not**
`hidden_size`, `num_hidden_layers`, `num_key_value_heads` or
`max_position_embeddings`. Weight size and architecture are therefore cheap
and available for the whole pool, while everything the KV-cache arithmetic
needs requires fetching the repository's raw `config.json` one repo at a time.
So: rank coarsely on the cheap tier, fetch configs for the finalists only,
then compute the real fit.

`internal/models` already owns the hard arithmetic — `EstimateVRAM`, `Fit` and
`evaluateTP`, with an `HFConfig` that is MoE- and hybrid-attention-aware. This
package does not reimplement any of it. Its job is to decide which
repositories to ask about, to get a remote `config.json` into a form
`models.ParseHFConfig` already reads, and to order the answers.

## Files touched

- `internal/recommend/engine.go` — new. The `Engine` type, the `Candidate`
  type, the pool cache and the public entry points.
- `internal/recommend/profile.go` — new. `Profile` and its construction.
- `internal/recommend/accel.go` — new. The GPU-arch → accelerated-formats
  table and the image veto.
- `internal/recommend/candidates.go` — new. Pool fetch, merge, dedupe, coarse
  filter and coarse rank. The cached pool itself lives on the `Engine` in
  `engine.go`, behind its mutex.
- `internal/recommend/config_cache.go` — new. The on-disk `config.json` cache.
- `internal/recommend/score.go` — new. The coarse score and the four
  objectives.
- `internal/recommend/verdict.go` — new. Verified / Unverified / Dropped and
  the reason strings.
- `internal/recommend/publishers.conf` — new. Trusted publishers, `go:embed`ed.
- `internal/recommend/*_test.go` — new. Per file.
- `internal/recommend/testdata/` — new. Recorded Hub payloads.
- `internal/huggingface/client.go` — add `Candidates` and `FetchConfigJSON`,
  and widen `ModelConfigMeta` with `Architectures []string` and
  `ModelType string`. The Hub already returns both under `config` and the
  struct discards them today, keeping only `QuantizationConfig`; the coarse
  tier needs the architecture to display. `Gated` needs no change —
  `ModelSearchResult.Gated` already exists.
- `variants/variants.go` — read `VARIANT_NO_ACCEL` into
  `Descriptor.NoAccel []string`.
- `internal/api/recommend.go` — new. `handleRecommend`,
  `handleRecommendRefresh`, and the engine held on `Server`.
- `internal/api/server.go` — register `GET /api/recommend` and
  `POST /api/recommend/refresh`.

`internal/models` is **not** modified. `ParseHFConfig(modelDir string)` is
already exported and reads only `config.json` from the directory it is given,
so the config cache writes each repository's `config.json` into its own
directory and calls the existing parser unchanged. A remote config therefore
goes through exactly the same code as a local one, with no second parser and
no bytes-variant to keep in step.

## Steps

1. **Profile**, in `profile.go`.

   ```go
   type Profile struct {
       Inventory   models.GPUInventory
       GPUName     string            // e.g. "Radeon AI PRO R9700"; may be ""
       GPUArch     string            // cfg.GPUArch, e.g. "gfx1201"
       Variant     string            // Descriptor.ID, e.g. "rdna4-clav"
       Accelerated []string          // computed by step 3, not supplied
       Archs       map[string]bool   // from phase 16
       ArchsKnown  bool
   }
   func NewProfile(inv models.GPUInventory, gpuName, gpuArch string,
                   d variants.Descriptor, archs map[string]bool,
                   archsKnown bool) Profile
   ```

   `NewProfile` fills `Accelerated` by running step 3's table over `gpuArch`
   and subtracting `d.NoAccel`; `Variant` comes from `d.ID`, which
   `variants.Descriptor` already carries (`variants/variants.go` reads it from
   `VARIANT_ID`) and which equals `env.Variant`. The descriptor is
   passed in rather than looked up so the package has one source for both the
   variant's identity and its veto, and so a test can supply either.

   Called from `internal/api` with `s.gpuInventory()`, `cfg.GPUArch`, the
   descriptor from `s.vllmEnv.Descriptor()`, and `s.SupportedArchs()`.
   `GPUName` comes from the first entry of `s.monitor.Current().GPU`, the same
   reading the dashboard's GPU card already renders; it is empty when the
   monitor has nothing, and the header omits it rather than showing a gap.

   `Profile` is serialised for the endpoint by an explicit view struct rather
   than by tagging it, because `Inventory` is a nested `models.GPUInventory`
   and step 17's `profile` object is flat. Both the type and the method are
   exported, because `internal/api` marshals the `Result` that holds one:

   ```go
   type ProfileView struct {
       GPUCount       int      `json:"gpu_count"`
       PerCardGB      float64  `json:"per_card_gb"`
       TotalVRAMGB    float64  `json:"total_vram_gb"`
       GPUName        string   `json:"gpu_name,omitempty"`
       GPUArch        string   `json:"gpu_arch"`
       Variant        string   `json:"variant"`
       Accelerated    []string `json:"accelerated"`
       ArchsKnown     bool     `json:"archs_known"`
       InventoryKnown bool     `json:"inventory_known"`
   }
   func (p Profile) View() ProfileView
   ```

   `TotalVRAMGB` is `Count × PerCardGB`, computed once here so the template
   never multiplies. When `Inventory.Known` is false the
   engine returns its result with `unavailable` set to
   `"the GPUs have not been read yet"` and both lists empty — the same
   mechanism as a network failure, so phase 18 renders one state rather than
   two. Guessing a card size is the exact mistake `GPUInventory`'s own doc
   comment records having been made before.

2. **The types everything else refers to**, in `engine.go` and
   `verdict.go`.

   ```go
   type Engine struct {
       hf      *huggingface.Client
       dataDir string        // for the config cache; passed at construction
       mu      sync.Mutex
       pool    *pool         // nil until the first build
   }
   func NewEngine(hf *huggingface.Client, dataDir string) *Engine

   type pool struct {
       profile     Profile
       generatedAt time.Time
       candidates  []Candidate
       orders      map[string][]int   // intent -> indices into candidates
   }

   type Candidate struct {
       // From the cheap tier.
       ID, Author, Format, Arch string
       Gated                    bool
       Downloads                int
       LastModified             time.Time
       WeightBytes              int64
       WeightGB                 float64 // WeightBytes / 1024³, for display
       WeightsKnown             bool
       ParamsB                  float64 // Safetensors.Total / 1e9; 0 if unknown
       // WeightOnly drives the Fastest penalty: the weights dequantize to
       // bf16 before the matmul, so the format buys memory and not speed.
       // Read from quantization_config, not the badge — a compressed-tensors
       // repo may hold either scheme and the badge says only
       // "compressed-tensors".
       WeightOnly bool

       // From the config fetch and the fit.
       HFConfig models.HFConfig
       Est      models.VRAMEstimate
       TP       int      // the recommended width; 0 when none fits
       // Copied from the recommended TPOption: RequiredGB, AvailableGB, SpareGB.
       Required, Available, Spare float64
       AffordableTokens int

       // The verdict.
       Verdict Verdict   // Verified, Unverified or Dropped
       Reason  string    // non-empty only when Unverified
       Accelerated bool
       // ArchSupported is meaningful only when Profile.ArchsKnown is true;
       // step 14 sets it there and leaves it false otherwise, which is why
       // the card keys its architecture clause off archs_known rather than
       // off this field.
       ArchSupported bool
   }
   ```

   `WeightOnly` is decided from `huggingface.QuantConfig`, which the cheap
   tier already carries and which already has every field needed — no new
   parsing:

   | condition | `WeightOnly` |
   |---|---|
   | `QuantMethod` is `awq` or `gptq` | true — both are weight-only by definition, whatever `Bits` says |
   | `QuantMethod` is `compressed-tensors` and `Format` is `pack-quantized` | true — the 4-bit packed layout |
   | `QuantMethod` is `compressed-tensors` and `Format` is `float-quantized` | false — the fp8 w8a8 layout |
   | `LoadIn4Bit` or `LoadIn8Bit` | true — bitsandbytes |
   | anything else, including absent `quantization_config` | false |

   The per-group `num_bits` that compressed-tensors nests under
   `config_groups` is deliberately not consulted: `Format` answers the same
   question at the top level, and the cheap tier's `config` does not carry the
   groups.

   ```go
   type Verdict string
   const (
       Verified   Verdict = "verified"
       Unverified Verdict = "unverified"
       Dropped    Verdict = "dropped"
   )
   ```

   The engine's public API is exactly three methods; everything else is
   unexported:

   ```go
   // Result builds or reuses the pool for p, then returns it ordered by intent.
   func (e *Engine) Result(ctx context.Context, p Profile, intent string) Result
   // Refresh discards any pool and rebuilds against p.
   func (e *Engine) Refresh(ctx context.Context, p Profile) Result
   // SeedFor is phase 19's entry point; it reports nothing before then.
   func (e *Engine) SeedFor(modelID string) (models.VLLMConfig, bool)
   ```

   ```go
   type Result struct {
       Profile     ProfileView `json:"profile"`
       Intent      string      `json:"intent"`
       GeneratedAt time.Time   `json:"generated_at"`
       Stale       bool        `json:"stale"`
       Unavailable string      `json:"unavailable"`
       Verified    []Candidate `json:"verified"`
       Unverified  []Candidate `json:"unverified"`
   }
   ```

   `Candidate`'s JSON tags produce step 17's two shapes from one struct.
   Every field falls in exactly one of three groups:

   | group | fields |
   |---|---|
   | always emitted | `ID`, `Author`, `Format`, `Gated`, `Downloads`, `LastModified`, `Accelerated`, `ArchSupported` |
   | `omitempty` | `TP`, `WeightGB`, `ParamsB`, `Required`, `Available`, `Spare`, `AffordableTokens`, `Arch`, `Reason` |
   | `json:"-"` | `WeightBytes`, `WeightsKnown`, `WeightOnly`, `HFConfig`, `Est`, `Verdict` |

   The two booleans are emitted unconditionally because the template must
   distinguish false from absent: phase 18 renders `not accelerated on
   <arch>` for `"accelerated": false`, which `omitempty` would make
   unreachable. `Reason` takes `omitempty` so a verified entry carries no
   `reason` key. The internal group never crosses the wire: `WeightBytes` and
   `WeightGB` are the same figure in two units, and `HFConfig` and `Est` are
   working state the feed has no use for. `Verdict` is dropped because which
   list an entry appears in already says it.

   `Result` carries a `ProfileView`, not the `Profile` itself, so the endpoint
   emits step 17's flat object rather than the nested inventory and the whole
   architecture set. It holds materialised slices rather than the `pool`'s
   `[]int` orders: the orders are the engine's internal storage, and a caller
   should not have to index back into a slice it does not hold.

   The `Profile` is passed per call rather than held, because it is built from
   live readings in `internal/api` and can change between requests; the pool
   remembers the one it was built against so step 16 can compare them.
   `dataDir` is supplied at construction from `cfg.DataDir`, so nothing in the
   package reads global configuration.

3. **Acceleration table** in `accel.go`: `gfx1200`, `gfx1201` → `fp8`;
   `gfx942`, `gfx950` → `fp8`; `gfx1100`, `gfx1101`, `gfx1102` → none;
   `sm_89`, `sm_90`, `sm_100`, `sm_120` → `fp8`; `sm_80`, `sm_86` → none.
   Anything unlisted returns none, which reads as "not accelerated" rather
   than "unknown" — the conservative direction, since claiming speed that is
   not there is worse than omitting speed that is. Subtract
   `Descriptor.NoAccel`. Format names are compared case-insensitively with
   `strings.EqualFold` throughout: the Hub's tags and `VARIANT_NO_ACCEL` are
   lowercase, while `DetectQuantFormat` produces display badges like `FP8`.
   `Profile.Accelerated` stores the lowercase form, which is what the JSON
   carries; `Candidate.Format` keeps the badge as rendered. Reading the veto
   requires adding it: add
   `NoAccel: strings.Fields(kv["VARIANT_NO_ACCEL"])` to the descriptor
   assembly in `variants/variants.go`, beside the existing
   `Caps`/`Capabilities` lines that already parse space-separated lists the
   same way. The value is lowercase format names separated by spaces, matching
   the badge values `DetectQuantFormat` produces. No shipped `variants/*.conf`
   sets the key — every current image has the kernels for what its silicon
   can do — so the field is empty everywhere until an image needs it. This is
   the machine-readable copy of the fact `help.html:121` currently states only
   in prose.

4. **Candidate buckets.** Always three groups, regardless of what is
   accelerated:

   - every accelerated format from step 3 (on `gfx1201`, `fp8`);
   - 4-bit weight-only, because it is how a model that does not otherwise fit
     gets to run at all;
   - unquantized, because on a 128 GB host a bf16 model is a legitimate and
     often best answer, and a pool built only from quantized repositories
     would never offer one.

   The first two expand to Hub tags through the existing
   `huggingface.QuantFilterTags`, which returns a list per bucket: `fp8` gives
   one tag, and 4-bit weight-only is the union of the `awq`, `gptq` and
   `compressed-tensors` buckets, giving three. One query is issued per tag,
   not per bucket. The third has no tag to map to — an
   unquantized repository is defined by the absence of a `quantization_config`,
   not by a tag — so it is queried with no quant `filter` beyond the
   `filter=transformers` the client already sends, and the results are then
   kept only where `DetectQuantFormat` returns the unquantized badge. That is
   a sieve rather than a filter, which is why it is used for this one bucket
   only: the other two have real tags and use them.

   On a profile with nothing accelerated the first group is empty and the
   other two still stand, so the pool is never reduced to 4-bit alone.

5. **Client methods.** Both are thin; all policy stays in `recommend`.

   ```go
   func (c *Client) Candidates(ctx context.Context, q CandidateQuery) ([]ModelSearchResult, error)
   type CandidateQuery struct {
       Tag   string // one Hub quant filter tag; empty for the unquantized bucket
       Sort  string // "downloads" or "lastModified"
       Limit int    // 50
   }
   func (c *Client) FetchConfigJSON(ctx context.Context, modelID, revision string) ([]byte, error)
   ```

   `Candidates` builds
   `/api/models?filter=transformers&sort=<sort>&direction=-1&limit=<limit>`,
   adding a second `&filter=<tag>` only when `Tag` is non-empty — the Hub ANDs
   repeated `filter` parameters, which is the behaviour `Search` already
   relies on. An empty `Tag` therefore yields the unquantized bucket's
   untagged query rather than a malformed one. Plus
   `expand[]=safetensors&expand[]=config&expand[]=downloads&expand[]=lastModified&expand[]=gated&expand[]=tags`,
   and calls `c.setAuth` exactly as `search()` does, so a configured token is
   honoured. `FetchConfigJSON` gets
   `https://huggingface.co/<modelID>/resolve/<revision>/config.json`, also
   through `setAuth`. Callers pass `"main"` as the revision; the parameter
   exists so a future caller can pin a commit without a signature change.

6. **Pool fetch.** One `Candidates` call per (bucket tag × sort), merged with
   first-occurrence-wins and deduped by repository id — the same merge shape
   `Client.Search` already uses for multi-tag buckets. Apply the existing
   `isGGUFOnly` filter to every result, so the overview's no-GGUF non-goal
   holds for the feed as it already does for search.

   On `gfx1201` the buckets expand to five queries per sort — `fp8`, `awq`,
   `gptq`, `compressed-tensors`, and the untagged unquantized one — so ten per
   refresh, up to ~500 results before deduping and ~8.5 MB at 854 KB per
   query. Deduping removes a good deal of that: `compressed-tensors` overlaps
   both the fp8 and the 4-bit buckets by design, since it is a container
   rather than a scheme.

7. **Coarse filter**, cheap tier only. Drop a candidate when
   `Safetensors.WeightBytes()` is known and, divided by 1024³ to reach GB,
   exceeds `Inventory.Count × Inventory.PerCardGB` — every card on the host,
   ignoring
   tensor-parallel divisibility entirely, because divisibility needs
   `num_key_value_heads` and that is not in the cheap tier. This is
   deliberately the most permissive possible bound: it drops only what cannot
   run on this hardware under any split, which is exactly what "proven bad"
   should mean at this stage. Anything it lets through is judged properly at
   steps 12–14. Do not drop on unknown size, and do not drop on architecture.

8. **Set `Accelerated`** on every surviving candidate, before the coarse
   score reads it: true when `Profile.Accelerated` contains the candidate's
   `Format`, compared with `strings.EqualFold` per step 3. It is cheap-tier
   data, so it is known for unverified candidates too, which is why step 2
   emits it unconditionally.

9. **Trusted publishers.** `internal/recommend/publishers.conf`, `go:embed`ed,
   one author per line, `#` for comments:

   ```
   # Authors whose quantized checkpoints are consistently servable by vLLM.
   # Presence is a ranking boost only; absence is never a penalty. Emptying
   # this file degrades the ordering without breaking the feed.
   RedHatAI
   neuralmagic
   Qwen
   mistralai
   meta-llama
   deepseek-ai
   google
   microsoft
   nvidia
   unsloth
   zai-org
   openai
   ibm-granite
   amd
   Intel
   moonshotai
   ```

   It names authors only, never models, so it cannot go stale as models are
   published. The file is embedded, so it must exist for the build; emptying
   it — leaving only comments — zeroes the `trustedPublisher` term and is the
   supported way to opt out.

10. **Coarse rank** and keep the top 40. This score only decides who gets a
   `config.json` fetch, so it is deliberately crude. Every term is normalised
   to `[0,1]` across the pool and the weights sum to 1:

   ```
   coarse = 0.40·downloadsPercentile
         + 0.25·recencyPercentile      // over lastModified
         + 0.25·accelerated            // 1 if the format is accelerated here
         + 0.10·trustedPublisher       // 1 if the author is in publishers.conf
   ```

   Every percentile in this phase, here and in step 15, is computed over the
   set being ranked at that moment: the surviving pool for the coarse score,
   and the verified candidates for each objective. Saying so once avoids two
   denominators producing two different orders from the same data.

   Forty is enough that all four objectives have real choice, and small enough
   that the fetch stays a few seconds at 8 concurrent.

   Candidates below the cut are discarded here and never become `Candidate`
   values: they are out of the running, not judged and rejected. The three
   verdicts in step 14 describe the forty finalists only, which is what the
   feed shows and what the counts in the UI refer to.

11. **Config fetch**, 8 concurrent through a bounded worker pool. The cache is
    a directory per repository revision:

    ```
    <dataDir>/recommend/configs/<key>/config.json
    ```

    where `<key>` is `url.PathEscape(modelID) + "@" + url.PathEscape(lastModified)`.
    `PathEscape` is named rather than left to the implementer because the
    modelID contains a `/` and the timestamp contains `:`, and both must
    survive as a single path segment. A `config.json` is immutable for a
    revision, so a hit never expires and a new `lastModified` is simply a new
    key. On a fetch or write error the candidate becomes Unverified with that
    reason rather than failing the refresh.

12. **Exact fit.** Set `ParamsB` from `Safetensors.Total / 1e9` — the Hub's
    own count, so the parameter term of the **Best quality** objective rests
    on a reported figure rather than a derived one. It is 0 when the Hub
    reports no total; note that this is not the same condition as
    `WeightsKnown`, which is also false when a dtype is unrecognised even
    though `Total` is present, so the two are set independently rather than
    from each other — and `WeightGB` from `WeightBytes / 1024³`, computed
    once here so neither the objectives nor the template divide. Then call
    `models.ParseHFConfig(<the cache directory>)`, then
    `models.EstimateVRAM`, then `models.Fit` against `Profile.Inventory`.
    Before calling `Fit`, set `VRAMEstimate.CheckpointGB` from
    `Safetensors.WeightBytes()` divided by 1024³ — that field is in GB and
    means "what the weights actually occupy", and the Hub figure is exactly
    that, so supplying it stops the formula being consulted for a number
    already known exactly.

13. **Tensor-parallel validity.** `Fit` enumerates the widths the host can
    supply; discard any returned `TPOption` whose `TP` does not divide
    `HFConfig.NumKeyValueHeads`, before choosing a recommended width. A
    four-card host is then never told a six-KV-head model fits at TP=4. The
    recommended width is the lowest surviving `TPOption` with `Fits == true`.

14. **Verdicts.** Set `ArchSupported` first: true when `Profile.ArchsKnown`
    is true and `HFConfig.Architectures[0]` is in `Profile.Archs`, false in
    every other case including when the registry was never read. Then:
    - **Verified** — size known, config fetched and parsed, at least one valid
      `TPOption` with `Fits == true`, and either `ArchsKnown` is false or
      `HFConfig.Architectures[0]` is in `Profile.Archs`.
    - **Unverified**, with exactly one reason, first match wins:
      `size unknown — the repository publishes no safetensors metadata`;
      `config unavailable — could not fetch config.json`;
      `config unreadable — config.json is missing the fields the fit needs`;
      `architecture <Name> is not in this image's model registry`.
    - **Dropped** — a finalist for which no valid `TPOption` fits. Candidates
      the coarse filter removed at step 7, and those below step 9's cut, are
      not `Candidate` values at all and carry no verdict; they were never in
      the running rather than judged and rejected.

15. **The four objectives** in `score.go`. Each takes an already-scored
    candidate and returns `[0,1]`; no objective reads the network or refetches
    anything. `accel` is 1 when the format is accelerated here, else 0;
    `weightOnly` is 1 when `Candidate.WeightOnly` is set, else 0 — the field
    from step 2, never re-derived from the format badge here.
    `headroom` is `TPOption.SpareGB / TPOption.AvailableGB` at the recommended
    width. `tpNorm` is `(TP − minTP) / (maxTP − minTP)` over the host's
    available widths, or 0 when there is only one.

    ```
    quality = 0.70·paramPercentile + 0.20·headroom + 0.10·accel
    fastest = 0.50·(1 − tpNorm) + 0.30·headroom + 0.20·accel
              − 0.40·weightOnly                // clamped to [0,1]
    context = affordableContextPercentile
    newest  = recencyPercentile
    ```

    **Affordable context** is the honest figure and is computed, not taken
    from the model's claim:

    ```
    affordableTokens = min(
        HFConfig.MaxPositionEmbeddings,
        floor(TPOption.SpareGB × 1024³ / VRAMEstimate.KVCachePerTokenB) )
    ```

    `KVCachePerTokenB` is already populated by `EstimateVRAM`. A model
    advertising a million tokens it cannot hold does not win this axis. Store
    it as `Candidate.AffordableTokens`, which is what the endpoint serialises
    as `context_tokens` and what phase 19 reads rather than recomputing.

    Compute and store all four orders at refresh time, so a chip press is a
    slice read.

16. **Cache, TTL and staleness.** Hold the scored pool on the engine behind a
    mutex with the `Profile` it was computed against and its `generatedAt`.

    - A request when `pool` is nil — the first of the process's life — builds
      it synchronously and serves it with `stale: false`.
    - A request within 6 hours of `generatedAt` serves the pool with
      `stale: false`.
    - A request after 6 hours serves **the same pool** with `stale: true` and
      does not rebuild. Serving a known-good ordering that is a few hours old
      beats making the user wait for a network round trip they did not ask
      for, and `stale` plus `generated_at` let phase 18 say so and offer
      Refresh.
    - A request whose `Profile` differs from the stored one discards the pool
      and rebuilds synchronously. A pool judged against different hardware is
      wrong in a way the user cannot see, so it is never served. `Profile`
      holds a map and a slice and so is not comparable with `==`; give it
      `func (p Profile) Key() string` joining the card count, per-card GB, GPU
      arch, variant, sorted accelerated formats, `ArchsKnown` and the number
      of known architectures with `|`, and compare those. The arch count
      rather than the whole set: the set only changes when the image does,
      which the variant already captures.
    - `POST /api/recommend/refresh` always rebuilds, then returns exactly what
      `GET /api/recommend?intent=quality` would return — the same body shape,
      so phase 18 can render it directly or ignore it and re-trigger the feed
      load.

17. **Endpoint.** `GET /api/recommend?intent=quality|fastest|context|newest`.
    Unknown or absent intent falls back to `quality` rather than erroring.

    ```json
    { "profile": { "gpu_count": 4, "per_card_gb": 32, "total_vram_gb": 128,
                   "gpu_name": "Radeon AI PRO R9700",
                   "gpu_arch": "gfx1201", "variant": "rdna4-clav",
                   "accelerated": ["fp8"], "archs_known": true,
                   "inventory_known": true },
      "intent": "quality", "generated_at": "...", "stale": false,
      "unavailable": "",
      "verified": [ { "id": "...", "author": "...", "format": "FP8",
                      "weight_gb": 31.97, "params_b": 32.8,
                      "tp": 2, "required_gb": 41.2, "available_gb": 64.0,
                      "spare_gb": 22.8, "context_tokens": 131072,
                      "accelerated": true, "arch": "Qwen3ForCausalLM",
                      "arch_supported": true, "gated": false,
                      "downloads": 0, "last_modified": "..." } ],
      "unverified": [ { "id": "...", "author": "...", "format": "FP8",
                        "gated": false, "accelerated": true,
                        "arch_supported": false, "downloads": 0,
                        "last_modified": "...", "arch": "Qwen3ForCausalLM",
                        "reason": "size unknown — ..." } ] }
    ```

    Both entries carry every field in step 2's always-emitted group, which is
    why the unverified one still reports `accelerated`, `arch_supported`,
    `downloads` and `last_modified`: one struct produces both shapes, and only
    the `omitempty` fields differ between them. An unverified entry's
    `arch_supported` is false because no check succeeded, not because one
    failed — the card says nothing about architecture in that case.

    `arch` is the class name, set from the cheap tier's
    `ModelConfigMeta.Architectures[0]` when the pool is built and overwritten
    from `HFConfig.Architectures[0]` once the config is fetched and parsed.
    The fetched config wins because it is the file vLLM itself reads; the
    cheap-tier value exists so an unverified candidate can still name its
    architecture. They agree in practice, and where they do not, the one that
    was actually checked against the registry is the one displayed.

18. **Failure.** Any network error returns 200 with both lists empty and
    `unavailable` naming the cause. The feed must never take the page down,
    and the search box below it must keep working.

## Build gate

```
gofmt -l ./internal ./cmd
go build ./...
go vet ./internal/...
go test ./...
```

## Test plan

All tests use recorded Hub payloads in `internal/recommend/testdata/`, captured
once from the live API, so the suite makes no network calls. Those fixtures are
test data, not shipped ranking inputs.

- **Unit, acceleration.** `gfx1201` → fp8; `gfx1100` → none; unknown arch →
  none. A descriptor with `NoAccel: ["fp8"]` on `gfx1201` → none.
- **Unit, buckets.** A profile with nothing accelerated still produces the
  4-bit and unquantized buckets. This is the guard for the gfx1100 case.
- **Unit, GGUF.** A GGUF-only repository in the recorded payload never reaches
  the pool.
- **Unit, coarse filter.** A candidate larger than `Count × PerCardGB` does
  not survive the filter; one just under it does; one of unknown size does.
  Assert survival, not a verdict — the filter runs before verdicts exist.
- **Unit, TP divisibility.** A config with 6 KV heads on a 4-card host yields
  no TP=4 option, and the recommended width is 2.
- **Unit, verdicts.** Each of the four Unverified reasons is produced by the
  condition that should produce it, in first-match order, and each carries a
  non-empty reason.
- **Unit, arch unknown.** With `ArchsKnown == false`, a candidate whose
  architecture is in nobody's set is still Verified. This is the guard that
  stops a fresh install showing an empty feed.
- **Unit, objectives.** Over a fixed set of scored candidates, assert the first
  result of each of the four orders, and that a weight-only model never leads
  **Fastest**.
- **Unit, WeightOnly.** An `awq` config sets it; a `compressed-tensors` config
  with `format: pack-quantized` sets it; one with `format: float-quantized`
  does not; an absent `quantization_config` does not.
- **Unit, affordable context.** A model advertising 1,000,000 tokens with
  headroom for 140,000 ranks on 140,000, and one advertising 8,192 with room
  for far more ranks on 8,192.
- **Unit, config cache key.** A modelID containing `/` and a timestamp
  containing `:` round-trip to a single readable path segment, and a second
  scoring run with the same `lastModified` issues no fetch while a changed one
  does.
- **Unit, staleness.** A pool 7 hours old is served with `stale: true` and no
  rebuild; a pool whose profile card count has changed is discarded and
  rebuilt; a nil pool is built synchronously on the first request.
- **Unit, profile key.** Two profiles differing only in the order of
  `Accelerated` produce the same `Key()`; differing in card count, per-card
  GB, GPU arch, variant or `ArchsKnown` produce different ones.
- **Unit, unquantized bucket.** A recorded payload containing both an FP8 and
  a bf16 repository yields the bf16 one from the unquantized bucket and does
  not double-count the FP8 one, which the fp8 bucket already returned.
- **Unit, unavailable.** A client returning a network error, and separately a
  profile with `Inventory.Known == false`, each yield 200, empty lists and a
  populated `unavailable`.
- **Manual.** `curl /api/recommend?intent=quality` on the reference machine.
  Confirm the top result fits, its stated TP is plausible, and no result
  exceeds 128 GB. Then `intent=fastest` and confirm the order changes with no
  new Hub traffic, watched with a request log.

## Commit

```
feat(recommend): rank Hub models against this machine
```

## Rollback

Revert the commit and remove the two routes. `internal/recommend` is a leaf
that only `internal/api` imports, so nothing else is affected. The
`VARIANT_NO_ACCEL` reader in `variants/variants.go` is additive and harmless
if left. Delete `<dataDir>/recommend/` to reclaim the config cache. Phases 15
and 16 stand on their own and are not reverted with this.

# Phase 15 — Exact model sizes from the Hub

**Depends on:** nothing · **Enables:** phase 17's coarse filter, and every size
figure the recommendation feed states. Ships user-visible value on its own.

## Goal

Replace the parameter-count formula in `internal/huggingface` with the exact
per-dtype counts the Hub already publishes, and point the existing search and
detail views at it. Today `estimateParamCount` models one dense MLP per layer
— `4*h*h` for attention, `3*h*inter` for the MLP, no expert term — so a
mixture-of-experts checkpoint is understated by roughly its expert count. The
Hub returns the true figure, broken down by dtype, for one extra query
parameter.

Verified against the live API while planning:

```
GET /api/models?search=Qwen3-32B-FP8&limit=2&expand[]=safetensors
  "safetensors": {
    "parameters": { "BF16": 1558406144, "F8_E4M3": 31205621760 },
    "total": 32764027904 }
```

That is an exact weight size with no formula, no MoE correction and no
bits-per-parameter table: `1558406144 × 2 + 31205621760 × 1 = 34,322,434,048`
bytes, 31.97 GiB. The dtype breakdown also removes the need to infer a width
from the format badge, which is what `quantBits` was doing.

This phase is deliberately confined to the HuggingFace package and the one
template that renders its numbers. No new package, no hardware awareness.

## Files touched

- `internal/huggingface/client.go` — add `Safetensors` to
  `ModelSearchResult` and `ModelDetail`; add `expand[]=safetensors` to the
  search query; delete `estimateParamCount`, `quantBits` and `estimateVRAM`;
  compute weight bytes from the dtype map.
- `internal/huggingface/safetensors.go` — new. The `Safetensors` type, the
  dtype→bits table, and `WeightBytes()`.
- `internal/huggingface/safetensors_test.go` — new. Table tests for the dtype
  arithmetic, the MoE case, and the absent-metadata case.
- `internal/huggingface/client_test.go` — update any assertions that referenced
  the deleted estimators.
- `internal/api/hf.go` — `hfModelDetail` carries the new label instead of
  `VRAMLabel`; `newHFModelDetail` fills it; `formatVRAM` is deleted with the
  field it formatted.
- `web/templates/partials/hf_results.html` — the "Est. VRAM" cell in the
  `hf_model_detail` template becomes "Weights" and renders the exact figure,
  with a "size unknown" state. The `hf_results` template in the same file,
  which draws the search result list, shows no size and is not touched.
- `internal/api/testdata/golden/` — regenerate any fragment holding a size
  figure, with `go test ./internal/api/ -update`.

## Steps

1. Create `internal/huggingface/safetensors.go` with:

   ```go
   type Safetensors struct {
       Parameters map[string]int64 `json:"parameters,omitempty"`
       Total      int64            `json:"total,omitempty"`
   }
   ```

2. Add `bitsPerDtype map[string]int64`, in **bits** rather than bytes so the
   4-bit entries stay integers and the sum never needs a float: `F64`/`I64`
   64; `F32`/`I32` 32; `F16`/`BF16`/`I16` 16; `F8_E4M3`, `F8_E5M2`, `I8`,
   `U8` 8; `F4`, `I4`, `U4` 4; `BOOL` 8. An unrecognised dtype makes the whole
   figure unknown rather than being skipped — silently omitting a tensor class
   would understate the model, which is the failure this phase exists to
   remove.

3. Add `func (s Safetensors) WeightBytes() (int64, bool)`. It returns
   `(0, false)` when `Parameters` is empty or holds a dtype not in the table.
   Otherwise it sums `count × bitsPerDtype[dtype]` into an `int64` of bits and
   divides by 8 at the end, so a 4-bit checkpoint is exact rather than rounded
   per tensor class. The bool is the honest answer to "do we know", and it is
   what the Unverified bucket keys off in phase 17.

4. In `client.go`, add the field to both `ModelSearchResult` and
   `ModelDetail`:

   ```go
   Safetensors *Safetensors `json:"safetensors,omitempty"`
   ```

5. Add `&expand[]=safetensors` to the search URL in `search()`. Leave
   `config=true` exactly as it is — it is what `DetectQuantFormat` reads, and
   this phase does not change format detection.

6. Delete `estimateParamCount`, `quantBits` and `estimateVRAM`. Replace their
   two call sites in `GetModel` (`client.go:308` and `client.go:320`):
   `detail.ParameterCount` comes from `Safetensors.Total`, and `VRAMEstGB` is
   replaced by `WeightsBytes int64` plus `WeightsKnown bool`, both filled from
   `Safetensors.WeightBytes()`. Bytes rather than gigabytes: it is what the
   function returns, what `FormatBytes` consumes in step 8, and what phase 17
   compares against VRAM, so converting here would only mean converting back.
   Delete the `VRAMEstGB` field rather than keeping it populated differently —
   a field named for an estimate that now holds a measurement is a trap for
   the next reader.

7. `GetModel` fetches one repo, so it can ask for `expand[]=safetensors` on
   the single-model endpoint too. Do that rather than carrying the figure
   across from the search result, so a detail view opened by direct URL is
   just as exact.

8. In `internal/api/hf.go`, replace `hfModelDetail.VRAMLabel` with
   `WeightsLabel string`, filled in `newHFModelDetail` by passing
   `detail.WeightsBytes` to the existing `huggingface.FormatBytes` helper —
   the label is
   computed in Go so the template holds no arithmetic, which is how
   `SizeLabel` beside it already works. Leave it empty when `WeightsKnown` is
   false, and delete `formatVRAM` along with the field it formatted.

9. Update the `hf_model_detail` template in `hf_results.html`: rename the
   "Est. VRAM" cell to "Weights" and bind it to `WeightsLabel`. When it is
   empty, render `size unknown` in muted text with a `title` explaining that
   the repository publishes no safetensors metadata. Do not fall back to a
   formula. The `hf_results` template in the same file renders no size figure
   and needs no change.

10. Leave "Download Size" alone. It comes from the file tree and is already
    exact; it answers a different question (bytes to transfer, including the
    tokenizer and configs) from weight bytes.

## Build gate

```
gofmt -l ./internal ./cmd
go build ./...
go vet ./internal/...
go test ./...
```

## Test plan

- **Unit, dtype arithmetic.** `WeightBytes` over the verified Qwen3-32B-FP8
  map — `{"BF16":1558406144,"F8_E4M3":31205621760}` — returns exactly
  `34322434048` bytes. Assert that integer, not a rounded GB.
- **Unit, 4-bit exactness.** A map of `{"I4": 3}` returns `1` byte
  (12 bits / 8, truncating), not `2`. The point is that the sum happens in
  bits before the single division, so no tensor class is rounded on its own.
- **Unit, MoE.** A parameters map totalling ~125B returns the true figure. The
  point of the test is that nothing in the path consults `hidden_size` or an
  expert count, so construct it from the map alone.
- **Unit, unknown.** `Parameters` nil → `(0, false)`. `Parameters` containing
  an unrecognised dtype → `(0, false)`, not a partial sum. This is the
  assertion that stops a future dtype being silently dropped.
- **Unit, absent.** A `ModelDetail` with `Safetensors: nil` yields
  `WeightsKnown == false` and an empty `WeightsLabel`, and does not panic.
- **Template.** Render `hf_model_detail` for a known and an unknown model;
  assert the known one shows a figure and the unknown one shows the muted
  state, following the existing golden-fragment pattern. `ModelSearchResult`
  gains `Safetensors` in this phase for phase 17's benefit, but renders no
  size, so the `hf_results` template's golden fragment should be unchanged —
  assert that too.
- **Manual.** Search for a known MoE repository on the live Hub and confirm
  the weights figure matches the repository's own model card, rather than the
  ~10× understatement shown today.

## Commit

```
feat(hf): take model sizes from the Hub instead of a formula
```

## Rollback

Self-contained: revert the commit. The deleted estimators come back with it,
and nothing outside `internal/huggingface`, `internal/api/hf.go` and
`hf_results.html` has changed.

Steps 1–5 are purely additive and safe to leave applied on their own. Step 6
is not: it deletes `estimateParamCount`, `quantBits`, `estimateVRAM` and the
`VRAMEstGB` field, so from step 6 onward the tree does not build until step 8
has replaced the field on `hfModelDetail` and step 9 has rebound the template.
Treat steps 6–9 as one unit.

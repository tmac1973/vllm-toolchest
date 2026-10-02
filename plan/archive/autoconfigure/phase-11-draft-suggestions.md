# Phase 11 — Offer the draft model a card asks for

**Depends on:** phase 08 (`WantDraft`, `DraftRepo`), phase 09 (`Result`),
phase 10 (the review) · **Enables:** speculative decoding in one more pass
for a model whose draft is not downloaded yet

## Goal

Phase 08 pairs a draft that is already installed. When the card recommends
speculative decoding with a draft that is *not* installed, the review so far
only says so. This phase finds the repository, lists it in the review with its
size and a Download button, and makes the next run pair it -- without reading
the card again, because the advice from the first run is kept.

## Files touched

- `internal/autoconfig/drafts.go` — new. `Hub`, `DraftSuggestion`,
  `FindDraftSuggestions`.
- `internal/autoconfig/drafts_test.go` — new.
- `internal/autoconfig/run.go` — `Deps.Hub`, `Deps.Installed`,
  `Result.Suggestions`, the call.
- `internal/api/autoconfig.go` — the download handler; suggestions in the
  review view.
- `internal/api/autoconfig_run.go` — pass `s.hfClient` as the `Hub`.
- `internal/api/server.go` — the route.
- `web/templates/partials/autoconfig_review.html` — the suggestions block.
- `web/templates/layout.html` — `.autoconfig-suggestion`.
- Golden recordings for the review with a suggestion.

## Steps

1. `drafts.go`:

   ```go
   type Hub interface {
       GetModel(ctx context.Context, modelID string) (*huggingface.ModelDetail, error)
   }
   type DraftSuggestion struct {
       Repo      string
       SizeLabel string // from the repository's downloadable files
       Method    string
       Why       string
       Installed bool   // already on disk but not compatible; see step 3
   }
   func FindDraftSuggestions(ctx context.Context, hub Hub, target *models.Model,
       wantMethod, draftRepo string, installed func(repo string) *models.Model) ([]DraftSuggestion, error)
   ```

2. Which repository. Only a repository the card itself names is suggested:
   `draftRepo` from phase 08, which is either the model value of the card's
   speculative config when that is `owner/name`, or the helper's `draft_repo`
   with a verified quote. There is no search by name and no guessing from the
   model's family: a drafter trained for a different checkpoint loads and then
   fails minutes in, and a wrong suggestion with a Download button on it is
   worse than none. When `draftRepo` is empty the function returns nothing and
   the existing note stands.

3. `hub.GetModel(ctx, draftRepo)`. A repository that does not exist returns
   no suggestion and no error: the card was wrong or out of date. Any other
   failure -- the network, a rate limit -- returns the error, and the run
   notes "The draft the card names, <repo>, could not be checked (<error>)."
   instead of claiming it does not exist. Otherwise one
   suggestion with the total size of its downloadable files. If that
   repository is already installed (it was not chosen in phase 08 because
   `DraftMismatch` was non-empty, or its draft method is not the one wanted),
   the suggestion is marked `Installed`, its `Why` is the mismatch sentence or
   "it is a <method> draft and the card asks for <wanted>", and the template
   shows it without a button.

4. `Deps` gains `Hub Hub` and `Installed func(repo string) *models.Model`;
   the server passes `s.hfClient` and a function returning the registry model
   with that ID, or nil. `Run`: after `Validate`, when `checked.WantDraft != ""`
   and `d.Hub != nil`,
   `Progress("Looking for the draft model the card recommends")` and store the
   suggestions on the result. The speculative note ends with exactly one of
   these, the first that applies:
   1. the Hub could not be asked: "The draft the card names, <repo>, could
      not be checked (<error>)." (step 3);
   2. a suggestion that is not installed: "It can be downloaded below.";
   3. only an installed suggestion: "An installed copy, <repo>, cannot be
      used: <why>." (step 3);
   4. a repository was named and not found: "The card names <repo>, but the
      Hub has no such repository.";
   5. none was named: "The card does not name a repository for it."

5. Review block, under the card table, when there are suggestions: the
   repository, the method, the size, the reason, and a button, each
   suggestion in its own `<div class="autoconfig-suggestion">`:

   ```html
   <button hx-post="/api/models/autoconfig/draft?id={{$.ModelID}}&repo={{.Repo}}"
           hx-target="closest .autoconfig-suggestion" hx-swap="outerHTML">Download</button>
   ```

6. `POST /api/models/autoconfig/draft`. The `repo` parameter must equal a
   suggestion on the held result for that model -- the handler never starts a
   transfer for an arbitrary repository from a query string. It calls
   `s.startTransfer(ctx, repo, "", false)`, returns the existing
   `download_started` partial for that transfer, and sets
   `HX-Trigger: downloadsChanged`. `transferBlocker`'s refusals (no space, a
   transfer already running) are shown in place of the button.

7. `.autoconfig-suggestion` in `layout.html`: a bordered block in the
   review's width, with the button on the right.

8. After the download. The draft registers through the normal completion
   hook and is recognised as a draft by `ParseHFConfig`. Nothing is
   reconfigured automatically. The suggestion's text says what to do: "When it
   has finished, run Autoconfigure again to pair it." That second run finds
   the same card hash, reuses the stored advice, touches no engine, and phase
   08's rule 12 now finds a compatible draft and proposes the speculative
   config. The earlier reading comes from phase 09's `lastAdvice`, or from the
   Autoconfig profile if the first result was saved; only a server restart
   with nothing saved makes the second run read the card again.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
go test ./internal/api -run 'TestGoldenPartials' -update
```

## Test plan

- `FindDraftSuggestions` with a fake Hub: a named repository that exists gives
  one suggestion with its size; one that does not exist gives none; an empty
  `draftRepo` gives none and makes no request; a repository already installed
  but mismatched is marked `Installed` with the mismatch sentence.
- `Run` with `WantDraft` set and a Hub: the suggestion is on the result and
  the note ends with the download sentence; with no repository named, the note
  ends with the other.
- The draft handler refuses a `repo` that is not on the held result, and one
  for a model with no held result.
- The draft handler with a fake Hub and downloader starts exactly one
  transfer for the suggested repository.
- Golden: the review with one downloadable suggestion and with one
  installed-but-incompatible suggestion.
- End to end in a test server: run with no draft installed (suggestion
  shown), register a compatible draft fixture, run again with the saved
  profile's advice: the helper fake is not called and the review has a ticked
  `speculative_config` row naming the draft's local path.

## Commit

```
feat(autoconfig): offer the draft model a card recommends when it is not installed
```

## Rollback

Revert the commit. The review returns to mentioning a missing draft as a note.
Drafts already downloaded are ordinary registry entries and are unaffected.

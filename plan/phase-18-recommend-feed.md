# Phase 18 — The recommendation feed

**Depends on:** phase 17 (the engine and its endpoint) · **Enables:** phase 19,
which needs a Download button on a card that knows its fit.

## Goal

Render the ranked result on the Download Models page: a "Recommended for this
machine" section above the search box, a row of four intent chips, verified
cards with their reasons, and an Unverified section below them. Search stays
exactly where it is and keeps working untouched.

The section leads with the profile it was computed against — `4× Radeon AI PRO
R9700 · 128 GB · gfx1201 · rdna4-clav` — so a recommendation that looks wrong
can be traced to the input that made it wrong. That line is the feature's own
falsifiability, and it is the reason the feed can be trusted at all.

## Files touched

- `web/templates/models_browse.html` — the recommend section above the
  existing search card; the section's CSS; the chip-switching wiring.
- `web/templates/partials/recommend_feed.html` — new. The profile line, the
  chips, the verified list, the unverified list, and the empty and
  unavailable states.
- `internal/api/recommend.go` — extend `handleRecommend` to render the
  partial for an htmx request and keep returning JSON otherwise, following
  the `handleServiceAdvice` pattern in `internal/api/advice_panel.go`.
  `handleRecommendRefresh` is deliberately left JSON-only; step 9 explains
  how the button is wired around that.
- `internal/api/recommend_render_test.go` — new. Template rendering across
  every state.
- `internal/api/testdata/golden/partial_recommend_feed*.html` — new golden
  fragments.
- `web/templates/help.html` — a short subsection under Models describing what
  the feed does and what "unverified" means.

## Steps

1. Add the section to `models_browse.html`, above the existing search
   `<article>`, as its own `<article class="recommend-feed">` containing a
   single `<div id="recommend-feed">`. The htmx attributes go on that div,
   not on the article:

   ```html
   <div id="recommend-feed"
        hx-get="/api/recommend?intent=quality"
        hx-trigger="load"
        hx-swap="innerHTML"></div>
   ```

   The div both is the swap target and carries the trigger, so a
   `htmx.trigger('#recommend-feed', 'load')` from elsewhere on the page
   reloads it. Putting the `hx-get` on the enclosing article instead would
   leave nothing listening on the target, and step 9's Refresh would appear to
   do nothing.

2. Render the profile line from the endpoint's `profile` object:
   `gpu_count`, `gpu_name`, `total_vram_gb`, `gpu_arch`, `variant` — giving
   `4× Radeon AI PRO R9700 · 128 GB · gfx1201 · rdna4-clav`. Omit the name
   when `gpu_name` is empty rather than leaving a gap. Unknown inventory is
   not a separate template state: phase 17 reports it through the same
   `unavailable` string as a network failure, so step 8's one state covers
   both and the template never renders a partial recommendation against
   invented hardware.

3. The four chips are buttons in a `role="group"`, each with
   `class="recommend-chip"`, `hx-get="/api/recommend?intent=..."`,
   `hx-target="#recommend-feed"` and `hx-swap="innerHTML"`. The class is what
   step 9's Refresh uses to find the active one. The active chip
   is marked with `aria-pressed="true"` and styled by that attribute rather
   than a class, so the state lives in one place. A chip press is a local
   request to this server, which reads back one of the four orders phase 17
   already computed: no HuggingFace traffic and no re-ranking. That is the
   overview's criterion, and the test plan checks it by watching for outbound
   Hub requests.

4. A verified card carries: repository id as the heading, the format badge
   reusing the existing badge partial, exact weight size from `weight_gb`, and
   the fit sentence. The sentence is three or four clauses joined by ` · `, in
   this order — the first three always present, the fourth only when the
   architecture was actually checked:

   | clause | when | text |
   |---|---|---|
   | fit | always | `fits at TP=<tp>` |
   | memory | always | `<required_gb> GB of <available_gb>` |
   | acceleration | `accelerated` true | `accelerated on <gpu_arch>` |
   | acceleration | `accelerated` false | `not accelerated on <gpu_arch>` |
   | architecture | `archs_known` true | `<arch> supported by this image` |
   | architecture | `archs_known` false | *omitted* |

   The acceleration clause is always present in one form or the other, because
   the overview promises each card states *whether* the format is accelerated
   here — silence would read as acceleration to anyone skimming, and a
   weight-only 4-bit model needs to say plainly that it buys memory rather
   than speed. The architecture clause is the one that can genuinely be
   omitted: when `archs_known` is false no check happened, and saying nothing
   is the honest report of that. A verified card is only verified on
   architecture when the check passed, so there is no "not supported" form.

   Both memory figures are totals across the cards in use at that width, not
   per-card shares — that is what `TPOption`'s own doc comment defines them as,
   so "41 GB of 64" at TP=2 means 41 of the 64 GB two cards provide. The
   architecture is named by its class, e.g. `Qwen3ForCausalLM`, because that
   is the name vLLM's registry is keyed by and therefore the name that was
   actually checked.

5. Gated repositories get a marker and a `title` explaining a token is
   needed, driven by the `gated` field phase 17 returns on both lists, so the
   feed does not recommend something the user cannot fetch. Match the wording
   the search results already use for a gated repository rather than inventing
   a second phrasing for the same condition.

6. The Unverified section is a separate `<details>`, collapsed by default,
   labelled with its count — *"7 could not be fully checked"*. Each row is the
   repository and its reason string. Collapsed by default because these are
   the ones we cannot vouch for, and expanding is the user asking to see them.

7. Each card gets a Download button posting to the existing
   `/api/hf/download` endpoint with the repository id, so the feed reuses the
   download path rather than growing one. Phase 19 extends what that download
   records; this phase only wires the button.

8. Two empty states, each a single explanatory line. When `unavailable` is
   non-empty, render it — that one string covers both no-network and
   GPUs-not-yet-read, because phase 17 reports them the same way. When
   `unavailable` is empty but `verified` is too, say the pool was built and
   nothing in it fits. Neither is an error page and neither hides the search
   box below.

9. A Refresh button in the section header, wired as
   `hx-post="/api/recommend/refresh"` with `hx-swap="none"` and
   `hx-on::after-request="document.querySelector('.recommend-chip[aria-pressed=\'true\']').click()"`.

   Two things are load-bearing here. The refresh route returns JSON, not the
   partial — phase 17 step 16 defines it that way and this phase extends only
   `handleRecommend` to render HTML — so `hx-swap` must be `none` or the raw
   JSON would be swapped onto the page. And the reload re-clicks the active
   chip rather than re-triggering the div directly: the div's own `hx-get` is
   fixed at `?intent=quality`, so triggering it would silently throw away a
   user's choice of Fastest, Longest context or Newest every time they pressed
   Refresh. Clicking the pressed chip reissues that chip's own request, so the
   intent survives with no second copy of the state.

   Show the `generated_at` age beside the button. When `stale` is true the section says so in
   one muted line — phase 17 serves a stale pool rather than rebuilding on its
   own, so this line plus the button is the whole of how a user gets a fresh
   one.

10. Style the section to match the Server tab's engine-notes strip: cards in
    the page flow, not a fixed-height pane, with the section growing to its
    content. Scope every selector under `.recommend-feed` so nothing leaks to
    other pages, the way `web/templates/server.html` scopes everything under
    `.server-top` and `.engine-notes`.

11. Add the help text. It should state plainly that the feed only shows what
    it believes will run; that Unverified means an input was missing rather
    than that the model is bad; and that every figure is an estimate computed
    before the model has ever run, which the engine's own measurements will
    refine once it has.

## Build gate

```
gofmt -l ./internal ./cmd
go build ./...
go vet ./internal/...
go test ./...
```

## Test plan

- **Template, verified.** A fixture with two verified candidates renders both
  fit sentences with real numbers and no empty clauses.
- **Template, acceleration clause.** An accelerated candidate renders
  `accelerated on gfx1201`; a non-accelerated one renders
  `not accelerated on gfx1201`. Neither is ever absent.
- **Template, arch clause omission.** A payload with `archs_known: false`
  renders no architecture clause, and no stray ` · ` separator is left behind
  — a dangling separator is the likely bug.
- **Template, unverified.** Each reason string reaches the page, and the
  `<details>` count matches the row count.
- **Template, empty states.** A populated `unavailable` (whether from unknown
  inventory or a network failure) renders one explanatory line and no cards; a
  pool with zero verified entries but a live profile renders its own line. In
  both cases assert the search box below is still present in the output.
- **Template, gated.** A gated repository renders its marker.
- **Render test** in the golden-fragment style already used for
  `partial_timings_list`, covering the verified, unverified and unavailable
  states.
- **Manual, Refresh keeps the intent.** Select Newest, press Refresh, and
  confirm the feed comes back ordered by Newest rather than reverting to Best
  quality — the failure this wiring exists to prevent.
- **Manual, no Hub traffic on chip switch.** Open the page with the container
  log visible, press all four chips, and confirm no outbound Hub request is
  logged after the initial load. This is the overview's success criterion and
  the one most likely to regress.
- **Manual, offline.** Block outbound network, reload, and confirm the page
  renders, the feed states why it is empty, and the search box below still
  submits.
- **Manual, narrow.** At 1280px the section stacks without horizontal scroll.

## Commit

```
feat(browse): recommend models that fit this machine
```

## Rollback

Revert the commit. `models_browse.html` returns to search-only and the
engine from phase 17 keeps serving JSON for anything else that wants it.
No data is written by this phase, so there is nothing to clean up.

Steps 2–11 build the partial, its states and its styling; step 1 is what makes
any of it reachable from the page. Applied in the written order, the section
exists before the partial it swaps into does, so treat steps 1–8 as one unit
and do step 1 last if the tree must stay servable in between. Steps 9–11
(Refresh, styling, help text) are additive and safe to leave applied alone.

# Phase 12 — A fix when the first start fails

**Depends on:** phase 01 (`SaveProfileFrom`), phase 08 (`process.RemoveFlag`),
phase 10 (the `.autoconfig-table` style) ·
**Enables:** acceptance (phase 14)

## Goal

A first guess on a model that has never run can be wrong. When a start fails
for a reason the engine itself explains -- the context does not fit the memory
left, the memory fraction asked for is more than is free, a flag is not
recognised -- the model's config panel shows a notice with the specific change
that answers it, and one button applies exactly that change. Nothing is
restarted and nothing is retried: the operator applies the fix and presses
Start.

The engine's explanations are already parsed by `internal/advice` and already
offered on the Server page, one field at a time. This phase attaches them to
the model they are about and widens what can be applied.

## Files touched

- `internal/api/start_fix.go` — new. `startFix`, `(*Server).startFixFor`,
  `handleApplyStartFix`.
- `internal/api/start_fix_test.go` — new.
- `internal/api/advice_apply.go` — `applyToConfig` learns
  `extra_flags_remove`.
- `internal/api/models_config_view.go` — `modelConfigView.StartFix`.
- `internal/api/server.go` — the route.
- `web/templates/partials/model_config.html` — the notice, above the VRAM
  banner.
- Golden recordings for the config panel with a fix.

## Steps

1. When a fix exists. `func (s *Server) startFixFor(m *models.Model) *startFix`
   returns nil unless the process manager's status names this model
   (`status.ModelID == m.ID`) **and** the start did not succeed: the state is
   `error`, or it is `starting` with `StartFailed` set. The advice the manager
   holds belongs to the last start and is cleared by the next one, so the
   notice disappears by itself as soon as the operator starts anything.

2. What it proposes. Walk `s.process.Advice()` for items of severity error
   and build at most one change per field:

   | advice item | change |
   |---|---|
   | field `max_model_len` with a `Suggested` value (the KV-cache ceiling the engine reported) | set `max_model_len` to that value rounded down to a multiple of 1,024 |
   | field `gpu_memory_utilization` with a `Suggested` value | set it to that value |
   | field `extra_flags` with `Suggested` naming an unrecognised flag | remove that flag, and its value, from extra flags |
   | field `max_model_len` with no value (a plain out-of-memory) | set `max_model_len` to half the current value rounded down to 1,024, never below 2,048 |

   These four shapes are what `internal/advice` produces today, from its rules
   for "is larger than the maximum number of tokens", "free memory on device
   … less than desired", "unrecognized arguments" and "out of memory"
   (`advice.go:358-658`); this phase adds no parsing.

   When both `max_model_len` rows apply, the one with the engine's ceiling
   wins: it is a measurement, and halving is a guess. An unrecognised flag
   that is not in the model's extra flags -- it came from a mapped field such
   as the Mamba cache mode -- is not a change: the notice explains it and adds
   "It comes from the <label> setting; clear that in Configure." with no Apply
   for it, so `applyToConfig` is only ever asked to remove a flag that is
   there. Items with no field, and
   warnings and notes, are not changes; the first
   error item with no field is shown as the notice's explanation when nothing
   else applies, with a link to the Server page's log. A proposed value equal
   to the current one is dropped. With no changes and no explanation, return
   nil.

   ```go
   type startFix struct {
       Why     string          // the engine's line, verbatim
       Changes []startFixChange
   }
   type startFixChange struct {
       Field, Label, Current, Proposed, Reason string
   }
   ```

3. `applyToConfig` gains one case, `extra_flags_remove`, whose value is a flag
   name beginning with `--`: it sets
   `cfg.ExtraFlags = process.RemoveFlag(cfg.ExtraFlags, value)` and errors
   when the flag is not present. The existing three cases are unchanged.

4. The notice in `model_config.html`, rendered when `.StartFix` is set, above
   the VRAM banner: "The last start failed." then `Why` in a `<code>` line,
   then a small table of the changes (current, proposed, reason) using phase
   10's `.autoconfig-table`, then **Apply this fix**. When there are no
   changes it shows only the explanation.

5. `POST /api/models/autoconfig/fix?id=`. It recomputes `startFixFor(m)` --
   the form carries no values, so a stale page cannot apply something the
   engine no longer says -- refuses when it is nil or the registry is
   read-only, applies each change through `applyToConfig`, then
   `UpdateConfig`, the estimate recompute and `Register`, as
   `handleApplyAdvice` does. It re-renders the config panel through
   `renderConfigPanel` with the banner "Applied. Start the model again when
   you are ready."

6. Keep the Autoconfig profile in step. If the model's active profile is
   Autoconfig and unmodified (`ActiveProfile` reports it not modified), read
   that profile, set its `Config` to the new live config,
   append one note per change with origin "this machine", and write it back
   with `SaveProfileFrom`. The notes are:
   - "Context lowered from N to M after a start failed: the engine reported
     room for M tokens." (a ceiling from the engine);
   - "Context halved from N to M after a start ran out of memory." (a bare
     out-of-memory);
   - "GPU memory utilization lowered from X to Y after a start failed: the
     engine reported only Y of the card free.";
   - "Removed <flag> after a start failed: this image's vLLM does not
     recognise it."
   Without this the panel would show the profile as modified and restoring it
   would bring the failing value back. When the Autoconfig profile is active
   but modified by hand, the live change is made and the profile is left
   alone: copying the fix into it would also copy the hand edits, which the
   operator has not saved there. A model whose active profile is anything
   else also gets the live change only.

7. The notice describes the start that failed, so it must go away once the
   configuration has changed since, whether by this fix or by hand. Compare
   `process.BuildArgs(m.StartConfig())` with `status.Args`, as
   `EnsureModelLoaded` already compares arguments: when they differ,
   `startFixFor` returns nil. After an applied fix they always differ, so the
   notice clears without any stored state, and a halving fix cannot be offered
   twice for one failure.

8. There is no marker on the model's card. The failure happens minutes after
   Start was pressed and nothing pushes it to the Models page; the notice is
   where the operator looks next, in Configure, and the Server page's own
   status already shows the start failed.

9. The Server page's advice panel is left as it is. It shows the same engine
   lines for whatever was started last and remains the place for advice about
   a model that is running.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
go test ./internal/api -run 'TestGoldenFragments|TestGoldenPartials' -update
```

## Test plan

With a `fakevllm` script that prints canned failure output and exits
non-zero, as `measurement_test.go` does for successes:

- The engine's "is larger than the maximum number of tokens" line with a
  ceiling of 187,432: the fix proposes `max_model_len` 187,392 and nothing
  else. Applying it sets the live config, and the notice is gone on the next
  render.
- "unrecognized arguments: --enable-reasoning": the fix removes that flag and
  leaves the other extra flags intact.
- A free-memory line: the fix sets the utilization to the suggested value.
- A bare out-of-memory with context 131,072: the fix proposes 65,536; at
  2,048 it proposes nothing and shows only the explanation.
- A failure with no recognised line: explanation only, no Apply button.
- The same failures for a *different* model than the one rendered: no notice.
- A successful start: no notice.
- Active profile Autoconfig: after applying, the profile's config equals the
  live config, `ActiveProfile` reports it unmodified, and the profile has one
  more note. Active profile "Mine": the profile is untouched.
- Applying on a read-only registry refuses and changes nothing.
- A stale page posting after a new start has cleared the advice gets the
  "nothing to apply" message.
- Goldens for the panel with a one-change fix and an explanation-only notice.
- After applying the halving fix, a render with the same failed status shows
  no notice.

## Commit

```
feat(autoconfig): offer a fix on the model when its start fails
```

## Rollback

Revert the commit. The Server page's advice panel still offers the engine's
suggestions one field at a time, as it did before.

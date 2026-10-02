# Phase 04 — Know which flags this image accepts

**Depends on:** nothing · **Enables:** marking a card's flags the image does
not have before they cost a failed start (read by the run, phase 09)

## Goal

Autoconfigure takes the whole `vllm serve` command from a model card. A card
written for a newer or a different image can name a flag this one does not
have, and today the only way to find out is a start that dies on
`unrecognized arguments`, one flag at a time. This phase asks the installed
vLLM for its accepted flags once per image, caches the list, and gives the rest
of the server a way to ask "does this image have `--x`". When the list cannot
be obtained, every flag passes, as `validateNamedBackends` already behaves for
an unknown variant.

## Files touched

- `internal/vllmenv/flags.go` — new. `(Env).ProbeServeFlags`, the help-output
  parser.
- `internal/vllmenv/flags_test.go` — new, using the `stubPython`-style
  executable stub already in `env_test.go`.
- `internal/vllmenv/testdata/serve-help-0.27.txt` — new. The output of
  `vllm serve --help=all` captured once from the 0.27 image on the
  workstation.
- `internal/api/serve_flags.go` — new. The cache file, `(*Server).serveFlags`,
  the `flagSupport` type.
- `internal/api/serve_flags_test.go` — new.
- `internal/api/server.go` — start the probe in the background at boot.

## Steps

1. `flags.go`: `func (e Env) ProbeServeFlags(timeout time.Duration) ([]string, error)`.
   Build the command with `e.ServeCommand("--help", nil)`, which yields the
   variant's launcher followed by `--help` in the model-path position -- for
   the generic launcher, `vllm serve --help`. Run it the way
   `ProbeDeviceName` runs its child: own process group, `cmd.Cancel` killing
   the group, `WaitDelay` of five seconds, `PYTHONUNBUFFERED=1`. Capture
   stdout and stderr together; argparse writes help to stdout, some wrappers
   to stderr.

2. If the captured text contains `--help=` (newer vLLM prints a grouped summary
   and offers `--help=all`), run once more with `--help=all` in place of
   `--help` and use that output instead.

3. `func parseServeFlags(help string) []string`: every match of
   `--[a-z][a-z0-9-]*` in the text, de-duplicated and sorted. Negated forms
   need no special handling: argparse prints a boolean pair as
   `--x, --no-x`, and the pattern finds both. Return an error from
   `ProbeServeFlags` unless the result has at least 20 flags and includes both
   `--max-model-len` and `--tensor-parallel-size`: anything less is not a help
   listing and must not be trusted as one.

4. `serve_flags.go`, the cache. File `<DataDir>/cache/serve-flags.json`:

   ```json
   {"variant": "rdna4-clav", "variant_version": "…", "probed_at": "…", "flags": ["--…"]}
   ```

   It is valid only while `s.vllmEnv.VariantVersion` is non-empty and both
   `variant` and `variant_version` equal `s.vllmEnv.Variant` and
   `s.vllmEnv.VariantVersion`. Written with tmp-then-rename. A variant that
   follows `:latest` changes its stamp on rebuild, which retires the cache. An
   image with no stamp is never written to the file, because nothing would
   say when the list went stale; its probe result is held in memory for the
   life of the process instead, and it is probed again on every boot.

5. `type flagSupport struct { known bool; set map[string]bool }` with
   `func (f flagSupport) Known() bool` and
   `func (f flagSupport) Has(flag string) bool`. `Has` strips a `=value`
   suffix, returns true whenever `known` is false, and otherwise looks the
   flag up. The always-present flags `BuildArgs` emits are not special-cased:
   if the image's help lists them they are in the set.

6. `func (s *Server) serveFlags() flagSupport` returns the in-memory list
   when this process has probed, and otherwise the file's list when valid. It never probes inline -- a caller in a request must not wait on a
   Python import -- and returns the unknown value when there is no valid
   cache yet.

7. `func (s *Server) refreshServeFlags()` loads the cache, and when it is
   missing or stale runs `ProbeServeFlags(90 * time.Second)`, stores the
   result and logs the flag count at info level. A failure is logged at warn
   with the first line of output and leaves support unknown. Call it in a
   goroutine from `NewServerWithEnv`, after the process manager exists.
   Guard with a mutex so a second call while one is running returns at once.

8. The probe runs whatever the engine is doing. Printing help returns from
   argument parsing before any device is touched, so it neither needs the
   cards free nor disturbs a serving model. The manual check in the test plan
   confirms that on real hardware; a probe that claimed VRAM would be a defect
   to fix in the probe, not a reason to schedule it.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

## Test plan

Automated:

- `parseServeFlags` on a captured `vllm serve --help` from 0.27 (checked in as
  `internal/vllmenv/testdata/serve-help-0.27.txt`) finds `--max-model-len`,
  `--kv-cache-dtype`, `--enable-auto-tool-choice`, `--no-async-scheduling` or
  its positive form, and no token that is not a flag.
- A stub launcher printing the grouped summary first and the full list on
  `--help=all` is called twice and yields the full list.
- A stub printing three lines yields the "not a help listing" error.
- A stub that never exits is killed at the timeout and the call returns.
- `flagSupport`: unknown passes everything; known rejects `--made-up` and
  accepts `--max-model-len=4096`.
- Cache: a file for another variant or another version is ignored; a valid one
  is returned without running anything; a probe failure writes no file.

Manual, on compute, once:

1. With the engine stopped, watch `GET /api/monitor/` while the server boots
   and probes: per-card `vram_used_mb` must not rise. Repeat with a model
   serving: the serving model must be unaffected.
2. Read `serve-flags.json`: the count is in the hundreds and includes
   `--enable-expert-offload`, a flag only this image has.

## Commit

```
feat(vllmenv): learn which serve flags the installed vLLM accepts
```

## Rollback

Revert the commit and delete `<DataDir>/cache/serve-flags.json`. With the code
gone the file is inert; with the file gone the code re-probes. Nothing reads
flag support until phase 09.

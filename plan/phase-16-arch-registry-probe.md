# Phase 16 — Architecture registry probe

**Depends on:** nothing · **Enables:** phase 17's architecture verdict. Can be
built in parallel with phase 15; numbered second because phase 15 ships
user-visible value and this does not.

## Goal

Learn which model architectures the running image's vLLM can actually load,
and cache the answer. This is the one input the recommendation feed needs that
the Hub cannot supply: architecture support is version-specific and lives
inside the container. This repository already records a case of it mattering —
`variants/rdna4-clav.conf` explains that the 28.04.9 image was needed because
it carries `qwen4_exp`, and a live Hub query during planning returned a
`Qwen4ExpForConditionalGeneration` repository, so the drift is real and
current.

Without this, the failure mode is expensive: a sixty-gigabyte download
followed by a start that dies on *"Model architecture X is not supported"*.

`vllmenv` already has a hardened probe harness for the device name, with
process-group creation, `SIGKILL` to the group and a `WaitDelay`, written
because importing vLLM spawns children that hold the stdout pipe open. This
phase reuses that shape. Unlike the device probe it never initialises the GPU
— it is an import and a registry read — so it costs seconds, not minutes, and
holds no HIP context.

This phase delivers the probe, the cache and an API endpoint. Nothing consumes
the result yet; phase 17 does.

## Files touched

- `internal/vllmenv/env.go` — add `supportedArchsProbe` (the Python snippet)
  and `func (e Env) ProbeSupportedArchs(timeout time.Duration) ([]string, error)`.
- `internal/vllmenv/env_test.go` — tests for output parsing and the empty case.
- `internal/api/arch_registry.go` — new. The `Server`-held state, its
  accessors, and `handleArchRegistry`.
- `internal/api/arch_registry_test.go` — new. The unknown state and the
  endpoint's shape.
- `cmd/vllmctl/main.go` — `readCachedArchs`/`writeCachedArchs` beside the
  existing `readCachedDeviceName`/`writeCachedDeviceName`, plus
  `resolveSupportedArchs`.
- `cmd/vllmctl/main_test.go` — cache key matching, stale rejection,
  missing-file tolerance.
- `internal/api/server.go` — register `GET /api/arch-registry` inside the
  existing `/api` route block.

## Steps

1. Add the probe body to `vllmenv`, as a `const` beside `deviceNameProbe`:

   ```python
   from vllm.model_executor.models import ModelRegistry
   for a in sorted(ModelRegistry.get_supported_archs()):
       print(a)
   ```

   No GPU initialisation, no `torch.cuda` call, no engine construction.

2. Add `ProbeSupportedArchs`, copying `ProbeDeviceName`'s process handling
   exactly: `exec.CommandContext` with the timeout, `PYTHONUNBUFFERED=1`,
   `SysProcAttr{Setpgid: true}`, `Cancel` sending `SIGKILL` to `-pid`, and
   `WaitDelay: 5 * time.Second`. Split stdout on newlines, trim, drop empties.
   Return an error when the list comes back empty — an empty registry is a
   broken probe, not a vLLM that supports nothing, and treating it as data
   would mark every model unverified.

3. Write `readCachedArchs` and `writeCachedArchs` in `cmd/vllmctl/main.go`,
   beside the device-name pair they are modelled on — the cache is read and
   written only by the boot goroutine, which lives in `package main`, and an
   unexported helper in `internal/api` would not be callable from there. The
   file format:

   ```
   <variant> <stamp version>
   ArchOne
   ArchTwo
   ...
   ```

   First line is the key. `readCachedArchs(dataDir, variant, version) []string`
   returns a single value and no error, returning nil for a missing file, an
   unreadable one, or one whose key does not match on both fields — the same
   shape as `readCachedDeviceName` in `cmd/vllmctl/main.go`, which returns
   `""` rather than an error for every one of those cases. A cache miss is not
   a failure; it just means probing. Path:
   `<dataDir>/arch-registry.txt`. `variant` is `env.Variant` and `version` is
   `env.VariantVersion`, the stamp `vllmenv.Detect` already reads — together
   they change whenever the image does, which is exactly when the registry
   can change.

4. Hold the result on `Server` behind a mutex, as a
   `struct { archs map[string]bool; known bool; source string }`. Expose
   `func (s *Server) SupportedArchs() (map[string]bool, bool)` for the verdict,
   and `func (s *Server) ArchSource() string` for the endpoint. The bool is
   "has the probe ever succeeded", and phase 17 uses it to decide between a
   real verdict and Unverified.

5. Add `func (s *Server) SetSupportedArchs(archs []string, source string)` for
   the boot goroutine to call, mirroring the existing `SetDeviceName`. `source`
   is `"cache"` when the archs came from the file and `"probe"` when they came
   from a fresh run. The field's zero value is the empty string, which
   `ArchSource` reports as `"none"` so the endpoint never has to infer where
   its data came from.

6. In `main.go`, after the existing device-name goroutine, add
   `go resolveSupportedArchs(cfg, env, srv)`. It reads the cache first and
   calls `srv.SetSupportedArchs(archs, "cache")` immediately on a hit; on a
   miss it calls `env.ProbeSupportedArchs(90 * time.Second)`, then
   `SetSupportedArchs(archs, "probe")`, then writes the cache. Ninety seconds
   rather than the device probe's three minutes: that one initialises
   hardware and this one is an import. On probe failure it logs at `Warn` and
   returns, leaving `known` false. It must never be fatal — a machine with no
   venv at all must still serve the UI.

7. Add `GET /api/arch-registry` returning
   `{"known": bool, "count": int, "archs": [...], "source": "probe"|"cache"|"none"}`.
   This is the phase's verifiable surface and the way to confirm the probe
   worked without reading logs.

## Build gate

```
gofmt -l ./internal ./cmd
go build ./...
go vet ./internal/...
go test ./...
```

## Test plan

- **Unit, parsing.** Probe output with blank lines, trailing newline and
  surrounding whitespace parses to a clean sorted list. Empty output returns
  an error rather than an empty list.
- **Unit, cache key.** A file whose first line names a different variant
  returns nil. Same for a different stamp version. A matching file returns the
  archs verbatim.
- **Unit, missing file.** `readCachedArchs` against a path that does not exist
  returns nil and no error, the way `readCachedDeviceName` does.
- **Unit, unknown state.** A `Server` that has never been given archs reports
  `known == false`, and `/api/arch-registry` returns `known: false` with a 200
  rather than an error.
- **Manual, on the real image.** Start the container, wait for the goroutine,
  and `curl /api/arch-registry`. Confirm the count is in the low hundreds and
  that `Qwen4ExpForConditionalGeneration` is present on the `rdna4-clav`
  image and absent on an older one. That contrast is the whole point of the
  probe and is the check that proves it is reading something real.
- **Manual, cache.** Restart and confirm `source` is `cache` and the container
  did not re-probe.

## Commit

```
feat(vllmenv): read the image's supported model architectures
```

## Rollback

Revert the commit. The cache file is inert once nothing reads it, but delete
`<dataDir>/arch-registry.txt` for cleanliness. Nothing else depends on this
phase yet, so it is safe to revert independently of phase 15.

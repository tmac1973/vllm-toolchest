# Phase 06 — Borrow the engine and give it back

**Depends on:** nothing · **Enables:** loading the helper model without
leaving the machine in a different state (phase 09)

## Goal

vLLM serves one model and holds the cards while it does. To read a model card
the helper has to be loaded, which means whatever is serving has to stop, and
the operator chose that it comes back afterwards. Nothing in the project does
this today: benchmark jobs leave the engine on their last configuration, the
context probe requires it stopped and never restarts it, and no subsystem
checks whether another is using the engine. This phase adds the one
abstraction -- stop what is serving, start something else, do a piece of work,
stop it, restart what was serving -- and the busy check that keeps two such
activities from colliding.

## Files touched

- `internal/api/engine_lease.go` — new. `engineLease`, `(*Server).engineBusy`,
  `(*Server).borrowEngine`.
- `internal/api/engine_lease_test.go` — new, against the `fakevllm` script
  pattern in `measurement_test.go`.
- `internal/api/server.go` — add the `lease engineLease` field.
- `internal/api/service.go` — Start, Stop and Restart handlers refuse while the
  engine is borrowed.
- `internal/api/bench.go`, `internal/api/bench_probe.go`,
  `internal/api/tuning.go` (`handleStartTuning`) — refuse to start while the
  engine is borrowed.

## Steps

1. `engineLease` is a mutex and a `holder string` ("" when free). Methods:
   `take(holder string) bool`, `release()`, `heldBy() string`.

2. `func (s *Server) engineBusy() string` returns a sentence for the first of
   these that is true, and `""` otherwise:
   - `s.benchSvc.ActiveRunID()` or `ActiveJobID()` reports one — "A benchmark
     is running and using the GPUs."
   - the context probe is active (`s.probe`'s active flag) — "A context probe
     is running."
   - `s.tuner.ActiveJob()` is non-nil (`internal/tuning/runner.go:139`) —
     "Kernel tuning is running."
   - `s.lease.heldBy() != ""` — "Autoconfigure is reading a model card."
   - the process state is `starting` or `stopping` — "The engine is still
     starting." / "The engine is still stopping."

3. The loan:

   ```go
   type engineLoan struct {
       ModelID   string   // what the process manager records as running
       ModelPath string
       Args      []string
       Env       []string
       StartWait time.Duration // how long to wait for it to come up
   }

   // work receives the engine's base URL once it is healthy.
   // err is the loan's own failure: the engine would not stop, the loaned
   // model would not start, or work failed. restore reports anything that
   // went wrong giving the engine back, and is "" when nothing did. They are
   // kept apart because the caller's result depends on the first and not on
   // the second: a helper that answered is still an answer if the model it
   // displaced could not be restarted.
   func (s *Server) borrowEngine(ctx context.Context, holder string, loan engineLoan,
       progress func(string), work func(ctx context.Context, baseURL string) error) (err error, restore string)
   ```

4. `borrowEngine`, in order:
   1. `engineBusy()` non-empty returns it as an error. Then `lease.take`.
      Release in a `defer`.
   2. Read `s.process.GetStatus()`. If the state is `running`, remember
      `prev = status.ModelID`; otherwise `prev = ""`. A state of `error` means
      nothing is serving and nothing is to be restored.
   3. If `prev != ""`: `progress("Stopping " + display name)`,
      `s.process.Stop()`, then poll `GetStatus` every second until `stopped`,
      for at most 60 seconds (the manager itself kills at 30). If it has not
      stopped by then, return "the engine did not stop within a minute" without
      starting anything, and attempt no restore: the model is still up or on
      its way down, and starting another would fail.
   4. `progress("Starting the helper model")`. Call `s.process.Start` with the
      loan directly -- not `startModel`, which would watch for a measurement
      and store it against a registry model. Poll every two seconds until the
      state is `running`; return an error on `error`, on `StartFailed`, on
      `Overdue`, when `StartWait` elapses, or when `ctx` is done. Include the
      manager's `Error` text.
   5. Call `work(ctx, "http://<cfg.VLLMHost>:<cfg.VLLMPort>")`.
   6. In a deferred function armed when step 4 begins -- so it runs whether or
      not anything was serving, and never after step 3's timeout, when nothing
      was started:
      if the state is `running` or `starting`, `progress("Stopping the helper
      model")`, `Stop()` and wait for `stopped` as in step 3; if it is
      `error`, there is nothing to stop. If the helper has not stopped within
      60 seconds, set `restore` to "the helper model did not stop, so <name>
      was not restarted" and restore nothing. Then, when
      `prev != ""`:
      - if that model is still in the registry and not orphaned,
        `progress("Restarting " + display name)` and `s.startModel(m)`. Do not
        wait for it to become ready: a large model takes minutes to reload and
        the caller's result does not depend on it. If `startModel` refuses,
        log it and set `restore` to "<name> was not restarted: <reason>";
      - otherwise set `restore` to "<id> was not restarted: it is no longer in
        the registry".
   7. The deferred restore must use a fresh context with its own 90-second
      bound, not `ctx`: when the caller's context has timed out or been
      cancelled, that is exactly when restoring matters.

5. `Restart` is not used anywhere in this path. `Manager.Restart` calls `Stop`
   first, and `Stop` returns an error from the `error` state, so a restart
   straight after a failed helper start would fail; `Stop` when running and
   `Start` otherwise avoids it.

6. Guards in the other direction. While `s.lease.heldBy() != ""`:
   - `handleServiceStart`, `handleServiceStop` and `handleServiceRestart`
     answer with the busy sentence through the message partial they already
     use for refusals (HTTP 409 for non-htmx callers);
   - starting a benchmark run or job, a context probe, or a tuning job is
     refused the same way, at the top of each start handler: the benchmark
     handlers in `internal/api/bench.go` and `bench_probe.go`, and
     `handleStartTuning` in `internal/api/tuning.go`.
   These are the only cross-checks added. Making benchmark, probe and tuner
   aware of *each other* is not this phase's job.

7. The auto-start path at boot runs before any lease can be held and is left
   alone. `cfg.AutoRestart` is stored but not read by the process manager, so
   a deliberate stop here does not trigger anything.

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

## Test plan

With two `fakevllm` scripts -- one standing for a served model, one for the
loan -- each printing "Application startup complete." and sleeping:

- Nothing serving: the loan starts, `work` runs, the engine is `stopped`
  afterwards, nothing is restarted.
- A model serving: it is stopped, the loan runs, and afterwards the process
  manager reports that model `starting` or `running` again with its own args.
- `work` returns an error: the loan is stopped and the model is restored, the
  error is returned as `err`, and `restore` is empty.
- The loan's script exits non-zero at once (state `error`): `work` is never
  called, the model is restored, the error names the start failure.
- `ctx` is cancelled while waiting for the loan: the loan is stopped and the
  model is restored, using the fresh context.
- The served model is deleted from the registry during the loan: `work`'s
  success is returned with `err` nil, no restart is attempted, and `restore`
  says why.
- `engineBusy` reports each reason; a second `borrowEngine` while one is held
  is refused; the Start, Stop and Restart handlers refuse while held, and work
  again after release.
- No measurement is recorded against the loan's model ID.

## Commit

```
feat(api): borrow the engine for another model and restore what was serving
```

## Rollback

Revert the commit. The handler guards only ever fire while a lease is held,
and nothing takes one until phase 09, so this is inert on its own.

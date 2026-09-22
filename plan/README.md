# Plans

Working notes, one per phase. A phase document is written before the work and
corrected during it, so it records what was decided and why — including the
things that turned out to be wrong.

## Active

- [`phase-10-variant-expansion.md`](phase-10-variant-expansion.md) — image
  variants for NVIDIA, AMD and Intel accelerators. Implemented on the
  `variant-manifests` branch; its §7 lists what is still unproven.
- [`phase-11-config-profiles.md`](phase-11-config-profiles.md) — named,
  per-model snapshots of a launch config, and the `models.json` schema gate
  they needed first.
- [`phase-12-per-model-env.md`](phase-12-per-model-env.md) — per-model
  environment variables, overriding the machine-wide ones. Planned, not built.
- [`phase-13-engine-advice.md`](phase-13-engine-advice.md) — reading what the
  engine measured and what it suggests, out of its own log output. The first
  slice of the "optimizer"; acting on any of it is deliberately a later phase.
- [`phase-14-measured-vram.md`](phase-14-measured-vram.md) — take the VRAM
  figures from the engine's own report instead of deriving them. A reversal of
  phase 13's assumption that the estimator could be calibrated into
  correctness; §1 is the evidence that it could not.
- [`recommend-models-overview.md`](recommend-models-overview.md) — the shared
  definition for phases 15–19: a hardware- and runtime-aware "Recommended for
  this machine" feed on the Download Models page, generated from the Hub
  rather than from a shipped list. Planned, not built.
- [`phase-15-exact-model-sizes.md`](phase-15-exact-model-sizes.md) — take model
  sizes from the Hub's `safetensors` metadata instead of a dense-transformer
  formula that understates a mixture-of-experts checkpoint by its expert
  count. Ships on its own; everything after it depends on the figures.
- [`phase-16-arch-registry-probe.md`](phase-16-arch-registry-probe.md) — read
  the running image's supported model architectures out of vLLM's own
  registry, by the probe-and-cache pattern the device name already uses.
  Independent of phase 15.
- [`phase-17-recommend-engine.md`](phase-17-recommend-engine.md) — the
  `internal/recommend` package: the machine profile, the candidate pool, the
  staged ranking the Hub's two-tier API forces, and the four objectives. Ends
  at a JSON endpoint, with no UI.
- [`phase-18-recommend-feed.md`](phase-18-recommend-feed.md) — the feed itself:
  the profile line that makes a wrong recommendation traceable, the four
  intent chips, and the verified and unverified lists.
- [`phase-19-fit-handoff.md`](phase-19-fit-handoff.md) — seed a downloaded
  model's config from the fit that ranked it. §"What seeding is not" records
  why measurement does not supersede it, which an earlier draft assumed.
- [`todo.md`](todo.md) — everything with no phase of its own.

## Archive

[`archive/`](archive/) holds the plans for work that has shipped. They are kept
rather than deleted because they explain why things are the way they are, which
is not recoverable from the code:

| | |
|---|---|
| [`implementation-plan.md`](archive/implementation-plan.md) | The original nine-phase outline |
| [`phase-01-scaffold-and-container.md`](archive/phase-01-scaffold-and-container.md) | Project scaffold, container foundation |
| [`phase-02-monitor-and-dashboard.md`](archive/phase-02-monitor-and-dashboard.md) | System monitor, dashboard |
| [`phase-03-huggingface-search-download.md`](archive/phase-03-huggingface-search-download.md) | HuggingFace search and download |
| [`phase-04-model-registry-and-config.md`](archive/phase-04-model-registry-and-config.md) | Model registry, per-model config |
| [`phase-05-process-management-and-service.md`](archive/phase-05-process-management-and-service.md) | vLLM process management, service control |
| [`phase-06-benchmarking.md`](archive/phase-06-benchmarking.md) | Benchmarking subsystem |
| [`phase-07-settings-and-config.md`](archive/phase-07-settings-and-config.md) | Settings, persistent configuration |
| [`phase-08-setup-script-and-polish.md`](archive/phase-08-setup-script-and-polish.md) | `setup.sh`, install flow |
| [`phase-09-nvidia-cuda-support.md`](archive/phase-09-nvidia-cuda-support.md) | NVIDIA CUDA as a second target |
| [`ui-parity-plan.md`](archive/ui-parity-plan.md) | Web UI parity with llama-toolchest |

Archived documents are left as they were written. Where one describes something
that has since changed — the compose file names in the phase 8 and 9 notes, for
instance, which moved from per-variant to per-vendor in phase 10 — the
document is the record of the decision at the time, not a description of the
code now.

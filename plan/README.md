# Plans

Working notes, one per phase. A phase document is written before the work and
corrected during it, so it records what was decided and why — including the
things that turned out to be wrong.

## Active

- [`todo.md`](todo.md) — where things stand, what is still open, and
  everything with no phase of its own.

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
| [`phase-10-variant-expansion.md`](archive/phase-10-variant-expansion.md) | Image variants for NVIDIA, AMD and Intel; §7 lists what was unproven |
| [`phase-11-config-profiles.md`](archive/phase-11-config-profiles.md) | Named per-model config profiles, the `models.json` schema gate |
| [`phase-12-per-model-env.md`](archive/phase-12-per-model-env.md) | Per-model environment variables |
| [`phase-13-engine-advice.md`](archive/phase-13-engine-advice.md) | Reading the engine's measurements and suggestions from its log (its §6 display was built later, as the advice panel) |
| [`phase-14-measured-vram.md`](archive/phase-14-measured-vram.md) | VRAM from the engine's own report, and why the estimator could not be calibrated into correctness |
| [`recommend-models-overview.md`](archive/recommend-models-overview.md) | The recommendation feed, phases 15–19 |
| [`phase-15-exact-model-sizes.md`](archive/phase-15-exact-model-sizes.md) | Exact model sizes from the Hub |
| [`phase-16-arch-registry-probe.md`](archive/phase-16-arch-registry-probe.md) | The image's supported architectures, probed and cached |
| [`phase-17-recommend-engine.md`](archive/phase-17-recommend-engine.md) | The recommend engine: profile, candidates, ranking |
| [`phase-18-recommend-feed.md`](archive/phase-18-recommend-feed.md) | The feed on Download Models |
| [`phase-19-fit-handoff.md`](archive/phase-19-fit-handoff.md) | Seeding a downloaded model through autoconfigure's planner |
| [`autoconfigure/`](archive/autoconfigure/overview.md) | Autoconfigure, its own phases 01–15; [`acceptance.md`](archive/autoconfigure/acceptance.md) records the hardware runs |

Archived documents are left as they were written. Where one describes something
that has since changed — the compose file names in the phase 8 and 9 notes, for
instance, which moved from per-variant to per-vendor in phase 10 — the
document is the record of the decision at the time, not a description of the
code now.

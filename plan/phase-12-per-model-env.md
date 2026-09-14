# Phase 12: Per-model environment variables

**Status:** implemented on the `per-model-env` branch, 2026-09-14, in the three
commits §8 lists. The row-editor question in §2 is still open and does not
block anything; the textarea shipped.

## About this document

Written 2026-09-14, out of a case where the existing environment layers could
not express what the model needed. Where this records a decision Tim made, it
says so; §2 marks the one question still open.

## 1. The case that produced it

`tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8` on the four-R9700 box will not start.
All four GPUs hit `torch.OutOfMemoryError` during layer construction, each
filling to 31.16 of 31.86 GiB inside the offloader: the model's 47.7 GB n-gram
embedding table stays on the GPUs. The model card avoids that with
`VLLM_PLE_CPU_OFFLOAD=1`, and pairs it with `CLAV_GDN=1`.

Neither is expressible today:

- `variants/rdna4-clav.conf` declares no `VARIANT_IMAGE_ENV`, and manifests are
  `go:embed`-ed, so adding one needs a rebuild and a redeploy.
- Settings → runtime environment would work, but it is **machine-wide**.
  `CLAV_GDN=1` would then apply to every model on that host, and
  `VLLM_PLE_CPU_OFFLOAD=1` is meaningless for any model without a PLE table.

The variable belongs to the model, so that is where it should be stored.

## 2. Decisions

| Question | Decision | Why |
|---|---|---|
| Scope | Per-model, overriding the machine-wide value of the same name | The motivating variables describe a checkpoint, not a host |
| Precedence | image manifest → machine-wide → variant knobs → **model** | The model is the most specific statement, so it wins |
| Storage | One string field on `VLLMConfig`, `KEY=VALUE` per line | `VLLMConfig` must stay comparable; see §3 |
| Validation | Reuse `EnvSet`: warn, never block | Matches the machine-wide box; an unknown name is the point of a free-form entry |

**Open, for Tim.** He asked for a row editor: a button to add a variable, and
edit/delete on each. This plan proposes a textarea first, for the reasons in
§5, with the row editor as a later change over the same stored string. The
storage and precedence below are the same either way, so this can be settled
after §3 and §4 are built.

## 3. Storage

`VLLMConfig` gains one field:

```go
	// Env is this model's own environment, KEY=VALUE per line, applied after
	// the machine-wide runtime environment and overriding it by name.
	Env string `json:"env,omitempty"`
```

**A string, not a map or a slice.** `VLLMConfig` is compared with `==` —
`Registry.ActiveProfile` (`internal/models/profiles.go:210`) reports profile
drift that way, and `pending_test.go:33`, `profiles_test.go` and
`models_profiles_test.go:107` all rely on it. A map or slice field stops the
package compiling. The comment on `ActiveProfile` already says that failure is
intended, and this is the first feature to meet it: the answer is to store the
env the way `Config.RuntimeEnvExtra` already stores its free-form block, not to
reshape the comparison.

Two things then come for free:

- **Profiles** capture it, because a profile snapshots `VLLMConfig`. "The
  config that produced these numbers" starts including the environment.
- **Backups** carry it, because they carry model configs.

Parsing reuses `parseExtraEnv` and `validEnvName` (`internal/config/runtime_env.go`)
unchanged; both are already the right shape and handle blank lines, comments
and malformed entries.

## 4. Precedence and the effective view

`launchEnv` (`internal/api/jobs_env.go` neighbours, see `launchEnv`) appends
image manifest, then machine-wide runtime, then knobs. The model's pairs append
last. `cmd.Env = append(os.Environ(), env...)` and os/exec resolves a duplicate
name to its last occurrence, so appending last *is* overriding — no merge step
is required for correctness.

A merge is still required to **show** it. `effectiveEnvLines`
(`internal/api/runtime_env_view.go:34`) already collapses the layers in launch
order, keeps the winning value, and annotates what each entry replaces from the
container's own environment. It gains a parameter for the model's pairs and
keeps every other property. The comment there about reading the direction the
wrong way round applies with more force once there are four layers.

The config panel then renders the same `effective_env` partial the Settings
page uses, showing what this model will actually launch with. That preview is
the part that prevents mistakes: a variable set in two places with different
values is otherwise invisible until the engine behaves oddly.

## 5. UI

A textarea in the Advanced fieldset, inside the autosaving form, named `env`,
placeholder `VLLM_PLE_CPU_OFFLOAD=1`. It autosaves exactly like every other
field, and the "Launch parameters apply on the next restart" line below the
form already covers it.

Below it, the effective-environment block for this model.

**Why not the row editor first.** Rows with add/edit/delete mean three
endpoints, a partial re-render, and escaping work, for data a textarea already
expresses. More to the point, every button that posts has to sit *outside* the
config form: the form autosaves on `change` with `hx-include="closest form"`,
so a control inside it fires a config save merely by being used, and its fields
ride along on every autosave. Phase 11 paid that cost deliberately for the
profile bar, where the payoff was a dropdown people browse. Here the payoff is
smaller and the same tax applies.

If the rows are still wanted afterwards, they are a pure presentation change:
parse the stored string into rows, render inputs, write it back. Nothing in §3
or §4 changes.

## 6. Validation

`EnvSet{Extra: cfg.Env}.Validate()` on save, with `Warnings()` surfaced in the
panel banner. Two additions to the existing warning list:

- `VLLM_WORKER_MULTIPROC_METHOD` and `PYTHONUNBUFFERED`, which
  `process.BuildEnv` sets for its own reasons — log capture and immediate
  error output. Overriding them breaks how vllmctl reads the engine, and the
  operator should be told rather than stopped.
- `HIP_VISIBLE_DEVICES` already has warning text; it applies here too.

Refusing is deliberately not on the table. The machine-wide box accepts unknown
names by design, and a per-model box exists for exactly the variables this
project has never heard of.

## 7. Benchmarks

`ConfigSnapshot` gains `Env string` as a recorded-only field, for the reason
the attention backend is recorded: it changes throughput, and two runs that
differ only in it would otherwise be indistinguishable. A `env` compare
dimension and CSV column follow the pattern phase 11 established — empty reads
as absent, so runs that set nothing add no column.

## 8. Commits

1. `VLLMConfig.Env`, parsing and the launch layer, with tests for precedence
   and for a name set in several layers at once.
2. The panel: textarea, the effective-environment preview, validation and
   warnings, golden regeneration.
3. The benchmark snapshot field, compare dimension and CSV column.

Planned as four; the effective-environment view and the textarea became one
commit because both change the config panel, and splitting them would
regenerate the golden recordings twice for one visible change.

Two things this turned up while being built:

- `effectiveEnvLines` re-implemented the same layering `launchEnv` builds.
  Adding a fifth layer to one of them would have let the preview drift from
  what actually launches, so both now read from one `configuredEnvPairs`.
- `launchEnv` took a quantization method; it takes the model now. The
  environment and the quant method have to come from the same record, and a
  caller that passed one and forgot the other would serve a model under an
  environment it was never configured with.

## 9. Testing

- Precedence: a name set in the manifest, in Settings and on the model resolves
  to the model's value, and `effectiveEnvLines` reports it once.
- A model with no env changes nothing about the launch environment.
- Round trip through the config PUT, including a malformed line, which warns
  and still saves the rest.
- The panel's autosave must not be disturbed: the existing
  `TestProfileControlsAreOutsideTheAutosavingForm` covers the profile bar; the
  env textarea is inside the form on purpose and needs the opposite assertion —
  that it is submitted with the form.
- Golden: the config panel gains the textarea and, for a fixture with env set,
  the effective block.

## 10. Out of scope

- The row editor (§5), pending Tim's call.
- Secrets. Anything put here lands in profiles and in backups, both of which
  are plain JSON. `HF_TOKEN` and friends stay in Settings, which already treats
  them separately.
- Per-model env for benchmark **jobs** as an overlay dimension. Recording it is
  in scope; sweeping it is not.

## 11. Related, not in this phase

The same investigation found `variants/rdna4-clav.conf` pinned to
`tcclaviger/vllm:28.02.2` (8 Sep) while the author's newest is `28.04.9`
(13 Sep), and the offload flags that model needs — `--enable-expert-offload`,
`--ple-nvme-offload` and their sizing options — only exist from `28.03.1` on.
Bumping a pin means re-verifying `VARIANT_VLLM_PIN` and `VARIANT_TUNER_REF`
against the new image, which is its own change. `VLLMCTL_BASE_IMAGE` already
overrides a variant's base image for a one-off test without touching the
manifest. A staleness check belongs in `setup.sh` — compare the manifest pin
against the registry and report — which `plan/todo.md` already carries as
"Support `setup.sh update`".

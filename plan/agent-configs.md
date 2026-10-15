# Coding agent configs

A port of llama-toolchest's *Agent Configs* (tmac1973/llama-toolchest#270,
commit 1b66ae4): a link in the Server tab's API Endpoint card opens a dialog
with downloadable config files that point opencode, Kilo Code, pi, Goose and
Crush at this server, each listing the served model with its context,
max output, and tool, image and reasoning support.

## What llama-toolchest built

| Piece | Where | What it does |
|---|---|---|
| `internal/agentconfig/agentconfig.go` | new package | `Model`, `Input`, `Agent` types; `MaxOutputFor` (ctx/4, clamped 1K–32K); `Agents` list, `Find`; comment `header` |
| `internal/agentconfig/agents.go` | | One generator per agent: opencode and Kilo (shared opencode JSON schema), pi (`models.json`), Goose (`custom_providers/*.json`), Crush (`crushrc` script) |
| `internal/api/agentconfigs.go` | | `GET /api/agent-configs` renders the dialog body; `GET /api/agent-configs/{agent}?base=` downloads one file. Base URL = `?base`, else `ExternalURL`, else the request's host; `/v1` appended; local-only hosts flagged |
| `web/templates/partials/agent_configs.html` | | Dialog body: URL field (re-renders on change), local-only warning, model list, API-key export line, one `<details>` per agent with download, save paths per OS, activation command, notes |
| `server.html`, `dashboard_cards.html` | | `<dialog>` + `openAgentConfigs()` fetching the partial on every open; *Agent Configs →* link |
| `server.go` | | `servedChatModels()` extracted so the Available Models card and the configs share one list |
| README, help page | | A *Coding agents* section and a help entry |
| tests | `agentconfig_test.go`, `agentconfigs_test.go` | Each generator's output parsed back; base-URL normalisation; local-only detection; 409 when nothing is served |

The API key is never written into a file: when one is set the files read it
from an environment variable, otherwise they carry the placeholder `none`
(several agents hide a provider whose key is empty).

## What is different here

1. **One model, not a router.** vLLM serves one model per process and nothing
   loads another on demand (see `handleLoadedModels`,
   `internal/api/capabilities.go`). The files therefore list exactly the model
   running now. There is no `servedChatModels()` to extract; the source is the
   same as `/api/service/loaded-models`: `process.GetStatus()` →
   `registry.Get(st.ModelID)` → `buildCapabilities(m, st.Args)`.
   The snapshot warning gets sharper: *switching the served model means
   downloading again.*
2. **The facts already exist.** `buildCapabilities` already reports, from the
   flags the engine was actually launched with:
   - `context_per_request` — the served `--max-model-len` (vLLM's KV pool is
     shared, so no division by parallel slots, unlike llama-toolchest);
   - `tools` — `--enable-auto-tool-choice` was passed;
   - `vision` — a vision model not started with `--language-model-only`;
   - `reasoning.supported` — from the chat template.
   `agentInput` maps those keys onto `agentconfig.Model` and does no detection
   of its own, so the configs and discovery can never disagree.
3. **Max output.** `MaxOutputFor` as in llama (ctx/4, 1K–32K). The plan was
   to lower it to the engine's default `max_tokens` when that is smaller; it
   is not, because that default only applies to a request that names none and
   caps nothing. An agent always sends its own.
4. **The helper model is excluded.** While autoconfigure has the helper
   (`models.HelperRepo`, served as `vllmctl-helper`) running, the dialog says
   the server is busy with autoconfigure rather than offering files for it.
   Drafts are never the served model, so need no check.
5. **Starting vs running.** Offer files only when the state is
   `StateRunning`. While starting, the dialog says so; the model is not
   answering yet and its launch args could still be refined.
6. **Names.** Provider ID/name `vllm-toolchest`; Goose ID `vllm_toolchest`;
   file names `vllm-toolchest.opencode.json`, `vllm_toolchest.json`,
   `vllm-toolchest.crushrc` (pi and Kilo keep their fixed `models.json` /
   `kilo.json`). Key variable **`VLLM_TOOLCHEST_API_KEY`** — *not*
   `VLLMCTL_API_KEY`, which is the server's own config override
   (`internal/config/config.go:333`); reusing it would make a client shell
   that sets it look like server configuration to anyone reading it.
7. **Request timeout.** vLLM never loads on first request, so Crush's
   `option request-timeout 600` is for long prompts on big models, not
   model loads; keep it and fix the comment.
8. **UI conventions.** This repo's dashboard card uses `copyFromButton(this)`
   with `data-copy` and `{{template "icon_copy"}}` rather than llama's
   `copy_button` template; the dialog follows the existing
   `engine-notes-modal` pattern in `server.html`.

## Steps

1. **`internal/agentconfig`** — copy both files from llama-toolchest, rename
   constants and file names per §6, adjust package doc and header text
   ("Generated by vllm-toolchest…", "the model the server was serving then").
   Port `agentconfig_test.go` (round-trip parse of every generator, key vs no
   key, `MaxOutputFor` bounds).
2. **`internal/api/agentconfigs.go`** — `agentInput` from the running model's
   capabilities (§1–5); `agentBaseURL`, `isLocalOnly` as in llama;
   `handleAgentConfigs`, `handleAgentConfigDownload` (409 with a reason when
   stopped, starting, or running the helper). Routes beside `/gpu-map` in
   `server.go`. Port `agentconfigs_test.go`, adding cases for the helper, a
   starting engine, tools off, and `--language-model-only`.
3. **Templates** — `partials/agent_configs.html` (model list becomes one
   line; reasons for no files per §4–5), the dialog and `openAgentConfigs()`
   in `server.html`, the *Agent Configs →* link in `dashboard_cards.html`.
   Add the partial to the golden-partials test.
4. **Docs** — README *Coding agents* section next to *Client
   auto-discovery*; help page entry.
5. **Verify on compute** (per the compute host notes): start
   Qwen3.8-Flash-Next with tool use on, download each file, and have at least
   opencode and one other agent make a tool call through it, with and without
   an API key. Record the agent versions tested in the README as llama does.

## Not in scope

- Listing every registered model so an agent can switch between them: every
  request for a model that isn't running fails here.
- Keeping files in sync automatically (e.g. an endpoint agents poll). The
  discovery API already serves clients that want that.

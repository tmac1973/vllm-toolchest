# Phase 05 — Read the model card and the command in it

**Depends on:** nothing · **Enables:** turning a card into proposals (phase
08)

## Goal

Get a model's card from the Hub, cut it down to what is about running the
model, and pull out every serve command it contains as structured data: which
flags map to a field this tool has, which do not, and which environment
variables came with it. None of this needs a helper model. It is deterministic
text handling, so a host too small for the helper still gets the card's
command, and the helper is left with the two jobs only a reader can do --
choosing between several commands, and reading advice written as prose.

The shapes handled here are the ones real cards use, checked on 2026-09-30
against the cards of the two models that serve on compute and of
`Qwen/Qwen3-14B-AWQ`:

- a `docker run … image:tag /app/models --flag …` inside an HTML
  `<pre><code>` block, with `-e KEY=VALUE` lines and HTML entities, and **no**
  `vllm serve` text at all, because the image's entrypoint is the server;
- two such commands in one card, for four cards and for two;
- a markdown fence holding several `vllm serve` commands separated by comment
  lines;
- a fragment such as `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ...
  --hf-overrides '{…}'`, where `...` stands for the rest of a command and the
  line documents one special-purpose flag;
- an option mentioned only inline in a sentence: "add
  `--speculative-config '{"method": "dflash", …}'`".

## Files touched

- `internal/huggingface/client.go` — add `ModelCard` and `BaseModel`.
- `internal/huggingface/card_test.go` — new, with an `httptest` Hub.
- `internal/process/parse_args.go` — new. `ParseArgs`, the inverse of
  `BuildArgs`, and `KnownServeFlags`.
- `internal/process/parse_args_test.go` — new.
- `internal/autoconfig/card.go` — new package. `Fetcher`, `Card`, `FetchCard`,
  `TrimCard`, `CardCharsForContext`.
- `internal/autoconfig/command.go` — new. `Command`, `ExtractCommands`,
  `CommandSettings`, `(Command).Settings`, `InlineFlags`.
- `internal/autoconfig/card_test.go`, `command_test.go` — new.
- `internal/autoconfig/testdata/` — excerpts of three real cards.

## Steps

1. `huggingface.Client`:
   - `func (c *Client) ModelCard(ctx context.Context, modelID string) (string, error)`
     — `GET <base>/<modelID>/raw/main/README.md` with `setAuth`. A 404 returns
     `"", nil`: a repository with no card is not an error.
   - `func (c *Client) BaseModel(ctx context.Context, modelID string) string`
     — `GET <base>/api/models/<modelID>?expand[]=cardData`, returning
     `cardData.base_model`. The field is a string in some repositories and a
     list in others; take the string, or the first element. Any failure
     returns `""`.

2. `process.ParseArgs(args []string) (cfg VLLMStartConfig, rest []string)`.
   Accept both `--flag value` and `--flag=value`. Map every flag in
   `BuildArgs` back to its field, using the table in `manager.go:656-755`:
   `--dtype`, `--max-model-len`, `--tensor-parallel-size`,
   `--gpu-memory-utilization`, `--enforce-eager`, `--trust-remote-code`,
   `--max-num-seqs`, `--quantization`, `--load-format`,
   `--enable-prefix-caching`, `--kv-cache-dtype`, `--enable-chunked-prefill`,
   `--max-num-batched-tokens`, `--enable-auto-tool-choice`,
   `--tool-call-parser`, `--reasoning-parser`, `--attention-backend`,
   `--mamba-cache-mode`, `--speculative-config`, `--compilation-config`,
   `--kv-cache-memory`, `--no-async-scheduling`, `--language-model-only`,
   `--tokenizer`, `--chat-template`, and `-tp` as the short form of the
   tensor-parallel size. Drop, without returning them in `rest`: `--host`,
   `--port`, `--served-model-name`, `--api-key`, `--model`, `--download-dir`
   and `--uvicorn-log-level`, each with its value, because this tool sets or
   owns them. Everything else goes to `rest`, a flag and its value kept
   together as consecutive elements. A flag is taken to have a value when the
   next element does not begin with `-`.
   `func KnownServeFlags() []string` returns the mapped and the dropped flags
   together; step 6 uses it to recognise a serve command that never says
   `vllm`.

3. `internal/autoconfig/card.go`, ported from llama-toolchest's
   `internal/autoconfig/card.go` with the same structure:

   ```go
   type Fetcher interface {
       ModelCard(ctx context.Context, modelID string) (string, error)
       BaseModel(ctx context.Context, modelID string) string
   }
   type Card struct {
       Text    string   // trimmed, what the helper reads
       Raw     string   // untrimmed, entities unescaped, all sources joined:
                        // what commands are extracted from and quotes are checked against
       Sources []string
       Hash    string   // sha256 of Raw, hex, first 16 characters
   }
   func FetchCard(ctx context.Context, f Fetcher, modelID string, maxChars int) Card
   ```

   Read the model's own card, then its base model's when `BaseModel` names a
   different repository. The model's own card comes first in both `Text` and
   `Raw`, because a quantizer's card is the one that says how to serve the
   quantized files. A model ID that is not `owner/name` has no Hub repository:
   return an empty `Card`.

4. `func TrimCard(md string) string` keeps llama-toolchest's rules -- strip
   front matter, HTML comments, image links and rows that are only links
   (badges); split at markdown headings, text before the first heading being a
   section of its own; keep a section only when it mentions a keyword; collapse
   three or more blank lines to two -- with three changes. HTML tags are
   stripped *after* `<pre>`, `<code>`, `<p>`, `<li>`, `<tr>` and `<br>` have
   been turned into line breaks, so an HTML card does not collapse into one
   line. Entities are unescaped with `html.UnescapeString`. And the keyword
   list is this engine's: `recommend, sampling, temperature, top_p, top-p,
   top_k, top-k, min_p, presence, repetition, vllm, serve, docker, context,
   max-model-len, thinking, reasoning, parser, tool, speculative, mtp, draft,
   dflash, eagle, kv-cache, tensor-parallel, offload, environment, best
   practice, usage, quickstart, deploy`. `CardCharsForContext(ctx int) int` is
   unchanged: `(ctx - 2048 - 1000) * 3`, floor 4,000, ceiling 48,000.
   `FetchCard` trims each card, joins them under a heading
   "## From the model card of <repo>", and cuts the joined text at the last
   paragraph break before `maxChars`, appending "[The rest of the model card
   was left out for length.]". The model's own card comes first, so it is the
   base model's that is cut when both do not fit.

5. `command.go`:

   ```go
   type Command struct {
       Raw     string   // as written, continuations joined
       Model   string   // the positional model argument
       Args    []string // everything after the model
       Env     []string // KEY=VALUE, in order of appearance
       Docker  bool     // taken from a docker run
       Partial bool     // the model argument is a placeholder ("...")
   }
   func ExtractCommands(raw string) []Command
   ```

   Code blocks are markdown fences **and** HTML `<pre>…</pre>` regions, read
   from `Card.Raw` with tags inside them removed and entities unescaped. Within
   a block: join lines ending in a backslash; drop lines beginning with `#`
   (they separate commands); each remaining logical line is a candidate.
   Tokenise with `process.SplitFlags`. Drop a trailing `&`, `|` or `>` and what
   follows.

6. A candidate line is a serve command in one of two forms.
   - **Direct.** It contains the tokens `vllm serve`, or
     `vllm.entrypoints.openai.api_server`. Leading `KEY=VALUE` tokens are its
     environment, together with earlier `export KEY=VALUE` lines in the same
     block. The token after `serve` is the model when it does not begin with
     `-`.
   - **Docker.** It begins `docker run`. Walk the tokens after `run`: a token
     beginning with `-` is a docker option; it takes the next token as its
     value unless it contains `=` or is one of the valueless options `--rm`,
     `-i`, `-t`, `-it`, `-d`, `--privileged`, `--init`, `--read-only`, `-P`.
     `-e` / `--env` values are collected as environment. The first token that
     is neither an option nor an option's value is the image. What follows the
     image is the serve command: if it starts with `vllm serve` strip those two
     tokens, then a first token not beginning with `-` is the model, and the
     rest are the arguments. It counts as a serve command only when the
     arguments contain at least two flags from `process.KnownServeFlags()` --
     which is what distinguishes a vLLM image from any other container.
   `Partial` is set when the model token is `...`, `…` or `<…>`-style
   placeholder text with no `/`. Identical commands (same `Args` and `Env`)
   are returned once, in order of first appearance.

7. `func (c Command) Settings() CommandSettings`:

   ```go
   type CommandSettings struct {
       Start process.VLLMStartConfig // known flags, from ParseArgs
       Rest  [][]string              // each unknown flag with its value
       Env   []string
   }
   ```

   `Rest` groups `ParseArgs`'s remainder so that each element is one flag and
   whatever value followed it. Deciding which of these become proposals and
   which become notes belongs to phase 08; this phase only produces the facts.

8. `func InlineFlags(raw string) [][]string` finds serve flags mentioned
   outside any command: inline code spans (markdown backticks and HTML
   `<code>` outside `<pre>`) whose text begins with `--` and whose first token
   is in `KnownServeFlags()`. Each is returned as a flag with its value. This
   is how "add `--speculative-config '{…}'`" in a sentence is seen at all.
   Unknown flags in prose are not collected: a code span such as `--help` or a
   flag of another server is far more often an aside than a recommendation,
   and only a flag this tool can name is worth a row.

9. Save excerpts under `internal/autoconfig/testdata/`, each cut down to the
   blocks that are read plus a few lines of the text around them, with the
   repository named in a first-line comment:
   `thinkingcap-27b-paro5.md` (one docker command, nine `-e` lines, the inline
   DFlash sentence, and the base card's fenced block of two `vllm serve`
   commands), `flash-next-mxfp4.md` (two docker commands and the partial
   `vllm serve ...` line) and `qwen3-14b-awq.md` (one direct command using
   `--enable-reasoning`, and one partial).

## Build gate

```
gofmt -l internal/
go vet ./...
go test ./...
```

## Test plan

- `ParseArgs(BuildArgs(c))` returns `c` with an empty `rest`, for a config
  with every field set except `ServedModelName`, which `ParseArgs` drops by
  design; a second case with it set checks that it is dropped and nothing
  else is lost. This keeps the two tables from drifting: a field added
  to `BuildArgs` later fails here until `ParseArgs` knows it.
- `ParseArgs` on `--max-model-len=4096 --enable-expert-offload
  --expert-offload-mem 46 --port 8000`: the context is set, `rest` is
  `["--enable-expert-offload", "--expert-offload-mem", "46"]`, and the port is
  gone.
- `ModelCard` returns the body, sends the token when one is set, and returns
  empty without error on 404. `BaseModel` handles the string form, the list
  form and a missing field.
- `FetchCard` reads two repositories when the base differs and one when it is
  the same or absent; `Hash` changes when either card changes.
- `TrimCard` on an HTML-only card keeps its `<pre>` block on separate lines
  with `<path>` unescaped; on a markdown card it drops "Citation" and keeps
  "Deployment".
- `thinkingcap-27b-paro5.md`: three commands. The first is `Docker`, its model
  is `/app/models`, its environment has nine entries including
  `GPU_MAX_HW_QUEUES=2`, `Settings().Start` has `TensorParallelSize 4`,
  `MaxModelLen 262144`, `KVCacheDtype "fp8"`, `ToolCallParser "qwen3_coder"`,
  `ReasoningParser "qwen3"` and the compilation config, `Rest` holds exactly
  the `--override-generation-config` pair, and `--port` and
  `--served-model-name` are nowhere. `InlineFlags` returns the
  `--speculative-config` pair.
- `flash-next-mxfp4.md`: two docker commands differing in
  `ROCR_VISIBLE_DEVICES`, and one `Partial` command whose environment is
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
- `qwen3-14b-awq.md`: one full command with `ReasoningParser "deepseek_r1"`
  and `--enable-reasoning` in `Rest`, and one `Partial`.
- A `docker run` of an unrelated image (`docker run -p 80:80 nginx`) and a
  block containing only `pip install vllm` yield nothing.

## Commit

```
feat(autoconfig): read a model card and the serve commands it contains
```

## Rollback

Revert the commit. The new package has no callers until phase 08, and the two
client methods and `ParseArgs` are additive.

package api

import (
	"fmt"
	"net/http"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
)

// handleBenchmarksAbout returns the disclosure modal contents: the
// internal prompt text and repetition prefix shown verbatim, the live
// preset table with computed durations, and an example uvx llama-benchy
// command per benchy preset. The user sees exactly what we run.
func (s *Server) handleBenchmarksAbout(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	fmt.Fprint(w, `<article>
  <header><strong>About benchmarks</strong></header>

  <h4>Benchmark sources</h4>
  <ul>
    <li><strong>internal-*</strong> presets run a Go HTTP loop that streams
    <code>/v1/chat/completions</code> against the loaded vLLM. TTFT is
    measured from the first SSE chunk; total time is wall clock. Token
    counts come from <code>stream_options.include_usage</code>.</li>
    <li><strong>benchy-*</strong> presets shell out to
    <a href="https://github.com/eugr/llama-benchy" target="_blank" rel="noopener">llama-benchy</a>
    via <code>uvx</code>. Engine-agnostic: the same preset can be run against
    llama.cpp, Ollama, or any OpenAI-compatible server for apples-to-apples
    comparison.</li>
  </ul>

  <h4>Internal preset prompt</h4>
  <p>Each request prepends a repetition prefix to defeat the prefix cache,
  then repeats the corpus text to fill the target token count
  (assumes ~4 chars/token).</p>
  <details>
    <summary>Prefix template</summary>
    <pre>`)
	fmt.Fprint(w, htmlEscape(benchmark.BenchPromptPrefixTemplate))
	fmt.Fprint(w, `</pre>
  </details>
  <details>
    <summary>Corpus text</summary>
    <pre style="white-space:pre-wrap;font-size:0.85em;">`)
	fmt.Fprint(w, htmlEscape(benchmark.BenchPromptText))
	fmt.Fprint(w, `</pre>
  </details>

  <h4>Presets</h4>
  <table>
    <thead><tr><th>Name</th><th>Source</th><th>Description</th></tr></thead>
    <tbody>`)
	for _, p := range benchmark.Presets() {
		fmt.Fprintf(w, `<tr><td><code>%s</code></td><td>%s</td><td><small>%s</small></td></tr>`,
			htmlEscape(p.Name), htmlEscape(p.EffectiveSource()), htmlEscape(p.Description))
	}
	fmt.Fprint(w, `</tbody>
  </table>

  <h4>Benchy commands</h4>
  <p>These are the exact <code>uvx</code> commands each benchy preset runs.
  The model identifier (<code>--model</code>) is the served-model-name vLLM
  reports via <code>/v1/models</code>; the tokenizer is the HuggingFace repo id.</p>`)

	for _, p := range benchmark.Presets() {
		if p.EffectiveSource() != benchmark.PresetSourceBenchy {
			continue
		}
		concurrency := p.Concurrency
		if len(concurrency) == 0 {
			concurrency = []int{1}
		}
		cmd := benchmark.FormatBenchyCommand(benchmark.BenchyConfig{
			BaseURL:         "http://localhost:8000/v1",
			APIKey:          "EMPTY",
			ServedModelName: "<served-model-name>",
			Tokenizer:       "<hf-repo-id>",
			PromptSizes:     p.PromptTokens,
			GenSizes:        []int{p.GenTokens},
			Runs:            p.Repetitions,
			Concurrency:     concurrency,
			SaveResultPath:  "<tempfile>",
		})
		fmt.Fprintf(w, `<details>
  <summary><code>%s</code></summary>
  <pre style="white-space:pre-wrap;font-size:0.85em;">%s</pre>
</details>`,
			htmlEscape(p.Name), htmlEscape(cmd))
	}

	fmt.Fprint(w, `</article>`)
}

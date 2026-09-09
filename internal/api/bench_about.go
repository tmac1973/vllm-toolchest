package api

import (
	"net/http"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
)

// presetRow is one line of the preset table.
type presetRow struct {
	Name        string
	Source      string
	Description string
}

// benchyCommandRow is the exact uvx command one benchy preset runs.
type benchyCommandRow struct {
	Name    string
	Command string
}

// handleBenchmarksAbout returns the disclosure modal contents: the internal
// prompt text and repetition prefix shown verbatim, the live preset table,
// and an example uvx llama-benchy command per benchy preset. The user sees
// exactly what we run.
func (s *Server) handleBenchmarksAbout(w http.ResponseWriter, r *http.Request) {
	presets := benchmark.Presets()

	rows := make([]presetRow, 0, len(presets))
	var commands []benchyCommandRow
	for _, p := range presets {
		rows = append(rows, presetRow{
			Name:        p.Name,
			Source:      p.EffectiveSource(),
			Description: p.Description,
		})

		if p.EffectiveSource() != benchmark.PresetSourceBenchy {
			continue
		}
		concurrency := p.Concurrency
		if len(concurrency) == 0 {
			concurrency = []int{1}
		}
		commands = append(commands, benchyCommandRow{
			Name: p.Name,
			Command: benchmark.FormatBenchyCommand(benchmark.BenchyConfig{
				BaseURL:         "http://localhost:8000/v1",
				APIKey:          "EMPTY",
				ServedModelName: "<served-model-name>",
				Tokenizer:       "<hf-repo-id>",
				PromptSizes:     p.PromptTokens,
				GenSizes:        []int{p.GenTokens},
				Runs:            p.Repetitions,
				Concurrency:     concurrency,
				SaveResultPath:  "<tempfile>",
			}),
		})
	}

	respondHTML(w)
	s.renderPartial(w, "bench_about", struct {
		PrefixTemplate string
		PromptText     string
		Presets        []presetRow
		BenchyCommands []benchyCommandRow
	}{
		PrefixTemplate: benchmark.BenchPromptPrefixTemplate,
		PromptText:     benchmark.BenchPromptText,
		Presets:        rows,
		BenchyCommands: commands,
	})
}

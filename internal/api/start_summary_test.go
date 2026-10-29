package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/advice"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// The start that prompted the summary: Qwen3.8-Flash-Next at TP=4, a YaRN
// context of 512K, sixteen streams, and a default output limit of 64K.
var qwenStart = advice.Measurements{
	KVCacheGB:         6.78,
	WeightsPerRankGB:  18.36,
	LoadSeconds:       180.136055,
	ConsumedGB:        21.81,
	PeakActivationGB:  1.1,
	GraphPoolGB:       0.33,
	MaxConcurrency:    1.71,
	ConcurrencyTokens: 524288,
	KVCacheTokens:     898205,
	PLEOffloadGB:      38.74,
	EngineVersion:     "0.29.0.dev0+g2bdbbc8080",
	DeviceGB:          31.86,
	BudgetGB:          30.9,
	BlockTokens:       832,
	Hybrid:            true,
	ExpertsResident:   24576,
	InitSeconds:       74.75,
	CompileSeconds:    9.5,
}

var qwenArgs = []string{
	"--max-model-len", "524288", "--tensor-parallel-size", "4",
	"--gpu-memory-utilization", "0.97", "--max-num-seqs", "16",
	"--enable-prefix-caching", "--kv-cache-dtype", "fp8",
	"--max-num-batched-tokens", "8192",
	`--speculative-config={"method": "mtp", "num_speculative_tokens": 3}`,
	"--override-generation-config", `{"max_tokens": 65536, "temperature": 0.8}`,
}

func summaryRowFor(t *testing.T, s *startSummary, label string) summaryRow {
	t.Helper()
	for _, sec := range s.Sections {
		for _, r := range sec.Rows {
			if strings.HasPrefix(r.Label, label) {
				return r
			}
		}
	}
	t.Fatalf("no %q row in the summary", label)
	return summaryRow{}
}

func TestStartSummaryAnswersThePerStreamQuestion(t *testing.T) {
	s := newStartSummary(qwenStart, qwenArgs)
	if s == nil {
		t.Fatal("a fully reported start produced no summary")
	}

	for label, want := range map[string]string{
		"Shared pool":                  "877K tokens",
		"Longest request":              "512K tokens",
		"Full-length requests at once": "1.71",
		"Each of 16 streams":           "55K tokens",
		"Default output limit":         "64K tokens",
		"Prefix cache reuse":           "in 832-token blocks",
		"Card":                         "31.9 GiB",
		"KV cache":                     "6.8 GiB",
		"Captured graphs":              "0.33 GiB",
		"In host RAM":                  "38.7 GiB",
		"Loading weights":              "3m 00s",
		"Engine setup":                 "1m 15s",
		"Experts":                      "all 24,576 on the GPUs",
		"KV cache format":              "fp8",
		"Speculative decoding":         "mtp, 3 draft tokens",
	} {
		if got := summaryRowFor(t, s, label).Value; got != want {
			t.Errorf("%s = %q, want %q", label, got, want)
		}
	}

	if r := summaryRowFor(t, s, "Default output limit"); !r.Warn {
		t.Error("an output limit longer than a stream's share is not flagged")
	}
	if r := summaryRowFor(t, s, "Each of 16 streams"); !r.Ours {
		t.Error("the per-stream share is our arithmetic and should say so")
	}
	if !strings.Contains(summaryRowFor(t, s, "Shared pool").Note, "recurrent state") {
		t.Error("a hybrid model's pool note does not mention the per-request state")
	}
	if want := "877K tokens of context, 55K each across 16 streams"; s.Headline != want {
		t.Errorf("headline = %q, want %q", s.Headline, want)
	}
}

// An output limit that fits inside a stream's share is not worth a row.
func TestStartSummaryOmitsAnOutputLimitThatFits(t *testing.T) {
	args := append([]string{}, qwenArgs...)
	args[len(args)-1] = `{"max_tokens": 32768}`
	s := newStartSummary(qwenStart, args)
	for _, sec := range s.Sections {
		for _, r := range sec.Rows {
			if r.Label == "Default output limit" {
				t.Errorf("a 32K limit under a 55K share was flagged: %+v", r)
			}
		}
	}
}

// Before the engine reports anything there is nothing to summarise, and the
// panel must say "loading" rather than show empty headings.
func TestStartSummaryIsNilBeforeAnyMeasurement(t *testing.T) {
	if s := newStartSummary(advice.Measurements{}, qwenArgs); s != nil {
		t.Errorf("an empty measurement produced a summary: %+v", s)
	}
}

func TestTokensK(t *testing.T) {
	for n, want := range map[int]string{
		832:     "832",
		8192:    "8K",
		5000:    "4.9K",
		56137:   "55K",
		524288:  "512K",
		898205:  "877K",
		1048576: "1,024K",
	} {
		if got := tokensK(n); got != want {
			t.Errorf("tokensK(%d) = %q, want %q", n, got, want)
		}
	}
}

// The template has to render the section on a quiet start too: that is the
// ordinary case, and the one where the figures are all there is to read.
func TestTheAdvicePanelRendersTheStartSummary(t *testing.T) {
	const id = "test/summary"
	output := strings.Join([]string{
		"INFO [gpu_worker.py:626] Available KV cache memory: 6.78 GiB",
		"INFO [kv_cache_utils.py:1718] GPU KV cache size: 898,205 tokens, Maximum concurrency for 524,288 tokens per request: 1.71x",
	}, "\n")
	s := testServerWithEngine(t, id, output, models.VLLMConfig{TensorParallelSize: 4, MaxModelLen: 524288, MaxNumSeqs: 16})
	s.initTemplates()
	waitForObserved(t, s, func() bool { return s.process.Measured().KVCacheTokens > 0 })

	req := httptest.NewRequest("GET", "/api/service/advice", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleServiceAdvice(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{"Shared pool", "877K tokens", "start-summary", "Nothing to report"} {
		if !strings.Contains(body, want) {
			t.Errorf("rendered panel is missing %q:\n%s", want, body)
		}
	}
}

// The pinning suggestion is a byte count; the panel shows it in GiB while the
// button still writes the exact bytes.
func TestKVCacheMemorySuggestionReadsInGiB(t *testing.T) {
	if got := suggestedLabel("kv_cache_memory", "6926419682"); got != "6.5 GiB" {
		t.Errorf("label = %q, want 6.5 GiB", got)
	}
	if got := suggestedLabel("max_model_len", "214240"); got != "214240" {
		t.Errorf("a non-byte suggestion was rewritten: %q", got)
	}
}

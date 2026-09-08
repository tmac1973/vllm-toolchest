package api

import (
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
)

// The HTML fragments in this package are being moved out of fmt.Fprintf and
// into html/template partials. That refactor is meant to be invisible, and
// "invisible" is only checkable against a recording — reading two renderings
// of a 700-line config panel side by side is not something to trust to the eye.
//
// So: render every fragment handler that can be driven from local state,
// against fixtures chosen to exercise the branches, and diff the bytes. Run
// with -update to re-record after a change that is meant to be visible.
var updateGolden = flag.Bool("update", false, "rewrite the golden fragment files")

// goldenFixtureModels are the registry entries every fragment renders against.
// They are picked for branch coverage, not realism:
//
//   - the FP8 model is the ordinary case: quantized, tool-capable, TP=2.
//   - the AWQ model exercises the Marlin branch in compatibleQuantOptions.
//   - the third is deliberately hostile. Its name and free-text config fields
//     carry the characters that break unescaped HTML, including the JSON
//     speculative config whose double quotes once closed the value="..."
//     attribute it was written into and truncated the field to `{`.
func goldenFixtureModels() []*models.Model {
	return []*models.Model{
		{
			ID:             "unsloth/Qwen3.8-27B-FP8",
			DisplayName:    "Qwen3.8-27B-FP8",
			LocalPath:      "/models/unsloth/Qwen3.8-27B-FP8",
			Enabled:        true,
			DownloadDate:   time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC),
			TotalSizeBytes: 30_900_000_000,
			HFConfig: models.HFConfig{
				Architectures:         []string{"Qwen3MoeForCausalLM"},
				ModelType:             "qwen3_moe",
				NumHiddenLayers:       64,
				AttentionLayers:       16,
				HiddenSize:            5120,
				IntermediateSize:      25600,
				NumAttentionHeads:     40,
				NumKeyValueHeads:      8,
				HeadDim:               128,
				MaxPositionEmbeddings: 262144,
				VocabSize:             151936,
				TorchDtype:            "bfloat16",
			},
			Quantization: models.QuantMeta{Method: "fp8", Bits: 8, Sym: true, BytesPerParam: 1.0},
			ToolUse: models.ToolUseMeta{
				HasToolSupport:  true,
				ToolCallParser:  "hermes",
				DetectionMethod: "chat_template",
			},
			VLLMConfig: models.VLLMConfig{
				Dtype:                "auto",
				MaxModelLen:          32768,
				TensorParallelSize:   2,
				GPUMemoryUtilization: 0.90,
				MaxNumSeqs:           16,
				EnableChunkedPrefill: true,
				KVCacheDtype:         "fp8",
				LoadFormat:           "auto",
				EnableAutoToolChoice: true,
				ToolCallParser:       "hermes",
			},
		},
		{
			ID:             "TheBloke/Mixtral-8x7B-AWQ",
			DisplayName:    "Mixtral-8x7B-AWQ",
			LocalPath:      "/models/TheBloke/Mixtral-8x7B-AWQ",
			Enabled:        true,
			DownloadDate:   time.Date(2026, 2, 14, 9, 30, 0, 0, time.UTC),
			TotalSizeBytes: 24_700_000_000,
			HFConfig: models.HFConfig{
				Architectures:         []string{"MixtralForCausalLM"},
				ModelType:             "mixtral",
				NumHiddenLayers:       32,
				HiddenSize:            4096,
				IntermediateSize:      14336,
				NumAttentionHeads:     32,
				NumKeyValueHeads:      8,
				HeadDim:               128,
				MaxPositionEmbeddings: 32768,
				VocabSize:             32000,
				TorchDtype:            "float16",
			},
			Quantization: models.QuantMeta{Method: "awq", Bits: 4, Sym: true, GroupSize: 128, BytesPerParam: 0.5},
			ToolUse: models.ToolUseMeta{
				HasToolSupport:  true,
				ToolCallParser:  "mistral",
				DetectionMethod: "architecture",
			},
			VLLMConfig: models.VLLMConfig{
				Dtype:                "float16",
				MaxModelLen:          8192,
				TensorParallelSize:   1,
				GPUMemoryUtilization: 0.85,
				MaxNumSeqs:           32,
				KVCacheDtype:         "auto",
				LoadFormat:           "auto",
				AttentionBackend:     "TRITON_ATTN",
			},
		},
		{
			ID:             `evil/model "x><script>`,
			DisplayName:    `Model & "Friends" <b>`,
			LocalPath:      "/models/evil/model",
			Enabled:        false,
			DownloadDate:   time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC),
			TotalSizeBytes: 15_000_000_000,
			Orphaned:       true,
			HFConfig: models.HFConfig{
				Architectures:         []string{"LlamaForCausalLM"},
				ModelType:             "llama",
				NumHiddenLayers:       32,
				HiddenSize:            4096,
				IntermediateSize:      11008,
				NumAttentionHeads:     32,
				NumKeyValueHeads:      32,
				HeadDim:               128,
				MaxPositionEmbeddings: 20000,
				VocabSize:             32000,
				TorchDtype:            "bfloat16",
			},
			Quantization: models.QuantMeta{Method: "none", BytesPerParam: 2.0},
			ToolUse:      models.ToolUseMeta{HasToolSupport: false},
			VLLMConfig: models.VLLMConfig{
				Dtype:                "bfloat16",
				MaxModelLen:          12345, // deliberately not one of the presets
				TensorParallelSize:   1,
				GPUMemoryUtilization: 0.92,
				MaxNumSeqs:           4,
				KVCacheDtype:         "auto",
				LoadFormat:           "auto",
				SpeculativeConfig:    `{"method":"mtp","num_speculative_tokens":8}`,
				CompilationConfig:    `{"cudagraph_capture_sizes":[1,2,4,8]}`,
				ReasoningParser:      "qwen3",
				ChatTemplate:         "/tmp/tpl & more.jinja",
				ExtraFlags:           `--swap-space 4 --served-model-name "my model"`,
				TrustRemoteCode:      true,
			},
		},
	}
}

// newGoldenServer builds a Server whose every dependency is local and
// deterministic: no network, no vLLM process, no GPU probe.
func newGoldenServer(t *testing.T, variant string) *Server {
	t.Helper()
	dir := t.TempDir()

	env := vllmenv.Env{Variant: vllmenv.VariantGeneric, VenvRoot: "/opt/vllm-venv"}
	if variant == vllmenv.VariantRadiance {
		env = vllmenv.Env{
			Variant:         vllmenv.VariantRadiance,
			RadianceVersion: "0.9.3",
			VenvRoot:        "/opt/vllm",
		}
	}

	s := &Server{
		cfg: &config.Config{
			DataDir:        dir,
			ExternalURL:    "http://compute:3000",
			VLLMHost:       "127.0.0.1",
			VLLMPort:       8000,
			GPUMemoryUtil:  0.90,
			MaxNumSeqs:     16,
			DefaultDtype:   "auto",
			ToolUseEnabled: true,
		},
		registry: models.NewRegistry(dir),
		monitor:  monitor.New(0),
		process:  process.NewManager("127.0.0.1", 8000),
		vllmEnv:  env,
	}
	s.bench = benchmark.NewStore(dir)
	s.benchSvc = benchmark.NewService(s.bench)
	s.initTemplates()

	for _, m := range goldenFixtureModels() {
		m.VRAMEstimate = models.EstimateVRAM(m)
		if err := s.registry.Register(m); err != nil {
			t.Fatalf("register %s: %v", m.ID, err)
		}
	}
	return s
}

// assertGolden compares body against testdata/golden/<name>.html, or rewrites
// it under -update.
func assertGolden(t *testing.T, name, body string) {
	t.Helper()
	path := filepath.Join("testdata", "golden", name+".html")

	if *updateGolden {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}

	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("no recording for %q — run: go test ./internal/api -run TestGoldenFragments -update (%v)", name, err)
	}
	if string(want) == body {
		return
	}

	// Most of these fragments are emitted today as one very long line, because
	// that is what a chain of Fprintf calls produces. A readable template does
	// not, so the common failure during the conversion is a difference in
	// whitespace and nothing else. Say which kind it is: whitespace-only is a
	// judgement call to review and re-record, anything else is a bug.
	kind := "output changed"
	if strings.TrimSpace(collapseSpace(string(want))) == strings.TrimSpace(collapseSpace(body)) {
		kind = "output changed in whitespace only (review, then re-record with -update)"
	}

	// Point at the first divergence. These fragments run to tens of kilobytes
	// and a full dump helps nobody.
	i := 0
	for i < len(want) && i < len(body) && want[i] == body[i] {
		i++
	}
	t.Errorf("%s: %s at byte %d of %d\n  want: %.160q\n  got:  %.160q",
		name, kind, i, len(want), sliceFrom(string(want), i), sliceFrom(body, i))
}

func sliceFrom(s string, i int) string {
	if i >= len(s) {
		return "<end of output>"
	}
	return s[i:]
}

// collapseSpace reduces every run of whitespace to a single space, leaving
// <pre> blocks alone — inside those, whitespace is the content.
//
// It deliberately collapses rather than strips: turning ">\n  <" into "><"
// would also hide a real change, since the space between a label's text and
// its control is rendered.
func collapseSpace(s string) string {
	var out strings.Builder
	for len(s) > 0 {
		open := strings.Index(s, "<pre")
		if open < 0 {
			out.WriteString(collapseRuns(s))
			break
		}
		close := strings.Index(s[open:], "</pre>")
		if close < 0 {
			out.WriteString(collapseRuns(s))
			break
		}
		close += open + len("</pre>")
		out.WriteString(collapseRuns(s[:open]))
		out.WriteString(s[open:close])
		s = s[close:]
	}
	return out.String()
}

func collapseRuns(s string) string {
	var out strings.Builder
	inSpace := false
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if !inSpace {
				out.WriteRune(' ')
				inSpace = true
			}
			continue
		}
		inSpace = false
		out.WriteRune(r)
	}
	return out.String()
}

type goldenCase struct {
	name    string
	variant string
	method  string
	target  string
	handler func(*Server, http.ResponseWriter, *http.Request)
}

// TestGoldenFragments renders every HTML fragment that can be produced from
// local state. Handlers that need the network (HuggingFace search) or a live
// vLLM process are not covered here.
func TestGoldenFragments(t *testing.T) {
	cases := []goldenCase{
		{name: "models_list", target: "/api/models", handler: (*Server).handleListModels},
		{name: "models_scan", method: "POST", target: "/api/models/scan", handler: (*Server).handleScanModels},
		{name: "dashboard", target: "/api/dashboard", handler: (*Server).handleDashboard},
		{name: "service_status", target: "/api/service/status", handler: (*Server).handleServiceStatus},
		{name: "monitor_bar", target: "/api/monitor", handler: (*Server).handleMonitorStatus},
		{name: "bench_about", target: "/api/benchmarks/about", handler: (*Server).handleBenchmarksAbout},
		{name: "bench_form", target: "/api/benchmarks/form", handler: (*Server).handleBenchmarkForm},
		{name: "bench_list", target: "/api/benchmarks/", handler: (*Server).handleListBenchmarks},
		{name: "job_form", target: "/api/benchmark-jobs/form", handler: (*Server).handleJobForm},
		{name: "job_list", target: "/api/benchmark-jobs/", handler: (*Server).handleListJobs},
		{name: "probe_form", target: "/api/benchmarks/probe-context/form", handler: (*Server).handleProbeForm},
	}

	// The config panel is the largest fragment and the one with the most
	// per-model branching, so record it for every fixture. Named by position:
	// one fixture's display name is deliberately full of characters that have
	// no business in a filename.
	for i, m := range goldenFixtureModels() {
		cases = append(cases, goldenCase{
			name:    fmt.Sprintf("config_panel_%d", i),
			target:  "/api/models/config-panel?id=" + url.QueryEscape(m.ID),
			handler: (*Server).handleModelConfigPanel,
		})
	}
	// It also branches on the image variant: radiance adds attention backends
	// and the all-reduce token-ceiling advice.
	cases = append(cases, goldenCase{
		name:    "config_panel_radiance",
		variant: vllmenv.VariantRadiance,
		target:  "/api/models/config-panel?id=" + url.QueryEscape("unsloth/Qwen3.8-27B-FP8"),
		handler: (*Server).handleModelConfigPanel,
	})

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newGoldenServer(t, tc.variant)
			method := tc.method
			if method == "" {
				method = "GET"
			}
			r := httptest.NewRequest(method, tc.target, nil)
			r.Header.Set("HX-Request", "true")
			w := httptest.NewRecorder()

			tc.handler(s, w, r)

			if w.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
			}
			assertGolden(t, tc.name, w.Body.String())
		})
	}
}

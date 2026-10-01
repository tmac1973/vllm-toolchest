package process

import (
	"reflect"
	"strings"
	"testing"
)

// Every field BuildArgs writes comes back. A field added to BuildArgs and not
// to ParseArgs fails here, which is what keeps the two from drifting.
func TestParseArgsInvertsBuildArgs(t *testing.T) {
	c := VLLMStartConfig{
		Dtype: "bfloat16", MaxModelLen: 131072, TensorParallelSize: 4,
		GPUMemoryUtilization: 0.92, EnforceEager: true, TrustRemoteCode: true,
		MaxNumSeqs: 8, Quantization: "awq", LoadFormat: "safetensors",
		EnablePrefixCaching: true, KVCacheDtype: "fp8", EnableChunkedPrefill: true,
		MaxNumBatchedTokens: 8192, EnableAutoToolChoice: true, ToolCallParser: "qwen3_coder",
		ReasoningParser: "qwen3", AttentionBackend: "TRITON_ATTN", MambaCacheMode: "align",
		SpeculativeConfig: `{"method": "mtp", "num_speculative_tokens": 3}`,
		CompilationConfig: `{"cudagraph_capture_sizes": [4]}`,
		KVCacheMemory:     6931621295, DisableAsyncScheduling: true, LanguageModelOnly: true,
		Tokenizer: "org/tok", ChatTemplate: "/t.jinja",
	}
	got, rest := ParseArgs(BuildArgs(c))
	if len(rest) != 0 {
		t.Errorf("rest = %q, want nothing", rest)
	}
	if !reflect.DeepEqual(got, c) {
		t.Errorf("round trip lost fields:\n got %+v\nwant %+v", got, c)
	}

	// The served name is this tool's to set, so it is dropped by design, and
	// nothing else is lost with it.
	c.ServedModelName = "org/model"
	got, rest = ParseArgs(BuildArgs(c))
	c.ServedModelName = ""
	if len(rest) != 0 || !reflect.DeepEqual(got, c) {
		t.Errorf("with a served name: rest=%q\n got %+v", rest, got)
	}
}

func TestParseArgsKeepsWhatItDoesNotKnow(t *testing.T) {
	got, rest := ParseArgs([]string{
		"--max-model-len=4096", "--enable-expert-offload", "--expert-offload-mem", "46",
		"--port", "8000", "--enable-reasoning", "--reasoning-parser", "deepseek_r1",
		"-tp", "2", "--max-num-seqs", "lots",
	})
	if got.MaxModelLen != 4096 || got.ReasoningParser != "deepseek_r1" || got.TensorParallelSize != 2 {
		t.Errorf("known flags not read: %+v", got)
	}
	want := []string{"--enable-expert-offload", "--expert-offload-mem", "46", "--enable-reasoning", "--max-num-seqs", "lots"}
	if !reflect.DeepEqual(rest, want) {
		t.Errorf("rest = %q\nwant  %q", rest, want)
	}
}

// vLLM takes a flag written with underscores as the dashed one, and cards
// write both.
func TestParseArgsTakesUnderscoredFlags(t *testing.T) {
	cfg, rest := ParseArgs([]string{"--load_format", "mistral", "--tokenizer_mode", "mistral", "--max_model_len=8192", "-tp", "2"})
	if cfg.LoadFormat != "mistral" || cfg.MaxModelLen != 8192 {
		t.Errorf("cfg = %+v", cfg)
	}
	if !strings.Contains(strings.Join(rest, " "), "--tokenizer-mode mistral") {
		t.Errorf("rest = %v", rest)
	}
}

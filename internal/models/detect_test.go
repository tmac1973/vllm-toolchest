package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// modelDirWith writes files (name -> contents) into a fresh model directory.
func modelDirWith(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// tokenizerConfig wraps a chat template the way tokenizer_config.json holds it.
func tokenizerConfig(t *testing.T, template any) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{"chat_template": template, "model_max_length": 131072})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// Excerpts of real chat templates: the parts that carry the tool-call markers.
const (
	// NousResearch/Hermes-3-Llama-3.1-8B: JSON inside <tool_call> tags.
	hermesTemplate = `{%- if tools %}<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags.
{%- endif %}`

	// mistralai/Mistral-7B-Instruct-v0.3.
	mistralTemplate = `{%- if tools is not none and (message == user_messages[-1]) %}
{{- "[AVAILABLE_TOOLS] [" }}{%- for tool in tools %}{{- tool|tojson }}{%- endfor %}{{- "][/AVAILABLE_TOOLS]" }}
{%- endif %}
{%- elif message.tool_calls is defined and message.tool_calls is not none %}{{- "[TOOL_CALLS] [" }}`

	// meta-llama/Llama-3.1-8B-Instruct's assistant tool-call turn, verbatim
	// from its tokenizer_config.json (via the ungated unsloth mirror). Custom
	// tools are called in JSON; <|python_tag|> only prefixes the built-in
	// ipython tools.
	llama31Template = `{%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}`
)

// The architecture is checked first because newer families share template
// markers with older ones but emit a different call format. Qwen3.5's
// template has <tool_call> like Hermes, yet its calls are XML.
func TestDetectToolUsePrefersTheArchitecture(t *testing.T) {
	dir := modelDirWith(t, map[string]string{"tokenizer_config.json": tokenizerConfig(t, hermesTemplate)})
	got := DetectToolUse(dir, "Qwen/Qwen3.5-27B", HFConfig{Architectures: []string{"Qwen3_5ForConditionalGeneration"}})
	want := ToolUseMeta{HasToolSupport: true, ToolCallParser: "qwen3_xml", DetectionMethod: "architecture"}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// With an architecture that names no parser, the chat template decides.
func TestDetectToolUseReadsTheChatTemplate(t *testing.T) {
	cases := []struct {
		name     string
		modelID  string
		template any
		want     string
	}{
		{"hermes", "NousResearch/Hermes-3-Llama-3.1-8B", hermesTemplate, "hermes"},
		{"mistral", "mistralai/Mistral-7B-Instruct-v0.3", mistralTemplate, "mistral"},
		{
			// Hermes-2-Pro ships named templates as a list; the tool_use one
			// carries the markers and must not be missed for not being
			// first.
			"named template list", "NousResearch/Hermes-2-Pro-Llama-3-8B",
			[]map[string]string{
				{"name": "default", "template": "{{bos_token}}{% for message in messages %}<|im_start|>{{ message['role'] }}{% endfor %}"},
				{"name": "tool_use", "template": hermesTemplate},
			},
			"hermes",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := modelDirWith(t, map[string]string{"tokenizer_config.json": tokenizerConfig(t, c.template)})
			got := DetectToolUse(dir, c.modelID, HFConfig{Architectures: []string{"LlamaForCausalLM"}})
			want := ToolUseMeta{HasToolSupport: true, ToolCallParser: c.want, DetectionMethod: "chat_template_regex"}
			if got != want {
				t.Errorf("got %+v, want %+v", got, want)
			}
		})
	}
}

// Llama 3.1 calls custom tools in JSON, which vLLM parses with llama3_json.
// Its template also has <|python_tag|> for the built-in ipython tools, which
// once selected "pythonic": every Llama 3.x model was set up with a parser
// that could not read its tool calls.
func TestDetectToolUseGivesLlama31ItsJSONParser(t *testing.T) {
	dir := modelDirWith(t, map[string]string{"tokenizer_config.json": tokenizerConfig(t, llama31Template)})
	got := DetectToolUse(dir, "meta-llama/Llama-3.1-8B-Instruct", HFConfig{Architectures: []string{"LlamaForCausalLM"}})
	if got.ToolCallParser != "llama3_json" {
		t.Errorf("got %+v, want llama3_json", got)
	}
}

// Without a tokenizer config, or with one that does not parse, the model's
// name is the last resort.
func TestDetectToolUseFallsBackToTheModelName(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
	}{
		{"no tokenizer config", nil},
		{"unparseable tokenizer config", map[string]string{"tokenizer_config.json": "{not json"}},
		{"template with no tool markers", map[string]string{"tokenizer_config.json": tokenizerConfig(t, "{{ messages[0]['content'] }}")}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := DetectToolUse(modelDirWith(t, c.files), "meta-llama/Llama-3.3-70B-Instruct", HFConfig{Architectures: []string{"LlamaForCausalLM"}})
			want := ToolUseMeta{HasToolSupport: true, ToolCallParser: "llama3_json", DetectionMethod: "known_model_family"}
			if got != want {
				t.Errorf("got %+v, want %+v", got, want)
			}
		})
	}
}

// A model nothing identifies gets no parser. Guessing one would enable
// --enable-auto-tool-choice with a parser that mangles its output.
func TestDetectToolUseOfAnUnknownModelIsNone(t *testing.T) {
	got := DetectToolUse(t.TempDir(), "acme/base-7b", HFConfig{Architectures: []string{"GPTNeoXForCausalLM"}})
	if got != (ToolUseMeta{}) {
		t.Errorf("got %+v, want no tool support", got)
	}
}

func TestDetectVisionRecognisesVisionModels(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		archs []string
	}{
		// Qwen2.5-VL ships a preprocessor_config.json for its image
		// processor; that alone is enough.
		{"preprocessor config", map[string]string{"preprocessor_config.json": `{"image_processor_type":"Qwen2VLImageProcessor"}`}, nil},
		// llava-hf/llava-1.5-7b-hf's processor_config.json.
		{"processor config", map[string]string{"processor_config.json": `{"image_token":"<image>","num_additional_image_tokens":1,"patch_size":14,"processor_class":"LlavaProcessor","vision_feature_select_strategy":"default"}`}, nil},
		// Gemma 3's config carries its vision tower; that is enough even
		// when no processor file was downloaded.
		{"vision_config in config.json", map[string]string{"config.json": `{"architectures":["Gemma3ForConditionalGeneration"],"vision_config":{"hidden_size":1152}}`}, nil},
		{"VL architecture", nil, []string{"Qwen2_5_VLForConditionalGeneration"}},
		{"LLaVA architecture", nil, []string{"LlavaForConditionalGeneration"}},
		{"Pixtral architecture", nil, []string{"PixtralForConditionalGeneration"}},
		{"InternVL architecture", nil, []string{"InternVLChatModel"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DetectVision(modelDirWith(t, c.files), HFConfig{Architectures: c.archs}); !got.IsVisionModel {
				t.Error("not detected as a vision model")
			}
		})
	}
}

// A text-only model must not be marked as vision: that offers image inputs
// the server will reject.
func TestDetectVisionLeavesTextModelsAlone(t *testing.T) {
	dir := modelDirWith(t, map[string]string{
		"config.json": `{"architectures":["Qwen3ForCausalLM"],"hidden_size":4096,"text_config":{}}`,
	})
	if got := DetectVision(dir, HFConfig{Architectures: []string{"Qwen3ForCausalLM"}}); got.IsVisionModel {
		t.Error("Qwen3-8B detected as a vision model")
	}
	// An unreadable config is not evidence of anything.
	bad := modelDirWith(t, map[string]string{"config.json": "{"})
	if got := DetectVision(bad, HFConfig{Architectures: []string{"LlamaForCausalLM"}}); got.IsVisionModel {
		t.Error("an unparseable config made a vision model")
	}
}

func ptr[T any](v T) *T { return &v }

func TestParseGenDefaultsReadsGenerationConfig(t *testing.T) {
	cases := []struct {
		name string
		body string
		want GenDefaults
	}{
		{
			// Qwen/Qwen3-8B generation_config.json, verbatim.
			"Qwen3-8B",
			`{"bos_token_id":151643,"do_sample":true,"eos_token_id":[151645,151643],"pad_token_id":151643,
			  "temperature":0.6,"top_k":20,"top_p":0.95,"transformers_version":"4.51.0"}`,
			GenDefaults{Temperature: ptr(0.6), TopP: ptr(0.95), TopK: ptr(20)},
		},
		{
			// meta-llama/Llama-3.1-8B-Instruct: no top_k, which must stay
			// unset rather than become 0 (top_k 0 means something to vLLM).
			"Llama-3.1-8B-Instruct",
			`{"bos_token_id":128000,"do_sample":true,"eos_token_id":[128001,128008,128009],
			  "temperature":0.6,"top_p":0.9,"transformers_version":"4.42.3"}`,
			GenDefaults{Temperature: ptr(0.6), TopP: ptr(0.9)},
		},
		{
			// Qwen/Qwen2.5-7B-Instruct carries a repetition penalty.
			"Qwen2.5-7B-Instruct",
			`{"bos_token_id":151643,"pad_token_id":151643,"do_sample":true,"eos_token_id":[151645,151643],
			  "repetition_penalty":1.05,"temperature":0.7,"top_p":0.8,"top_k":20}`,
			GenDefaults{Temperature: ptr(0.7), TopP: ptr(0.8), TopK: ptr(20), RepetitionPenalty: ptr(1.05)},
		},
		{
			// An explicit greedy default is a value, not an absence.
			"explicit zero temperature and max_new_tokens",
			`{"temperature":0.0,"max_new_tokens":2048}`,
			GenDefaults{Temperature: ptr(0.0), MaxNewTokens: ptr(2048)},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ParseGenDefaults(modelDirWith(t, map[string]string{"generation_config.json": c.body}))
			gj, _ := json.Marshal(got)
			wj, _ := json.Marshal(c.want)
			if string(gj) != string(wj) {
				t.Errorf("got %s, want %s", gj, wj)
			}
		})
	}
}

// No generation_config.json, or a broken one, leaves every default unset so
// vLLM's own defaults apply.
func TestParseGenDefaultsWithoutAUsableFileIsEmpty(t *testing.T) {
	for name, dir := range map[string]string{
		"missing": t.TempDir(),
		"broken":  modelDirWith(t, map[string]string{"generation_config.json": `{"temperature":`}),
	} {
		if got := ParseGenDefaults(dir); got != (GenDefaults{}) {
			t.Errorf("%s: got %+v, want all unset", name, got)
		}
	}
}

// One field of an unexpected type costs that field only. top_k written as
// 20.0 is still the count 20; a temperature written as a string is dropped
// while the rest of the file is kept.
func TestParseGenDefaultsKeepsWhatItCanRead(t *testing.T) {
	dir := modelDirWith(t, map[string]string{"generation_config.json": `{
		"temperature": "0.7", "top_p": 0.95, "top_k": 20.0, "max_new_tokens": 1.5
	}`})
	g := ParseGenDefaults(dir)
	if g.Temperature != nil {
		t.Errorf("temperature = %v, want unset for a string", *g.Temperature)
	}
	if g.TopP == nil || *g.TopP != 0.95 {
		t.Errorf("top_p = %v, want 0.95", g.TopP)
	}
	if g.TopK == nil || *g.TopK != 20 {
		t.Errorf("top_k = %v, want 20", g.TopK)
	}
	if g.MaxNewTokens != nil {
		t.Errorf("max_new_tokens = %v, want unset for a fraction", *g.MaxNewTokens)
	}
}

// Audio models ship a preprocessor_config.json too, for their feature
// extractor. That alone must not make Whisper a vision model.
func TestDetectVisionIgnoresAnAudioFeatureExtractor(t *testing.T) {
	// openai/whisper-large-v3's preprocessor_config.json, abridged.
	dir := modelDirWith(t, map[string]string{"preprocessor_config.json": `{"chunk_length":30,"feature_extractor_type":"WhisperFeatureExtractor","feature_size":128,"hop_length":160,"n_fft":400,"n_samples":480000,"nb_max_frames":3000,"padding_side":"right","padding_value":0.0,"processor_class":"WhisperProcessor","return_attention_mask":false,"sampling_rate":16000}`})
	if DetectVision(dir, HFConfig{Architectures: []string{"WhisperForConditionalGeneration"}}).IsVisionModel {
		t.Error("an audio model was detected as a vision model")
	}
}

// Qwen's <tool_call> wraps JSON in Qwen3 -- thinking variants too -- and XML
// in Qwen3 Coder and Qwen3.5. Each excerpt is the tool-call instruction from
// the repository's own chat template, verbatim, with the <think> it opens
// where the template has one. <think> once chose the XML parser, which sent
// Qwen3's JSON calls to a parser that could not read them.
func TestDetectToolUseTellsQwensTwoToolCallFormatsApart(t *testing.T) {
	cases := []struct {
		repo, template, want string
	}{
		{"Qwen/Qwen3-8B", `<tool_call></tool_call>
{{- '<think>\n' }}`, "hermes"},
		{"Qwen/Qwen3-30B-A3B-Thinking-2507", `<tool_call></tool_call>
{{- '<think>\n' }}`, "hermes"},
		{"Qwen/Qwen3-Coder-30B-A3B-Instruct", `<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>`, "qwen3_coder"},
		{"Qwen/Qwen3.5-27B", `<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>
{{- '<think>\n' }}`, "qwen3_xml"},
	}
	for _, c := range cases {
		t.Run(c.repo, func(t *testing.T) {
			dir := modelDirWith(t, map[string]string{"tokenizer_config.json": tokenizerConfig(t, c.template)})
			// An architecture that names no parser, so the template decides.
			got := DetectToolUse(dir, "local/model", HFConfig{Architectures: []string{"Qwen3ForCausalLM"}})
			if got.ToolCallParser != c.want {
				t.Errorf("got %+v, want %s", got, c.want)
			}
		})
	}
}

// Newer repositories keep the template in chat_template.jinja and leave it
// out of tokenizer_config.json. Detection must read it there rather than fall
// back to guessing from the name.
func TestDetectToolUseReadsAChatTemplateFile(t *testing.T) {
	dir := modelDirWith(t, map[string]string{
		"tokenizer_config.json": `{"model_max_length": 131072}`,
		"chat_template.jinja":   mistralTemplate,
	})
	got := DetectToolUse(dir, "local/model", HFConfig{Architectures: []string{"MistralForCausalLM"}})
	want := ToolUseMeta{HasToolSupport: true, ToolCallParser: "mistral", DetectionMethod: "chat_template_regex"}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

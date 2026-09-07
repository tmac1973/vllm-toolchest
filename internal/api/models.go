package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	list := s.registry.List()

	if !isHTMX(r) {
		respondJSON(w, list)
		return
	}

	respondHTML(w)
	if len(list) == 0 {
		fmt.Fprint(w, `<p>No models registered. <a href="/models/browse">Download models from HuggingFace &rarr;</a></p>`)
		return
	}

	fmt.Fprint(w, `<table><thead><tr>
    <th>Model</th><th>Quant</th><th>Size</th><th>VRAM</th><th>Tools</th><th></th>
  </tr></thead><tbody>`)

	for _, m := range list {
		quantBadge := quantBadgeHTML(m.Quantization)
		vramLabel := vramLabelHTML(m.VRAMEstimate)
		toolBadge := safeHTML("")
		if m.ToolUse.HasToolSupport {
			toolBadge = safeHTML(fmt.Sprintf(`<ins title="%s">tool use</ins>`,
				esc(m.ToolUse.ToolCallParser)))
		}

		orphanBadge := safeHTML("")
		if m.Orphaned {
			orphanBadge = ` <del>missing</del>`
		}

		// Model IDs and display names come from HuggingFace and land in both
		// attributes and text below, so this printer escapes them.
		row := htmlPrinter(w)
		sid := safeID(m.ID)
		row(`<tr>
      <td>
        <a href="#" hx-get="/api/models/config-panel?id=%s" hx-target="#config-%s" hx-swap="innerHTML">
          <strong>%s</strong>
        </a>%s
        <br><small style="opacity:0.6;">%s</small>
      </td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
      <td>
        <button class="secondary outline" style="padding:0.15rem 0.5rem;font-size:0.75rem;"
                hx-delete="/api/models/delete?id=%s"
                hx-confirm="Delete %s? This removes all model files."
                hx-target="closest tr"
                hx-swap="outerHTML">Delete</button>
      </td>
    </tr>
    <tr id="config-%s-row"><td colspan="6"><div id="config-%s"></div></td></tr>`,
			m.ID, sid,
			m.DisplayName, orphanBadge,
			m.ID,
			quantBadge,
			huggingface.FormatBytes(m.TotalSizeBytes),
			vramLabel,
			toolBadge,
			m.ID, m.DisplayName,
			sid, sid)
	}

	fmt.Fprint(w, `</tbody></table>`)
}

func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	respondJSON(w, m)
}

func (s *Server) handleModelConfigPanel(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	respondHTML(w)
	c := m.VLLMConfig
	sid := safeID(m.ID)

	maxCtx := m.HFConfig.MaxPositionEmbeddings
	if maxCtx == 0 {
		maxCtx = 4096
	}
	modelLen := c.MaxModelLen
	if modelLen == 0 {
		modelLen = maxCtx
	}

	// Escapes every string argument. The config fields below are free text the
	// operator typed -- a JSON speculative config is nothing but double quotes,
	// and interpolating one raw closed the value="..." attribute it was being
	// written into, so the browser kept only the leading brace.
	p := htmlPrinter(w)

	p(`<article style="margin:0.5rem 0;">
  <header style="display:flex;justify-content:space-between;align-items:center;">
    <span>Configuration: %s</span>
    <small id="save-status-%s" style="opacity:0.7;"></small>
  </header>
  <div style="margin-bottom:1rem;padding:0.75rem;border-radius:0.25rem;background:var(--pico-card-sectioning-background-color);">
    <strong>VRAM Estimate:</strong> %.1f GB weights + %.1f GB overhead = <strong>%.1f GB</strong> &mdash; %s
  </div>
  <form hx-put="/api/models/config?id=%s" hx-trigger="change" hx-target="#config-%s" hx-swap="innerHTML" hx-include="closest form">`,
		m.DisplayName, sid,
		m.VRAMEstimate.WeightMemoryGB, m.VRAMEstimate.ActivationGB, m.VRAMEstimate.TotalSingleGPUGB, m.VRAMEstimate.FitLabel,
		m.ID, sid)

	// ── Core ──
	p(`<fieldset><legend>Core</legend><div class="grid">`)

	// dtype
	p(`<label title="Data type for model weights. 'auto' uses the model's native dtype (usually bfloat16). Use float16 for older GPUs that don't support bfloat16.">dtype <select name="dtype">`)
	for _, opt := range []string{"auto", "float16", "bfloat16", "float32"} {
		p(`<option value="%s"%s>%s</option>`, opt, selected(c.Dtype == opt), opt)
	}
	p(`</select></label>`)

	// Context length dropdown
	ctxOptions := []int{2048, 4096, 8192, 16384, 32768, 65536, 131072}
	// Ensure model max is in list, filter to <= max
	hasMax := false
	for _, v := range ctxOptions {
		if v == maxCtx {
			hasMax = true
		}
	}
	if !hasMax && maxCtx > 0 {
		var tmp []int
		inserted := false
		for _, v := range ctxOptions {
			if !inserted && maxCtx < v {
				tmp = append(tmp, maxCtx)
				inserted = true
			}
			tmp = append(tmp, v)
		}
		if !inserted {
			tmp = append(tmp, maxCtx)
		}
		ctxOptions = tmp
	}
	if maxCtx > 0 {
		var tmp []int
		for _, v := range ctxOptions {
			if v <= maxCtx {
				tmp = append(tmp, v)
			}
		}
		ctxOptions = tmp
	}
	isCustomCtx := true
	for _, v := range ctxOptions {
		if v == modelLen {
			isCustomCtx = false
		}
	}

	p(`<label title="Maps to vLLM's --max-model-len. Maximum sequence length (prompt + generation). Lower values use less VRAM for KV cache and let the engine warm up faster. Model max is %d.">Context length <small>(--max-model-len, max: %d)</small>`, maxCtx, maxCtx)
	p(`<select name="max_model_len" id="ctx-sel-%s" onchange="var c=document.getElementById('ctx-cust-%s');if(this.value==='custom'){c.style.display='';c.name='max_model_len';this.name='';}else{c.style.display='none';c.name='';this.name='max_model_len';}">`, sid, sid)
	for _, v := range ctxOptions {
		label := fmt.Sprintf("%d", v)
		if v >= 1024 {
			label = fmt.Sprintf("%dK", v/1024)
		}
		if v == maxCtx {
			label += " (max)"
		}
		p(`<option value="%d"%s>%s</option>`, v, selected(!isCustomCtx && v == modelLen), label)
	}
	customDisplay := "display:none;"
	customName := ""
	if isCustomCtx {
		customDisplay = ""
		customName = "max_model_len"
	}
	p(`<option value="custom"%s>Custom...</option></select>`, selected(isCustomCtx))
	p(`<input type="number" id="ctx-cust-%s" name="%s" value="%d" min="256" max="%d" style="%smargin-top:0.25rem;" placeholder="Custom context length">`, sid, customName, modelLen, maxCtx, customDisplay)
	p(`</label></div>`)

	// Tensor parallel + GPU memory util
	p(`<div class="grid">`)
	p(`<label title="Split the model across multiple GPUs. TP must evenly divide the model's attention heads. Higher TP reduces per-GPU memory but adds inter-GPU communication overhead.">Tensor parallel <select name="tensor_parallel_size">`)
	for _, tp := range []int{1, 2, 4, 8} {
		label := fmt.Sprintf("%d GPU", tp)
		if tp > 1 {
			label += "s"
		}
		p(`<option value="%d"%s>%s</option>`, tp, selected(c.TensorParallelSize == tp), label)
	}
	p(`</select></label>`)

	p(`<label title="Fraction of GPU memory vLLM is allowed to use (0.1-0.99). Higher = more KV cache (longer contexts) but risk OOM. 0.90 is a safe default.">GPU memory utilization`)
	p(`<input type="range" name="gpu_memory_utilization" min="0.1" max="0.99" step="0.01" value="%.2f" oninput="this.nextElementSibling.textContent=this.value">`, c.GPUMemoryUtilization)
	p(`<small>%.2f</small></label>`, c.GPUMemoryUtilization)
	p(`</div></fieldset>`)

	// ── Performance ──
	p(`<fieldset><legend>Performance</legend><div class="grid">`)
	p(`<label title="Disable HIP/CUDA graph compilation. Slower steady-state but faster startup. Enable if you get graph compilation errors."><input type="checkbox" name="enforce_eager" role="switch"%s> Enforce eager mode</label>`, checked(c.EnforceEager))
	p(`<label title="Cache KV blocks for shared prefixes (system prompts). Speeds up requests sharing the same prefix. Safe for most workloads."><input type="checkbox" name="enable_prefix_caching" role="switch"%s> Prefix caching</label>`, checked(c.EnablePrefixCaching))
	p(`</div>`)

	p(`<label title="Max concurrent sequences (requests). Lower = less memory, more predictable latency. Single user: 1-4. Multi-user: 16-64.">Max concurrent sequences <select name="max_num_seqs">`)
	for _, n := range []int{1, 4, 8, 16, 32, 64, 128, 256} {
		p(`<option value="%d"%s>%d</option>`, n, selected(c.MaxNumSeqs == n), n)
	}
	p(`</select></label>`)

	// Both of these existed in the config and in BuildArgs but had no form
	// field, so every save through this panel wiped them: the batched-token
	// budget was reset to 0, and chunked prefill -- read from a checkbox that
	// was never rendered -- was forced off.
	p(`<label title="Prefill in chunks so a long prompt does not monopolise a step. Normally leave on; vLLM V1 enables it by default.">
<input type="checkbox" name="enable_chunked_prefill" role="switch"%s> Enable chunked prefill</label>`,
		checked(c.EnableChunkedPrefill))
	p(`<label title="Tokens per prefill chunk. 0 lets vLLM choose.">Max batched tokens <input type="number" name="max_num_batched_tokens" value="%d" min="0" step="256" placeholder="0 = auto"></label>`,
		c.MaxNumBatchedTokens)

	// The ceiling depends on the model's hidden size and the tensor-parallel
	// size, so it cannot be written into a static tooltip -- and exceeding it
	// is silent, which is exactly the kind of thing that should not be left to
	// the operator to derive from a third-party source file.
	if advice, warn := batchedTokenAdvice(
		s.vllmEnv.IsRadiance(), m.HFConfig.HiddenSize,
		c.TensorParallelSize, c.MaxNumBatchedTokens,
	); advice != "" {
		if warn {
			p(`<small style="display:block;margin-top:-0.5rem;margin-bottom:0.5rem;color:var(--pico-del-color);"><strong>Warning:</strong> %s</small>`, advice)
		} else {
			p(`<small style="display:block;margin-top:-0.5rem;margin-bottom:0.5rem;opacity:0.7;">%s</small>`, advice)
		}
	}
	p(`</fieldset>`)

	// ── Quantization ──
	p(`<fieldset><legend>Quantization <a href="#" onclick="document.getElementById('quant-help-%s').showModal();return false;" style="font-size:0.75rem;text-decoration:none;" title="What are quantization methods?">&#9432;</a></legend><div class="grid">`, sid)

	quantOpts := compatibleQuantOptions(m.Quantization.Method, m.Quantization.Sym, m.Quantization.Bits)
	p(`<label title="Quantization method for inference. Options are filtered to what this model supports based on its format.">Method <select name="quantization">`)
	for _, opt := range quantOpts {
		p(`<option value="%s"%s>%s</option>`, opt.val, selected(c.Quantization == opt.val), opt.label)
	}
	p(`</select></label>`)

	p(`<label title="Data type for KV cache. 'auto' uses FP16. FP8 halves cache memory for longer contexts, with minimal quality tradeoff.">KV cache dtype <select name="kv_cache_dtype">`)
	for _, opt := range []string{"auto", "fp8", "fp8_e5m2", "fp8_e4m3"} {
		p(`<option value="%s"%s>%s</option>`, opt, selected(c.KVCacheDtype == opt), opt)
	}
	p(`</select></label></div></fieldset>`)

	// ── Tool Use ──
	// Determine effective parser: config override > auto-detected
	effectiveParser := c.ToolCallParser
	if effectiveParser == "" {
		effectiveParser = m.ToolUse.ToolCallParser
	}
	toolEnabled := c.EnableAutoToolChoice && effectiveParser != ""

	p(`<fieldset><legend>Tool Use</legend>`)

	if m.ToolUse.HasToolSupport {
		p(`<p style="margin-bottom:0.5rem;"><small>Detected: <strong>%s</strong> parser (via %s)</small></p>`,
			m.ToolUse.ToolCallParser, m.ToolUse.DetectionMethod)
	} else {
		p(`<p style="margin-bottom:0.5rem;"><small style="opacity:0.6;">No tool use support detected in this model's chat template.</small></p>`)
	}

	p(`<label title="Enable OpenAI-compatible tool/function calling. Sets both --enable-auto-tool-choice and --tool-call-parser. The parser is auto-detected from the model's chat template.">`)
	p(`<input type="checkbox" name="enable_auto_tool_choice" role="switch"%s`, checked(toolEnabled))
	// When toggled on, auto-set the parser from detection; when off, clear it
	p(` onchange="var ps=this.closest('form').querySelector('[name=tool_call_parser]');if(this.checked){ps.value='%s';}else{ps.value='';}">`, m.ToolUse.ToolCallParser)
	p(` Enable tool use</label>`)

	// Parser override -- collapsed by default, expandable for advanced users
	p(`<details style="margin-top:0.5rem;"><summary style="font-size:0.85rem;cursor:pointer;">Parser override</summary>`)
	p(`<label title="Override the auto-detected parser. Only change this if auto-detection got it wrong. Using the wrong parser will break tool calling.">`)
	p(`<select name="tool_call_parser">`)
	// Auto option appears outside any optgroup so it's always at the top.
	autoSelected := c.ToolCallParser == "" || c.ToolCallParser == m.ToolUse.ToolCallParser
	p(`<option value=""%s>(auto: %s)</option>`, selected(autoSelected), m.ToolUse.ToolCallParser)
	// Groups ordered most-common first. Parser names must match vLLM's
	// registration names in vllm/tool_parsers/__init__.py.
	for _, grp := range []struct {
		label string
		opts  []struct{ val, label string }
	}{
		{"Common", []struct{ val, label string }{
			{"hermes", "hermes — Hermes, NousResearch, Qwen 2.5, plain Qwen 3"},
			{"qwen3_xml", "qwen3_xml — Qwen 3.5+, Qwen thinking variants, MiMo"},
			{"qwen3_coder", "qwen3_coder — Qwen 3 Coder"},
			{"llama3_json", "llama3_json — Llama 3.1 / 3.2 / 3.3"},
			{"llama4_pythonic", "llama4_pythonic — Llama 4"},
			{"llama4_json", "llama4_json — Llama 4 (json output)"},
			{"mistral", "mistral — Mistral, Mixtral"},
			{"pythonic", "pythonic — Python-style function calls"},
			{"openai", "openai — OpenAI-compatible JSON"},
		}},
		{"DeepSeek", []struct{ val, label string }{
			{"deepseek_v3", "deepseek_v3 — DeepSeek V3, R1"},
			{"deepseek_v31", "deepseek_v31 — DeepSeek V3.1"},
			{"deepseek_v32", "deepseek_v32 — DeepSeek V3.2"},
			{"deepseek_v4", "deepseek_v4 — DeepSeek V4"},
		}},
		{"Granite", []struct{ val, label string }{
			{"granite", "granite — IBM Granite"},
			{"granite4", "granite4 — IBM Granite 4"},
			{"granite-20b-fc", "granite-20b-fc — IBM Granite 20B FC"},
		}},
		{"GLM", []struct{ val, label string }{
			{"glm45", "glm45 — GLM 4.5 MoE"},
			{"glm47", "glm47 — GLM 4.7 MoE"},
		}},
		{"Cohere", []struct{ val, label string }{
			{"cohere_command3", "cohere_command3 — Command R / R+"},
			{"cohere_command4", "cohere_command4 — Command R 4"},
		}},
		{"Other", []struct{ val, label string }{
			{"gemma4", "gemma4 — Gemma 4"},
			{"functiongemma", "functiongemma — FunctionGemma"},
			{"phi4_mini_json", "phi4_mini_json — Phi-4 Mini"},
			{"internlm", "internlm — InternLM"},
			{"jamba", "jamba — Jamba (AI21)"},
			{"kimi_k2", "kimi_k2 — Moonshot Kimi K2"},
			{"minimax", "minimax — MiniMax"},
			{"minimax_m2", "minimax_m2 — MiniMax M2"},
			{"hunyuan_a13b", "hunyuan_a13b — Tencent Hunyuan A13B"},
			{"hy_v3", "hy_v3 — Tencent Hunyuan V3"},
			{"olmo3", "olmo3 — AI2 OLMo 3"},
			{"longcat", "longcat — LongCat Flash"},
			{"ernie45", "ernie45 — Baidu Ernie 4.5"},
			{"lfm2", "lfm2 — Liquid LFM-2"},
			{"xlam", "xlam — Salesforce xLAM"},
			{"seed_oss", "seed_oss — ByteDance Seed-OSS"},
			{"step3", "step3 — StepFun Step3"},
			{"step3p5", "step3p5 — StepFun Step3.5"},
			{"mimo", "mimo — Xiaomi MiMo (uses Qwen3 XML)"},
			{"apertus", "apertus — Apertus"},
			{"gigachat3", "gigachat3 — GigaChat 3"},
			{"poolside_v1", "poolside_v1 — Poolside V1"},
		}},
	} {
		p(`<optgroup label="%s">`, grp.label)
		for _, opt := range grp.opts {
			isSelected := c.ToolCallParser == opt.val && c.ToolCallParser != m.ToolUse.ToolCallParser
			p(`<option value="%s"%s>%s</option>`, opt.val, selected(isSelected), opt.label)
		}
		p(`</optgroup>`)
	}
	p(`</select></label></details>`)
	p(`</fieldset>`)

	// ── Advanced ──
	p(`<fieldset><legend>Advanced</legend>`)
	p(`<label title="Execute custom Python code from the model's HF repo. Required by some models (Yi, InternLM) but is a security risk."><input type="checkbox" name="trust_remote_code" role="switch"%s> Trust remote code</label>
<small style="display:block;margin-top:-0.5rem;margin-bottom:0.5rem;opacity:0.7;">Warning: executes arbitrary code from the model repo</small>`, checked(c.TrustRemoteCode))
	p(`<div class="grid">`)
	p(`<label title="Override vLLM's attention backend. Leave on auto unless a model or image needs a specific one.">Attention backend
<select name="attention_backend">`)
	for _, opt := range attentionBackendOptions(s.vllmEnv.IsRadiance()) {
		p(`<option value="%s"%s>%s</option>`, opt.Val, selected(c.AttentionBackend == opt.Val), opt.Label)
	}
	p(`</select></label>`)
	p(`<label title="Parser for models that emit a separate reasoning/thinking channel, e.g. qwen3 or deepseek_r1.">Reasoning parser <input type="text" name="reasoning_parser" value="%s" placeholder="(none)"></label>`, c.ReasoningParser)
	p(`</div>`)

	p(`<div class="grid">`)
	p(`<label title="Required alongside prefix caching on hybrid linear-attention (gated-delta-net / mamba) models. 'align' snapshots the recurrent state at block boundaries so those layers become cacheable too.">Mamba cache mode
<select name="mamba_cache_mode">`)
	for _, opt := range []struct{ val, label string }{
		{"", "(vLLM default)"},
		{"align", "align — makes hybrid models prefix-cacheable"},
	} {
		p(`<option value="%s"%s>%s</option>`, opt.val, selected(c.MambaCacheMode == opt.val), opt.label)
	}
	p(`</select></label>`)
	p(`<label title="Pin the KV cache pool size in bytes. 0 lets vLLM size it, which under-reports free VRAM on ROCm. Clear this before measuring anything memory-related — while set, the pool stops responding to other memory changes.">KV cache memory (bytes) <input type="number" name="kv_cache_memory" value="%d" min="0" step="1048576" placeholder="0 = auto"></label>`, c.KVCacheMemory)
	p(`</div>`)

	p(`<label title="Raw JSON for --speculative-config. Lossless: the target model verifies every drafted token.">Speculative config (JSON) <input type="text" name="speculative_config" value="%s" placeholder='{&#34;method&#34;:&#34;mtp&#34;,&#34;num_speculative_tokens&#34;:8}'></label>`, c.SpeculativeConfig)
	p(`<label title="Raw JSON for --compilation-config. Most useful for trimming the CUDA-graph capture ladder to sizes this serve can actually reach.">Compilation config (JSON) <input type="text" name="compilation_config" value="%s" placeholder='{&#34;cudagraph_capture_sizes&#34;:[1,2,4,8,16,32]}'></label>`, c.CompilationConfig)

	p(`<div class="grid">`)
	p(`<label title="Pass --no-async-scheduling. Required when a speculative config sets disable_padded_drafter_batch, which is incompatible with async scheduling."><input type="checkbox" name="disable_async_scheduling" role="switch"%s> Disable async scheduling</label>`, checked(c.DisableAsyncScheduling))
	p(`<label title="Serve a vision-language checkpoint text-only, skipping its vision tower."><input type="checkbox" name="language_model_only" role="switch"%s> Language model only</label>`, checked(c.LanguageModelOnly))
	p(`</div>`)

	p(`<div class="grid">`)
	p(`<label title="Path to a Jinja chat template that overrides the one in the model repo.">Chat template <input type="text" name="chat_template" value="%s" placeholder="(use the model's own)"></label>`, c.ChatTemplate)
	p(`<label title="Load the tokenizer from a different path or HF repo than the model.">Tokenizer <input type="text" name="tokenizer" value="%s" placeholder="(use the model's own)"></label>`, c.Tokenizer)
	p(`</div>`)

	p(`<label title="Raw CLI flags appended to vllm serve. Quoted values are kept together, so JSON with spaces survives. e.g. --disable-log-requests --swap-space 4">Extra flags <input type="text" name="extra_flags" value="%s" placeholder="--disable-log-requests --swap-space 4"></label>`, c.ExtraFlags)
	p(`</fieldset>`)

	// ── Effective flags (read-only) ──
	effParser := c.ToolCallParser
	if effParser == "" && c.EnableAutoToolChoice {
		effParser = m.ToolUse.ToolCallParser
	}
	effCfg := c.StartConfig()
	effCfg.MaxModelLen = modelLen
	effCfg.ToolCallParser = effParser

	args := process.BuildArgs(effCfg)
	modelPath := process.ResolveModelPath(m.LocalPath)
	// Mirror process.Manager.Start exactly -- this box is what the operator
	// reads to reason about a failed launch, so a preview that differs from
	// the real argv is worse than none.
	bin, argv := s.vllmEnv.ServeCommand(modelPath, append(
		[]string{"--host", "0.0.0.0", "--port", fmt.Sprintf("%d", s.cfg.VLLMPort)}, args...))
	cmdLine := bin
	for _, a := range argv {
		cmdLine += " " + a
	}

	p(`<fieldset><legend>Effective Command</legend>`)
	p(`<pre style="font-size:0.75rem;white-space:pre-wrap;word-break:break-all;padding:0.5rem;background:var(--pico-code-background-color);border-radius:0.25rem;user-select:all;cursor:pointer;" title="Click to select all">%s</pre>`, cmdLine)
	p(`</fieldset>`)

	p(`</form>`)

	// ── Quant help modal ──
	p(`<dialog id="quant-help-%s"><article style="max-width:600px;">`, sid)
	p(`<header><button aria-label="Close" rel="prev" onclick="this.closest('dialog').close();"></button><strong>Quantization Methods</strong></header>`)
	p(`<table style="font-size:0.85rem;"><thead><tr><th>Method</th><th>Bits</th><th>Description</th></tr></thead><tbody>`)
	p(`<tr><td><strong>FP16/BF16</strong></td><td>16</td><td>Full precision. Best quality, highest VRAM. BF16 preferred for newer GPUs.</td></tr>`)
	p(`<tr><td><strong>AWQ</strong></td><td>4</td><td>Activation-aware Weight Quantization. Good quality/size tradeoff. Pre-quantized models on HF.</td></tr>`)
	p(`<tr><td><strong>GPTQ</strong></td><td>4</td><td>Post-Training Quantization. Similar to AWQ, depends on calibration data.</td></tr>`)
	p(`<tr><td><strong>Marlin</strong></td><td>4</td><td>Optimized kernel for compatible GPTQ/AWQ. Faster inference, same model files. Needs symmetric quant.</td></tr>`)
	p(`<tr><td><strong>FP8</strong></td><td>8</td><td>8-bit float. Half VRAM of FP16, minimal quality loss. Native on Ada/RDNA4.</td></tr>`)
	p(`<tr><td><strong>BitsAndBytes</strong></td><td>4/8</td><td>Dynamic quantization at load. No pre-quantized model needed. Slower load, requires eager mode.</td></tr>`)
	p(`<tr><td><strong>SqueezeLLM</strong></td><td>4</td><td>Older method. Rarely used with newer models.</td></tr>`)
	p(`<tr><td><strong>compressed_tensors</strong></td><td>mixed</td><td>vLLM native format. Mixed precision across layers.</td></tr>`)
	p(`</tbody></table>`)
	p(`<footer><button onclick="this.closest('dialog').close();">Close</button></footer></article></dialog>`)
	p(`</article>`)
}

func (s *Server) handleUpdateModelConfig(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	var cfg models.VLLMConfig
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "json") {
		json.NewDecoder(r.Body).Decode(&cfg)
	} else {
		r.ParseForm()
		cfg = models.VLLMConfig{
			Dtype:                  r.FormValue("dtype"),
			MaxModelLen:            formInt(r, "max_model_len"),
			TensorParallelSize:     formInt(r, "tensor_parallel_size"),
			GPUMemoryUtilization:   formFloat(r, "gpu_memory_utilization"),
			EnforceEager:           r.FormValue("enforce_eager") == "on",
			EnablePrefixCaching:    r.FormValue("enable_prefix_caching") == "on",
			EnableChunkedPrefill:   r.FormValue("enable_chunked_prefill") == "on",
			MaxNumSeqs:             formInt(r, "max_num_seqs"),
			MaxNumBatchedTokens:    formInt(r, "max_num_batched_tokens"),
			Quantization:           r.FormValue("quantization"),
			LoadFormat:             r.FormValue("load_format"),
			KVCacheDtype:           r.FormValue("kv_cache_dtype"),
			TrustRemoteCode:        r.FormValue("trust_remote_code") == "on",
			EnableAutoToolChoice:   r.FormValue("enable_auto_tool_choice") == "on",
			ToolCallParser:         r.FormValue("tool_call_parser"),
			ReasoningParser:        r.FormValue("reasoning_parser"),
			AttentionBackend:       r.FormValue("attention_backend"),
			MambaCacheMode:         r.FormValue("mamba_cache_mode"),
			SpeculativeConfig:      strings.TrimSpace(r.FormValue("speculative_config")),
			CompilationConfig:      strings.TrimSpace(r.FormValue("compilation_config")),
			KVCacheMemory:          int64(formInt(r, "kv_cache_memory")),
			DisableAsyncScheduling: r.FormValue("disable_async_scheduling") == "on",
			LanguageModelOnly:      r.FormValue("language_model_only") == "on",
			Tokenizer:              r.FormValue("tokenizer"),
			ChatTemplate:           r.FormValue("chat_template"),
			ExtraFlags:             r.FormValue("extra_flags"),
		}
		if cfg.LoadFormat == "" {
			cfg.LoadFormat = "auto"
		}
		if cfg.KVCacheDtype == "" {
			cfg.KVCacheDtype = "auto"
		}
		if cfg.Dtype == "" {
			cfg.Dtype = "auto"
		}
	}

	// Auto-fill parser from model detection when tool use is enabled
	if cfg.EnableAutoToolChoice && cfg.ToolCallParser == "" {
		cfg.ToolCallParser = m.ToolUse.ToolCallParser
	}
	// Clear parser when tool use is disabled
	if !cfg.EnableAutoToolChoice {
		cfg.ToolCallParser = ""
	}

	if err := s.registry.UpdateConfig(id, cfg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Recompute VRAM estimate
	m.VLLMConfig = cfg
	m.VRAMEstimate = models.EstimateVRAM(m)
	s.registry.Register(m)

	if isHTMX(r) {
		// Re-render the full config panel so effective command updates
		s.handleModelConfigPanel(w, r)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if err := s.registry.Delete(id, true); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		return // empty response removes the row
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleScanModels(w http.ResponseWriter, r *http.Request) {
	s.registry.Maintenance()
	list := s.registry.List()

	if isHTMX(r) {
		respondHTML(w)
		fmt.Fprintf(w, `<p><ins>Scan complete. %d models registered.</ins></p>`, len(list))
		return
	}
	respondJSON(w, map[string]int{"count": len(list)})
}

type quantOption struct {
	val   string
	label string
}

// compatibleQuantOptions returns quantization methods compatible with the model's format.
func compatibleQuantOptions(detectedMethod string, sym bool, bits int) []quantOption {
	switch detectedMethod {
	case "awq":
		opts := []quantOption{
			{"", "auto-detect (AWQ)"},
			{"awq", "AWQ"},
		}
		// Marlin works with 4-bit symmetric AWQ
		if bits == 4 {
			opts = append(opts, quantOption{"marlin", "Marlin (optimized AWQ kernel)"})
		}
		return opts

	case "gptq":
		opts := []quantOption{
			{"", "auto-detect (GPTQ)"},
			{"gptq", "GPTQ"},
		}
		// Marlin works with 4-bit symmetric GPTQ without desc_act
		if bits == 4 && sym {
			opts = append(opts, quantOption{"marlin", "Marlin (optimized GPTQ kernel)"})
		}
		return opts

	case "fp8":
		return []quantOption{
			{"", "auto-detect (FP8)"},
			{"fp8", "FP8"},
		}

	case "bitsandbytes":
		return []quantOption{
			{"", "auto-detect (BitsAndBytes)"},
			{"bitsandbytes", "BitsAndBytes"},
		}

	case "compressed_tensors":
		return []quantOption{
			{"", "auto-detect (compressed_tensors)"},
			{"compressed_tensors", "compressed_tensors"},
		}

	case "squeezellm":
		return []quantOption{
			{"", "auto-detect (SqueezeLLM)"},
			{"squeezellm", "SqueezeLLM"},
		}

	case "gguf":
		return []quantOption{
			{"", "auto-detect (GGUF)"},
			{"gguf", "GGUF"},
		}

	default:
		// FP16/BF16 unquantized model -- can do dynamic quantization
		return []quantOption{
			{"", "None (full precision)"},
			{"bitsandbytes", "BitsAndBytes (dynamic 4/8-bit at load)"},
			{"fp8", "FP8 (dynamic 8-bit, needs GPU support)"},
		}
	}
}

func quantBadgeHTML(q models.QuantMeta) safeHTML {
	label := strings.ToUpper(q.Method)
	if label == "NONE" || label == "" {
		label = "FP16"
	}
	if q.Bits > 0 {
		label += fmt.Sprintf(" %d-bit", q.Bits)
	}
	if q.GGUFQuantType != "" {
		label = "GGUF " + q.GGUFQuantType
	}
	color := quantBadgeColor(q.Method)
	return safeHTML(fmt.Sprintf(`<span style="display:inline-block;padding:0.1rem 0.4rem;border-radius:0.2rem;font-size:0.7rem;background:%s;color:#fff;">%s</span>`, color, esc(label)))
}

func vramLabelHTML(est models.VRAMEstimate) safeHTML {
	if est.WeightMemoryGB == 0 {
		return "—"
	}
	color := "#2d8a4e"
	if est.NeedsTP2 {
		color = "#b86e00"
	}
	if est.TooLarge {
		color = "#b83d3d"
	}
	return safeHTML(fmt.Sprintf(`<span style="color:%s;">%.1f GB<br><small>%s</small></span>`,
		color, est.TotalSingleGPUGB, esc(est.FitLabel)))
}

func safeID(id string) string {
	r := strings.NewReplacer(
		"/", "--",
		".", "-",
		":", "-",
		" ", "-",
	)
	return r.Replace(id)
}

func checked(v bool) safeHTML {
	if v {
		return " checked"
	}
	return ""
}

func selected(v bool) safeHTML {
	if v {
		return " selected"
	}
	return ""
}

func formInt(r *http.Request, key string) int {
	v := r.FormValue(key)
	if v == "" {
		return 0
	}
	var n int
	fmt.Sscanf(v, "%d", &n)
	return n
}

func formFloat(r *http.Request, key string) float64 {
	v := r.FormValue(key)
	if v == "" {
		return 0
	}
	var f float64
	fmt.Sscanf(v, "%f", &f)
	return f
}

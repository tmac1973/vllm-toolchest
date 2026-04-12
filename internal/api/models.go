package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
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
		toolBadge := ""
		if m.ToolUse.HasToolSupport {
			toolBadge = fmt.Sprintf(`<ins title="%s">tool use</ins>`, m.ToolUse.ToolCallParser)
		}

		orphanBadge := ""
		if m.Orphaned {
			orphanBadge = ` <del>missing</del>`
		}

		sid := safeID(m.ID)
		fmt.Fprintf(w, `<tr>
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

	fmt.Fprintf(w, `<article style="margin:0.5rem 0;">
  <header>Configuration: %s</header>

  <div style="margin-bottom:1rem;padding:0.75rem;border-radius:0.25rem;background:var(--pico-card-sectioning-background-color);">
    <strong>VRAM Estimate:</strong> %.1f GB weights + %.1f GB overhead = <strong>%.1f GB</strong>
    &mdash; %s
  </div>

  <form hx-put="/api/models/config?id=%s" hx-swap="none">
    <fieldset>
      <legend>Core</legend>
      <div class="grid">
        <label>dtype
          <select name="dtype">`,
		m.DisplayName,
		m.VRAMEstimate.WeightMemoryGB, m.VRAMEstimate.ActivationGB, m.VRAMEstimate.TotalSingleGPUGB,
		m.VRAMEstimate.FitLabel,
		m.ID)

	for _, opt := range []string{"auto", "float16", "bfloat16", "float32"} {
		sel := ""
		if c.Dtype == opt {
			sel = " selected"
		}
		fmt.Fprintf(w, `<option value="%s"%s>%s</option>`, opt, sel, opt)
	}

	maxCtx := m.HFConfig.MaxPositionEmbeddings
	if maxCtx == 0 {
		maxCtx = 4096
	}
	modelLen := c.MaxModelLen
	if modelLen == 0 {
		modelLen = maxCtx
	}

	fmt.Fprintf(w, `</select>
        </label>
        <label>Context length <small>(max: %d)</small>
          <input type="number" name="max_model_len" value="%d" min="256" max="%d">
        </label>
      </div>
      <div class="grid">
        <label>Tensor parallel
          <select name="tensor_parallel_size">
            <option value="1"%s>1 GPU</option>
            <option value="2"%s>2 GPUs (TP=2)</option>
          </select>
        </label>
        <label>GPU memory utilization
          <input type="range" name="gpu_memory_utilization" min="0.1" max="0.99" step="0.01" value="%.2f"
                 oninput="this.nextElementSibling.textContent=this.value">
          <small>%.2f</small>
        </label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Performance</legend>
      <div class="grid">
        <label>
          <input type="checkbox" name="enforce_eager" role="switch"%s>
          Enforce eager mode
        </label>
        <label>
          <input type="checkbox" name="enable_prefix_caching" role="switch"%s>
          Prefix caching
        </label>
      </div>
      <label>Max concurrent sequences
        <select name="max_num_seqs">`,
		maxCtx, modelLen, maxCtx,
		selected(c.TensorParallelSize == 1), selected(c.TensorParallelSize == 2),
		c.GPUMemoryUtilization, c.GPUMemoryUtilization,
		checked(c.EnforceEager), checked(c.EnablePrefixCaching))

	for _, n := range []int{1, 4, 8, 16, 32, 64, 128, 256} {
		sel := ""
		if c.MaxNumSeqs == n {
			sel = " selected"
		}
		fmt.Fprintf(w, `<option value="%d"%s>%d</option>`, n, sel, n)
	}

	fmt.Fprintf(w, `</select>
      </label>
    </fieldset>

    <fieldset>
      <legend>Quantization</legend>
      <div class="grid">
        <label>Method
          <select name="quantization">`)

	for _, opt := range []string{"", "awq", "gptq", "fp8", "gguf", "bitsandbytes", "marlin", "squeezellm", "compressed_tensors"} {
		sel := ""
		if c.Quantization == opt {
			sel = " selected"
		}
		label := opt
		if label == "" {
			label = "auto-detect"
		}
		fmt.Fprintf(w, `<option value="%s"%s>%s</option>`, opt, sel, label)
	}

	fmt.Fprintf(w, `</select>
        </label>
        <label>KV cache dtype
          <select name="kv_cache_dtype">`)

	for _, opt := range []string{"auto", "fp8", "fp8_e5m2", "fp8_e4m3"} {
		sel := ""
		if c.KVCacheDtype == opt {
			sel = " selected"
		}
		fmt.Fprintf(w, `<option value="%s"%s>%s</option>`, opt, sel, opt)
	}

	fmt.Fprintf(w, `</select>
        </label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Tool Use</legend>
      <div class="grid">
        <label>
          <input type="checkbox" name="enable_auto_tool_choice" role="switch"%s>
          Enable auto tool choice
        </label>
        <label>Tool call parser
          <select name="tool_call_parser">
            <option value=""%s>(none)</option>
            <option value="hermes"%s>hermes</option>
            <option value="llama3_json"%s>llama3_json</option>
            <option value="mistral"%s>mistral</option>
            <option value="granite"%s>granite</option>
            <option value="internlm"%s>internlm</option>
            <option value="jamba"%s>jamba</option>
            <option value="pythonic"%s>pythonic</option>
          </select>
        </label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Advanced</legend>
      <label>
        <input type="checkbox" name="trust_remote_code" role="switch"%s>
        Trust remote code <small>(security risk)</small>
      </label>
      <label>Extra flags
        <input type="text" name="extra_flags" value="%s" placeholder="--disable-log-requests --swap-space 4">
      </label>
    </fieldset>

    <button type="submit">Save Configuration</button>
  </form>
</article>`,
		checked(c.EnableAutoToolChoice),
		selected(c.ToolCallParser == ""),
		selected(c.ToolCallParser == "hermes"),
		selected(c.ToolCallParser == "llama3_json"),
		selected(c.ToolCallParser == "mistral"),
		selected(c.ToolCallParser == "granite"),
		selected(c.ToolCallParser == "internlm"),
		selected(c.ToolCallParser == "jamba"),
		selected(c.ToolCallParser == "pythonic"),
		checked(c.TrustRemoteCode),
		c.ExtraFlags)
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
			Dtype:                r.FormValue("dtype"),
			MaxModelLen:          formInt(r, "max_model_len"),
			TensorParallelSize:   formInt(r, "tensor_parallel_size"),
			GPUMemoryUtilization: formFloat(r, "gpu_memory_utilization"),
			EnforceEager:         r.FormValue("enforce_eager") == "on",
			EnablePrefixCaching:  r.FormValue("enable_prefix_caching") == "on",
			EnableChunkedPrefill: r.FormValue("enable_chunked_prefill") == "on",
			MaxNumSeqs:           formInt(r, "max_num_seqs"),
			Quantization:         r.FormValue("quantization"),
			LoadFormat:           r.FormValue("load_format"),
			KVCacheDtype:         r.FormValue("kv_cache_dtype"),
			TrustRemoteCode:      r.FormValue("trust_remote_code") == "on",
			EnableAutoToolChoice: r.FormValue("enable_auto_tool_choice") == "on",
			ToolCallParser:       r.FormValue("tool_call_parser"),
			ExtraFlags:           r.FormValue("extra_flags"),
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

	if err := s.registry.UpdateConfig(id, cfg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Recompute VRAM estimate
	m.VLLMConfig = cfg
	m.VRAMEstimate = models.EstimateVRAM(m)
	s.registry.Register(m)

	if isHTMX(r) {
		respondHTML(w)
		fmt.Fprint(w, `<p><ins>Configuration saved.</ins></p>`)
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

func quantBadgeHTML(q models.QuantMeta) string {
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
	return fmt.Sprintf(`<span style="display:inline-block;padding:0.1rem 0.4rem;border-radius:0.2rem;font-size:0.7rem;background:%s;color:#fff;">%s</span>`, color, label)
}

func vramLabelHTML(est models.VRAMEstimate) string {
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
	return fmt.Sprintf(`<span style="color:%s;">%.1f GB<br><small>%s</small></span>`,
		color, est.TotalSingleGPUGB, est.FitLabel)
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

func checked(v bool) string {
	if v {
		return " checked"
	}
	return ""
}

func selected(v bool) string {
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

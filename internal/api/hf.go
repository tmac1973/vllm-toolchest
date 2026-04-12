package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
)

func (s *Server) handleHFSearch(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprint(w, `<p>Enter a search query above.</p>`)
			return
		}
		respondJSON(w, []any{})
		return
	}

	results, err := s.hfClient.Search(r.Context(), query)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><mark>Search error: %s</mark></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	if !isHTMX(r) {
		respondJSON(w, results)
		return
	}

	groups := huggingface.GroupResults(results)
	respondHTML(w)

	if len(groups) == 0 {
		fmt.Fprint(w, `<p>No models found.</p>`)
		return
	}

	for _, g := range groups {
		primary := g.Variants[0]
		sid := safeID(primary.ID)

		gatedBadge := ""
		if primary.Gated.IsGated() {
			gatedBadge = ` <small style="color:var(--pico-del-color);">[gated]</small>`
		}

		fmt.Fprintf(w, `<article style="margin-bottom:0.5rem;">
  <header style="padding:0.5rem 1rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <strong>%s</strong>%s
        <br><small style="opacity:0.7;">%s &middot; %s downloads &middot; %s likes</small>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:0.15rem;">`,
			primary.ID, gatedBadge,
			primary.Author,
			formatCount(primary.Downloads), formatCount(primary.Likes))

		// Variant badges -- each is clickable and loads that specific variant
		for _, v := range g.Variants {
			label := v.QuantFormat
			if label == "" {
				label = "FP16"
			}
			color := quantBadgeColor(label)
			vSid := safeID(v.ID)
			fmt.Fprintf(w, `<a href="#" hx-get="/api/hf/model?id=%s" hx-target="#detail-%s" hx-swap="innerHTML" title="%s" style="display:inline-block;padding:0.15rem 0.5rem;border-radius:0.2rem;font-size:0.7rem;background:%s;color:#fff;text-decoration:none;cursor:pointer;">%s</a>`,
				v.ID, sid, v.ID, color, label)
			_ = vSid
		}

		fmt.Fprintf(w, `</div>
    </div>
  </header>
  <div id="detail-%s" style="padding:0 1rem;"></div>
</article>`, sid)

		if len(g.Variants) == 0 {
			continue
		}
	}
}

func (s *Server) handleHFModel(w http.ResponseWriter, r *http.Request) {
	modelID := r.URL.Query().Get("id")
	if modelID == "" {
		http.Error(w, "missing id", http.StatusBadRequest)
		return
	}

	detail, err := s.hfClient.GetModel(r.Context(), modelID)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><mark>Error loading model details: %s</mark></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	if !isHTMX(r) {
		respondJSON(w, detail)
		return
	}

	respondHTML(w)

	quantInfo := "FP16/BF16 (unquantized)"
	if detail.QuantFormat != "" {
		quantInfo = detail.QuantFormat
		if detail.QuantConfig != nil && detail.QuantConfig.Bits > 0 {
			quantInfo += fmt.Sprintf(" %d-bit", detail.QuantConfig.Bits)
			if detail.QuantConfig.GroupSize > 0 {
				quantInfo += fmt.Sprintf(" (group_size=%d)", detail.QuantConfig.GroupSize)
			}
		}
	}

	gatedWarning := ""
	if detail.Gated.IsGated() && s.cfg.HFToken == "" {
		gatedWarning = `<p><mark>This is a gated model. Configure your HF token in <a href="/settings">Settings</a> to download.</mark></p>`
	}

	vramWarning := ""
	metrics := s.monitor.Current()
	if detail.VRAMEstGB > 0 && len(metrics.GPU) > 0 {
		gpuVRAM := float64(metrics.GPU[0].VRAMTotalMB) / 1024
		if detail.VRAMEstGB > gpuVRAM {
			vramWarning = fmt.Sprintf(`<p><mark>Estimated VRAM (%.1f GB) exceeds GPU memory (%.0f GB). Consider a quantized variant or TP=2.</mark></p>`,
				detail.VRAMEstGB, gpuVRAM)
		}
	}

	fmt.Fprintf(w, `%s%s
<div class="grid" style="margin-bottom:0.5rem;">
  <div><small>Architecture</small><br><strong>%s</strong></div>
  <div><small>Quantization</small><br><strong>%s</strong></div>
  <div><small>Est. VRAM</small><br><strong>%s</strong></div>
  <div><small>Download Size</small><br><strong>%s</strong></div>
</div>`,
		gatedWarning, vramWarning,
		orDash(detail.Architecture),
		quantInfo,
		formatVRAM(detail.VRAMEstGB),
		huggingface.FormatBytes(detail.TotalSize))

	// Collapsible file list
	downloadableFiles := 0
	for _, f := range detail.Files {
		if f.Category != "skip" {
			downloadableFiles++
		}
	}

	fmt.Fprintf(w, `<details style="margin-bottom:0.5rem;">
  <summary>%d files to download</summary>
  <table style="font-size:0.85rem;">
    <thead><tr><th>File</th><th>Size</th><th>Type</th></tr></thead>
    <tbody>`, downloadableFiles)

	for _, f := range detail.Files {
		if f.Category == "skip" {
			continue
		}
		fmt.Fprintf(w, `<tr><td><code>%s</code></td><td>%s</td><td>%s</td></tr>`,
			f.Filename, huggingface.FormatBytes(f.Size), f.Category)
	}
	fmt.Fprint(w, `</tbody></table></details>`)

	// Download button
	sid := safeID(detail.ID)
	disabled := ""
	if detail.Gated.IsGated() && s.cfg.HFToken == "" {
		disabled = ` disabled`
	}
	fmt.Fprintf(w, `<div id="dl-%s">
  <button hx-post="/api/hf/download?model_id=%s"
          hx-target="#dl-%s"
          hx-swap="innerHTML"
          style="margin:0;"%s>Download (%s)</button>
</div>`,
		sid,
		detail.ID, sid, disabled,
		huggingface.FormatBytes(detail.TotalSize))
}

func (s *Server) handleHFDownload(w http.ResponseWriter, r *http.Request) {
	// Accept model_id from query param, form body, or JSON body
	modelID := r.URL.Query().Get("model_id")
	if modelID == "" {
		r.ParseForm()
		modelID = r.FormValue("model_id")
	}
	if modelID == "" {
		var req struct {
			ModelID string `json:"model_id"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		modelID = req.ModelID
	}

	if modelID == "" {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprint(w, `<p><mark>Missing model_id</mark></p>`)
			return
		}
		http.Error(w, "missing model_id", http.StatusBadRequest)
		return
	}

	detail, err := s.hfClient.GetModel(r.Context(), modelID)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><mark>Error: %s</mark></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// Filter to downloadable files
	var filesToDownload []huggingface.ModelFile
	hasSafetensors := false
	for _, f := range detail.Files {
		if strings.HasSuffix(strings.ToLower(f.Filename), ".safetensors") {
			hasSafetensors = true
			break
		}
	}

	for _, f := range detail.Files {
		if f.Category == "skip" {
			continue
		}
		if hasSafetensors && f.Category == "weight" &&
			strings.HasSuffix(strings.ToLower(f.Filename), ".bin") {
			continue
		}
		if strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			continue
		}
		filesToDownload = append(filesToDownload, f)
	}

	if len(filesToDownload) == 0 {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprint(w, `<p><mark>No downloadable weight files found.</mark></p>`)
			return
		}
		http.Error(w, "no downloadable files", http.StatusBadRequest)
		return
	}

	downloadID, err := s.downloader.Start(modelID, filesToDownload)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprintf(w, `<p><mark>Error: %s</mark></p>`, err)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		// Show inline progress polling + trigger the active-downloads section to refresh
		w.Header().Set("HX-Trigger", "download-started")
		fmt.Fprintf(w, `<div hx-get="/api/hf/download/%s/progress" hx-trigger="load, every 2s" hx-swap="innerHTML">
  <progress value="0" max="100" style="margin:0;"></progress>
  <small>Starting download...</small>
</div>`, downloadID)
		return
	}

	respondJSON(w, map[string]string{"download_id": downloadID})
}

func (s *Server) handleHFDownloadProgress(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	d := s.downloader.GetProgress(id)
	if d == nil {
		if isHTMX(r) {
			respondHTML(w)
			fmt.Fprint(w, `<p><ins>Download complete!</ins> <a href="/models">View in Models &rarr;</a></p>`)
			return
		}
		http.Error(w, "download not found", http.StatusNotFound)
		return
	}

	if !isHTMX(r) {
		respondJSON(w, d)
		return
	}

	respondHTML(w)
	pct := 0
	if d.TotalBytes > 0 {
		pct = int(d.BytesDownloaded * 100 / d.TotalBytes)
	}

	switch d.Status {
	case "complete":
		fmt.Fprint(w, `<p><ins>Download complete!</ins> <a href="/models">View in Models &rarr;</a></p>`)
	case "failed":
		fmt.Fprintf(w, `<p><del>Download failed: %s</del></p>`, d.Error)
	case "cancelled":
		fmt.Fprint(w, `<p>Download cancelled.</p>`)
	default:
		speed := huggingface.FormatBytes(d.SpeedBPS) + "/s"
		total := huggingface.FormatBytes(d.TotalBytes)
		downloaded := huggingface.FormatBytes(d.BytesDownloaded)
		// Keep polling
		fmt.Fprintf(w, `<div hx-get="/api/hf/download/%s/progress" hx-trigger="every 2s" hx-swap="innerHTML">
  <progress value="%d" max="100" style="margin:0;"></progress>
  <small>%s / %s (%s) &mdash; %d%% &mdash; %d/%d files</small>
</div>`, id, pct, downloaded, total, speed, pct, d.CompletedFiles, d.TotalFiles)
	}
}

// handleHFActiveDownloads returns progress for all active downloads (used by both browse and models pages).
func (s *Server) handleHFActiveDownloads(w http.ResponseWriter, r *http.Request) {
	downloads := s.downloader.ActiveDownloads()

	if !isHTMX(r) {
		respondJSON(w, downloads)
		return
	}

	respondHTML(w)
	activeCount := 0
	for _, dl := range downloads {
		if dl.Status == "complete" {
			continue
		}
		activeCount++
		pct := 0
		if dl.TotalBytes > 0 {
			pct = int(dl.BytesDownloaded * 100 / dl.TotalBytes)
		}
		speed := huggingface.FormatBytes(dl.SpeedBPS) + "/s"
		fmt.Fprintf(w, `<article style="margin-bottom:0.5rem;padding:0.75rem 1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <strong>%s</strong>
    <button class="secondary outline" style="padding:0.15rem 0.5rem;font-size:0.75rem;"
            hx-delete="/api/hf/download/%s" hx-target="closest article" hx-swap="outerHTML">Cancel</button>
  </div>
  <progress value="%d" max="100" style="margin:0.25rem 0;"></progress>
  <small>%s / %s (%s) &mdash; %d%% &mdash; %d/%d files</small>
</article>`,
			dl.ModelID, dl.ID, pct,
			huggingface.FormatBytes(dl.BytesDownloaded),
			huggingface.FormatBytes(dl.TotalBytes),
			speed, pct, dl.CompletedFiles, dl.TotalFiles)
	}
}

func (s *Server) handleHFDownloadCancel(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.downloader.Cancel(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if isHTMX(r) {
		respondHTML(w)
		return // empty = remove the article
	}
	w.WriteHeader(http.StatusNoContent)
}

func quantBadgeColor(format string) string {
	switch strings.ToUpper(format) {
	case "AWQ":
		return "#2d8a4e"
	case "GPTQ":
		return "#2d6a8a"
	case "FP8":
		return "#7a3db8"
	case "BNB-4BIT", "BNB-8BIT":
		return "#b8a33d"
	default:
		return "#555"
	}
}

func formatCount(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func formatVRAM(gb float64) string {
	if gb <= 0 {
		return "Unknown"
	}
	return fmt.Sprintf("~%.1f GB", gb)
}

func orDash(s string) string {
	if s == "" {
		return "\u2014"
	}
	return s
}

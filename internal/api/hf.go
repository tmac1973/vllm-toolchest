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
	for _, g := range groups {
		primary := g.Variants[0]
		gatedBadge := ""
		if primary.Gated != "" {
			gatedBadge = ` <small>[gated]</small>`
		}

		fmt.Fprintf(w, `<article style="margin-bottom:0.5rem;">
  <header style="padding:0.5rem 1rem;">
    <strong>%s</strong>%s
    <small style="opacity:0.7;">%s &middot; %s downloads &middot; %s likes</small>
  </header>
  <div style="padding:0.25rem 1rem 0.5rem;">`,
			primary.ID, gatedBadge, primary.Author,
			formatCount(primary.Downloads), formatCount(primary.Likes))

		// Show variant badges
		seen := map[string]bool{}
		for _, v := range g.Variants {
			label := v.QuantFormat
			if label == "" {
				label = "FP16"
			}
			if seen[label] {
				continue
			}
			seen[label] = true
			color := quantBadgeColor(label)
			fmt.Fprintf(w, `<a href="#" hx-get="/api/hf/model?id=%s" hx-target="#model-detail" hx-swap="innerHTML" style="display:inline-block;padding:0.15rem 0.5rem;margin:0.1rem;border-radius:0.25rem;font-size:0.75rem;background:%s;color:#fff;text-decoration:none;">%s</a>`,
				v.ID, color, label)
		}

		fmt.Fprint(w, `</div></article>`)
	}

	if len(groups) == 0 {
		fmt.Fprint(w, `<p>No models found.</p>`)
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
			fmt.Fprintf(w, `<article><p><mark>Error: %s</mark></p></article>`, err)
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
	// Quant info
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
	if detail.Gated != "" && s.cfg.HFToken == "" {
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

	fmt.Fprintf(w, `<article>
  <header><strong>%s</strong></header>
  %s%s
  <div class="grid">
    <div>
      <small>Architecture</small>
      <p>%s</p>
    </div>
    <div>
      <small>Quantization</small>
      <p>%s</p>
    </div>
    <div>
      <small>Est. VRAM</small>
      <p>%s</p>
    </div>
    <div>
      <small>Download Size</small>
      <p>%s</p>
    </div>
  </div>`,
		detail.ID,
		gatedWarning, vramWarning,
		orDash(detail.Architecture),
		quantInfo,
		formatVRAM(detail.VRAMEstGB),
		huggingface.FormatBytes(detail.TotalSize))

	// File table
	fmt.Fprint(w, `<table><thead><tr><th>File</th><th>Size</th><th>Type</th></tr></thead><tbody>`)
	for _, f := range detail.Files {
		if f.Category == "skip" {
			continue
		}
		fmt.Fprintf(w, `<tr><td>%s</td><td>%s</td><td>%s</td></tr>`,
			f.Filename, huggingface.FormatBytes(f.Size), f.Category)
	}
	fmt.Fprint(w, `</tbody></table>`)

	// Download button
	safeID := strings.ReplaceAll(detail.ID, "/", "--")
	disabled := ""
	if detail.Gated != "" && s.cfg.HFToken == "" {
		disabled = ` disabled`
	}
	fmt.Fprintf(w, `<button hx-post="/api/hf/download" hx-vals='{"model_id":"%s"}' hx-target="#dl-%s" hx-swap="innerHTML"%s>Download Model (%s)</button>
  <div id="dl-%s"></div>
</article>`,
		detail.ID, safeID, disabled, huggingface.FormatBytes(detail.TotalSize), safeID)
}

func (s *Server) handleHFDownload(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID string   `json:"model_id"`
		Files   []string `json:"files,omitempty"` // optional: specific files
	}

	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "json") {
		json.NewDecoder(r.Body).Decode(&req)
	} else {
		r.ParseForm()
		req.ModelID = r.FormValue("model_id")
	}

	if req.ModelID == "" {
		http.Error(w, "missing model_id", http.StatusBadRequest)
		return
	}

	// Fetch file list if not specified
	detail, err := s.hfClient.GetModel(r.Context(), req.ModelID)
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
		// Skip .bin weights if safetensors available
		if hasSafetensors && f.Category == "weight" &&
			strings.HasSuffix(strings.ToLower(f.Filename), ".bin") {
			continue
		}
		if len(req.Files) > 0 {
			found := false
			for _, name := range req.Files {
				if name == f.Filename {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		filesToDownload = append(filesToDownload, f)
	}

	downloadID, err := s.downloader.Start(req.ModelID, filesToDownload)
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
		fmt.Fprintf(w, `<div hx-ext="sse" sse-connect="/api/hf/download/%s/progress" sse-swap="progress">
  <progress value="0" max="100"></progress>
  <small>Starting download...</small>
</div>`, downloadID)
		return
	}

	respondJSON(w, map[string]string{"download_id": downloadID})
}

func (s *Server) handleHFDownloadProgress(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	ch, err := s.downloader.Subscribe(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	defer s.downloader.Unsubscribe(id, ch)

	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	for {
		select {
		case progress, ok := <-ch:
			if !ok {
				return
			}

			if isHTMX(r) {
				pct := 0
				if progress.TotalBytes > 0 {
					pct = int(progress.BytesDownloaded * 100 / progress.TotalBytes)
				}
				speed := huggingface.FormatBytes(progress.SpeedBPS) + "/s"
				total := huggingface.FormatBytes(progress.TotalBytes)
				downloaded := huggingface.FormatBytes(progress.BytesDownloaded)

				switch progress.Status {
				case "complete":
					sse.SendEvent("progress",
						fmt.Sprintf(`<p><ins>Download complete!</ins> <a href="/models">View in Models &rarr;</a></p>`))
					return
				case "failed":
					sse.SendEvent("progress",
						fmt.Sprintf(`<p><del>Download failed: %s</del></p>`, progress.Error))
					return
				case "cancelled":
					sse.SendEvent("progress", `<p>Download cancelled.</p>`)
					return
				default:
					html := fmt.Sprintf(`<progress value="%d" max="100"></progress>
<small>%s / %s (%s) &mdash; %d%% &mdash; %d/%d files</small>`,
						pct, downloaded, total, speed, pct,
						progress.CompletedFiles, progress.TotalFiles)
					sse.SendEvent("progress", html)
				}
			} else {
				data, _ := json.Marshal(progress)
				sse.SendEvent("progress", string(data))
				if progress.Status == "complete" || progress.Status == "failed" || progress.Status == "cancelled" {
					return
				}
			}

		case <-r.Context().Done():
			return
		}
	}
}

func (s *Server) handleHFActiveDownloads(w http.ResponseWriter, r *http.Request) {
	downloads := s.downloader.ActiveDownloads()

	if !isHTMX(r) {
		respondJSON(w, downloads)
		return
	}

	respondHTML(w)
	if len(downloads) == 0 {
		return // empty is fine for htmx
	}

	for _, dl := range downloads {
		if dl.Status == "complete" {
			continue
		}
		pct := 0
		if dl.TotalBytes > 0 {
			pct = int(dl.BytesDownloaded * 100 / dl.TotalBytes)
		}
		fmt.Fprintf(w, `<article style="margin-bottom:0.5rem;padding:0.5rem 1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <strong>%s</strong>
    <button class="secondary outline" style="padding:0.15rem 0.5rem;font-size:0.75rem;" hx-delete="/api/hf/download/%s" hx-swap="none">Cancel</button>
  </div>
  <progress value="%d" max="100" style="margin:0.25rem 0;"></progress>
  <small>%s / %s &mdash; %d%% &mdash; %d/%d files</small>
</article>`,
			dl.ModelID, dl.ID, pct,
			huggingface.FormatBytes(dl.BytesDownloaded),
			huggingface.FormatBytes(dl.TotalBytes),
			pct, dl.CompletedFiles, dl.TotalFiles)
	}
}

func (s *Server) handleHFDownloadCancel(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.downloader.Cancel(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func quantBadgeColor(format string) string {
	switch strings.ToUpper(format) {
	case "AWQ":
		return "#2d8a4e"
	case "GPTQ":
		return "#2d6a8a"
	case "GGUF":
		return "#b86e00"
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
		return "—"
	}
	return s
}

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
			s.renderPartial(w, "plain_message", "Enter a search query above.")
			return
		}
		respondJSON(w, []any{})
		return
	}

	// The filter goes to the Hub, not just applied to what comes back — see
	// Client.Search.
	quantFilter := r.URL.Query().Get("quant")
	results, err := s.hfClient.Search(r.Context(), query, quantFilter)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "notice", fmt.Sprintf("Search error: %s", err))
			return
		}
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// The Hub's tag filter is broader than the bucket, so narrow what came
	// back to what actually belongs in it.
	if quantFilter != "" {
		var filtered []huggingface.ModelSearchResult
		for _, res := range results {
			if huggingface.MatchesQuantFilter(res.QuantFormat, res.Tags, quantFilter) {
				filtered = append(filtered, res)
			}
		}
		results = filtered
	}

	if !isHTMX(r) {
		respondJSON(w, results)
		return
	}

	respondHTML(w)
	s.renderPartial(w, "hf_results", struct {
		Groups      []hfResultGroup
		QuantFilter string
	}{
		Groups:      hfResultGroups(huggingface.GroupResults(results)),
		QuantFilter: huggingface.QuantFilterLabel(quantFilter),
	})
}

// hfResultGroup is one repo family in the search results: the primary repo,
// plus one badge per distinct quantization somebody published it in.
type hfResultGroup struct {
	ID        string
	SafeID    string
	Author    string
	Downloads string
	Likes     string
	Gated     bool
	Variants  []hfResultVariant
}

type hfResultVariant struct {
	ID     string
	Format string
	Color  string
}

func hfResultGroups(groups []huggingface.ModelGroup) []hfResultGroup {
	out := make([]hfResultGroup, 0, len(groups))
	for _, g := range groups {
		if len(g.Variants) == 0 {
			continue
		}
		primary := g.Variants[0]
		row := hfResultGroup{
			ID:        primary.ID,
			SafeID:    safeID(primary.ID),
			Author:    primary.Author,
			Downloads: formatCount(primary.Downloads),
			Likes:     formatCount(primary.Likes),
			Gated:     primary.Gated.IsGated(),
		}
		// One badge per quantization format. Several repos often publish the
		// same format; the first one wins, which is the primary author's.
		seen := map[string]bool{}
		for _, v := range g.Variants {
			format := v.QuantFormat
			if format == huggingface.QuantUnknown {
				// The repo published no config and carried no marker. That is
				// not evidence of full precision, and saying "FP16" here is
				// how a search for unquantized models filled up with 4-bit
				// ones.
				format = "unknown"
			}
			if seen[format] {
				continue
			}
			seen[format] = true
			row.Variants = append(row.Variants, hfResultVariant{
				ID: v.ID, Format: format, Color: quantBadgeColor(format),
			})
		}
		out = append(out, row)
	}
	return out
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
			s.renderPartial(w, "notice", fmt.Sprintf("Error loading model details: %s", err))
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
	s.renderPartial(w, "hf_model_detail", s.newHFModelDetail(detail))
}

// hfModelDetail is the expanded panel under a search result: what the model
// is, what it will cost, and the button that starts the download.
type hfModelDetail struct {
	ID           string
	SafeID       string
	Architecture string
	QuantInfo    string
	VRAMLabel    string
	SizeLabel    string
	Files        []hfDetailFile
	// GatedWarning is set when the model needs a token this server does not
	// have; Disabled then keeps the button from offering a download that
	// cannot succeed.
	GatedWarning bool
	VRAMWarning  string
	Disabled     bool
}

type hfDetailFile struct {
	Filename  string
	SizeLabel string
	Category  string
}

func (s *Server) newHFModelDetail(detail *huggingface.ModelDetail) hfModelDetail {
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

	// Compared against the first GPU only: this is a "will it obviously not
	// fit" warning, not the tensor-parallel planning the model card does.
	var vramWarning string
	if metrics := s.monitor.Current(); detail.VRAMEstGB > 0 && len(metrics.GPU) > 0 {
		gpuVRAM := float64(metrics.GPU[0].VRAMTotalMB) / 1024
		if detail.VRAMEstGB > gpuVRAM {
			vramWarning = fmt.Sprintf(
				"Estimated VRAM (%.1f GB) exceeds GPU memory (%.0f GB). Consider a quantized variant or TP=2.",
				detail.VRAMEstGB, gpuVRAM)
		}
	}

	gated := detail.Gated.IsGated() && s.cfg.HFToken == ""

	v := hfModelDetail{
		ID:           detail.ID,
		SafeID:       safeID(detail.ID),
		Architecture: orDash(detail.Architecture),
		QuantInfo:    quantInfo,
		VRAMLabel:    formatVRAM(detail.VRAMEstGB),
		SizeLabel:    huggingface.FormatBytes(detail.TotalSize),
		GatedWarning: gated,
		VRAMWarning:  vramWarning,
		Disabled:     gated,
	}
	for _, f := range detail.Files {
		if f.Category == "skip" {
			continue
		}
		v.Files = append(v.Files, hfDetailFile{
			Filename:  f.Filename,
			SizeLabel: huggingface.FormatBytes(f.Size),
			Category:  f.Category,
		})
	}
	return v
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
			s.renderPartial(w, "notice", "Missing model_id")
			return
		}
		http.Error(w, "missing model_id", http.StatusBadRequest)
		return
	}

	detail, err := s.hfClient.GetModel(r.Context(), modelID)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "notice", fmt.Sprintf("Error: %s", err))
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
			s.renderPartial(w, "notice", "No downloadable weight files found.")
			return
		}
		http.Error(w, "no downloadable files", http.StatusBadRequest)
		return
	}

	downloadID, err := s.downloader.Start(modelID, filesToDownload)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "notice", fmt.Sprintf("Error: %s", err))
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		// Show inline progress polling + trigger the active-downloads section to refresh
		w.Header().Set("HX-Trigger", "download-started")
		s.renderPartial(w, "download_started", downloadID)
		return
	}

	respondJSON(w, map[string]string{"download_id": downloadID})
}

// downloadView is one download's progress, shaped for the templates. Percent
// is precomputed: a template cannot divide without turning integers into
// floats first.
type downloadView struct {
	ID              string
	ModelID         string
	Status          string
	Error           string
	Percent         int
	DownloadedLabel string
	TotalLabel      string
	SpeedLabel      string
	CompletedFiles  int
	TotalFiles      int
}

func newDownloadView(d huggingface.DownloadProgress) downloadView {
	pct := 0
	if d.TotalBytes > 0 {
		pct = int(d.BytesDownloaded * 100 / d.TotalBytes)
	}
	return downloadView{
		ID:              d.ID,
		ModelID:         d.ModelID,
		Status:          d.Status,
		Error:           d.Error,
		Percent:         pct,
		DownloadedLabel: huggingface.FormatBytes(d.BytesDownloaded),
		TotalLabel:      huggingface.FormatBytes(d.TotalBytes),
		SpeedLabel:      huggingface.FormatBytes(d.SpeedBPS) + "/s",
		CompletedFiles:  d.CompletedFiles,
		TotalFiles:      d.TotalFiles,
	}
}

func (s *Server) handleHFDownloadProgress(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	d := s.downloader.GetProgress(id)
	if d == nil {
		// The download has already been reaped, which for the poller that got
		// here means it finished.
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "download_progress", downloadView{Status: "complete"})
			return
		}
		http.Error(w, "download not found", http.StatusNotFound)
		return
	}

	if !isHTMX(r) {
		respondJSON(w, d)
		return
	}

	view := newDownloadView(*d)
	view.ID = id
	respondHTML(w)
	s.renderPartial(w, "download_progress", view)
}

// downloadRow is one line of the downloads panel: either a transfer in
// flight, or one that stopped partway and still has bytes on disk.
type downloadRow struct {
	downloadView
	Active      bool
	OnDiskLabel string
	PartFiles   int
}

// handleHFDownloads renders the downloads panel, shown above both the models
// and browse pages.
func (s *Server) handleHFDownloads(w http.ResponseWriter, r *http.Request) {
	active := s.downloader.ActiveDownloads()

	if !isHTMX(r) {
		respondJSON(w, map[string]any{
			"active":     active,
			"incomplete": s.downloader.ListIncomplete(),
		})
		return
	}

	rows := []downloadRow{}
	running := map[string]bool{}
	for _, dl := range active {
		if dl.Status != "downloading" {
			continue
		}
		running[dl.ModelID] = true
		rows = append(rows, downloadRow{downloadView: newDownloadView(dl), Active: true})
	}
	// Anything with partial files and nothing moving them is resumable. A
	// model that is downloading right now also has .part files, so the running
	// set is what keeps it from being listed twice.
	for _, inc := range s.downloader.ListIncomplete() {
		if running[inc.ModelID] {
			continue
		}
		rows = append(rows, downloadRow{
			downloadView: downloadView{ModelID: inc.ModelID},
			OnDiskLabel:  huggingface.FormatBytes(inc.OnDisk),
			PartFiles:    inc.PartFiles,
		})
	}

	respondHTML(w)
	s.renderPartial(w, "downloads_panel", struct{ Rows []downloadRow }{rows})
}

// handleHFDiscardIncomplete deletes a stalled download's partial files. It is
// the only path that removes them: pausing and failing deliberately leave them
// so the transfer can pick up where it stopped.
func (s *Server) handleHFDiscardIncomplete(w http.ResponseWriter, r *http.Request) {
	modelID := r.URL.Query().Get("model_id")
	if modelID == "" {
		http.Error(w, "missing model_id", http.StatusBadRequest)
		return
	}
	// Only ever for a model that is not in the registry: a registered model's
	// directory holds its weights, and this must not be a way to delete those.
	if _, registered := s.registry.Get(modelID); registered {
		http.Error(w, "model is registered — remove it from the Models page instead", http.StatusConflict)
		return
	}
	if err := s.downloader.Discard(modelID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleHFDownloadCancel(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.downloader.Cancel(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	// Remove any partial registry entry (model ID uses / not --)
	modelID := strings.ReplaceAll(id, "--", "/")
	s.registry.Delete(modelID, false) // files already cleaned up by downloader

	if isHTMX(r) {
		respondHTML(w)
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
	case "FP8":
		return "#7a3db8"
	case "BNB-4BIT", "BNB-8BIT", "BITSANDBYTES":
		return "#b8a33d"
	case "COMPRESSED-TENSORS":
		return "#b5622d"
	case "MXFP4", "NVFP4":
		return "#a8397c"
	case "QUARK", "AUTOROUND", "MODELOPT":
		return "#4a6b8a"
	case "UNKNOWN":
		// Deliberately dimmer than the rest: it is an absence of information,
		// not a format.
		return "#3f3f3f"
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

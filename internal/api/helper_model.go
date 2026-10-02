package api

import (
	"fmt"
	"log/slog"
	"net/http"
	"sync/atomic"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// helperModel is the app's helper model, or nil when none is installed. A
// model becomes the helper by being downloaded as one, from Settings; the
// same repository downloaded from the search page is an ordinary model.
func (s *Server) helperModel() *models.Model {
	for _, m := range s.registry.List() {
		if m.Helper && !m.Orphaned {
			return m
		}
	}
	return nil
}

// helperWanted is set while Settings has asked for the helper's download, so
// the completion hook knows to claim it.
type helperWanted struct{ atomic.Bool }

// onTransferComplete is the downloader's completion hook: register the model
// as any download is, then claim the helper if that is what this was.
func (s *Server) onTransferComplete(downloadID, modelID, modelDir string) {
	_, existed := s.registry.Get(modelID)
	recordTransfer(s.registry)(downloadID, modelID, modelDir)
	if !existed {
		s.seedFromFeed(modelID)
	}
	if modelID == models.HelperRepo && s.wantHelper.Load() {
		if err := s.registry.SetHelper(modelID, true); err != nil {
			slog.Warn("the helper model downloaded but could not be marked as the helper", "error", err)
			return
		}
		s.wantHelper.Store(false)
	}
}

// helperUtil is the memory fraction the helper runs at on this host.
func (s *Server) helperUtil() float64 {
	return models.HelperUtil(s.cfg.GPUMemoryUtil)
}

// helperPanelData is what the helper_model_panel partial renders.
type helperPanelData struct {
	Repo        string
	Installed   bool
	SizeLabel   string
	DownloadID  string // a transfer in flight
	TooLarge    string // HelperFits's sentence when it does not fit
	Message     string
	MessageKind string // "ok" or "error"
}

func (s *Server) helperPanel(msg, kind string) helperPanelData {
	d := helperPanelData{Repo: models.HelperRepo, Message: msg, MessageKind: kind}
	if s.downloader != nil && s.downloader.ActiveModelIDs()[models.HelperRepo] {
		d.DownloadID = huggingface.DownloadID(models.HelperRepo)
	}
	if h := s.helperModel(); h != nil {
		d.Installed = true
		d.SizeLabel = huggingface.FormatBytes(h.TotalSizeBytes)
	}
	if ok, why := models.HelperFits(s.gpuInventory(), s.helperUtil()); !ok {
		d.TooLarge = why
	}
	return d
}

func (s *Server) renderHelperPanel(w http.ResponseWriter, msg, kind string) {
	respondHTML(w)
	s.renderPartial(w, "helper_model_panel", s.helperPanel(msg, kind))
}

// handleHelperPanel renders the Settings section for the helper model.
func (s *Server) handleHelperPanel(w http.ResponseWriter, r *http.Request) {
	s.renderHelperPanel(w, "", "")
}

// handleDownloadHelper starts the helper's download, or claims a copy the
// operator already has.
func (s *Server) handleDownloadHelper(w http.ResponseWriter, r *http.Request) {
	if s.helperModel() != nil {
		s.renderHelperPanel(w, "", "")
		return
	}
	if m, ok := s.registry.Get(models.HelperRepo); ok && !m.Orphaned {
		if err := s.registry.SetHelper(m.ID, true); err != nil {
			s.renderHelperPanel(w, "Not marked as the helper: "+err.Error(), "error")
			return
		}
		s.renderHelperPanel(w, "The copy already installed is now the helper model.", "ok")
		return
	}
	s.wantHelper.Store(true)
	if _, err := s.startTransfer(r.Context(), models.HelperRepo, "", false); err != nil {
		s.wantHelper.Store(false)
		s.renderHelperPanel(w, "The download did not start: "+err.Error(), "error")
		return
	}
	w.Header().Set("HX-Trigger", "downloadsChanged")
	s.renderHelperPanel(w, "", "")
}

// handleRemoveHelper deletes the helper model and its files.
func (s *Server) handleRemoveHelper(w http.ResponseWriter, r *http.Request) {
	h := s.helperModel()
	if h == nil {
		s.renderHelperPanel(w, "", "")
		return
	}
	size := h.TotalSizeBytes
	if err := s.registry.Delete(h.ID, true); err != nil {
		s.renderHelperPanel(w, "Not removed: "+err.Error(), "error")
		return
	}
	s.renderHelperPanel(w, fmt.Sprintf("Removed, freeing %s.", huggingface.FormatBytes(size)), "ok")
}

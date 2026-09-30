package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"regexp"

	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// commitSHA is a full git commit id, the only form of revision accepted from
// a request: it goes into a URL, and it is meant to pin one snapshot.
var commitSHA = regexp.MustCompile(`^[0-9a-f]{40}$`)

// transferError is a refusal to start a transfer, with the status it should
// be reported under.
type transferError struct {
	status int
	msg    string
}

func (e *transferError) Error() string { return e.msg }

func transferStatus(err error) int {
	var te *transferError
	if errors.As(err, &te) {
		return te.status
	}
	return http.StatusInternalServerError
}

// transferBlocker is why a transfer into modelID cannot start right now, or
// "" when it can. The update panel shows it in place of its button and
// startTransfer enforces it, so the two cannot disagree.
func (s *Server) transferBlocker(modelID string, plan huggingface.Plan) string {
	if m, registered := s.registry.Get(modelID); registered {
		// Downloads always land under the models directory. A model
		// registered from somewhere else would get a second, partial copy
		// there while the files it is served from stayed as they were.
		if filepath.Clean(m.LocalPath) != filepath.Clean(s.downloader.ModelDir(modelID)) {
			return fmt.Sprintf("This model's files are in %s, outside the models directory downloads are written to.", m.LocalPath)
		}
		// The engine has the checkpoint open. Nothing is swapped in until the
		// whole update has arrived, but that last step would still change the
		// files under a running server.
		if st := s.process.GetStatus(); (st.State == process.StateRunning || st.State == process.StateStarting) && st.ModelID == modelID {
			return "This model is being served. Stop the server before updating its files."
		}
	}
	// -1 means free space is unknown, which gates nothing; see newHFModelDetail.
	if avail := s.downloader.AvailableForDownload(); avail >= 0 && plan.FetchBytes() > avail {
		return fmt.Sprintf("Not enough free disk space: needs %s, and only %s is available after the %s safety margin and any in-flight downloads.",
			huggingface.FormatBytes(plan.FetchBytes()), huggingface.FormatBytes(avail),
			huggingface.FormatBytes(huggingface.DiskSafetyMarginBytes))
	}
	return ""
}

// startTransfer lists a repo's files at revision and starts bringing the
// model directory in line with them. It is the one way a download begins,
// whether the model is new, half-fetched or already registered.
//
// An empty revision means the repo as it stands now.
func (s *Server) startTransfer(ctx context.Context, modelID, revision string, removeStale bool) (string, error) {
	if revision != "" && !commitSHA.MatchString(revision) {
		return "", &transferError{http.StatusBadRequest, "revision must be a full commit id"}
	}
	// Already running: hand back the transfer in flight rather than judging
	// it against a disk budget it is itself using up.
	if s.downloader.ActiveModelIDs()[modelID] {
		return huggingface.DownloadID(modelID), nil
	}

	revision, files, err := s.hfClient.GetFiles(ctx, modelID, revision)
	if err != nil {
		return "", &transferError{http.StatusBadGateway, err.Error()}
	}
	if len(files) == 0 {
		return "", &transferError{http.StatusBadRequest, "No downloadable weight files found."}
	}

	plan := s.downloader.Plan(modelID, revision, files)
	if why := s.transferBlocker(modelID, plan); why != "" {
		return "", &transferError{http.StatusConflict, why}
	}

	_, registered := s.registry.Get(modelID)
	return s.downloader.Start(huggingface.Request{
		ModelID:     modelID,
		Revision:    revision,
		Files:       files,
		RemoveStale: removeStale,
		Update:      registered,
	})
}

// modelUpdateView is the panel under a model card that says what has changed
// upstream since the model was downloaded.
type modelUpdateView struct {
	ID          string
	SafeID      string
	DisplayName string

	// Revision is the commit the comparison was made against. The update is
	// started against the same one, so what gets fetched is what was shown.
	Revision      string
	RevisionShort string
	// SyncedShort is the commit the files were last brought in line with, or
	// "" for a model downloaded before that was recorded.
	SyncedShort string
	CommitsURL  string

	// Rows is the files with something to do; files known to be current are
	// only counted.
	Rows         []modelUpdateRow
	CurrentCount int
	FetchCount   int
	FetchLabel   string
	VerifyCount  int
	VerifyLabel  string
	Stale        []string

	// Blocked is why the update cannot start now, shown instead of the button.
	Blocked string
}

type modelUpdateRow struct {
	Filename  string
	SizeLabel string
	State     string
	Note      string
}

// UpToDate reports that there is nothing to fetch, check or clear away.
func (v modelUpdateView) UpToDate() bool {
	return len(v.Rows) == 0 && len(v.Stale) == 0
}

func shortRevision(rev string) string {
	if len(rev) > 8 {
		return rev[:8]
	}
	return rev
}

func (s *Server) newModelUpdateView(modelID, displayName string, plan huggingface.Plan) modelUpdateView {
	v := modelUpdateView{
		ID:            modelID,
		SafeID:        safeID(modelID),
		DisplayName:   displayName,
		Revision:      plan.Revision,
		RevisionShort: shortRevision(plan.Revision),
		SyncedShort:   shortRevision(plan.LocalRevision),
		Stale:         plan.Stale,
		FetchLabel:    huggingface.FormatBytes(plan.FetchBytes()),
		Blocked:       s.transferBlocker(modelID, plan),
	}
	if u := hfModelURL(modelID); u != "" {
		v.CommitsURL = u + "/commits/main"
	}

	var verifyBytes int64
	for _, f := range plan.Files {
		row := modelUpdateRow{Filename: f.Filename, SizeLabel: huggingface.FormatBytes(f.Size)}
		switch f.State {
		case huggingface.FileCurrent:
			v.CurrentCount++
			continue
		case huggingface.FileNew:
			row.State = "new"
			v.FetchCount++
		case huggingface.FileChanged:
			row.State = "changed"
			v.FetchCount++
		case huggingface.FileUnverified:
			row.State = "to verify"
			row.Note = "same size as upstream; fetched only if its contents differ"
			v.VerifyCount++
			verifyBytes += f.Size
		}
		if f.State != huggingface.FileUnverified && f.PartBytes > 0 {
			row.Note = huggingface.FormatBytes(min(f.PartBytes, f.Size)) + " already fetched"
		}
		v.Rows = append(v.Rows, row)
	}
	v.VerifyLabel = huggingface.FormatBytes(verifyBytes)
	return v
}

// handleModelUpdateCheck compares a registered model with its repo as it
// stands now. It reads no file contents, so it answers at once even for a
// checkpoint that would take minutes to hash.
func (s *Server) handleModelUpdateCheck(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}

	revision, files, err := s.hfClient.GetFiles(r.Context(), id, "")
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "notice", fmt.Sprintf("Could not check Hugging Face for updates: %s", err))
			return
		}
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	plan := s.downloader.Plan(id, revision, files)

	if !isHTMX(r) {
		respondJSON(w, plan)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "model_update", s.newModelUpdateView(id, displayNameOf(m), plan))
}

// handleModelUpdate starts fetching what handleModelUpdateCheck reported.
func (s *Server) handleModelUpdate(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if _, ok := s.registry.Get(id); !ok {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	// FormValue, not the query alone: the checkbox arrives in the body.
	removeStale := r.FormValue("remove_stale") == "true"

	downloadID, err := s.startTransfer(r.Context(), id, r.FormValue("revision"), removeStale)
	if err != nil {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "notice", err.Error())
			return
		}
		http.Error(w, err.Error(), transferStatus(err))
		return
	}

	if !isHTMX(r) {
		respondJSON(w, map[string]string{"download_id": downloadID})
		return
	}
	// The downloads panel at the top of the page is where progress shows.
	w.Header().Set("HX-Trigger", "downloadsChanged")
	respondHTML(w)
	s.renderPartial(w, "plain_message", "Update started — progress is in the Downloads panel at the top of the page.")
}

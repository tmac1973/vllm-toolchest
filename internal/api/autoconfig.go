package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/tmac1973/vllm-toolchest/internal/autoconfig"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// contextClassOption is one context size the start panel offers.
type contextClassOption struct {
	Value, Label, Help string
	Checked            bool
}

func contextClassOptions(maxCtx int) []contextClassOption {
	maxLabel := "Maximum — the largest this model supports, as much as fits"
	if maxCtx > 0 {
		maxLabel = fmt.Sprintf("Maximum — up to %s tokens, as much as fits", groupThousands(maxCtx))
	}
	return []contextClassOption{
		{Value: "short", Label: "Short — about 8,000 tokens", Help: "A few pages of text. Uses the least memory, leaving the most for concurrent requests."},
		{Value: "medium", Label: "Medium — about 32,000 tokens", Help: "A long document or a long conversation. A good default.", Checked: true},
		{Value: "long", Label: "Long — about 128,000 tokens", Help: "A small codebase or a book chapter. Needs much more memory."},
		{Value: "max", Label: maxLabel, Help: "The model's own limit, reduced only as far as needed to fit."},
	}
}

// groupThousands writes 262144 as 262,144.
func groupThousands(n int) string {
	s := strconv.Itoa(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// autoconfigDialogView is the start panel.
type autoconfigDialogView struct {
	ModelID, SafeID, ModelName string
	Classes                    []contextClassOption
	// Busy, TooLarge, Serving and HelperUsable are the lines that say what
	// reading the card will cost, shown in that order.
	Busy          string
	TooLarge      string
	HelperMissing bool
	Serving       string
	HelperUsable  bool
	// HasReading offers "Read the card again".
	HasReading bool
}

func (s *Server) autoconfigDialog(m *models.Model) autoconfigDialogView {
	v := autoconfigDialogView{
		ModelID: m.ID, SafeID: safeID(m.ID), ModelName: displayNameOf(m),
		Classes: contextClassOptions(m.HFConfig.MaxPositionEmbeddings),
		Busy:    s.engineBusy(),
	}
	helper := s.helperModel()
	ok, why := models.HelperFits(s.gpuInventory(), s.helperUtil())
	switch {
	case helper == nil:
		v.HelperMissing = true
	case !ok:
		v.TooLarge = why
	default:
		v.HelperUsable = true
		if st := s.process.GetStatus(); st.State == process.StateRunning && st.ModelID != "" {
			v.Serving = s.displayName(st.ModelID)
		}
	}
	s.autoconf.mu.Lock()
	_, v.HasReading = s.autoconf.lastAdvice[m.ID]
	s.autoconf.mu.Unlock()
	if p, ok := s.registry.Profile(m.ID, models.AutoconfigProfileName); ok && p.Autoconfig != nil && len(p.Autoconfig.Advice) > 0 {
		v.HasReading = true
	}
	return v
}

type autoconfigProgressView struct {
	ModelID, SafeID, ModelName, Progress string
}

// autoconfigMessageView is a short outcome in the autoconfigure slot.
type autoconfigMessageView struct {
	ModelID, SafeID string
	Banner          panelBanner
}

// hwRow is one line of the hardware table.
type hwRow struct {
	Label, Field    string
	Current         string
	Proposed        string
	ProposedNarrow  string
	ProposedOffload string
	Why, WhyNarrow  string
	WhyOffload      string
	CardUsed        string
}

// hardwareTableView is the "This machine" table, shared with refinement.
type hardwareTableView struct {
	SafeID    string
	HasNarrow bool
	Rows      []hwRow
}

var hardwareFieldsShown = []struct{ field, label string }{
	{"tensor_parallel_size", "GPUs (tensor parallel)"},
	{"max_model_len", "Context length"},
	{"kv_cache_dtype", "KV cache dtype"},
	{"gpu_memory_utilization", "GPU memory utilization"},
	{"max_num_seqs", "Max concurrent sequences"},
}

func hardwareValue(c models.VLLMConfig, field string) string {
	switch field {
	case "tensor_parallel_size":
		return strconv.Itoa(max(1, c.TensorParallelSize))
	case "max_model_len":
		if c.MaxModelLen == 0 {
			return "model default"
		}
		return groupThousands(c.MaxModelLen)
	case "kv_cache_dtype":
		if c.KVCacheDtype == "" {
			return "auto"
		}
		return c.KVCacheDtype
	case "gpu_memory_utilization":
		if c.GPUMemoryUtilization == 0 {
			return fmt.Sprintf("%.2f", config.DefaultGPUMemoryUtil)
		}
		return fmt.Sprintf("%.2f", c.GPUMemoryUtilization)
	case "max_num_seqs":
		return strconv.Itoa(c.MaxNumSeqs)
	case "expert_offload":
		if process.HasFlag(c.ExtraFlags, models.ExpertOffloadFlag) {
			return "on"
		}
		return "off"
	case "kv_cache_memory":
		if c.KVCacheMemory == 0 {
			return "sized by the engine"
		}
		return fmt.Sprintf("%.1f GB", float64(c.KVCacheMemory)/(1<<30))
	}
	return ""
}

func notesFor(notes []models.ProfileNote, field string) string {
	out := ""
	for _, n := range notes {
		if n.Field == field {
			if out != "" {
				out += " "
			}
			out += n.Reason
		}
	}
	return out
}

// hardwareTable builds the table from the current config, the plan, and the
// notes about what the card used.
func hardwareTable(safe string, current models.VLLMConfig, plan models.FitPlan, cardNotes []models.ProfileNote) hardwareTableView {
	v := hardwareTableView{SafeID: safe, HasNarrow: plan.Narrow != nil}
	fields := hardwareFieldsShown
	if plan.All.Offload || plan.Offload != nil {
		fields = append(append([]struct{ field, label string }{}, fields...), struct{ field, label string }{"expert_offload", "Expert offload"})
	}
	var machineNotes []models.ProfileNote
	for _, n := range cardNotes {
		if n.Origin == "this machine" {
			machineNotes = append(machineNotes, n)
		}
	}
	widthWhy := func(w models.WidthPlan, field string) string {
		notes := append(append([]models.ProfileNote{}, w.Notes...), plan.Notes...)
		if field == "expert_offload" {
			return strings.TrimSpace(notesFor(notes, "extra_flags") + " " + notesFor(notes, "speculative_config"))
		}
		return notesFor(notes, field)
	}
	for _, f := range fields {
		r := hwRow{
			Label: f.label, Field: f.field,
			Current:  hardwareValue(current, f.field),
			Proposed: hardwareValue(plan.All.Config, f.field),
			Why:      widthWhy(plan.All, f.field),
			CardUsed: notesFor(machineNotes, f.field),
		}
		r.ProposedNarrow, r.WhyNarrow = r.Proposed, r.Why
		r.ProposedOffload, r.WhyOffload = r.Proposed, r.Why
		if plan.Narrow != nil {
			r.ProposedNarrow = hardwareValue(plan.Narrow.Config, f.field)
			r.WhyNarrow = widthWhy(*plan.Narrow, f.field)
		}
		if plan.Offload != nil {
			r.ProposedOffload = hardwareValue(plan.Offload.Config, f.field)
			r.WhyOffload = widthWhy(*plan.Offload, f.field)
		}
		v.Rows = append(v.Rows, r)
	}
	if current.KVCacheMemory > 0 {
		v.Rows = append(v.Rows, hwRow{
			Label: "KV cache memory (pinned)", Field: "kv_cache_memory",
			Current: hardwareValue(current, "kv_cache_memory"), Proposed: "sized by the engine",
			ProposedNarrow: "sized by the engine", ProposedOffload: "sized by the engine",
			Why:        "A pinned pool would override the engine's own sizing, which the context is planned against.",
			WhyNarrow:  "A pinned pool would override the engine's own sizing, which the context is planned against.",
			WhyOffload: "A pinned pool would override the engine's own sizing, which the context is planned against.",
		})
	}
	return v
}

// cardRowView is one line of the "From the model card" table.
type cardRowView struct {
	Key, Label, Current, Proposed, Reason, Quote, Warning string
	Ticked                                                bool
}

type widthOption struct {
	Value, Label string
	Checked      bool
}

// autoconfigReviewView is the review of a finished run.
type autoconfigReviewView struct {
	ModelID, SafeID, ModelName string
	Sources                    string
	FirstGuess                 bool
	PlanKnown                  bool
	PlanWhy                    string
	Widths                     []widthOption
	Hardware                   hardwareTableView
	CardRows                   []cardRowView
	FieldNotes                 []fieldNotes
	GeneralNotes               []string
	Summary                    []string
	// ImageWarning says the card needs an image this machine does not run.
	ImageWarning string
	// HelperAnswer is the helper's answer as it gave it, indented, for
	// seeing why a reading was or was not used; "" when it gave none.
	HelperAnswer string
	Suggestions  []autoconfig.DraftSuggestion
}

type fieldNotes struct {
	Field string
	Notes []string
}

func requestsLabel(n int) string {
	switch n {
	case 0:
		return "room for an unknown number of full-context requests"
	case 1:
		return "room for 1 full-context request"
	}
	return fmt.Sprintf("room for %d full-context requests", n)
}

func (s *Server) autoconfigReview(m *models.Model, res *autoconfig.Result) autoconfigReviewView {
	v := autoconfigReviewView{
		ModelID: m.ID, SafeID: safeID(m.ID), ModelName: displayNameOf(m),
		FirstGuess: res.Plan.Known && res.Plan.FirstGuess,
		PlanKnown:  res.Plan.Known, PlanWhy: res.Plan.Why,
		Hardware: hardwareTable(safeID(m.ID), res.Base, res.Plan, res.Notes),

		ImageWarning: res.ImageWarning,
	}

	switch {
	case len(res.CardSources) == 0:
		v.Sources = "No model card was read."
	default:
		v.Sources = "Read the model card of " + joinList(res.CardSources) + ". "
		switch res.AdviceFrom {
		case "helper":
			v.Sources += "The helper model read its text."
		case "previous":
			v.Sources += "The reading of its text from the last run was reused."
		default:
			v.Sources += "Its text was not read; its command was."
		}
	}

	if len(res.Advice) > 0 {
		var b bytes.Buffer
		if json.Indent(&b, res.Advice, "", "  ") == nil {
			v.HelperAnswer = b.String()
		}
	}

	if res.Plan.Known {
		all := res.Plan.All
		label := fmt.Sprintf("All %d cards — %s", all.TP, requestsLabel(all.FullContextRequests))
		if all.TP == 1 {
			label = "One card — " + requestsLabel(all.FullContextRequests)
		}
		if all.Offload {
			label = fmt.Sprintf("All %d cards, experts in system RAM — one request at a time, slower", all.TP)
		}
		v.Widths = append(v.Widths, widthOption{Value: "all", Checked: true, Label: label})
		if n := res.Plan.Narrow; n != nil {
			v.Widths = append(v.Widths, widthOption{Value: "narrow",
				Label: fmt.Sprintf("%d %s — %s, leaves %d free", n.TP, cardsWord(n.TP), requestsLabel(n.FullContextRequests), all.TP-n.TP)})
		}
		if o := res.Plan.Offload; o != nil {
			v.Widths = append(v.Widths, widthOption{Value: "offload",
				Label: fmt.Sprintf("All %d cards, experts in system RAM — %s tokens, one request at a time, slower", o.TP, groupThousands(o.ContextTokens))})
		}
	}

	for _, r := range res.Rows {
		reason := r.Reason
		if cur := r.Current(res.Base); cur != "" && cur == r.Proposed() {
			reason += " Unticked, it is removed."
		}
		v.CardRows = append(v.CardRows, cardRowView{
			Key: r.Key, Label: r.Label(), Current: r.Current(res.Base), Proposed: r.Proposed(),
			Reason: reason, Quote: r.Quote, Warning: r.Warning, Ticked: r.Ticked,
		})
	}

	v.Suggestions = res.Suggestions
	byField := map[string][]string{}
	var order []string
	for _, n := range res.Notes {
		switch {
		case n.Origin == "helper summary":
			v.Summary = append(v.Summary, n.Reason)
		case n.Origin == "this machine" && isHardwareField(n.Field) && res.Plan.Known:
			// Shown in the hardware table's Card column. With no plan there
			// is no table, and they are listed with the other notes.
		case n.Field == "":
			v.GeneralNotes = append(v.GeneralNotes, n.Reason)
		default:
			if _, seen := byField[n.Field]; !seen {
				order = append(order, n.Field)
			}
			byField[n.Field] = append(byField[n.Field], n.Reason)
		}
	}
	for _, f := range order {
		v.FieldNotes = append(v.FieldNotes, fieldNotes{Field: f, Notes: byField[f]})
	}
	return v
}

func isHardwareField(f string) bool {
	for _, h := range hardwareFieldsShown {
		if h.field == f {
			return true
		}
	}
	return f == "kv_cache_memory"
}

func cardsWord(n int) string {
	if n == 1 {
		return "card"
	}
	return "cards"
}

func joinList(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	}
	out := ""
	for i, it := range items {
		switch {
		case i == 0:
			out = it
		case i == len(items)-1:
			out += " and " + it
		default:
			out += ", " + it
		}
	}
	return out
}

// renderAutoconfigMessage renders a short outcome in the slot. Refusals are
// answered 200: htmx does not swap a non-2xx response, and a refusal nobody
// sees looks like a button that does nothing.
func (s *Server) renderAutoconfigMessage(w http.ResponseWriter, id string, b panelBanner) {
	respondHTML(w)
	s.renderPartial(w, "autoconfig_message", autoconfigMessageView{ModelID: id, SafeID: safeID(id), Banner: b})
}

// handleAutoconfigDialog shows the start panel, or the progress or review of
// a run already held for this model.
func (s *Server) handleAutoconfigDialog(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	m, ok := s.registry.Get(id)
	if !ok {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "That model is no longer in the registry."})
		return
	}
	if run, ok := s.autoconfigSnapshot(); ok && run.modelID == id {
		s.renderAutoconfigRun(w, m, run)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "autoconfig_dialog", s.autoconfigDialog(m))
}

// handleAutoconfigStart starts a run.
func (s *Server) handleAutoconfigStart(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if !s.parseForm(w, r) {
		return
	}
	class := models.ParseContextClass(r.FormValue("context_class"))
	if err := s.startAutoconfig(id, class, r.FormValue("reread") == "on"); err != nil {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "Not started: " + err.Error()})
		return
	}
	m, _ := s.registry.Get(id)
	run, _ := s.autoconfigSnapshot()
	s.renderAutoconfigRun(w, m, run)
}

// handleAutoconfigStatus is polled while a run is going; once it is done it
// returns the review, which does not poll.
func (s *Server) handleAutoconfigStatus(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	run, ok := s.autoconfigSnapshot()
	m, found := s.registry.Get(id)
	if !ok || run.modelID != id || !found {
		respondHTML(w)
		return
	}
	s.renderAutoconfigRun(w, m, run)
}

func (s *Server) renderAutoconfigRun(w http.ResponseWriter, m *models.Model, run autoconfigRun) {
	switch {
	case !run.done:
		respondHTML(w)
		s.renderPartial(w, "autoconfig_progress", autoconfigProgressView{
			ModelID: m.ID, SafeID: safeID(m.ID), ModelName: displayNameOf(m), Progress: run.progress,
		})
	case run.err != nil || run.result == nil:
		msg := "Autoconfigure did not finish."
		if run.err != nil {
			msg = "Autoconfigure did not finish: " + run.err.Error()
		}
		s.clearAutoconfigRun(m.ID)
		s.renderAutoconfigMessage(w, m.ID, panelBanner{Error: msg})
	default:
		respondHTML(w)
		s.renderPartial(w, "autoconfig_review", s.autoconfigReview(m, run.result))
	}
}

// handleAutoconfigSave saves the proposal as the Autoconfig profile, and with
// apply=1 also makes it the live config.
func (s *Server) handleAutoconfigSave(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if !s.parseForm(w, r) {
		return
	}

	run, ok := s.autoconfigSnapshot()
	if !ok || run.modelID != id || !run.done || run.result == nil {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "There is no finished Autoconfigure result to save. Run it again."})
		return
	}
	// Checked before anything is changed: a refusal reported after a
	// mutation leaves this process disagreeing with what is on disk.
	if reason := s.registry.ReadOnly(); reason != "" {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "Not saved: the registry is read-only (" + reason + ")."})
		return
	}

	ticked := map[string]bool{}
	for _, key := range r.Form["row"] {
		ticked[key] = true
	}
	for _, row := range run.result.Rows {
		if !ticked[row.Key] {
			ticked[row.Key] = false
		}
	}
	width := r.FormValue("width")
	res := run.result
	profile, plan, err := res.Profile(width, ticked, models.ProfileMeta{
		Variant: s.vllmEnv.Variant, VariantVersion: s.vllmEnv.VariantVersion,
	})
	if err != nil {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "Not saved: " + err.Error()})
		return
	}
	cfg := profile.Config
	d, known := s.vllmEnv.Descriptor()
	if err := validateNamedBackends(d, known, cfg.SpeculativeConfig, cfg.ExtraFlags); err != nil {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "Not saved: " + err.Error()})
		return
	}

	replaced, err := s.registry.SaveProfileFrom(id, models.AutoconfigProfileName, profile)
	if err != nil {
		s.renderAutoconfigMessage(w, id, panelBanner{Error: "Not saved: " + err.Error()})
		return
	}
	msg := "Saved as the Autoconfig profile"
	if replaced {
		msg += ", replacing the earlier one"
	}
	if r.FormValue("apply") == "1" {
		if _, err := s.registry.ApplyProfile(id, models.AutoconfigProfileName); err != nil {
			s.renderAutoconfigMessage(w, id, panelBanner{Error: msg + ", but it could not be applied: " + err.Error()})
			return
		}
		if m, ok := s.registry.Get(id); ok {
			m.VRAMEstimate = models.EstimateVRAM(m, s.configuredEnvPairs(m))
			s.registry.Register(m)
		}
		msg += " and now in use. It takes effect the next time this model starts."
	} else {
		msg += ". Restore it from Configure when you want to use it."
	}
	if w0 := autoconfig.ChosenWidth(res.Plan, width); w0 != nil {
		if w1 := autoconfig.ChosenWidth(plan, width); w1 != nil &&
			(w1.TP != w0.TP || w1.ContextTokens != w0.ContextTokens) {
			msg += fmt.Sprintf(" The fit was recalculated for the settings you kept: context %s, %d %s.",
				groupThousands(w1.ContextTokens), w1.TP, cardsWord(w1.TP))
		}
	}

	s.clearAutoconfigRun(id)
	w.Header().Set("HX-Trigger", "modelsChanged")
	s.renderAutoconfigMessage(w, id, panelBanner{OK: msg, Warning: envBlockWarning(cfg.Env)})
}

// handleAutoconfigDiscard drops a finished result.
func (s *Server) handleAutoconfigDiscard(w http.ResponseWriter, r *http.Request) {
	s.clearAutoconfigRun(r.URL.Query().Get("id"))
	respondHTML(w)
}

// handleAutoconfigDraft downloads a draft the review suggests. The repository
// must be one the held result suggests: a transfer is never started for an
// arbitrary repository named in a query string.
func (s *Server) handleAutoconfigDraft(w http.ResponseWriter, r *http.Request) {
	id, repo := r.URL.Query().Get("id"), r.URL.Query().Get("repo")
	respondHTML(w)
	run, ok := s.autoconfigSnapshot()
	suggested := false
	if ok && run.modelID == id && run.result != nil {
		for _, sg := range run.result.Suggestions {
			suggested = suggested || (sg.Repo == repo && !sg.Installed)
		}
	}
	if !suggested {
		s.renderPartial(w, "error_message", "That draft is not one this result suggests.")
		return
	}
	downloadID, err := s.startTransfer(r.Context(), repo, "", false)
	if err != nil {
		s.renderPartial(w, "error_message", "The download did not start: "+err.Error())
		return
	}
	w.Header().Set("HX-Trigger", "downloadsChanged")
	s.renderPartial(w, "download_started", downloadID)
}

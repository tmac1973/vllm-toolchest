package api

import (
	"github.com/tmac1973/vllm-toolchest/variants"
)

// The Settings page's feature-knob panel used to be 107 lines of hand-written
// HTML that named one image's twelve switches. Every control is now built
// here, from the running variant's manifest.
//
// The rule that keeps it simple: every dynamic lookup is resolved in Go, so
// the template only ranges and prints. No map indexing by a computed key, no
// arithmetic for the two-column rows, no comparing a stored value against an
// option. html/template resolves field names at execute time, not parse time,
// which means a lookup it cannot do renders a blank control in production and
// nothing catches it — so it does none.

// knobOptionView is one <option>. Selected is precomputed and Label already
// carries any annotation, so the template needs neither `eq` nor `index`.
type knobOptionView struct {
	Value    string
	Label    string
	Selected bool
}

// knobView is one control.
type knobView struct {
	Field       string // the form field name, e.g. "knob_use_r4d"
	Label       string
	Help        string
	IsSelect    bool
	Options     []knobOptionView
	Value       string // text knobs only
	Placeholder string // text knobs only
}

// knobGroupView is one visual group. Rows is pre-chunked two per row, which
// reproduces the panel's existing <div class="grid"> pairs without asking the
// template to count.
type knobGroupView struct {
	Title string
	Note  string
	Rows  [][]knobView
}

// knobSectionView is the whole panel, or nil when there is nothing to draw.
//
// That nil is what replaced `{{if .IsRadiance}}`: a variant with no manifest
// and a variant whose manifest declares no knobs both render nothing at all,
// rather than an empty box that looks like a bug.
type knobSectionView struct {
	Title    string
	Note     string
	Version  string
	DocURL   string
	DocLabel string
	Groups   []knobGroupView
}

// knobsPerRow matches the two-column layout the panel has always used.
const knobsPerRow = 2

// knobSection builds the panel for the running image.
func (s *Server) knobSection() *knobSectionView {
	d, ok := variants.Get(s.vllmEnv.Variant)
	if !ok || len(d.Knobs) == 0 {
		return nil
	}

	vals := s.cfg.KnobValues(d.ID)

	out := &knobSectionView{
		Title:    d.Label,
		Note:     d.Note,
		Version:  s.vllmEnv.RadianceVersion,
		DocURL:   d.DocURL,
		DocLabel: d.DocLabel,
	}

	for _, g := range d.Groups {
		gv := knobGroupView{Title: g.Title, Note: g.Note}
		var row []knobView
		for _, k := range g.Knobs {
			row = append(row, knobControl(k, vals[k.ID]))
			if len(row) == knobsPerRow {
				gv.Rows = append(gv.Rows, row)
				row = nil
			}
		}
		if len(row) > 0 {
			gv.Rows = append(gv.Rows, row)
		}
		out.Groups = append(out.Groups, gv)
	}
	return out
}

func knobControl(k variants.Knob, current string) knobView {
	v := knobView{
		Field:       "knob_" + k.ID,
		Label:       k.Label,
		Help:        k.Help,
		Placeholder: k.Placeholder,
	}

	if k.Kind != variants.KindSelect {
		v.Value = current
		return v
	}

	v.IsSelect = true
	for _, o := range k.Options {
		label := o.Label
		// Annotate the recommendation in the label rather than as separate
		// markup: it is how an operator discovers the manifest has an opinion
		// at all, before there is a button that applies them wholesale.
		if k.Recommended != "" && o.Value == k.Recommended {
			label += " — recommended"
		}
		v.Options = append(v.Options, knobOptionView{
			Value:    o.Value,
			Label:    label,
			Selected: o.Value == current,
		})
	}
	return v
}

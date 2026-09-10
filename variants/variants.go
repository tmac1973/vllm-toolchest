// Package variants reads the image-variant manifests in this directory.
//
// A manifest describes one image vllm-toolchest can be built on: where its
// base image comes from, what hardware it runs on, and which feature switches
// ("knobs") its Settings panel should offer. Each fact is declared exactly
// once, here, and every consumer generates from it — the Go config layer, the
// Settings renderer, the generated documentation, and setup.sh.
//
// The files live at the top of the repo rather than under internal/ on
// purpose: they are operator-facing artifacts that people read and hand-edit,
// and go:embed cannot reach outside its own package directory anyway.
//
// # The grammar
//
// A manifest is a strict subset of bash, so setup.sh can `source` it directly
// with no jq, yq or python on the host:
//
//	NAME='value'   one per line, value always single-quoted
//	# comment
//	<blank>
//
// Bash single-quoted strings are fully literal, which is what lets a tooltip
// carry commas, double quotes, em-dashes, # and $ without escaping. The one
// character that cannot appear is the ASCII apostrophe, and this parser
// rejects it rather than letting bash and Go disagree about where the value
// ends. This is not a shell parser: it accepts the restricted grammar bash
// also happens to accept, and rejects everything else.
package variants

import (
	"embed"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync"
)

//go:embed *.conf
var manifestFS embed.FS

// Unset is the token a manifest writes for "leave the image default alone".
// It is "-" rather than "" because a space-separated value list cannot carry
// an empty token, and because a recommendation of "leave it alone" has to be
// distinguishable from no recommendation having been written down.
const Unset = "-"

// Kind is the control a knob renders as.
type Kind string

const (
	KindSelect Kind = "select"
	KindText   Kind = "text"
)

// Option is one entry of a select knob. Value is "" for the unset option: the
// manifest's "-" is decoded here, once, so nothing downstream sees the
// sentinel.
type Option struct {
	Value string
	Label string
}

// Knob is one feature switch.
type Knob struct {
	// ID is the lowercase form of the manifest slug. It is the yaml key
	// under knobs.<variant> and the suffix of the settings form field.
	ID string

	// Env is the environment variable the image reads. It is deliberately
	// not derived from a prefix: a variant may own a variable named
	// anything, and the Intel images in particular share nothing with the
	// RADIANCE_* naming.
	Env string

	Label string
	Help  string
	Kind  Kind

	// Options is nil for KindText.
	Options []Option

	// Placeholder is KindText only.
	Placeholder string

	// Recommended is what an "apply recommended settings" action would set.
	// Empty means the recommendation is the image's own default — which is
	// a different statement from nobody having written a recommendation.
	Recommended string

	// Group is the manifest group slug this knob belongs to.
	Group string
}

// Group is a contiguous run of knobs sharing a heading. An empty Title renders
// a separator with no heading, which is how the existing radiance panel splits
// kernels from drafting from host settings.
type Group struct {
	ID    string
	Title string
	Note  string
	Knobs []Knob
}

// Descriptor is one variant's whole manifest.
type Descriptor struct {
	ID       string
	Label    string
	Summary  string
	Vendor   string
	Tier     string
	DocURL   string
	DocLabel string
	Note     string

	BaseImage    string
	Dockerfile   string
	NeedsFlatten bool
	VLLMPin      string

	GFXTargets []string
	HostArch   string
	Priority   int

	VenvRoot          string
	Launcher          []string
	StampFile         string
	Caps              []string
	AttentionBackends []string

	HostReqs []HostReq

	// Knobs is in declaration order, which is the single authoritative
	// ordering for the UI, for env emission and for generated docs.
	Knobs []Knob

	// Groups is Knobs partitioned into contiguous runs.
	Groups []Group
}

// HostReq is one pre-build check setup.sh runs against the host. Severity is
// carried per check, in the manifest, so a false block on hardware nobody here
// owns is a one-line edit rather than a code change and a release.
type HostReq struct {
	Kind     string // gfx, arch, rocm_min, nvidia_driver_min, compute_cap, intel_firmware
	Severity string // block | warn
	Value    string
	Message  string
}

// Knob returns the knob with the given id.
func (d Descriptor) Knob(id string) (Knob, bool) {
	for _, k := range d.Knobs {
		if k.ID == id {
			return k, true
		}
	}
	return Knob{}, false
}

// EnvNames returns every environment variable this variant's knobs own, in
// manifest order. The compose pass-through lists are checked against this.
func (d Descriptor) EnvNames() []string {
	out := make([]string, 0, len(d.Knobs))
	for _, k := range d.Knobs {
		out = append(out, k.Env)
	}
	return out
}

// Recommended returns knob id -> value for every knob whose recommendation is
// something other than the image default. This is the whole API an "apply
// recommended settings" action needs.
func (d Descriptor) Recommended() map[string]string {
	out := map[string]string{}
	for _, k := range d.Knobs {
		if k.Recommended != "" {
			out[k.ID] = k.Recommended
		}
	}
	return out
}

var (
	once     sync.Once
	loaded   []Descriptor
	loadErr  error
	byID     map[string]Descriptor
	envOwner map[string]string
)

// All returns every manifest, sorted by id. It panics on a malformed embedded
// manifest: a manifest that does not parse is a build-time authoring mistake,
// and TestAllManifestsAreValid exists to catch it in CI with a better message
// than a nil dereference at runtime.
func All() []Descriptor {
	load()
	if loadErr != nil {
		panic("variants: " + loadErr.Error())
	}
	return loaded
}

// Get returns the descriptor for a variant id. An unknown id — an image built
// before a manifest existed, or an operator override naming something we do
// not ship — resolves to nothing rather than an error, because every consumer
// treats "no manifest" as "no knobs to offer".
func Get(id string) (Descriptor, bool) {
	load()
	if loadErr != nil {
		return Descriptor{}, false
	}
	d, ok := byID[id]
	return d, ok
}

// EnvNameSet maps every environment variable declared as a knob, across every
// variant, to "<variant id>.<knob id>". This replaces prefix matching: the old
// RADIANCE_ prefix rule wrongly claimed ownership of RADIANCE_IMAGE and
// RADIANCE_VERSION, which are compose build args and not knobs at all.
func EnvNameSet() map[string]string {
	load()
	if loadErr != nil {
		return nil
	}
	return envOwner
}

// Err reports a manifest problem without panicking, for the validation test.
func Err() error {
	load()
	return loadErr
}

func load() {
	once.Do(func() {
		entries, err := manifestFS.ReadDir(".")
		if err != nil {
			loadErr = err
			return
		}
		byID = map[string]Descriptor{}
		envOwner = map[string]string{}

		for _, e := range entries {
			name := e.Name()
			if e.IsDir() || !strings.HasSuffix(name, ".conf") {
				continue
			}
			b, err := manifestFS.ReadFile(name)
			if err != nil {
				loadErr = err
				return
			}
			id := strings.TrimSuffix(path.Base(name), ".conf")
			d, err := parse(id, string(b))
			if err != nil {
				loadErr = fmt.Errorf("%s: %w", name, err)
				return
			}
			if err := d.Validate(); err != nil {
				loadErr = fmt.Errorf("%s: %w", name, err)
				return
			}
			for _, k := range d.Knobs {
				if prior, dup := envOwner[k.Env]; dup {
					loadErr = fmt.Errorf("%s: %s is already declared by %s; "+
						"env names must be unique across variants so config "+
						"loading can attribute one without knowing the running image",
						name, k.Env, prior)
					return
				}
				envOwner[k.Env] = d.ID + "." + k.ID
			}
			byID[d.ID] = d
			loaded = append(loaded, d)
		}
		sort.Slice(loaded, func(i, j int) bool { return loaded[i].ID < loaded[j].ID })
	})
}

// assignRE matches the one statement form a manifest may contain.
var assignRE = regexp.MustCompile(`^([A-Za-z_][A-Za-z0-9_]*)='([^']*)'$`)

// parseFile turns a manifest into its raw key/value pairs.
func parseFile(src string) (map[string]string, error) {
	out := map[string]string{}
	for i, line := range strings.Split(src, "\n") {
		t := strings.TrimSpace(line)
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		m := assignRE.FindStringSubmatch(t)
		if m == nil {
			// Name the likely cause rather than just the line: every way
			// this fails is a quoting mistake, and saying which one saves
			// an author from staring at a line that looks fine.
			switch {
			case strings.Contains(t, `="`):
				return nil, fmt.Errorf("line %d: values must be single-quoted, not double-quoted: %s", i+1, t)
			case strings.Count(t, "'") > 2:
				return nil, fmt.Errorf("line %d: values may not contain an ASCII apostrophe (use double quotes in prose): %s", i+1, t)
			default:
				return nil, fmt.Errorf("line %d: expected NAME='value', got: %s", i+1, t)
			}
		}
		if prior, dup := out[m[1]]; dup && prior != m[2] {
			return nil, fmt.Errorf("line %d: %s is assigned twice", i+1, m[1])
		}
		out[m[1]] = m[2]
	}
	return out, nil
}

// knobAttrs are the attribute suffixes a KNOB_ key may end with. The list is
// what makes slug ambiguity detectable: KNOB_USE_R4D_AR_QUANT_ENV is only
// unambiguous because no attribute is named QUANT.
var knobAttrs = []string{
	"ENV", "GROUP", "TYPE", "LABEL", "HELP", "VALUES", "PLACEHOLDER", "RECOMMENDED",
}

func parse(id string, src string) (Descriptor, error) {
	kv, err := parseFile(src)
	if err != nil {
		return Descriptor{}, err
	}

	d := Descriptor{
		ID:                kv["VARIANT_ID"],
		Label:             kv["VARIANT_LABEL"],
		Summary:           kv["VARIANT_SUMMARY"],
		Vendor:            kv["VARIANT_VENDOR"],
		Tier:              kv["VARIANT_TIER"],
		DocURL:            kv["VARIANT_DOC_URL"],
		DocLabel:          kv["VARIANT_DOC_LABEL"],
		Note:              kv["VARIANT_NOTE"],
		BaseImage:         kv["VARIANT_BASE_IMAGE"],
		Dockerfile:        kv["VARIANT_DOCKERFILE"],
		NeedsFlatten:      kv["VARIANT_NEEDS_FLATTEN"] == "1",
		VLLMPin:           kv["VARIANT_VLLM_PIN"],
		GFXTargets:        strings.Fields(kv["VARIANT_GFX_TARGETS"]),
		HostArch:          kv["VARIANT_HOST_ARCH"],
		VenvRoot:          kv["VARIANT_VENV_ROOT"],
		Launcher:          strings.Fields(kv["VARIANT_LAUNCHER"]),
		StampFile:         kv["VARIANT_STAMP_FILE"],
		Caps:              strings.Fields(kv["VARIANT_CAPS"]),
		AttentionBackends: strings.Fields(kv["VARIANT_ATTENTION_BACKENDS"]),
	}
	if d.ID == "" {
		d.ID = id
	}
	if p := kv["VARIANT_PRIORITY"]; p != "" {
		if _, err := fmt.Sscanf(p, "%d", &d.Priority); err != nil {
			return Descriptor{}, fmt.Errorf("VARIANT_PRIORITY: %q is not a number", p)
		}
	}

	// HOSTREQ_1, HOSTREQ_2, ... are read in order until one is missing, so a
	// gap truncates the list rather than silently skipping a check.
	for i := 1; ; i++ {
		raw, ok := kv[fmt.Sprintf("HOSTREQ_%d", i)]
		if !ok {
			break
		}
		parts := strings.SplitN(raw, "|", 4)
		if len(parts) != 4 {
			return Descriptor{}, fmt.Errorf("HOSTREQ_%d: expected kind|severity|value|message, got %q", i, raw)
		}
		if parts[1] != "block" && parts[1] != "warn" {
			return Descriptor{}, fmt.Errorf("HOSTREQ_%d: severity must be block or warn, got %q", i, parts[1])
		}
		d.HostReqs = append(d.HostReqs, HostReq{
			Kind: parts[0], Severity: parts[1], Value: parts[2], Message: parts[3],
		})
	}

	for _, slug := range strings.Fields(kv["KNOBS"]) {
		k, err := parseKnob(kv, slug)
		if err != nil {
			return Descriptor{}, fmt.Errorf("knob %s: %w", slug, err)
		}
		d.Knobs = append(d.Knobs, k)
	}
	d.Groups = groupKnobs(kv, d.Knobs)

	return d, nil
}

func parseKnob(kv map[string]string, slug string) (Knob, error) {
	at := func(attr string) string { return kv["KNOB_"+slug+"_"+attr] }

	k := Knob{
		ID:          strings.ToLower(slug),
		Env:         at("ENV"),
		Label:       at("LABEL"),
		Help:        at("HELP"),
		Kind:        Kind(at("TYPE")),
		Placeholder: at("PLACEHOLDER"),
		Group:       at("GROUP"),
	}
	if rec := at("RECOMMENDED"); rec != Unset {
		k.Recommended = rec
	}

	switch k.Kind {
	case KindSelect:
		for _, tok := range strings.Fields(at("VALUES")) {
			k.Options = append(k.Options, Option{
				Value: optionValue(tok),
				Label: optionLabel(at, tok),
			})
		}
	case KindText:
		// Caught here rather than in Validate: the options are discarded
		// when building a text knob, so by then the mistake is invisible.
		if at("VALUES") != "" {
			return Knob{}, fmt.Errorf("a text knob may not declare VALUES")
		}
	case "":
		return Knob{}, fmt.Errorf("missing KNOB_%s_TYPE", slug)
	default:
		return Knob{}, fmt.Errorf("unknown type %q (want select or text)", k.Kind)
	}
	return k, nil
}

func optionValue(tok string) string {
	if tok == Unset {
		return ""
	}
	return tok
}

// optionLabel resolves a per-option label. An explicit
// KNOB_<SLUG>_LABEL_<TOKEN> wins; otherwise the common tokens get their
// conventional wording so a plain on/off knob needs three lines, not six.
func optionLabel(at func(string) string, tok string) string {
	key := strings.ToUpper(tok)
	if tok == Unset {
		key = "UNSET"
	}
	if l := at("LABEL_" + key); l != "" {
		return l
	}
	switch tok {
	case Unset:
		return "image default"
	case "1":
		return "on"
	case "0":
		return "off"
	default:
		return tok
	}
}

// groupKnobs partitions knobs into contiguous runs sharing a group slug.
// Contiguity is enforced by Validate, so this cannot silently merge two
// separated runs of the same group into one.
func groupKnobs(kv map[string]string, knobs []Knob) []Group {
	var out []Group
	for _, k := range knobs {
		if n := len(out); n > 0 && out[n-1].ID == k.Group {
			out[n-1].Knobs = append(out[n-1].Knobs, k)
			continue
		}
		out = append(out, Group{
			ID:    k.Group,
			Title: kv["GROUP_"+k.Group+"_TITLE"],
			Note:  kv["GROUP_"+k.Group+"_NOTE"],
			Knobs: []Knob{k},
		})
	}
	return out
}

// Validate enforces the manifest invariants that a typo would otherwise turn
// into a runtime surprise on hardware nobody here can test.
func (d Descriptor) Validate() error {
	if d.ID == "" {
		return fmt.Errorf("VARIANT_ID is required")
	}
	if d.Tier != "" && d.Tier != "tested" && d.Tier != "community" && d.Tier != "experimental" {
		return fmt.Errorf("VARIANT_TIER: %q is not tested, community or experimental", d.Tier)
	}

	seenEnv := map[string]bool{}
	seenID := map[string]bool{}
	slugs := map[string]bool{}
	for _, k := range d.Knobs {
		slugs[strings.ToUpper(k.ID)] = true
	}

	for _, k := range d.Knobs {
		if k.Env == "" {
			return fmt.Errorf("knob %s: KNOB_%s_ENV is required", k.ID, strings.ToUpper(k.ID))
		}
		if seenEnv[k.Env] {
			return fmt.Errorf("knob %s: %s is declared twice in this variant", k.ID, k.Env)
		}
		seenEnv[k.Env] = true
		if seenID[k.ID] {
			return fmt.Errorf("knob %s is listed twice in KNOBS", k.ID)
		}
		seenID[k.ID] = true
		if k.Label == "" {
			return fmt.Errorf("knob %s: a label is required, it is the control's only visible name", k.ID)
		}

		// A slug that is another slug plus an attribute name makes
		// KNOB_<slug>_<attr> ambiguous for both bash and this parser.
		up := strings.ToUpper(k.ID)
		for _, attr := range knobAttrs {
			if other, cut := strings.CutSuffix(up, "_"+attr); cut && slugs[other] {
				return fmt.Errorf("knob %s collides with knob %s: KNOB_%s_* is ambiguous with KNOB_%s_%s",
					k.ID, strings.ToLower(other), up, other, attr)
			}
		}

		if k.Kind == KindSelect {
			if len(k.Options) == 0 {
				return fmt.Errorf("knob %s: a select needs KNOB_%s_VALUES", k.ID, up)
			}
			// The unset option must come first so "leave the image
			// default alone" is what a fresh install shows.
			if k.Options[0].Value != "" {
				return fmt.Errorf("knob %s: the first value must be %q (the unset sentinel), got %q",
					k.ID, Unset, k.Options[0].Value)
			}
			valid := map[string]bool{}
			for _, o := range k.Options[1:] {
				if o.Value == "" {
					return fmt.Errorf("knob %s: %q may only appear once, as the first value", k.ID, Unset)
				}
				valid[o.Value] = true
			}
			if k.Recommended != "" && !valid[k.Recommended] {
				return fmt.Errorf("knob %s: recommended value %q is not one of its options", k.ID, k.Recommended)
			}
		}
	}

	// Contiguity: a group may not appear, stop, and reappear, or the panel
	// would render the same heading twice.
	seenGroup := map[string]bool{}
	for i, g := range d.Groups {
		if seenGroup[g.ID] {
			return fmt.Errorf("group %s is split: its knobs must be contiguous in KNOBS", g.ID)
		}
		seenGroup[g.ID] = true
		_ = i
	}
	return nil
}

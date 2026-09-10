package variants

import (
	"strings"
	"testing"
)

// TestAllManifestsAreValid is the replacement for the compile-time safety a
// code-generation step would have given. Every manifest ships to hardware
// nobody here can test, so a typo has to fail in CI rather than at install.
func TestAllManifestsAreValid(t *testing.T) {
	if err := Err(); err != nil {
		t.Fatalf("embedded manifests do not load: %v", err)
	}
	all := All()
	if len(all) == 0 {
		t.Fatal("no manifests were embedded; the go:embed pattern is wrong")
	}
	for _, d := range all {
		if d.Label == "" {
			t.Errorf("%s: VARIANT_LABEL is required, it is the menu entry", d.ID)
		}
		if d.Summary == "" {
			t.Errorf("%s: VARIANT_SUMMARY is required, it is the menu's one-line explanation", d.ID)
		}
	}
}

func TestGetUnknownVariant(t *testing.T) {
	if _, ok := Get("nonesuch"); ok {
		t.Error("an undeclared variant must not resolve to a descriptor")
	}
}

// A variant with no knobs is a first-class case, not a gap: the Settings panel
// must render nothing rather than an empty box.
func TestGenericDeclaresNoKnobs(t *testing.T) {
	d, ok := Get("generic")
	if !ok {
		t.Fatal("generic manifest missing")
	}
	if len(d.Knobs) != 0 || len(d.Groups) != 0 {
		t.Errorf("generic should declare no knobs, got %d knobs in %d groups",
			len(d.Knobs), len(d.Groups))
	}
	if len(d.Recommended()) != 0 {
		t.Error("a variant with no knobs has nothing to recommend")
	}
}

func TestRadianceGroupsAndOptions(t *testing.T) {
	d, ok := Get("radiance")
	if !ok {
		t.Fatal("radiance manifest missing")
	}

	// Group shape mirrors the panel the hand-written template drew: six
	// kernel switches, four drafting ones, two host ones.
	want := []struct {
		id    string
		count int
	}{{"KERNELS", 6}, {"DRAFTING", 4}, {"HOST", 2}}
	if len(d.Groups) != len(want) {
		t.Fatalf("groups = %d, want %d", len(d.Groups), len(want))
	}
	for i, w := range want {
		if d.Groups[i].ID != w.id || len(d.Groups[i].Knobs) != w.count {
			t.Errorf("group %d = %s/%d, want %s/%d",
				i, d.Groups[i].ID, len(d.Groups[i].Knobs), w.id, w.count)
		}
	}
	if d.Groups[1].Title == "" {
		t.Error("the drafting group needs its heading; only the unnamed groups render as a bare separator")
	}

	// skinny_gemm is the one four-state knob, and the reason the option list
	// is data rather than an on/off bool.
	k, ok := d.Knob("skinny_gemm")
	if !ok {
		t.Fatal("skinny_gemm missing")
	}
	if len(k.Options) != 4 {
		t.Fatalf("skinny_gemm options = %d, want 4", len(k.Options))
	}
	if k.Options[0].Value != "" || k.Options[0].Label != "image default" {
		t.Errorf("first option = %+v, want the unset sentinel labelled %q", k.Options[0], "image default")
	}
	if k.Options[2].Value != "all" || k.Options[2].Label != "all shapes" {
		t.Errorf("third option = %+v, want all/%q", k.Options[2], "all shapes")
	}

	// Default option wording, so an on/off knob needs no label lines.
	r4d, _ := d.Knob("use_r4d")
	if r4d.Options[1].Label != "on" || r4d.Options[2].Label != "off" {
		t.Errorf("use_r4d options = %q/%q, want on/off",
			r4d.Options[1].Label, r4d.Options[2].Label)
	}

	// A text knob keeps its placeholder and declares no options.
	tau, _ := d.Knob("draft_tau")
	if tau.Kind != KindText || len(tau.Options) != 0 || tau.Placeholder == "" {
		t.Errorf("draft_tau = %+v, want a text knob with a placeholder and no options", tau)
	}
}

// "-" means the recommendation is the image's own default, which must not be
// confused with an empty recommendation slot that would then be applied.
func TestUnsetRecommendationIsNotAValue(t *testing.T) {
	d, _ := Get("radiance")
	for _, k := range d.Knobs {
		if k.Recommended == Unset {
			t.Errorf("knob %s: the %q sentinel leaked into Recommended", k.ID, Unset)
		}
	}
	for id, v := range d.Recommended() {
		if v == "" {
			t.Errorf("knob %s: Recommended() must omit knobs with no recommendation, not map them to empty", id)
		}
	}
}

func TestEnvNamesAreOwnedUniquely(t *testing.T) {
	owner := EnvNameSet()
	if owner["RADIANCE_USE_R4D"] != "radiance.use_r4d" {
		t.Errorf("RADIANCE_USE_R4D owner = %q, want radiance.use_r4d", owner["RADIANCE_USE_R4D"])
	}
	// The old prefix rule claimed these, and was wrong: they are compose
	// build args, not knobs.
	for _, n := range []string{"RADIANCE_IMAGE", "RADIANCE_VERSION"} {
		if _, ok := owner[n]; ok {
			t.Errorf("%s is not a knob and must not be claimed as one", n)
		}
	}
}

func TestParseRejectsBadQuoting(t *testing.T) {
	for _, tc := range []struct {
		name, src, want string
	}{
		{"double quotes", "VARIANT_ID=\"x\"", "single-quoted"},
		{"unquoted", "VARIANT_ID=x", "expected NAME="},
		{"apostrophe in value", "VARIANT_NOTE='the image's default'", "apostrophe"},
		{"trailing text", "VARIANT_ID='x' # trailing", "expected NAME="},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseFile(tc.src)
			if err == nil {
				t.Fatalf("parsed %q, want an error", tc.src)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestParseAcceptsCommentsAndBlanks(t *testing.T) {
	kv, err := parseFile("# a comment\n\nVARIANT_ID='x'\n   \n# another\nVARIANT_LABEL='L'\n")
	if err != nil {
		t.Fatalf("parseFile: %v", err)
	}
	if kv["VARIANT_ID"] != "x" || kv["VARIANT_LABEL"] != "L" {
		t.Errorf("kv = %v", kv)
	}
}

// The characters a tooltip actually needs. Single quoting is what buys these,
// and it is why the existing help text's "--" workarounds can go away.
func TestParseKeepsProsePunctuationLiteral(t *testing.T) {
	const want = `Routes small-M projections. "all" differs at a ULP — notably in_proj_ba, $x, #1 (really).`
	kv, err := parseFile("KNOB_X_HELP='" + want + "'")
	if err != nil {
		t.Fatalf("parseFile: %v", err)
	}
	if got := kv["KNOB_X_HELP"]; got != want {
		t.Errorf("help = %q, want %q", got, want)
	}
}

func TestValidateRejects(t *testing.T) {
	base := "VARIANT_ID='t'\nVARIANT_LABEL='T'\n"
	for _, tc := range []struct {
		name, src, want string
	}{
		{
			"select without the unset sentinel first",
			base + "KNOBS='A'\nKNOB_A_ENV='E'\nKNOB_A_TYPE='select'\nKNOB_A_LABEL='A'\nKNOB_A_VALUES='1 0'\n",
			"unset sentinel",
		},
		{
			"recommended value outside the option list",
			base + "KNOBS='A'\nKNOB_A_ENV='E'\nKNOB_A_TYPE='select'\nKNOB_A_LABEL='A'\nKNOB_A_VALUES='- 1 0'\nKNOB_A_RECOMMENDED='2'\n",
			"not one of its options",
		},
		{
			"text knob with values",
			base + "KNOBS='A'\nKNOB_A_ENV='E'\nKNOB_A_TYPE='text'\nKNOB_A_LABEL='A'\nKNOB_A_VALUES='- 1'\n",
			"may not declare VALUES",
		},
		{
			"duplicate env within a variant",
			base + "KNOBS='A B'\nKNOB_A_ENV='E'\nKNOB_A_TYPE='text'\nKNOB_A_LABEL='A'\nKNOB_B_ENV='E'\nKNOB_B_TYPE='text'\nKNOB_B_LABEL='B'\n",
			"declared twice",
		},
		{
			"split group",
			base + "KNOBS='A B C'\n" +
				"KNOB_A_ENV='E1'\nKNOB_A_TYPE='text'\nKNOB_A_LABEL='A'\nKNOB_A_GROUP='G'\n" +
				"KNOB_B_ENV='E2'\nKNOB_B_TYPE='text'\nKNOB_B_LABEL='B'\nKNOB_B_GROUP='H'\n" +
				"KNOB_C_ENV='E3'\nKNOB_C_TYPE='text'\nKNOB_C_LABEL='C'\nKNOB_C_GROUP='G'\n",
			"must be contiguous",
		},
		{
			"ambiguous slug",
			base + "KNOBS='A A_HELP'\n" +
				"KNOB_A_ENV='E1'\nKNOB_A_TYPE='text'\nKNOB_A_LABEL='A'\n" +
				"KNOB_A_HELP_ENV='E2'\nKNOB_A_HELP_TYPE='text'\nKNOB_A_HELP_LABEL='B'\n",
			"ambiguous",
		},
		{
			"missing label",
			base + "KNOBS='A'\nKNOB_A_ENV='E'\nKNOB_A_TYPE='text'\n",
			"label is required",
		},
		{
			"bad tier",
			"VARIANT_ID='t'\nVARIANT_LABEL='T'\nVARIANT_TIER='probably fine'\nKNOBS=''\n",
			"not tested, community or experimental",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, err := parse("t", tc.src)
			if err == nil {
				err = d.Validate()
			}
			if err == nil {
				t.Fatal("accepted an invalid manifest")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestHostReqParsing(t *testing.T) {
	d, ok := Get("radiance")
	if !ok {
		t.Fatal("radiance manifest missing")
	}
	if len(d.HostReqs) != 1 {
		t.Fatalf("host reqs = %d, want 1", len(d.HostReqs))
	}
	got := d.HostReqs[0]
	if got.Kind != "gfx" || got.Severity != "block" || got.Value != "gfx1201" {
		t.Errorf("host req = %+v, want the gfx1201 hard block that exists today", got)
	}

	if _, err := parse("t", "VARIANT_ID='t'\nVARIANT_LABEL='T'\nKNOBS=''\nHOSTREQ_1='gfx|maybe|x|m'\n"); err == nil {
		t.Error("accepted a severity that is neither block nor warn")
	}
	if _, err := parse("t", "VARIANT_ID='t'\nVARIANT_LABEL='T'\nKNOBS=''\nHOSTREQ_1='gfx|block|x'\n"); err == nil {
		t.Error("accepted a host req with too few fields")
	}
}

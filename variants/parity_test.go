package variants_test

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/variants"
)

// The radiance manifest has to describe exactly what config.RadianceConfig
// describes today — same knob ids, same environment variables — because that
// equivalence is what makes the config migration a reparent rather than a
// rename. If these ever disagree, an operator's saved switches would move to
// keys nothing reads, silently, and the first symptom would be a model served
// with the stock kernels.
//
// This test exists for the window in which both representations are live. It
// should be deleted along with RadianceConfig once the knob work lands.

func TestManifestMatchesRadianceConfigYAMLTags(t *testing.T) {
	d, ok := variants.Get("radiance")
	if !ok {
		t.Fatal("radiance manifest missing")
	}

	var want []string
	rt := reflect.TypeOf(config.RadianceConfig{})
	for i := range rt.NumField() {
		tag := rt.Field(i).Tag.Get("yaml")
		if tag == "" || tag == "-" {
			t.Fatalf("%s has no yaml tag; the knob id is derived from it", rt.Field(i).Name)
		}
		want = append(want, strings.SplitN(tag, ",", 2)[0])
	}

	var got []string
	for _, k := range d.Knobs {
		got = append(got, k.ID)
	}

	sort.Strings(want)
	sorted := append([]string(nil), got...)
	sort.Strings(sorted)
	if !reflect.DeepEqual(sorted, want) {
		t.Errorf("knob ids do not match RadianceConfig yaml tags\n manifest: %v\n   config: %v", sorted, want)
	}
}

func TestManifestMatchesRadianceConfigEnvNames(t *testing.T) {
	d, ok := variants.Get("radiance")
	if !ok {
		t.Fatal("radiance manifest missing")
	}

	// Env() skips empty fields, so fill every one to get the whole table.
	var rc config.RadianceConfig
	rv := reflect.ValueOf(&rc).Elem()
	for i := range rv.NumField() {
		rv.Field(i).SetString("x")
	}

	var want []string
	for _, pair := range rc.Env() {
		want = append(want, strings.SplitN(pair, "=", 2)[0])
	}

	got := d.EnvNames()
	if len(got) != len(want) {
		t.Fatalf("env names: manifest has %d, RadianceConfig.Env has %d", len(got), len(want))
	}

	sortedGot := append([]string(nil), got...)
	sortedWant := append([]string(nil), want...)
	sort.Strings(sortedGot)
	sort.Strings(sortedWant)
	if !reflect.DeepEqual(sortedGot, sortedWant) {
		t.Errorf("env names do not match\n manifest: %v\n   config: %v", sortedGot, sortedWant)
	}
}

// Ordering is allowed to differ — nothing depends on the order the switches
// are appended in, since no name appears twice — but a reader comparing the
// two should be told rather than left to wonder.
func TestManifestEnvOrderIsDocumentedWhereItDiffers(t *testing.T) {
	d, _ := variants.Get("radiance")

	var rc config.RadianceConfig
	rv := reflect.ValueOf(&rc).Elem()
	for i := range rv.NumField() {
		rv.Field(i).SetString("x")
	}
	var legacy []string
	for _, pair := range rc.Env() {
		legacy = append(legacy, strings.SplitN(pair, "=", 2)[0])
	}

	if reflect.DeepEqual(d.EnvNames(), legacy) {
		return
	}
	t.Logf("manifest emits knobs in a different order to RadianceConfig.Env, which is fine:"+
		"\n manifest: %v\n   config: %v", d.EnvNames(), legacy)
}

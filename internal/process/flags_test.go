package process

import (
	"reflect"
	"testing"
)

func TestJoinFlagsRoundTrips(t *testing.T) {
	for _, args := range [][]string{
		{"--override-generation-config", `{"max_tokens": 65536, "temperature": 0.6}`},
		{"--x", "it's here", "--y", `a "quoted" \ thing`},
		{"--enable-expert-offload", "--expert-offload-mem", "46"},
		{"--empty", ""},
	} {
		if got := SplitFlags(JoinFlags(args)); !reflect.DeepEqual(got, args) {
			t.Errorf("round trip of %q gave %q (joined %s)", args, got, JoinFlags(args))
		}
	}
}

func TestSetAndRemoveFlag(t *testing.T) {
	extra := `--keep-me --override-generation-config '{"temperature": 1}' --expert-offload-mem=46`

	got := SetFlag(extra, []string{"--override-generation-config", `{"temperature": 0.6}`})
	want := []string{"--keep-me", "--override-generation-config", `{"temperature": 0.6}`, "--expert-offload-mem=46"}
	if !reflect.DeepEqual(SplitFlags(got), want) {
		t.Errorf("replace: %q", SplitFlags(got))
	}

	got = SetFlag(extra, []string{"--new", "1"})
	if a := SplitFlags(got); len(a) != 6 || a[4] != "--new" || a[5] != "1" {
		t.Errorf("append: %q", a)
	}

	got = RemoveFlag(extra, "--expert-offload-mem")
	if a := SplitFlags(got); len(a) != 3 || HasFlag(got, "--expert-offload-mem") {
		t.Errorf("remove inline: %q", a)
	}
	got = RemoveFlag(extra, "--override-generation-config")
	if a := SplitFlags(got); !reflect.DeepEqual(a, []string{"--keep-me", "--expert-offload-mem=46"}) {
		t.Errorf("remove with value: %q", a)
	}
	if !HasFlag(extra, "--keep-me") || HasFlag(extra, "--absent") {
		t.Error("HasFlag")
	}
}

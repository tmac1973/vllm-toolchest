package monitor

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// The GPU index used to come from rocm-smi's "device" column, which on the
// R9700 box reported card0 for every row: all four GPUs landed at index 0 and
// each then read GPU 0's VRAM out of sysfs. The PCI bus address is the key that
// actually identifies a card, so this is the mapping that has to hold.
func TestKFDIndexByBDF(t *testing.T) {
	tmp := t.TempDir()
	// PCI device directories named by bus address, as sysfs names them, with
	// renderD* symlinks pointing at them the way /sys/class/drm does.
	bdfs := []string{"0000:c7:00.0", "0000:83:00.0", "0000:43:00.0", "0000:03:00.0"}
	var kfdDirs []string
	for i, bdf := range bdfs {
		target := filepath.Join(tmp, bdf)
		if err := os.Mkdir(target, 0o755); err != nil {
			t.Fatal(err)
		}
		link := filepath.Join(tmp, "renderD"+strconv.Itoa(128+i))
		if err := os.Symlink(target, link); err != nil {
			t.Fatal(err)
		}
		kfdDirs = append(kfdDirs, link)
	}

	m := kfdIndexByBDF(kfdDirs)
	if len(m) != len(bdfs) {
		t.Fatalf("kfdIndexByBDF returned %d entries, want %d", len(m), len(bdfs))
	}
	// The index is the position in KFD order, which is deliberately not the
	// ascending bus order rocm-smi hands rows back in.
	for want, bdf := range bdfs {
		if got, ok := m[bdf]; !ok || got != want {
			t.Errorf("m[%q] = %d, %v; want %d, true", bdf, got, ok, want)
		}
	}
	if _, ok := m["0000:c9:00.0"]; ok {
		t.Error("a device absent from the KFD list must not be in the map")
	}
}

// A device rocm-smi reports but KFD does not know cannot be placed, and the
// collection has to be rejected rather than guessed at.
func TestKFDIndexByBDFSkipsBrokenLinks(t *testing.T) {
	tmp := t.TempDir()
	good := filepath.Join(tmp, "0000:03:00.0")
	if err := os.Mkdir(good, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(tmp, "renderD128")
	if err := os.Symlink(good, link); err != nil {
		t.Fatal(err)
	}
	dangling := filepath.Join(tmp, "renderD129")
	if err := os.Symlink(filepath.Join(tmp, "gone"), dangling); err != nil {
		t.Fatal(err)
	}

	m := kfdIndexByBDF([]string{link, dangling})
	if len(m) != 1 {
		t.Fatalf("got %d entries, want 1 (the dangling link must be dropped)", len(m))
	}
	if m["0000:03:00.0"] != 0 {
		t.Errorf("m[0000:03:00.0] = %d, want 0", m["0000:03:00.0"])
	}
}

func TestReadVRAMFromDir(t *testing.T) {
	tmp := t.TempDir()
	write := func(name, val string) {
		if err := os.WriteFile(filepath.Join(tmp, name), []byte(val+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("mem_info_vram_used", "60817408")     // 58 MiB
	write("mem_info_vram_total", "34209202176") // 32624 MiB

	used, total := readVRAMFromDir(tmp)
	if used != 58 {
		t.Errorf("used = %d MB, want 58", used)
	}
	if total != 32624 {
		t.Errorf("total = %d MB, want 32624", total)
	}
}

// A device directory with no VRAM files reports zero rather than failing —
// the caller renders "0.0 / 0.0 GB", which is visibly wrong in a way a
// silently-wrong number is not.
func TestReadVRAMFromDirMissing(t *testing.T) {
	used, total := readVRAMFromDir(t.TempDir())
	if used != 0 || total != 0 {
		t.Errorf("got %d/%d, want 0/0", used, total)
	}
}

func TestKFDNodeID(t *testing.T) {
	for _, tc := range []struct {
		path string
		want int
	}{
		{"/sys/class/kfd/kfd/topology/nodes/0/properties", 0},
		{"/sys/class/kfd/kfd/topology/nodes/3/properties", 3},
		{"/sys/class/kfd/kfd/topology/nodes/10/properties", 10},
	} {
		if got := kfdNodeID(tc.path); got != tc.want {
			t.Errorf("kfdNodeID(%q) = %d, want %d", tc.path, got, tc.want)
		}
	}
}

const rocminfoIntelHost = `
ROCk module is loaded
=====================
HSA Agents
==========
*******
Agent 1
*******
  Name:                    Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Marketing Name:          Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Device Type:             CPU
  Pool Info:
    Pool 1
      Name:                nothing
*******
Agent 2
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
`

// The CPU agent used to be skipped by blacklisting names starting with "AMD
// Ryzen" or "AMD EPYC", so on an Intel host the Xeon became GPU 0 and every
// real card shifted down one.
func TestParseROCmGPUNamesSkipsNonAMDCPU(t *testing.T) {
	names := parseROCmGPUNames(rocminfoIntelHost)
	want := []string{"AMD Radeon AI PRO R9700", "AMD Radeon AI PRO R9700"}
	if len(names) != len(want) {
		t.Fatalf("got %d names %q, want %d", len(names), names, len(want))
	}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("names[%d] = %q, want %q", i, names[i], want[i])
		}
	}
}

// A GPU agent with no marketing name falls back to its gfx target rather than
// rendering blank.
func TestParseROCmGPUNamesFallsBackToAgentName(t *testing.T) {
	const out = `
*******
Agent 1
*******
  Name:                    gfx1201
  Marketing Name:
  Device Type:             GPU
`
	names := parseROCmGPUNames(out)
	if len(names) != 1 || names[0] != "gfx1201" {
		t.Errorf("got %q, want [gfx1201]", names)
	}
}

func TestParseROCmGPUNamesNoGPUs(t *testing.T) {
	const out = `
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X
  Device Type:             CPU
`
	if names := parseROCmGPUNames(out); len(names) != 0 {
		t.Errorf("got %q, want none", names)
	}
}

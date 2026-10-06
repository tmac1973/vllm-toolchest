package monitor

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/testutil"
)

// The parser is inline in Collect, which runs nvidia-smi by name. Putting a
// fake nvidia-smi first on PATH tests the parsing on captured output without
// a GPU, and without changing how Collect finds the tool.
func fakeNvidiaSMI(t *testing.T, output string) {
	t.Helper()
	script := testutil.WriteScript(t, "cat <<'EOF'\n"+output+"EOF\n")
	dir := t.TempDir()
	if err := os.Symlink(script, filepath.Join(dir, "nvidia-smi")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

// Captured from `nvidia-smi --query-gpu=index,name,utilization.gpu,
// memory.used,memory.total,temperature.gpu,power.draw
// --format=csv,noheader,nounits` on a two-card box: fields are ", "
// separated, the name keeps its spaces, and power has decimals.
func TestNvidiaCollectParsesEveryCard(t *testing.T) {
	fakeNvidiaSMI(t, ""+
		"0, NVIDIA GeForce RTX 4090, 97, 22817, 24564, 71, 412.35\n"+
		"1, NVIDIA RTX A6000, 0, 4, 49140, 34, 21.08\n")

	got, err := (&nvidiaBackend{}).Collect()
	if err != nil {
		t.Fatal(err)
	}
	want := []GPUInfo{
		{Index: 0, Name: "NVIDIA GeForce RTX 4090", UtilPercent: 97, VRAMUsedMB: 22817, VRAMTotalMB: 24564, TempC: 71, PowerW: 412.35},
		{Index: 1, Name: "NVIDIA RTX A6000", UtilPercent: 0, VRAMUsedMB: 4, VRAMTotalMB: 49140, TempC: 34, PowerW: 21.08},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got  %+v\nwant %+v", got, want)
	}
}

// Datacenter and some laptop parts answer [N/A] or [Not Supported] for the
// readings they lack. Those read as zero; the card itself, and the readings it
// does have, must still come through.
func TestNvidiaCollectReadsUnavailableFieldsAsZero(t *testing.T) {
	fakeNvidiaSMI(t, ""+
		"0, NVIDIA H100 80GB HBM3, [N/A], 1024, 81559, 30, [N/A]\n"+
		"1, NVIDIA GeForce RTX 3050 Laptop GPU, 12, 512, 4096, [N/A], [Not Supported]\n")

	got, err := (&nvidiaBackend{}).Collect()
	if err != nil {
		t.Fatal(err)
	}
	want := []GPUInfo{
		{Index: 0, Name: "NVIDIA H100 80GB HBM3", VRAMUsedMB: 1024, VRAMTotalMB: 81559, TempC: 30},
		{Index: 1, Name: "NVIDIA GeForce RTX 3050 Laptop GPU", UtilPercent: 12, VRAMUsedMB: 512, VRAMTotalMB: 4096},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got  %+v\nwant %+v", got, want)
	}
}

// No output -- a driver with no visible devices -- is no GPUs, not an error
// and not a phantom card at index 0.
func TestNvidiaCollectOfEmptyOutputIsNoGPUs(t *testing.T) {
	fakeNvidiaSMI(t, "")
	got, err := (&nvidiaBackend{}).Collect()
	if err != nil || len(got) != 0 {
		t.Errorf("got %+v, %v; want no GPUs and no error", got, err)
	}
}

// A line that is not a GPU row (a warning nvidia-smi prints first, say) is
// skipped rather than parsed into a card.
func TestNvidiaCollectSkipsLinesThatAreNotRows(t *testing.T) {
	fakeNvidiaSMI(t, "WARNING: infoROM is corrupted at gpu 0000:01:00.0\n"+
		"0, NVIDIA L4, 5, 300, 23034, 40, 16.20\n")
	got, err := (&nvidiaBackend{}).Collect()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "NVIDIA L4" {
		t.Errorf("got %+v, want just the L4", got)
	}
}

// When nvidia-smi fails -- the driver went away under a running container --
// Collect says so rather than reporting no GPUs.
func TestNvidiaCollectReportsAFailingNvidiaSMI(t *testing.T) {
	script := testutil.WriteScript(t, "echo 'NVIDIA-SMI has failed' >&2\nexit 9\n")
	dir := t.TempDir()
	if err := os.Symlink(script, filepath.Join(dir, "nvidia-smi")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	if got, err := (&nvidiaBackend{}).Collect(); err == nil {
		t.Errorf("got %+v and no error", got)
	}
}

// A row short of the seven columns asked for is skipped. The guard was once
// one short of the last index read, and a six-field row panicked the server.
func TestNvidiaCollectSurvivesARowMissingTheLastField(t *testing.T) {
	fakeNvidiaSMI(t, "0, NVIDIA L4, 5, 300, 23034, 40\n")
	if _, err := (&nvidiaBackend{}).Collect(); err != nil {
		t.Fatal(err)
	}
}

// The name is the one free-text column, and a ", " inside it must not shift
// the numbers after it.
func TestNvidiaCollectReadsANameWithACommaInIt(t *testing.T) {
	fakeNvidiaSMI(t, "1, NVIDIA RTX 6000 Ada, 48GB, 7, 1024, 49140, 51, 88.5\n")
	gpus, err := (&nvidiaBackend{}).Collect()
	if err != nil || len(gpus) != 1 {
		t.Fatalf("got %+v, %v", gpus, err)
	}
	g := gpus[0]
	if g.Name != "NVIDIA RTX 6000 Ada, 48GB" || g.UtilPercent != 7 || g.VRAMTotalMB != 49140 || g.PowerW != 88.5 {
		t.Errorf("parsed %+v", g)
	}
}

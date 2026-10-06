package tuning

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// Real config.json values, so a change to the shape maths shows up against
// numbers someone can check on the Hub.
var (
	// Qwen/Qwen3-30B-A3B config.json: hidden_size 2048, intermediate_size
	// 6144, moe_intermediate_size 768, num_experts 128, 32 query heads, 4 KV
	// heads, head_dim 128.
	qwen3MoE = models.HFConfig{
		HiddenSize: 2048, IntermediateSize: 6144, MoEIntermediate: 768, NumExperts: 128,
		NumAttentionHeads: 32, NumKeyValueHeads: 4, HeadDim: 128,
	}
	// meta-llama/Llama-3.1-8B-Instruct config.json: hidden_size 4096,
	// intermediate_size 14336, 32 heads, 8 KV heads, and no head_dim, so it is
	// derived as 4096/32 = 128.
	llama31_8B = models.HFConfig{
		HiddenSize: 4096, IntermediateSize: 14336,
		NumAttentionHeads: 32, NumKeyValueHeads: 8,
	}
	// Qwen/Qwen3-4B config.json: hidden_size 2560, intermediate_size 9728, 32
	// heads, 8 KV heads and an explicit head_dim of 128. 32*128 = 4096 is not
	// the hidden size, which is the case where head_dim must not be derived.
	qwen3_4B = models.HFConfig{
		HiddenSize: 2560, IntermediateSize: 9728,
		NumAttentionHeads: 32, NumKeyValueHeads: 8, HeadDim: 128,
	}
)

// Each expected list is worked out by hand from DeriveShapes' documented
// matmuls -- Q (heads*hd/tp, h), KV (kv*hd/tp, h), O (h, heads*hd/tp),
// gate/up (inter/tp, h), down (h, inter/tp) -- then deduplicated, filtered to
// multiples of the block and sorted by N then K.
func TestDeriveShapesSplitsAttentionAndMLPAcrossRanks(t *testing.T) {
	cases := []struct {
		name           string
		cfg            models.HFConfig
		tp             int
		blockN, blockK int
		want           []Shape
	}{
		{
			// Q 32*128=4096, KV 4*128=512, inter 6144: nothing is split.
			name: "Qwen3-30B-A3B tp1", cfg: qwen3MoE, tp: 1, blockN: 128, blockK: 128,
			want: []Shape{{512, 2048}, {2048, 4096}, {2048, 6144}, {4096, 2048}, {6144, 2048}},
		},
		{
			// Q 2048 and O 2048 coincide at (2048,2048) and are tuned once;
			// KV 4 heads split as 2 per rank = 256.
			name: "Qwen3-30B-A3B tp2", cfg: qwen3MoE, tp: 2, blockN: 128, blockK: 128,
			want: []Shape{{256, 2048}, {2048, 2048}, {2048, 3072}, {3072, 2048}},
		},
		{
			// One KV head per rank: 128.
			name: "Qwen3-30B-A3B tp4", cfg: qwen3MoE, tp: 4, blockN: 128, blockK: 128,
			want: []Shape{{128, 2048}, {1024, 2048}, {1536, 2048}, {2048, 1024}, {2048, 1536}},
		},
		{
			// 4 KV heads do not divide over 8 ranks, so vLLM replicates them
			// and KV keeps its tp=1 shape (512,2048) -- which here is also
			// Q's shape (32*128/8 = 512), so it appears once.
			name: "Qwen3-30B-A3B tp8 replicates KV", cfg: qwen3MoE, tp: 8, blockN: 128, blockK: 128,
			want: []Shape{{512, 2048}, {768, 2048}, {2048, 512}, {2048, 768}},
		},
		{
			// With a 512 block, inter/8 = 768 is not a multiple and the
			// kernel never sees gate/up or down: they are not tuned.
			name: "Qwen3-30B-A3B tp8 block512 drops unaligned", cfg: qwen3MoE, tp: 8, blockN: 512, blockK: 512,
			want: []Shape{{512, 2048}, {2048, 512}},
		},
		{
			// A non-square block filters N and K separately: (128,2048)
			// fails N%256, (2048,1024) and (2048,1536) pass K%128.
			name: "Qwen3-30B-A3B tp4 block256x128", cfg: qwen3MoE, tp: 4, blockN: 256, blockK: 128,
			want: []Shape{{1024, 2048}, {1536, 2048}, {2048, 1024}, {2048, 1536}},
		},
		{
			// Q and O are both (4096,4096) at tp=1.
			name: "Llama-3.1-8B tp1 derives head_dim", cfg: llama31_8B, tp: 1, blockN: 128, blockK: 128,
			want: []Shape{{1024, 4096}, {4096, 4096}, {4096, 14336}, {14336, 4096}},
		},
		{
			name: "Llama-3.1-8B tp2", cfg: llama31_8B, tp: 2, blockN: 128, blockK: 128,
			want: []Shape{{512, 4096}, {2048, 4096}, {4096, 2048}, {4096, 7168}, {7168, 4096}},
		},
		{
			// tp=3 truncates 4096/3 and 14336/3 to unaligned widths, and 8 KV
			// heads replicate: only the replicated KV shape is left.
			name: "Llama-3.1-8B tp3 keeps only aligned shapes", cfg: llama31_8B, tp: 3, blockN: 128, blockK: 128,
			want: []Shape{{1024, 4096}},
		},
		{
			// The explicit head_dim gives Q 4096 wide against a 2560 hidden.
			name: "Qwen3-4B tp1 explicit head_dim", cfg: qwen3_4B, tp: 1, blockN: 128, blockK: 128,
			want: []Shape{{1024, 2560}, {2560, 4096}, {2560, 9728}, {4096, 2560}, {9728, 2560}},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := DeriveShapes(c.cfg, c.tp, c.blockN, c.blockK)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("DeriveShapes = %v, want %v", got, c.want)
			}
		})
	}
}

// A model whose config.json was never read has no hidden size, and there is
// nothing to tune: the tuning page must show zero shapes, not garbage.
func TestDeriveShapesReturnsNothingWithoutAHiddenSize(t *testing.T) {
	if got := DeriveShapes(models.HFConfig{NumAttentionHeads: 32}, 1, 128, 128); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

// Older configs omit intermediate_size, num_key_value_heads and head_dim. The
// fallbacks are 4*hidden, MHA (kv = heads) and hidden/heads.
func TestDeriveShapesFillsMissingFieldsWithTheirConventionalDefaults(t *testing.T) {
	cfg := models.HFConfig{HiddenSize: 1024, NumAttentionHeads: 8}
	// head_dim 128, kv 8 -> Q = KV = O = (1024,1024); inter 4096.
	want := []Shape{{1024, 1024}, {1024, 4096}, {4096, 1024}}
	if got := DeriveShapes(cfg, 1, 128, 128); !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// A tensor-parallel size of zero comes from a model saved before the field
// was set. It means one GPU, and must not divide by zero.
func TestDeriveShapesTreatsANonPositiveTPAsOne(t *testing.T) {
	want := DeriveShapes(llama31_8B, 1, 128, 128)
	for _, tp := range []int{0, -2} {
		if got := DeriveShapes(llama31_8B, tp, 128, 128); !reflect.DeepEqual(got, want) {
			t.Errorf("tp=%d: got %v, want the tp=1 shapes %v", tp, got, want)
		}
	}
}

// vLLM fuses q/k/v into one QKVParallelLinear and gate/up into one
// MergedColumnParallelLinear, so the block-FP8 GEMM it runs for those has N =
// (heads+2*kv)*hd/tp and N = 2*inter/tp. Its own tuner,
// benchmarks/kernels/benchmark_w8a8_block_fp8.py, lists DeepSeek-V3's gate_up
// as (18432*2 // tp, 7168) for that reason. A config tuned at the unfused N
// is never looked up.
func TestGateAndUpAreTunedAtTheFusedWidthVLLMRuns(t *testing.T) {
	t.Skip("known bug: DeriveShapes emits unfused gate/up and q/k/v widths vLLM never looks up")
	got := DeriveShapes(llama31_8B, 1, 128, 128)
	want := Shape{N: 2 * 14336, K: 4096}
	for _, s := range got {
		if s == want {
			return
		}
	}
	t.Errorf("no fused gate_up shape %v in %v", want, got)
}

// Qwen3-30B-A3B has no dense MLP: every layer is MoE (decoder_sparse_step 1,
// mlp_only_layers empty), and the experts run vLLM's fused-MoE kernel with its
// own config files. intermediate_size is unused, so (6144,2048) and
// (2048,6144) are tuned for a layer the model does not have.
func TestAnAllMoEModelIsNotTunedForADenseMLPItDoesNotHave(t *testing.T) {
	t.Skip("known bug: DeriveShapes ignores NumExperts and tunes an unused intermediate_size")
	for _, s := range DeriveShapes(qwen3MoE, 1, 128, 128) {
		if s.N == 6144 || s.K == 6144 {
			t.Errorf("dense MLP shape %v tuned for an all-MoE model", s)
		}
	}
}

// The runtime looks the file up by exact name. One character off and the
// tuned config is silently ignored, so the format is pinned to vLLM's.
func TestConfigFilenameMatchesVLLMsLookupName(t *testing.T) {
	got := ConfigFilename(Shape{N: 512, K: 2048}, "NVIDIA_H100_80GB_HBM3", 128, 128)
	want := "N=512,K=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json"
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
	// Block dimensions are written N first, then K.
	if got := ConfigFilename(Shape{N: 1, K: 2}, "AMD_Instinct_MI300X", 256, 128); !strings.HasSuffix(got, "block_shape=[256,128].json") {
		t.Errorf("block order: %q", got)
	}
}

// installFixture lays out a data dir with tuned JSONs for device and an empty
// vLLM configs dir.
func installFixture(t *testing.T, device string, files ...string) (dataDir, configsDir string) {
	t.Helper()
	root := t.TempDir()
	dataDir = filepath.Join(root, "data")
	src := filepath.Join(dataDir, "tuned-kernels", device)
	configsDir = filepath.Join(root, "configs")
	for _, d := range []string{src, configsDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(src, f), []byte(`{"tuned":"`+f+`"}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dataDir, configsDir
}

func linkTarget(t *testing.T, path string) string {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("lstat %s: %v", path, err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("%s is not a symlink", path)
	}
	target, err := os.Readlink(path)
	if err != nil {
		t.Fatal(err)
	}
	return target
}

// The tuned configs live on the data volume and vLLM reads them from its own
// site-packages, so every JSON is linked across. Anything else in the
// directory is not a kernel config and stays put.
func TestInstallTunedConfigsLinksEveryJSONIntoVLLM(t *testing.T) {
	const dev = "NVIDIA_H100_80GB_HBM3"
	a := ConfigFilename(Shape{N: 512, K: 2048}, dev, 128, 128)
	b := ConfigFilename(Shape{N: 2048, K: 4096}, dev, 128, 128)
	dataDir, configsDir := installFixture(t, dev, a, b, "notes.txt")
	if err := os.Mkdir(filepath.Join(dataDir, "tuned-kernels", dev, "sub.json"), 0o755); err != nil {
		t.Fatal(err)
	}

	n, warn, err := InstallTunedConfigs(dataDir, dev, configsDir)
	if err != nil || warn != "" || n != 2 {
		t.Fatalf("got n=%d warn=%q err=%v, want 2 installed and no warning", n, warn, err)
	}
	for _, f := range []string{a, b} {
		want := filepath.Join(dataDir, "tuned-kernels", dev, f)
		if got := linkTarget(t, filepath.Join(configsDir, f)); got != want {
			t.Errorf("%s -> %s, want %s", f, got, want)
		}
	}
	for _, f := range []string{"notes.txt", "sub.json"} {
		if _, err := os.Lstat(filepath.Join(configsDir, f)); err == nil {
			t.Errorf("%s was installed; only JSON files are configs", f)
		}
	}

	// It runs on every boot: a second run replaces its own links and
	// reports the same count rather than failing on them.
	n, warn, err = InstallTunedConfigs(dataDir, dev, configsDir)
	if err != nil || warn != "" || n != 2 {
		t.Errorf("second run: n=%d warn=%q err=%v", n, warn, err)
	}
}

// A symlink left from an earlier install -- the data dir moved, or the file
// was retuned -- is replaced, not kept pointing at the old place.
func TestInstallTunedConfigsReplacesAStaleSymlink(t *testing.T) {
	const dev = "dev"
	f := "N=1,K=1.json"
	dataDir, configsDir := installFixture(t, dev, f)
	dst := filepath.Join(configsDir, f)
	if err := os.Symlink("/nonexistent/old.json", dst); err != nil {
		t.Fatal(err)
	}

	n, warn, err := InstallTunedConfigs(dataDir, dev, configsDir)
	if err != nil || warn != "" || n != 1 {
		t.Fatalf("n=%d warn=%q err=%v", n, warn, err)
	}
	if got, want := linkTarget(t, dst), filepath.Join(dataDir, "tuned-kernels", dev, f); got != want {
		t.Errorf("link -> %s, want %s", got, want)
	}
}

// A regular file of the same name is one vLLM (or the radiance image) shipped.
// Replacing it would lose data that is not ours, so it is skipped and named.
func TestInstallTunedConfigsNeverOverwritesAShippedFile(t *testing.T) {
	const dev = "dev"
	f := "N=1,K=1.json"
	dataDir, configsDir := installFixture(t, dev, f, "N=2,K=2.json")
	dst := filepath.Join(configsDir, f)
	if err := os.WriteFile(dst, []byte("upstream"), 0o644); err != nil {
		t.Fatal(err)
	}

	n, warn, err := InstallTunedConfigs(dataDir, dev, configsDir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("installed %d, want 1 (the other file)", n)
	}
	if !strings.Contains(warn, f) || !strings.Contains(warn, "upstream regular file") {
		t.Errorf("warning %q does not name the skipped file", warn)
	}
	if b, _ := os.ReadFile(dst); string(b) != "upstream" {
		t.Errorf("shipped file now holds %q", b)
	}
}

// Nothing tuned yet is the first-run case, not an error.
func TestInstallTunedConfigsWithNoTunedDirDoesNothing(t *testing.T) {
	configsDir := t.TempDir()
	n, warn, err := InstallTunedConfigs(t.TempDir(), "dev", configsDir)
	if n != 0 || warn != "" || err != nil {
		t.Errorf("n=%d warn=%q err=%v, want a silent no-op", n, warn, err)
	}
}

// On a dev box there is no vLLM venv. That is reported, not failed.
func TestInstallTunedConfigsWithoutVLLMSkipsWithAWarning(t *testing.T) {
	dataDir, _ := installFixture(t, "dev", "N=1,K=1.json")
	missing := filepath.Join(t.TempDir(), "no-vllm")
	n, warn, err := InstallTunedConfigs(dataDir, "dev", missing)
	if n != 0 || err != nil || !strings.Contains(warn, missing) {
		t.Errorf("n=%d warn=%q err=%v, want a skip naming %s", n, warn, err, missing)
	}
}

// An empty configs dir means the generic image's default path.
func TestInstallTunedConfigsDefaultsTheConfigsDir(t *testing.T) {
	if _, err := os.Stat(DefaultVLLMConfigsDir); err == nil {
		t.Skip("the default vLLM configs dir exists here; installing into it is not a unit test")
	}
	dataDir, _ := installFixture(t, "dev", "N=1,K=1.json")
	_, warn, err := InstallTunedConfigs(dataDir, "dev", "")
	if err != nil || !strings.Contains(warn, DefaultVLLMConfigsDir) {
		t.Errorf("warn=%q err=%v, want a skip naming the default dir", warn, err)
	}
}

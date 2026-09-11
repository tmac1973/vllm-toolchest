package models

import (
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// Kernel tuning only affects the block-quantized FP8 GEMM. Matching on the
// quantization method alone marked per-channel FP8 and 4-bit weight-only
// compressed-tensors data as tunable — both take a different code path, so the
// page offered a tuning run that would have changed nothing.
func TestIsBlockFP8(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config string
		want   bool
		why    string
	}{
		{
			name:   "native fp8 with a block shape",
			config: `{"quantization_config":{"quant_method":"fp8","weight_block_size":[128,128]}}`,
			want:   true,
			why:    "the case the tuner exists for",
		},
		{
			name:   "native fp8 without one",
			config: `{"quantization_config":{"quant_method":"fp8"}}`,
			want:   false,
			why:    "per-tensor FP8 does not reach the block kernel",
		},
		{
			name: "compressed-tensors, block strategy",
			config: `{"quantization_config":{"quant_method":"compressed-tensors","format":"float-quantized",
				"config_groups":{"group_0":{"weights":{"num_bits":8,"type":"float","strategy":"block","block_structure":[128,128]}}}}}`,
			want: true,
			why:  "block_structure is where compressed-tensors states it",
		},
		{
			name: "compressed-tensors, channel strategy",
			config: `{"quantization_config":{"quant_method":"compressed-tensors","format":"float-quantized",
				"config_groups":{"group_0":{"weights":{"num_bits":8,"type":"float","strategy":"channel"}}}}}`,
			want: false,
			why:  "per-channel FP8 is still FP8, and still not tunable",
		},
		{
			name: "compressed-tensors, group strategy at 4 bits",
			config: `{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized",
				"config_groups":{"group_0":{"weights":{"num_bits":4,"type":"int","strategy":"group","group_size":128}}}}}`,
			want: false,
			why:  "weight-only 4-bit wearing the compressed-tensors label",
		},
		{
			name: "blockwise but 4-bit",
			config: `{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized",
				"config_groups":{"group_0":{"weights":{"num_bits":4,"type":"int","strategy":"block","block_structure":[128,128]}}}}}`,
			want: false,
			why:  "a block shape alone is not enough; the kernel is FP8",
		},
		{
			name:   "awq",
			config: `{"quantization_config":{"quant_method":"awq","bits":4,"group_size":128}}`,
			want:   false,
			why:    "never a candidate",
		},
		{
			name:   "unquantized",
			config: `{"torch_dtype":"bfloat16"}`,
			want:   false,
			why:    "nothing to tune",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q := DetectQuantization(writeConfig(t, tc.config), "test/model")
			if got := q.IsBlockFP8(); got != tc.want {
				t.Errorf("IsBlockFP8() = %v, want %v — %s (method=%q bits=%d block=%v)",
					got, tc.want, tc.why, q.Method, q.Bits, q.WeightBlockSize)
			}
		})
	}
}

// A malformed or partial block shape must not read as blockwise: acting on it
// would derive shapes from a number that is not there.
func TestIsBlockFP8RejectsMalformedShapes(t *testing.T) {
	for _, tc := range []struct {
		name string
		q    QuantMeta
	}{
		{"one dimension", QuantMeta{Method: "fp8", WeightBlockSize: []int{128}}},
		{"zero dimension", QuantMeta{Method: "fp8", WeightBlockSize: []int{128, 0}}},
		{"negative", QuantMeta{Method: "fp8", WeightBlockSize: []int{-1, 128}}},
		{"empty", QuantMeta{Method: "fp8", WeightBlockSize: []int{}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.q.IsBlockFP8() {
				t.Errorf("%v accepted as a block shape", tc.q.WeightBlockSize)
			}
		})
	}
}

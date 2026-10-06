"""Tests for the tuner wrapper's search-space pruning.

Run with: python3 -m unittest discover -s scripts -p 'test_*.py'
They need neither torch nor the upstream tuner: outside the image the wrapper
imports with nothing patched, and the pruning rule is a plain function.
"""
import itertools
import unittest

import tune_fp8_wrapper as w


def upstream_space():
    """benchmark_w8a8_block_fp8.get_configs_compute_bound, as vLLM ships it."""
    return [
        {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k,
         "GROUP_SIZE_M": g, "num_warps": warps, "num_stages": stages}
        for stages, m, k, n, warps, g in itertools.product(
            [2, 3, 4, 5], [16, 32, 64, 128, 256], [64, 128], [32, 64, 128, 256], [4, 8], [1, 16, 32, 64])
    ]


def tile(m, n, warps):
    return {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "num_warps": warps}


class FitsRegisters(unittest.TestCase):
    def test_a_tile_that_spills_is_left_out(self):
        # 256x256 on 4 warps of 32: 512 accumulators a thread, double the
        # register file -- the config the tuner spent minutes compiling.
        self.assertFalse(w.fits_registers(tile(256, 256, 4), 32))
        self.assertFalse(w.fits_registers(tile(256, 128, 4), 32))

    def test_tiles_within_the_register_file_are_kept(self):
        self.assertTrue(w.fits_registers(tile(128, 128, 4), 32))   # 128 a thread
        self.assertTrue(w.fits_registers(tile(256, 128, 8), 32))   # 128 a thread
        self.assertTrue(w.fits_registers(tile(16, 32, 4), 32))

    def test_a_wider_wavefront_keeps_more(self):
        # On CDNA (wavefront 64) the same tile is spread over twice the
        # threads: 256x128 on 4 warps is 128 a thread and fits.
        self.assertTrue(w.fits_registers(tile(256, 128, 4), 64))
        self.assertFalse(w.fits_registers(tile(256, 256, 4), 64))

    def test_only_the_largest_tiles_go(self):
        space = upstream_space()
        kept = [c for c in space if w.fits_registers(c, 32)]
        dropped = [c for c in space if not w.fits_registers(c, 32)]
        self.assertEqual(len(space), 1280)
        # 256x256 at either warp count, and 256x128 / 128x256 on 4 warps:
        # 4 tile-and-warp pairs x 2 BLOCK_K x 4 GROUP_SIZE_M x 4 stages.
        self.assertEqual(len(dropped), 4 * 2 * 4 * 4)
        self.assertTrue(all(c["BLOCK_SIZE_M"] * c["BLOCK_SIZE_N"] >= 128 * 256 for c in dropped))
        self.assertEqual(len(kept), 1280 - 128)


if __name__ == "__main__":
    unittest.main()

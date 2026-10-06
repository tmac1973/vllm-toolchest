#!/usr/bin/env python3
"""Thin wrapper around vLLM's benchmark_w8a8_block_fp8.py that lets us pass
arbitrary (N, K) weight shapes instead of the DeepSeek-V3 defaults baked into
the upstream script, and leaves out kernel configs that cannot win.

Sits next to benchmark_w8a8_block_fp8.py on the container's filesystem
(/opt/vllm-tuner/) so the import resolves without sys.path gymnastics.
"""
import argparse
import importlib
import sys
import types

# The most fp32 accumulators a config may give each thread. The register file
# holds 256 per thread on every GPU this targets, and the A and B fragments
# need room beside the accumulators; past this the kernel spills, runs slower
# than a smaller tile, and so is never the config chosen.
MAX_ACCUMULATORS_PER_THREAD = 128


def fits_registers(config, warp_size):
    """Whether config's output tile fits in its threads' registers."""
    threads = config["num_warps"] * warp_size
    return config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"] // threads <= MAX_ACCUMULATORS_PER_THREAD


def _warp_size():
    """The device's warp (wavefront) size: 32 on RDNA and NVIDIA, 64 on CDNA.
    When it cannot be read, 64 -- the larger size prunes the fewest configs."""
    try:
        import torch
        return int(getattr(torch.cuda.get_device_properties(0), "warp_size", 64) or 64)
    except Exception:
        return 64


def _install():
    """Import the upstream tuner and narrow its search space, or return None
    where it is not installed (tests, a checkout outside the image).

    This runs at import, not in main(): the tuner works in spawned processes,
    one per GPU, and a spawned process imports this file afresh (as
    __mp_main__) before it runs its share. A patch made only in main() would
    exist in the parent alone, and every worker would search the full space.

    The configs left out are the ones whose tiles spill. A 256x256 tile on 4
    warps of 32 asks each thread for 512 accumulators; LLVM then spends
    minutes per config on a megabyte of IR, and on 2026-10-06 the first
    shape's first batch size had spent over 20 minutes in such compiles at
    under half done, on a config that could not be the fastest.
    """
    sys.path.insert(0, "/opt/vllm-tuner")
    try:
        bench = importlib.import_module("benchmark_w8a8_block_fp8")
    except ImportError:
        return None
    warp = _warp_size()
    full = bench.get_configs_compute_bound

    def pruned():
        return [c for c in full() if fits_registers(c, warp)]

    bench.get_configs_compute_bound = pruned
    bench._full_search_space = full
    return bench


bench = _install()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", required=True,
                   help="comma-separated N:K pairs, e.g. 4096:5120,8704:5120")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--block-k", type=int, default=128)
    p.add_argument("--save-path", default="/data/tuned-kernels")
    p.add_argument("--out-dtype", default="bfloat16",
                   choices=["float32", "float16", "half", "bfloat16"])
    args = p.parse_args()

    if bench is None:
        print("benchmark_w8a8_block_fp8 not found in /opt/vllm-tuner", file=sys.stderr)
        return 1

    shape_pairs = []
    for pair in args.shapes.split(","):
        n, k = pair.split(":")
        shape_pairs.append((int(n), int(k)))

    # Replace its hardcoded DeepSeek shape source. main() calls this in this
    # process, so unlike the search space it needs no patch at import.
    def _shapes(_tp_size):
        return shape_pairs
    bench.get_weight_shapes = _shapes

    print(f"[tuner] searching {len(bench.get_configs_compute_bound())} of "
          f"{len(bench._full_search_space())} kernel configs per batch size; "
          f"the rest spill (warp size {_warp_size()})", flush=True)

    # Build the args namespace the upstream main() expects.
    fake_args = types.SimpleNamespace(
        tp_size=args.tp_size,
        input_type="fp8",
        out_dtype=args.out_dtype,
        block_n=args.block_n,
        block_k=args.block_k,
        batch_size=None,
        save_path=args.save_path,
    )
    bench.main(fake_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

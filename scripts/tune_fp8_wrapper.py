#!/usr/bin/env python3
"""Thin wrapper around vLLM's benchmark_w8a8_block_fp8.py that lets us pass
arbitrary (N, K) weight shapes instead of the DeepSeek-V3 defaults baked into
the upstream script.

Sits next to benchmark_w8a8_block_fp8.py on the container's filesystem
(/opt/vllm-tuner/) so the import resolves without sys.path gymnastics.
"""
import argparse
import importlib
import sys
import types


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

    shape_pairs = []
    for pair in args.shapes.split(","):
        n, k = pair.split(":")
        shape_pairs.append((int(n), int(k)))

    # Import the upstream tuner. Lives in /opt/vllm-tuner/ alongside this
    # wrapper (copied from vLLM source during container build).
    sys.path.insert(0, "/opt/vllm-tuner")
    bench = importlib.import_module("benchmark_w8a8_block_fp8")

    # Replace its hardcoded DeepSeek shape source.
    def _shapes(_tp_size):
        return shape_pairs
    bench.get_weight_shapes = _shapes

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

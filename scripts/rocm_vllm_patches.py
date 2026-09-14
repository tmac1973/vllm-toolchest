#!/usr/bin/env python3
"""Build-time edits to a vLLM source tree, for the from-source ROCm image.

Run from Dockerfile.rocm against a freshly cloned vLLM checkout, before the
wheel is built. vLLM's ROCm support is written for datacenter Instinct parts;
these edits stop it leaving consumer Radeon cards on untuned paths.

Which files need touching was originally learned from
github.com/kyuz0/amd-r9700-vllm-toolboxes. This implementation is our own, and
it deliberately carries far less than that script did, because most of what it
did is no longer needed:

  - forcing ROCm detection. The image installs the amdsmi bindings into its
    venv and verifies they import, so at runtime, with cards present, vLLM
    selects the ROCm platform on its own.
  - the no-GPU architecture fallback. This one was measured on the first build
    rather than reasoned about, and the reasoning it replaces was wrong: amdsmi
    enumerates the host's cards even with no /dev/kfd in the build container,
    so ROCm IS selected on a GPU-less builder, vllm/platforms/rocm.py IS
    imported, and resolving _GCN_ARCH there ends in torch.cuda raising "No CUDA
    GPUs are available". Neither VLLM_TARGET_DEVICE=cpu nor hiding the cards
    with ROCR_VISIBLE_DEVICES avoids it. The build's own smoke test therefore
    treats exactly that error as the expected outcome -- see the end of
    Dockerfile.rocm -- rather than patching vLLM to satisfy it. The consequence
    to know: `import vllm` cannot complete in this image without a usable GPU.
  - pinning device_type to "cuda", which upstream now does itself.
  - the mwaitxintrin.h include, the INT8 config fallback and the HIP_FOUND
    CMake check: all three no longer match anything upstream.

Anything an edit cannot apply is an error, not a shrug. The script it replaces
printed NOOP and carried on, so an upstream rename produced an image that was
quietly unpatched. Use --lenient to downgrade that to a warning, knowingly.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Marker written into every edit, so a re-run recognises its own work.
MARK = "vllm-toolchest:"


@dataclass(frozen=True)
class Edit:
    """One change to one file in the vLLM tree."""

    id: str
    path: str
    why: str
    # Rewrites the file text. Returns the text unchanged when its anchor is
    # gone, which the runner reports as a failure.
    apply: Callable[[str], str]
    # Substring proving this edit is already in the text.
    marker: str
    # Only applied when the image is being built for one of these targets.
    # Empty means every target.
    arches: frozenset[str] = field(default_factory=frozenset)


def _once(text: str, anchor: str, replacement: str) -> str:
    """Replace the first occurrence of anchor, or return text untouched."""
    return text.replace(anchor, replacement, 1) if anchor in text else text


# ── The edits ───────────────────────────────────────────────────────────────

_FP8_ANCHOR = """    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
"""

_FP8_FALLBACK = '''    # vllm-toolchest: no tuned block-FP8 configs ship for RDNA parts, and the
    # untuned default is materially slower. Borrow MI300X's, which is what the
    # kernel was tuned against and the closest entry that exists. The swap is
    # done on the path so it stays correct whatever spelling upstream uses for
    # the block shape in the file name.
    if not os.path.exists(config_file_path) and any(
        tag in device_name for tag in ("gfx12", "Radeon", "Graphics")
    ):
        borrowed = config_file_path.replace(
            f"device_name={device_name}", "device_name=AMD_Instinct_MI300X"
        )
        if borrowed != config_file_path and os.path.exists(borrowed):
            config_file_path = borrowed
'''

_AITER_ANCHOR = "IS_AITER_FOUND = is_aiter_found()\n"

_AITER_MAP = '''
# vllm-toolchest: AITER's Triton kernels are selected by device, and its table
# has no entry for gfx1201. Point it at MI350X, the architecture those kernels
# were written for and the one this card is closest to. setdefault so a future
# AITER that knows gfx1201 keeps its own answer.
if IS_AITER_FOUND:
    try:
        from aiter.ops.triton.utils import arch_info as _vt_arch_info

        _vt_arch_info._ARCH_TO_DEVICE.setdefault("gfx1201", "MI350X")
    except Exception:  # noqa: BLE001 - absence or a rename must not break import
        pass
'''

_MI3XX_ANCHOR = '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])'

_MI3XX_WITH_GFX1201 = (
    '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])'
    # vllm-toolchest: gfx1201 reaches the AITER and FP8 Triton paths gated on
    # this predicate, and runs them correctly.
    ' or "gfx1201" in _GCN_ARCH'
)

_DEVICE_NAME_ANCHOR = '''    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        asic_info_device_id: str = asic_info["device_id"]
        if asic_info_device_id in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[asic_info_device_id]
        return asic_info["market_name"]
'''

_DEVICE_NAME_FIXED = '''    def get_device_name(cls, device_id: int = 0) -> str:
        # vllm-toolchest: AITER keys its JIT kernel cache on this string and
        # recognises the "AMD-gfx..." spelling, not the marketing name amdsmi
        # reports. This image is built for one architecture, so answer with it.
        return "AMD-gfx1201"
'''

# transformers moved these behind a lazy top-level __getattr__, which some
# versions resolve poorly when vLLM imports them at module scope. Importing
# from the defining modules is equivalent and unambiguous. Ordered: the
# combined forms must be tried before the single-name ones.
_TRANSFORMERS_IMPORTS = [
    (
        "from transformers import GenerationConfig, PretrainedConfig",
        "from transformers.generation import GenerationConfig\n"
        "from transformers.configuration_utils import PretrainedConfig",
    ),
    (
        "from transformers import PretrainedConfig, GenerationConfig",
        "from transformers.generation import GenerationConfig\n"
        "from transformers.configuration_utils import PretrainedConfig",
    ),
    (
        "from transformers import PretrainedConfig",
        "from transformers.configuration_utils import PretrainedConfig",
    ),
    (
        "from transformers import GenerationConfig",
        "from transformers.generation import GenerationConfig",
    ),
]


def _split_transformers_imports(text: str) -> str:
    for anchor, replacement in _TRANSFORMERS_IMPORTS:
        if anchor in text:
            return text.replace(anchor, replacement, 1)
    return text


EDITS: list[Edit] = [
    Edit(
        id="fp8-borrow-mi300x-configs",
        path="vllm/model_executor/layers/quantization/utils/fp8_utils.py",
        why="use MI300X's tuned block-FP8 configs on RDNA",
        apply=lambda t: _once(t, _FP8_ANCHOR, _FP8_ANCHOR + _FP8_FALLBACK),
        marker=f"{MARK} no tuned block-FP8 configs",
    ),
    Edit(
        id="transformers-import-from-source-modules",
        path="vllm/transformers_utils/config.py",
        why="import GenerationConfig/PretrainedConfig from their own modules",
        apply=_split_transformers_imports,
        marker="from transformers.configuration_utils import PretrainedConfig",
    ),
    Edit(
        id="mi3xx-paths-include-gfx1201",
        path="vllm/platforms/rocm.py",
        why="let gfx1201 take the AITER and FP8 Triton paths",
        apply=lambda t: _once(t, _MI3XX_ANCHOR, _MI3XX_WITH_GFX1201),
        marker='or "gfx1201" in _GCN_ARCH',
        arches=frozenset({"gfx1201"}),
    ),
    Edit(
        id="device-name-for-aiter-jit",
        path="vllm/platforms/rocm.py",
        why="report AMD-gfx1201, which AITER's JIT cache expects",
        apply=lambda t: _once(t, _DEVICE_NAME_ANCHOR, _DEVICE_NAME_FIXED),
        marker=f"{MARK} AITER keys its JIT kernel cache",
        arches=frozenset({"gfx1201"}),
    ),
    Edit(
        id="aiter-treats-gfx1201-as-mi350x",
        path="vllm/_aiter_ops.py",
        why="give AITER a device entry for gfx1201",
        apply=lambda t: _once(t, _AITER_ANCHOR, _AITER_ANCHOR + _AITER_MAP),
        marker=f"{MARK} AITER's Triton kernels are selected by device",
        arches=frozenset({"gfx1201"}),
    ),
]


# ── Runner ──────────────────────────────────────────────────────────────────

APPLIED, ALREADY, SKIPPED, FAILED = "applied", "already", "skipped", "FAILED"


def targets_for(arch_spec: str) -> set[str]:
    """The gfx targets this build covers. PYTORCH_ROCM_ARCH allows a
    semicolon-separated list, and any of them may gate an edit."""
    return {part.strip() for part in arch_spec.replace(",", ";").split(";") if part.strip()}


def run(tree: Path, arch_spec: str, dry_run: bool = False) -> list[tuple[Edit, str, str]]:
    """Apply every edit. Returns (edit, status, detail) per edit, in order."""
    targets = targets_for(arch_spec)
    results: list[tuple[Edit, str, str]] = []

    # Group by file so a file with several edits is read and written once.
    for path_str in dict.fromkeys(e.path for e in EDITS):
        path = tree / path_str
        edits = [e for e in EDITS if e.path == path_str]

        original: str | None = None
        text = ""
        if path.exists():
            original = text = path.read_text()

        for edit in edits:
            if edit.arches and not (edit.arches & targets):
                results.append((edit, SKIPPED, f"not built for {'/'.join(sorted(edit.arches))}"))
                continue
            if original is None:
                results.append((edit, FAILED, f"{path_str} does not exist"))
                continue
            if edit.marker in text:
                results.append((edit, ALREADY, "marker present"))
                continue
            updated = edit.apply(text)
            if updated == text:
                results.append((edit, FAILED, "anchor not found — upstream moved"))
                continue
            text = updated
            results.append((edit, APPLIED, edit.why))

        if original is not None and text != original and not dry_run:
            path.write_text(text)

    return results


def report(results: list[tuple[Edit, str, str]]) -> int:
    width = max((len(e.id) for e, _, _ in results), default=0)
    for edit, status, detail in results:
        print(f"  {status:<7} {edit.id:<{width}}  {detail}")
    return sum(1 for _, status, _ in results if status == FAILED)


# ── Self-test ───────────────────────────────────────────────────────────────
# Fixtures are the upstream text each anchor was written against. They keep the
# anchors honest without a vLLM checkout, and prove every edit is idempotent.

_FIXTURES = {
    "vllm/model_executor/layers/quantization/utils/fp8_utils.py": (
        "    device_name = get_device_name_as_file_name()\n"
        '    json_file_name = f"N={N},K={K},device_name={device_name}.json"\n\n'
        + _FP8_ANCHOR
        + "    if os.path.exists(config_file_path):\n        pass\n"
    ),
    "vllm/transformers_utils/config.py": (
        "from transformers import GenerationConfig, PretrainedConfig\n"
        "from transformers.configuration_utils import ALLOWED_LAYER_TYPES\n"
    ),
    "vllm/platforms/rocm.py": (
        "_GCN_ARCH = _get_gcn_arch()\n"
        + _MI3XX_ANCHOR
        + "\n_ON_GFX9 = True\n\n    @classmethod\n"
        + _DEVICE_NAME_ANCHOR
    ),
    "vllm/_aiter_ops.py": ("import ctypes\n\n" + _AITER_ANCHOR + "\n\nclass _DlInfo:\n    pass\n"),
}


def self_test() -> int:
    import tempfile

    failures = 0
    for arch, expect_gated in (("gfx1201", True), ("gfx1100", False)):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            for rel, body in _FIXTURES.items():
                dest = tree / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body)

            first = run(tree, arch)
            for edit, status, detail in first:
                gated = bool(edit.arches)
                want = APPLIED if (not gated or expect_gated) else SKIPPED
                if status != want:
                    print(f"  FAIL  [{arch}] {edit.id}: {status} ({detail}), want {want}")
                    failures += 1

            # Re-running must change nothing: the build re-runs on every cache
            # miss, and a second pass that stacks edits would be silent damage.
            snapshot = {rel: (tree / rel).read_text() for rel in _FIXTURES}
            for edit, status, detail in run(tree, arch):
                if status not in (ALREADY, SKIPPED):
                    print(f"  FAIL  [{arch}] {edit.id}: not idempotent ({status})")
                    failures += 1
            for rel, before in snapshot.items():
                if (tree / rel).read_text() != before:
                    print(f"  FAIL  [{arch}] {rel} changed on the second run")
                    failures += 1

    # A missing file has to be loud, not invisible.
    with tempfile.TemporaryDirectory() as tmp:
        if not any(status == FAILED for _, status, _ in run(Path(tmp), "gfx1201")):
            print("  FAIL  an empty tree should fail every edit")
            failures += 1

    print("self-test: " + ("OK" if failures == 0 else f"{failures} failure(s)"))
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=".", help="root of the vLLM checkout")
    ap.add_argument(
        "--arch",
        default=os.environ.get("GPU_ARCH", ""),
        help="gfx target(s) this image is built for; defaults to $GPU_ARCH",
    )
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument(
        "--lenient",
        action="store_true",
        help="an edit whose anchor is gone warns instead of failing the build",
    )
    ap.add_argument("--self-test", action="store_true", help="check the edits against fixtures")
    args = ap.parse_args()

    if args.self_test:
        return 1 if self_test() else 0

    print(f"vLLM ROCm build edits (arch={args.arch or 'unset'}, tree={args.tree})")
    failed = report(run(Path(args.tree), args.arch, args.dry_run))
    if not failed:
        return 0
    if args.lenient:
        print(f"warning: {failed} edit(s) could not be applied; continuing (--lenient)")
        return 0
    print(
        f"error: {failed} edit(s) could not be applied.\n"
        "  vLLM has moved under them. Re-check each against the current tree and\n"
        "  update or drop it; build with --lenient only if you accept an image\n"
        "  without them."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

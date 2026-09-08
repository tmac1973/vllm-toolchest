#!/usr/bin/env python3
# Source: github.com/kyuz0/amd-r9700-vllm-toolboxes (scripts/patch_vllm.py)
# Build-time patches for AMD Radeon R9700 (gfx1201) vLLM compilation.
# Run from the root of a freshly cloned vLLM repo. All patches are idempotent.

import re
import sys
from pathlib import Path

_OK = 0
_WARN = 0


def _patch(path_str: str, description: str, fn):
    global _OK, _WARN
    p = Path(path_str)
    if not p.exists():
        print(f"  SKIP  {path_str}  (file not found)")
        _WARN += 1
        return
    original = p.read_text()
    result = fn(original)
    if result is None or result == original:
        print(f"  NOOP  {path_str}  ({description} - already applied or no match)")
        _OK += 1
        return
    p.write_text(result)
    print(f"  OK    {path_str}  ({description})")
    _OK += 1


def patch_1_init_py(txt: str) -> str:
    txt = txt.replace("import amdsmi", "# import amdsmi")
    txt = re.sub(r"is_rocm = .*", "is_rocm = True", txt)
    txt = re.sub(
        r"if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:", "if True:", txt
    )
    txt = txt.replace("amdsmi.amdsmi_init()", "pass")
    txt = txt.replace("amdsmi.amdsmi_shut_down()", "pass")

    monkeypatch = (
        "\n\n"
        "# Monkeypatch torch.library._clear_torch_ops_cache and _del_library to prevent shutdown crashes\n"
        "try:\n"
        "    import torch.library\n"
        "    if hasattr(torch.library, '_clear_torch_ops_cache'):\n"
        "        _orig_clear_cache = torch.library._clear_torch_ops_cache\n"
        "        def _safe_clear_torch_ops_cache(op_defs):\n"
        "            try:\n"
        "                _orig_clear_cache(op_defs)\n"
        "            except Exception:\n"
        "                pass\n"
        "        torch.library._clear_torch_ops_cache = _safe_clear_torch_ops_cache\n"
        "    if hasattr(torch.library, '_del_library'):\n"
        "        _orig_del_library = torch.library._del_library\n"
        "        def _safe_del_library(*args, **kwargs):\n"
        "            try:\n"
        "                _orig_del_library(*args, **kwargs)\n"
        "            except Exception:\n"
        "                pass\n"
        "        torch.library._del_library = _safe_del_library\n"
        "except Exception:\n"
        "    pass\n"
    )
    if "safe_clear_torch_ops_cache" not in txt:
        txt += monkeypatch

    return txt


def patch_2_rocm_py_mock(txt: str) -> str:
    # This used to prepend `sys.modules["amdsmi"] = MagicMock()` here, from
    # when the image had no amdsmi bindings and vLLM's ROCm platform would not
    # import without them.
    #
    # It has to stay removed. The Dockerfile now copies the real bindings into
    # the venv, and the mock shadowed them: setting sys.modules ahead of this
    # module's own `from amdsmi import ...` means every amdsmi call returned a
    # MagicMock rather than a number. vLLM's get_device_total_memory() queries
    # amdsmi inside a try/except and falls back to torch.cuda on failure — but
    # a MagicMock does not fail, so the fallback never ran and the mock escaped
    # into arithmetic:
    #
    #   File "vllm/engine/arg_utils.py", line 2674, in get_batch_defaults
    #     if device_memory >= 160 * GiB_bytes:
    #   TypeError: '>=' not supported between instances of 'MagicMock' and 'int'
    #
    # Without the mock both paths are correct: with the bindings present vLLM
    # reads real VRAM, and without them the import raises, vLLM warns, and it
    # falls back to torch.cuda. Nothing here needs amdsmi anyway — patch_4
    # hardcodes _GCN_ARCH, and patch_1 already removes the amdsmi dependency
    # from platform detection.

    txt = re.sub(r'device_type\s*:\s*str\s*=\s*"[^"]*"', 'device_type: str = "cuda"', txt)
    txt = re.sub(r'device_name\s*:\s*str\s*=\s*"[^"]*"', 'device_name: str = "gfx1201"', txt)

    if "AMD-gfx1201" not in txt:
        txt += (
            "\n"
            "    @classmethod\n"
            "    def get_device_name(cls, device_id: int = 0) -> str:\n"
            '        return "AMD-gfx1201"\n'
        )
    return txt


def patch_3_transformers_config(txt: str) -> str:
    txt = txt.replace(
        "from transformers import GenerationConfig, PretrainedConfig",
        "from transformers.generation import GenerationConfig\n"
        "from transformers.configuration_utils import PretrainedConfig",
    )
    txt = txt.replace(
        "from transformers import PretrainedConfig, GenerationConfig",
        "from transformers.generation import GenerationConfig\n"
        "from transformers.configuration_utils import PretrainedConfig",
    )
    txt = txt.replace(
        "from transformers import PretrainedConfig",
        "from transformers.configuration_utils import PretrainedConfig",
    )
    txt = txt.replace(
        "from transformers import GenerationConfig",
        "from transformers.generation import GenerationConfig",
    )
    return txt


def patch_4_gcn_arch(txt: str) -> str:
    return re.sub(r"_GCN_ARCH\s*=\s*_get_gcn_arch\(\)", '_GCN_ARCH = "gfx1201"', txt)


def patch_5_spinloop(txt: str) -> str:
    return txt.replace("#include <mwaitxintrin.h>", "#include <x86intrin.h>")


def patch_6_on_mi3xx(txt: str) -> str:
    return txt.replace(
        '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])',
        '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950", "gfx1201"])',
    )


def patch_7_aiter_ops(txt: str) -> str:
    marker = "map gfx1201 to MI350X"
    if marker in txt:
        return txt

    replacement = (
        "IS_AITER_FOUND = is_aiter_found()\n"
        "\n"
        "# R9700/RDNA4: map gfx1201 to MI350X in AITER arch detection\n"
        "if IS_AITER_FOUND:\n"
        "    try:\n"
        "        import aiter.ops.triton.utils.arch_info as _arch_info\n"
        '        _arch_info._ARCH_TO_DEVICE["gfx1201"] = "MI350X"\n'
        "    except (ImportError, AttributeError):\n"
        "        pass"
    )
    return txt.replace("IS_AITER_FOUND = is_aiter_found()", replacement)


def patch_8_fp8_utils(txt: str) -> str:
    marker = 'fallback_device_name = "AMD_Instinct_MI300X"'
    if marker in txt:
        return txt

    target = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )"
    )
    replacement = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )\n"
        "    # R9700/RDNA4: fall back to MI300X tuning configs\n"
        '    if not os.path.exists(config_file_path) and ("gfx12" in device_name or "Radeon" in device_name or "Graphics" in device_name):\n'
        '        fallback_device_name = "AMD_Instinct_MI300X"\n'
        "        fallback_json = f\"N={N},K={K},device_name={fallback_device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json\"\n"
        "        fallback_path = os.path.join(\n"
        '            os.path.dirname(os.path.realpath(__file__)), "configs", fallback_json\n'
        "        )\n"
        "        if os.path.exists(fallback_path):\n"
        "            config_file_path = fallback_path"
    )
    return txt.replace(target, replacement)


def patch_10_hip_found(txt: str) -> str:
    """Recent PyTorch's cmake/public/LoadHIP.cmake stopped setting HIP_FOUND
    and now sets PYTORCH_FOUND_HIP instead (after migrating from
    find_package(HIP) to find_package(hip CONFIG)). vLLM's CMakeLists.txt
    still checks HIP_FOUND, so the build fails with 'Can't find CUDA or HIP
    installation' even though HIP is configured fine. Accept both names."""
    if "PYTORCH_FOUND_HIP" in txt:
        return txt
    txt = txt.replace(
        "if (NOT HIP_FOUND AND CUDA_FOUND)",
        "if (NOT HIP_FOUND AND NOT PYTORCH_FOUND_HIP AND CUDA_FOUND)",
    )
    txt = txt.replace(
        "elseif(HIP_FOUND)",
        "elseif(HIP_FOUND OR PYTORCH_FOUND_HIP)",
    )
    return txt


def patch_9_int8_utils(txt: str) -> str:
    marker = 'fallback_device_name = "AMD_Instinct_MI300X"'
    if marker in txt:
        return txt

    # int8_utils uses "block_shape=[{block_n}, {block_k}]" (with space)
    target = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )"
    )
    replacement = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )\n"
        "    # R9700/RDNA4: fall back to MI300X tuning configs\n"
        '    if not os.path.exists(config_file_path) and ("gfx12" in device_name or "Radeon" in device_name or "Graphics" in device_name):\n'
        '        fallback_device_name = "AMD_Instinct_MI300X"\n'
        "        fallback_json = f\"N={N},K={K},device_name={fallback_device_name},dtype=int8_w8a8,block_shape=[{block_n}, {block_k}].json\"\n"
        "        fallback_path = os.path.join(\n"
        '            os.path.dirname(os.path.realpath(__file__)), "configs", fallback_json\n'
        "        )\n"
        "        if os.path.exists(fallback_path):\n"
        "            config_file_path = fallback_path"
    )
    return txt.replace(target, replacement)


def main():
    print("=" * 60)
    print("patch_vllm.py - R9700 (gfx1201) build-time patches")
    print("=" * 60)

    _patch("vllm/platforms/__init__.py", "force ROCm, bypass amdsmi", patch_1_init_py)

    def patch_rocm_combined(txt):
        txt = patch_2_rocm_py_mock(txt)
        txt = patch_4_gcn_arch(txt)
        txt = patch_6_on_mi3xx(txt)
        return txt
    _patch("vllm/platforms/rocm.py", "device name + GCN arch + MI3XX gate", patch_rocm_combined)

    _patch(
        "vllm/transformers_utils/config.py",
        "fix GenerationConfig imports",
        patch_3_transformers_config,
    )
    _patch("csrc/spinloop.cpp", "fix mwaitxintrin.h for ROCm Clang", patch_5_spinloop)
    _patch("vllm/_aiter_ops.py", "map gfx1201 to MI350X", patch_7_aiter_ops)
    _patch(
        "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
        "MI300X FP8 config fallback",
        patch_8_fp8_utils,
    )
    _patch(
        "vllm/model_executor/layers/quantization/utils/int8_utils.py",
        "MI300X INT8 config fallback",
        patch_9_int8_utils,
    )
    _patch(
        "CMakeLists.txt",
        "accept PYTORCH_FOUND_HIP in addition to HIP_FOUND",
        patch_10_hip_found,
    )

    print()
    print(f"Done.  {_OK} patches processed, {_WARN} warnings.")
    if _WARN > 0:
        print("(Warnings are expected if some files are absent in this vLLM version)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

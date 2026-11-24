#!/usr/bin/env python3
"""Turn the repo-root pyproject.toml into a GPU wheel variant.

Usage: python installers/pip/gpu_variant.py {cuda12|rocm6|sycl}

scikit-build-core overrides (keyed on CLIO_WHEEL_GPU in pyproject.toml) pick
the CMake defines, but they cannot change [project].name or dependencies, so
CI runs this on its checkout before cibuildwheel. Each variant is a separate
PyPI distribution that installs the same `iowarp_core` package; installing
two variants into one environment overwrites one with the other, the same
trade-off cupy-cuda12x/cupy-rocm make.
"""
import pathlib
import re
import sys

VARIANTS = {
    # libcudart.so.12 for the shared CUDA runtime the GPU libs link.
    "cuda12": ("iowarp-core-cuda12", ["nvidia-cuda-runtime-cu12>=12.8"]),
    # ROCm has no PyPI runtime; users install ROCm 6.x from AMD.
    "rocm6": ("iowarp-core-rocm6", []),
    # libsycl.so.8 + Unified Runtime from Intel's PyPI wheels (oneAPI 2025.x).
    "sycl": ("iowarp-core-sycl", ["intel-sycl-rt>=2025.0,<2026"]),
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print(f"usage: {sys.argv[0]} {{{'|'.join(VARIANTS)}}}", file=sys.stderr)
        return 2
    name, extra = VARIANTS[sys.argv[1]]
    path = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = path.read_text()
    text, n = re.subn(r'(?m)^name = "iowarp-core"$', f'name = "{name}"', text)
    if n != 1:
        print("error: [project] name line not found", file=sys.stderr)
        return 1
    if extra:
        deps = "".join(f'    "{d}",\n' for d in extra)
        text, n = re.subn(r"(?m)^(dependencies = \[\n)", r"\g<1>" + deps, text)
        if n != 1:
            print("error: dependencies list not found", file=sys.stderr)
            return 1
    path.write_text(text)
    print(f"pyproject.toml -> {name} (+{extra or 'no extra deps'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

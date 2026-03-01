"""CLI entry point for the chimaera command.

Locates the chimaera binary bundled in the wheel and exec's it,
ensuring IOWarp shared libraries are on the library search path.
"""

import os
import sys


def main():
    """Execute the chimaera binary."""
    # In scikit-build-core editable installs, __file__ resolves to the
    # workspace source tree, but cmake-built artifacts (binary, .so libs)
    # live in site-packages/iowarp_core/.  Anchor to site-packages by
    # using the __file__ of a cmake-built extension module (.so), which the
    # editable finder always serves from site-packages (known_wheel_files).
    bin_path = None
    lib_dir = None
    for _extmod in ("wrp_cte_core_ext", "wrp_cee"):
        try:
            import importlib as _il
            _m = _il.import_module(_extmod)
            # .so lives at site-packages/<name>.so (top-level module).
            # iowarp_core/bin/chimaera is a sibling of that .so file.
            _sp = os.path.dirname(os.path.abspath(_m.__file__))
            _candidate = os.path.join(_sp, "iowarp_core", "bin", "chimaera")
            if os.path.exists(_candidate):
                bin_path = _candidate
                lib_dir = os.path.join(_sp, "iowarp_core", "lib")
                break
        except (ImportError, AttributeError):
            continue

    if bin_path is None:
        # Fallback: __file__-based path works for regular (non-editable) installs
        # where _cli.py itself lives inside site-packages/iowarp_core/.
        package_dir = os.path.dirname(os.path.abspath(__file__))
        bin_path = os.path.join(package_dir, "bin", "chimaera")
        lib_dir = os.path.join(package_dir, "lib")

    if not os.path.exists(bin_path):
        print("Error: chimaera binary not found at", bin_path, file=sys.stderr)
        sys.exit(1)

    # Ensure IOWarp libs are on the library path
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = (
            lib_dir + ":" + ld_path if ld_path else lib_dir
        )

    # Replace this process with the chimaera binary
    os.execve(bin_path, [bin_path] + sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()

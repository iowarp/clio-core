"""CLI entry points for bundled IOWarp binaries.

Locates a named binary bundled in the wheel (under ``iowarp_core/bin/``)
and exec's it after ensuring IOWarp shared libraries are on the library
search path.  Each ``[project.scripts]`` entry in pyproject.toml routes
to a thin wrapper around ``_exec_iowarp_bin``.
"""

import importlib
import os
import sys

_IS_WINDOWS = sys.platform == "win32"
_BIN_SUFFIX = ".exe" if _IS_WINDOWS else ""


def _find_bin(bin_dir, name):
    """Return path to ``<bin_dir>/<name>`` honoring the platform exe suffix."""
    candidate = os.path.join(bin_dir, name + _BIN_SUFFIX)
    if os.path.exists(candidate):
        return candidate
    # Also accept already-suffixed names in case callers pass `chimaera.exe`.
    if _BIN_SUFFIX and not name.endswith(_BIN_SUFFIX):
        alt = os.path.join(bin_dir, name)
        if os.path.exists(alt):
            return alt
    return None


def _resolve_paths(name):
    """Return ``(bin_path, bin_dir, lib_dir)`` for the named binary.

    In scikit-build-core editable installs, ``__file__`` resolves to the
    workspace source tree, but cmake-built artifacts (binaries, shared libs)
    live in ``site-packages/iowarp_core/``.  Anchor to site-packages by
    importing a cmake-built extension module — the editable finder always
    serves those from site-packages — and walking from its on-disk location.
    """
    for extmod in ("clio_cte_core_ext", "clio_cee"):
        try:
            mod = importlib.import_module(extmod)
        except (ImportError, AttributeError):
            continue
        sp = os.path.dirname(os.path.abspath(mod.__file__))
        bin_dir = os.path.join(sp, "iowarp_core", "bin")
        bin_path = _find_bin(bin_dir, name)
        if bin_path is not None:
            return bin_path, bin_dir, os.path.join(sp, "iowarp_core", "lib")

    # Fallback for regular (non-editable) installs where _cli.py lives
    # inside site-packages/iowarp_core/ itself.
    package_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(package_dir, "bin")
    return _find_bin(bin_dir, name), bin_dir, os.path.join(package_dir, "lib")


def _prepare_env(bin_dir, lib_dir):
    """Make sibling DLLs / .so deps discoverable to the spawned binary."""
    if _IS_WINDOWS:
        # On Windows, dependent DLLs are searched on PATH (DefaultDllImportSearchPaths
        # for the spawned process). Prepend our bin/ + lib/ so loader finds them.
        # LD_LIBRARY_PATH does nothing on Win32.
        sep = os.pathsep
        existing = os.environ.get("PATH", "")
        prefix_parts = [p for p in (bin_dir, lib_dir) if os.path.isdir(p)]
        if prefix_parts:
            os.environ["PATH"] = sep.join(prefix_parts + ([existing] if existing else []))
    else:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir and lib_dir not in existing:
            os.environ["LD_LIBRARY_PATH"] = (
                lib_dir + ":" + existing if existing else lib_dir
            )


def _exec_iowarp_bin(name):
    """Find the named binary in the wheel's bin/ and execute it."""
    bin_path, bin_dir, lib_dir = _resolve_paths(name)
    if bin_path is None:
        suffix = _BIN_SUFFIX or "(no suffix)"
        print(
            f"Error: {name}{suffix} binary not found under {bin_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    _prepare_env(bin_dir, lib_dir)

    if _IS_WINDOWS:
        # os.execv on Windows doesn't truly replace the process — the parent
        # exits immediately and the child detaches, which breaks pipes and
        # exit codes for callers (pip console-script wrappers, shells). Use
        # subprocess so we can propagate the child's exit code reliably.
        import subprocess
        proc = subprocess.run([bin_path] + sys.argv[1:])
        sys.exit(proc.returncode)
    else:
        os.execve(bin_path, [bin_path] + sys.argv[1:], os.environ)


def main():
    """Entry point for the ``chimaera`` console script."""
    _exec_iowarp_bin("chimaera")


def cte_bench_main():
    """Entry point for the ``clio_cte_bench`` console script."""
    _exec_iowarp_bin("clio_cte_bench")


def run_thrpt_main():
    """Entry point for the ``clio_run_thrpt_bench`` console script."""
    _exec_iowarp_bin("clio_run_thrpt_bench")


if __name__ == "__main__":
    main()

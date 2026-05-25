"""CLI entry points for bundled IOWarp binaries.

Locates a named binary under ``iowarp_core/bin/`` (auto-adding the
``.exe`` suffix on Windows), wires up the platform's shared-library
search path, and runs it. Each ``[project.scripts]`` entry in
pyproject.toml routes to a thin wrapper around ``_exec_iowarp_bin``.
"""

import importlib
import os
import subprocess
import sys


def _locate_bin(name):
    """Return (bin_path, bin_dir, lib_dir) for the named binary.

    Tries the site-packages route via an installed cmake-built ext
    module first (works around editable installs where ``__file__``
    points at the source tree), then falls back to the directory of
    this _cli.py file (normal wheel installs).
    """
    exe = name + (".exe" if sys.platform == "win32" else "")

    for _extmod in ("clio_cte_core_ext", "clio_cee"):
        try:
            _m = importlib.import_module(_extmod)
            _sp = os.path.dirname(os.path.abspath(_m.__file__))
            _candidate = os.path.join(_sp, "iowarp_core", "bin", exe)
            if os.path.exists(_candidate):
                return (
                    _candidate,
                    os.path.dirname(_candidate),
                    os.path.join(_sp, "iowarp_core", "lib"),
                )
        except (ImportError, AttributeError):
            continue

    # Fallback: regular wheel install where _cli.py is inside
    # site-packages/iowarp_core/ itself.
    package_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(package_dir, "bin", exe)
    return candidate, os.path.join(package_dir, "bin"), os.path.join(package_dir, "lib")


def _exec_iowarp_bin(name):
    """Find ``iowarp_core/bin/<name>[.exe]``, set the platform's
    dynamic-loader search path, and run it.
    """
    bin_path, bin_dir, lib_dir = _locate_bin(name)
    if not os.path.exists(bin_path):
        exe = name + (".exe" if sys.platform == "win32" else "")
        print(f"Error: {exe} binary not found at {bin_path}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    if sys.platform == "win32":
        # CMake places .dll files in bin/ next to .exe files on Windows,
        # so the standard DLL search (which checks the .exe's own dir
        # first) handles the IOWarp-owned DLLs. Third-party vcpkg DLLs
        # (yaml-cpp, libzmq, libsodium, msgpack) get bundled by
        # delvewheel into a sibling `iowarp_core.libs/` directory at
        # site-packages root — find it and prepend to PATH so the
        # spawned chimaera.exe / *_bench.exe can resolve those imports
        # too. (Python imports of .pyd modules use the os.add_dll_directory
        # registration that delvewheel installs into __init__.py, which
        # doesn't propagate to subprocess children.)
        search_dirs = [bin_dir, lib_dir]
        pkg_dir = os.path.dirname(bin_dir)  # site-packages/iowarp_core
        sp_root = os.path.dirname(pkg_dir)  # site-packages
        delvewheel_libs = os.path.join(sp_root, "iowarp_core.libs")
        if os.path.isdir(delvewheel_libs):
            search_dirs.append(delvewheel_libs)
        existing = env.get("PATH", "")
        env["PATH"] = os.pathsep.join(
            search_dirs + [existing] if existing else search_dirs
        )
        # os.execve has odd Windows semantics: cmd.exe doesn't really
        # see the replaced process, argv quoting goes through MSVCRT,
        # and the parent prompt returns control immediately. Use
        # subprocess.run and forward the exit code instead.
        result = subprocess.run([bin_path] + sys.argv[1:], env=env)
        sys.exit(result.returncode)

    if sys.platform == "darwin":
        env_var = "DYLD_LIBRARY_PATH"
    else:
        env_var = "LD_LIBRARY_PATH"
    existing = env.get(env_var, "")
    env[env_var] = (
        lib_dir + os.pathsep + existing if existing else lib_dir
    )
    os.execve(bin_path, [bin_path] + sys.argv[1:], env)


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

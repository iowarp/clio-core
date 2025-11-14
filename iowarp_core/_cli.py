"""
CLI wrapper for IOWarp Core binaries.

This module provides entry points that execute the bundled IOWarp binaries
with the correct library paths set.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_package_paths():
    """Get the paths to the package lib and bin directories."""
    # Get the directory where this module is installed
    module_dir = Path(__file__).parent.resolve()

    lib_dir = module_dir / "lib"
    bin_dir = module_dir / "bin"

    return lib_dir, bin_dir


def run_binary(binary_name):
    """
    Run a bundled or system-installed binary with the correct library paths.

    Args:
        binary_name: Name of the binary to execute
    """
    lib_dir, bin_dir = get_package_paths()
    binary_path = bin_dir / binary_name

    # Check if binary is bundled in package directory
    if not binary_path.exists():
        # Fall back to system-installed binary (from source distribution)
        import shutil
        system_binary = shutil.which(binary_name)
        if system_binary:
            binary_path = Path(system_binary)
        else:
            print(f"Error: Binary '{binary_name}' not found", file=sys.stderr)
            print(f"  - Not bundled at: {bin_dir / binary_name}", file=sys.stderr)
            print(f"  - Not found in PATH", file=sys.stderr)
            print(f"Make sure iowarp-core is installed correctly.", file=sys.stderr)
            sys.exit(1)

    # Set up environment with library path
    env = os.environ.copy()

    # Build library path including conda environment if available
    lib_paths = []

    # Add bundled lib directory if it exists (for bundled binary distributions)
    if lib_dir.exists():
        lib_paths.append(str(lib_dir))

    # Add conda lib directory for dependencies (HDF5, MPI, etc.)
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = Path(conda_prefix) / "lib"
        if conda_lib.exists():
            lib_paths.append(str(conda_lib))

    # Add lib directory to library path (only if we have paths to add)
    if lib_paths:
        if sys.platform.startswith("linux"):
            ld_library_path = env.get("LD_LIBRARY_PATH", "")
            new_paths = ":".join(lib_paths)
            if ld_library_path:
                env["LD_LIBRARY_PATH"] = f"{new_paths}:{ld_library_path}"
            else:
                env["LD_LIBRARY_PATH"] = new_paths
        elif sys.platform == "darwin":
            dyld_library_path = env.get("DYLD_LIBRARY_PATH", "")
            new_paths = ":".join(lib_paths)
            if dyld_library_path:
                env["DYLD_LIBRARY_PATH"] = f"{new_paths}:{dyld_library_path}"
            else:
                env["DYLD_LIBRARY_PATH"] = new_paths

    # Execute the binary with arguments passed from command line
    try:
        result = subprocess.run(
            [str(binary_path)] + sys.argv[1:],
            env=env,
            check=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error executing {binary_name}: {e}", file=sys.stderr)
        sys.exit(1)


def wrp_cae_omni():
    """Entry point for wrp_cae_omni binary."""
    run_binary("wrp_cae_omni")


def wrp_cte():
    """Entry point for wrp_cte binary (Hermes)."""
    run_binary("wrp_cte")


def wrp_runtime():
    """Entry point for wrp_runtime binary (Chimaera)."""
    run_binary("wrp_runtime")


def chi_refresh_repo():
    """Entry point for chi_refresh_repo binary."""
    run_binary("chi_refresh_repo")


def chimaera_compose():
    """Entry point for chimaera_compose binary."""
    run_binary("chimaera_compose")


def chimaera_start_runtime():
    """Entry point for chimaera_start_runtime binary."""
    run_binary("chimaera_start_runtime")


def chimaera_stop_runtime():
    """Entry point for chimaera_stop_runtime binary."""
    run_binary("chimaera_stop_runtime")


def test_binary_assim():
    """Entry point for test_binary_assim binary."""
    run_binary("test_binary_assim")


def test_error_handling():
    """Entry point for test_error_handling binary."""
    run_binary("test_error_handling")


def test_hdf5_assim():
    """Entry point for test_hdf5_assim binary."""
    run_binary("test_hdf5_assim")


def test_range_assim():
    """Entry point for test_range_assim binary."""
    run_binary("test_range_assim")

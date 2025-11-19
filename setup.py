#!/usr/bin/env python3
"""
Setup script for iowarp-core package.
Builds and installs C++ components using CMake in the correct order.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.dist import Distribution
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class CMakeExtension(Extension):
    """Extension class for CMake-based C++ projects."""

    def __init__(self, name, sourcedir="", repo_url="", **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)
        self.repo_url = repo_url


class CustomSDist(sdist):
    """Custom sdist command that ensures git submodules are fully included."""

    def run(self):
        """Ensure git submodules are initialized before creating source distribution."""
        # Check if we're in a git repository
        if os.path.exists(".git"):
            print("\n" + "="*60)
            print("Ensuring git submodules are included in source distribution")
            print("="*60 + "\n")

            try:
                # Initialize and update submodules to ensure they're present
                subprocess.check_call(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                print("✓ Git submodules initialized successfully")

                # List submodules to verify
                result = subprocess.run(
                    ["git", "submodule", "status"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.stdout:
                    print("\nSubmodules found:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"  {line}")

                print("\nNote: MANIFEST.in will include submodule files in the tarball")
                print("")

            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not initialize git submodules: {e}")
                print("Source distribution may be incomplete!")
        else:
            print("Not a git repository - skipping submodule initialization")

        # Call parent sdist command to create the distribution
        super().run()


class CMakeBuild(build_ext):
    """Custom build command that builds IOWarp core using CMake presets."""

    # Single repository for all components
    REPO_URL = "https://github.com/iowarp/core"

    def run(self):
        """Build IOWarp core following the quick installation steps."""
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build iowarp-core. "
                "Install with: pip install cmake"
            )

        # Create build directory
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Build the unified core
        self.build_iowarp_core(build_temp)

    def build_iowarp_core(self, build_temp):
        """Build IOWarp core using install.sh script."""
        print(f"\n{'='*60}")
        print(f"Building IOWarp Core using install.sh")
        print(f"{'='*60}\n")

        # Determine install prefix based on whether we're bundling binaries
        # For IOWarp Core, install directly to the Python environment (sys.prefix)
        # This is the standard approach for packages with C++ libraries that need to be
        # found by CMake and linked by other applications.
        # Libraries → {sys.prefix}/lib/
        # Headers → {sys.prefix}/include/
        # CMake configs → {sys.prefix}/lib/cmake/
        bundle_binaries = os.environ.get("IOWARP_BUNDLE_BINARIES", "OFF").upper() == "ON"
        if bundle_binaries:
            # Install to a staging directory that we'll copy into the wheel
            # (Not recommended for IOWarp - use IOWARP_BUNDLE_BINARIES=ON to enable)
            install_prefix = build_temp / "install"
        else:
            # Install directly to Python environment prefix
            # This ensures libraries/headers are in standard locations
            install_prefix = Path(sys.prefix).absolute()

        print(f"Install prefix: {install_prefix}")

        # Find install.sh in package root
        package_root = Path(__file__).parent.absolute()
        install_script = package_root / "install.sh"

        if not install_script.exists():
            raise RuntimeError(f"install.sh not found at {install_script}")

        # Make install.sh executable
        install_script.chmod(0o755)

        # Prepare environment for install.sh
        env = os.environ.copy()
        env["INSTALL_PREFIX"] = str(install_prefix)
        # INSTALL_PREFIX points to venv root, e.g., /path/to/venv
        # This ensures standard layout:
        #   - Binaries:    {venv}/bin/
        #   - Libraries:   {venv}/lib/
        #   - Headers:     {venv}/include/
        #   - CMake:       {venv}/lib/cmake/
        # Python bindings install to site-packages via nanobind

        # Determine number of parallel jobs
        if hasattr(self, "parallel") and self.parallel:
            env["BUILD_JOBS"] = str(self.parallel)
        else:
            import multiprocessing
            env["BUILD_JOBS"] = str(multiprocessing.cpu_count())

        print(f"\nRunning install.sh with:")
        print(f"  INSTALL_PREFIX={env['INSTALL_PREFIX']}")
        print(f"  BUILD_JOBS={env['BUILD_JOBS']}\n")

        # Run install.sh and capture output for debugging
        result = subprocess.run(
            [str(install_script)],
            cwd=package_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Print all install.sh output
        if result.stdout:
            print(result.stdout)

        # Check for errors
        if result.returncode != 0:
            print(f"\nERROR: install.sh failed with exit code {result.returncode}\n")
            sys.exit(result.returncode)

        print(f"\nIOWarp core built and installed successfully!\n")

        # If bundling is enabled, copy binaries to package
        if bundle_binaries:
            self.copy_binaries_to_package(build_temp)

    def copy_binaries_to_package(self, build_temp):
        """Copy built binaries and headers into the Python package for wheel bundling."""
        print("\n" + "="*60)
        print("Copying binaries to package directory")
        print("="*60 + "\n")

        install_prefix = build_temp / "install"
        package_dir = Path(self.build_lib) / "iowarp_core"

        # Create directories in the package
        lib_dir = package_dir / "lib"
        include_dir = package_dir / "include"
        bin_dir = package_dir / "bin"

        lib_dir.mkdir(parents=True, exist_ok=True)
        include_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)

        # Copy libraries
        src_lib_dir = install_prefix / "lib"
        if src_lib_dir.exists():
            print(f"Copying libraries from {src_lib_dir} to {lib_dir}")
            for lib_file in src_lib_dir.rglob("*"):
                if lib_file.is_file() or lib_file.is_symlink():
                    # Copy .so, .a, and .dylib files
                    if lib_file.suffix in [".so", ".a", ".dylib"] or ".so." in lib_file.name:
                        rel_path = lib_file.relative_to(src_lib_dir)
                        dest = lib_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        # Remove existing file/symlink to avoid conflicts
                        if dest.exists() or dest.is_symlink():
                            dest.unlink()
                        # Copy file or symlink
                        if lib_file.is_symlink():
                            os.symlink(os.readlink(lib_file), dest)
                        else:
                            shutil.copy2(lib_file, dest)
                        print(f"  Copied: {rel_path}")

        # Copy lib64 if it exists (some systems use lib64)
        src_lib64_dir = install_prefix / "lib64"
        if src_lib64_dir.exists():
            print(f"Copying libraries from {src_lib64_dir} to {lib_dir}")
            for lib_file in src_lib64_dir.rglob("*"):
                if lib_file.is_file() or lib_file.is_symlink():
                    if lib_file.suffix in [".so", ".a", ".dylib"] or ".so." in lib_file.name:
                        rel_path = lib_file.relative_to(src_lib64_dir)
                        dest = lib_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        # Remove existing file/symlink to avoid conflicts
                        if dest.exists() or dest.is_symlink():
                            dest.unlink()
                        # Copy file or symlink
                        if lib_file.is_symlink():
                            os.symlink(os.readlink(lib_file), dest)
                        else:
                            shutil.copy2(lib_file, dest)
                        print(f"  Copied: {rel_path}")

        # Copy headers
        src_include_dir = install_prefix / "include"
        if src_include_dir.exists():
            print(f"Copying headers from {src_include_dir} to {include_dir}")
            for header_file in src_include_dir.rglob("*"):
                if header_file.is_file():
                    rel_path = header_file.relative_to(src_include_dir)
                    dest = include_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(header_file, dest)

        # Copy binaries/executables
        src_bin_dir = install_prefix / "bin"
        if src_bin_dir.exists():
            print(f"Copying binaries from {src_bin_dir} to {bin_dir}")
            # Binaries to exclude from distribution (test executables)
            exclude_binaries = {
                "test_binary_assim",
                "test_error_handling",
                "test_hdf5_assim",
                "test_range_assim",
            }
            for bin_file in src_bin_dir.rglob("*"):
                if bin_file.is_file():
                    # Skip test binaries
                    if bin_file.name in exclude_binaries:
                        print(f"  Skipped test binary: {bin_file.name}")
                        continue

                    rel_path = bin_file.relative_to(src_bin_dir)
                    dest = bin_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bin_file, dest)
                    # Make executable
                    dest.chmod(dest.stat().st_mode | 0o111)
                    print(f"  Copied: {rel_path}")

        # Copy conda dependencies
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            print(f"\n" + "="*60)
            print("Copying conda dependencies")
            print("="*60 + "\n")

            conda_lib_dir = Path(conda_prefix) / "lib"
            if conda_lib_dir.exists():
                # List of library patterns to copy (dependencies needed by IOWarp)
                lib_patterns = [
                    "libboost_*.so*",
                    "libhdf5*.so*",
                    "libmpi*.so*",
                    "libzmq*.so*",
                    "libsodium.so*",  # Required by ZeroMQ
                    "libyaml*.so*",  # Note: libyaml-cpp is built from source (see skip_libs below)
                    "libz.so*",
                    "libsz.so*",
                    "libaec.so*",
                    "libcurl.so*",
                    "libssh2.so*",  # Required by libcurl
                    "libnghttp2.so*",  # Required by libcurl
                    "libssl.so*",  # OpenSSL SSL library
                    "libcrypto.so*",  # OpenSSL crypto library
                    "libopen-*.so*",  # OpenMPI libraries
                    "libpmix*.so*",  # PMIx for OpenMPI
                    "libhwloc*.so*",  # Hardware locality for MPI
                    "libevent*.so*",  # Event notification library
                    "libfabric*.so*",  # Networking for MPI
                    "libefa.so*",  # Elastic Fabric Adapter (if available)
                    "libpsm*.so*",  # Intel PSM/PSM2 (if available)
                    "libucx*.so*",  # Unified Communication X
                    "libucp*.so*",  # UCX Protocol layer
                    "libucc*.so*",  # Unified Collective Communication
                    "libucs*.so*",  # UCX Services layer
                    "libuct*.so*",  # UCX Transport layer
                    "libucm*.so*",  # UCX Memory layer
                    "libicu*.so*",  # ICU (International Components for Unicode)
                    "libnuma*.so*",  # NUMA support
                    "librdmacm.so*",  # RDMA connection manager
                    "libibverbs.so*",  # InfiniBand verbs
                    "libstdc++.so*",
                    "libgcc_s.so*",
                    "libgfortran.so*",
                    "libquadmath.so*",
                ]

                copied_libs = set()
                # Libraries we build from source and should not copy from conda
                skip_libs = {
                    "libyaml-cpp.so",
                    "libyaml-cpp.so.0.8",
                    "libyaml-cpp.so.0.8.0",
                }

                for pattern in lib_patterns:
                    for lib_file in conda_lib_dir.glob(pattern):
                        lib_name = lib_file.name

                        # Skip libraries we build from source
                        if lib_name in skip_libs:
                            print(f"  Skipping conda {lib_name} (using built-from-source version)")
                            continue

                        if lib_file.is_file() and not lib_file.is_symlink():
                            if lib_name not in copied_libs:
                                dest = lib_dir / lib_name
                                shutil.copy2(lib_file, dest)
                                copied_libs.add(lib_name)
                                print(f"  Copied conda dependency: {lib_name}")
                        elif lib_file.is_symlink():
                            # Copy symlinks as well
                            if lib_name not in copied_libs:
                                dest = lib_dir / lib_name
                                # Remove existing file/symlink to avoid conflicts
                                if dest.exists() or dest.is_symlink():
                                    dest.unlink()
                                target = lib_file.readlink()
                                # If target is relative, keep it relative
                                if not target.is_absolute():
                                    dest.symlink_to(target)
                                else:
                                    # If absolute, just copy the file it points to
                                    shutil.copy2(lib_file, dest)
                                copied_libs.add(lib_name)
                                print(f"  Copied conda dependency (symlink): {lib_name}")

                print(f"\nTotal conda dependencies copied: {len(copied_libs)}")

        # Note: RPATH is set to absolute paths by CMake at build time
        # - CMake sets absolute RPATH (${CMAKE_INSTALL_PREFIX}/lib) for source builds
        # - This is intended for source-only builds, not relocatable binary distributions
        # - If creating relocatable packages, RPATH would need to be fixed post-build
        print("\nBinary copying complete!")
        print("Note: Libraries have absolute RPATH set by CMake build system\n")


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform-specific tags."""

    def has_ext_modules(self):
        """Always return True to indicate this is a platform-specific package.

        This forces setuptools to generate platform-specific wheel tags instead of
        pure-python 'any' tags. Required for packages with compiled C/C++ extensions.
        """
        return True


class CustomBdistWheel(_bdist_wheel):
    """Custom bdist_wheel that automatically generates manylinux tags.

    This eliminates the need for post-build wheel renaming scripts and ensures
    proper platform tags are set during the build process.
    """

    def finalize_options(self):
        """Override to set platform-specific wheel tags based on the target system."""
        super().finalize_options()

        # Only apply manylinux tags on Linux
        if platform.system() == 'Linux':
            machine = platform.machine()

            # Map machine architecture to manylinux platform tag
            # manylinux_2_17 is compatible with most modern Linux systems (RHEL 7+, Ubuntu 16.04+)
            if machine == 'x86_64':
                self.plat_name = 'manylinux_2_17_x86_64'
            elif machine == 'aarch64':
                self.plat_name = 'manylinux_2_17_aarch64'
            elif machine == 'ppc64le':
                self.plat_name = 'manylinux_2_17_ppc64le'
            elif machine == 's390x':
                self.plat_name = 'manylinux_2_17_s390x'
            else:
                # For unknown architectures, use the generic linux tag
                print(f"Warning: Unknown architecture {machine}, using generic linux tag")
                self.plat_name = f'linux_{machine}'


# Create extensions list
# Always include the CMake build extension so that source distributions work correctly.
# The IOWARP_BUNDLE_BINARIES flag controls whether binaries are bundled into the wheel
# or installed to the system prefix (for source installs).
ext_modules = [
    CMakeExtension(
        "iowarp_core._native",
        sourcedir=".",
    )
]
cmdclass = {
    "build_ext": CMakeBuild,
    "sdist": CustomSDist,
    "bdist_wheel": CustomBdistWheel,
}


if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        distclass=BinaryDistribution,
    )

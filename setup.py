#!/usr/bin/env python3
"""
Setup script for iowarp-core package.
Builds and installs C++ components using CMake in the correct order.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class CMakeExtension(Extension):
    """Extension class for CMake-based C++ projects."""

    def __init__(self, name, sourcedir="", repo_url="", **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)
        self.repo_url = repo_url


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

        # If bundling binaries, copy them to the package directory
        if os.environ.get("IOWARP_BUNDLE_BINARIES", "OFF").upper() == "ON":
            self.copy_binaries_to_package(build_temp)

    def build_iowarp_core(self, build_temp):
        """Build IOWarp core using CMake presets from included source or clone."""
        print(f"\n{'='*60}")
        print(f"Building IOWarp Core")
        print(f"{'='*60}\n")

        # Set up directories
        source_dir = build_temp / "iowarp-core"
        build_dir = source_dir / "build"

        # Determine install prefix based on whether we're bundling binaries
        bundle_binaries = os.environ.get("IOWARP_BUNDLE_BINARIES", "ON").upper() == "ON"
        if bundle_binaries:
            # Install to a staging directory that we'll copy into the wheel
            install_prefix = build_temp / "install"
        else:
            # Install to system prefix (for editable installs)
            install_prefix = Path(sys.prefix).absolute()

        # Check if source is in the package root directory (direct copy)
        package_root = Path(__file__).parent.absolute()
        root_cmake = package_root / "CMakeLists.txt"

        if not source_dir.exists():
            if root_cmake.exists():
                # Use source from package root (direct copy approach)
                print(f"Using C++ source from package root: {package_root}")
                print(f"Copying source to build directory...")
                # Copy all C++ source directories
                for item in ["CMakeLists.txt", "CMakePresets.json", "context-runtime",
                           "context-transfer-engine", "context-assimilation-engine",
                           "context-transport-primitives", "context-exploration-engine",
                           "external", "docker", "ai-prompts"]:
                    src_path = package_root / item
                    if src_path.exists():
                        dest_path = source_dir / item
                        if src_path.is_dir():
                            shutil.copytree(src_path, dest_path, symlinks=True, dirs_exist_ok=True)
                        else:
                            source_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_path, dest_path)
                print(f"Source copied to {source_dir}")
            else:
                # Clone repository (fallback for old source distributions)
                print(f"Source not found in package root, cloning {self.REPO_URL}...")
                subprocess.check_call(["git", "clone", "--recursive", self.REPO_URL, str(source_dir)])
        else:
            print(f"Using existing source at {source_dir}")

        # Determine build type
        build_type = os.environ.get("IOWARP_BUILD_TYPE", "release").lower()
        preset = f"{build_type}"

        print(f"Configuring with CMake preset: {preset}")

        # Configure using CMake preset
        cmake_preset_args = [
            "cmake",
            f"--preset={preset}",
        ]

        # Additional configuration options
        additional_args = []

        # Override install prefix
        additional_args.append(f"-DCMAKE_INSTALL_PREFIX={install_prefix}")

        # Set RPATH for bundled binaries to find their libraries
        if bundle_binaries:
            # Disable building tests for distribution builds
            additional_args.append("-DWRP_CORE_ENABLE_TESTS=OFF")

            rpaths = []
            if sys.platform.startswith("linux"):
                rpaths.append("$ORIGIN/../lib")
                conda_prefix = os.environ.get("CONDA_PREFIX")
                if conda_prefix:
                    rpaths.append(f"{conda_prefix}/lib")
            elif sys.platform == "darwin":
                rpaths.append("@loader_path/../lib")
                conda_prefix = os.environ.get("CONDA_PREFIX")
                if conda_prefix:
                    rpaths.append(f"{conda_prefix}/lib")

            if rpaths:
                rpath = ":".join(rpaths) if sys.platform.startswith("linux") else ";".join(rpaths)
                additional_args.extend([
                    f"-DCMAKE_INSTALL_RPATH={rpath}",
                    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE",
                    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE",
                ])

        # Add HDF5_ROOT to use conda's HDF5 if available
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            # Explicitly ignore paths that might have incompatible HDF5
            home_dir = str(Path.home())
            hdf5_lib = f"{conda_prefix}/lib/libhdf5.so"
            additional_args.extend([
                f"-DHDF5_ROOT={conda_prefix}",
                f"-DHDF5_C_LIBRARY={hdf5_lib}",
                f"-DHDF5_INCLUDE_DIR={conda_prefix}/include",
                f"-DCMAKE_PREFIX_PATH={conda_prefix}",
                f"-DCMAKE_IGNORE_PATH=/usr/lib/x86_64-linux-gnu/hdf5;{home_dir}/hdf5-install;{home_dir}/hdf5;/opt/hdf5",
                "-DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=FALSE",
            ])

        # Apply additional CMake arguments if preset configuration needs overrides
        if additional_args:
            # First configure with preset
            subprocess.check_call(cmake_preset_args, cwd=source_dir)
            # Then apply additional configuration
            cmake_config_args = ["cmake", "build"] + additional_args
            print(f"Applying additional configuration: {' '.join(additional_args)}")
            subprocess.check_call(cmake_config_args, cwd=source_dir)
        else:
            subprocess.check_call(cmake_preset_args, cwd=source_dir)

        # Build
        print(f"\nBuilding IOWarp core...")
        build_args = ["cmake", "--build", "build"]

        # Determine number of parallel jobs
        if hasattr(self, "parallel") and self.parallel:
            build_args.extend(["--parallel", str(self.parallel)])
        else:
            # Use all available cores
            import multiprocessing
            build_args.extend(["--parallel", str(multiprocessing.cpu_count())])

        subprocess.check_call(build_args, cwd=source_dir)

        # Install
        print(f"\nInstalling IOWarp core...")
        install_args = ["cmake", "--install", "build", "--prefix", str(install_prefix)]
        subprocess.check_call(install_args, cwd=source_dir)

        print(f"\nIOWarp core built and installed successfully!\n")

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

        # Fix RPATH in all bundled libraries to prefer bundled dependencies
        print(f"\n" + "="*60)
        print("Fixing RPATH in bundled libraries")
        print("="*60 + "\n")

        # Check if patchelf is available
        patchelf_available = True
        try:
            subprocess.run(["patchelf", "--version"],
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  Warning: patchelf not found, skipping RPATH fixes")
            print("  Libraries may fail to find bundled dependencies")
            patchelf_available = False

        if patchelf_available:
            # Fix RPATH for all .so files in lib directory
            for lib_file in lib_dir.rglob("*.so*"):
                if lib_file.is_file() and not lib_file.is_symlink():
                    try:
                        # Set RPATH to look in same directory first
                        # $ORIGIN means the directory containing the library
                        subprocess.run([
                            "patchelf",
                            "--set-rpath", "$ORIGIN:$ORIGIN/..:$ORIGIN/../lib",
                            "--force-rpath",
                            str(lib_file)
                        ], capture_output=True, check=True)
                        print(f"  Fixed RPATH: {lib_file.name}")
                    except subprocess.CalledProcessError as e:
                        # Some files may not be ELF files or may not have RPATH
                        # This is fine, just skip them
                        pass

            # Fix RPATH for binaries
            for bin_file in bin_dir.rglob("*"):
                if bin_file.is_file() and not bin_file.is_symlink():
                    try:
                        # Set RPATH to look in ../lib
                        subprocess.run([
                            "patchelf",
                            "--set-rpath", "$ORIGIN/../lib",
                            "--force-rpath",
                            str(bin_file)
                        ], capture_output=True, check=True)
                        print(f"  Fixed RPATH: bin/{bin_file.name}")
                    except subprocess.CalledProcessError:
                        pass

        print("\nBinary copying complete!\n")


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
}


if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )

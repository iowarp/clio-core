# IOWarp Core - vcpkg Integration

This directory contains the vcpkg port definition for IOWarp Core. There are two ways to use vcpkg with this project:

1. **For developers** — build IOWarp Core from source with dependencies provided by vcpkg
2. **For consumers** — install IOWarp Core as a vcpkg package into your own project

## For Developers

This directory contains the `vcpkg.json` manifest that declares build dependencies. The Windows CMake presets point here via `VCPKG_MANIFEST_DIR`. When you configure with a vcpkg-enabled CMake preset, vcpkg automatically installs the dependencies into `build/vcpkg_installed/` — no manual `vcpkg install` step is needed.

### Prerequisites

- [vcpkg](https://github.com/microsoft/vcpkg) installed with the `VCPKG_ROOT` environment variable set
- Visual Studio 2022 with the "Desktop development with C++" workload (Windows)

### Quick Start (Windows)

```powershell
# Configure — vcpkg installs dependencies automatically
cmake --preset=windows-debug

# Build
cmake --build build --config Debug

# Run tests
cd build && ctest -C Debug
```

### How It Works

The CMake presets (`windows-debug`, `windows-release`) set `toolchainFile` to the vcpkg toolchain and `VCPKG_MANIFEST_DIR` to `installers/vcpkg/`. When CMake runs, the toolchain reads the manifest and installs the listed dependencies before configuration proceeds.

This mirrors how Conda provides dependencies in the Linux DevContainer — developers get a one-command setup without manually installing each library.

### Dependencies (`installers/vcpkg/vcpkg.json`)

- zeromq
- yaml-cpp
- cereal
- hdf5
- catch2

## For Consumers

To install IOWarp Core as a library in your own project, use the overlay port in this directory.

### Install from Local Overlay

```bash
# Clone the IOWarp Core repository
git clone https://github.com/iowarp/core.git
cd core

# Install using vcpkg with overlay
vcpkg install iowarp-core --overlay-ports=installers/vcpkg
```

### Use in CMake

After installation, add to your `CMakeLists.txt`:

```cmake
find_package(iowarp-core CONFIG REQUIRED)

target_link_libraries(your_target PRIVATE
    chimaera::admin_client
    chimaera::bdev_client
    wrp_cte::core_client
)
```

### Port Dependencies

The overlay port (`installers/vcpkg/vcpkg.json`) installs:
- vcpkg-cmake, vcpkg-cmake-config (build tools)
- zeromq, yaml-cpp, cereal, hdf5, catch2

Note: The full overlay port currently only builds on Linux/macOS. On Windows, use the developer workflow described above.

## License

BSD-3-Clause (see LICENSE file in repository root)

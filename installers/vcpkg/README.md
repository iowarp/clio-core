# IOWarp Core - vcpkg Port

This directory contains the vcpkg port definition for IOWarp Core.

## Installing via vcpkg

### Option 1: Install from vcpkg registry (future)

Once IOWarp Core is added to the vcpkg registry:

```bash
vcpkg install iowarp-core
```

### Option 2: Install from local overlay

To install from this repository:

```bash
# Clone the IOWarp Core repository
git clone https://github.com/iowarp/core.git
cd core

# Install using vcpkg with overlay
vcpkg install iowarp-core --overlay-ports=installers/vcpkg
```

## Using in CMake Projects

After installation, use in your CMakeLists.txt:

```cmake
find_package(iowarp-core CONFIG REQUIRED)

target_link_libraries(your_target PRIVATE
    chimaera::admin_client
    chimaera::bdev_client
    wrp_cte::core_client
)
```

## Dependencies

The port automatically installs all required dependencies:
- boost-fiber
- boost-context
- boost-system
- boost-filesystem
- zeromq
- yaml-cpp
- cereal
- hdf5
- catch2

## Platform Support

- Linux: ✅ Supported
- macOS: ✅ Supported
- Windows: ❌ Not supported (marked as `"supports": "!windows"`)

## Build Options

The vcpkg port builds IOWarp Core with production settings:
- Tests disabled
- Benchmarks disabled
- Python bindings disabled
- Documentation disabled
- Code coverage disabled

## Contributing

To contribute improvements to this vcpkg port, submit a pull request to the IOWarp Core repository.

## License

BSD-3-Clause (see LICENSE file in repository root)

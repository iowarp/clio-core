# IOWarp Core Pip Installer

Build and install IOWarp Core as a self-contained Python wheel.

## Quick Start

### Prerequisites

Install development packages for the external dependencies. On Ubuntu/Debian:

```bash
sudo apt-get install libyaml-cpp-dev libzmq3-dev libaio-dev
```

On RHEL/CentOS/manylinux:

```bash
yum install yaml-cpp-devel yaml-cpp-static zeromq-devel libaio-devel
```

The `pip-release` CMake preset enables `WRP_CORE_STATIC_DEPS=ON`, which links
these dependencies statically so the resulting wheel is self-contained.

### Build and Install

```bash
cd installers/pip
pip install -v .
```

### Verify

```bash
python -c "import iowarp_core; print(iowarp_core.get_version())"
chimaera --help
```

## CI / Wheel Building

The GitHub Actions workflow `.github/workflows/build-pip.yml` uses
[cibuildwheel](https://cibuildwheel.readthedocs.io/) to produce manylinux_2_28
wheels for CPython 3.10-3.13.

External deps are linked statically via `WRP_CORE_STATIC_DEPS=ON` (set in the
`pip-release` preset), so the wheel only contains IOWarp's own `.so` files.

### Manual cibuildwheel

```bash
pip install cibuildwheel
cd installers/pip
CIBW_BUILD="cp312-manylinux_x86_64" \
CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_28 \
CIBW_BEFORE_ALL="yum install -y yaml-cpp-devel yaml-cpp-static zeromq-devel libaio-devel" \
cibuildwheel --output-dir wheelhouse
```

## RPATH Fixup

After building, use `fix_rpaths.sh` to set `$ORIGIN`-relative RPATHs if
scikit-build-core doesn't handle them automatically:

```bash
bash fix_rpaths.sh /path/to/site-packages
```

## Wheel Layout

```
iowarp_core/
  __init__.py        # Library path setup
  _cli.py            # chimaera CLI entry point
  _config.py         # Config file resolution
  lib/               # IOWarp shared libraries (static deps baked in)
  ext/               # Python extension modules
  bin/               # CLI executables
  data/              # Default configuration
```

## Customization

### Build Options

Pass extra CMake args via `scikit-build-core`:

```bash
pip install -v . --config-settings=cmake.args="-DWRP_CORE_ENABLE_CAE=OFF"
```

### Static vs Shared Dependencies

The `pip-release` preset enables `WRP_CORE_STATIC_DEPS=ON`. To use shared
dependencies instead (e.g., for development):

```bash
pip install -v . --config-settings=cmake.define.WRP_CORE_STATIC_DEPS=OFF
```

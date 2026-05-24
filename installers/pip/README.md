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

The `pip-release` CMake preset enables `CLIO_CORE_STATIC_DEPS=ON`, which links
these dependencies statically so the resulting wheel is self-contained.

### Build and Install

```bash
cd installers/pip
pip install -v .
```

### Verify

```bash
python -c "import iowarp_core; print(iowarp_core.get_version())"
clio_run --help
```

### Windows-specific install notes

On Windows the wheel builds against vcpkg-provided dependencies. Build it with:

```powershell
pip wheel . --no-deps --wheel-dir dist `
  --config-settings=cmake.define.CMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  --config-settings=cmake.define.VCPKG_MANIFEST_DIR="$(Resolve-Path installers/vcpkg)" `
  --config-settings=cmake.define.VCPKG_TARGET_TRIPLET=x64-windows `
  --config-settings=cmake.define.CLIO_CORE_STATIC_DEPS=OFF `
  --config-settings=cmake.define.CLIO_CORE_ENABLE_IO_URING=OFF
pip install --force-reinstall --no-deps dist/iowarp_core-*-win_amd64.whl
```

If pip emits a warning that scripts (`clio_run.exe`, `clio_cte_bench.exe`, …)
were installed into a directory that is not on `PATH`, you need to add that
directory yourself — pip refuses to do it for you. This is especially common
with the Windows Store distribution of Python, which forces `--user` installs
into `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.X_…\LocalCache\local-packages\Python3XX\Scripts`.

Either:

1. Add the warned-about Scripts directory to your user `PATH` once
   (`setx PATH "%PATH%;<scripts-dir>"`), open a new shell, and re-run.
2. Or invoke the entry points via `python -c`:
   `python -c "from iowarp_core._cli import clio_run_main; clio_run_main()" --help`.

## CI / Wheel Building

The GitHub Actions workflow `.github/workflows/build-pip.yml` uses
[cibuildwheel](https://cibuildwheel.readthedocs.io/) to produce manylinux_2_28
wheels for CPython 3.10-3.13.

External deps are linked statically via `CLIO_CORE_STATIC_DEPS=ON` (set in the
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
  _cli.py            # clio_run / benchmark CLI entry points
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

The `pip-release` preset enables `CLIO_CORE_STATIC_DEPS=ON`. To use shared
dependencies instead (e.g., for development):

```bash
pip install -v . --config-settings=cmake.define.CLIO_CORE_STATIC_DEPS=OFF
```

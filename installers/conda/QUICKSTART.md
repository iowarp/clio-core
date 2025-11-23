# IOWarp Core Conda - Quick Reference

## One-Command Build & Install

```bash
cd conda && ./conda-local.sh
```

That's it! The script handles everything interactively.

## What the Script Does

1. ✅ Checks conda environment is active
2. ✅ Installs conda-build if needed
3. ✅ Initializes git submodules automatically
4. ✅ Builds the conda package from local source
5. ✅ Asks if you want to install now
6. ✅ Provides troubleshooting if build fails

## Manual Commands

### Build Package

```bash
conda build conda/ -c conda-forge
```

### Install Package

```bash
# Option 1: Use local cache
conda install --use-local iowarp-core

# Option 2: Install from specific file
conda install /path/to/iowarp-core-*.tar.bz2

# Option 3: Create new environment
conda create -n iowarp-env iowarp-core --use-local -c conda-forge
```

### Verify Installation

```bash
# List installed package
conda list iowarp-core

# Test command-line tools
chimaera_start_runtime --help
wrp_cte --help
wrp_cae_omni --help
```

### Uninstall

```bash
conda remove iowarp-core
```

## Prerequisites

- Active conda environment
- Git submodules initialized (script does this automatically)

## Build Time

First build: 10-30 minutes (depending on system)
Subsequent builds: 5-15 minutes (cached dependencies)

## Common Issues

### "No conda environment detected"
```bash
conda activate base  # or your preferred environment
```

### "Submodules not initialized"
```bash
git submodule update --init --recursive
```

### Build fails
```bash
# Try with debug output
conda build conda/ -c conda-forge --debug
```

## Environment Variables

Customize the build by setting environment variables:

```bash
# Enable MPI support
export WRP_CORE_ENABLE_MPI=ON
conda build conda/ -c conda-forge

# Enable compression libraries
export WRP_CORE_ENABLE_COMPRESS=ON
conda build conda/ -c conda-forge

# Enable all tests
export WRP_CORE_ENABLE_TESTS=ON
conda build conda/ -c conda-forge
```

## File Locations

- **Recipe:** `/workspace/conda/meta.yaml`
- **Build script:** `/workspace/conda/build.sh`
- **CMake preset:** `conda` (see `/workspace/CMakePresets.json`)
- **Package cache:** `$CONDA_PREFIX/conda-bld/`

## Getting Help

- Full documentation: `conda/README.md`
- CMake options: See main `CMakeLists.txt`
- Build system guide: See CLAUDE.md section on Conda

## Quick Tips

💡 **Tip 1:** The script automatically uses the "conda" CMake preset optimized for conda builds

💡 **Tip 2:** Building from local source means you can test changes immediately without pushing to GitHub

💡 **Tip 3:** Use `conda build --output` to see where the package will be saved

💡 **Tip 4:** The package includes RPATH so it's relocatable - no LD_LIBRARY_PATH needed

## Next Steps After Install

```bash
# Check installation
conda list iowarp-core

# Run example
chimaera_start_runtime

# Link against IOWarp in your project
# In your CMakeLists.txt:
# find_package(iowarp-core REQUIRED)
# target_link_libraries(your_app chimaera::admin_client)
```

## Rebuilding After Changes

```bash
# After modifying source code, simply rebuild:
cd conda && ./conda-local.sh

# The script will rebuild and ask if you want to reinstall
```

---

**For more detailed information, see `conda/README.md`**

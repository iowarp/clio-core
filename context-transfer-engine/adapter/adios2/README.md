# IOWarp ADIOS2 Adapter

## Overview
This adapter provides integration between IOWarp's Context Transfer Engine (CTE) and ADIOS2 through a plugin engine interface.

## Status
**Implementation Complete - Compilation Blocked by ADIOS2 Installation Issue**

The adapter code has been fully implemented and updated to work with the current CTE API. However, compilation is currently blocked by a broken ADIOS2 installation in the conda environment.

## Build Configuration

### CMake Options
- `CLIO_CTE_ENABLE_ADIOS2_ADAPTER`: Enable/disable ADIOS2 adapter (default: OFF)
- The adapter does NOT require ELF support (it's a plugin, not an interceptor)

### Dependencies
- ADIOS2 (required)
- MPI (optional - adapter will build without MPI support if not available)
- IOWarp CTE core client library

### Building
```bash
cmake --preset=debug -DCLIO_CTE_ENABLE_ADIOS2_ADAPTER=ON
make iowarp_engine
```

## Known Issues

### ADIOS2 Installation Issue
The ADIOS2 installation in the current conda environment has broken headers. The `Variable.h` header contains incorrect C++ syntax:

```cpp
// INCORRECT (current ADIOS2 headers):
template <class T>
class Variable {
    Variable<T>(const std::string &name, ...);  // ❌ Wrong
    ~Variable<T>() = default;                    // ❌ Wrong
};

// CORRECT C++ syntax should be:
template <class T>
class Variable {
    Variable(const std::string &name, ...);     // ✅ Correct
    ~Variable() = default;                       // ✅ Correct
};
```

**Workaround**: Install a different version of ADIOS2 or patch the headers.

## Implementation Details

### API Updates
The adapter has been updated to work with the current CTE API:
- Removed `mctx` (MemoryCtx) parameters - no longer used in CTE API
- Updated to use `CLIO_CTE_CLIENT` global singleton
- Fixed buffer allocation to use `AllocateBuffer()` instead of template version
- Updated constructor to use move semantics for `adios2::helper::Comm`

### Key Features
- Async and sync Put/Get operations
- Tag-based blob organization (ADIOS file = CTE tag)
- Step-based data management
- Shared memory buffer handling for efficient data transfer

## Architecture

The adapter creates a bridge between ADIOS2's plugin engine interface and IOWarp's CTE:

1. **ADIOS2 File → CTE Tag**: Each ADIOS2 file/session maps to a CTE tag
2. **ADIOS2 Variable → CTE Blob**: Each variable write becomes a blob in CTE
3. **Step Management**: ADIOS2 steps are encoded in blob names

## Future Work
- Test with a working ADIOS2 installation
- Add comprehensive unit tests
- Optimize async operations with proper Future tracking
- Add performance benchmarks

## Contact
For issues or questions, please contact the IOWarp development team.

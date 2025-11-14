"""
IOWarp Core - High-performance distributed I/O and task execution runtime.

This package provides Python bindings and installation support for the IOWarp
ecosystem, which includes:

- Context Transport Primitives (HermesShm): Shared memory IPC primitives
- Runtime (Chimaera): Distributed task execution runtime
- Context Transfer Engine (Hermes): Multi-tiered I/O buffering system
- Context Assimilation Engine: Data ingestion and processing engine

All components are built from C++ source using CMake during installation.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("iowarp-core")
except PackageNotFoundError:
    # Package is not installed, use a fallback version
    __version__ = "0.0.0.dev0"

__author__ = "IOWarp Team"
__license__ = "BSD-3-Clause"

# Component versions and information
COMPONENTS = {
    "context-transport-primitives": {
        "description": "High-performance shared memory library with IPC primitives",
        "url": "https://github.com/iowarp/context-transport-primitives",
    },
    "runtime": {
        "description": "Distributed task execution runtime (Chimaera)",
        "url": "https://github.com/iowarp/runtime",
    },
    "context-transfer-engine": {
        "description": "Multi-tiered I/O buffering system (Hermes)",
        "url": "https://github.com/iowarp/context-transfer-engine",
    },
    "context-assimilation-engine": {
        "description": "Data ingestion and processing engine",
        "url": "https://github.com/iowarp/context-assimilation-engine",
    },
}


def get_component_info():
    """Get information about installed IOWarp components."""
    return COMPONENTS


def get_version():
    """Get the version of iowarp-core package."""
    return __version__


__all__ = [
    "__version__",
    "COMPONENTS",
    "get_component_info",
    "get_version",
]

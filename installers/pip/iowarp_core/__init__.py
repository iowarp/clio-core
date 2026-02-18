"""IOWarp Core - Context Management Platform.

Sets up library search paths so IOWarp shared libraries and Python
extensions can be loaded without system-wide installation.
"""

import ctypes
import os
import sys

__version__ = "1.0.0"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_PACKAGE_DIR, "lib")
_EXT_DIR = os.path.join(_PACKAGE_DIR, "ext")
_BIN_DIR = os.path.join(_PACKAGE_DIR, "bin")
_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")


def _setup():
    """Configure library and extension paths at import time."""
    # Add lib/ to LD_LIBRARY_PATH so dlopen finds IOWarp shared libs
    if os.path.isdir(_LIB_DIR):
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if _LIB_DIR not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = (
                _LIB_DIR + ":" + ld_path if ld_path else _LIB_DIR
            )

        # Pre-load the core library with RTLD_GLOBAL so symbols are
        # available to all subsequently loaded IOWarp libraries
        _core_lib = os.path.join(_LIB_DIR, "libchimaera_cxx.so")
        if os.path.exists(_core_lib):
            ctypes.CDLL(_core_lib, mode=ctypes.RTLD_GLOBAL)

    # Add ext/ to sys.path so 'import wrp_cte_core_ext' works
    if os.path.isdir(_EXT_DIR) and _EXT_DIR not in sys.path:
        sys.path.insert(0, _EXT_DIR)


_setup()


def get_version():
    """Return the package version string."""
    return __version__


def get_lib_dir():
    """Return the path to the IOWarp shared library directory."""
    return _LIB_DIR


def get_ext_dir():
    """Return the path to the Python extension modules directory."""
    return _EXT_DIR


def get_bin_dir():
    """Return the path to the IOWarp binary directory."""
    return _BIN_DIR


def get_data_dir():
    """Return the path to the IOWarp data directory."""
    return _DATA_DIR


def cte_available():
    """Check if the Context Transfer Engine extension is available."""
    ext_path = os.path.join(_EXT_DIR, "wrp_cte_core_ext")
    # Check for any .so file matching the extension name
    if os.path.isdir(_EXT_DIR):
        for f in os.listdir(_EXT_DIR):
            if f.startswith("wrp_cte_core_ext") and f.endswith(".so"):
                return True
    return False

"""IOWarp Core - Context Management Platform.

Sets up library search paths so IOWarp shared libraries and Python
extensions can be loaded without system-wide installation.
"""

import ctypes
import os
import shutil
import sys

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("iowarp-core")
except Exception:
    __version__ = "0.0.0-dev"

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

        # Pre-load shared libraries in dependency order with RTLD_GLOBAL
        # so symbols are available to all subsequently loaded IOWarp libraries.
        # LD_LIBRARY_PATH changes above only affect child processes, so we
        # must explicitly load each library for the current process.
        for _lib_name in [
            "libhermes_shm_host.so",
            "libchimaera_cxx.so",
        ]:
            _lib_path = os.path.join(_LIB_DIR, _lib_name)
            if os.path.exists(_lib_path):
                ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

    # Add ext/ to sys.path so 'import wrp_cte_core_ext' works
    if os.path.isdir(_EXT_DIR) and _EXT_DIR not in sys.path:
        sys.path.insert(0, _EXT_DIR)

    # Seed ~/.chimaera/chimaera.yaml from the bundled default if missing.
    # The C++ runtime looks for this file when no env var override is set.
    _user_conf_dir = os.path.expanduser("~/.chimaera")
    _user_conf = os.path.join(_user_conf_dir, "chimaera.yaml")
    _bundled_default = os.path.join(_DATA_DIR, "chimaera_default.yaml")
    if not os.path.exists(_user_conf) and os.path.exists(_bundled_default):
        try:
            os.makedirs(_user_conf_dir, exist_ok=True)
            shutil.copy2(_bundled_default, _user_conf)
        except OSError:
            pass  # read-only home, containerised, etc.


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

"""IOWarp Core - Context Management Platform.

Sets up library search paths so IOWarp shared libraries and Python
extensions can be loaded without system-wide installation.

Usage::

    import wrp_cee as cee          # Context Exploration Engine
    import wrp_cte_core_ext        # Context Transfer Engine
"""

import ctypes
import importlib
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

# Extension modules that live in ext/ and can be imported via
# "from iowarp_core import <name>".
_EXT_MODULES = {"wrp_cee", "wrp_cte_core_ext"}


def _setup():
    """Configure library and extension paths at import time."""
    # Add lib/ to LD_LIBRARY_PATH so dlopen finds IOWarp shared libs
    if os.path.isdir(_LIB_DIR):
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if _LIB_DIR not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = (
                _LIB_DIR + ":" + ld_path if ld_path else _LIB_DIR
            )

        # Pre-load ALL shared libraries in dependency order with RTLD_GLOBAL
        # so symbols are globally visible to subsequently loaded extensions.
        # LD_LIBRARY_PATH changes above only affect child processes, so we
        # must explicitly load each library for the current process.
        # Python loads extension modules with RTLD_LOCAL by default, which
        # hides symbols from transitive dependencies and breaks nanobind
        # modules like wrp_cee that depend on multiple IOWarp libraries.
        for _lib_name in [
            "libhermes_shm_host.so",
            "libchimaera_cxx.so",
            "libchimaera_admin_client.so",
            "libchimaera_admin_runtime.so",
            "libchimaera_bdev_client.so",
            "libchimaera_bdev_runtime.so",
            "libwrp_cte_core_client.so",
            "libwrp_cte_core_runtime.so",
            "libwrp_cte_cae_config.so",
            "libwrp_cae_core_client.so",
            "libwrp_cae_core_runtime.so",
            "libwrp_cee_api.so",
        ]:
            _lib_path = os.path.join(_LIB_DIR, _lib_name)
            if os.path.exists(_lib_path):
                ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

    # Add ext/ to sys.path so extension modules can be found by import
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


# PEP 562: "from iowarp_core import wrp_cee" lazily loads the extension.
def __getattr__(name):
    if name in _EXT_MODULES:
        return importlib.import_module(name)
    raise AttributeError(f"module 'iowarp_core' has no attribute {name!r}")


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
    if os.path.isdir(_EXT_DIR):
        for f in os.listdir(_EXT_DIR):
            if f.startswith("wrp_cte_core_ext") and f.endswith(".so"):
                return True
    return False


def cee_available():
    """Check if the Context Exploration Engine extension is available."""
    if os.path.isdir(_EXT_DIR):
        for f in os.listdir(_EXT_DIR):
            if f.startswith("wrp_cee") and f.endswith(".so"):
                return True
    return False

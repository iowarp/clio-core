"""IOWarp Core - Context Management Platform.

Sets up library search paths so IOWarp shared libraries and Python
extensions can be loaded without system-wide installation.

Usage::

    import clio_cee as cee          # Context Exploration Engine
    import clio_cte_core_ext        # Context Transfer Engine
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
_EXT_MODULES = {"clio_cee", "clio_cte_core_ext", "chimaera_runtime_ext"}


_IS_WINDOWS = sys.platform == "win32"


def _setup_windows_dll_path():
    """Make IOWarp + vcpkg DLLs discoverable to the extension loader.

    On Windows the .pyd extensions in ext/ are linked against IOWarp DLLs that
    install to bin/ (alongside the .exe binaries — the Windows convention), and
    against vcpkg-provided runtime DLLs that get copied there too. The Python
    loader doesn't look in arbitrary package directories, so we register them
    with os.add_dll_directory (the Win32-correct replacement for LD_LIBRARY_PATH;
    PATH is not consulted by default since Python 3.8 secure DLL search).
    """
    for _d in (_BIN_DIR, _LIB_DIR):
        if os.path.isdir(_d):
            os.add_dll_directory(_d)


def _setup_posix_lib_path():
    """LD_LIBRARY_PATH + RTLD_GLOBAL preload chain (Linux/macOS only)."""
    if not os.path.isdir(_LIB_DIR):
        return
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
    # modules like clio_cee that depend on multiple IOWarp libraries.
    for _lib_name in [
        "libclio_ctp_host.so",
        "libchimaera_cxx.so",
        "libclio_admin_client.so",
        "libclio_admin_runtime.so",
        "libchimaera_bdev_client.so",
        "libchimaera_bdev_runtime.so",
        "libclio_cte_core_client.so",
        "libclio_cte_core_runtime.so",
        "libclio_cte_cae_config.so",
        "libclio_cae_core_client.so",
        "libclio_cae_core_runtime.so",
        "libclio_cee_api.so",
    ]:
        _lib_path = os.path.join(_LIB_DIR, _lib_name)
        if os.path.exists(_lib_path):
            ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)


def _setup():
    """Configure library and extension paths at import time."""
    if _IS_WINDOWS:
        _setup_windows_dll_path()
    else:
        _setup_posix_lib_path()

    # Add ext/ to sys.path so extension modules can be found by import
    if os.path.isdir(_EXT_DIR) and _EXT_DIR not in sys.path:
        sys.path.insert(0, _EXT_DIR)

    # Seed the per-user default config from the bundled default if missing.
    # Both ~/.clio/clio.yaml (preferred) AND ~/.chimaera/chimaera.yaml
    # (legacy) are seeded so the runtime's lookup hits a file regardless of
    # which layout the user has migrated to. The C++ runtime checks the new
    # path first; see ConfigManager::GetServerConfigPath.
    _bundled_default = os.path.join(_DATA_DIR, "chimaera_default.yaml")
    if os.path.exists(_bundled_default):
        for _dir, _name in (("~/.clio", "clio.yaml"),
                            ("~/.chimaera", "chimaera.yaml")):
            _user_conf_dir = os.path.expanduser(_dir)
            _user_conf = os.path.join(_user_conf_dir, _name)
            if not os.path.exists(_user_conf):
                try:
                    os.makedirs(_user_conf_dir, exist_ok=True)
                    shutil.copy2(_bundled_default, _user_conf)
                except OSError:
                    pass  # read-only home, containerised, etc.


_setup()


# PEP 562: "from iowarp_core import clio_cee" lazily loads the extension.
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


_EXT_SUFFIX = ".pyd" if _IS_WINDOWS else ".so"


def _ext_present(prefix):
    if not os.path.isdir(_EXT_DIR):
        return False
    return any(
        f.startswith(prefix) and f.endswith(_EXT_SUFFIX)
        for f in os.listdir(_EXT_DIR)
    )


def cte_available():
    """Check if the Context Transfer Engine extension is available."""
    return _ext_present("clio_cte_core_ext")


def cee_available():
    """Check if the Context Exploration Engine extension is available."""
    return _ext_present("clio_cee")

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


def _setup():
    """Configure library and extension paths at import time.

    Per-platform shape:
      - Linux: prepend lib/ to LD_LIBRARY_PATH (for child procs), then
        dlopen each lib in dependency order with RTLD_GLOBAL so nanobind
        extension modules find transitive symbols. Python loads .so
        extensions with RTLD_LOCAL by default, which hides transitive
        deps and breaks clio_cee on Linux.
      - macOS: skip the explicit preload — every dylib in the wheel
        carries @loader_path/../lib rpaths set at link time, so dlopen()
        resolves transitively without help. Still prepend
        DYLD_LIBRARY_PATH for child procs (SIP may strip it for system
        binaries; harmless for our spawned children).
      - Windows: DLLs live alongside .pyd files in bin/ (CMake's Windows
        convention). Python 3.8+ restricts LoadLibraryEx to the .pyd's
        own dir + system32 by default — register bin/ and lib/ via
        os.add_dll_directory so .pyd extensions find clio_*.dll deps.
        Also prepend PATH for child procs spawned by the chimaera
        wrapper.
    """
    if sys.platform == "win32":
        # Register DLL search directories for in-process LoadLibrary
        # (Python extension loads).
        for _d in (_BIN_DIR, _LIB_DIR):
            if os.path.isdir(_d):
                try:
                    os.add_dll_directory(_d)
                except (OSError, AttributeError):
                    pass  # add_dll_directory is Py3.8+ and Windows-only.
        # Prepend bin/ + lib/ to PATH so child processes (the wrapped
        # chimaera/clio_*_bench) inherit a search path that finds the
        # same DLLs.
        _paths = [d for d in (_BIN_DIR, _LIB_DIR) if os.path.isdir(d)]
        if _paths:
            existing = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join(
                _paths + ([existing] if existing else [])
            )
    elif os.path.isdir(_LIB_DIR):
        # POSIX (Linux + macOS).
        _env_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        ld_path = os.environ.get(_env_var, "")
        if _LIB_DIR not in ld_path:
            os.environ[_env_var] = (
                _LIB_DIR + os.pathsep + ld_path if ld_path else _LIB_DIR
            )

        # Linux-only: explicit RTLD_GLOBAL preload so nanobind extensions
        # see transitive symbols. macOS resolves these via @loader_path
        # rpaths baked into each dylib at link time.
        if sys.platform.startswith("linux"):
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


_EXT_SUFFIXES = (".so", ".pyd")  # POSIX (.so on Linux + macOS), Windows (.pyd)


def cte_available():
    """Check if the Context Transfer Engine extension is available."""
    if os.path.isdir(_EXT_DIR):
        for f in os.listdir(_EXT_DIR):
            if f.startswith("clio_cte_core_ext") and f.endswith(_EXT_SUFFIXES):
                return True
    return False


def cee_available():
    """Check if the Context Exploration Engine extension is available."""
    if os.path.isdir(_EXT_DIR):
        for f in os.listdir(_EXT_DIR):
            if f.startswith("clio_cee") and f.endswith(_EXT_SUFFIXES):
                return True
    return False

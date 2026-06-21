"""Configuration resolution helpers for IOWarp.

Provides utilities for locating the Clio server configuration file,
checking standard locations in order of precedence. Mirrors the C++
runtime's ConfigManager::GetServerConfigPath.
"""

import os


def find_config():
    """Find the Clio server configuration file.

    Search order:
    1. CLIO_SERVER_CONF env var (preferred), or CLIO_SERVER_CONF (legacy)
    2. ~/.clio/clio.yaml        (new canonical user config)
    3. ~/.clio/clio_run.yaml    (legacy filename in new dir)
    4. ~/.clio/clio.yaml    (new filename in legacy dir)
    5. ~/.clio/clio_run.yaml (legacy)
    6. Bundled default in the package data/ directory

    Returns:
        str: Path to the configuration file, or None if not found.
    """
    # 1. Environment variable override (new name, then legacy fallback)
    for var in ("CLIO_SERVER_CONF", "CLIO_SERVER_CONF"):
        env_conf = os.environ.get(var)
        if env_conf and os.path.isfile(env_conf):
            return env_conf

    # 2-5. User-local config — new dir/name combos first, legacy last
    for rel in (
        "~/.clio/clio.yaml",
        "~/.clio/clio_run.yaml",
        "~/.clio/clio.yaml",
        "~/.clio/clio_run.yaml",
    ):
        path = os.path.expanduser(rel)
        if os.path.isfile(path):
            return path

    # 6. Bundled default
    package_dir = os.path.dirname(os.path.abspath(__file__))
    default_conf = os.path.join(package_dir, "data", "clio_default.yaml")
    if os.path.isfile(default_conf):
        return default_conf

    return None


def get_default_config():
    """Return the path to the bundled default configuration file.

    Returns:
        str: Path to clio_default.yaml in the package data directory.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, "data", "clio_default.yaml")

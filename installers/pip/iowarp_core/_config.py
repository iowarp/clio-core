"""Configuration resolution helpers for IOWarp.

Provides utilities for locating the Chimaera server configuration file,
checking standard locations in order of precedence.
"""

import os


def find_config():
    """Find the Chimaera configuration file.

    Search order:
    1. CHI_SERVER_CONF environment variable
    2. ~/.chimaera/chimaera.yaml
    3. Bundled default in the package data/ directory

    Returns:
        str: Path to the configuration file, or None if not found.
    """
    # 1. Environment variable override
    env_conf = os.environ.get("CHI_SERVER_CONF")
    if env_conf and os.path.isfile(env_conf):
        return env_conf

    # 2. User-local config
    user_conf = os.path.expanduser("~/.chimaera/chimaera.yaml")
    if os.path.isfile(user_conf):
        return user_conf

    # 3. Bundled default
    package_dir = os.path.dirname(os.path.abspath(__file__))
    default_conf = os.path.join(package_dir, "data", "chimaera_default.yaml")
    if os.path.isfile(default_conf):
        return default_conf

    return None


def get_default_config():
    """Return the path to the bundled default configuration file.

    Returns:
        str: Path to chimaera_default.yaml in the package data directory.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, "data", "chimaera_default.yaml")

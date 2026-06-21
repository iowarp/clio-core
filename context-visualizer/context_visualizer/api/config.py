"""GET /api/config -- return the active Chimaera YAML config as JSON."""

import os
from pathlib import Path

from flask import Blueprint, jsonify

bp = Blueprint("config", __name__)


def _load_config():
    """Find and parse the active YAML config file.

    Search order matches the runtime
    (see ConfigManager::GetServerConfigPath in context-runtime/src/config_manager.cc):
      1. CLIO_SERVER_CONF env (preferred)
      2. CLIO_SERVER_CONF env (legacy)
      3. ~/.clio/clio.yaml         (new canonical user config)
      4. ~/.clio/chimaera.yaml     (legacy filename in new dir)
      5. ~/.chimaera/clio.yaml     (new filename in legacy dir)
      6. ~/.chimaera/chimaera.yaml (legacy)
    """
    import yaml

    home = Path.home()
    candidates = [
        os.environ.get("CLIO_SERVER_CONF"),
        os.environ.get("CLIO_SERVER_CONF"),
        str(home / ".clio" / "clio.yaml"),
        str(home / ".clio" / "chimaera.yaml"),
        str(home / ".chimaera" / "clio.yaml"),
        str(home / ".chimaera" / "chimaera.yaml"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path) as fh:
                return yaml.safe_load(fh)
    return None


@bp.route("/config")
def get_config():
    cfg = _load_config()
    if cfg is None:
        return jsonify({"error": "no config file found", "searched": [
            "CLIO_SERVER_CONF",
            "CLIO_SERVER_CONF",
            "~/.clio/clio.yaml",
            "~/.clio/chimaera.yaml",
            "~/.chimaera/clio.yaml",
            "~/.chimaera/chimaera.yaml",
        ]}), 404
    return jsonify({"config": cfg})

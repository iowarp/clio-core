"""GET /api/config -- return the active Clio YAML config as JSON."""

import os
from pathlib import Path

from flask import Blueprint, jsonify

bp = Blueprint("config", __name__)


def _load_config():
    """Find and parse the active YAML config file.

    Search order matches the runtime
    (see ConfigManager::GetServerConfigPath in context-runtime/src/config_manager.cc):
      1. CLIO_SERVER_CONF env
      2. ~/.clio/clio.yaml  (canonical user config)
    """
    import yaml

    home = Path.home()
    candidates = [
        os.environ.get("CLIO_SERVER_CONF"),
        str(home / ".clio" / "clio.yaml"),
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
            "~/.clio/clio.yaml",
        ]}), 404
    return jsonify({"config": cfg})

"""GET /api/config -- return the active Chimaera YAML config as JSON."""

import os
from pathlib import Path

from flask import Blueprint, jsonify

bp = Blueprint("config", __name__)


def _load_config():
    """Find and parse the active YAML config file.

    Search order matches the runtime:
      1. CHI_SERVER_CONF env
      2. WRP_RUNTIME_CONF env
      3. ~/.chimaera/chimaera.yaml
    """
    import yaml

    candidates = [
        os.environ.get("CHI_SERVER_CONF"),
        os.environ.get("WRP_RUNTIME_CONF"),
        str(Path.home() / ".chimaera" / "chimaera.yaml"),
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
            "CHI_SERVER_CONF",
            "WRP_RUNTIME_CONF",
            "~/.chimaera/chimaera.yaml",
        ]}), 404
    return jsonify({"config": cfg})

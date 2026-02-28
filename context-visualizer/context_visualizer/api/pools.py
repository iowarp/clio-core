"""GET /api/pools -- pool listing from the runtime config compose section."""

from flask import Blueprint, jsonify

from .config import _load_config

bp = Blueprint("pools", __name__)


@bp.route("/pools")
def get_pools():
    cfg = _load_config()
    if cfg is None:
        return jsonify({"error": "config not found"}), 404

    compose = cfg.get("compose", []) or []
    pools = []
    for entry in compose:
        pools.append({
            "mod_name": entry.get("mod_name", ""),
            "pool_name": entry.get("pool_name", ""),
            "pool_id": entry.get("pool_id", ""),
            "pool_query": entry.get("pool_query", ""),
        })

    return jsonify({"pools": pools})

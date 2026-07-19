"""Safe-bdev dashboard API.

Read endpoints (Phase 3a):
  GET  /api/safe_bdev/pools            -- safe_bdev pools from the compose config
  GET  /api/safe_bdev/<pool_id>/stats  -- live recovery progress + member roster

Write endpoints (Phase 3b -- manual member management from the dashboard):
  POST   /api/safe_bdev/<pool_id>/add_member
  DELETE /api/safe_bdev/<pool_id>/remove_member
  POST   /api/safe_bdev/<pool_id>/replace_member   (remove + auto-recover)

The write endpoints call clio_client wrappers over the runtime's
AddBdev/RemoveBdev/RecoverBdev tasks. They return 501 until the runtime
python bindings expose those calls (clio_client raises NotImplementedError).
"""

from flask import Blueprint, jsonify, request

from .. import clio_client
from .config import _load_config

bp = Blueprint("safe_bdev", __name__)


def _safe_bdev_pools():
    """Enumerate safe_bdev pools declared in the active compose config."""
    cfg = _load_config()
    pools = []
    if cfg:
        for entry in cfg.get("compose", []) or []:
            if entry.get("mod_name") == "clio_safe_bdev":
                pools.append({
                    "pool_name": entry.get("pool_name", ""),
                    "pool_id": str(entry.get("pool_id", "")),
                    "max_failures": entry.get("max_failures"),
                })
    return pools


@bp.route("/safe_bdev/pools")
def safe_bdev_pools():
    return jsonify({"pools": _safe_bdev_pools()})


@bp.route("/safe_bdev/<pool_id>/stats")
def safe_bdev_stats(pool_id):
    """Live safe_bdev stats for one pool.

    Shape (recovery fields drive the in-flight / remaining dashboard):
      { "stats": { "pool_name", "max_failures", "recovery_active",
                   "recovery_ops_total", "recovery_ops_completed",
                   "recovery_ops_in_flight", "recovery_ops_remaining",
                   "members": [ {role, index, pool_name, pool_id, state,
                                 recovering} ] } }
    """
    try:
        stats = clio_client.get_safe_bdev_stats(pool_id)
    except Exception as exc:  # noqa: BLE001 -- dashboard must not 500 hard
        return jsonify({"error": str(exc), "stats": {}}), 200
    return jsonify({"stats": stats})


def _member_action(pool_id, fn, *args):
    """Run a clio_client write helper, mapping NotImplementedError -> 501."""
    try:
        result = fn(pool_id, *args)
    except NotImplementedError as exc:
        return jsonify({"success": False, "error": str(exc)}), 501
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": str(exc)}), 500
    return jsonify({"success": True, "result": result})


@bp.route("/safe_bdev/<pool_id>/add_member", methods=["POST"])
def add_member(pool_id):
    """Grow the array: add a new member bdev (data or parity)."""
    data = request.get_json(silent=True) or {}
    return _member_action(
        pool_id, clio_client.safe_bdev_add_member,
        data.get("member_path"), data.get("capacity", "256MB"),
        int(data.get("node_id", 0)), int(data.get("as_parity", 0)),
    )


@bp.route("/safe_bdev/<pool_id>/remove_member", methods=["DELETE"])
def remove_member(pool_id):
    """Take a member out of service (mark faulty / unlink)."""
    data = request.get_json(silent=True) or {}
    return _member_action(
        pool_id, clio_client.safe_bdev_remove_member,
        str(data.get("target_pool_id", "")), int(data.get("was_faulty", 1)),
    )


@bp.route("/safe_bdev/<pool_id>/replace_member", methods=["POST"])
def replace_member(pool_id):
    """Replace a failed member: remove it, compose a fresh bdev, and
    auto-recover the lost shards onto it (the 'add another to trigger
    recovery' flow). Recovery progress then shows up in the stats poll."""
    data = request.get_json(silent=True) or {}
    return _member_action(
        pool_id, clio_client.safe_bdev_replace_member,
        str(data.get("failed_pool_id", "")), data.get("member_path"),
        data.get("capacity", "256MB"), int(data.get("node_id", 0)),
    )

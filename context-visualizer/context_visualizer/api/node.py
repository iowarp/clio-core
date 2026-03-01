"""Per-node API endpoints: workers, system_stats, bdev_stats."""

import os

from flask import Blueprint, jsonify, request

from .. import chimaera_client

bp = Blueprint("node", __name__)


def _node_is_alive(node_id):
    """TCP liveness check before querying a node.

    Looks up the node's IP from the hostfile and attempts a TCP connect
    to the RPC port.  The dashboard is a separate process and keeps
    running even when the local runtime is down.
    """
    # Look up IP from hostfile
    hostfile = os.environ.get("CONTAINER_HOSTFILE", "")
    if not hostfile:
        return True  # No hostfile — can't check, let it try
    try:
        with open(hostfile) as f:
            lines = [line.strip() for line in f if line.strip()]
        if node_id < len(lines):
            ip = lines[node_id]
            alive = chimaera_client.check_nodes_alive([ip], port=9413, timeout=1)
            return 0 in alive
    except Exception:
        pass
    return True  # Can't determine — let it try


def _worker_stats(node_id):
    """Get worker stats, falling back to local if node_id is 0."""
    if node_id == 0:
        return chimaera_client.get_worker_stats()
    return chimaera_client.get_worker_stats_for_node(node_id)


def _system_stats(node_id, min_event_id=0):
    """Get system stats, falling back to local if node_id is 0."""
    if node_id == 0:
        return chimaera_client.get_system_stats("local", min_event_id)
    return chimaera_client.get_system_stats_for_node(node_id, min_event_id)


def _bdev_stats(node_id):
    """Get bdev stats, falling back to local if node_id is 0."""
    if node_id == 0:
        return chimaera_client.get_bdev_stats("local")
    return chimaera_client.get_bdev_stats_for_node(node_id)


@bp.route("/node/<int:node_id>/workers")
def get_node_workers(node_id):
    if not _node_is_alive(node_id):
        return jsonify({"error": "node_down", "workers": [], "summary": {
            "count": 0, "queued": 0, "blocked": 0, "processed": 0,
        }}), 503

    try:
        raw = _worker_stats(node_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    workers = []
    for _cid, data in raw.items():
        if isinstance(data, list):
            workers.extend(data)
        elif isinstance(data, dict):
            workers.append(data)

    return jsonify({
        "workers": workers,
        "summary": {
            "count": len(workers),
            "queued": sum(w.get("num_queued_tasks", 0) for w in workers),
            "blocked": sum(w.get("num_blocked_tasks", 0) for w in workers),
            "processed": sum(w.get("num_tasks_processed", 0) for w in workers),
        },
    })


@bp.route("/node/<int:node_id>/system_stats")
def get_node_system_stats(node_id):
    if not _node_is_alive(node_id):
        return jsonify({"error": "node_down", "entries": []}), 503

    min_event_id = request.args.get("min_event_id", 0, type=int)
    try:
        raw = _system_stats(node_id, min_event_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    entries = []
    for _cid, data in raw.items():
        if isinstance(data, list):
            entries.extend(data)
        elif isinstance(data, dict):
            entries.append(data)

    return jsonify({"entries": entries})


@bp.route("/node/<int:node_id>/bdev_stats")
def get_node_bdev_stats(node_id):
    if not _node_is_alive(node_id):
        return jsonify({"error": "node_down", "devices": []}), 503

    try:
        raw = _bdev_stats(node_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    devices = []
    for _cid, data in raw.items():
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("stats"):
                    entry = dict(item["stats"])
                    entry["pool_id"] = item.get("pool_id", "")
                    devices.append(entry)
        elif isinstance(data, dict) and data.get("stats"):
            entry = dict(data["stats"])
            entry["pool_id"] = data.get("pool_id", "")
            devices.append(entry)

    return jsonify({"devices": devices})

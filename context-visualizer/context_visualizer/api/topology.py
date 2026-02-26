"""GET /api/topology -- cluster topology overview."""

import socket

from flask import Blueprint, jsonify

from .. import chimaera_client

bp = Blueprint("topology", __name__)


@bp.route("/topology")
def get_topology():
    try:
        raw = chimaera_client.get_system_stats_all()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    # Each container's response is an array of system_stats entries.
    # Extract the latest entry per node (identified by node_id).
    # Fallback: if node_id is missing (old runtime), use container_id as key.
    nodes = {}
    for cid, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Prefer node_id from response; fall back to container id
            nid = entry.get("node_id")
            if nid is None:
                nid = cid
            nid_str = str(nid)
            # Keep the latest entry per node (highest event_id)
            prev = nodes.get(nid_str)
            if prev is None or entry.get("event_id", 0) > prev.get("event_id", 0):
                nodes[nid_str] = entry

    # Fallback hostname when the runtime doesn't supply one
    local_hostname = socket.gethostname()

    result = []
    for nid_str, entry in nodes.items():
        # Derive a numeric node_id for the frontend URL
        raw_nid = entry.get("node_id")
        if raw_nid is not None:
            node_id = int(raw_nid)
        else:
            # Best-effort: try parsing the container id string, default to 0
            try:
                node_id = int(nid_str)
            except (ValueError, TypeError):
                node_id = 0

        result.append({
            "node_id": node_id,
            "hostname": entry.get("hostname") or local_hostname,
            "ip_address": entry.get("ip_address", ""),
            "cpu_usage_pct": entry.get("cpu_usage_pct", 0),
            "ram_usage_pct": entry.get("ram_usage_pct", 0),
            "gpu_count": entry.get("gpu_count", 0),
            "gpu_usage_pct": entry.get("gpu_usage_pct", 0),
            "hbm_usage_pct": entry.get("hbm_usage_pct", 0),
        })

    return jsonify({"nodes": result})

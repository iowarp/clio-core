"""GET /api/topology -- cluster topology overview + node management."""

import os
import socket

from flask import Blueprint, jsonify

from .. import chimaera_client

bp = Blueprint("topology", __name__)


def _read_hostfile():
    """Read the hostfile and return a list of (node_id, ip_address) tuples."""
    hostfile = os.environ.get("CONTAINER_HOSTFILE", "")
    if not hostfile:
        return []
    try:
        with open(hostfile) as f:
            lines = [line.strip() for line in f if line.strip()]
        return [(i, ip) for i, ip in enumerate(lines)]
    except Exception:
        return []


def _get_node_ip(node_id):
    """Get the IP address for a node from the hostfile."""
    node_id = int(node_id)
    for nid, ip in _read_hostfile():
        if nid == node_id:
            return ip
    if node_id == 0:
        return "127.0.0.1"
    return None


@bp.route("/topology")
def get_topology():
    # 1. Get the full expected node list from the hostfile
    hostfile_nodes = _read_hostfile()

    # 2. TCP liveness check — fast, does not go through the C++ runtime
    #    so it won't block when nodes are dead
    ip_list = [ip for _, ip in hostfile_nodes]
    alive_set = set()
    if ip_list:
        try:
            alive_set = chimaera_client.check_nodes_alive(ip_list, port=9413, timeout=1)
        except Exception:
            pass

    # 3. Query system_stats only for nodes confirmed alive via TCP
    alive_node_ids = [nid for idx, (nid, ip) in enumerate(hostfile_nodes) if idx in alive_set]
    live_nodes = {}  # node_id -> entry
    if alive_node_ids:
        try:
            live_nodes = chimaera_client.get_system_stats_per_node(alive_node_ids, timeout=3)
        except Exception:
            pass

    # Fallback: no hostfile — broadcast (legacy, may block if nodes are dead)
    if not hostfile_nodes:
        try:
            raw = chimaera_client.get_system_stats_all()
            for cid, entries in raw.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    nid = entry.get("node_id")
                    if nid is None:
                        nid = cid
                    nid_int = int(nid)
                    prev = live_nodes.get(nid_int)
                    if prev is None or entry.get("event_id", 0) > prev.get("event_id", 0):
                        live_nodes[nid_int] = entry
        except Exception:
            pass

    local_hostname = socket.gethostname()

    # 4. Merge: hostfile gives us the full list, TCP + stats tell us who's alive
    result = []
    if hostfile_nodes:
        for idx, (node_id, ip) in enumerate(hostfile_nodes):
            tcp_alive = idx in alive_set
            entry = live_nodes.get(node_id)
            if tcp_alive and entry:
                result.append({
                    "node_id": node_id,
                    "hostname": entry.get("hostname") or ip,
                    "ip_address": entry.get("ip_address") or ip,
                    "cpu_usage_pct": entry.get("cpu_usage_pct", 0),
                    "ram_usage_pct": entry.get("ram_usage_pct", 0),
                    "gpu_count": entry.get("gpu_count", 0),
                    "gpu_usage_pct": entry.get("gpu_usage_pct", 0),
                    "hbm_usage_pct": entry.get("hbm_usage_pct", 0),
                    "alive": True,
                })
            elif tcp_alive:
                # TCP says alive but stats query failed — still mark alive
                result.append({
                    "node_id": node_id,
                    "hostname": ip,
                    "ip_address": ip,
                    "cpu_usage_pct": 0,
                    "ram_usage_pct": 0,
                    "gpu_count": 0,
                    "gpu_usage_pct": 0,
                    "hbm_usage_pct": 0,
                    "alive": True,
                })
            else:
                result.append({
                    "node_id": node_id,
                    "hostname": ip,
                    "ip_address": ip,
                    "cpu_usage_pct": 0,
                    "ram_usage_pct": 0,
                    "gpu_count": 0,
                    "gpu_usage_pct": 0,
                    "hbm_usage_pct": 0,
                    "alive": False,
                })
    else:
        # No hostfile — only show nodes that responded (legacy behavior)
        for nid_int, entry in live_nodes.items():
            result.append({
                "node_id": nid_int,
                "hostname": entry.get("hostname") or local_hostname,
                "ip_address": entry.get("ip_address", ""),
                "cpu_usage_pct": entry.get("cpu_usage_pct", 0),
                "ram_usage_pct": entry.get("ram_usage_pct", 0),
                "gpu_count": entry.get("gpu_count", 0),
                "gpu_usage_pct": entry.get("gpu_usage_pct", 0),
                "hbm_usage_pct": entry.get("hbm_usage_pct", 0),
                "alive": True,
            })

    return jsonify({"nodes": result})


@bp.route("/topology/node/<node_id>/shutdown", methods=["POST"])
def shutdown_node(node_id):
    ip = _get_node_ip(node_id)
    if ip is None:
        return jsonify({"error": f"Node {node_id} not found or has no IP"}), 404

    try:
        result = chimaera_client.shutdown_node(ip)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code


@bp.route("/topology/node/<node_id>/restart", methods=["POST"])
def restart_node(node_id):
    print(f"[topology] POST restart node_id={node_id}", flush=True)
    ip = _get_node_ip(node_id)
    print(f"[topology] Resolved node_id={node_id} -> ip={ip}", flush=True)
    if ip is None:
        return jsonify({"error": f"Node {node_id} not found or has no IP"}), 404

    try:
        result = chimaera_client.restart_node(ip)
    except Exception as exc:
        print(f"[topology] restart_node raised: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500

    print(f"[topology] restart_node result: success={result.get('success')} "
          f"rc={result.get('returncode')} stderr={result.get('stderr', '')[:200]}", flush=True)

    # NOTE: We intentionally do NOT call reinit() here.
    # 1. CHIMAERA_INIT has a static guard that prevents re-initialization,
    #    so reinit() is effectively a no-op after the first init.
    # 2. After Phase 2 failover, the C++ client is connected to a surviving
    #    node and can route physical:N queries to any node (including the
    #    restarted one) without needing a direct local connection.

    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code

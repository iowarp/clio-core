"""GET /api/system -- high-level system overview."""

from flask import Blueprint, jsonify

from .. import chimaera_client

bp = Blueprint("system", __name__)


@bp.route("/system")
def get_system():
    connected = chimaera_client.is_connected()

    info = {
        "connected": connected,
        "workers": 0,
        "queued": 0,
        "blocked": 0,
        "processed": 0,
    }

    if not connected:
        # Try to connect on first request
        try:
            raw = chimaera_client.get_worker_stats()
            connected = True
        except Exception as exc:
            return jsonify({"connected": False, "error": str(exc)}), 503
    else:
        try:
            raw = chimaera_client.get_worker_stats()
        except Exception as exc:
            return jsonify({"connected": True, "error": str(exc)}), 503

    workers = []
    for _cid, data in raw.items():
        if isinstance(data, list):
            workers.extend(data)
        elif isinstance(data, dict):
            workers.append(data)

    info["connected"] = True
    info["workers"] = len(workers)
    info["queued"] = sum(w.get("queued", 0) for w in workers)
    info["blocked"] = sum(w.get("blocked", 0) for w in workers)
    info["processed"] = sum(w.get("processed", 0) for w in workers)

    return jsonify(info)

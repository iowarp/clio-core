"""GET /api/workers -- live worker statistics."""

from flask import Blueprint, jsonify

from .. import chimaera_client

bp = Blueprint("workers", __name__)


@bp.route("/workers")
def get_workers():
    try:
        raw = chimaera_client.get_worker_stats()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    # Flatten container results into a single worker list
    workers = []
    for _cid, data in raw.items():
        if isinstance(data, list):
            workers.extend(data)
        elif isinstance(data, dict):
            workers.append(data)

    total_queued = sum(w.get("queued", 0) for w in workers)
    total_blocked = sum(w.get("blocked", 0) for w in workers)
    total_processed = sum(w.get("processed", 0) for w in workers)

    return jsonify({
        "workers": workers,
        "summary": {
            "count": len(workers),
            "queued": total_queued,
            "blocked": total_blocked,
            "processed": total_processed,
        },
    })

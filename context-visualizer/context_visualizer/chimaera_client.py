"""Wrapper around chimaera_runtime_ext for the visualizer."""

import threading

try:
    import msgpack
except ImportError:
    msgpack = None

_lock = threading.Lock()
_chi = None
_admin = None
_init_done = False


def _ensure_init():
    """Lazy-initialize the Chimaera client connection."""
    global _chi, _admin, _init_done
    if _init_done:
        return
    with _lock:
        if _init_done:
            return
        import chimaera_runtime_ext as chi
        _chi = chi
        ok = chi.chimaera_init(chi.ChimaeraMode.kClient)
        if not ok:
            raise RuntimeError("chimaera_init(kClient) failed -- is the runtime running?")
        _admin = chi.AdminClient(chi.PoolId(0, 0))
        _init_done = True


def _decode_results(results):
    """Decode a {container_id: bytes} dict into {container_id: decoded_data}."""
    decoded = {}
    for cid, blob in results.items():
        if msgpack is not None and isinstance(blob, (bytes, bytearray)):
            try:
                decoded[str(cid)] = msgpack.unpackb(blob, raw=False)
            except Exception:
                decoded[str(cid)] = blob
        else:
            decoded[str(cid)] = blob
    return decoded


def is_connected():
    """Return True if the client has been initialized."""
    return _init_done


def get_worker_stats():
    """Query worker_stats from the admin pool (local node)."""
    _ensure_init()
    results = _admin.monitor(_chi.PoolQuery.Local(), "worker_stats")
    return _decode_results(results)


def get_pool_worker_stats(pool_id_str, routing="local"):
    """Query worker_stats for a specific pool via pool_stats:// URI."""
    _ensure_init()
    uri = f"pool_stats://{pool_id_str}:{routing}:worker_stats"
    results = _admin.monitor(_chi.PoolQuery.Local(), uri)
    return _decode_results(results)


def get_status():
    """Query general status from the admin pool."""
    _ensure_init()
    results = _admin.monitor(_chi.PoolQuery.Local(), "status")
    return _decode_results(results)


def finalize():
    """Clean shutdown of the Chimaera client."""
    global _init_done, _admin
    if _init_done and _chi is not None:
        _chi.chimaera_finalize()
        _admin = None
        _init_done = False

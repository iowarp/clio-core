"""Wrapper around chimaera_runtime_ext for the visualizer."""

import concurrent.futures
import os
import socket
import threading

# The dashboard client should never block retrying a dead runtime.
# 0 = fail immediately; the dashboard relies on TCP liveness checks instead.
os.environ.setdefault("CHI_CLIENT_RETRY_TIMEOUT", "0")
os.environ.setdefault("CHI_CLIENT_TRY_NEW_SERVERS", "16")

try:
    import msgpack
except ImportError:
    msgpack = None

# Default timeout (seconds) for async_monitor calls.  Broadcasts that include
# dead nodes can block for 30+ seconds waiting on retries, so we cap them.
_MONITOR_TIMEOUT = 10

_lock = threading.Lock()
_chi = None
_init_done = False

# Single shared worker thread for ALL C++ client calls.
# The C++ IPC layer is not thread-safe (ReconnectToNewHost mutates shared
# transport state), so we must serialize all calls.  The GIL is released
# inside task.wait() so the main thread can still enforce timeouts and
# serve Flask requests while the worker blocks in C++.
_chi_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _ensure_init():
    """Lazy-initialize the Chimaera client connection."""
    global _chi, _init_done
    if _init_done:
        return
    with _lock:
        if _init_done:
            return
        import chimaera_runtime_ext as chi
        _chi = chi
        # 0 = kClient — pass a plain int to avoid any nanobind enum objects
        ok = chi.chimaera_init(0)
        if not ok:
            raise RuntimeError("chimaera_init(kClient) failed -- is the runtime running?")
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


def _do_monitor(pool_query, query, timeout):
    """Run async_monitor + wait in the calling thread (used by _monitor)."""
    task = _chi.async_monitor(pool_query, query)
    results = task.wait(timeout)
    return results


def _monitor(pool_query, query, timeout=_MONITOR_TIMEOUT):
    """Execute an async_monitor call with a timeout.

    The timeout is passed to both the C++ Wait() (so the worker thread
    is freed promptly) and the Python future.result() (so the caller
    is not blocked longer than necessary).
    """
    _ensure_init()
    future = _chi_worker.submit(_do_monitor, pool_query, query, timeout)
    try:
        results = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"Monitor({pool_query!r}, {query!r}) timed out after {timeout}s"
        )
    return _decode_results(results)


def is_connected():
    """Return True if the client has been initialized."""
    return _init_done



def reinit():
    """Tear down and reset the C++ client so the next call reconnects.

    Call this after the local runtime has been restarted so the client
    picks up the fresh process instead of using stale state.
    """
    global _init_done, _chi
    with _lock:
        if _init_done and _chi is not None:
            try:
                _chi.chimaera_finalize()
            except Exception:
                pass
        _init_done = False
        _chi = None


def get_worker_stats():
    """Query worker_stats from the admin pool (local node)."""
    return _monitor("local", "worker_stats")


def get_pool_worker_stats(pool_id_str, routing="local"):
    """Query worker_stats for a specific pool via pool_stats:// URI."""
    return _monitor("local", f"pool_stats://{pool_id_str}:{routing}:worker_stats")


def get_status():
    """Query general status from the admin pool."""
    return _monitor("local", "status")


def get_system_stats(pool_query="local", min_event_id=0):
    """Query system_stats from the admin pool."""
    return _monitor(pool_query, f"system_stats:{min_event_id}")


def get_system_stats_all(min_event_id=0):
    """Broadcast system_stats to all nodes (each self-identifies via hostname/ip/node_id)."""
    return _monitor("broadcast", f"system_stats:{min_event_id}")


def check_nodes_alive(ip_list, port=9413, timeout=1):
    """Check which nodes are alive by attempting a TCP connection to their RPC port.

    Returns a set of indices (matching ip_list positions) that are alive.
    This avoids the C++ runtime's IPC layer which blocks on dead nodes.
    """
    alive = set()

    def _check_one(idx, ip):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((ip, port))
            s.close()
            return idx
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(ip_list), 1)) as pool:
        futures = [pool.submit(_check_one, i, ip) for i, ip in enumerate(ip_list)]
        for f in concurrent.futures.as_completed(futures, timeout=timeout + 2):
            try:
                idx = f.result()
                if idx is not None:
                    alive.add(idx)
            except Exception:
                pass

    return alive


def get_system_stats_per_node(node_ids, min_event_id=0, timeout=3):
    """Query system_stats for each node sequentially.

    Returns {node_id: entry_dict} for nodes that responded, skipping
    dead nodes instead of blocking the whole request.

    Queries are serialized through the shared _chi_worker to avoid
    concurrent access to the C++ IPC layer.  The topology endpoint
    already filters to alive-only nodes via TCP checks, so the
    sequential delay is minimal.
    """
    _ensure_init()
    results = {}

    for node_id in node_ids:
        try:
            raw = _monitor(f"physical:{node_id}", f"system_stats:{min_event_id}", timeout=timeout)
            latest = None
            for cid, entries in raw.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            latest = entry
            if latest is not None:
                results[node_id] = latest
        except Exception:
            pass

    return results


def get_bdev_stats(pool_query="local"):
    """Query bdev_stats from the admin pool."""
    return _monitor(pool_query, "bdev_stats")


def get_worker_stats_for_node(node_id):
    """Query worker_stats for a specific node."""
    return _monitor(f"physical:{node_id}", "worker_stats")


def get_system_stats_for_node(node_id, min_event_id=0):
    """Query system_stats for a specific node."""
    return _monitor(f"physical:{node_id}", f"system_stats:{min_event_id}")


def get_bdev_stats_for_node(node_id):
    """Query bdev_stats for a specific node."""
    return _monitor(f"physical:{node_id}", "bdev_stats")


def get_host_info(node_id):
    """Query host info (hostname, ip_address, node_id) for a specific node."""
    decoded = _monitor(f"physical:{node_id}", "get_host_info")
    for entry in decoded.values():
        if isinstance(entry, dict):
            return entry
    return {}


def _do_stop_runtime(pool_query, grace_period_ms):
    """Run stop_runtime on the shared worker thread."""
    _chi.stop_runtime(pool_query, grace_period_ms)


def shutdown_node(node_id, grace_period_ms=5000):
    """Shutdown a node via AsyncStopRuntime through the C++ client.

    Sends the same task the C++ unit tests use — no SSH required.
    Fire-and-forget: the target runtime dies before it can reply.
    Serialized through _chi_worker to avoid concurrent C++ access.
    """
    _ensure_init()
    pool_query = f"physical:{node_id}"
    try:
        future = _chi_worker.submit(_do_stop_runtime, pool_query, grace_period_ms)
        future.result(timeout=5)
    except Exception as exc:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "success": True,
        "returncode": 0,
        "stdout": f"StopRuntime sent to node {node_id}",
        "stderr": "",
    }


def restart_node(ip_address, port=9413):
    """Restart a node's Chimaera runtime (non-blocking).

    Assumes the runtime is already dead (shutdown_node was called first).
    Launches ``chimaera runtime restart`` (WAL replay) so the node rejoins
    the existing cluster. Returns immediately — the dashboard's topology
    polling (TCP-based) will detect when the node comes back.
    """
    import os
    import subprocess

    local_ip = os.environ.get("NODE_IP", "")
    is_local = (ip_address == local_ip)

    print(f"[restart_node] Starting restart for {ip_address}:{port} "
          f"(local={is_local})", flush=True)

    if is_local:
        print("[restart_node] Launching local runtime via Popen", flush=True)
        try:
            subprocess.Popen(
                ["chimaera", "runtime", "restart"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            print(f"[restart_node] Popen failed: {exc}", flush=True)
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Failed to launch runtime: {exc}",
            }
    else:
        env_parts = []
        for var in ("PATH", "LD_LIBRARY_PATH",
                    "CHI_SERVER_CONF", "CHI_NUM_CONTAINERS",
                    "CONTAINER_HOSTFILE"):
            val = os.environ.get(var)
            if val:
                env_parts.append(f"{var}={val}")
        env_str = ("export " + " ".join(env_parts) + " && ") if env_parts else ""
        remote_cmd = (
            f"{env_str}"
            f"nohup setsid chimaera runtime restart "
            f"</dev/null >/dev/null 2>&1 & disown; exit 0"
        )
        cmd = [
            "ssh", "-T", "-n",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            ip_address,
            remote_cmd,
        ]
        print(f"[restart_node] Starting via SSH: {' '.join(cmd)}", flush=True)
        try:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            print(f"[restart_node] SSH exception: {exc}", flush=True)
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"SSH start command failed: {exc}",
            }

    print("[restart_node] Runtime launch initiated — returning immediately",
          flush=True)
    return {
        "success": True,
        "returncode": 0,
        "stdout": "Restart initiated; topology polling will detect when ready",
        "stderr": "",
    }


def finalize():
    """Clean shutdown of the Chimaera client."""
    global _init_done
    if _init_done and _chi is not None:
        _chi.chimaera_finalize()
        _init_done = False

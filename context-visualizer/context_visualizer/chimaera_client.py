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


def _do_monitor(pool_query, query):
    """Run async_monitor + wait in the calling thread (used by _monitor)."""
    task = _chi.async_monitor(pool_query, query)
    results = task.wait()
    return results


def _monitor(pool_query, query, timeout=_MONITOR_TIMEOUT):
    """Execute an async_monitor call with a timeout.

    Both async_monitor() and .wait() can block when the runtime or a
    target node is dead, so the entire operation runs in a worker thread
    with shutdown(wait=False) to avoid blocking the caller.

    Server-side failover is handled by the C++ layer (CHI_CLIENT_TRY_NEW_SERVERS).
    """
    _ensure_init()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_do_monitor, pool_query, query)
    try:
        results = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        raise TimeoutError(
            f"Monitor({pool_query!r}, {query!r}) timed out after {timeout}s"
        )
    pool.shutdown(wait=False)
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
    """Query system_stats for each node individually in parallel.

    Returns {node_id: entry_dict} for nodes that responded, skipping
    dead nodes instead of blocking the whole request.
    """
    _ensure_init()
    results = {}

    def _query_one(node_id):
        try:
            raw = _monitor(f"physical:{node_id}", f"system_stats:{min_event_id}", timeout=timeout)
            # Use the last (most recent) entry from the ring buffer
            latest = None
            for cid, entries in raw.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            latest = entry
            if latest is not None:
                return (node_id, latest)
        except Exception:
            pass
        return (node_id, None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(node_ids)) as pool:
        futures = [pool.submit(_query_one, nid) for nid in node_ids]
        for f in concurrent.futures.as_completed(futures, timeout=timeout + 2):
            try:
                nid, entry = f.result()
                if entry is not None:
                    results[nid] = entry
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


def shutdown_node(ip_address, grace_period_ms=5000):
    """Shutdown a remote node via SSH.

    First attempts graceful shutdown with 'chimaera runtime stop'.
    Exit codes 0, 134 (SIGABRT), and 255 (SSH closed) are success.
    If graceful stop fails, forcefully kills the runtime process.
    """
    import subprocess

    # Step 1: Try graceful shutdown
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        ip_address,
        f"export PATH=/workspace/build/bin:$PATH LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH && chimaera runtime stop --grace-period {grace_period_ms}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode in (0, 134, 255):
            return {
                "success": True,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
    except subprocess.TimeoutExpired:
        pass

    # Step 2: Graceful stop failed or timed out — force kill
    # Use -x (exact name match) to avoid killing the container's main bash
    # process, which also has 'chimaera runtime start' in its command line.
    kill_cmd = [
        "ssh", "-T", "-n",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        ip_address,
        "pkill -9 -x chimaera 2>/dev/null; sleep 0.5; "
        "! pgrep -x chimaera >/dev/null 2>&1",
    ]
    try:
        kill_result = subprocess.run(
            kill_cmd, capture_output=True, text=True, timeout=10)
        # pgrep exits 1 when no processes found (good — runtime is dead)
        return {
            "success": True,
            "returncode": kill_result.returncode,
            "stdout": "Force-killed runtime process",
            "stderr": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Both graceful and forced shutdown failed: {exc}",
        }


def restart_node(ip_address, port=9413):
    """Restart a node's Chimaera runtime (non-blocking).

    Detects whether the target is the local node (by comparing against
    NODE_IP env var) and uses a direct subprocess.Popen for local restarts
    instead of SSH, which avoids SSH session hangs during leader init.

    Uses ``chimaera runtime restart`` (WAL replay) so the node rejoins the
    existing cluster rather than trying to bootstrap a new one.

    Returns immediately after launching the process — the dashboard's
    topology polling (TCP-based) will detect when the node comes back.
    This prevents blocking the Flask server and freezing the entire UI.
    """
    import os
    import time
    import subprocess

    log_file = "/tmp/chimaera_restart.log"
    local_ip = os.environ.get("NODE_IP", "")
    is_local = (ip_address == local_ip)

    print(f"[restart_node] Starting restart for {ip_address}:{port} "
          f"(local={is_local})", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Kill any existing runtime process to free the port.
    # Use -x (exact executable name) to avoid killing the container's
    # main bash process whose command line also contains 'chimaera'.
    # ------------------------------------------------------------------
    if is_local:
        print("[restart_node] Step 1: Killing local runtime", flush=True)
        try:
            kill_result = subprocess.run(
                ["pkill", "-9", "-x", "chimaera"],
                capture_output=True, text=True, timeout=5)
            print(f"[restart_node] Step 1: Kill rc={kill_result.returncode}",
                  flush=True)
        except Exception as exc:
            print(f"[restart_node] Step 1: Kill exception (OK): {exc}",
                  flush=True)
        time.sleep(1)
    else:
        kill_cmd = [
            "ssh", "-T", "-n",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            ip_address,
            "pkill -9 -x chimaera 2>/dev/null; sleep 1",
        ]
        print(f"[restart_node] Step 1: Killing via SSH: "
              f"{' '.join(kill_cmd)}", flush=True)
        try:
            kill_result = subprocess.run(
                kill_cmd, capture_output=True, text=True, timeout=10)
            print(f"[restart_node] Step 1: Kill rc={kill_result.returncode}",
                  flush=True)
        except Exception as exc:
            print(f"[restart_node] Step 1: Kill exception (OK): {exc}",
                  flush=True)

    # ------------------------------------------------------------------
    # Step 2: Launch the runtime in the background and return immediately.
    # The dashboard's topology polling (TCP liveness) will detect when the
    # node is back up — no need to block here.
    # ------------------------------------------------------------------
    if is_local:
        env = os.environ.copy()
        env["PATH"] = f"/workspace/build/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"/workspace/build/bin:{env.get('LD_LIBRARY_PATH', '')}")
        print("[restart_node] Step 2: Launching local runtime via Popen",
              flush=True)
        try:
            with open(log_file, "w") as log_f:
                subprocess.Popen(
                    ["/workspace/build/bin/chimaera", "runtime", "restart"],
                    stdin=subprocess.DEVNULL,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
        except Exception as exc:
            print(f"[restart_node] Step 2: Popen failed: {exc}", flush=True)
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Failed to launch runtime: {exc}",
            }
    else:
        env_parts = [
            "export PATH=/workspace/build/bin:$PATH",
            "LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH",
        ]
        for var in ("CHI_SERVER_CONF", "CHI_NUM_CONTAINERS",
                    "CONTAINER_HOSTFILE"):
            val = os.environ.get(var)
            if val:
                env_parts.append(f"{var}={val}")
            print(f"[restart_node] env {var}={val!r}", flush=True)
        env_str = " ".join(env_parts)
        remote_cmd = (
            f"{env_str} && "
            f"nohup setsid /workspace/build/bin/chimaera runtime restart "
            f"</dev/null >{log_file} 2>&1 & disown; exit 0"
        )
        cmd = [
            "ssh", "-T", "-n",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            ip_address,
            remote_cmd,
        ]
        print(f"[restart_node] Step 2: Starting via SSH: "
              f"{' '.join(cmd)}", flush=True)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15)
        except Exception as exc:
            print(f"[restart_node] Step 2: SSH exception: {exc}", flush=True)
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"SSH start command failed: {exc}",
            }
        if result.returncode != 0:
            print(f"[restart_node] Step 2: SSH rc={result.returncode}",
                  flush=True)
            return {
                "success": False,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
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

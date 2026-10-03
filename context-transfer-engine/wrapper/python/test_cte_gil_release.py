#!/usr/bin/env python3
"""GIL release and typed-future regressions for clio_cte_core_ext.

Reproduces iowarp/clio-core#1096 and #793 against a real, separately spawned
runtime daemon:

  1. Async query results: AsyncTagQuery / AsyncBlobQuery used to return no
     rows (and client-mode AsyncGetBlob misses zeros) because the binding
     waited on a type-erased Future<Task>. They must return the rows.
  2. GIL release: with the daemon suspended, a blocking call on a worker
     thread must not freeze the main thread (~0 heartbeats before the fix).
  3. Future.wait(max_sec) must honour max_sec against a suspended daemon AND
     a dead one, returning None instead of blocking.

Steps 2-3 need psutil (to suspend the daemon); without it they are skipped.

Usage:
    python3 test_cte_gil_release.py      # run from the dir holding the module
"""
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.getcwd())  # prefer the freshly-built module next to us
if os.name == "nt":
    # Python 3.8+ no longer resolves an extension's dependent DLLs via PATH;
    # a dev build keeps them (vcpkg, the module's siblings) on PATH.
    for _d in [os.getcwd()] + os.environ.get("PATH", "").split(os.pathsep):
        if _d and os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass

SUSPEND_SEC = 2.0     # how long the daemon stays frozen
MIN_HEARTBEATS = 50   # 10 ms beats; a GIL-holding call yields ~0-1


def free_port() -> int:
    """Pick a base port whose +1 and +3 neighbours are free too."""
    for base in range(11413, 11900, 7):
        ok = True
        for p in (base, base + 1, base + 3):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
                    break
        if ok:
            return base
    raise RuntimeError("no free port cluster")


def write_config(port: int) -> str:
    """Write a RAM-only CTE config (no pyyaml dependency)."""
    cfg = f"""networking:
  port: {port}
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "ram::gil_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "64MB"
  - mod_name: clio_cte_core
    pool_name: clio_cte_core
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "ram::gil_ram_tier"
        bdev_type: ram
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: max_bw
"""
    fd, path = tempfile.mkstemp(prefix="clio_gil_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(cfg)
    return path


READY_MARK = "CTE Core container created and initialized"


def wait_for_daemon(log_path: str, daemon, timeout: float = 90.0) -> None:
    """Block until the daemon has composed its CTE pool.

    Connecting straight after spawning races startup: on macOS clio_init
    does not retry a not-yet-bound Unix socket, and a client that attaches
    before the compose finishes finds no storage targets (PutBlob rc=11).
    The daemon logs READY_MARK once the CTE container is up.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if daemon.poll() is not None:
            raise RuntimeError(f"daemon exited early (rc={daemon.returncode})")
        try:
            with open(log_path, errors="replace") as f:
                if READY_MARK in f.read():
                    time.sleep(0.5)  # let the compose task reply
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError("daemon did not come up; see " + log_path)


def find_clio_run(bin_dir: str) -> str:
    """Locate the clio_run executable next to the module."""
    for name in ("clio_run.exe", "clio_run"):
        p = os.path.join(bin_dir, name)
        if os.path.exists(p):
            return p
    raise RuntimeError(f"clio_run not found in {bin_dir}")


def heartbeats_during(fn) -> tuple:
    """Run fn on a worker thread; count 10 ms main-thread heartbeats."""
    done = threading.Event()
    err = []

    def work():
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - reported below
            err.append(e)
        finally:
            done.set()

    # Start the clock BEFORE the thread: a binding that holds the GIL keeps
    # this thread from running at all until the call returns.
    beats, t0 = 0, time.monotonic()
    threading.Thread(target=work, daemon=True).start()
    while not done.is_set():
        time.sleep(0.01)
        beats += 1
    return beats, time.monotonic() - t0, err


def resume_later(pid: int, delay: float) -> None:
    """Resume a suspended pid from ANOTHER process: a Python thread here
    would need the very GIL a non-releasing binding holds."""
    subprocess.Popen([sys.executable, "-c",
                      "import time, psutil; time.sleep(%f); "
                      "psutil.Process(%d).resume()" % (delay, pid)])


def check(cond: bool, msg: str, failures: list) -> None:
    print(("PASS: " if cond else "FAIL: ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def main() -> int:
    bin_dir = os.getcwd()
    port = free_port()
    cfg = write_config(port)
    env = dict(os.environ, CLIO_SERVER_CONF=cfg, CLIO_PORT=str(port),
               CLIO_BIND_ADDR="127.0.0.1", CLIO_REPO_PATH=bin_dir)
    os.environ.update(CLIO_SERVER_CONF=cfg, CLIO_PORT=str(port),
                      CLIO_BIND_ADDR="127.0.0.1", CLIO_REPO_PATH=bin_dir,
                      CLIO_WITH_RUNTIME="0", CLIO_WAIT_SERVER="60",
                      CLIO_CLIENT_RETRY_TIMEOUT="5")
    # Import only after the environment is final: a module linked against a
    # different C runtime snapshots the environment when it loads.
    import clio_cte_core_ext as cte
    daemon_log = cfg + ".daemon.log"
    log_f = open(daemon_log, "w")
    daemon = subprocess.Popen([find_clio_run(bin_dir), "start"], env=env,
                              stdout=log_f, stderr=subprocess.STDOUT)
    failures = []
    try:
        wait_for_daemon(daemon_log, daemon)
        assert cte.clio_init(cte.RuntimeMode.kClient, False), "clio_init"
        assert cte.initialize_cte(cfg, cte.PoolQuery.Dynamic()), "init cte"
        client = cte.get_cte_client()
        tag = cte.Tag("gil_probe_tag")
        payload = bytes(range(256)) * 16
        tag.PutBlob("b0", payload, 0)
        tag.PutBlob("b1", payload, 0)
        tid = tag.GetTagId()

        # 1. Typed futures: results must arrive.
        tags = client.AsyncTagQuery("gil_probe_.*").result()
        check("gil_probe_tag" in tags, f"AsyncTagQuery rows {tags}", failures)
        blobs = client.AsyncBlobQuery("gil_probe_.*", "b.*").result()
        check(sorted(b for _, b in blobs) == ["b0", "b1"],
              f"AsyncBlobQuery rows {blobs}", failures)
        names = client.AsyncGetContainedBlobs(tid).result()
        check(sorted(names) == ["b0", "b1"],
              f"AsyncGetContainedBlobs {names}", failures)
        size = client.AsyncGetBlobSize(tid, "b0").result()
        check(size == len(payload), f"AsyncGetBlobSize {size}", failures)
        data = client.AsyncGetBlob(tid, "b1", len(payload), 0).result()
        check(data == payload, "AsyncGetBlob bytes match", failures)
        new_tid = client.AsyncGetOrCreateTag("gil_probe_tag").result()
        # UniqueId has no Python __eq__; compare its fields.
        check((new_tid.major_, new_tid.minor_) == (tid.major_, tid.minor_),
              f"AsyncGetOrCreateTag -> {new_tid.major_}.{new_tid.minor_}",
              failures)
        check(client.AsyncDelBlob(tid, "b0").wait() == 0,
              "Future.wait returns the return code", failures)

        try:
            import psutil
        except ImportError:
            print("SKIP: psutil unavailable; GIL/timeout checks skipped")
            return 1 if failures else 0
        proc = psutil.Process(daemon.pid)

        # 2. Blocking calls release the GIL while the daemon is frozen.
        blocking = {
            "Tag(name)": lambda: cte.Tag("gil_probe_new_tag"),
            "Tag.GetContainedBlobs": tag.GetContainedBlobs,
            "Tag.GetBlob": lambda: tag.GetBlob("b1", len(payload), 0),
            "Tag.PutBlob": lambda: tag.PutBlob("b2", payload, 0),
            "Client.TagQuery": lambda: client.TagQuery("gil_.*"),
            "Client.DelBlob": lambda: client.DelBlob(tid, "b2"),
            "AsyncGetBlob.result": lambda: client.AsyncGetBlob(
                tid, "b1", len(payload), 0).result(),
        }
        for name, fn in blocking.items():
            proc.suspend()
            resume_later(proc.pid, SUSPEND_SEC)
            beats, el, err = heartbeats_during(fn)
            # A call served from the client's SHM cache never reaches the
            # frozen daemon and returns at once -- also fine.
            check(not err and (beats >= MIN_HEARTBEATS
                               or el < SUSPEND_SEC / 2),
                  f"{name}: {beats} heartbeats in {el:.2f}s "
                  f"(err={err or None})", failures)

        # 3a. wait(max_sec) against a suspended daemon.
        proc.suspend()
        fut = client.AsyncGetContainedBlobs(tid)
        t0 = time.monotonic()
        rc = fut.wait(0.5)
        el = time.monotonic() - t0
        proc.resume()
        check(rc is None and el < 1.5,
              f"wait(0.5) on a suspended daemon -> {rc} in {el:.2f}s",
              failures)
        check(fut.wait(10.0) == 0, "wait() completes once resumed", failures)

        # 3b. wait(max_sec) against a dead daemon.
        # Freeze it first so the request cannot complete before the kill.
        proc.suspend()
        fut = client.AsyncGetContainedBlobs(tid)
        proc.kill()
        daemon.wait(10)
        t0 = time.monotonic()
        rc = fut.wait(2.0)
        el = time.monotonic() - t0
        check(rc is None and el < 6.0,
              f"wait(2.0) on a dead daemon -> {rc} in {el:.2f}s", failures)

        # 3c. A blocking op against the dead daemon raises within the
        # client retry window instead of hanging or crashing (#722): a
        # killed runtime leaves its segment files behind, and reconnect
        # used to re-attach them and wait forever.
        t0 = time.monotonic()
        raised = False
        try:
            tag.PutBlob("after_death", payload, 0)
        except RuntimeError:
            raised = True
        el = time.monotonic() - t0
        check(raised and el < 30.0,
              f"PutBlob on a dead daemon raised={raised} in {el:.1f}s",
              failures)
    finally:
        if daemon.poll() is None:
            daemon.kill()
        log_f.close()
        for path in (cfg, daemon_log):
            try:
                os.remove(path)
            except OSError:
                pass
    print("RESULT:", "FAIL" if failures else "OK", flush=True)
    sys.stdout.flush()
    # The client still holds a connection to a daemon this test killed;
    # skip interpreter teardown so its outcome cannot mask the result.
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Async CTE client API tests.

Exercises the Client.Async*() methods, which return a Future you can overlap
other work against (submit N, then wait/result later). Covers:
  * AsyncPutBlob / AsyncGetBlob round-trip (byte-exact, non-UTF-8 payload)
  * overlapped submission (many in flight, then drain) — the whole point of async
  * Future.done() polling and Future.wait() return codes
  * AsyncDelBlob status

Usage:
    python3 test_cte_async.py
    CLIO_WITH_RUNTIME=0 python3 test_cte_async.py   # external runtime
"""
import os
import socket
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())  # prefer the freshly-built module next to us


def should_initialize_runtime() -> bool:
    val = os.getenv("CLIO_WITH_RUNTIME")
    return val is None or str(val).lower() not in ("0", "false", "no", "off")


def setup_environment_paths(cte) -> None:
    module_file = getattr(cte, "__file__", None)
    if not module_file:
        return
    bin_dir = os.path.dirname(os.path.abspath(module_file))
    os.environ["CLIO_REPO_PATH"] = bin_dir
    existing = os.getenv("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir


def find_available_port(start: int = 9309, end: int = 9380) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports in {start}-{end}")


def initialize_runtime(cte) -> bool:
    import yaml
    config = {
        "networking": {"port": find_available_port()},
        "runtime": {"num_threads": 4, "queue_depth": 1024},
        "compose": [
            {"mod_name": "clio_bdev", "pool_name": "ram::chi_default_bdev",
             "pool_query": "local", "pool_id": "301.0",
             "bdev_type": "ram", "capacity": "128MB"},
            {"mod_name": "clio_cte_core", "pool_name": "clio_cte_core",
             "pool_query": "local", "pool_id": "512.0",
             "storage": [{"path": "ram::cte_ram_tier1", "bdev_type": "ram",
                          "capacity_limit": "128MB", "score": 1.0}],
             "dpe": {"dpe_type": "max_bw"}},
        ],
    }
    cfg_path = os.path.join(tempfile.gettempdir(), "clio_async_conf.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
    os.environ["CLIO_SERVER_CONF"] = cfg_path
    if not cte.clio_init(cte.RuntimeMode.kClient, True):
        return False
    time.sleep(0.5)
    return cte.initialize_cte(cfg_path, cte.PoolQuery.Dynamic())


def run_test(cte) -> int:
    client = cte.get_cte_client()
    tag_id = cte.Tag("async_tag").GetTagId()
    payload = bytes(range(256)) * 32  # 8 KiB, deliberately NOT valid UTF-8

    # --- single AsyncPutBlob -> AsyncGetBlob round-trip --------------------
    pf = client.AsyncPutBlob(tag_id, "blob0", payload, 0)
    assert pf.wait() == 0, "AsyncPutBlob returned non-zero"

    gf = client.AsyncGetBlob(tag_id, "blob0", len(payload), 0)
    got = gf.result()
    assert isinstance(got, bytes), f"result must be bytes, got {type(got).__name__}"
    assert got == payload, "async round-trip byte mismatch"
    print(f"OK single async round-trip ({len(got)} bytes)")

    # --- overlapped submission: many puts in flight, then drain -----------
    n = 16
    blobs = [f"blob_{i}" for i in range(n)]
    datas = [bytes([i]) * (1024 + i) for i in range(n)]
    put_futs = [client.AsyncPutBlob(tag_id, blobs[i], datas[i], 0)
                for i in range(n)]  # all submitted before any wait
    for i, f in enumerate(put_futs):
        assert f.wait() == 0, f"overlapped put {i} failed"
    print(f"OK {n} overlapped async puts")

    get_futs = [client.AsyncGetBlob(tag_id, blobs[i], len(datas[i]), 0)
                for i in range(n)]  # all in flight at once
    for i, f in enumerate(get_futs):
        assert f.result() == datas[i], f"overlapped get {i} byte mismatch"
    print(f"OK {n} overlapped async gets (byte-exact)")

    # --- done() polling ----------------------------------------------------
    pf2 = client.AsyncPutBlob(tag_id, "blob_poll", payload, 0)
    spins = 0
    while not pf2.done():
        spins += 1
        if spins > 1_000_000:
            raise AssertionError("Future.done() never became True")
    assert pf2.wait() == 0
    print("OK Future.done() polling")

    # --- AsyncDelBlob status ----------------------------------------------
    df = client.AsyncDelBlob(tag_id, "blob0")
    assert df.wait() == 0, "AsyncDelBlob returned non-zero"
    print("OK async delete")

    # --- list-result futures (query / telemetry) --------------------------
    tq = client.AsyncTagQuery(".*", 0, cte.PoolQuery.Local())
    tags = tq.result()
    assert isinstance(tags, list), f"AsyncTagQuery result not a list: {type(tags).__name__}"
    assert "async_tag" in tags, f"expected tag missing from {tags}"

    bq = client.AsyncBlobQuery(".*", ".*", 0, cte.PoolQuery.Local())
    pairs = bq.result()
    assert isinstance(pairs, list), "AsyncBlobQuery result not a list"

    tel = client.AsyncPollTelemetryLog(0)
    assert isinstance(tel.result(), list), "AsyncPollTelemetryLog result not a list"
    print("OK async list-result futures (tag/blob query, telemetry)")

    return 0


def main() -> int:
    try:
        import clio_cte_core_ext as cte
    except ImportError as e:
        print(f"FAIL: cannot import clio_cte_core_ext: {e}")
        return 1
    if should_initialize_runtime():
        setup_environment_paths(cte)
        if not initialize_runtime(cte):
            print("FAIL: runtime initialization")
            return 1
    return run_test(cte)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL: {e}")
        sys.exit(1)

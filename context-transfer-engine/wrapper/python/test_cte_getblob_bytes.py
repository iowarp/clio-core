#!/usr/bin/env python3
"""Regression test for Tag.GetBlob returning raw bytes.

The Python binding once returned std::string, which nanobind decodes as UTF-8 —
raising UnicodeDecodeError on any non-text blob. This puts a deliberately
non-UTF-8 payload via PutBlob and verifies GetBlob returns it as `bytes`,
byte-exact. Against the old binding this test errors out (UnicodeDecodeError);
against the fix it passes.

Usage:
    python3 test_cte_getblob_bytes.py
    CLIO_WITH_RUNTIME=0 python3 test_cte_getblob_bytes.py   # external runtime
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


def find_available_port(start: int = 9209, end: int = 9280) -> int:
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
    cfg_path = os.path.join(tempfile.gettempdir(), "clio_getblob_bytes_conf.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
    os.environ["CLIO_SERVER_CONF"] = cfg_path
    if not cte.chimaera_init(cte.ChimaeraMode.kClient, True):
        return False
    time.sleep(0.5)
    return cte.initialize_cte(cfg_path, cte.PoolQuery.Dynamic())


def run_test(cte) -> int:
    tag = cte.Tag("getblob_bytes_roundtrip")
    payload = bytes(range(256)) * 64  # 16 KiB, deliberately NOT valid UTF-8
    tag.PutBlob("blob", payload, 0)

    size = tag.GetBlobSize("blob")
    assert size == len(payload), f"GetBlobSize {size} != {len(payload)}"

    got = tag.GetBlob("blob", size, 0)
    assert isinstance(got, bytes), f"GetBlob must return bytes, got {type(got).__name__}"
    assert got == payload, "binary round-trip mismatch"
    print(f"OK GetBlob returned {len(got)} bytes (byte-exact, non-UTF-8 round-trip)")
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

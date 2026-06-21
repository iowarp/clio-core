#!/usr/bin/env python3
"""
Python bindings smoke test for CTE's TemporalSearch API.

Puts blobs in three time-stamped batches (past / present / future relative
to a synthetic "now"), then exercises TemporalSearch with various time
windows and asserts:

  - The returned object is a list[TemporalSearchResult].
  - Results are sorted ascending by last_modified (handler invariant).
  - Open-ended bounds (0) are treated as "no constraint on that side".
  - max_entries cap is respected.
  - A window that matches nothing returns [].
  - tag_regex and blob_regex filtering work correctly.

The test controls timestamps by writing blobs with a known
last_modified value.  Because CTE sets last_modified on every PutBlob
call from the wall clock we cannot inject an arbitrary timestamp
through the normal PutBlob path.  Instead we put the blobs, record
the wall-clock ns *around* each group's writes, and use those
brackets as the time window — leaving a small guard band so clock
jitter doesn't produce flaky results.

Usage:
    python3 test_cte_temporal_search.py
    CLIO_WITH_RUNTIME=0 python3 test_cte_temporal_search.py   # external runtime

Environment:
    CLIO_WITH_RUNTIME (default 1) — start runtime inside this process
    CLIO_SERVER_CONF              — override the generated test config
"""

import os
import socket
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())


# ---------------------------------------------------------------------------
# Setup helpers (mirrors test_cte_semantic_search.py)
# ---------------------------------------------------------------------------

def should_initialize_runtime() -> bool:
    val = os.getenv("CLIO_WITH_RUNTIME")
    if val is None:
        return True
    return str(val).lower() not in ("0", "false", "no", "off")


def setup_environment_paths(cte) -> None:
    module_file = getattr(cte, "__file__", None)
    if not module_file:
        return
    bin_dir = os.path.dirname(os.path.abspath(module_file))
    os.environ["CLIO_REPO_PATH"] = bin_dir
    existing = os.getenv("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{bin_dir}:{existing}" if existing else bin_dir
    )


def find_available_port(start: int = 9129, end: int = 9200) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports in {start}-{end}")


def generate_test_config() -> str:
    import yaml
    tmp = tempfile.gettempdir()
    port = find_available_port()
    config = {
        "networking": {"port": port},
        "runtime": {"num_threads": 4, "queue_depth": 1024},
        "compose": [
            {
                "mod_name": "clio_bdev",
                "pool_name": "ram::chi_default_bdev",
                "pool_query": "local",
                "pool_id": "301.0",
                "bdev_type": "ram",
                "capacity": "128MB",
            },
            {
                "mod_name": "clio_cte_core",
                "pool_name": "clio_cte_core",
                "pool_query": "local",
                "pool_id": "512.0",
                "storage": [
                    {
                        "path": "ram::cte_ram_tier1",
                        "bdev_type": "ram",
                        "capacity_limit": "128MB",
                        "score": 1.0,
                    }
                ],
                "dpe": {"dpe_type": "max_bw"},
            },
        ],
    }
    cfg_path = os.path.join(tmp, "clio_temporal_conf.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
    os.environ["CLIO_SERVER_CONF"] = cfg_path
    return cfg_path


def initialize_runtime(cte) -> bool:
    config_path = generate_test_config()
    if not cte.clio_init(cte.RuntimeMode.kClient, True):
        return False
    time.sleep(0.5)
    if not cte.initialize_cte(config_path, cte.PoolQuery.Dynamic()):
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS = 1_000_000_000  # nanoseconds per second


def now_ns() -> int:
    # CTE's GetCurrentTimeNs() uses std::chrono::steady_clock, which maps to
    # CLOCK_MONOTONIC on Linux. time.monotonic_ns() uses the same clock, so
    # the brackets we record here are directly comparable to blob timestamps.
    return time.monotonic_ns()


def put_group(tag, prefix: str, count: int, payload: bytes = b"x") -> tuple[int, int]:
    """Put `count` blobs named ``<prefix>_0`` … ``<prefix>_{count-1}``.

    Returns (begin_ns, end_ns) wall-clock brackets bracketing the writes.
    A 10 ms guard band is subtracted/added so the caller can use these
    as TemporalSearch time windows without hitting clock-jitter edges.
    """
    guard = 10 * 1_000_000  # 10 ms in ns
    begin = now_ns() - guard
    for i in range(count):
        tag.PutBlob(f"{prefix}_{i}", payload, 0)
    end = now_ns() + guard
    return begin, end


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run_test(cte) -> int:
    client = cte.get_cte_client()

    # ------------------------------------------------------------------ #
    # 1. Write three groups with measurable gaps between them.            #
    # ------------------------------------------------------------------ #
    tag_a = cte.Tag("temporal_tag_a")
    tag_b = cte.Tag("temporal_tag_b")

    # Group 1 — "old" blobs in tag_a
    g1_begin, g1_end = put_group(tag_a, "old", 3)
    time.sleep(0.05)   # 50 ms gap so timestamps are distinct

    # Group 2 — "mid" blobs in tag_a
    g2_begin, g2_end = put_group(tag_a, "mid", 3)
    time.sleep(0.05)

    # Group 3 — "new" blobs in tag_b (different tag)
    g3_begin, g3_end = put_group(tag_b, "new", 3)

    print(f"PUT 3 groups: old=[{g1_begin},{g1_end}]  "
          f"mid=[{g2_begin},{g2_end}]  new=[{g3_begin},{g3_end}]")

    # ------------------------------------------------------------------ #
    # 2. Basic type / sort invariants                                     #
    # ------------------------------------------------------------------ #
    results = client.TemporalSearch(
        tag_regex="temporal_tag_a",
        blob_regex=".*",
        time_begin=g1_begin,
        time_end=g2_end,
        pool_query=cte.PoolQuery.Local(),
    )

    assert isinstance(results, list), \
        f"Expected list, got {type(results).__name__}"
    print(f"GOT {len(results)} results for tag_a window [g1_begin, g2_end]")
    for i, r in enumerate(results):
        print(f"  #{i}  blob={r.blob_name}  ts={r.last_modified}  "
              f"tag_id=({r.tag_id.major_},{r.tag_id.minor_})")

    # Should contain both old_* and mid_* blobs (6 total)
    assert len(results) == 6, \
        f"Expected 6 results (old+mid), got {len(results)}"

    # Verify field presence
    for i, r in enumerate(results):
        assert hasattr(r, "tag_id"),       f"result {i} missing tag_id"
        assert hasattr(r, "blob_name"),    f"result {i} missing blob_name"
        assert hasattr(r, "last_modified"), f"result {i} missing last_modified"
        assert isinstance(r.blob_name, str)
        assert isinstance(r.last_modified, int)

    # Ascending sort invariant
    for i in range(1, len(results)):
        assert results[i].last_modified >= results[i - 1].last_modified, (
            f"results not sorted ascending: "
            f"results[{i-1}].last_modified={results[i-1].last_modified} > "
            f"results[{i}].last_modified={results[i].last_modified}")
    print("OK sort order (ascending last_modified)")

    # ------------------------------------------------------------------ #
    # 3. Only the "old" window                                           #
    # ------------------------------------------------------------------ #
    old_results = client.TemporalSearch(
        tag_regex="temporal_tag_a",
        blob_regex="^old_.*",
        time_begin=g1_begin,
        time_end=g1_end,
        pool_query=cte.PoolQuery.Local(),
    )
    assert len(old_results) == 3, \
        f"Expected 3 old blobs, got {len(old_results)}"
    for r in old_results:
        assert r.blob_name.startswith("old_"), \
            f"Unexpected blob in old window: {r.blob_name}"
    print("OK old-only window returned 3 blobs")

    # ------------------------------------------------------------------ #
    # 4. max_entries cap                                                 #
    # ------------------------------------------------------------------ #
    capped = client.TemporalSearch(
        tag_regex="temporal_tag_a",
        blob_regex=".*",
        time_begin=g1_begin,
        time_end=g2_end,
        max_entries=2,
        pool_query=cte.PoolQuery.Local(),
    )
    assert len(capped) == 2, \
        f"max_entries=2 not honored: got {len(capped)}"
    # Cap preserves the earliest entries (ascending sort then truncate)
    for r in capped:
        assert r.blob_name.startswith("old_"), \
            f"Expected oldest blobs after cap, got {r.blob_name}"
    print("OK max_entries=2 cap honored and kept earliest blobs")

    # ------------------------------------------------------------------ #
    # 5. Open lower bound (time_begin=0)                                 #
    # ------------------------------------------------------------------ #
    open_lower = client.TemporalSearch(
        tag_regex="temporal_tag_a",
        blob_regex=".*",
        time_begin=0,
        time_end=g1_end,
        pool_query=cte.PoolQuery.Local(),
    )
    assert len(open_lower) == 3, \
        f"Open-lower-bound expected 3, got {len(open_lower)}"
    print("OK open lower bound (time_begin=0) returned 3 blobs")

    # ------------------------------------------------------------------ #
    # 6. Open upper bound (time_end=0)                                   #
    # ------------------------------------------------------------------ #
    open_upper = client.TemporalSearch(
        tag_regex="temporal_tag_a",
        blob_regex=".*",
        time_begin=g2_begin,
        time_end=0,
        pool_query=cte.PoolQuery.Local(),
    )
    assert len(open_upper) == 3, \
        f"Open-upper-bound expected 3 (mid only), got {len(open_upper)}"
    for r in open_upper:
        assert r.blob_name.startswith("mid_"), \
            f"Expected mid blobs with open upper bound, got {r.blob_name}"
    print("OK open upper bound (time_end=0) returned 3 mid blobs")

    # ------------------------------------------------------------------ #
    # 7. tag_regex filters to a different tag                            #
    # ------------------------------------------------------------------ #
    tag_b_results = client.TemporalSearch(
        tag_regex="temporal_tag_b",
        blob_regex=".*",
        time_begin=g3_begin,
        time_end=g3_end,
        pool_query=cte.PoolQuery.Local(),
    )
    assert len(tag_b_results) == 3, \
        f"tag_b expected 3 blobs, got {len(tag_b_results)}"
    for r in tag_b_results:
        assert r.blob_name.startswith("new_"), \
            f"Expected new_ blobs in tag_b, got {r.blob_name}"
    print("OK tag_regex filtering (temporal_tag_b) returned 3 blobs")

    # ------------------------------------------------------------------ #
    # 8. Window that matches nothing returns []                          #
    # ------------------------------------------------------------------ #
    far_future = time.monotonic_ns() + 3600 * _NS
    empty = client.TemporalSearch(
        tag_regex=".*",
        blob_regex=".*",
        time_begin=far_future,
        time_end=far_future + _NS,
        pool_query=cte.PoolQuery.Local(),
    )
    assert isinstance(empty, list)
    assert len(empty) == 0, \
        f"Future window expected [], got {len(empty)} results"
    print("OK no-match window returned empty list")

    # ------------------------------------------------------------------ #
    # 9. blob_regex that matches nothing returns []                      #
    # ------------------------------------------------------------------ #
    no_blob = client.TemporalSearch(
        tag_regex=".*",
        blob_regex="^nonexistent_.*",
        time_begin=g1_begin,
        time_end=g3_end,
        pool_query=cte.PoolQuery.Local(),
    )
    assert isinstance(no_blob, list)
    assert len(no_blob) == 0, \
        f"Non-matching blob_regex expected [], got {len(no_blob)} results"
    print("OK non-matching blob_regex returned empty list")

    print("\nAll TemporalSearch tests PASSED")
    return 0


def main() -> int:
    if should_initialize_runtime():
        try:
            import clio_cte_core_ext as cte
        except ImportError as e:
            print(f"FAIL: cannot import clio_cte_core_ext: {e}")
            return 1
        setup_environment_paths(cte)
        if not initialize_runtime(cte):
            print("FAIL: runtime initialization")
            return 1
    else:
        import clio_cte_core_ext as cte

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

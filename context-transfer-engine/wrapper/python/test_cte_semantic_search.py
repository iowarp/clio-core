#!/usr/bin/env python3
"""
Python bindings smoke test for CTE's SemanticSearch API.

Puts 10 short documents under one tag via Tag.PutBlob, then calls
Client.SemanticSearch with a natural-language query restricted to the
matching blob names and verifies:

  - The returned object is a list[SemanticSearchResult].
  - Results are sorted by descending score (handler invariant).
  - The space/moon-themed documents (doc0 Apollo, doc5 Hubble) appear
    in the top-k when the query is about space exploration.

The labeling pipeline (CAE -> Ollama -> {name}_label blobs) is exercised
separately by the C++ test_cae_semantic_search; this test scores the
raw document text so it doesn't need Ollama to run.

Usage:
    python3 test_cte_semantic_search.py
    CLIO_WITH_RUNTIME=0 python3 test_cte_semantic_search.py   # external runtime

Environment:
    CLIO_WITH_RUNTIME (default 1) — start runtime inside this process
    CLIO_SERVER_CONF              — override the generated test config
"""

import os
import socket
import sys
import tempfile
import time

# Prefer the freshly-built module sitting next to us in build/bin over
# any older copy in the venv site-packages — same trick test_bindings.py
# uses. CTest sets WORKING_DIRECTORY to the .so's directory.
sys.path.insert(0, os.getcwd())


# ---------------------------------------------------------------------------
# Setup helpers (mirrors test_bindings.py)
# ---------------------------------------------------------------------------

def should_initialize_runtime() -> bool:
    val = os.getenv("CLIO_WITH_RUNTIME")
    if val is None:
        return True
    return str(val).lower() not in ("0", "false", "no", "off")


def setup_environment_paths(cte) -> None:
    """Tell the runtime where to find the ChiMod .so files."""
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
    """Write a compose-style YAML so the CTE core comes up with a RAM
    storage tier already configured. The compose block creates pool
    512.0 (clio_cte_core); CLIO_CTE_CLIENT_INIT later resolves to that
    existing pool by ID and the bench's PutBlob calls have a target."""
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
    cfg_path = os.path.join(tmp, "clio_semsearch_conf.yaml")
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
    # Storage targets come up automatically from the compose `devices:`
    # block in generate_test_config(); we don't call Client.RegisterTarget
    # explicitly here because that binding currently returns a
    # std_atomic<u32> that nanobind can't convert. (Unrelated pre-
    # existing issue — see test_bindings.py for the same workaround.)
    return True


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

CORPUS = [
    # 0 — space
    ("doc0",
     "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and "
     "Buzz Aldrin walked on the lunar surface while Michael Collins "
     "orbited overhead. The lunar exploration mission was televised."),
    # 1
    ("doc1",
     "The Wright brothers achieved the first controlled, sustained flight "
     "of a powered, heavier-than-air aircraft on December 17, 1903 at "
     "Kitty Hawk, North Carolina."),
    # 2
    ("doc2",
     "The fall of the Berlin Wall on November 9, 1989 marked the symbolic "
     "end of the Cold War division of Europe."),
    # 3
    ("doc3",
     "Marie Curie was a Polish and naturalized-French physicist and "
     "chemist who conducted pioneering research on radioactivity."),
    # 4
    ("doc4",
     "The Great Wall of China is a series of fortifications stretching "
     "across the historical northern borders of ancient Chinese states."),
    # 5 — space
    ("doc5",
     "The Hubble Space Telescope was launched into low Earth orbit in "
     "1990, revolutionizing our understanding of the cosmos including "
     "lunar imaging and distant galaxies."),
    # 6
    ("doc6",
     "The printing press was invented by Johannes Gutenberg around 1440 "
     "and helped catalyze the spread of knowledge during the Renaissance."),
    # 7
    ("doc7",
     "The Industrial Revolution began in Britain in the late 18th "
     "century with the mechanization of textile manufacturing."),
    # 8
    ("doc8",
     "The Roman Empire reached its territorial peak under Emperor Trajan "
     "in 117 AD, spanning much of Europe, North Africa, and the Middle East."),
    # 9
    ("doc9",
     "The Statue of Liberty was a gift from France to the United States "
     "dedicated in 1886, designed by Frederic Auguste Bartholdi."),
]


def run_test(cte) -> int:
    """Returns 0 on success, non-zero on failure. Asserts loudly."""

    # Put all 10 docs into one tag.
    tag = cte.Tag("semsearch_corpus")
    for name, body in CORPUS:
        tag.PutBlob(name, body.encode("utf-8"), 0)
    print(f"PUT {len(CORPUS)} documents under tag 'semsearch_corpus'")

    # SemanticSearch is a Client method, not a Tag method.
    client = cte.get_cte_client()
    query = "moon space telescope lunar exploration"

    # Type / sort invariants
    results = client.SemanticSearch(
        tag_regex=".*",
        blob_regex="^doc[0-9]+$",   # only the corpus blobs, no labels
        query_text=query,
        k=5,
        pool_query=cte.PoolQuery.Local(),
    )

    assert isinstance(results, list), \
        f"Expected list, got {type(results).__name__}"
    print(f"GOT {len(results)} results for query: {query!r}")
    for i, r in enumerate(results):
        print(f"  #{i}  blob={r.blob_name}  score={r.score:.6f}  "
              f"tag_id=({r.tag_id.major_},{r.tag_id.minor_})")

    assert len(results) > 0, "SemanticSearch returned 0 results"
    assert len(results) <= 5, f"k=5 cap not honored: got {len(results)}"

    for i, r in enumerate(results):
        assert hasattr(r, "tag_id"), f"result {i} missing tag_id"
        assert hasattr(r, "blob_name"), f"result {i} missing blob_name"
        assert hasattr(r, "score"), f"result {i} missing score"
        assert isinstance(r.blob_name, str)
        assert isinstance(r.score, float)

    for i in range(1, len(results)):
        assert results[i].score <= results[i - 1].score, (
            f"results not sorted descending: "
            f"results[{i-1}].score={results[i-1].score} < "
            f"results[{i}].score={results[i].score}")

    # Top hits should be space-themed. BM25 over short docs can be
    # noisy, but with "moon / space / telescope / lunar / exploration"
    # in the query, doc0 (Apollo) and doc5 (Hubble) overlap on
    # multiple terms — they should outrank Marie Curie or the Roman
    # Empire by a wide margin.
    relevant = {"doc0", "doc5"}
    top_names = {r.blob_name for r in results}
    assert relevant & top_names, (
        f"Expected at least one of {relevant} in top-{len(results)}; "
        f"got {top_names}")
    print(f"OK relevance check: top-5 contains "
          f"{sorted(relevant & top_names)}")

    # Empty-query corner case — every score should be 0, results count
    # may be the full corpus or capped by k, but the binding shouldn't
    # crash.
    empty = client.SemanticSearch(
        tag_regex=".*", blob_regex="^doc[0-9]+$",
        query_text="", k=3, pool_query=cte.PoolQuery.Local())
    assert isinstance(empty, list)
    for r in empty:
        assert r.score == 0.0, (
            f"empty query expected score=0, got {r.score} for {r.blob_name}")
    print(f"OK empty-query returned {len(empty)} zero-scored results")

    # Blob-regex that matches nothing should return [].
    none = client.SemanticSearch(
        tag_regex=".*", blob_regex="^nonexistent_.*",
        query_text="moon", k=5, pool_query=cte.PoolQuery.Local())
    assert isinstance(none, list)
    assert len(none) == 0, f"no-match expected [], got {len(none)} results"
    print("OK no-match returned empty list")

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

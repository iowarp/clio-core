"""Render a CLIO compose YAML for one composition.

A composition is the set of choices the live controller can make on a single node:
    placement    "dram" | "disk"       which CTE tier holds the stream
    cache        bool                  clio_cte_cache at the top of the chain
    indexer      bool                  clio_cte_indexer in the chain
    compression  "none" | codec name   clio_cte_compressor with default_compress

The SSD tier is long_term (a durable device; cache copies may only use temporary tiers, so they
stay in DRAM) and the cache keeps owner-local copies (owner_copies, single node): without both the
cache ChiMod is a pass-through on one node. SSC_DISK_PERSISTENCE / SSC_CACHE_OWNER_COPIES restore
the old setup.

lz4 is composed with the "fast" preset: CLIO's "balanced" lz4 is LZ4-HC, which is slower
than zstd and not what the SSC describes as lz4.

The chain is cache(563) -> indexer(564) -> compressor(562) -> core(512), with absent
stages skipped. Clients bind to the chain top through CLIO_CTE_POOL.
"""

import os

CORE, COMPRESSOR, CACHE, INDEXER = "512.0", "562.0", "563.0", "564.0"

DEFAULT = {"placement": "dram", "cache": False, "indexer": False, "compression": "none"}


def chain_top(c):
    for on, pid in ((c["cache"], CACHE), (c["indexer"], INDEXER),
                    (c["compression"] != "none", COMPRESSOR)):
        if on:
            return pid
    return CORE


def render(c, root, dram_gb=None, disk_gb=None):
    """root: directory for the disk tier and metadata logs (inside the container). Tier sizes
    default to 1 GB DRAM / 2 GB disk; SSC_DRAM_GB / SSC_DISK_GB set another deployment."""
    dram_gb = dram_gb or os.environ.get("SSC_DRAM_GB", "1")
    disk_gb = disk_gb or os.environ.get("SSC_DISK_GB", "2")
    # both tiers exist; placement is expressed by tier score (DPE max_bw fills the highest
    # score first), so the "other" tier still catches overflow like a real deployment
    dram_score, disk_score = (1.0, 0.2) if c["placement"] == "dram" else (0.2, 1.0)
    y = f"""networking:
  port: 9413
runtime:
  num_threads: 4
  queue_depth: 1024
viz:
  enabled: true
  port: 8080
  bind: "127.0.0.1"
compose:
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "{CORE}"
    storage:
      - path: "ram::cte_ram_tier"
        bdev_type: "ram"
        capacity_limit: "{dram_gb}g"
        score: {dram_score}
      - path: "{root}/cte_disk_tier.dat"
        bdev_type: "file"
        capacity_limit: "{disk_gb}GB"
        score: {disk_score}
        persistence_level: "{os.environ.get('SSC_DISK_PERSISTENCE', 'long_term')}"
    performance:
      metadata_log_path: "{root}/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
    targets:
      neighborhood: 1
      poll_period_ms: 1000
"""
    return y + interceptor_entries(c, root)


def interceptor_entries(c, root):
    """The interceptor pools of a composition (everything above core), as compose entries."""
    y = ""
    nxt = CORE
    if c["compression"] != "none":
        y += f"""  - mod_name: clio_cte_compressor
    pool_name: clio_cte_compressor
    pool_query: local
    pool_id: "{COMPRESSOR}"
    next_pool_id: "{nxt}"
    tracking_enabled: false
    default_compress: "{c['compression']}"
    default_preset: "{'fast' if c['compression'] == 'lz4' else 'balanced'}"
"""
        nxt = COMPRESSOR
    if c["indexer"]:
        y += f"""  - mod_name: clio_cte_indexer
    pool_name: clio_cte_indexer
    pool_query: local
    pool_id: "{INDEXER}"
    next_pool_id: "{nxt}"
    index_log_path: "{root}/cte_indexer_index"
"""
        nxt = INDEXER
    if c["cache"]:
        y += f"""  - mod_name: clio_cte_cache
    pool_name: clio_cte_cache
    pool_query: local
    pool_id: "{CACHE}"
    next_pool_id: "{nxt}"
    min_score: 0.5
    owner_copies: {os.environ.get('SSC_CACHE_OWNER_COPIES', 'true')}
"""
    return y


def interceptors_file(c, root):
    """A compose file with only the interceptor pools, for clio_run compose start/rm on a
    running runtime (core and its tiers stay up). None when the chain is core alone."""
    e = interceptor_entries(c, root)
    return "compose:\n" + e if e else None


def name(c):
    return (f"{c['placement']}-{'cache' if c['cache'] else 'nocache'}-"
            f"{'idx' if c['indexer'] else 'noidx'}-{c['compression']}")

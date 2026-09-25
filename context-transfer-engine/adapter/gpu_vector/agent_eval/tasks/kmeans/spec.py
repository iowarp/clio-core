"""Grading spec for the k-means task: decks, references, and checks."""

import os

# ITERS is the task default (4); GRADE_ITERS re-grades the same deliverables
# with longer timed runs (the flag is part of the task contract).
DIMS, K = 32, 16
ITERS = int(os.environ.get("GRADE_ITERS", "4"))

# (name, data_mb, cache_mb). The last two hold 8x more data than cache.
# Small decks: what the agents were told the grader uses (<= 512 MB).
SMALL_DECKS = [
    ("resident_128", 128, 256),
    ("ooc8x_256", 256, 32),
    ("ooc8x_512", 512, 64),
]
# Large decks: the same deliverables at laptop scale, graded alone on the idle
# GPU. Beyond the range the prompt promised, so failures here are reported
# separately. 6 GB is the most the in-core baseline can hold on an 8 GB GPU.
LARGE_DECKS = [
    ("resident_2g", 2048, 4096),
    ("ooc8x_4g", 4096, 512),
    ("ooc8x_6g", 6144, 768),
]
# XL deck: 16 GB through a 4 GB cache, long enough (>= 2 min per run at 100
# iterations) that nothing but steady-state paging shows. Too big for the
# in-core baseline on an 8 GB GPU, so only the paged reference applies.
XL_DECKS = [
    ("ooc4x_16g", 16384, 4096),
]
DECKS = {"large": LARGE_DECKS, "xl": XL_DECKS}.get(os.environ.get("GRADE_DECKS"), SMALL_DECKS)
INCORE_MAX_MB = 6144   # the most an 8 GB GPU holds with the whole point set resident


def bench_args(data_mb, cache_mb):
    """Flags passed to the agent's bench for one deck."""
    return ["--data-mb", str(data_mb), "--cache-mb", str(cache_mb),
            "--dims", str(DIMS), "--clusters", str(K), "--iters", str(ITERS)]


def cpu_ref_cmd(bindir, data_mb):
    """Independent CPU reference; prints `REF counts=... csum=...`."""
    return [f"{bindir}/kmeans_ref", str(data_mb), str(DIMS), str(K), str(ITERS)]


# Our paged implementation's best measured configuration at each budget.
# Its real cache is blocks x max(slots, 8) x page (the frame floor is 8 per
# block), so small budgets need small pages or few blocks; these were chosen
# by measuring the alternatives and verified to fault and evict out of core.
PAGED_REF_CFG = {  # (data_mb, cache_mb): (page_kb, blocks, slots)
    (128, 256): (512, 64, 8),
    (256, 32): (256, 16, 8),
    (512, 64): (256, 32, 8),
    (2048, 4096): (1024, 64, 64),
    (4096, 512): (1024, 64, 8),
    (6144, 768): (1024, 96, 8),
    (16384, 4096): (1024, 64, 64),
}


def paged_ref_cmd(data_mb, cache_mb):
    """Our paged implementation at the same cache budget and host-RAM store."""
    page_kb, blocks, slots = PAGED_REF_CFG[(data_mb, cache_mb)]
    assert page_kb * blocks * max(slots, 8) == cache_mb * 1024, "reference cache != budget"
    return ["/workspace/build-nc/bin/clio_kmeans_paged_newcoro",
            "--data-mb", str(data_mb), "--iters", str(ITERS),
            "--dims", str(DIMS), "--clusters", str(K),
            "--page-kb", str(page_kb), "--blocks", str(blocks), "--slots", str(slots),
            "--publish-seed", "--hbm-mb", "16"]


def incore_ref_cmd(bindir, data_mb):
    """The in-core baseline: the whole point set resident in device memory.
    None when the deck cannot fit on the device."""
    if data_mb > INCORE_MAX_MB:
        return None
    return ["mpirun", "-n", "1", f"{bindir}/kmeans_mpi",
            "--data-mb", str(data_mb), "--iters", str(ITERS),
            "--dims", str(DIMS), "--clusters", str(K)]


def parse_paged_ref_ms(out):
    """ms per iteration from the paged reference's KMEANS line."""
    for line in out.splitlines():
        if line.startswith("KMEANS mode="):
            kv = dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)
            return float(kv["ms"]) / ITERS
    return None


def parse_incore_ms(out):
    """ms per iteration from the in-core baseline's summary line."""
    import re
    m = re.search(r"(\d+) iters in ([0-9.]+) ms", out)
    return float(m.group(2)) / int(m.group(1)) if m else None


def check(result, ref, deck):
    """Correctness verdicts for one parsed RESULT against the CPU reference.
    Returns a dict of check name -> bool."""
    name, data_mb, cache_mb = deck
    out = {}
    out["counts_exact"] = result.get("counts") == ref["counts"]
    try:
        out["csum_close"] = abs(float(result["csum"]) - float(ref["csum"])) <= 0.5
    except (KeyError, ValueError):
        out["csum_close"] = False
    try:
        evicts = int(result["evicts"])
        faults = int(result["faults"])
    except (KeyError, ValueError):
        evicts = faults = -1
    if cache_mb < data_mb:
        # Out of core: the cache must actually have been exercised.
        out["evicted"] = evicts > 0
        out["faulted"] = faults > 0
    return out


def metric(result):
    """The performance number compared across implementations (lower is better)."""
    return float(result["ms_per_iter"])

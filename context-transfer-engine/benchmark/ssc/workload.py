"""Workload phases driven through the CTE Python client (bound to the chain top).

Every phase writes real data from one Gray-Scott campaign (gs_campaign.py; simulated once and
cached), as the stages of a simulation campaign would:
    gs_output        U/V field snapshots (4 MB), written once, never read back. The campaign
                     advances with the window index: early snapshots are smooth (zstd 1.7x),
                     late ones are patterned (zstd 1.1x)
    gs_analysis      1 MB tiles of recent snapshots, written once, then read many times
    gs_products      the snapshots as zstd-compressed products (incompressible)
    gs_log           the simulation's per-rank step log, flushed in 16 KB writes; searched later
    gs_archive       the campaign log archived in 4 MB chunks (4 GB, fits the tiers only when
                     compressed: zstd 9x, lz4 5x)
    gs_field_reread  (generalization) full snapshots re-read for analysis
    gs_events        (generalization) spot-detection event records, 4 KB writes; searched later

run_phase() records a client-side op trace (every application put/get, nothing the chain
does internally) and returns wall time, logical MB, MB/s and failed puts. delete_phase()
frees the blobs so repeats start from the same tier state.
"""

import os
import random
import time

import numpy as np

MB = 1 << 20
CAMPAIGN = os.environ.get("GS_CAMPAIGN", "/ssc-bench/gs_campaign")

PHASES = {
    # name: (blob_bytes, n_blobs, reads, data kind)
    "gs_output": (4 * MB, 256, 0, "snapshot"),
    "gs_analysis": (1 * MB, 64, 6000, "tile"),
    "gs_products": (4 * MB, 256, 0, "products.bin"),
    "gs_log": (16 * 1024, 8000, 0, "log.txt"),
    "gs_archive": (4 * MB, 1024, 0, "log.txt"),
    # generalization phases: added after every design decision was frozen, never used to tune
    "gs_field_reread": (4 * MB, 128, 512, "snapshot"),
    "gs_events": (4 * 1024, 16000, 0, "events.bin"),
    # second application (DLIO / h5bench-style), added after SSC v10 was frozen; never in the
    # SSC's node memory, so its windows only see experience from the Gray-Scott workloads
    "dl_samples": (256 * 1024, 4096, 12288, "products.bin"),   # training set, 3 shuffled epochs
    "dl_ckpt": (16 * MB, 32, 0, "weights"),                     # model checkpoint, write-only
    "h5_particles": (8 * MB, 256, 256, "snapshot2"),            # particle dump, re-read once
    # third application: WfCommons workflows (see WF_TRACES); data from wf_data.py
    "mt_project": (4 * MB, 180, 1872, "fits.bin"),              # Montage mProject images
    "mt_fit": (1024, 4230, 4230, "fit.txt"),                    # Montage mDiffFit plane fits
    "ep_fastq": (6 * MB, 165, 165, "fastq.txt"),                # Epigenomics split FASTQ
    "ep_map": (512 * 1024, 660, 660, "map.txt"),                # Epigenomics alignments
    "kg_vcf": (4 * MB, 512, 512, "vcf.txt"),                    # 1000Genome chromosome VCF
    "kg_indiv": (28 * 1024, 2600, 3900, "products.bin"),        # 1000Genome individuals (tar.gz)
}
# Where the WfCommons phases come from (WfInstances, Pegasus on Chameleon). File sizes and reuse
# (readers per file) are the traces'; counts are scaled so one window is a meaningful amount of I/O.
WF_TRACES = {
    "mt_project": "montage-chameleon-2mass-02d-001: mProject, 180 files, median 4.0 MB, 10.4 readers/file",
    "mt_fit": "montage-chameleon-2mass-02d-001: mDiffFit, 423 files, median 0.3 KB, 1 reader/file (x10)",
    "ep_fastq": "epigenomics-chameleon-hep-4seq-50k-001: fastqSplit/filterContams, 165 files, 6.4 MB, 1 reader",
    "ep_map": "epigenomics-chameleon-hep-4seq-50k-001: map, 165 files, median 488 KB, 1 reader (x4)",
    "kg_vcf": "1000genome-chameleon-10ch-250k-001: ALL.chrN.250000.vcf, 2.4 GB, 25 readers (2 GB window)",
    "kg_indiv": "1000genome-chameleon-10ch-250k-001: individuals, 260 files, median 27 KB, 1.5 readers (x10)",
}
BASE_PHASES = ("gs_output", "gs_analysis", "gs_products", "gs_log", "gs_archive")
GS_PHASES = BASE_PHASES + ("gs_field_reread", "gs_events")
APP2_PHASES = ("dl_samples", "dl_ckpt", "h5_particles")
APP3_PHASES = ("mt_project", "mt_fit", "ep_fastq", "ep_map", "kg_vcf", "kg_indiv")
# workloads the SSC may remember as node experience (the frozen design: Gray-Scott only)
MEMORY_PHASES = GS_PHASES

_cache = {}


def _load(name):
    if name not in _cache:
        path = os.path.join(CAMPAIGN, name)
        _cache[name] = np.load(path, mmap_mode="r") if name.endswith(".npy") else open(path, "rb").read()
    return _cache[name]


def _weights(i):
    """Checkpoint-like float32 tensor: eight consecutive snapshots as float32 with small noise in
    the low mantissa bits (trained weights compress poorly)."""
    key = ("weights", i)
    if key not in _cache:
        s = _load("snapshots.npy")
        x = np.concatenate([np.asarray(s[(i + j) % len(s)], dtype=np.float32).ravel() for j in range(8)])
        x = x + np.random.default_rng(i).standard_normal(x.size).astype(np.float32) * 3e-3
        _cache[key] = x.astype(np.float32).tobytes()
    return _cache[key]


def stage(seed):
    """Campaign position of window `seed` (the window index): the first snapshot it writes."""
    return min(60, seed * 8 // 5)


def payload(kind, n_bytes, seed, part):
    """Distinct buffer `part` (0-3) of a window. Snapshots and tiles come from the campaign
    position of the window; byte streams (logs, products, events) from an offset that moves on
    with the window, wrapping around the file."""
    if kind == "snapshot":
        return _load("snapshots.npy")[stage(seed) + part].tobytes()[:n_bytes]
    if kind == "snapshot2":
        s = _load("snapshots.npy")
        i = stage(seed) + part
        return (s[i % len(s)].tobytes() + s[(i + 1) % len(s)].tobytes())[:n_bytes]
    if kind == "weights":
        return _weights(stage(seed) + part)[:n_bytes]
    if kind == "tile":
        snap = _load("snapshots.npy")[stage(seed) + part % 2].tobytes()
        return snap[(part // 2) * n_bytes:(part // 2 + 1) * n_bytes]
    data = _load(kind)
    off = ((seed * 4 + part) * n_bytes) % max(1, len(data) - n_bytes)
    return data[off:off + n_bytes]


class ClientTrace:
    """Client-side op trace, the analogue of an adapter-level I/O trace (e.g. the HDF5 VOL trace)."""

    def __init__(self):
        self.ops = []          # (op, size)
        self.t0 = time.time()

    def summary(self):
        puts = [s for o, s in self.ops if o == "put"]
        gets = [s for o, s in self.ops if o == "get"]
        bp, bg = sum(puts), sum(gets)
        return {"n_put": len(puts), "n_get": len(gets), "bytes_put": bp, "bytes_get": bg,
                "mean_put": bp / len(puts) if puts else 0, "mean_get": bg / len(gets) if gets else 0,
                "reuse": bg / bp if bp else 0.0, "seconds": time.time() - self.t0}


def run_phase(cte, phase, tag_name, seed=0, trace=None, start=0, should_pause=None):
    """start: first blob index (a resumed window). should_pause(i): checked every 32 puts; when
    it returns True the window stops before its reads and the result carries paused_at."""
    blob_bytes, n_blobs, reads, kind = PHASES[phase]
    # a few distinct payloads, reused round-robin, so generation is not timed
    bufs = [payload(kind, blob_bytes, seed, i) for i in range(4)]
    tag = cte.Tag(tag_name)
    failed = 0
    t0 = time.perf_counter()
    for i in range(start, n_blobs):
        try:
            tag.PutBlob(f"b{i}", bufs[i % len(bufs)])
        except RuntimeError:
            failed += 1    # tier full: the put was refused
        if trace is not None:
            trace.ops.append(("put", blob_bytes))
        if should_pause is not None and (i + 1) % 32 == 0 and i + 1 < n_blobs and should_pause(i):
            wall = time.perf_counter() - t0
            mb = (i + 1 - start - failed) * blob_bytes / MB
            return {"phase": phase, "wall_s": wall, "write_s": wall, "read_s": 0.0, "logical_mb": mb,
                    "mb_per_s": mb / wall if wall else 0.0, "failed_puts": failed, "n_puts": i + 1 - start,
                    "paused_at": i + 1}
    t_write = time.perf_counter() - t0
    rng = random.Random(seed)
    t1 = time.perf_counter()
    failed_gets = 0
    for _ in range(reads):
        i = rng.randrange(n_blobs)
        try:
            tag.GetBlob(f"b{i}", blob_bytes)
        except RuntimeError:
            failed_gets += 1   # the blob's put was refused (tier full): the read finds nothing
        if trace is not None:
            trace.ops.append(("get", blob_bytes))
    t_read = time.perf_counter() - t1
    moved = (n_blobs - start - failed + reads) * blob_bytes / MB
    wall = t_write + t_read
    return {"phase": phase, "wall_s": wall, "write_s": t_write, "read_s": t_read, "logical_mb": moved,
            "mb_per_s": moved / wall if wall else 0.0, "failed_puts": failed, "n_puts": n_blobs - start,
            "failed_gets": failed_gets}


def delete_phase(cte, client, phase, tag_name):
    tag = cte.Tag(tag_name)
    tid = tag.GetTagId()
    for i in range(PHASES[phase][1]):
        client.DelBlob(tid, f"b{i}")

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
    "cy_season": (304 * 1024, 1600, 640, "cycles.txt"),         # Cycles crop-season outputs
    "sk_realign": (80 * 1024, 1000, 10000, "products.bin"),     # SoyKB realigned BAMs (compressed)
    "sk_gvcf": (43 * 1024, 2000, 4000, "vcf.txt"),              # SoyKB per-sample gVCFs
    "sg_decon": (4608, 4000, 4000, "sac.bin"),                  # Seismology receiver functions (SAC)
    "sra_fastq": (8 * MB, 192, 192, "fastq.txt"),               # SRAsearch fasterq-dump reads
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
    "cy_season": "cycles-chameleon-10l-1c-9p-001: cycles/fertilizer_increase, 1600 files, median 304 KB, 0.4 readers",
    "sk_realign": "soykb-chameleon-10fastq-10ch-001: indel_realign, 10 files, median 80 KB, 10 readers (x100)",
    "sk_gvcf": "soykb-chameleon-10fastq-10ch-001: haplotype_caller, 100 files, median 43 KB, 2 readers (x20)",
    "sg_decon": "seismology-chameleon-100p-001: sG1IterDecon, 100 files, median 4.6 KB, 1 reader (x40)",
    "sra_fastq": "srasearch-chameleon-10a-001: fasterq-dump, 20 files, median 511 MB, 1 reader (8 MB blobs, 1.5 GB window)",
}
# Large scale (SSC_SCALE=large, the 16 GB DRAM node setups): file counts from the largest WfInstances
# instances (Cycles 10l-3c-12p, Seismology 1100p, SoyKB 30fastq-20ch, SRAsearch 50a, 1000Genome 10
# chromosomes, Montage 2MASS at 10x the mosaic), per-file sizes as the traces record them, except SoyKB's
# realigned BAMs, which the trace run produced from subsampled reads (80 KB); they are scaled to 4 MB
# blobs of a 6 GB per-window BAM set. Reuse (reads per blob) is the trace's. Windows are 0.05-24 GB
# against 16 GB DRAM + 48 GB second tier, so the same decisions (fit, spill, compress, cache) arise at
# realistic sizes.
LARGE = {
    "mt_project": (4 * MB, 1500, 15600, "fits.bin"),            # 6 GB, 10.4 readers per image
    "mt_fit": (1024, 42300, 42300, "fit.txt"),                  # 41 MB of 1 KB plane fits
    "ep_fastq": (6 * MB, 1700, 1700, "fastq.txt"),              # 10 GB of split FASTQ
    "ep_map": (512 * 1024, 6600, 6600, "map.txt"),              # 3.3 GB of alignments (searched)
    "kg_vcf": (4 * MB, 6000, 6000, "vcf.txt"),                  # 24 GB: 10 chromosome VCFs (21x compressible)
    "kg_indiv": (28 * 1024, 26000, 39000, "products.bin"),      # 0.7 GB of per-individual results
    "cy_season": (304 * 1024, 16200, 6480, "cycles.txt"),       # 4.8 GB, the largest Cycles instance
    "sk_realign": (4 * MB, 1500, 15000, "products.bin"),        # 6 GB realigned BAMs, 10 readers each
    "sk_gvcf": (43 * 1024, 20000, 40000, "vcf.txt"),            # 0.86 GB of gVCFs (searched)
    "sg_decon": (4608, 11000, 11000, "sac.bin"),                # 1,100 stations x 10 traces
    "sra_fastq": (8 * MB, 3000, 3000, "fastq.txt"),             # 24 GB of downloaded reads (50 accessions)
}
SCALE = os.environ.get("SSC_SCALE", "small")
if SCALE == "large":
    PHASES.update(LARGE)

# the WfCommons campaign: seven workflows' I/O phases sharing one node
WF_PHASES = ("mt_project", "mt_fit", "ep_fastq", "ep_map", "kg_vcf", "kg_indiv",
             "cy_season", "sk_realign", "sk_gvcf", "sg_decon", "sra_fastq")
WF_NEW_PHASES = WF_PHASES[6:]
WF_FAMILY = {"mt": "Montage", "ep": "Epigenomics", "kg": "1000Genome", "cy": "Cycles", "sk": "SoyKB",
             "sg": "Seismology", "sra": "SRAsearch", "gs": "Gray-Scott", "dl": "DL", "h5": "DL"}


def family(phase):
    """The workflow (application) a phase belongs to; experience leaves the whole workflow out."""
    return WF_FAMILY[phase.split("_")[0]]
BASE_PHASES = ("gs_output", "gs_analysis", "gs_products", "gs_log", "gs_archive")
GS_PHASES = BASE_PHASES + ("gs_field_reread", "gs_events")
APP2_PHASES = ("dl_samples", "dl_ckpt", "h5_particles")
APP3_PHASES = ("mt_project", "mt_fit", "ep_fastq", "ep_map", "kg_vcf", "kg_indiv")
# workloads the SSC may remember as node experience (the frozen design: Gray-Scott only)
MEMORY_PHASES = GS_PHASES
LEAVE_OUT = "phase"     # "phase" (frozen v10/v12 runs) | "workflow": no phase of the same workflow


def remembered(phase):
    """The phases the SSC may use as experience when deciding for `phase`."""
    if LEAVE_OUT == "workflow":
        return [p for p in MEMORY_PHASES if family(p) != family(phase)]
    return [p for p in MEMORY_PHASES if p != phase]
# a mixed campaign: the phases of all three applications that want different stacks on one node
# (archive must compress, analysis/particles/checkpoint want the cache, alignments want the index,
# the VCF and fits want plain DRAM, the training samples no cache); no single fixed stack fits all
MIX_PHASES = ("gs_archive", "gs_analysis", "h5_particles", "dl_ckpt", "dl_samples", "ep_map", "kg_vcf", "mt_fit")

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

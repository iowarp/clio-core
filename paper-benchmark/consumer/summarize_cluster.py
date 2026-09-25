#!/usr/bin/env python3
"""Summarize a cluster.sh campaign: per arm, the write, the cold read of the
consumer's fields, the clustering, and how close its clusters are to the
reference's.

  summarize_cluster.py <out dir>      # writes <out>/summary.csv, prints a table

WRITE is figure 9's bar: the driver's total minus its input read and minus the
H2D staging and cuSZ's per-chunk resource-manager construction -- the union of
the compressor's per-chunk h2d intervals and the cuSZ wrapper's stamped
stream/manager/release spans (CLIO_CUSZ_PHASE_LOG), taken TOGETHER inside the
measured window, so time where both ran is removed once. Baseline reports its
own staging. READ is the driver's timer around
the get/decompress calls of the cold read-back; its io / decompress split is
the runtime's per-chunk phase log, summed (the read-back is sequential).
Totals carry one-off noise that has nothing to do with the codec -- a random
~300 ms stall on one chunk, and how much of the final fdatasync the kernel's
background writeback had already done (0.02-0.85 s) -- so beside each mean
+- sd the summary gives the MEDIAN over reps and the median PER-CHUNK wall
time over every chunk of every rep, which no single event can move.

REPEATED READS: cluster.sh reads each arm back READS times (read.log, then
read_2.log ...), each cold. `read` columns are READ 1 alone, comparable with
single-read campaigns; `reads_sum` is the per-rep total over every verified
read (what a consumer that returns N times pays); `read_all_med` is the
median single read over all reads of all reps.

CLUSTER is k-means over every frame, H2D of the decoded fields included. ARI is
the adjusted Rand index against the reference labels, averaged over frames;
seed ARI beside it is the reference's own seed-to-seed ARI, averaged the same
way -- k-means rerun on the ORIGINAL data from another seed -- the floor an
arm's ARI is read against. max|err| is the largest element error of any
decoded value against the dump.
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

ARMS = ["Baseline", "Best fixed nvCOMP", "Worst fixed nvCOMP", "ndzip", "cuSZp3",
        "cuSZ", "NeuroPress", "NeuroPress+DT"]


def slug(label):
    """Result-dir tag of an arm label, as cluster.sh derives it."""
    s = label.lower().translate(str.maketrans(" ()+", "____"))
    return re.sub("_+", "_", s).rstrip("_")


def intervals(path, start_col, ms_col, w0, w1, keep=lambda r: True):
    """[start, end) ns intervals of a stamped log, clipped to [w0, w1]."""
    iv = []
    if not os.path.exists(path):
        return iv
    with open(path) as f:
        for r in csv.DictReader(f):
            if not keep(r) or not r.get(ms_col) or not r.get(start_col):
                continue
            a = float(r[start_col])
            b = a + float(r[ms_col]) * 1e6
            a, b = max(a, w0), min(b, w1)
            if b > a:
                iv.append((a, b))
    return iv


def union_s(iv):
    """Total length of a set of ns intervals, overlaps counted once, in s."""
    tot, end = 0.0, None
    for a, b in sorted(iv):
        if end is None or a > end:
            tot += b - a
            end = b
        elif b > end:
            tot += b - end
            end = b
    return tot / 1e9


def write_side(d):
    """(write s, stored ratio, cuSZ setup s removed) from one run's logs."""
    log = open(os.path.join(d, "stdout.log")).read()
    t = re.search(r"time: read ([\d.]+) s\s+stage\+compress ([\d.]+) s\s+total ([\d.]+) s", log)
    w = re.search(r"window: start_ns (\d+)\s+end_ns (\d+)", log)
    s = re.search(r"stored \d+ blob\(s\), (\d+) B in -> (\d+) B", log)
    if not (t and w and s):
        return None
    w0, w1 = int(w.group(1)), int(w.group(2))
    setup = intervals(os.path.join(d, "cusz_setup.csv"), "start_ns", "ms", w0, w1)
    h2d = re.search(r"time: h2d ([\d.]+) s", log)
    if h2d:
        excluded_s = float(h2d.group(1)) + union_s(setup)
    else:
        excluded_s = union_s(setup + intervals(
            os.path.join(d, "phases.csv"), "h2d_start_ns", "h2d_ms", w0, w1,
            keep=lambda r: r["path"] == "write"))
    return (float(t.group(3)) - float(t.group(1)) - excluded_s,
            int(s.group(1)) / int(s.group(2)), union_s(setup))


def read_side(d):
    """(read s, io s, decompress s, bytes read, ratio of them, codecs)."""
    log = open(os.path.join(d, "read.log")).read()
    m = re.search(r"get\+decompress: ([\d.]+) ms", log)
    if not m or not log.count("VERIFIED"):
        return None
    io = dt = 0.0
    p = os.path.join(d, "phase_read.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            io += float(r["io_ms"] or 0)
            dt += float(r["decompress_ms"] or 0)
    inb = stored = 0
    codecs = {}
    for r in csv.DictReader(open(os.path.join(d, "blobs_read.csv"))):
        inb += int(r["bytes"])
        stored += int(r["stored"])
        codecs[r["codec"]] = codecs.get(r["codec"], 0) + 1
    if stored == 0:          # nothing was read back: a failed rep, not a fast one
        return None
    read_s = float(m.group(1)) / 1e3 - cusz_read_setup_s(p)
    return read_s, io / 1e3, dt / 1e3, inb, inb / stored, codecs


def cusz_read_setup_s(phase_log):
    """cuSZ decode-manager builds logged beside one read's phase log, in s.

    The read-back is one chunk at a time, so their sum IS elapsed time and
    comes straight off that read (see cluster.sh: construction is excluded on
    both sides).
    """
    p = phase_log[:-len(".csv")] + "_cusz_setup.csv"
    if not os.path.exists(p):
        return 0.0
    return sum(float(r["ms"]) for r in csv.DictReader(open(p))
               if r["phase"] in ("dmgr", "mgr", "release", "stream")) / 1e3


def later_reads(d):
    """(seconds, phase log) of reads 2..N of one rep, verified ones only."""
    out = []
    i = 2
    while os.path.exists(os.path.join(d, f"read_{i}.log")):
        log = open(os.path.join(d, f"read_{i}.log")).read()
        m = re.search(r"get\+decompress: ([\d.]+) ms", log)
        if m and log.count("VERIFIED") and "VERIFIED: 0 of" not in log:
            ph = os.path.join(d, f"phase_read_{i}.csv")
            out.append((float(m.group(1)) / 1e3 - cusz_read_setup_s(ph), ph))
        i += 1
    return out


STALL_MS = 100.0   # a chunk this slow is a one-off stall, not codec work


def stalls(path, path_col=None):
    """Chunks of a phase log slower than STALL_MS (optionally one path only)."""
    if not os.path.exists(path):
        return 0
    return sum(1 for r in csv.DictReader(open(path))
               if (path_col is None or r["path"] == path_col)
               and float(r["wall_ms"] or 0) > STALL_MS)


def chunk_walls(path, path_col=None):
    """Per-chunk wall times (ms) of a phase log, optionally one path only."""
    if not os.path.exists(path):
        return []
    return [float(r["wall_ms"]) for r in csv.DictReader(open(path))
            if (path_col is None or r["path"] == path_col) and r["wall_ms"]]


def median(v):
    """Median of a list, or '' when empty."""
    return st.median(v) if v else ""


def mean(v):
    """Mean of a list, or '' when empty."""
    return st.mean(v) if v else ""


def sd(v):
    """Sample standard deviation, or '' below two values."""
    return st.stdev(v) if len(v) > 1 else ""


def fmt(v, p=2):
    """mean +- sd of a list, or n/a."""
    if not v:
        return "n/a"
    return f"{st.mean(v):.{p}f} +- {(st.stdev(v) if len(v) > 1 else 0.0):.{p}f}"


def main():
    out = sys.argv[1]
    ref = json.load(open(os.path.join(out, "reference.json")))
    ref_read = sum(f["file_read_ms"] for f in ref["frames"]) / 1e3
    ref_km = sum(f["kmeans_ms"] for f in ref["frames"]) / 1e3
    nfr = len(ref["frames"])
    seed_ari = st.mean(f["seed_ari"] for f in ref["frames"])
    print(f"{ref['workload']}: {nfr} frame(s), fields {', '.join(ref['fields'])}, {ref['device']}")
    print(f"reference (the dumps, cold, no Clio): read {ref_read:.2f} s, k-means {ref_km:.2f} s,"
          f" seed-to-seed ARI {seed_ari:.4f} (mean over frames)")
    rows = []
    print(f"\n{'arm':20s} {'write s':>14s} {'w med':>6s} {'w/chunk':>7s} {'ratio':>6s} {'r-ratio':>7s}"
          f" {'read s':>14s} {'r med':>6s} {'r/chunk':>7s} {'n':>2s} {'sum reads':>9s}"
          f" {'io s':>6s} {'dec s':>6s} {'GB/s':>5s} {'kmeans s':>8s} {'ARI':>8s}"
          f" {'max|err|':>9s}  read codecs")
    for arm in ARMS:
        runs = sorted(glob.glob(os.path.join(out, "runs", slug(arm), "r*")))
        if not runs:
            continue
        W, R, IO, DT, KM, ratio, rratio, ARI, AGREE, ERR, GBS, SET, WS, RS, WC, RC, codecs = (
            [] for _ in range(17))
        RSUM, RALL, NREAD = [], [], []
        codecs = {}
        for d in runs:
            ws = write_side(d) if os.path.exists(os.path.join(d, "stdout.log")) else None
            if ws:
                W.append(ws[0]); ratio.append(ws[1]); SET.append(ws[2])
                WS.append(stalls(os.path.join(d, "phases.csv"), "write"))
                WC.extend(chunk_walls(os.path.join(d, "phases.csv"), "write"))
            rs = read_side(d) if os.path.exists(os.path.join(d, "read.log")) else None
            if rs:
                R.append(rs[0]); IO.append(rs[1]); DT.append(rs[2])
                GBS.append(rs[3] / rs[0] / 1e9); rratio.append(rs[4])
                more = later_reads(d)
                RS.append(stalls(os.path.join(d, "phase_read.csv"))
                          + sum(stalls(p) for _, p in more))
                RC.extend(chunk_walls(os.path.join(d, "phase_read.csv")))
                for _, p in more:
                    RC.extend(chunk_walls(p))
                reads = [rs[0]] + [t for t, _ in more]
                RSUM.append(sum(reads)); RALL.extend(reads); NREAD.append(len(reads))
                for k, v in rs[5].items():
                    codecs[k] = codecs.get(k, 0) + v
            j = os.path.join(d, "cluster.json")
            if os.path.exists(j):
                fr = json.load(open(j))["frames"]
                if len(fr) == nfr:
                    KM.append(sum(f["kmeans_ms"] for f in fr) / 1e3)
                    ARI.append(st.mean(f["ari"] for f in fr))
                    AGREE.append(st.mean(f["agree"] for f in fr))
                    ERR.append(max(max(f["max_err"].values()) for f in fr))
        n = sum(codecs.values()) or 1
        top = ", ".join(f"{k.replace('nvcomp-', '')} {100 * v / n:.0f}%"
                        for k, v in sorted(codecs.items(), key=lambda kv: -kv[1])[:3])
        print(f"{arm:20s} {fmt(W):>14s} {median(W) or 0:6.2f} {median(WC) or 0:7.2f}"
              f" {mean(ratio) or 0:6.2f} {mean(rratio) or 0:7.2f}"
              f" {fmt(R):>14s} {median(R) or 0:6.2f} {median(RC) or 0:7.2f}"
              f" {min(NREAD) if NREAD else 0:2d} {median(RSUM) or 0:9.2f}"
              f" {mean(IO) or 0:6.2f} {mean(DT) or 0:6.2f}"
              f" {mean(GBS) or 0:5.2f} {mean(KM) or 0:8.2f}"
              f" {mean(ARI) if ARI else float('nan'):8.5f}"
              f" {max(ERR) if ERR else float('nan'):9.3g}  {top}")
        rows.append({"arm": arm, "reps": len(runs),
                     "write_s": mean(W), "write_sd": sd(W), "ratio": mean(ratio),
                     "write_med_s": median(W), "write_min_s": min(W) if W else "",
                     "write_max_s": max(W) if W else "",
                     "write_chunk_med_ms": median(WC),
                     "cusz_setup_removed_s": mean(SET),
                     "read_ratio": mean(rratio),
                     "read_s": mean(R), "read_sd": sd(R),
                     "read_med_s": median(R), "read_min_s": min(R) if R else "",
                     "read_max_s": max(R) if R else "",
                     "read_chunk_med_ms": median(RC),
                     "reads_n": min(NREAD) if NREAD else "",
                     "reads_sum_s": mean(RSUM), "reads_sum_med_s": median(RSUM),
                     "reads_sum_min_s": min(RSUM) if RSUM else "",
                     "reads_sum_max_s": max(RSUM) if RSUM else "",
                     "read_all_med_s": median(RALL),
                     "read_io_s": mean(IO), "read_decompress_s": mean(DT),
                     "read_gb_s": mean(GBS),
                     "kmeans_s": mean(KM), "kmeans_sd": sd(KM),
                     "ari": mean(ARI), "agree": mean(AGREE), "seed_ari": seed_ari,
                     "max_err": max(ERR) if ERR else "",
                     "clustered_reps": len(ARI), "read_codecs": top,
                     "write_stall_chunks": sum(WS), "read_stall_chunks": sum(RS),
                     # A compressor that stored most of what is read back raw
                     # is not being measured -- its fallback path is.
                     "raw_share": (sum(v for k, v in codecs.items() if k.startswith("raw"))
                                   / (sum(codecs.values()) or 1))})
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["arm"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nreference read (the dumps): {ref_read:.2f} s -> {out}/summary.csv")


if __name__ == "__main__":
    main()

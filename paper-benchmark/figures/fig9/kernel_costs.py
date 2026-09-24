#!/usr/bin/env python3
"""Per-chunk kernel cost and codec choice, from a campaign's blobs.csv files.

Regenerates the tables in KERNELS.md sections 2-4. Reads ONLY blobs.csv, which
every figure-9 arm writes at no cost -- unlike selection.csv, which carries the
exact quantize/shuffle flags but FNV-hashes every input and output byte and is
therefore off in any timed arm.

  ./kernel_costs.py --camp Nyx=/work/hdd/.../fig9-full-nyx-09222322/nyx \
                    --camp VPIC=/work/hdd/.../fig9-full-vpic-40g-.../vpic

WHY preprocessing is inferred rather than read: the task context that the replay
driver sees (core_tasks.h) carries compress_lib_ and a SUMMED
actual_preproc_time_ms_, never the chosen action's two booleans. On a LOSSLESS
arm the inference is exact -- quantize is masked to -INFINITY without a positive
bound, so any nonzero preprocessing time must be byte-shuffle. On a lossy arm it
is ambiguous and this script says so instead of guessing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys


def find_blobs(root: str, arm: str) -> str | None:
    """Locate an arm's per-chunk log.

    :param root: the campaign's per-workload directory
    :param arm: arm tag, e.g. "neuropress" or "np_only"
    :return: path to blobs.csv, or None when the arm did not run
    """
    for dirpath, _, names in os.walk(os.path.join(root, arm)):
        if "blobs.csv" in names:
            return os.path.join(dirpath, "blobs.csv")
    return None


def load(path: str) -> list[dict]:
    """Read a blobs.csv into rows."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def kernel_share(rows: list[dict], bar_s: float | None) -> dict:
    """Sum the CUDA-event brackets over every chunk of one arm.

    compress_ms brackets the codec launch alone and preproc_ms the quantize and
    byte-shuffle kernels; neither covers allocation, manager setup or staging.
    Negative values are sentinels for "not measured" and are floored at zero.

    :param rows: blobs.csv rows
    :param bar_s: the arm's total_min*60 from fig9.csv, or None
    :return: dict of seconds per phase plus the share of the bar
    """
    def total(col):
        return sum(max(0.0, float(r[col])) for r in rows) / 1000.0
    codec, preproc, h2d = total("compress_ms"), total("preproc_ms"), total("h2d_ms")
    out = {"chunks": len(rows), "codec": codec, "preproc": preproc,
           "h2d": h2d, "sum": codec + preproc + h2d}
    out["share"] = (100.0 * out["sum"] / bar_s) if bar_s else None
    return out


def choices(rows: list[dict], top: int) -> list[tuple]:
    """Codec choice counts with mean per-chunk cost and ratio.

    :param rows: blobs.csv rows
    :param top: how many codecs to report
    :return: [(codec, count, share_pct, mean_ms, mean_ratio)], most-chosen first
    """
    cnt = collections.Counter(r["codec"] for r in rows)
    ms = collections.defaultdict(list)
    rat = collections.defaultdict(list)
    for r in rows:
        ms[r["codec"]].append(float(r["compress_ms"]))
        rat[r["codec"]].append(float(r["ratio"]))
    n = max(1, len(rows))
    out = []
    for codec, c in cnt.most_common(top):
        out.append((codec, c, 100.0 * c / n,
                    sum(ms[codec]) / len(ms[codec]),
                    sum(rat[codec]) / len(rat[codec])))
    return out


def preproc_share(rows: list[dict]) -> float:
    """Fraction of chunks that ran ANY preprocessing kernel, as a percentage."""
    n = max(1, len(rows))
    return 100.0 * sum(1 for r in rows if float(r["preproc_ms"]) > 0) / n


def bar_seconds(root: str, arm: str) -> float | None:
    """The arm's measured wall clock, read back from its own fig9.csv."""
    path = os.path.join(root, arm, "fig9.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("total_min")]
    return float(rows[-1]["total_min"]) * 60.0 if rows else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camp", action="append", required=True, metavar="NAME=DIR",
                    help="workload name and its campaign directory; repeatable")
    ap.add_argument("--arm", default="neuropress", help="arm tag (default neuropress)")
    ap.add_argument("--lossless-arm", default="np_only",
                    help="lossless arm for the exact byte-shuffle share")
    ap.add_argument("--top", type=int, default=4, help="codecs to report (default 4)")
    args = ap.parse_args()

    camps = []
    for spec in args.camp:
        if "=" not in spec:
            print(f"--camp wants NAME=DIR, got {spec!r}", file=sys.stderr)
            return 2
        name, _, path = spec.partition("=")
        camps.append((name, path))

    print(f"== kernel time as a share of the bar  (arm: {args.arm})")
    print(f"{'workload':8s} {'chunks':>7s} {'codec_s':>8s} {'preproc_s':>10s} "
          f"{'sum_s':>7s} {'bar_s':>7s} {'share':>7s}")
    for name, root in camps:
        blobs = find_blobs(root, args.arm)
        if not blobs:
            print(f"{name:8s}  no {args.arm} arm")
            continue
        k = kernel_share(load(blobs), bar_seconds(root, args.arm))
        share = f"{k['share']:6.1f}%" if k["share"] else "     --"
        print(f"{name:8s} {k['chunks']:7d} {k['codec']:8.2f} {k['preproc']:10.2f} "
              f"{k['sum']:7.2f} {(bar_seconds(root, args.arm) or 0):7.1f} {share:>7s}")

    print(f"\n== codec choice, cost and ratio  (arm: {args.arm})")
    for name, root in camps:
        blobs = find_blobs(root, args.arm)
        if not blobs:
            continue
        rows = load(blobs)
        print(f"  {name}  ({len(rows)} chunks, "
              f"{len(set(r['codec'] for r in rows))} distinct codecs)")
        for codec, c, pct, ms, ratio in choices(rows, args.top):
            print(f"     {codec:20s} {c:6d}  {pct:5.1f}%   "
                  f"{ms:7.2f} ms/chunk   ratio {ratio:8.3f}")

    print(f"\n== preprocessing share")
    print(f"   on {args.lossless_arm} (eb=0) this is EXACT byte-shuffle: quantize")
    print(f"   cannot run without a positive bound. On {args.arm} it may be either.")
    for name, root in camps:
        parts = []
        for arm in (args.lossless_arm, args.arm):
            blobs = find_blobs(root, arm)
            if blobs:
                parts.append(f"{arm} {preproc_share(load(blobs)):5.1f}%")
        print(f"  {name:8s} {'   '.join(parts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Did exploration measure EVERY action on EVERY chunk, with real numbers?

Global coverage is not the question. A log can show all 32 actions across the
run while no single chunk saw more than four of them -- that is exactly what
K=3 produces, and it is what makes the cost model confounded into the
observation. So every check below is PER CHUNK.

"Real numbers" is the second half. A row exists for every candidate the
selector considered, but a row is not a measurement: ratio, ct_ms and dt_ms
are filled in only when the candidate was actually run, and dt_ms in
particular is measured only when CLIO_NEUROPRESS_EXPLORE_MEASURE_DT is on --
otherwise it carries the model's prediction or a sentinel. A run that
"explored" 32 actions and measured none of their decompression times would
pass a coverage check and answer no question worth asking.

Usage:  validate_exploration.py <explore.csv> [expected_k]
"""
import csv, sys
from collections import Counter, defaultdict

ACTION = ("lib_name", "shuffle", "quantize")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(path, expected_k=31):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{path}: empty"); return 1

    by_chunk = defaultdict(list)
    for r in rows:
        by_chunk[r["blob"]].append(r)

    want_rows = expected_k + 1          # K alternatives + the primary
    all_actions = {(r["lib_name"], r["shuffle"], r["quantize"]) for r in rows}

    print(f"=== {path}")
    print(f"    {len(rows)} rows over {len(by_chunk)} chunks")
    print(f"    distinct actions in the log: {len(all_actions)}/32")

    # --- 1. per-chunk completeness ---------------------------------------
    bad_count, bad_dupe, per_chunk_actions = [], [], []
    for blob, rs in by_chunk.items():
        acts = Counter((r["lib_name"], r["shuffle"], r["quantize"]) for r in rs)
        per_chunk_actions.append(len(acts))
        if len(rs) != want_rows:
            bad_count.append((blob, len(rs)))
        dupes = {a: n for a, n in acts.items() if n > 1}
        if dupes:
            bad_dupe.append((blob, dupes))

    lo, hi = min(per_chunk_actions), max(per_chunk_actions)
    print(f"\n    [1] rows per chunk == {want_rows}: "
          f"{'PASS' if not bad_count else f'FAIL ({len(bad_count)} chunks)'}")
    if bad_count[:3]:
        for b, n in bad_count[:3]:
            print(f"        {b}: {n} rows")
    print(f"    [2] distinct actions per chunk: min {lo}, max {hi} of 32 -> "
          f"{'PASS' if lo == 32 else 'FAIL'}")
    print(f"    [3] no action measured twice in a chunk: "
          f"{'PASS' if not bad_dupe else f'FAIL ({len(bad_dupe)} chunks)'}")

    # --- 2. both halves of the space, per chunk ---------------------------
    bad_split = []
    for blob, rs in by_chunk.items():
        q = Counter(r["quantize"] for r in rs)
        if q.get("0", 0) != 16 or q.get("1", 0) != 16:
            bad_split.append((blob, dict(q)))
    print(f"    [4] 16 lossless + 16 quantize per chunk: "
          f"{'PASS' if not bad_split else f'FAIL ({len(bad_split)} chunks)'}")
    if bad_split[:3]:
        for b, q in bad_split[:3]:
            print(f"        {b}: {q}")

    # --- 3. are the numbers MEASURED? -------------------------------------
    print("\n    measured columns (a row is not a measurement):")
    for col, want_positive in (("ratio", True), ("ct_ms", True),
                               ("dt_ms", True), ("cost", False)):
        vals = [num(r.get(col)) for r in rows]
        missing = sum(1 for v in vals if v is None)
        nonpos = sum(1 for v in vals if v is not None and v <= 0)
        good = len(vals) - missing - (nonpos if want_positive else 0)
        flag = "PASS" if good == len(vals) else "CHECK"
        print(f"      {col:9} present {len(vals)-missing:5}/{len(vals)}"
              f"   non-positive {nonpos:5}   -> {flag}")

    # dt_ms is the one that silently degrades to a prediction.
    dt = [num(r.get("dt_ms")) for r in rows]
    pdt = [num(r.get("pred_dt_ms")) for r in rows]
    identical = sum(1 for a, b in zip(dt, pdt)
                    if a is not None and b is not None and a == b)
    print(f"      dt_ms == pred_dt_ms on {identical}/{len(rows)} rows"
          f"  ({'measured' if identical < len(rows) * 0.5 else 'LOOKS PREDICTED'})")

    # --- 4. lossy quality, on the quantized half --------------------------
    qz = [r for r in rows if r.get("quantize") == "1"]
    if qz:
        meas = sum(1 for r in qz if r.get("quality_measured") == "1")
        psnr = sum(1 for r in qz if (num(r.get("psnr_db")) or -1) > 0)
        print(f"\n    quantized rows: {len(qz)}"
              f"   quality_measured=1 on {meas}"
              f"   psnr_db>0 on {psnr}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 31
    sys.exit(main(sys.argv[1], k))

#!/usr/bin/env python3
"""Gate the explore log's quality columns.

Quality is DEFINED only for a quantizing action. Every nvcomp codec is
lossless and the byte shuffle is a permutation, so a lossless row's round trip
is exact by construction -- a number there measures the codec against its own
input, not the data against its original. Those rows must read NA.

A quantizing row is the opposite: it MUST carry the measurement, because it is
the only thing that can witness a bound violation. -1 there was the bug this
gate exists to catch -- the primary's row was written in DynamicSchedule,
before the primary was compressed, so its quality did not exist yet.

Exit 0 only if every check passes.
"""
import csv, sys
from collections import Counter, defaultdict

QCOLS = ["meas_rmse", "meas_max_error", "meas_psnr_db",
         "meas_ssim", "meas_ssim_deviation"]

def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("   FAIL: no rows"); return 1
    chunks = {r["blob"] for r in rows}
    qz = [r for r in rows if r["quantize"] == "1"]
    ll = [r for r in rows if r["quantize"] == "0"]
    print(f"   {len(rows)} rows over {len(chunks)} chunks "
          f"({len(qz)} quantized, {len(ll)} lossless)")

    fails = []

    # ---- 1. structure: nothing lost by deferring the primary's write ----
    per = Counter(r["blob"] for r in rows)
    if set(per.values()) != {32}:
        fails.append(f"rows per chunk is {sorted(set(per.values()))}, want 32")
    roles = Counter(r["role"] for r in rows)
    if roles.get("primary", 0) != len(chunks):
        fails.append(f"{roles.get('primary',0)} primary rows, want {len(chunks)}")
    acts = defaultdict(set)
    for r in rows:
        acts[r["blob"]].add((r["lib_name"], r["quantize"], r["shuffle"]))
    if {len(v) for v in acts.values()} != {32}:
        fails.append("some chunk does not cover all 32 actions")
    print(f"   [1] 32 rows/chunk, one primary each, 32 distinct actions: "
          f"{'FAIL' if fails else 'PASS'}")

    # ---- 2. quantized rows carry a real measurement ---------------------
    bad = [r for r in qz if any(r[c] in ("NA", "-1", "") for c in QCOLS)]
    nq = [r for r in qz if r["quality_measured"] != "1"]
    if bad:
        fails.append(f"{len(bad)} quantized rows with NA/-1 quality")
        for r in bad[:3]:
            print(f"       e.g. {r['blob']} {r['lib_name']} role={r['role']} "
                  f"rmse={r['meas_rmse']}")
    if nq:
        fails.append(f"{len(nq)} quantized rows with quality_measured != 1")
    print(f"   [2] all {len(qz)} quantized rows measured, none NA/-1: "
          f"{'FAIL' if (bad or nq) else 'PASS'}")

    # ---- 3. the primary specifically -- the row that used to be -1 ------
    pq = [r for r in rows if r["role"] == "primary" and r["quantize"] == "1"]
    pbad = [r for r in pq if any(r[c] in ("NA", "-1", "") for c in QCOLS)]
    if pbad:
        fails.append(f"{len(pbad)} quantized PRIMARY rows still unmeasured")
    print(f"   [3] all {len(pq)} quantized primary rows measured: "
          f"{'FAIL' if pbad else 'PASS'}")

    # ---- 4. lossless rows are NA, not a number --------------------------
    lbad = [r for r in ll if any(r[c] != "NA" for c in QCOLS)]
    lqm = [r for r in ll if r["quality_measured"] != "NA"]
    if lbad:
        fails.append(f"{len(lbad)} lossless rows with a number instead of NA")
        for r in lbad[:3]:
            print(f"       e.g. {r['blob']} {r['lib_name']} "
                  f"rmse={r['meas_rmse']}")
    if lqm:
        fails.append(f"{len(lqm)} lossless rows with quality_measured != NA")
    print(f"   [4] all {len(ll)} lossless rows NA: "
          f"{'FAIL' if (lbad or lqm) else 'PASS'}")

    # ---- 5. the measurements are sane, not just present -----------------
    # Only rows that actually carry a number: an NA here is already a [2]
    # failure, and float("NA") would abort the gate before it reported.
    vals = [float(r["meas_max_error"]) for r in qz
            if r["meas_max_error"] not in ("NA", "")]
    if not vals:
        fails.append("no quantized row carries a numeric max_error")
        print("   [5] SKIPPED: nothing numeric to check")
        print("\n   GATE FAILED:")
        for f in fails:
            print(f"     - {f}")
        return 1
    eb = sorted({r["eb_encoded"] for r in qz})
    over = [v for v in vals if v > float(eb[0]) * 1.001] if len(eb) == 1 else []
    if len(eb) != 1:
        fails.append(f"quantized rows carry {len(eb)} different bounds: {eb}")
    if over:
        fails.append(f"{len(over)} quantized rows exceed the bound {eb[0]}")
    print(f"   [5] eb={eb[0] if len(eb)==1 else eb}, worst max|err| "
          f"{max(vals):.4e}, none over bound: {'FAIL' if over else 'PASS'}")

    if fails:
        print("\n   GATE FAILED:")
        for f in fails:
            print(f"     - {f}")
        return 1
    print("\n   GATE PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

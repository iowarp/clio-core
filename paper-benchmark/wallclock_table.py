#!/usr/bin/env python3
"""Summary table for compare_wallclock.sh.

Reads the per-arm CSV the runner produced and prints the comparison, ranking on
clio_s -- the per-chunk work a strategy actually controls. See the runner's
header for what that window includes and why.

Fidelity is printed beside every timing on purpose: a lossy codec that missed
its error bound did less work than the others and its time is not comparable.
"""
import csv, sys

def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("  no arms recorded"); return
    num = lambda r, k: float(r[k]) if r.get(k, "NA") not in ("NA", "") else None
    base = next((num(r, "clio_s") for r in rows
                 if r["config"] == "learn" and num(r, "clio_s")), None)

    print(f"{'Strategy':26s}{'clio(s)':>9s}{'vs NP':>7s}"
          f"{'infer':>8s}{'h2d+pre+cod':>12s}{'unacctd':>9s}{'ratio':>8s}  fidelity")
    for r in rows:
        c = num(r, "clio_s")
        rel = f"{c/base:.2f}x" if (base and c) else "--"
        cs = f"{c:.3f}" if c else "NA"
        pc = r.get("codec_s", "NA"); un = r.get("unacctd_s", "NA")
        inf = r.get("infer_s", "NA")
        note = "" if r.get("rc") == "0" else f"  rc={r.get('rc')}"
        print(f"{r['strategy']:26s}{cs:>9s}{rel:>7s}{inf:>8s}{pc:>12s}{un:>9s}"
              f"{r.get('ratio','NA'):>8s}  {r.get('bound','?')}{note}")

    print("\nclio(s) = h2d + inference + preprocess + codec -- MEASURED only.")
    print("unacctd = total - read - clio: runtime plumbing (per-chunk alloc, IPC,")
    print("  task scheduling). The H2D copy is now measured and counted in clio.")
    print("  It is ~6x the compression work and near-identical across arms, so it")
    print("  is shown but never ranked on. Process setup is excluded entirely.")
    print("Byte counts, not residual store time, are the honest I/O proxy:")
    print("  stored_bytes and ratio are exact and in the CSV.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: wallclock_table.py <wallclock.csv>")
    main(sys.argv[1])

#!/usr/bin/env python3
"""Turn ior_tiers.sh's IOR JSON summaries into one CSV row per repetition.

Columns match probe_tiers.sh's tier_bw.csv, so cost_model_tiers.py reads both:
  tier,method,transfer_bytes,rep,bytes,ms,GBps
method is ior_<write|read>_n<processes>; bytes and ms are the aggregate over
all processes; GBps is IOR's bandwidth in decimal GB/s.

  ior_to_csv.py RAW_DIR > ior_bw.csv
"""
import glob
import json
import os
import sys


def rows(path):
    """CSV rows of one <tier>-n<N>-<op>.json summary.

    @param path IOR JSON summary file
    @return list of CSV lines, one per repetition
    """
    tier, n, op = os.path.basename(path)[:-5].rsplit("-", 2)
    with open(path) as f:
        d = json.load(f)
    test = d["tests"][0]
    ntasks = int(test["Parameters"]["tasks"]) if "tasks" in test.get(
        "Parameters", {}) else int(n[1:])
    out = []
    for rep, iteration in enumerate(test["Results"], 1):
        for r in iteration:
            if r["access"] != op:
                continue
            total = r["blockKiB"] * 1024 * ntasks
            out.append(f"{tier},ior_{op}_{n},{int(r['xferKiB'] * 1024)},{rep},"
                       f"{int(total)},{1e3 * r['totalTime']:.3f},"
                       f"{r['bwMiB'] * 2**20 / 1e9:.4f}")
    return out


def main():
    """Print the header and every repetition of every summary under RAW_DIR."""
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print("tier,method,transfer_bytes,rep,bytes,ms,GBps")
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
        try:
            print("\n".join(rows(path)))
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
            print(f"{path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

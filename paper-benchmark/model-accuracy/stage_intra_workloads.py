#!/usr/bin/env python3
"""Stage gen_intra_workloads.py's native arrays as one 4 MiB chunk stream each.

    stage_intra_workloads.py [WORKLOAD ...] [--src DIR] [--out DIR]

Every array file NNN_<name>.<type> of a workload, in NNN order (the order the
application produces them), is cut into 4 MiB chunks written as
<out>/<workload>/fields/NNN_<name>.<type>__cKKKK__dt-<type>.chunk, so the
lexical order of the names is the replay order and run_v2_workloads.sh's
--dtype-from-name sees each chunk's element type. The last chunk of an array
keeps its real (shorter) length; empty arrays are skipped.
"""
import argparse
import os

SRC = "/mnt/nvme0/np-data-new"
OUT = "/mnt/nvme0/v2-work"
CHUNK = 4 << 20
WORKLOADS = ["omics-pbmc", "gnn-igbh", "analytics-tpch", "climate-era5"]


def stage(src, out):
    """Cut one workload's arrays into chunk files; returns (chunks, bytes)."""
    os.makedirs(out, exist_ok=True)
    n = total = 0
    for f in sorted(os.listdir(src)):
        dtype = f.rsplit(".", 1)[-1]
        p = os.path.join(src, f)
        size = os.path.getsize(p)
        if size == 0:
            continue
        with open(p, "rb") as fh:
            k = 0
            while True:
                b = fh.read(CHUNK)
                if not b:
                    break
                name = f"{f}__c{k:04d}__dt-{dtype}.chunk"
                with open(os.path.join(out, name), "wb") as o:
                    o.write(b)
                k += 1
                n += 1
                total += len(b)
    return n, total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workloads", nargs="*", default=WORKLOADS)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    for w in a.workloads:
        n, total = stage(os.path.join(a.src, w), os.path.join(a.out, w, "fields"))
        print(f"{w:16s} {n:5d} chunks  {total / 2**30:6.2f} GiB")


if __name__ == "__main__":
    main()

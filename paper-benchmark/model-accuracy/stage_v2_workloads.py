#!/usr/bin/env python3
"""Stage the codec sweep's measured chunks of every new-workloads dataset as
flat files for run_v2_workloads.sh.

    stage_v2_workloads.py [--out /mnt/nvme0/v2-work]

For each dataset of plot_new_workloads.DATASETS, every chunk the sweep
measured (~/np-newsweep/<ds>/chunks.txt, 4 MiB chunks) is copied from its
source file to <out>/<ds>/fields/<path>__c<k>__dt-<type>.chunk, where <type>
is the array's element type (from the sweep's array-type map; "bin" arrays:
LAMMPS dumps are float64, NanoAOD columns carry the type in their name).
Lexical order of the names is the replay order (time order for the sims).
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "codec-sweep", "new-workloads"))
import plot_new_workloads as pw  # noqa: E402

CHUNK = 4 << 20
TYPES = ("bf16", "f16", "f32", "f64", "i8", "u8", "i16", "u16", "i32", "u32",
         "i64", "u64")


def source_dir(ds):
    """Where the sweep read this dataset's arrays."""
    if ds.startswith("ref-"):
        return os.path.expanduser(f"~/np-data/{ds[4:]}/fields")
    return os.path.expanduser(f"~/np-data/new/{ds}")


def dtype_of(ds, path, product):
    """Element type of one array."""
    ext = product.rsplit(".", 1)[-1]
    if ext in TYPES:
        return ext
    if ds.startswith("ref-lammps"):
        return "f64"
    m = re.search(r"_(bf16|f16|f32|f64|i8|u8|i16|u16|i32|u32|i64|u64)_", path)
    if m:
        return m.group(1)
    return "i32" if "ints" in path else "f32"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/mnt/nvme0/v2-work")
    a = ap.parse_args()
    total = 0
    for ds in pw.DATASETS:
        sweep = os.path.expanduser(f"~/np-newsweep/{ds}")
        product = dict(l.strip().split(",", 1) for l in open(f"{sweep}/map.csv"))
        out = os.path.join(a.out, ds, "fields")
        os.makedirs(out, exist_ok=True)
        n = 0
        for line in open(f"{sweep}/chunks.txt"):
            path, k = line.split()
            dt = dtype_of(ds, path, product.get(f"{path}#{k}", ""))
            with open(os.path.join(source_dir(ds), path), "rb") as f:
                f.seek(int(k) * CHUNK)
                data = f.read(CHUNK)
            name = f"{path.replace('/', '__')}__c{k}__dt-{dt}.chunk"
            with open(os.path.join(out, name), "wb") as f:
                f.write(data)
            n += 1
            total += len(data)
        print(f"{ds:28s} {n:4d} chunks")
    print(f"staged {total / 2**30:.1f} GiB in {a.out}")


if __name__ == "__main__":
    main()

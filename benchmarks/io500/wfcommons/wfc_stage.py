#!/usr/bin/env python3
"""Write a WfBench workflow's input files (the files no task produces).

  mpiexec -n R ... wfc_stage.py <workflow.json> <workdir> R

Rank r of R (r from PMI_RANK or PALS_RANKID; R from the argument, since
the Aurora launcher sets no size variable for non-MPI programs) writes every R-th input
file, largest first, with the exact size the workflow expects. The content
is random bytes from one 16 MiB os.urandom buffer per rank (incompressible;
none of the three systems deduplicates). Each rank prints one line:
  STAGE rank=<r> files=<n> bytes=<b> secs=<s>
"""
import json
import os
import sys
import time

BUF = 16 << 20


def rank_size(nranks):
    """This process's rank and the number of ranks (nranks if given)."""
    r = os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", "0"))
    n = nranks or os.environ.get("PMI_SIZE", os.environ.get("PALS_SIZE", "1"))
    return int(r), int(n)


def staged_files(path):
    """(name, size) of every input file, largest first."""
    s = json.load(open(path))["workflow"]["specification"]
    produced = {f for t in s["tasks"] for f in t.get("outputFiles", [])}
    files = [(f["id"], f["sizeInBytes"]) for f in s["files"] if f["id"] not in produced]
    return sorted(files, key=lambda x: (-x[1], x[0]))


def main():
    """Write this rank's share of the input files into the work directory."""
    js, wd = sys.argv[1], sys.argv[2]
    r, n = rank_size(sys.argv[3] if len(sys.argv) > 3 else None)
    os.makedirs(wd, exist_ok=True)
    buf = os.urandom(BUF)
    t0 = time.time()
    nf = nb = 0
    for i, (name, size) in enumerate(staged_files(js)):
        if i % n != r:
            continue
        with open(os.path.join(wd, name), "wb") as f:
            left = size
            while left > 0:
                k = min(left, BUF)
                f.write(buf[:k])
                left -= k
        nf += 1
        nb += size
    print(f"STAGE rank={r} files={nf} bytes={nb} secs={time.time() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()

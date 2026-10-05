#!/usr/bin/env python3
"""Keep each workload's exhaustive search and fixed-codec runs, so they are run
once and reused.

    baseline_store.py save    DATASET exhaustive|fixed|oracle [--run DIR]
    baseline_store.py has     DATASET exhaustive|fixed [--codec SPEC]
    baseline_store.py restore DATASET exhaustive|fixed [--codec SPEC] [--run DIR]
    baseline_store.py best    DATASET
    baseline_store.py list

Store: /mnt/nvme0/v2-work/baselines/<DATASET>/exhaustive/ and
.../fixed/<codec>/, each holding the run's logs (stdout.log, runtime.log,
compose.yaml, blobs.csv, phases.csv, v2_pred.csv, v2_measured.csv,
selection.csv when present) and meta.json: the input fingerprint (sorted chunk
names and sizes), the chunk count, git commit and whether the tree had local
changes, the date, the run settings (selection log, tiers, GPU clock) and, for
an exhaustive search, the ranking of every setting by total balanced 4-tier
cost (candidates.csv) and the best single codec. The stored blobs themselves
(chi_bdev.dat, cte_tier.dat*) are not kept.

`has` succeeds only when a baseline exists AND its fingerprint matches the
staged input now, so changed data is never served from an old baseline.
`restore` copies the logs back to runs/<DATASET>_<kind>_nolog, where the
comparison scripts read them.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/nvme0/v2-work"
STORE = os.path.join(ROOT, "baselines")
LOGS = ["stdout.log", "runtime.log", "compose.yaml", "blobs.csv", "phases.csv",
        "v2_pred.csv", "v2_measured.csv", "selection.csv"]


def fingerprint(ds):
    """(sha256 of sorted chunk names and sizes, chunk count) of the input."""
    d = os.path.join(ROOT, ds, "fields")
    h = hashlib.sha256()
    names = sorted(os.listdir(d))
    for n in names:
        h.update(f"{n}\t{os.path.getsize(os.path.join(d, n))}\n".encode())
    return h.hexdigest(), len(names)


def slug(spec):
    """A codec spec as a directory name."""
    return re.sub(r"[^A-Za-z0-9=._-]+", "_", spec.strip())


def where(ds, kind, codec=None):
    """Store directory of one baseline."""
    if kind in ("exhaustive", "oracle"):
        return os.path.join(STORE, ds, kind)
    return os.path.join(STORE, ds, "fixed", slug(codec))


def git_state():
    """(commit, tree has local changes) of the clio-core checkout."""
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    run = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True,
                                    text=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))


def gpu_clock():
    """Current SM clock, as nvidia-smi reports it."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def fixed_codec(run):
    """The one setting a fixed run stored."""
    import pandas as pd
    specs = pd.read_csv(os.path.join(run, "v2_measured.csv")).spec.unique()
    if len(specs) != 1:
        sys.exit(f"{run} stored {len(specs)} settings, not one: not a fixed run")
    return specs[0]


def save(ds, kind, run):
    """Copy a finished run's logs into the store, with meta.json."""
    if not os.path.exists(os.path.join(run, "stdout.log")):
        sys.exit(f"no run at {run}")
    codec = fixed_codec(run) if kind == "fixed" else None
    dst = where(ds, kind, codec)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in LOGS:
        if os.path.exists(os.path.join(run, f)):
            shutil.copy2(os.path.join(run, f), dst)
    fp, n = fingerprint(ds)
    commit, dirty = git_state()
    compose = open(os.path.join(run, "compose.yaml")).read()
    meta = {"dataset": ds, "kind": kind, "codec": codec, "fingerprint": fp,
            "chunks": n, "git_commit": commit, "git_local_changes": dirty,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "source_run": run,
            "gpu_clock_at_save": gpu_clock(),
            "selection_log": os.path.exists(os.path.join(run, "selection.csv")),
            "tiers": "12e6:1,1e6:3,0.5e6:3,0.25e6:3",
            "verified": bool(re.search(r"VERIFIED: (\d+) of \1 ",
                                       open(os.path.join(run, "stdout.log")).read())),
            "compose_excerpt": [l.strip() for l in compose.splitlines()
                                if "neuropress_" in l]}
    if kind == "exhaustive":
        sys.path.insert(0, HERE)
        import compare_fixed_vs_learn as c
        cand = c.candidates(run)
        cand.to_csv(os.path.join(dst, "candidates.csv"))
        meta["best_codec"] = cand.index[0]
        meta["best_cost_ms"] = float(cand.cost_ms.iloc[0])
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=2)
    print(f"saved {ds} {kind}{' ' + codec if codec else ''} -> {dst}")


def valid(ds, kind, codec=None):
    """meta.json of a baseline whose fingerprint matches the input, or None."""
    p = os.path.join(where(ds, kind, codec), "meta.json")
    if not os.path.exists(p):
        return None
    meta = json.load(open(p))
    return meta if meta["fingerprint"] == fingerprint(ds)[0] else None


def restore(ds, kind, codec, run):
    """Copy a stored baseline's logs to the run directory the scripts read."""
    meta = valid(ds, kind, codec)
    if meta is None:
        sys.exit(f"no valid {kind} baseline for {ds}")
    os.makedirs(run, exist_ok=True)
    for f in os.listdir(where(ds, kind, codec)):
        shutil.copy2(os.path.join(where(ds, kind, codec), f), run)
    print(f"restored {ds} {kind} (saved {meta['saved']}, commit "
          f"{meta['git_commit'][:8]}) -> {run}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["save", "has", "restore", "best", "list"])
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("kind", nargs="?", choices=["exhaustive", "fixed", "oracle"])
    ap.add_argument("--codec", default=None)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    run = a.run or (os.path.join(ROOT, "runs", f"{a.dataset}_{a.kind}_nolog")
                    if a.dataset and a.kind else None)
    if a.action == "save":
        save(a.dataset, a.kind, run)
    elif a.action == "has":
        sys.exit(0 if valid(a.dataset, a.kind, a.codec) else 1)
    elif a.action == "restore":
        restore(a.dataset, a.kind, a.codec, run)
    elif a.action == "best":
        meta = valid(a.dataset, "exhaustive")
        if meta is None:
            sys.exit(1)
        print(meta["best_codec"])
    else:
        for ds in sorted(os.listdir(STORE)) if os.path.isdir(STORE) else []:
            for root, _, files in os.walk(os.path.join(STORE, ds)):
                if "meta.json" in files:
                    m = json.load(open(os.path.join(root, "meta.json")))
                    ok = "valid" if valid(ds, m["kind"], m["codec"]) else "STALE"
                    print(f"{ds:26s} {m['kind']:10s} {m.get('codec') or m.get('best_codec', ''):30s}"
                          f" {m['chunks']:5d} chunks  {m['saved']}  {ok}")


if __name__ == "__main__":
    main()

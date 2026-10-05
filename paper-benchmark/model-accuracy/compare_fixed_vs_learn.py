#!/usr/bin/env python3
"""One workload: best single codec (by the 4-tier cost model) vs NeuroPress v2
learning, per chunk and in total, all measured through Clio.

    compare_fixed_vs_learn.py DATASET [--tag TAG]

Reads runs/<DATASET>_{exhaustive,fixed,learn}_<TAG>:
  exhaustive  every setting measured on every chunk: per-setting total cost
              (compress + decompress + bytes / (ratio x tier bandwidth); a
              setting that does not shrink the chunk is stored raw), ranked;
              the cheapest is the best single codec
  fixed       the whole workload stored with that codec
  learn       NeuroPress v2, learning on, exploration off
Per chunk (phases.csv, write path): measured wall-clock ms and its parts
(compress, NeuroPress prediction, learning updates, the decompress that
measures learning labels, storage I/O, rest); read path: decompress and
wall ms. Totals add the measured end-to-end write / read time (stdout.log).
Writes <DATASET>_fixed_vs_learn_{candidates,chunks}.csv next to the runs.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

RUNS = "/mnt/nvme0/v2-work/runs"


def run_dir(ds, mode, tag):
    """Run directory of one mode."""
    return os.path.join(RUNS, f"{ds}_{mode}" + (f"_{tag}" if tag else ""))


def candidates(ex):
    """All settings ranked by total cost over the chunks (exhaustive run)."""
    pred = pd.read_csv(os.path.join(ex, "v2_pred.csv"), usecols=["blob", "bytes", "tier_bw"])
    m = pd.read_csv(os.path.join(ex, "v2_measured.csv"))
    raw = (m.role == "primary") & ((m.comp_ms <= 0) | (m.ratio <= 1))
    m = m[~raw & (m.decomp_ms > 0)].drop_duplicates(["blob", "setting"])
    m = m.merge(pred[["blob", "bytes"]], on="blob")
    io = m.bytes / m.tier_bw
    kept = m.ratio > 1
    m["cost"] = np.where(kept, m.comp_ms + m.decomp_ms + io / m.ratio, m.comp_ms + io)
    n = pred.blob.nunique()
    g = m.groupby("spec").agg(chunks=("blob", "nunique"), cost_ms=("cost", "sum"),
                              compress_ms=("comp_ms", "sum"),
                              decompress_ms=("decomp_ms", "sum"),
                              stored_bytes=("bytes", lambda b: 0))
    g["stored_bytes"] = m.assign(sb=np.where(kept, m.bytes / m.ratio, m.bytes)) \
        .groupby("spec").sb.sum()
    g["ratio"] = pred.bytes.sum() / g.stored_bytes
    store = pd.DataFrame({"chunks": [n], "cost_ms": [(pred.bytes / pred.tier_bw).sum()],
                          "compress_ms": [0.0], "decompress_ms": [0.0],
                          "stored_bytes": [pred.bytes.sum()], "ratio": [1.0]},
                         index=["store (raw)"])
    g = pd.concat([g[g.chunks == n], store]).sort_values("cost_ms")
    return g.drop(columns="chunks")


def wall(run):
    """(write s, read s) measured end to end."""
    text = open(os.path.join(run, "stdout.log")).read()
    w = re.search(r"stage\+compress ([0-9.]+) s", text)
    r = re.search(r"READ 1/1.*?get\+decompress: ([0-9.]+) ms", text, re.S)
    return float(w.group(1)), float(r.group(1)) / 1e3


def per_chunk(run):
    """Per-chunk write and read timings from phases.csv."""
    ph = pd.read_csv(os.path.join(run, "phases.csv"))
    w = ph[ph.path == "write"].drop_duplicates("chunk_id").set_index("chunk_id")
    r = ph[ph.path == "read"].drop_duplicates("chunk_id").set_index("chunk_id")
    cols = {"wall_ms": "write_wall_ms", "compress_ms": "compress_ms",
            "nn_ms": "predict_ms", "sgd_ms": "learn_update_ms",
            "sgd_gpu_ms": "learn_update_gpu_ms", "label_ms": "learn_label_ms",
            "io_ms": "write_io_ms", "h2d_ms": "h2d_ms", "sgd_updates": "updates",
            "explore_ms": "explore_ms", "explored": "explored"}
    out = w[[c for c in cols if c in w]].rename(columns=cols).fillna(0.0)
    out["read_wall_ms"] = r.wall_ms
    out["decompress_ms"] = r.decompress_ms
    # Compression ratio of what was stored (original / stored bytes, header
    # included), from the replay's per-blob report.
    b = pd.read_csv(os.path.join(run, "blobs.csv")).drop_duplicates("blob")
    b = b.set_index("blob")
    out["bytes"] = b.bytes
    out["stored_bytes"] = b.stored
    out["ratio"] = b.bytes / b.stored
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--tag", default="nolog")
    ap.add_argument("--with-explore", action="store_true",
                    help="also report the learn + explore run (off by default)")
    a = ap.parse_args()
    ds, tag = a.dataset, a.tag
    cand = candidates(run_dir(ds, "exhaustive", tag))
    pd.set_option("display.width", 200)
    print(f"== {ds}: all settings by total 4-tier cost (exhaustive run, {tag})")
    print(cand.round(2).to_string())
    best = cand.index[0]
    print(f"\nbest single codec: {best}  cost {cand.cost_ms.iloc[0]:.1f} ms")
    cand.to_csv(os.path.join(RUNS, f"{ds}_fixed_vs_learn_candidates.csv"))
    fx, le, lx = (run_dir(ds, m, tag) for m in ("fixed", "learn", "learnexp"))
    used = pd.read_csv(os.path.join(fx, "v2_measured.csv")).spec.unique()
    print(f"fixed run used: {list(used)}")
    a_fx, a_le = per_chunk(fx), per_chunk(le)
    both = a_fx.join(a_le, lsuffix="_fixed", rsuffix="_learn", how="inner")
    both.to_csv(os.path.join(RUNS, f"{ds}_fixed_vs_learn_chunks.csv"))
    rows = []
    arms = [("best single codec", fx, a_fx), ("NeuroPress learning", le, a_le)]
    if a.with_explore and os.path.exists(os.path.join(lx, "stdout.log")):
        arms.append(("NeuroPress learn + explore", lx, per_chunk(lx)))
    for name, run, d in arms:
        w, r = wall(run)
        rows.append({"method": name, "write_s": w, "read_s": r, "e2e_s": w + r,
                     "compress_s": d.compress_ms.sum() / 1e3,
                     "decompress_s": d.decompress_ms.sum() / 1e3,
                     "predict_s": d.get("predict_ms", 0).sum() / 1e3,
                     "learn_update_s": d.get("learn_update_ms", 0).sum() / 1e3,
                     "learn_update_gpu_s": d.get("learn_update_gpu_ms", 0).sum() / 1e3,
                     "learn_label_s": d.get("learn_label_ms", 0).sum() / 1e3,
                     "explore_s": d.get("explore_ms", 0).sum() / 1e3,
                     "explored": int(d.get("explored", 0).sum()),
                     "ratio": d.bytes.sum() / d.stored_bytes.sum(),
                     "updates": int(d.get("updates", 0).sum()),
                     "chunks_updated": int((d.get("updates", 0) > 0).sum()),
                     "chunks": len(d)})
    t = pd.DataFrame(rows).set_index("method")
    print("\n== totals (seconds; e2e = measured write + read)")
    print(t.round(3).to_string())
    print("\n== per chunk (ms): median / mean")
    for col in ("write_wall_ms", "read_wall_ms", "compress_ms", "decompress_ms", "ratio"):
        print(f"  {col:16s} fixed {both[col + '_fixed'].median():7.3f} / {both[col + '_fixed'].mean():7.3f}"
              f"   learn {both[col + '_learn'].median():7.3f} / {both[col + '_learn'].mean():7.3f}")
    for col in ("predict_ms", "learn_update_ms", "learn_update_gpu_ms", "learn_label_ms"):
        if col in a_le:
            print(f"  {col:20s} learn {a_le[col].median():7.3f} / {a_le[col].mean():7.3f}")


if __name__ == "__main__":
    main()

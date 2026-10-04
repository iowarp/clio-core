#!/usr/bin/env python3
"""Evaluate the saved NeuroPress v2 model on real chunks the sweep measured.

    eval_nn_v2_real.py --sweep-dir ~/np-newsweep/ref-vpic-126-2000
                       --src ~/np-data/vpic-126-2000/fields [--name vpic]
                       [--limit N] [--model .../model_v2.nnwt]

Nothing is retrained. For every chunk in the sweep directory (configs.csv +
configs_shuffle.csv, chunks "path#k" of --chunk bytes), the four v2 inputs are
computed from the source file exactly as for training (build_corpus_csv.py's
file_stats on the chunk's float32 values, then the transforms in
model_v2.json), the saved .nnwt is run in NumPy, and its 45 x 3 predictions
are compared with the measured values of the same 45 settings.

Writes results/nn-v2/real_eval_<name>_{accuracy,selection,chunks}.csv:
  accuracy   MAPE and median APE per target: all, per codec, per field, and
             for chunks inside / outside the training feature range
  selection  top-1 and regret of picking by predicted cost (12, 1, 0.5,
             0.25 GB/s) or ratio, against the best single setting for these
             chunks chosen with hindsight
  chunks     per chunk: features, in-range flag, picks and regrets
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "codec-sweep"))
import nn_v2_cv as cv  # noqa: E402
import train_nn_v2 as tr  # noqa: E402
from build_corpus_csv import file_stats  # noqa: E402

WEIGHTS = os.path.join(HERE, "..", "..", "context-transport-primitives", "src",
                       "compress", "model", "weights", "v2", "model_v2.nnwt")
TARGETS = ["comp_ms", "decomp_ms", "ratio"]


def canon(spec):
    """@return a setting with its key=value tokens sorted ("ans shuffle=byte")."""
    t = spec.split()
    return " ".join(t[:1] + sorted(t[1:]))


def chunk_features(src, name, chunk):
    """The four raw v2 inputs of one chunk "path#k".

    @return [log2 bytes, entropy, log10(mad/range), log10(d2/range)] and the
            untransformed (entropy, mad, d2)
    """
    path, k = name.rsplit("#", 1)
    k = int(k)
    with open(os.path.join(src, path), "rb") as f:
        f.seek(k * chunk)
        raw = f.read(chunk)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"nnv2_{os.getpid()}.bin")
    with open(tmp, "wb") as f:
        f.write(raw)
    e, m, d = file_stats(tmp)
    os.remove(tmp)
    x = [np.log2(len(raw)), e, np.log10(m + cv.EPS), np.log10(d + cv.EPS)]
    return x, (e, m, d)


def load_measured(sweep_dir, settings):
    """Measured log targets per chunk for the model's settings.

    @return (chunk names, bytes per chunk, log array chunks x S x 3)
    """
    parts = [pd.read_csv(os.path.join(sweep_dir, f), float_precision="round_trip",
                         low_memory=False)
             for f in ("configs.csv", "configs_shuffle.csv")
             if os.path.exists(os.path.join(sweep_dir, f))]
    d = pd.concat(parts, ignore_index=True)
    d["settings"] = d["settings"].fillna("")
    d["setting"] = (d["algorithm"] + " " + d["settings"]).str.strip().map(canon)
    d = d[d["setting"].isin(set(settings)) & (d["ok"] == 1)]
    d["ratio"] = d["bytes"] / d["comp_bytes"]
    full = d.groupby("file")["setting"].nunique()
    names = sorted(full[full == len(settings)].index)
    d = d[d["file"].isin(names)].drop_duplicates(["file", "setting"])
    cube = [d.pivot(index="file", columns="setting", values=t).loc[names, settings]
            .to_numpy(float) for t in TARGETS]
    nbytes = d.drop_duplicates("file").set_index("file").loc[names, "bytes"].to_numpy(float)
    return names, nbytes, np.log(np.stack(cube, axis=-1))


def training_range(stats_csv):
    """Min / max of the four transformed inputs over the training corpus."""
    s = pd.read_csv(stats_csv)
    size = s["file"].str.extract(r"_(\d+)(kb|mb)\.bin$")
    nb = size[0].astype(float) * np.where(size[1] == "mb", 2**20, 2**10)
    x = np.column_stack([np.log2(nb), s["entropy"], np.log10(s["mad"] + cv.EPS),
                         np.log10(s["d2"] + cv.EPS)])
    return x.min(axis=0), x.max(axis=0)


def ape_rows(groups, ape):
    """APE summaries per group label array (None = all)."""
    rows = []
    for by, labels in groups.items():
        keys = ["all"] if labels is None else sorted(set(labels))
        for key in keys:
            sel = np.ones(ape.shape[:2], bool) if labels is None else None
            if labels is not None:
                lab = np.asarray(labels)
                if lab.ndim == 1 and len(lab) == ape.shape[0]:
                    sel = np.repeat((lab == key)[:, None], ape.shape[1], axis=1)
                else:
                    sel = np.repeat((lab == key)[None, :], ape.shape[0], axis=0)
            if not sel.any():
                continue
            for t, name in enumerate(cv.NAMES):
                v = ape[..., t][sel]
                rows.append({"by": by, "key": key, "target": name, "n": int(v.size),
                             "mape": float(v.mean()), "median_ape": float(np.median(v))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", default="vpic")
    ap.add_argument("--chunk", type=int, default=4 << 20)
    ap.add_argument("--limit", type=int, default=0, help="first N chunks only")
    ap.add_argument("--model", default=WEIGHTS)
    ap.add_argument("--stats", default="/mnt/nvme0/synthetic-9800-stats.csv")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "codec-sweep", "results", "nn-v2"))
    a = ap.parse_args()
    meta = json.load(open(os.path.splitext(a.model)[0] + ".json"))
    settings = [canon(s) for s in meta["outputs"]["settings"]]
    codec = np.array([s.split()[0] for s in settings])

    names, nbytes, true = load_measured(os.path.expanduser(a.sweep_dir), settings)
    if a.limit:
        names, nbytes, true = names[:a.limit], nbytes[:a.limit], true[:a.limit]
    feats = [chunk_features(os.path.expanduser(a.src), n, a.chunk) for n in names]
    X = np.array([f[0] for f in feats])
    pred = tr.numpy_forward(a.model, X).astype(np.float64).reshape(len(names), len(settings), 3)
    lo, hi = training_range(a.stats)
    inside = ((X >= lo) & (X <= hi)).all(axis=1)
    field = np.array([n.rsplit("#", 1)[0].rsplit("_", 1)[-1] for n in names])

    ape = np.abs(np.exp(pred) - np.exp(true)) / np.exp(true)
    acc = ape_rows({"all": None, "codec": codec, "field": field,
                    "in_training_range": np.where(inside, "inside", "outside")}, ape)
    fixed = {"ratio": int((nbytes[:, None] / np.exp(true[..., 2])).sum(axis=0).argmin())}
    for bw in cv.BWS:
        fixed[f"cost@{bw:g}GB/s"] = int(cv.costs(true, nbytes, bw).sum(axis=0).argmin())
    sel = ([dict(r, model="v2") for r in cv.selection(pred, true, nbytes, None)] +
           [dict(r, model="best single setting (hindsight)",
                 setting=settings[fixed[r["objective"]]])
            for r in cv.selection(None, true, nbytes, fixed)])

    ch = pd.DataFrame({"chunk": names, "field": field, "inside_range": inside,
                       "entropy": [f[1][0] for f in feats], "mad_over_range": [f[1][1] for f in feats],
                       "d2_over_range": [f[1][2] for f in feats]})
    for bw in [None] + cv.BWS:
        key = "ratio" if bw is None else f"cost@{bw:g}GB/s"
        tc = -true[..., 2] if bw is None else cv.costs(true, nbytes, bw)
        pc = -pred[..., 2] if bw is None else cv.costs(pred, nbytes, bw)
        ch[f"{key}_true_best"] = [settings[i] for i in tc.argmin(axis=1)]
        ch[f"{key}_picked"] = [settings[i] for i in pc.argmin(axis=1)]

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"real_eval_{a.name}")
    pd.DataFrame(acc).to_csv(base + "_accuracy.csv", index=False)
    pd.DataFrame(sel).to_csv(base + "_selection.csv", index=False)
    ch.to_csv(base + "_chunks.csv", index=False)

    acc = pd.DataFrame(acc)
    pd.set_option("display.width", 200)
    print(f"{a.name}: {len(names)} chunks x {len(settings)} settings; "
          f"{int(inside.sum())} inside the training feature range")
    print("feature min/max here vs training:")
    for i, n in enumerate(["log2 bytes", "entropy", "log10 mad/range", "log10 d2/range"]):
        print(f"  {n:16s} {X[:, i].min():8.3f} .. {X[:, i].max():8.3f}   train {lo[i]:8.3f} .. {hi[i]:8.3f}")
    for by in ("all", "in_training_range", "codec", "field"):
        t = acc[acc.by == by].pivot(index="key", columns="target", values="mape").mul(100).round(1)
        print(f"\nMAPE % by {by}:\n{t[cv.NAMES].to_string()}")
    s = pd.DataFrame(sel).set_index(["objective", "model"])[["top1", "regret_mean", "regret_total"]]
    print("\nselection:\n" + s.round(4).to_string())
    print(f"\nwrote {os.path.abspath(base)}_*.csv")


if __name__ == "__main__":
    main()

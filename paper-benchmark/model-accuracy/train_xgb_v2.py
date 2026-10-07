#!/usr/bin/env python3
"""Train the XGBoost baseline on exactly the data of the NeuroPress v2 predictor.

    train_xgb_v2.py [--csv corpus_v2.csv.xz] [--out-dir DIR] [--results DIR] [--folds 5] [--seed 0]
                    [--max-trees 3000] [--depth 6] [--lr 0.1] [--jobs 64]

Data, inputs, targets and folds are NeuroPress v2's (nn_v2_cv.py / train_nn_v2.py), so the two
models differ only in the learner:
  rows     corpus_v2.csv (9800 synthetic float32 files x 45 lossless settings), read with
           nn_v2_cv.load; no benchmark workload data
  inputs   the four NeuroPress features (log2 chunk bytes, byte entropy, log10 MAD/range,
           log10 mean |2nd difference|/range) plus the setting, one-hot (45), one row per
           (file, setting) -- as model A of nn_v2_cv.py and as the XGBoost baseline's
           one-hot library encoding
  targets  natural log of compress ms, decompress ms and compression ratio, as measured; one
           booster per target
  folds    nn_v2_cv.make_folds (the 1960 generator patterns dealt into 5 folds, seed 0), 10 % of
           the training patterns held out for early stopping (nn_v2_cv.split_val, seed + fold)
  fit      the XGBoost baseline's hyper-parameters (depth 6, learning rate 0.1, hist), trees up to
           --max-trees with early stopping (50 rounds) on the held-out patterns; the final model is
           trained on every file with the median best tree count of the folds per target, as
           NeuroPress v2 uses the median best epoch of its folds
Accuracy is reported as NeuroPress v2's: the absolute percentage error of each target on unseen
files (the held-out fold), mean over the folds +- standard deviation.

Writes:
  OUT_DIR/xgb_v2_<target>.ubj.xz the boosters (XGBoost's binary UBJ format, xz-compressed; load_booster
                                 reads them)
  OUT_DIR/xgb_v2.json            features, setting order, targets, transforms, tree counts,
                                 hyper-parameters and the CV accuracy
  RESULTS/cv_accuracy.csv        per fold and target: MAPE and median APE (also per codec)
  RESULTS/cv_trees.csv           per fold and target: best tree count
"""
import argparse
import json
import lzma
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import nn_v2_cv as cv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EARLY = 50


def rows(X, files, n_settings):
    """@return the (file, setting) design matrix: the four features + one-hot setting, file-major."""
    f = np.repeat(files, n_settings)
    s = np.tile(np.arange(n_settings), len(files))
    return np.hstack([X[f], np.eye(n_settings)[s]]).astype(np.float32)


def params(a):
    """@return the booster parameters (the XGBoost baseline's, on CPU)."""
    return {"objective": "reg:squarederror", "max_depth": a.depth, "eta": a.lr, "tree_method": "hist",
            "nthread": a.jobs, "seed": a.seed}


def fit_target(a, data, fit, val, t, n_trees=None):
    """Fit one target's booster on the fit files, early-stopped on the val files (or n_trees fixed).

    @return (booster, best tree count)
    """
    X, Y = data["X"], data["Y"]
    S = Y.shape[1]
    dfit = xgb.DMatrix(rows(X, fit, S), label=Y[fit, :, t].ravel())
    if n_trees is not None:
        b = xgb.train(params(a), dfit, num_boost_round=n_trees)
        return b, n_trees
    dval = xgb.DMatrix(rows(X, val, S), label=Y[val, :, t].ravel())
    b = xgb.train(params(a), dfit, num_boost_round=a.max_trees, evals=[(dval, "val")],
                  early_stopping_rounds=EARLY, verbose_eval=False)
    return b, b.best_iteration + 1


def predict(b, data, files, n_trees):
    """@return predicted log target, files x settings."""
    S = data["Y"].shape[1]
    p = b.predict(xgb.DMatrix(rows(data["X"], files, S)), iteration_range=(0, n_trees))
    return p.reshape(len(files), S)


def cross_validate(a, data):
    """5-fold CV as nn_v2_cv.py: @return (accuracy rows, tree rows)."""
    groups = data["files"]["group"].to_numpy()
    fold = cv.make_folds(groups, a.folds, a.seed)
    acc, trees = [], []
    for k in range(a.folds):
        test = np.where(fold == k)[0]
        train = np.where(fold != k)[0]
        fit, val = cv.split_val(train, groups, a.seed + k)
        pred = np.empty((len(test), data["Y"].shape[1], 3))
        for t in range(3):
            t0 = time.time()
            b, n = fit_target(a, data, fit, val, t)
            pred[..., t] = predict(b, data, test, n)
            trees.append({"fold": k, "target": cv.NAMES[t], "best_trees": n, "fit_s": time.time() - t0})
            print(f"fold {k} {cv.NAMES[t]}: {n} trees, {time.time() - t0:.0f} s", flush=True)
        acc += cv.accuracy_rows("xgboost", k, pred, data["Y"][test], data, test)
    return pd.DataFrame(acc), pd.DataFrame(trees)


def load_booster(model_dir, name):
    """@return the booster of one target (name: comp_time, decomp_time or ratio) from model_dir:
    xgb_v2_<name>.ubj.xz (what save() writes and the repo keeps), or an uncompressed
    xgb_v2_<name>.json when one is there."""
    j = os.path.join(model_dir, f"xgb_v2_{name}.json")
    if os.path.exists(j):
        return xgb.Booster(model_file=j)
    with lzma.open(os.path.join(model_dir, f"xgb_v2_{name}.ubj.xz")) as f:
        return xgb.Booster(model_file=bytearray(f.read()))


def save(a, data, boosters, n_trees, cv_summary):
    """Write the boosters (UBJ, xz-compressed) and the metadata (xgb_v2.json)."""
    os.makedirs(a.out_dir, exist_ok=True)
    for t, b in enumerate(boosters):
        with lzma.open(os.path.join(a.out_dir, f"xgb_v2_{cv.NAMES[t]}.ubj.xz"), "wb", preset=9) as f:
            f.write(b.save_raw("ubj"))
    meta = {"model": "XGBoost baseline on the NeuroPress v2 training data",
            "training_csv": os.path.relpath(a.csv, REPO), "rows": int(data["Y"].shape[0] * data["Y"].shape[1]),
            "files": int(data["Y"].shape[0]),
            "inputs": ["log2(chunk bytes)", "Shannon entropy of the chunk's bytes, bits/byte",
                       "log10(MAD / range + 1e-12)", "log10(mean |2nd difference| / range + 1e-12)"]
                      + [f"one-hot setting {i}" for i in range(len(data["settings"]))],
            "settings": data["settings"],
            "targets": {cv.NAMES[t]: f"natural log of {cv.TARGETS[t]}" for t in range(3)},
            "boosters": {cv.NAMES[t]: f"xgb_v2_{cv.NAMES[t]}.ubj.xz" for t in range(3)},
            "trees": {cv.NAMES[t]: int(n_trees[t]) for t in range(3)},
            "hyper_parameters": {"max_depth": a.depth, "eta": a.lr, "tree_method": "hist",
                                 "early_stopping_rounds": EARLY, "max_trees": a.max_trees},
            "cv": cv_summary, "xgboost_version": xgb.__version__}
    with open(os.path.join(a.out_dir, "xgb_v2.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=os.path.join(HERE, "nn-v2-training", "corpus_v2.csv.xz"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "context-transport-primitives", "src", "compress", "model",
                                                      "weights", "xgb_v2"))
    ap.add_argument("--results", default=os.path.join(HERE, "..", "codec-sweep", "results", "xgb-v2"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-trees", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--jobs", type=int, default=64)
    a = ap.parse_args()
    data = cv.load(a.csv)
    print(f"{data['Y'].shape[0]} files x {data['Y'].shape[1]} settings from {a.csv}", flush=True)
    acc, trees = cross_validate(a, data)
    os.makedirs(a.results, exist_ok=True)
    acc.to_csv(os.path.join(a.results, "cv_accuracy.csv"), index=False)
    trees.to_csv(os.path.join(a.results, "cv_trees.csv"), index=False)
    allr = acc[acc.by == "all"]
    summary = {n: {"mape_mean": float(g.mape.mean()), "mape_std": float(g.mape.std()),
                   "median_ape": float(g.median_ape.mean())} for n, g in allr.groupby("target")}
    print("CV MAPE (unseen data): " + ", ".join(f"{n} {100 * summary[n]['mape_mean']:.2f}% +- "
                                                f"{100 * summary[n]['mape_std']:.2f}" for n in cv.NAMES), flush=True)
    n_trees = [int(np.median(trees[trees.target == n].best_trees)) for n in cv.NAMES]
    every = np.arange(data["Y"].shape[0])
    boosters = [fit_target(a, data, every, None, t, n_trees[t])[0] for t in range(3)]
    in_pred = np.stack([predict(b, data, every, n) for b, n in zip(boosters, n_trees)], axis=-1)
    ape = np.abs(np.exp(in_pred) - np.exp(data["Y"])) / np.exp(data["Y"])
    print("in-sample MAPE: " + ", ".join(f"{n} {100 * ape[..., t].mean():.2f}%" for t, n in enumerate(cv.NAMES)))
    save(a, data, boosters, n_trees, summary)
    print("wrote", os.path.abspath(a.out_dir), "trees", dict(zip(cv.NAMES, n_trees)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

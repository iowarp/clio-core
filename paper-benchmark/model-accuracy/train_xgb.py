#!/usr/bin/env python3
"""Retrain the XGBoost baseline on the SAME rows that seed HCompress.

The shipped checkpoint (NeuroPress/neural_net/weights/xgb_model.pkl) came with
the upstream checkout, and nothing records what it was trained on: upstream's
xgboost/train.py documents `--csv benchmark_results-100k.csv`, not the 600k
corpus this directory's accuracy table scores it against. This rebuilds it so
the three models in that table share their training data:

  rows   benchmark_results_600k.csv -> success only -> 16 KiB files dropped ->
         files sorted, shuffled with RandomState(42), first 80% = train.
         Exactly prepare_inputs.build_synthetic's split, i.e. the rows
         HCompress's seed.csv is drawn from; asserted below, not assumed.
  model  the SHIPPED checkpoint's format, so models_offline.XGBoostBaseline
         loads it unchanged: 15 features (8 one-hot algorithms, quant, shuffle,
         log10 eb with 1e-7 for lossless rows, log2 bytes, entropy, mad,
         second derivative), continuous ones standardized on the training
         rows; outputs log1p-transformed (times floored at 1 ms, as upstream)
         and standardized. Upstream's current encode_and_split uses an integer
         algorithm id and raw eb/size instead, which that loader would misread.
  fit    upstream's hyperparameters: 300 trees, depth 6, lr 0.1, early
         stopping 20 rounds on the held-out files -- as upstream does, so the
         held-out ("synthetic") scores share upstream's mild optimism; the
         five workload campaigns are untouched by it.

Only the four outputs the table scores are trained: compression time,
decompression time, ratio, PSNR.

Usage:
    train_xgb.py --corpus benchmark_results_600k.csv --out DIR \
                 [--seed-csv .../out/inputs/seed.csv] [--jobs N]
Writes DIR/xgb_model.pkl and DIR/train_xgb_metrics.csv.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_inputs import MIN_TRAINED_SIZE, SPLIT_SEED, VAL_FRACTION  # noqa: E402
from models_offline import ALGORITHMS, LOSSLESS_EB  # noqa: E402

FEATURES = ([f"alg_{a}" for a in ALGORITHMS] +
            ["quant_enc", "shuffle_enc", "error_bound_enc", "data_size_enc",
             "entropy", "mad", "second_derivative"])
CONTINUOUS = ["error_bound_enc", "data_size_enc", "entropy", "mad",
              "second_derivative"]
OUTPUTS = ["comp_time_log", "decomp_time_log", "ratio_log", "psnr_clamped"]


def split_files(df):
    """Train/val file sets, by the procedure build_synthetic and upstream share.

    @param df corpus rows, already filtered to success and >= MIN_TRAINED_SIZE.
    @return (train_files, val_files) as sets.
    """
    files = sorted(df.file.unique())
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(files)
    k = int(len(files) * (1 - VAL_FRACTION))
    return set(files[:k]), set(files[k:])


def encode_x(d):
    """The shipped checkpoint's 15 raw (unstandardized) input columns.

    @param d corpus rows.
    @return float64 matrix, columns in FEATURES order.
    """
    idx = d.algorithm.map({a: i for i, a in enumerate(ALGORITHMS)}).to_numpy()
    if np.isnan(idx.astype(float)).any():
        raise ValueError("corpus has an algorithm outside ALGORITHMS")
    onehot = np.zeros((len(d), len(ALGORITHMS)))
    onehot[np.arange(len(d)), idx.astype(int)] = 1.0
    q = (d.quantization == "linear").to_numpy(dtype=float)
    eb = np.where(q > 0, d.error_bound.to_numpy(dtype=float), LOSSLESS_EB)
    return np.column_stack([
        onehot, q, (d.shuffle.to_numpy(dtype=float) > 0).astype(float),
        np.log10(eb), np.log2(d.original_size.to_numpy(dtype=float)),
        d.entropy.to_numpy(dtype=float), d.mad.to_numpy(dtype=float),
        d.second_derivative.to_numpy(dtype=float)])


def encode_y(d):
    """The four scored outputs, transformed exactly as upstream's pipeline.

    @param d corpus rows.
    @return float64 matrix, columns in OUTPUTS order.
    """
    psnr = (pd.to_numeric(d.psnr_db, errors="coerce")
            .replace([np.inf, -np.inf], 120.0).fillna(120.0).clip(upper=120.0))
    return np.column_stack([
        np.log1p(d.compression_time_ms.clip(lower=1)),
        np.log1p(d.decompression_time_ms.clip(lower=1)),
        np.log1p(d.compression_ratio.clip(lower=0)),
        psnr.to_numpy(dtype=float)])


def main():
    """Load, split, fit one regressor per output, save in the shipped format."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed-csv", default=None,
                    help="HCompress's seed.csv from prepare_inputs.py; when "
                         "given, its row count must equal this train split's")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(a.corpus)
    n_all = len(df)
    df = df[df["success"] == True]                              # noqa: E712
    df = df[df.original_size >= MIN_TRAINED_SIZE].copy()
    train_files, val_files = split_files(df)
    tr = df[df.file.isin(train_files)]
    va = df[df.file.isin(val_files)]
    print(f"corpus {n_all} rows -> {len(df)} kept; files train "
          f"{len(train_files)} / val {len(val_files)}; rows train {len(tr)} "
          f"/ val {len(va)}  ({time.time() - t0:.1f} s)")

    # The rows HCompress is seeded from, rebuilt the way prepare_inputs does.
    if a.seed_csv:
        seed = pd.read_csv(a.seed_csv, usecols=["bytes"])
        if len(seed) != len(tr):
            raise SystemExit(f"split mismatch: seed.csv has {len(seed)} rows, "
                             f"this train split {len(tr)}")
        print(f"  matches HCompress seed.csv ({len(seed)} rows)")

    Xtr, Xva = encode_x(tr), encode_x(va)
    ci = [FEATURES.index(c) for c in CONTINUOUS]
    x_means = np.zeros(len(FEATURES))
    x_stds = np.ones(len(FEATURES))
    x_means[ci] = Xtr[:, ci].mean(axis=0)
    x_stds[ci] = np.maximum(Xtr[:, ci].std(axis=0), 1e-8)
    Xtr, Xva = (Xtr - x_means) / x_stds, (Xva - x_means) / x_stds

    Ytr, Yva = encode_y(tr), encode_y(va)
    y_means, y_stds = Ytr.mean(axis=0), np.maximum(Ytr.std(axis=0), 1e-8)
    Ytr, Yva = (Ytr - y_means) / y_stds, (Yva - y_means) / y_stds

    models, rows = {}, []
    for i, name in enumerate(OUTPUTS):
        t1 = time.time()
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             early_stopping_rounds=20, verbosity=0,
                             n_jobs=a.jobs)
        m.fit(Xtr, Ytr[:, i], eval_set=[(Xva, Yva[:, i])], verbose=False)
        p = m.predict(Xva)
        ss_res = float(np.sum((Yva[:, i] - p) ** 2))
        ss_tot = float(np.sum((Yva[:, i] - Yva[:, i].mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        models[name] = m
        rows.append({"output": name, "best_iteration": m.best_iteration,
                     "val_r2_normalized": r2, "seconds": time.time() - t1})
        print(f"  {name:16s} best_iter {m.best_iteration:3d}  val R2 {r2:.4f}  "
              f"({time.time() - t1:.1f} s)")

    path = os.path.join(a.out, "xgb_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"models": models, "x_means": x_means, "x_stds": x_stds,
                     "y_means": y_means, "y_stds": y_stds,
                     "feature_names": FEATURES}, f)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "train_xgb_metrics.csv"),
                              index=False)
    print(f"wrote {path}  (total {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()

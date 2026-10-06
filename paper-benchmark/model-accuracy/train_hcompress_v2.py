#!/usr/bin/env python3
"""Train the HCompress cost model on the NeuroPress v2 training corpus.

    train_hcompress_v2.py [--csv corpus_v2.csv.xz] [--out-dir DIR]
                          [--regularization 1e-3] [--ratio-cap 0] [--folds 5] [--seed 0]

HCompress (Devarajan et al., IPDPS 2020) learns a codec's cost from three
inputs only, and this script gives it exactly those:
  * the compression library: the key Clio uses, "<algorithm>|q<0/1>|s<0/1>"
    (algorithm without its parameters, quantize bit, shuffle bit; see
    HCompressCcpPredictor::LibraryKey and prepare_inputs.library_key);
  * the data type (one-hot of the CSV's dtype column);
  * the data size (log2 of the bytes).
No statistic of the data itself (entropy, MAD, curvature) is used.

The fit is the one HCompressCcpPredictor::Seed() makes: per target a ridge
regression (A + C I) w = b over [1, log2(bytes), one-hot library (, one-hot
dtype)], targets compression speed (MB/s = bytes / (ms x 1000)), decompression
speed (MB/s) and compression ratio (capped only with --ratio-cap > 0). A
prediction is made as PredictFor() makes it: time = bytes / (max(1 MB/s,
speed) x 1000), ratio clamped to [0.1, 1e5]; how often a bound applies is
printed.

Writes to DIR (default the repo's model/weights/hcompress_v2):
  hcompress_ccp_seed.json        library + size, Clio's format (weights and each
                                 head's RLS matrix P): load it with
                                 hcompress_model_path = DIR
  hcompress_ccp_dtype_seed.json  library + dtype + size: the same format plus a
                                 "dtypes" list and the dtype columns after the
                                 library columns (Clio's loader does not read
                                 this one yet)
  hcompress_v2_accuracy.csv      in-sample and 5-fold CV error per target
                                 (the folds of nn_v2_cv.py, by file pattern)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nn_v2_cv as cv  # noqa: E402

MIN_SPEED_MBPS = 1.0      # CcpConfig::min_speed_mbps
MIN_RATIO, MAX_RATIO = 0.1, 1e5
HEADS = (("compress_time", "w_compress_speed_mbps"),
         ("decompress_time", "w_decompress_speed_mbps"),
         ("ratio", "w_compression_ratio"))


def library_key(d):
    """@return Clio's HCompress key per row: bare algorithm, quantize and shuffle bits."""
    algo = d["algorithm"].astype(str).str.replace("^nvcomp-", "", regex=True)
    q = np.where(d["quantization"].astype(str).str.lower().isin(["none", "0", ""]), "0", "1")
    s = np.where(d["shuffle"].astype(str).str.lower().isin(["none", "0", ""]), "0", "1")
    return algo + "|q" + q + "|s" + s


def load(path):
    """@return the corpus rows with the model's inputs and targets."""
    d = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    d["key"] = library_key(d)
    d["group"] = (d["palette"] + "|" + d["bin_width"].astype(str) + "|" +
                  d["perturbation"].astype(str) + "|" + d["fill_mode"])
    d["bytes"] = d["original_size"].astype(float)
    return d


def vocab(values):
    """@return the distinct values in first-seen order (Seed()'s Index() order)."""
    return list(dict.fromkeys(values))


def encode(d, libs, dtypes):
    """@return the regression rows [1, log2(bytes), one-hot library, one-hot dtype]."""
    n = len(d)
    li = {k: i for i, k in enumerate(libs)}
    ti = {k: i for i, k in enumerate(dtypes)}
    x = np.zeros((n, 2 + len(libs) + len(dtypes)))
    x[:, 0] = 1.0
    x[:, 1] = np.log2(d["bytes"].to_numpy())
    r = np.arange(n)
    lib_col = d["key"].map(li)
    ok = lib_col.notna().to_numpy()   # a library outside the vocabulary: no column
    x[r[ok], 2 + lib_col[ok].astype(int).to_numpy()] = 1.0
    if dtypes:
        t_col = d["dtype"].map(ti)
        okt = t_col.notna().to_numpy()
        x[r[okt], 2 + len(libs) + t_col[okt].astype(int).to_numpy()] = 1.0
    return x


def targets(d, ratio_cap):
    """@return [compress speed, decompress speed, ratio] per row (NaN = not measured)."""
    b = d["bytes"].to_numpy()
    ct = d["compression_time_ms"].to_numpy(float)
    dt = d["decompression_time_ms"].to_numpy(float)
    r = d["compression_ratio"].to_numpy(float)
    cs = np.where(ct > 0, b / (ct * 1000.0), np.nan)
    ds = np.where(dt > 0, b / (dt * 1000.0), np.nan)
    rr = np.where(r > 0, np.minimum(r, ratio_cap) if ratio_cap > 0 else r, np.nan)
    return np.column_stack([cs, ds, rr])


def ridge(x, y, c):
    """@return (w solving (X'X + c I) w = X'y, P = (X'X + c I)^-1, rows used),
    over the rows where y is measured; P is the RLS state the seed leaves."""
    ok = np.isfinite(y)
    xx, yy = x[ok], y[ok]
    a = xx.T @ xx + (c if c > 0 else 1e-3) * np.eye(x.shape[1])
    return np.linalg.solve(a, xx.T @ yy), np.linalg.inv(a), int(ok.sum())


def fit(d, c, ratio_cap, with_dtype):
    """@return model dict: vocabularies, weights per head, samples per head."""
    libs = vocab(d["key"])
    dtypes = vocab(d["dtype"]) if with_dtype else []
    x = encode(d, libs, dtypes)
    y = targets(d, ratio_cap)
    ws, ps, ns = [], [], []
    for k in range(3):
        w, pk, n = ridge(x, y[:, k], c)
        ws.append(w)
        ps.append(pk)
        ns.append(n)
    return {"libraries": libs, "dtypes": dtypes, "w": ws, "p": ps, "samples": ns,
            "rows": len(d)}


def predict(model, d):
    """@return predicted (compress ms, decompress ms, ratio) as PredictFor() computes
    them, plus the count of rows where each bound applied."""
    x = encode(d, model["libraries"], model["dtypes"])
    b = d["bytes"].to_numpy()
    cs, ds, r = (x @ w for w in model["w"])
    bound = {"compress_speed_floor": int((cs < MIN_SPEED_MBPS).sum()),
             "decompress_speed_floor": int((ds < MIN_SPEED_MBPS).sum()),
             "ratio_clamp": int(((r < MIN_RATIO) | (r > MAX_RATIO)).sum())}
    ct = b / (np.maximum(MIN_SPEED_MBPS, cs) * 1000.0)
    dt = b / (np.maximum(MIN_SPEED_MBPS, ds) * 1000.0)
    return np.column_stack([ct, dt, np.clip(r, MIN_RATIO, MAX_RATIO)]), bound


def ape(d, pred):
    """@return absolute percentage errors [compress ms, decompress ms, ratio] per row."""
    true = d[["compression_time_ms", "decompression_time_ms", "compression_ratio"]].to_numpy(float)
    return np.abs(pred - true) / true


def write_json(model, path, c, ratio_cap):
    """Write the seed in the layout HCompressCcpPredictor::Save() writes."""
    def nums(v):
        return ", ".join(repr(float(x)) for x in v)
    dt = model["dtypes"]
    note = ("Expected Compression Cost, HCompress (IPDPS 2020): linear regression on the"
            " compression library (one-hot)" + (", the data type (one-hot)" if dt else "") +
            " and the data size (log2 bytes); outputs compression speed (MB/s),"
            " decompression speed (MB/s) and compression ratio; no quality output."
            " Trained by train_hcompress_v2.py on the NeuroPress v2 corpus.")
    lines = ["{",
             '  "model_type": "hcompress_ccp",',
             '  "version": "2.0",',
             f'  "note": "{note}",',
             f'  "seed_rows": {model["rows"]},',
             f'  "regularization": {c!r},',
             '  "forget_factor": 1,',
             '  "feedback_interval": 1,',
             '  "feedback_updates": 0,',
             f'  "ratio_target_cap": {ratio_cap!r},',
             '  "libraries": [' + ", ".join(f'"{k}"' for k in model["libraries"]) + "],"]
    if dt:
        lines.append('  "dtypes": [' + ", ".join(f'"{k}"' for k in dt) + "],")
    lines.append(f'  "samples": [{model["samples"][0]}, {model["samples"][1]}, {model["samples"][2]}],')
    for k, (_, key) in enumerate(HEADS):
        lines.append(f'  "{key}": [{nums(model["w"][k])}],')
    # each head's P = (A + C I)^-1 (row-major), so feedback in Clio continues
    # from the seed instead of from the prior (1/C) I
    for k, (_, key) in enumerate(HEADS):
        pkey = key.replace("w_", "p_", 1)
        lines.append(f'  "{pkey}": [{nums(model["p"][k].ravel())}]' + ("," if k < 2 else ""))
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def accuracy(d, fold, c, ratio_cap, with_dtype):
    """@return rows of in-sample and cross-validated MAPE / median APE per target."""
    rows = []
    full = fit(d, c, ratio_cap, with_dtype)
    p, bound = predict(full, d)
    a = ape(d, p)
    for k, (name, _) in enumerate(HEADS):
        rows.append({"model": "dtype" if with_dtype else "library+size", "split": "in-sample",
                     "target": name, "mape": a[:, k].mean(), "median_ape": np.median(a[:, k])})
    cv_ape = np.empty_like(a)
    for f in np.unique(fold):
        test = fold == f
        m = fit(d[~test], c, ratio_cap, with_dtype)
        cv_ape[test] = ape(d[test], predict(m, d[test])[0])
    for k, (name, _) in enumerate(HEADS):
        rows.append({"model": "dtype" if with_dtype else "library+size", "split": "cv",
                     "target": name, "mape": cv_ape[:, k].mean(),
                     "median_ape": np.median(cv_ape[:, k])})
    return rows, full, bound


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=os.path.join(HERE, "nn-v2-training", "corpus_v2.csv.xz"))
    ap.add_argument("--out-dir", default=os.path.join(
        HERE, "..", "..", "context-transport-primitives", "src", "compress", "model",
        "weights", "hcompress_v2"))
    ap.add_argument("--regularization", type=float, default=1e-3,
                    help="ridge C, as CcpConfig::regularization")
    ap.add_argument("--ratio-cap", type=float, default=0.0,
                    help="cap on the ratio target (0 = none, the paper's)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = load(a.csv)
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"{len(d)} rows, {d['file'].nunique()} files, {d['key'].nunique()} library keys, "
          f"dtypes {sorted(d['dtype'].unique())}, sizes {sorted(d['bytes'].astype(int).unique())}")
    files = d.drop_duplicates("file").set_index("file")
    fold_of_file = dict(zip(files.index, cv.make_folds(files["group"].to_numpy(), a.folds, a.seed)))
    fold = d["file"].map(fold_of_file).to_numpy()
    rows = []
    for with_dtype, name in ((False, "hcompress_ccp_seed.json"), (True, "hcompress_ccp_dtype_seed.json")):
        r, model, bound = accuracy(d, fold, a.regularization, a.ratio_cap, with_dtype)
        rows += r
        write_json(model, os.path.join(a.out_dir, name), a.regularization, a.ratio_cap)
        print(f"wrote {name}: {len(model['libraries'])} libraries"
              + (f", {len(model['dtypes'])} dtype(s)" if with_dtype else "")
              + f", bounds applied in-sample {bound}")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out_dir, "hcompress_v2_accuracy.csv"), index=False)
    with pd.option_context("display.width", 160):
        print((t.assign(mape=100 * t.mape, median_ape=100 * t.median_ape)
               .rename(columns={"mape": "MAPE %", "median_ape": "median APE %"})
               .round(2).to_string(index=False)))
    # with one dtype in the data the dtype column only duplicates the intercept
    p0, _ = predict(fit(d, a.regularization, a.ratio_cap, False), d)
    p1, _ = predict(fit(d, a.regularization, a.ratio_cap, True), d)
    print(f"largest relative difference between the two models' predictions: "
          f"{np.max(np.abs(p1 - p0) / p0):.3g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a train_xgb.py checkpoint as flat trees clio-core can evaluate.

clio-core's compress layer carries no XGBoost dependency (the build has
CLIO_CTP_ENABLE_XGBOOST off, and the pip wheel ships no C headers), so the
runtime selector (XgbTreePredictor, xgb_tree_predictor.{h,cc}) walks the trees
itself. This writes them in a line-based text format that needs no JSON parser:

    clio-xgb-trees 1
    features <F> <name> ...
    x_means <F doubles>
    x_stds <F doubles>
    lossless_eb <double>
    outputs <K>
    output <name> <y_mean> <y_std> <base_score> <n_trees>
    tree <n_nodes>
    <left> <right> <missing> <feature> <threshold> <leaf>     (n_nodes lines;
                                                                left = -1: leaf)
    ...
    tests <N>
    <F raw inputs> <K normalized predictions>                  (N lines)

Only the trees up to each model's best_iteration are written, because that is
what XGBRegressor.predict() evaluates after early stopping. A node sends x to
`left` when x < threshold (XGBoost's rule, compared in float32), and to
`missing` when x is NaN.

The test vectors are XGBoost's OWN predictions on held-out rows, and this
script re-derives them from the exported trees before writing: if the flat
trees do not reproduce predict(), nothing is written.

Usage: export_xgb_trees.py --model DIR/xgb_model.pkl --corpus CSV [--tests N]
Writes DIR/xgb_trees.txt next to the checkpoint.
"""
import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models_offline import LOSSLESS_EB  # noqa: E402
from prepare_inputs import MIN_TRAINED_SIZE  # noqa: E402
from train_xgb import OUTPUTS, encode_x, split_files  # noqa: E402


def flatten(tree_json):
    """One XGBoost JSON dump tree -> flat node arrays indexed by nodeid.

    @param tree_json a tree from Booster.get_dump(dump_format="json").
    @return list of (left, right, missing, feature, threshold, leaf) rows.
    """
    nodes = {}

    def walk(n):
        nid = n["nodeid"]
        if "leaf" in n:
            nodes[nid] = (-1, -1, -1, -1, 0.0, float(n["leaf"]))
            return
        feat = int(n["split"].lstrip("f"))
        nodes[nid] = (int(n["yes"]), int(n["no"]), int(n["missing"]), feat,
                      float(n["split_condition"]), 0.0)
        for c in n["children"]:
            walk(c)

    walk(tree_json)
    return [nodes[i] for i in range(len(nodes))]


def eval_tree(rows, x):
    """Walk one flat tree for one float32 input row."""
    i = 0
    while rows[i][0] != -1:
        left, right, missing, feat, thr, _ = rows[i]
        v = x[feat]
        i = missing if np.isnan(v) else (left if v < np.float32(thr) else right)
    return rows[i][5]


def main():
    """Load the checkpoint, flatten its trees, prove them, write the file."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tests", type=int, default=200)
    a = ap.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ck = pickle.load(open(a.model, "rb"))
    feats = list(ck["feature_names"])
    x_means = np.asarray(ck["x_means"], dtype=np.float64)
    x_stds = np.asarray(ck["x_stds"], dtype=np.float64)
    names = list(ck["models"].keys())
    if names != OUTPUTS:
        raise SystemExit(f"checkpoint outputs {names}, expected {OUTPUTS}")

    # Held-out rows for the test vectors: the same split train_xgb.py used.
    df = pd.read_csv(a.corpus)
    df = df[(df["success"] == True) &                          # noqa: E712
            (df.original_size >= MIN_TRAINED_SIZE)].copy()
    _, val_files = split_files(df)
    va = df[df.file.isin(val_files)].sample(n=a.tests, random_state=7)
    x_raw = encode_x(va)
    xn32 = ((x_raw - x_means) / x_stds).astype(np.float32)

    blocks, preds = [], {}
    for k, name in enumerate(names):
        m = ck["models"][name]
        bst = m.get_booster()
        cfg = json.loads(bst.save_config())
        base = float(cfg["learner"]["learner_model_param"]["base_score"]
                     .strip("[]"))
        n_trees = m.best_iteration + 1
        dumps = bst.get_dump(dump_format="json")[:n_trees]
        trees = [flatten(json.loads(d)) for d in dumps]
        want = m.predict(xn32)
        got = np.array([base + sum(eval_tree(t, x) for t in trees)
                        for x in xn32])
        err = float(np.max(np.abs(got - want)))
        print(f"  {name:16s} {n_trees} trees, base {base:.6g}, "
              f"max |flat - predict()| = {err:.3g}")
        if err > 1e-4:
            raise SystemExit(f"{name}: flat trees do not reproduce predict()")
        preds[name] = want
        blocks.append((name, float(ck["y_means"][k]), float(ck["y_stds"][k]),
                       base, trees))

    out = os.path.join(os.path.dirname(os.path.abspath(a.model)),
                       "xgb_trees.txt")
    g = lambda v: format(float(v), ".17g")  # noqa: E731
    with open(out, "w") as f:
        f.write("clio-xgb-trees 1\n")
        f.write(f"features {len(feats)} {' '.join(feats)}\n")
        f.write("x_means " + " ".join(g(v) for v in x_means) + "\n")
        f.write("x_stds " + " ".join(g(v) for v in x_stds) + "\n")
        f.write(f"lossless_eb {g(LOSSLESS_EB)}\n")
        f.write(f"outputs {len(blocks)}\n")
        for name, ym, ys, base, trees in blocks:
            f.write(f"output {name} {g(ym)} {g(ys)} {g(base)} {len(trees)}\n")
            for t in trees:
                f.write(f"tree {len(t)}\n")
                for l, r, mi, fe, th, lf in t:
                    f.write(f"{l} {r} {mi} {fe} {g(th)} {g(lf)}\n")
        f.write(f"tests {len(xn32)}\n")
        for i in range(len(xn32)):
            f.write(" ".join(g(v) for v in x_raw[i]) + " " +
                    " ".join(g(preds[n][i]) for n in names) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

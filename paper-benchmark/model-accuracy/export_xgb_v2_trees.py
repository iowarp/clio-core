#!/usr/bin/env python3
"""Export the XGBoost v2 baseline (train_xgb_v2.py) as flat trees that Clio can evaluate.

    export_xgb_v2_trees.py [--model-dir DIR] [--csv corpus_v2.csv.xz] [--tests 24]

clio-core carries no XGBoost dependency, so the runtime selector (XgbV2Predictor) walks the trees
itself, as XgbTreePredictor does for the old 8-algorithm model. Inputs per (chunk, setting): the
four NeuroPress v2 features (log2 chunk bytes, byte entropy, log10 MAD/range, log10 mean |2nd
difference|/range) as features 0-3, then the setting one-hot as features 4-48 (in the order of the
'settings' lines). Outputs: natural log of compress ms, decompress ms and compression ratio.

Format (text, no JSON parser needed):

    clio-xgb-v2-trees 1
    chunk_features 4
    settings <S>
    <setting spec>                                   (S lines, one-hot order)
    outputs 3
    output <name> <base_score> <n_trees>
    tree <n_nodes>
    <left> <right> <missing> <feature> <threshold> <leaf>   (n_nodes lines; left = -1: leaf)
    ...
    tests <N>
    <4 chunk features> <S x 3 predictions, setting-major>   (N lines)

A node sends x to `left` when x < threshold (compared in float32) and to `missing` when x is NaN.
One tree walk serves all settings at once: the settings that reach a node go the same way at a
chunk-feature split; at a one-hot split only the setting it tests can go the other way. The
exporter evaluates the flat trees that way and refuses to write unless they reproduce XGBoost's
own predict() on the test vectors (held-out-style rows of the corpus) to 1e-4.
"""
import argparse
import json
import os

import numpy as np
import xgboost as xgb

import nn_v2_cv as cv
from train_xgb_v2 import load_booster

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NF = 4   # chunk features before the one-hot


def flat_trees(booster):
    """@return (base_score, [trees]) of one booster; a tree = arrays (left, right, missing, feature,
    threshold, leaf)."""
    m = json.loads(booster.save_raw("json"))["learner"]
    base = float(m["learner_model_param"]["base_score"].strip("[]"))
    trees = []
    for t in m["gradient_booster"]["model"]["trees"]:
        left = np.array(t["left_children"], np.int32)
        right = np.array(t["right_children"], np.int32)
        miss = np.where(np.array(t["default_left"], bool), left, right).astype(np.int32)
        feat = np.array(t["split_indices"], np.int32)
        cond = np.array(t["split_conditions"], np.float32)
        leaf = np.where(left < 0, cond, np.float32(0)).astype(np.float32)
        trees.append((left, right, miss, feat, cond, leaf))
    return base, trees


def eval_all(trees, base, x, n_settings):
    """@return the prediction for every setting of one chunk (features x[0:4]), walking each tree
    once with the set of settings that reach each node."""
    out = np.full(n_settings, base, np.float64)
    xs = x.astype(np.float32)
    every = (1 << n_settings) - 1
    for left, right, miss, feat, thr, leaf in trees:
        stack = [(0, every)]
        while stack:
            n, mask = stack.pop()
            if left[n] < 0:
                for s in range(n_settings):
                    if mask >> s & 1:
                        out[s] += leaf[n]
                continue
            f = int(feat[n])
            if f < NF:
                v = xs[f]
                nxt = miss[n] if np.isnan(v) else (left[n] if v < thr[n] else right[n])
                stack.append((nxt, mask))
                continue
            s = f - NF   # the one setting whose one-hot is 1 here
            one = left[n] if np.float32(1) < thr[n] else right[n]
            zero = left[n] if np.float32(0) < thr[n] else right[n]
            hit = mask & (1 << s)
            if one == zero:
                stack.append((one, mask))
            else:
                if hit:
                    stack.append((one, hit))
                if mask & ~hit:
                    stack.append((zero, mask & ~hit))
    return out


def write(path, settings, outputs, tests):
    """Write the text file (see the module doc)."""
    with open(path, "w") as f:
        f.write("clio-xgb-v2-trees 1\n")
        f.write(f"chunk_features {NF}\n")
        f.write(f"settings {len(settings)}\n")
        for s in settings:
            f.write(s + "\n")
        f.write(f"outputs {len(outputs)}\n")
        for name, base, trees in outputs:
            f.write(f"output {name} {base!r} {len(trees)}\n")
            for left, right, miss, feat, thr, leaf in trees:
                f.write(f"tree {len(left)}\n")
                for i in range(len(left)):
                    if left[i] < 0:
                        f.write(f"-1 -1 -1 0 0 {float(leaf[i])!r}\n")
                    else:
                        f.write(f"{left[i]} {right[i]} {miss[i]} {feat[i]} {float(thr[i])!r} 0\n")
        f.write(f"tests {len(tests)}\n")
        for x, pred in tests:
            f.write(" ".join(repr(float(v)) for v in x) + " " + " ".join(repr(float(v)) for v in pred.ravel()) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=os.path.join(REPO, "context-transport-primitives", "src", "compress",
                                                        "model", "weights", "xgb_v2"))
    ap.add_argument("--csv", default=os.path.join(HERE, "nn-v2-training", "corpus_v2.csv.xz"))
    ap.add_argument("--tests", type=int, default=24)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.model_dir, "xgb_v2.json")))
    settings = meta["settings"]
    S = len(settings)
    data = cv.load(a.csv)
    assert data["settings"] == settings, "the corpus' setting order differs from the model's"
    rng = np.random.default_rng(7)
    files = rng.choice(len(data["X"]), size=a.tests, replace=False)
    outputs, preds = [], np.zeros((a.tests, S, 3))
    for t, name in enumerate(cv.NAMES):
        b = load_booster(a.model_dir, name)
        base, trees = flat_trees(b)
        rows = np.hstack([np.repeat(data["X"][files], S, axis=0), np.tile(np.eye(S), (a.tests, 1))]).astype(np.float32)
        ref = b.predict(xgb.DMatrix(rows)).reshape(a.tests, S)
        for i, fi in enumerate(files):
            mine = eval_all(trees, base, data["X"][fi], S)
            err = np.abs(mine - ref[i]).max()
            if err > 1e-4:
                raise SystemExit(f"{name}: flat trees differ from XGBoost by {err:.2e} on test {i}; nothing written")
        preds[..., t] = ref
        outputs.append((name, base, trees))
        print(f"{name}: {len(trees)} trees, {sum(len(x[0]) for x in trees)} nodes, matches XGBoost on {a.tests} files",
              flush=True)
    out = os.path.join(a.model_dir, "xgb_v2_trees.txt")
    write(out, settings, outputs, [(data["X"][fi], preds[i]) for i, fi in enumerate(files)])
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""5-fold cross-validation of NeuroPress v2 candidate predictors.

    nn_v2_cv.py [--csv corpus_v2.csv] [--out DIR] [--pred-out NPZ] [--seed S]

Data: corpus_v2.csv (build_corpus_csv.py on the float32 synthetic sweep), one
row per (file, setting); 9800 files x 45 lossless settings.

Inputs (the four NeuroPress features, nothing new): log2 chunk size, byte
entropy, log10 MAD/range, log10 mean |2nd difference|/range (the log only
spreads two heavily skewed columns; the values are the CSV's), plus which
setting is predicted.
Targets: natural log of compress ms, decompress ms and compression ratio, as
measured (no floor, cap or rounding).

Folds: the 1960 generator patterns (palette, bin width, perturbation, fill)
are dealt into 5 folds with a seeded shuffle; all 5 sizes and all 45 settings
of a pattern share a fold. Input and target scaling are fitted on the training
folds only; 10% of the training patterns are held out for early stopping.

Models (same folds):
  baseline  per (setting, size) geometric mean of the training folds -- uses
            no data statistic, so it shows what entropy / MAD / 2nd diff add
  A         one network per row: 4 features + one-hot setting (49) ->
            64 x 4 ReLU -> 3 outputs (NeuroPress's hidden size and depth)
  B         one network per chunk: 4 features -> 64 x 4 ReLU -> 135 outputs
            (3 per setting), every setting in one pass

Reported on each held-out fold:
  accuracy   mean and median absolute percentage error of each target, also
             per codec and per size
  selection  per file, the setting with the lowest predicted cost
             ct + dt + compressed bytes / bandwidth (12, 1, 0.5, 0.25 GB/s) or
             the highest predicted ratio: top-1 (picked the true best) and
             regret (true cost of the pick / true best - 1), per-file mean
             and over the fold's total; plus the best single setting chosen
             on the training folds as the fixed-codec reference
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

TARGETS = ["compression_time_ms", "decompression_time_ms", "compression_ratio"]
NAMES = ["comp_time", "decomp_time", "ratio"]
BWS = [12.0, 1.0, 0.5, 0.25]
EPS = 1e-12
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    """Read the corpus and lay it out per file.

    @param path  corpus_v2.csv
    @return dict with files (DataFrame), settings (list), X (F x 4 raw
            features), Y (F x S x 3 log targets), codec (per setting)
    """
    d = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    d["settings"] = d["settings"].fillna("")
    d["setting"] = (d["algorithm"] + " " + d["settings"]).str.strip()
    d["group"] = (d["palette"] + "|" + d["bin_width"].astype(str) + "|" +
                  d["perturbation"].astype(str) + "|" + d["fill_mode"])
    files = d.drop_duplicates("file").set_index("file")[
        ["group", "original_size", "entropy", "mad", "second_derivative"]]
    settings = sorted(d["setting"].unique())
    cube = [d.pivot(index="file", columns="setting", values=t)
            .loc[files.index, settings].to_numpy(dtype=np.float64)
            for t in TARGETS]
    raw = np.stack(cube, axis=-1)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("missing or non-positive (file, setting) measurement")
    x = np.column_stack([np.log2(files["original_size"].to_numpy(float)),
                         files["entropy"].to_numpy(float),
                         np.log10(files["mad"].to_numpy(float) + EPS),
                         np.log10(files["second_derivative"].to_numpy(float) + EPS)])
    codec = [s.split()[0] for s in settings]
    return {"files": files, "settings": settings, "X": x, "Y": np.log(raw),
            "codec": codec}


def make_folds(groups, k, seed):
    """@return fold index per file: patterns shuffled with seed, dealt round-robin."""
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of = {g: i % k for i, g in enumerate(ug[perm])}
    return np.array([fold_of[g] for g in groups])


def mlp(n_in, n_out, width=64, depth=4):
    """@return depth hidden ReLU layers of width, then a linear layer."""
    layers, w = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(w, width), nn.ReLU()]
        w = width
    layers.append(nn.Linear(w, n_out))
    return nn.Sequential(*layers)


def split_val(train_idx, groups, seed, frac=0.1):
    """Hold out frac of the training patterns for early stopping.

    @return (fit indices, validation indices) into the file arrays
    """
    ug = np.unique(groups[train_idx])
    rng = np.random.default_rng(seed + 1)
    val_g = set(rng.choice(ug, size=max(1, int(frac * len(ug))), replace=False))
    is_val = np.array([g in val_g for g in groups[train_idx]])
    return train_idx[~is_val], train_idx[is_val]


def train_loop(model, step, val_loss, n_batches, max_epochs, patience, lr):
    """Adam with early stopping on the validation loss.

    @param step       step(b) -> training loss tensor for batch number b
    @param val_loss   () -> float validation loss
    @param n_batches  () -> batches in this epoch (reshuffles inside)
    @return (best epoch, best validation loss)
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_ep, best_state = float("inf"), 0, None
    for ep in range(max_epochs):
        model.train()
        for b in range(n_batches()):
            opt.zero_grad()
            loss = step(b)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = val_loss()
        if v < best - 1e-6:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    return best_ep, best


class Scaler:
    """Mean / std standardisation fitted on training rows."""

    def __init__(self, a, axis=0):
        self.mu = a.mean(axis=axis)
        self.sd = a.std(axis=axis) + 1e-8

    def fwd(self, a):
        return (a - self.mu) / self.sd

    def inv(self, a):
        return a * self.sd + self.mu


def fit_a(data, fit, val, test, dev, seed):
    """Model A: one row per (file, setting).

    @return predicted log targets for test files, S x 3 each; epochs; params
    """
    X, Y = data["X"], data["Y"]
    S = Y.shape[1]
    xs, ys = Scaler(X[fit]), Scaler(Y[fit].reshape(-1, 3))
    tx = torch.tensor(xs.fwd(X), dtype=torch.float32, device=dev)
    ty = torch.tensor(ys.fwd(Y.reshape(-1, 3)).reshape(Y.shape),
                      dtype=torch.float32, device=dev)
    eye = torch.eye(S, device=dev)
    torch.manual_seed(seed)
    model = mlp(4 + S, 3).to(dev)
    pairs = torch.tensor(np.array([(f, s) for f in fit for s in range(S)]), device=dev)
    vpairs = torch.tensor(np.array([(f, s) for f in val for s in range(S)]), device=dev)
    bs, order = 4096, {"p": None}

    def rows(p):
        return torch.cat([tx[p[:, 0]], eye[p[:, 1]]], dim=1), ty[p[:, 0], p[:, 1]]

    def n_batches():
        order["p"] = pairs[torch.randperm(len(pairs), device=dev)]
        return (len(pairs) + bs - 1) // bs

    def step(b):
        xb, yb = rows(order["p"][b * bs:(b + 1) * bs])
        return ((model(xb) - yb) ** 2).mean()

    def val_loss():
        xb, yb = rows(vpairs)
        return ((model(xb) - yb) ** 2).mean().item()

    ep, _ = train_loop(model, step, val_loss, n_batches, 200, 15, 1e-3)
    tp = torch.tensor(np.array([(f, s) for f in test for s in range(S)]), device=dev)
    with torch.no_grad():
        out = model(rows(tp)[0]).cpu().numpy()
    pred = ys.inv(out).reshape(len(test), S, 3)
    return pred, ep, sum(p.numel() for p in model.parameters())


def fit_b(data, fit, val, test, dev, seed):
    """Model B: one row per file, 3 outputs per setting.

    @return predicted log targets for test files, S x 3 each; epochs; params
    """
    X, Y = data["X"], data["Y"]
    S = Y.shape[1]
    flat = Y.reshape(len(Y), -1)
    xs, ys = Scaler(X[fit]), Scaler(flat[fit])
    tx = torch.tensor(xs.fwd(X), dtype=torch.float32, device=dev)
    ty = torch.tensor(ys.fwd(flat), dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    model = mlp(4, 3 * S).to(dev)
    tfit = torch.tensor(fit, device=dev)
    tval = torch.tensor(val, device=dev)
    bs, order = 256, {"p": None}

    def n_batches():
        order["p"] = tfit[torch.randperm(len(tfit), device=dev)]
        return (len(tfit) + bs - 1) // bs

    def step(b):
        i = order["p"][b * bs:(b + 1) * bs]
        return ((model(tx[i]) - ty[i]) ** 2).mean()

    def val_loss():
        return ((model(tx[tval]) - ty[tval]) ** 2).mean().item()

    ep, _ = train_loop(model, step, val_loss, n_batches, 3000, 100, 1e-3)
    with torch.no_grad():
        out = model(tx[torch.tensor(test, device=dev)]).cpu().numpy()
    pred = ys.inv(out).reshape(len(test), S, 3)
    return pred, ep, sum(p.numel() for p in model.parameters())


def fit_baseline(data, fit, test):
    """Per (setting, size) mean of the log targets over the training files."""
    size = data["files"]["original_size"].to_numpy()
    Y = data["Y"]
    pred = np.empty((len(test),) + Y.shape[1:])
    for sz in np.unique(size):
        m = Y[fit][size[fit] == sz].mean(axis=0)
        pred[size[test] == sz] = m
    return pred


def costs(logy, nbytes, bw):
    """True or predicted cost in ms per (file, setting) at bw GB/s."""
    ct, dt, r = np.exp(logy[..., 0]), np.exp(logy[..., 1]), np.exp(logy[..., 2])
    return ct + dt + (nbytes[:, None] / r) / (bw * 1e6)


def selection(pred, true, nbytes, fixed_pick):
    """Top-1 and regret of picking by predicted cost (or ratio).

    @param fixed_pick  objective -> setting index of the training folds' best
                       single setting (None for the model's own picks)
    @return list of dict rows, one per objective
    """
    out = []
    objectives = [("ratio", None)] + [(f"cost@{bw:g}GB/s", bw) for bw in BWS]
    for name, bw in objectives:
        if bw is None:
            tc = -true[..., 2]          # minimise compressed size
            pc = -pred[..., 2] if pred is not None else None
        else:
            tc = costs(true, nbytes, bw)
            pc = costs(pred, nbytes, bw) if pred is not None else None
        best = tc.argmin(axis=1)
        pick = (np.full(len(tc), fixed_pick[name]) if pred is None
                else pc.argmin(axis=1))
        rows = np.arange(len(tc))
        if bw is None:  # regret in compressed bytes: best ratio / picked ratio
            reg = np.exp(true[rows, best, 2] - true[rows, pick, 2]) - 1
            tot = ((nbytes / np.exp(true[rows, pick, 2])).sum() /
                   (nbytes / np.exp(true[rows, best, 2])).sum() - 1)
        else:
            reg = tc[rows, pick] / tc[rows, best] - 1
            tot = tc[rows, pick].sum() / tc[rows, best].sum() - 1
        out.append({"objective": name, "top1": float((pick == best).mean()),
                    "regret_mean": float(reg.mean()),
                    "regret_median": float(np.median(reg)),
                    "regret_total": float(tot)})
    return out


def fixed_best(data, fit):
    """Best single setting on the training files, per objective."""
    Y, nbytes = data["Y"][fit], data["files"]["original_size"].to_numpy(float)[fit]
    res = {"ratio": int((nbytes[:, None] / np.exp(Y[..., 2])).sum(axis=0).argmin())}
    for bw in BWS:
        res[f"cost@{bw:g}GB/s"] = int(costs(Y, nbytes, bw).sum(axis=0).argmin())
    return res


def accuracy_rows(model, fold, pred, true, data, test):
    """APE summaries: overall, per codec, per size."""
    ape = np.abs(np.exp(pred) - np.exp(true)) / np.exp(true)  # F x S x 3
    size = data["files"]["original_size"].to_numpy()[test]
    codec = np.array(data["codec"])
    rows = []
    for t, name in enumerate(NAMES):
        a = ape[..., t]
        rows.append({"model": model, "fold": fold, "target": name, "by": "all",
                     "key": "all", "mape": a.mean(), "median_ape": np.median(a)})
        for c in np.unique(codec):
            v = a[:, codec == c]
            rows.append({"model": model, "fold": fold, "target": name, "by": "codec",
                         "key": c, "mape": v.mean(), "median_ape": np.median(v)})
        for sz in np.unique(size):
            v = a[size == sz]
            rows.append({"model": model, "fold": fold, "target": name, "by": "size",
                         "key": str(sz), "mape": v.mean(), "median_ape": np.median(v)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=os.path.join(HERE, "nn-v2-training", "corpus_v2.csv.xz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "codec-sweep", "results", "nn-v2"))
    ap.add_argument("--pred-out", default="/mnt/nvme0/corpus-sweep/nn-v2/cv_predictions.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(a.pred_out), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.monotonic()
    data = load(a.csv)
    F, S = data["Y"].shape[:2]
    groups = data["files"]["group"].to_numpy()
    fold = make_folds(groups, a.folds, a.seed)
    nbytes = data["files"]["original_size"].to_numpy(float)
    print(f"{F} files x {S} settings, {len(np.unique(groups))} patterns, "
          f"{a.folds} folds, device {dev}; loaded in {1e3 * (time.monotonic() - t0):.0f} ms")

    preds = {m: np.zeros_like(data["Y"]) for m in ("baseline", "A", "B")}
    acc, sel, info = [], [], []
    for k in range(a.folds):
        test = np.where(fold == k)[0]
        train = np.where(fold != k)[0]
        fit, val = split_val(train, groups, a.seed + k)
        fixed = fixed_best(data, train)
        sel += [dict(r, model="fixed best setting", fold=k)
                for r in selection(None, data["Y"][test], nbytes[test], fixed)]
        for m in ("baseline", "A", "B"):
            t1 = time.monotonic()
            if m == "baseline":
                p, ep, n_par = fit_baseline(data, train, test), 0, 0
            elif m == "A":
                p, ep, n_par = fit_a(data, fit, val, test, dev, a.seed + k)
            else:
                p, ep, n_par = fit_b(data, fit, val, test, dev, a.seed + k)
            ms = 1e3 * (time.monotonic() - t1)
            preds[m][test] = p
            acc += accuracy_rows(m, k, p, data["Y"][test], data, test)
            sel += [dict(r, model=m, fold=k)
                    for r in selection(p, data["Y"][test], nbytes[test], None)]
            info.append({"model": m, "fold": k, "params": n_par, "best_epoch": ep,
                         "train_ms": ms, "n_fit": len(fit), "n_test": len(test)})
            mape = {r["target"]: r["mape"] for r in acc
                    if r["model"] == m and r["fold"] == k and r["by"] == "all"}
            print(f"fold {k} {m:8s} params {n_par:6d} epoch {ep:4d} {ms:9.0f} ms  "
                  + "  ".join(f"{t} MAPE {100 * v:6.2f}%" for t, v in mape.items()),
                  flush=True)

    acc, sel, info = pd.DataFrame(acc), pd.DataFrame(sel), pd.DataFrame(info)
    acc.to_csv(os.path.join(a.out, "cv_accuracy.csv"), index=False)
    sel.to_csv(os.path.join(a.out, "cv_selection.csv"), index=False)
    info.to_csv(os.path.join(a.out, "cv_training.csv"), index=False)
    np.savez_compressed(a.pred_out, fold=fold, settings=np.array(data["settings"]),
                        files=data["files"].index.to_numpy(), true=data["Y"], **preds)

    pd.set_option("display.width", 200)
    s = acc[acc.by == "all"].groupby(["model", "target"])[["mape", "median_ape"]]
    print("\nAccuracy over 5 folds (mean +- std of the fold values, %):")
    print((100 * s.agg(["mean", "std"])).round(2).to_string())
    g = sel.groupby(["objective", "model"])[["top1", "regret_mean", "regret_total"]]
    print("\nSelection on held-out files (mean over folds):")
    print(g.mean().round(4).to_string())
    print(f"\nwrote {os.path.abspath(a.out)} and {a.pred_out}; "
          f"total {1e3 * (time.monotonic() - t0):.0f} ms")


if __name__ == "__main__":
    main()

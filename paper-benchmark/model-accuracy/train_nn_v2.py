#!/usr/bin/env python3
"""Train the NeuroPress v2 predictor (model B) on the whole corpus and save it.

    train_nn_v2.py [--csv corpus_v2.csv] [--out-dir DIR] [--epochs N] [--seed S]

Model B as cross-validated by nn_v2_cv.py: 4 data features -> 64 x 4 ReLU ->
3 outputs per setting (natural log of compress ms, decompress ms and ratio),
every setting in one forward pass. Trained on all 9800 files with the CV
hyper-parameters (Adam, lr 1e-3, batch 256); with no data left for early
stopping, it runs a fixed number of epochs: the median best epoch of the 5 CV
folds (results/nn-v2/cv_training.csv) unless --epochs is given.

Writes to DIR:
  model_v2.nnwt   little-endian binary, NNWT container version 3:
                    u32 magic 0x4E4E5754 ('NNWT'), u32 version 3,
                    u32 n_layers, u32 dims[n_layers + 1] (input, hidden...,
                    output), u32 feature_set (1 = the four features below),
                    u32 outputs_per_setting (3), u32 n_settings,
                    u32 table_bytes, the settings as table_bytes of
                    '\n'-separated canonical specs (output order),
                    f32 x_mean[in], x_std[in], y_mean[out], y_std[out],
                    then per layer f32 W[out][in] (row-major) and f32 b[out]
  model_v2.json   everything needed to use it: feature definitions, output
                  order (setting s -> outputs 3s, 3s+1, 3s+2), the settings
                  list, transforms, training details and the CV accuracy
  model_v2.pt     the PyTorch state dict
After writing, the binary is read back and run in NumPy; its outputs must
match PyTorch's on every file.
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import pandas as pd
import torch

import nn_v2_cv as cv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAGIC, VERSION = 0x4E4E5754, 3
FEATURE_SET, OUTPUTS_PER_SETTING = 1, 3
FEATURES = [
    "log2(chunk bytes)",
    "Shannon entropy of the chunk's bytes, bits/byte (0-8)",
    "log10(mean |x - mean(x)| / (max(x) - min(x)) + 1e-12), x as float32 "
    "values in float64; range 0 -> 1",
    "log10(mean |x[i+2] - 2 x[i+1] + x[i]| / (max(x) - min(x)) + 1e-12); "
    "range 0 -> 1",
]


def train_full(data, epochs, seed, dev):
    """Fit model B on every file for a fixed number of epochs.

    @return (model, x scaler, y scaler, train ms)
    """
    X, Y = data["X"], data["Y"]
    flat = Y.reshape(len(Y), -1)
    xs, ys = cv.Scaler(X), cv.Scaler(flat)
    tx = torch.tensor(xs.fwd(X), dtype=torch.float32, device=dev)
    ty = torch.tensor(ys.fwd(flat), dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    model = cv.mlp(X.shape[1], flat.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n, bs = len(X), 256
    t0 = time.monotonic()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, bs):
            i = perm[b:b + bs]
            opt.zero_grad()
            loss = ((model(tx[i]) - ty[i]) ** 2).mean()
            loss.backward()
            opt.step()
        if ep % 250 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:5d}  train loss {loss.item():.5f}", flush=True)
    model.eval()
    return model, xs, ys, 1e3 * (time.monotonic() - t0)


def linears(model):
    """@return the model's nn.Linear layers in order."""
    return [m for m in model if isinstance(m, torch.nn.Linear)]


def write_nnwt(path, dims, settings, norms, layers):
    """Write the version-3 NNWT binary described in the module docstring.

    @param dims      layer widths, input first
    @param settings  canonical setting specs in output order
    @param norms     (x_mean, x_std, y_mean, y_std)
    @param layers    [(W [out][in], b [out]), ...]
    """
    if dims[-1] != OUTPUTS_PER_SETTING * len(settings):
        raise ValueError("output width does not match the settings table")
    table = "\n".join(settings).encode()
    f32 = lambda a: np.asarray(a, dtype="<f4").tobytes()
    with open(path, "wb") as f:
        f.write(struct.pack("<3I", MAGIC, VERSION, len(layers)))
        f.write(struct.pack(f"<{len(dims)}I", *dims))
        f.write(struct.pack("<4I", FEATURE_SET, OUTPUTS_PER_SETTING, len(settings), len(table)))
        f.write(table)
        for v in norms:
            f.write(f32(v))
        for w, b in layers:
            f.write(f32(w))
            f.write(f32(b))


def model_layers(model):
    """@return (dims, [(W, b), ...]) of a trained model as NumPy arrays."""
    lin = linears(model)
    dims = [lin[0].in_features] + [m.out_features for m in lin]
    return dims, [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
                  for m in lin]


def read_nnwt(path):
    """Read a version-3 NNWT file back.

    @return (dims, x_mean, x_std, y_mean, y_std, [(W, b), ...], settings)
    """
    buf = open(path, "rb").read()
    magic, ver, nl = struct.unpack_from("<3I", buf, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not an NNWT v3 file")
    off = 12
    dims = list(struct.unpack_from(f"<{nl + 1}I", buf, off))
    off += 4 * (nl + 1)
    fset, per, n_set, tbytes = struct.unpack_from("<4I", buf, off)
    off += 16
    settings = buf[off:off + tbytes].decode().split("\n")
    off += tbytes
    if fset != FEATURE_SET or per != OUTPUTS_PER_SETTING or len(settings) != n_set \
            or dims[-1] != per * n_set:
        raise ValueError("feature set or settings table does not match the network")

    def take(n):
        nonlocal off
        a = np.frombuffer(buf, dtype="<f4", count=n, offset=off)
        off += 4 * n
        return a

    xm, xsd, ym, ysd = take(dims[0]), take(dims[0]), take(dims[-1]), take(dims[-1])
    layers = []
    for i in range(nl):
        w = take(dims[i + 1] * dims[i]).reshape(dims[i + 1], dims[i])
        layers.append((w, take(dims[i + 1])))
    if off != len(buf):
        raise ValueError(f"{len(buf) - off} trailing bytes")
    return dims, xm, xsd, ym, ysd, layers, settings


def numpy_forward(path, x_raw):
    """Run the saved file in float32 NumPy: log targets, files x outputs."""
    _, xm, xsd, ym, ysd, layers, _ = read_nnwt(path)
    h = ((x_raw.astype(np.float32) - xm) / xsd).astype(np.float32)
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h * ysd + ym


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default="/mnt/nvme0/corpus-sweep/corpus_v2.csv")
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "context-transport-primitives", "src", "compress", "model", "weights", "v2"))
    ap.add_argument("--cv-dir", default=os.path.join(HERE, "..", "codec-sweep", "results", "nn-v2"))
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cvt = pd.read_csv(os.path.join(a.cv_dir, "cv_training.csv"))
    cv_epochs = cvt.loc[cvt.model == "B", "best_epoch"].tolist()
    epochs = a.epochs or int(np.median(cv_epochs)) + 1
    data = cv.load(a.csv)
    F, S = data["Y"].shape[:2]
    print(f"{F} files x {S} settings; training model B for {epochs} epochs "
          f"(CV best epochs {cv_epochs}) on {dev}")

    model, xs, ys, train_ms = train_full(data, epochs, a.seed, dev)
    base = os.path.join(a.out_dir, "model_v2")
    dims, layers = model_layers(model)
    write_nnwt(base + ".nnwt", dims, data["settings"], (xs.mu, xs.sd, ys.mu, ys.sd), layers)
    torch.save(model.state_dict(), base + ".pt")

    with torch.no_grad():
        tx = torch.tensor(xs.fwd(data["X"]), dtype=torch.float32, device=dev)
        torch_out = ys.inv(model(tx).cpu().numpy().astype(np.float64))
    np_out = numpy_forward(base + ".nnwt", data["X"])
    diff = float(np.abs(np_out - torch_out).max())
    if diff > 1e-3:
        raise SystemExit(f"saved weights do not reproduce the model: max diff {diff}")

    pred = torch_out.reshape(F, S, 3)
    ape = np.abs(np.exp(pred) - np.exp(data["Y"])) / np.exp(data["Y"])
    fit_mape = {n: float(ape[..., t].mean()) for t, n in enumerate(cv.NAMES)}
    acc = pd.read_csv(os.path.join(a.cv_dir, "cv_accuracy.csv"))
    b = acc[(acc.model == "B") & (acc.by == "all")].groupby("target")
    cv_mape = {t: {"mean": float(g.mape.mean()), "std": float(g.mape.std()),
                   "median_ape": float(g.median_ape.mean())} for t, g in b}

    meta = {
        "name": "NeuroPress v2 (model B)",
        "format": {"file": "model_v2.nnwt", "magic": "0x4E4E5754", "version": VERSION,
                   "feature_set": FEATURE_SET,
                   "layout": "u32 magic, u32 version, u32 n_layers, u32 dims[n_layers+1], "
                             "u32 feature_set, u32 outputs_per_setting, u32 n_settings, "
                             "u32 table_bytes, settings table ('\\n'-separated specs), "
                             "f32 x_mean[in], x_std[in], y_mean[out], y_std[out], "
                             "per layer f32 W[out][in] row-major then f32 b[out]; little-endian"},
        "architecture": {"dims": dims, "hidden_activation": "ReLU", "output": "linear",
                         "params": int(sum(p.numel() for p in model.parameters()))},
        "inputs": {"features": FEATURES,
                   "normalise": "x_norm = (x - x_mean) / x_std, then the network"},
        "outputs": {"denormalise": "y = y_norm * y_std + y_mean",
                    "order": "setting s -> outputs 3s, 3s+1, 3s+2",
                    "per_setting": ["ln(compress ms)", "ln(decompress ms)", "ln(compression ratio)"],
                    "inverse": "exp(y)",
                    "times": "GPU kernel time (CUDA events), median of 3 reps, "
                             "byte/bit shuffle and un-shuffle included",
                    "settings": data["settings"]},
        "training": {"data": os.path.abspath(a.csv), "files": F, "rows": F * S,
                     "dtype": "float32, synthetic NeuroPress corpus (9800 files, 16 KiB-4 MiB)",
                     "optimizer": "Adam lr 1e-3, MSE on standardised log targets",
                     "batch": 256, "epochs": epochs, "seed": a.seed,
                     "train_ms": train_ms, "gpu": torch.cuda.get_device_name(0)
                     if dev.type == "cuda" else "cpu"},
        "accuracy": {"cv_5fold_mape": cv_mape, "cv_best_epochs": cv_epochs,
                     "in_sample_mape": fit_mape,
                     "note": "CV values are the expected accuracy on unseen data; "
                             "in-sample values are on the training data itself"},
        "verify": {"numpy_vs_torch_max_abs_diff_log": diff},
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"trained in {train_ms:.0f} ms; saved {base}.nnwt ({os.path.getsize(base + '.nnwt')} B), "
          f".json, .pt; numpy reload matches torch (max diff {diff:.2e} in log units)")
    print("in-sample MAPE: " + ", ".join(f"{k} {100 * v:.2f}%" for k, v in fit_mape.items()))
    print("CV MAPE (unseen data): " + ", ".join(
        f"{k} {100 * v['mean']:.2f}% +- {100 * v['std']:.2f}" for k, v in cv_mape.items()))


if __name__ == "__main__":
    main()

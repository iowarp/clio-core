#!/usr/bin/env python3
"""The two shipped models, run offline over arbitrary rows.

Neither model is retrained here. Both are loaded from the artifacts the project
ships and evaluated as they are:

  NeuroPressNN   context-transport-primitives/src/compress/model/weights/model.nnwt
                 (byte-identical to upstream NeuroPress's own file)
  XGBoost        NeuroPress/neural_net/weights/xgb_model.pkl

WHY OFFLINE AT ALL, when an exploration log already carries the NN's
predictions: the log carries three of them (ratio, compression time,
decompression time) and not the fourth. Its `psnr_db` column is the ANALYTICAL
PSNR derived from (range, error bound), not the network's PSNR head, so a PSNR
row scored from the log would be scoring a different quantity. Running the
network here yields all four from one model state, and gives the static
counterpart to HCompress's seed-only row: the shipped weights, before online
learning has moved them.

THE FORWARD PASS IS THE KERNEL'S, NOT A REIMPLEMENTATION OF THE IDEA. Every
step below is taken from neuropress_nn_gpu_kernels.cu, because the clamps are
not incidental: the time floor and the ratio cap change the reported error by
more than most modelling choices do.

  NeuroPressStandardize  s = (raw - x_mean)/max(x_std,1e-8), then the SOFT
                         feature bound: above hi, hi + log1p(s-hi); below lo,
                         lo - log1p(lo-s). lo/hi are the training min/max in
                         sigma units. 8 MiB chunks are ABOVE the training
                         maximum (4 MiB), so this bound is active on every
                         chunk of every real workload here.
  forward                4 hidden layers of 64, ReLU; linear output of 8.
  inverse                expm1 on outputs 0,1,2 after de-standardising; output
                         3 (PSNR) is identity.
  clamps                 sanity first (ct,dt in [1e-6,1e6]; ratio in [0.1,1e5];
                         psnr in [0,120]), then policy (ct,dt floored at
                         pred_time_floor = 1 ms; ratio capped at 100).

The lossless error-bound SENTINEL matters as much: a lossless candidate is fed
error_bound = 1e-7, not 0, because that is how the model was trained
(neural_net/core/configs.py) and what the kernel does at inference.
"""
from __future__ import annotations

import argparse
import pickle
import struct
import sys
import warnings

import numpy as np
import pandas as pd

#: Training/inference order of the 8 algorithms (neural_net/core/configs.py).
ALGORITHMS = ["lz4", "snappy", "deflate", "gdeflate", "zstd", "ans",
              "cascaded", "bitcomp"]
#: The lossless error-bound sentinel, at inference and in training.
LOSSLESS_EB = 1e-7
#: Policy clamps (RankingWeights defaults / NeuroPressCost::kMinTimeMs).
PRED_TIME_FLOOR_MS = 1.0
RATIO_CAP = 100.0


# ---------------------------------------------------------------------------
# NeuroPress dense net, from model.nnwt
# ---------------------------------------------------------------------------
class NeuroPressNN:
    """The shipped .nnwt weights and the kernel's forward pass."""

    def __init__(self, path: str):
        b = open(path, "rb").read()
        magic, ver, nlayers, idim, hdim, odim = struct.unpack("<6I", b[:24])
        if magic != 0x4E4E5754:
            raise ValueError(f"{path}: not an NNWT file (magic {magic:#x})")
        self.version, self.idim, self.hdim, self.odim = ver, idim, hdim, odim
        off = 24

        def take(n):
            nonlocal off
            a = np.frombuffer(b, dtype="<f4", count=n, offset=off).astype(np.float64)
            off += 4 * n
            return a

        self.x_means, self.x_stds = take(idim), take(idim)
        self.y_means, self.y_stds = take(odim), take(odim)
        fan = [(hdim, idim)] + [(hdim, hdim)] * (nlayers - 2) + [(odim, hdim)]
        self.W, self.b = [], []
        for (o, i) in fan:
            self.W.append(take(o * i).reshape(o, i))
            self.b.append(take(o))
        self.x_mins, self.x_maxs = take(idim), take(idim)
        if off != len(b):
            raise ValueError(f"{path}: {len(b) - off} trailing byte(s); layout mismatch")

    def standardize(self, X: np.ndarray) -> np.ndarray:
        sd = np.maximum(self.x_stds, 1e-8)
        s = (X - self.x_means) / sd
        lo = (self.x_mins - self.x_means) / sd
        hi = (self.x_maxs - self.x_means) / sd
        # NeuroPressSoftBoundSigma: log1p growth outside the training box, so an
        # out-of-range input is compressed towards the boundary instead of
        # extrapolating linearly.
        out = np.where(s > hi, hi + np.log1p(np.maximum(s - hi, 0.0)), s)
        out = np.where(s < lo, lo - np.log1p(np.maximum(lo - s, 0.0)), out)
        return out

    def raw_outputs(self, X: np.ndarray) -> np.ndarray:
        h = self.standardize(X)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(0.0, h @ W.T + b)
        return h @ self.W[-1].T + self.b[-1]

    def predict(self, X: np.ndarray, time_floor=PRED_TIME_FLOOR_MS,
                ratio_cap=RATIO_CAP) -> pd.DataFrame:
        y = self.raw_outputs(X) * self.y_stds + self.y_means
        ct = np.expm1(y[:, 0])
        dt = np.expm1(y[:, 1])
        ratio = np.expm1(y[:, 2])
        psnr = y[:, 3]
        # Sanity clamps first, then policy -- the kernel's order.
        ct = np.clip(ct, 1e-6, 1e6)
        dt = np.clip(dt, 1e-6, 1e6)
        ratio = np.clip(ratio, 0.1, 1e5)
        psnr = np.clip(psnr, 0.0, 120.0)
        return pd.DataFrame({
            "pred_ct_ms": np.maximum(time_floor, ct),
            "pred_dt_ms": np.maximum(time_floor, dt),
            "pred_ratio": np.minimum(ratio_cap, ratio),
            "pred_psnr_db": psnr,
        })


def nn_inputs(algo: pd.Series, quant: pd.Series, shuffle: pd.Series,
              error_bound: pd.Series, size: pd.Series, entropy: pd.Series,
              mad: pd.Series, second_deriv: pd.Series) -> np.ndarray:
    """The network's 8 raw inputs, in its own order."""
    q = quant.to_numpy(dtype=float)
    eb = np.where(q > 0, error_bound.to_numpy(dtype=float), LOSSLESS_EB)
    return np.column_stack([
        algo.to_numpy(dtype=float),
        q,
        (shuffle.to_numpy(dtype=float) > 0).astype(float),
        eb,
        size.to_numpy(dtype=float),
        entropy.to_numpy(dtype=float),
        mad.to_numpy(dtype=float),
        second_deriv.to_numpy(dtype=float),
    ])


# ---------------------------------------------------------------------------
# XGBoost, from xgb_model.pkl
# ---------------------------------------------------------------------------
class XGBoostBaseline:
    """Upstream's trained XGBoost checkpoint: one regressor per output.

    The checkpoint stores `feature_names`, which pins the encoding:

        alg_lz4 ... alg_bitcomp, quant_enc, shuffle_enc,
        error_bound_enc, data_size_enc, entropy, mad, second_derivative

    The two encoded columns are NOT documented anywhere, and the pickle keeps
    only their standardisation statistics -- so they are recovered from those
    statistics, which pin them exactly:

      error_bound_enc  mean -3.2485, std 2.2768. log10 over the four training
                       bounds {1e-7, 1e-3, 1e-2, 1e-1} gives {-7,-3,-2,-1},
                       whose mean is -3.25 and std 2.2776. No other candidate
                       encoding (ln, raw) comes near either statistic.
      data_size_enc    mean 19.006, std 2.2387. log2 over the training sizes
                       64 KiB..4 MiB gives {16..22}, mean 19.0. ln would give
                       ~13.2.
    """

    OUTPUTS = {
        "comp_time_log": ("pred_ct_ms", "expm1"),
        "decomp_time_log": ("pred_dt_ms", "expm1"),
        "ratio_log": ("pred_ratio", "expm1"),
        "psnr_clamped": ("pred_psnr_db", "identity"),
    }

    def __init__(self, path: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ck = pickle.load(open(path, "rb"))
        self.models = ck["models"]
        self.x_means = np.asarray(ck["x_means"], dtype=np.float64)
        self.x_stds = np.asarray(ck["x_stds"], dtype=np.float64)
        self.feature_names = list(ck["feature_names"])
        ym, ys = ck["y_means"], ck["y_stds"]
        names = list(self.models.keys())
        self.y_means = {n: float(np.asarray(ym)[i]) for i, n in enumerate(names)}
        self.y_stds = {n: float(np.asarray(ys)[i]) for i, n in enumerate(names)}

    def encode(self, algo, quant, shuffle, error_bound, size, entropy, mad,
               second_deriv) -> np.ndarray:
        n = len(algo)
        onehot = np.zeros((n, len(ALGORITHMS)))
        onehot[np.arange(n), algo.to_numpy(dtype=int)] = 1.0
        q = quant.to_numpy(dtype=float)
        eb = np.where(q > 0, error_bound.to_numpy(dtype=float), LOSSLESS_EB)
        X = np.column_stack([
            onehot,
            q,
            (shuffle.to_numpy(dtype=float) > 0).astype(float),
            np.log10(eb),
            np.log2(size.to_numpy(dtype=float)),
            entropy.to_numpy(dtype=float),
            mad.to_numpy(dtype=float),
            second_deriv.to_numpy(dtype=float),
        ])
        return (X - self.x_means) / self.x_stds

    def predict(self, Xn: np.ndarray, time_floor=PRED_TIME_FLOOR_MS,
                ratio_cap=RATIO_CAP) -> pd.DataFrame:
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for internal, (name, inv) in self.OUTPUTS.items():
                m = self.models[internal]
                y = (np.asarray(m.predict(Xn), dtype=np.float64)
                     * self.y_stds[internal] + self.y_means[internal])
                out[name] = np.expm1(y) if inv == "expm1" else y
        # The SAME clamps the NN's predictions carry. Applying them to one model
        # and not the other would make the table a comparison of policies.
        out["pred_ct_ms"] = np.maximum(time_floor, np.clip(out["pred_ct_ms"], 1e-6, 1e6))
        out["pred_dt_ms"] = np.maximum(time_floor, np.clip(out["pred_dt_ms"], 1e-6, 1e6))
        out["pred_ratio"] = np.minimum(ratio_cap, np.clip(out["pred_ratio"], 0.1, 1e5))
        out["pred_psnr_db"] = np.clip(out["pred_psnr_db"], 0.0, 120.0)
        return pd.DataFrame(out)


def main() -> int:
    """Self-check: dimensions, and a round trip through both models."""
    ap = argparse.ArgumentParser(description="print both models' shapes")
    ap.add_argument("--nnwt", required=True)
    ap.add_argument("--xgb", required=True)
    a = ap.parse_args()
    nn = NeuroPressNN(a.nnwt)
    print(f"nnwt v{nn.version}: {nn.idim} -> [{nn.hdim}] x {len(nn.W)-1} -> {nn.odim}")
    print("  x_mins", nn.x_mins, "\n  x_maxs", nn.x_maxs)
    xgb = XGBoostBaseline(a.xgb)
    print(f"xgb: {len(xgb.feature_names)} features, {len(xgb.models)} outputs")
    print("  ", xgb.feature_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())

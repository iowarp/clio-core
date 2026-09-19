#!/usr/bin/env python3
"""Build one evaluation frame per setting, plus HCompress's seed.

    prepare_inputs.py --corpus benchmark_results_600k.csv --campaign DIR \
                      --out DIR [--workload nyx ...]

Every model in the table is scored on the SAME rows, so the rows are built once
here and never rebuilt per model. One row is one (chunk, candidate
configuration) pair:

    chunk seq library distribution data_type data_format bytes executed
    ct_ms dt_ms ratio psnr_db                       <- MEASURED, NaN if not
    algo_idx quantize shuffle error_bound entropy mad second_deriv
    log_pred_ct_ms log_pred_dt_ms log_pred_ratio    <- the RUN's own NN output

SETTINGS

  synthetic  the held-out split of the corpus our model was trained on,
             reproducing upstream's own split exactly (neural_net/core/data.py
             encode_and_split: filter success, sort files, shuffle with
             RandomState(42), last 20% of FILES are validation). Splitting by
             file, not by row, is what keeps the 64 configurations of one
             buffer from straddling the split.

             16 KiB files are DROPPED. The shipped weights' own input box says
             they were not trained on them: x_mins[4] = 65,536 with the corpus
             containing 16,384. Keeping them would put rows in the
             "in-distribution" block that are out of distribution for the model
             being measured.

  vpic nyx lammps warpx ai
             the measured campaign. All 32 configurations were measured per
             chunk, so `ratio`, `ct_ms` and `dt_ms` are measurements for every
             candidate and not just for the one that ran.

WHAT "EXECUTED" MEANS, AND WHY IT IS COMPUTED RATHER THAN ASSUMED

HCompress's feedback sees "the measured cost of the executed choice". In the
campaign that is the row the model picked (role=primary). The corpus has no
executed choice -- it is an exhaustive profile -- so the pick is computed the
way the deployed runtime computes it: the NN's own predictions scored by its
own cost model, cost = max(1ms,ct) + max(1ms,dt) + bytes/(min(ratio,100)*5e6),
and the argmin is what a run would have executed. Marking an arbitrary row
instead would hand the baseline feedback from a configuration nothing would
have run.

INPUTS THAT ARE NOT AVAILABLE, and are reported as such rather than guessed:

  data_format   Neither side has one. The corpus profiles raw .bin buffers and
                the campaign compresses raw in-situ/replay buffers; there is no
                container, no HDF5, no format label anywhere. The column is
                written empty, HCompress encodes no term for it, and the table
                reports the input as unavailable.
  data_type     float32 everywhere, in the corpus (its only dtype) and in every
                workload. It is written, but being constant it can carry no
                information -- the table reports that too.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_offline import ALGORITHMS, LOSSLESS_EB, NeuroPressNN, nn_inputs  # noqa: E402

#: Cost-model constants, from NeuroPressCost / RankingWeights defaults.
BW_BYTES_PER_MS = 5e6
MIN_TIME_MS = 1.0
RATIO_CAP = 100.0
#: Upstream's split (neural_net/core/data.py).
SPLIT_SEED = 42
VAL_FRACTION = 0.2
#: Outside the shipped weights' input box (nnwt x_mins[4]).
MIN_TRAINED_SIZE = 65536

WORKLOADS = ["vpic", "nyx", "lammps", "warpx", "ai"]
ROW_COLUMNS = ["chunk", "seq", "library", "distribution", "data_type",
               "data_format", "bytes", "executed", "ct_ms", "dt_ms", "ratio",
               "psnr_db", "algo_idx", "quantize", "shuffle", "error_bound",
               "entropy", "mad", "second_deriv", "log_pred_ct_ms",
               "log_pred_dt_ms", "log_pred_ratio"]
EVAL_COLUMNS = ["chunk", "seq", "data_type", "data_format", "library",
                "distribution", "bytes", "ct_ms", "dt_ms", "ratio", "executed"]


def library_key(algorithm: pd.Series, quantize, shuffle) -> pd.Series:
    """The configured compression option, as HCompress keys its seed.

    Algorithm plus the two preprocessor bits, and deliberately NOT the error
    bound: HCompress's input list has no error bound at all, so one key covers
    every bound -- in the seed that averages the profiler's three bounds, and
    at evaluation the campaign's 0.05 lands on that same key. Keying on the
    bare codec instead would force one number to cover the quantized and
    lossless variants together, which no reading of the paper requires and
    which would understate the baseline.
    """
    q = np.where(np.asarray(quantize, dtype=float) > 0, "1", "0")
    s = np.where(np.asarray(shuffle, dtype=float) > 0, "1", "0")
    algo = algorithm.astype(str).str.replace("^nvcomp-", "", regex=True)
    return pd.Series([f"{a}|q{qq}|s{ss}" for a, qq, ss in zip(algo, q, s)],
                     index=algorithm.index)


def nn_pick(rows: pd.DataFrame, nn: NeuroPressNN) -> np.ndarray:
    """Per chunk, the configuration the deployed runtime would have executed."""
    X = nn_inputs(rows.algo_idx, rows.quantize, rows.shuffle, rows.error_bound,
                  rows.bytes, rows.entropy, rows.mad, rows.second_deriv)
    p = nn.predict(X)
    cost = (np.maximum(MIN_TIME_MS, p.pred_ct_ms)
            + np.maximum(MIN_TIME_MS, p.pred_dt_ms)
            + rows.bytes.to_numpy(dtype=float)
            / (np.minimum(RATIO_CAP, p.pred_ratio) * BW_BYTES_PER_MS))
    d = pd.DataFrame({"chunk": rows.chunk.to_numpy(), "cost": cost})
    best = d.groupby("chunk", sort=False)["cost"].transform("min")
    # Ties: the first candidate in file order wins, as a stable_sort does.
    pick = (d.cost <= best).to_numpy()
    first = np.zeros(len(d), dtype=bool)
    seen = set()
    for i, (c, ok) in enumerate(zip(d.chunk, pick)):
        if ok and c not in seen:
            seen.add(c)
            first[i] = True
    return first.astype(int)


# ---------------------------------------------------------------------------
def build_synthetic(corpus: str, nn: NeuroPressNN):
    df = pd.read_csv(corpus)
    n_all = len(df)
    df = df[df["success"] == True].copy()                      # noqa: E712
    n_ok = len(df)
    small = df.original_size < MIN_TRAINED_SIZE
    df = df[~small].copy()

    files = sorted(df.file.unique())
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(files)
    k = int(len(files) * (1 - VAL_FRACTION))
    train_files, val_files = set(files[:k]), set(files[k:])

    def frame(d: pd.DataFrame) -> pd.DataFrame:
        q = (d.quantization == "linear").astype(int)
        out = pd.DataFrame({
            "chunk": d.file.to_numpy(),
            "library": library_key(d.algorithm, q, d.shuffle).to_numpy(),
            "distribution": d.palette.to_numpy(),
            "data_type": d.dtype.to_numpy(),
            "data_format": "",
            "bytes": d.original_size.to_numpy(dtype=float),
            "ct_ms": d.compression_time_ms.to_numpy(dtype=float),
            "dt_ms": d.decompression_time_ms.to_numpy(dtype=float),
            "ratio": d.compression_ratio.to_numpy(dtype=float),
            # inf on a lossless row: the reconstruction is exact, so there is
            # no finite PSNR to predict OR to score against.
            "psnr_db": pd.to_numeric(d.psnr_db, errors="coerce")
                         .replace([np.inf, -np.inf], np.nan).to_numpy(),
            "algo_idx": d.algorithm.map({a: i for i, a in enumerate(ALGORITHMS)}).to_numpy(),
            "quantize": q.to_numpy(),
            "shuffle": d.shuffle.to_numpy(dtype=float),
            "error_bound": d.error_bound.to_numpy(dtype=float),
            "entropy": d.entropy.to_numpy(dtype=float),
            "mad": d.mad.to_numpy(dtype=float),
            "second_deriv": d.second_derivative.to_numpy(dtype=float),
            "log_pred_ct_ms": np.nan,
            "log_pred_dt_ms": np.nan,
            "log_pred_ratio": np.nan,
        })
        # A stable order: file, then the corpus's own row order inside it.
        out["seq"] = np.arange(len(out))
        return out

    tr, va = frame(df[df.file.isin(train_files)]), frame(df[df.file.isin(val_files)])
    va = va.sort_values(["chunk", "seq"]).reset_index(drop=True)
    va["executed"] = nn_pick(va, nn)
    tr["executed"] = 0
    print(f"corpus {corpus}: {n_all} rows, {n_ok} successful, "
          f"{int(small.sum())} dropped as 16 KiB (outside the shipped weights' "
          f"input box)\n  files {len(files)} -> train {len(train_files)} / "
          f"val {len(val_files)}; rows train {len(tr)} / val {len(va)}")
    return tr, va



def align_seed_time_convention(seed, nn, mode):
    """Put the profiler seed's TIME labels in the convention the campaign measures.

    The corpus and the shipped network do not agree about what a compression
    time is, and the disagreement is not small: the corpus's
    `compression_time_ms` has log1p mean 3.098 (about 21.4 ms) while the
    network's own stored `y_means` for that head is 1.347 (about 2.85 ms), and
    the campaign measures codec kernel time at about 1.56 ms. Seeding HCompress
    from the raw column therefore starts it 6-7x out in units the campaign never
    uses, while our own model starts on-target -- so the baseline spends its
    first chunks correcting an error we handed it, and its seed-only row
    measures a convention mismatch rather than the model.

    The obvious repair -- recover codec time as time minus per-call overhead --
    is NOT available: regressing the column on size gives R^2 of 0.000 to 0.027
    per algorithm with intercepts of 20-32 ms, so the size-dependent part is
    buried in noise and subtracting the intercept yields negative times. One
    global factor per head is all this corpus supports, and it is applied here
    rather than hidden: each time column is scaled so its log1p mean matches the
    corresponding head of the network both models are being compared against.
    That removes the offset without inventing a per-codec calibration neither
    paper specifies.

    `none` keeps the raw columns, which is what earlier campaigns reported.
    """
    if mode == "none":
        print("  seed times: RAW corpus convention (per-call); not the campaign's")
        return seed
    out = seed.copy()
    for col, head in (("ct_ms", 0), ("dt_ms", 1)):
        v = pd.to_numeric(out[col], errors="coerce")
        ok = v.notna() & (v > 0)
        if not ok.any():
            continue
        # Match in log1p space, where the head's statistics are stored. The
        # factor is SOLVED for rather than taken as expm1(want)/expm1(have):
        # these labels are strongly right-skewed (log1p median 1.36 against a
        # log1p mean of 3.10), so exponentiating the means overshoots -- it put
        # the median at 0.39 ms, under both the network's 2.85 and the
        # campaign's measured 1.56.
        x = v[ok].to_numpy(dtype=float)
        want = float(nn.y_means[head])
        have = float(np.log1p(x).mean())
        lo, hi = 1e-6, 1e6
        for _ in range(200):
            mid = (lo * hi) ** 0.5
            if float(np.log1p(x * mid).mean()) < want:
                lo = mid
            else:
                hi = mid
        scale = (lo * hi) ** 0.5
        out.loc[ok, col] = x * scale
        print(f"  seed {col}: log1p mean {have:.3f} -> "
              f"{float(np.log1p(x * scale).mean()):.3f} (target {want:.3f}); "
              f"x{scale:.4g}; median {np.median(x):.3f} -> "
              f"{np.median(x) * scale:.3f} ms")
    return out

def build_workload(campaign: str, wl: str) -> pd.DataFrame:
    base = os.path.join(campaign, wl, wl)
    e = pd.read_csv(os.path.join(base, "explore.csv"))
    dist = pd.read_csv(os.path.join(base, "dist.csv"), usecols=["blob", "dist_class"])

    # Penalty rows were never measured; fig8_trace.py drops them the same way.
    penalty = (e.ct_ms <= 0) & (e.dt_ms < 0) & (e.cost >= 1e5)
    kept = e[~penalty].copy()

    q = kept.quantize.astype(int)
    out = pd.DataFrame({
        "chunk": kept.blob.to_numpy(),
        "seq": kept.seq.to_numpy(),
        "library": library_key(kept.lib_name, q, kept.shuffle).to_numpy(),
        "data_type": "float32",
        "data_format": "",
        "bytes": kept.chunk_bytes.to_numpy(dtype=float),
        "executed": (kept.role == "primary").astype(int).to_numpy(),
        "ct_ms": kept.ct_ms.where(kept.ct_ms > 0).to_numpy(dtype=float),
        "dt_ms": kept.dt_ms.where(kept.dt_ms > 0).to_numpy(dtype=float),
        "ratio": kept.ratio.where(kept.ratio > 0).to_numpy(dtype=float),
        # MEASURED quality only. `psnr_db` in the log is the ANALYTICAL value
        # derived from (range, bound) and is a different quantity; using it
        # would score every model against a formula rather than a measurement.
        "psnr_db": np.where(kept.get("quality_measured", 0) == 1,
                            pd.to_numeric(kept.get("meas_psnr_db"), errors="coerce"),
                            np.nan),
        "algo_idx": kept.algo_idx.to_numpy(dtype=float),
        "quantize": q.to_numpy(),
        "shuffle": kept.shuffle.to_numpy(dtype=float),
        "error_bound": kept.eb_encoded.to_numpy(dtype=float),
        "entropy": kept.entropy.to_numpy(dtype=float),
        "mad": kept.mad.to_numpy(dtype=float),
        "second_deriv": kept.second_deriv.to_numpy(dtype=float),
        "log_pred_ct_ms": kept.pred_ct_ms.to_numpy(dtype=float),
        "log_pred_dt_ms": kept.pred_dt_ms.to_numpy(dtype=float),
        "log_pred_ratio": kept.pred_ratio.to_numpy(dtype=float),
    })
    out = out.merge(dist.rename(columns={"blob": "chunk", "dist_class": "distribution"}),
                    on="chunk", how="left")
    out["distribution"] = out["distribution"].fillna("")
    # Chunk order = the order the run executed in, which is what feedback
    # needs. Grouped, not merely sorted by seq: the replay driver pipelines
    # chunks, so two chunks' sweep rows can interleave in the log (they do for
    # exactly one Nyx chunk). A frame where one chunk's rows are split around
    # another's has no single point at which that chunk's feedback is due.
    out["_start"] = out.groupby("chunk")["seq"].transform("min")
    out = (out.sort_values(["_start", "chunk", "seq"], kind="stable")
              .drop(columns="_start").reset_index(drop=True))
    miss = int((out.distribution == "").sum())
    psnr_n = int(out.psnr_db.notna().sum())
    print(f"{wl}: {out.chunk.nunique()} chunks, {len(out)} rows, "
          f"{out.library.nunique()} configurations; measured dt {int(out.dt_ms.notna().sum())}, "
          f"measured PSNR {psnr_n}"
          + (f"; {miss} rows WITHOUT a distribution class" if miss else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workload", action="append", default=None)
    ap.add_argument("--seed-time-convention", default="nn",
                    choices=["nn", "none"],
                    help="nn: rescale the seed's time labels to the convention "
                         "the shipped network encodes, which is the one the "
                         "campaign measures. none: the raw per-call columns.")
    ap.add_argument("--nnwt", default="/u/imuradli/clio-core/context-transport-primitives/"
                                     "src/compress/model/weights/model.nnwt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nn = NeuroPressNN(a.nnwt)

    train, val = build_synthetic(a.corpus, nn)
    # HCompress's seed: the offline profiler's rows, which here ARE the rows our
    # own model was trained on -- the same corpus, the same split, same side.
    cols = ["data_type", "data_format", "library", "distribution", "bytes",
            "ct_ms", "dt_ms", "ratio"]
    seed = align_seed_time_convention(train[cols], nn, a.seed_time_convention)
    seed.to_csv(os.path.join(a.out, "seed.csv"), index=False, float_format="%.9g")
    print(f"  seed.csv: {len(seed)} profiler row(s)")

    # A SECOND seed, restricted to the profiler's largest buffers.
    #
    # HCompress regresses a SPEED and has no size input, which is sound only if
    # speed is size-independent. In this corpus it is not: the time labels are
    # dominated by a fixed per-call cost (256x more bytes moves the label 1.6x),
    # so speed scales almost linearly with size and roughly half the speed
    # variance is size alone. Seeding across five sizes therefore hands the
    # baseline a target it cannot represent. An offline profiler would instead
    # benchmark buffers like the ones the deployment compresses -- 8 MiB here --
    # so the largest profiled size is the closest available and the most
    # favourable honest choice. Both seeds are reported.
    largest = train[train.bytes >= train.bytes.max()][cols]
    largest.to_csv(os.path.join(a.out, "seed_largest.csv"), index=False,
                   float_format="%.9g")
    print(f"  seed_largest.csv: {len(largest)} row(s) at "
          f"{int(train.bytes.max())} bytes (the profiler's largest buffer)")

    settings = {"synthetic": val}
    for wl in (a.workload or WORKLOADS):
        try:
            settings[wl] = build_workload(a.campaign, wl)
        except FileNotFoundError as exc:
            print(f"{wl}: SKIPPED -- {exc}", file=sys.stderr)

    for name, rows in settings.items():
        # Every consumer of these two files joins them positionally, so the
        # invariant they rely on is checked here, once, rather than trusted.
        runs = int((rows.chunk != rows.chunk.shift()).sum())
        if runs != rows.chunk.nunique():
            raise SystemExit(f"{name}: chunk rows are not contiguous "
                             f"({runs} runs for {rows.chunk.nunique()} chunks)")
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        rows.reindex(columns=ROW_COLUMNS).to_csv(
            os.path.join(d, "rows.csv"), index=False, float_format="%.9g")
        rows.reindex(columns=EVAL_COLUMNS).to_csv(
            os.path.join(d, "eval.csv"), index=False, float_format="%.9g")
    print(f"\nwrote {len(settings)} setting(s) to {a.out}: {', '.join(settings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

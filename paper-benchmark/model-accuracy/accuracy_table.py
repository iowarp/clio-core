#!/usr/bin/env python3
"""The prediction-accuracy table: MAPE and R2 per metric, per setting, per model.

    accuracy_table.py --inputs DIR --out DIR [--no-policy-clamp]

Reads each setting's rows.csv (measured values, model inputs and the run's own
logged NN predictions) and hcompress.csv (the baseline's predictions), runs the
two shipped models offline, and emits:

    accuracy_long.csv    one row per (setting, model, metric)
    accuracy_table.tex   a tabular, one block per setting
    accuracy_table.md    the same thing, for reading in a terminal

FOUR MODELS, AND WHY THERE IS A FIFTH ROW

  NeuroPress NN            the shipped .nnwt weights, no online update. The
                           static counterpart to HCompress's seed-only row, and
                           the only NN state that can report all four metrics
                           from one model.
  XGBoost                  upstream's trained checkpoint, also static.
  HCompress CCP (seed)     fitted on the profiler rows = the corpus our model
                           was trained on.
  HCompress CCP (+fb)      the same, updated from measured outcomes during the
                           run.
  NeuroPress NN + online   the predictions the RUN ITSELF made, taken from the
    learning (as deployed)  exploration log, with online SGD active. This is
                           the adaptive counterpart to (+fb). It has no PSNR
                           because the log does not carry the network's PSNR
                           head -- its psnr_db column is the analytical value
                           derived from (range, bound), a different quantity.

EVERY MODEL IS CLAMPED THE SAME WAY. The deployed policy floors a predicted
time at 1 ms and caps a predicted ratio at 100x, and the NN's logged
predictions already carry those clamps from the kernel. Applying them to one
model and not another would make the table a comparison of policies rather than
of models, so they are applied to all. --no-policy-clamp reports the same table
without them; the floor in particular is not cosmetic, because most measured
compression times here are UNDER 1 ms.

MAPE is the mean over rows where the metric was actually MEASURED. Nothing is
imputed: a metric a model does not predict is reported as n/a, and a metric
nobody measured is reported as a row count of 0.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_offline import (NeuroPressNN, XGBoostBaseline, nn_inputs,  # noqa: E402
                            PRED_TIME_FLOOR_MS, RATIO_CAP)

#: The reported settings are the five REAL workloads.
#:
#: The synthetic corpus is not one of them. It is the data every static model
#: was fitted on -- our NN's and XGBoost's training set, and HCompress's
#: profiler seed -- so a block scored on its held-out split measures how well
#: each model reproduces its own training distribution, which is a different
#: question from the one this table is for. prepare_inputs.py still writes
#: inputs/synthetic/ so the split is on the record and can be scored on demand
#: by adding it back here.
SETTINGS = [("vpic", "VPIC"), ("nyx", "Nyx"), ("lammps", "LAMMPS"),
            ("warpx", "WarpX"), ("ai", "AI")]
#: (label, column suffix) in the order the table's rows must appear.
MODELS = ["NeuroPress NN", "XGBoost", "HCompress CCP (seed only)",
          "HCompress CCP (+ feedback)",
          "NeuroPress NN + online learning (as deployed)"]
#: key, the rows.csv column holding the MEASURED value, label. "cost" has no
#: column: it is computed from the three measured components by cost() below.
METRICS = [("comp_time", "ct_ms", "Comp time"),
           ("decomp_time", "dt_ms", "Decomp time"),
           ("ratio", "ratio", "Ratio"),
           ("cost", "", "Cost")]

#: PSNR is not reported. The cost model is what selection actually minimises,
#: so it is the metric that decides whether a predictor is good enough to
#: choose with; and HCompress has no quality output to compare against anyway.
#: The PSNR columns remain computable -- rows.csv still carries the measured
#: values and models_offline.py still returns the heads -- by putting
#: ("psnr", "psnr_db", "PSNR") back in METRICS.


def mape(pred: np.ndarray, act: np.ndarray, cap: float = 0.0) -> float:
    """Mean absolute percentage error, optionally with a PER-ROW ceiling.

    `cap` clips each row's APE before the mean, which is the repo's own
    convention for this statistic (plot_fig8.py's per-chunk MAPE_CLIP) and not
    the same thing as clipping the reported mean. Clipping the mean would put
    every model above the ceiling on the same number and destroy the comparison
    exactly where the models differ most; clipping per row keeps a model whose
    errors are all 30% distinguishable from one whose errors are half 10% and
    half 2000%. The uncapped value is always kept beside it in the CSV.
    """
    ape = np.abs(pred - act) / np.abs(act) * 100.0
    return float(np.mean(np.minimum(ape, cap) if cap > 0 else ape))


def smape(pred: np.ndarray, act: np.ndarray) -> float:
    """Symmetric MAPE: 100 * mean(|p-a| / (|p|+|a|)), bounded to [0, 100].

    The bound is INTRINSIC, not a ceiling imposed afterwards, and that is the
    point. A per-row cap at 100 puts every model whose errors exceed 1x on the
    same number -- XGBoost and HCompress seed-only both land on exactly 100 in
    the cost panel -- which hides how differently they behave. sMAPE keeps
    separating them all the way up:

        p = a      ->   0        p = 10a  ->  81.8
        p = 2a     ->  33.3      p = 100a ->  98.0
        p = 5a     ->  66.7      p -> inf -> 100

    It is a standard forecasting metric, so it needs no defending in the paper,
    and it is monotone in the error, so the ORDER of the models is the same as
    under uncapped MAPE. Raw MAPE, capped and uncapped, stays in the CSV.
    """
    den = np.abs(pred) + np.abs(act)
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(pred[ok] - act[ok]) / den[ok]) * 100.0)


def medape(pred: np.ndarray, act: np.ndarray) -> float:
    return float(np.median(np.abs(pred - act) / np.abs(act)) * 100.0)


def r2(pred: np.ndarray, act: np.ndarray) -> float:
    ss_tot = float(np.sum((act - act.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - np.sum((act - pred) ** 2) / ss_tot)


#: The deployed cost model (NeuroPressCost / RankingWeights defaults): the
#: quantity selection actually minimises.
BW_BYTES_PER_MS = 5e6


def cost(ct, dt, ratio, nbytes):
    """max(floor,ct) + max(floor,dt) + bytes/(min(ratio,cap)*bw), elementwise."""
    return (np.maximum(PRED_TIME_FLOOR_MS, ct) + np.maximum(PRED_TIME_FLOOR_MS, dt)
            + nbytes / (np.minimum(RATIO_CAP, np.maximum(ratio, 0.1)) * BW_BYTES_PER_MS))


def correctness(rows: pd.DataFrame, preds: dict) -> list:
    """Does the model PICK the configuration that measurement says is cheapest?

    MAPE answers "how wrong is each number"; this answers the question a
    selector actually asks. Both are reported because they can disagree
    sharply: a model can be badly calibrated in absolute terms and still rank
    correctly, which is why a cost MAPE alone cannot settle whether a predictor
    is good enough to choose with.

    Scored only on candidates whose compression time, decompression time AND
    ratio were all measured, so the measured cost is a measurement throughout.
    A tie in measured cost counts as correct: several configurations really can
    be equally cheap once the ratio saturates the cap, and calling the second
    of two identical costs an error would penalise the tie-break rather than
    the model.
    """
    act_ct = rows.ct_ms.to_numpy(dtype=float)
    act_dt = rows.dt_ms.to_numpy(dtype=float)
    act_ratio = rows.ratio.to_numpy(dtype=float)
    nbytes = rows.bytes.to_numpy(dtype=float)
    ok = (np.isfinite(act_ct) & (act_ct > 0) & np.isfinite(act_dt) & (act_dt > 0)
          & np.isfinite(act_ratio) & (act_ratio > 0))
    measured = cost(act_ct, act_dt, act_ratio, nbytes)

    chunk = rows.chunk.to_numpy()
    starts = np.flatnonzero(np.r_[True, chunk[1:] != chunk[:-1]])
    bounds = np.r_[starts, len(chunk)]

    out = []
    for model, p in preds.items():
        pc = cost(p["ct_ms"], p["dt_ms"], p["ratio"], nbytes)
        top1 = top3 = n = 0
        regret = []
        for i in range(len(bounds) - 1):
            sl = slice(bounds[i], bounds[i + 1])
            m = ok[sl] & np.isfinite(pc[sl])
            if m.sum() < 2:
                continue
            meas = measured[sl][m]
            pred = pc[sl][m]
            best = meas.min()
            pick = int(np.argmin(pred))            # ties: first, as a stable sort
            n += 1
            if meas[pick] <= best * (1 + 1e-9):
                top1 += 1
            k = min(3, len(pred))
            if np.isin(np.argsort(pred, kind="stable")[:k], np.flatnonzero(meas <= best * (1 + 1e-9))).any():
                top3 += 1
            regret.append(meas[pick] / best - 1.0)
        if n == 0:
            continue
        r = np.asarray(regret) * 100.0
        out.append({"model": model, "chunks": n, "top1_pct": 100.0 * top1 / n,
                    "top3_pct": 100.0 * top3 / n, "mean_regret_pct": float(r.mean()),
                    "median_regret_pct": float(np.median(r))})
    return out


def predictions(rows: pd.DataFrame, hc: pd.DataFrame, nn: NeuroPressNN,
                xgb: XGBoostBaseline, clamp: bool) -> dict:
    """Every model's predictions for every row, aligned to `rows`."""
    floor = PRED_TIME_FLOOR_MS if clamp else 0.0
    cap = RATIO_CAP if clamp else 1e5
    out = {}

    X = nn_inputs(rows.algo_idx, rows.quantize, rows.shuffle, rows.error_bound,
                  rows.bytes, rows.entropy, rows.mad, rows.second_deriv)
    p = nn.predict(X, time_floor=floor, ratio_cap=cap)
    out["NeuroPress NN"] = {"ct_ms": p.pred_ct_ms.to_numpy(),
                            "dt_ms": p.pred_dt_ms.to_numpy(),
                            "ratio": p.pred_ratio.to_numpy(),
                            "psnr_db": p.pred_psnr_db.to_numpy()}

    Xn = xgb.encode(rows.algo_idx, rows.quantize, rows.shuffle, rows.error_bound,
                    rows.bytes, rows.entropy, rows.mad, rows.second_deriv)
    q = xgb.predict(Xn, time_floor=floor, ratio_cap=cap)
    out["XGBoost"] = {"ct_ms": q.pred_ct_ms.to_numpy(),
                      "dt_ms": q.pred_dt_ms.to_numpy(),
                      "ratio": q.pred_ratio.to_numpy(),
                      "psnr_db": q.pred_psnr_db.to_numpy()}

    # hcompress.csv is written row-for-row from the same eval.csv, so it aligns
    # positionally; the assertion keeps a silent misalignment from becoming a
    # published number. `library` is checked as well as `chunk`, because the 32
    # rows of one chunk all carry the same chunk id: a permutation WITHIN a
    # chunk -- which is exactly what hcompress_ccp_eval.cc's regroup-by-chunk
    # would produce from an out-of-order eval.csv -- passes a chunk-only check
    # while attributing every prediction to the wrong configuration.
    if len(hc) != len(rows):
        raise SystemExit(f"hcompress.csv has {len(hc)} rows, rows.csv has {len(rows)}")
    if not (hc.chunk.to_numpy() == rows.chunk.to_numpy()).all():
        raise SystemExit("hcompress.csv is not aligned with rows.csv (chunk order)")
    if "library" in hc and not (hc.library.to_numpy() == rows.library.to_numpy()).all():
        raise SystemExit("hcompress.csv is not aligned with rows.csv (library order "
                         "within a chunk)")
    for label, suffix in [("HCompress CCP (seed only)", "seed"),
                          ("HCompress CCP (+ feedback)", "fb")]:
        out[label] = {
            "ct_ms": np.maximum(floor, hc[f"pred_ct_ms_{suffix}"].to_numpy()),
            "dt_ms": np.maximum(floor, hc[f"pred_dt_ms_{suffix}"].to_numpy()),
            "ratio": np.minimum(cap, hc[f"pred_ratio_{suffix}"].to_numpy()),
            "psnr_db": np.full(len(rows), np.nan),
        }

    # The cost model applies its own floor and cap to BOTH sides, so a cost is
    # comparable even where the components are clamped differently.
    nbytes = rows.bytes.to_numpy(dtype=float)
    for m in out:
        out[m]["cost"] = cost(out[m]["ct_ms"], out[m]["dt_ms"],
                              out[m]["ratio"], nbytes)

    # The run's own predictions. Already clamped by the kernel that made them,
    # so --no-policy-clamp cannot un-clamp them; that is stated in the report
    # rather than worked around.
    out["NeuroPress NN + online learning (as deployed)"] = {
        "ct_ms": rows.log_pred_ct_ms.to_numpy(dtype=float),
        "dt_ms": rows.log_pred_dt_ms.to_numpy(dtype=float),
        "ratio": rows.log_pred_ratio.to_numpy(dtype=float),
        "psnr_db": np.full(len(rows), np.nan),
    }
    onl = out["NeuroPress NN + online learning (as deployed)"]
    onl["cost"] = cost(onl["ct_ms"], onl["dt_ms"], onl["ratio"], nbytes)
    return out


# ---------------------------------------------------------------------------
# Default model locations, resolved so this runs on a machine that is not the
# one it was written on.
#
#   model.nnwt     ships IN this repo, so it is found relative to THIS file.
#   xgb_model.pkl  does NOT: it belongs to the upstream NeuroPress checkout.
#                  Set NEUROPRESS_DIR (default ~/NeuroPress) or pass --xgb.
#
# Both defaults used to be absolute paths under one author's home directory,
# which breaks silently on any other machine -- argparse happily accepts the
# path and the load fails later, far from the cause.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NNWT = os.path.normpath(os.path.join(
    _HERE, "..", "..", "context-transport-primitives",
    "src", "compress", "model", "weights", "model.nnwt"))
DEFAULT_XGB = os.path.join(
    os.environ.get("NEUROPRESS_DIR", os.path.expanduser("~/NeuroPress")),
    "neural_net", "weights", "xgb_model.pkl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-policy-clamp", action="store_true")
    ap.add_argument("--mape-cap", type=float, default=0.0,
                    help="per-row APE ceiling, in %% (0 = none). 100 bounds "
                         "every MAPE at 100%% while keeping models below the "
                         "ceiling distinguishable.")
    ap.add_argument("--nnwt", default=DEFAULT_NNWT)
    ap.add_argument("--xgb", default=DEFAULT_XGB)
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--hc-suffix", default="",
                    help="read hcompress<suffix>.csv, so seeding variants can "
                         "coexist instead of overwriting each other")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clamp = not a.no_policy_clamp
    nn, xgb = NeuroPressNN(a.nnwt), XGBoostBaseline(a.xgb)

    records, coverage, correct = [], [], []
    for key, label in SETTINGS:
        d = os.path.join(a.inputs, key)
        rp = os.path.join(d, "rows.csv")
        hp = os.path.join(d, f"hcompress{a.hc_suffix}.csv")
        if not (os.path.exists(rp) and os.path.exists(hp)):
            print(f"{key}: SKIPPED (missing {os.path.basename(rp)} or "
                  f"{os.path.basename(hp)})", file=sys.stderr)
            continue
        rows = pd.read_csv(rp)
        hc = pd.read_csv(hp)
        preds = predictions(rows, hc, nn, xgb, clamp)

        n_cost = int((np.isfinite(rows.ct_ms) & (rows.ct_ms > 0)
                      & np.isfinite(rows.dt_ms) & (rows.dt_ms > 0)
                      & np.isfinite(rows.ratio) & (rows.ratio > 0)).sum())
        cov = {"setting": label, "chunks": int(rows.chunk.nunique()),
               "n_cost": n_cost,
               "rows": len(rows), "configs": int(rows.library.nunique()),
               "unseen_distribution_rows": int(hc.unseen_distribution.sum())
               if "unseen_distribution" in hc else 0}
        for metric, col, _ in METRICS:
            if metric == "cost":
                continue   # filled in below, from the three-component mask
            cov[f"n_{metric}"] = int((rows[col].notna() & (rows[col] > 0)).sum())
        coverage.append(cov)

        for rec in correctness(rows, preds):
            rec["setting"] = label
            correct.append(rec)

        # Measured cost: only where all three components were measured, so it
        # is a measurement throughout rather than a mix of measured and floored.
        mct = rows.ct_ms.to_numpy(dtype=float)
        mdt = rows.dt_ms.to_numpy(dtype=float)
        mratio = rows.ratio.to_numpy(dtype=float)
        nbytes = rows.bytes.to_numpy(dtype=float)
        all3 = (np.isfinite(mct) & (mct > 0) & np.isfinite(mdt) & (mdt > 0)
                & np.isfinite(mratio) & (mratio > 0))
        measured_cost = np.where(all3, cost(mct, mdt, mratio, nbytes), np.nan)

        for model in MODELS:
            for metric, col, _ in METRICS:
                if metric == "cost":
                    act = measured_cost
                    pr = preds[model]["cost"]
                else:
                    act = rows[col].to_numpy(dtype=float)
                    pr = preds[model][col]
                m = np.isfinite(act) & (act > 0) & np.isfinite(pr)
                rec = {"setting": label, "model": model, "metric": metric,
                       "n": int(m.sum())}
                if m.sum() == 0:
                    rec.update(mape=np.nan, mape_uncapped=np.nan,
                               smape=np.nan, median_ape=np.nan, r2=np.nan,
                               note="not measured in this setting")
                else:
                    rec.update(mape=mape(pr[m], act[m], a.mape_cap),
                               mape_uncapped=mape(pr[m], act[m]),
                               smape=smape(pr[m], act[m]),
                               median_ape=medape(pr[m], act[m]),
                               r2=r2(pr[m], act[m]), note="")
                records.append(rec)

    if not records:
        print("nothing to report", file=sys.stderr)
        return 1
    long = pd.DataFrame(records)
    long["mape_cap"] = a.mape_cap
    cov = pd.DataFrame(coverage)
    tag = a.tag or ("" if clamp else "_unclamped")
    long.to_csv(os.path.join(a.out, f"accuracy_long{tag}.csv"), index=False,
                float_format="%.4f")
    cov.to_csv(os.path.join(a.out, f"coverage{tag}.csv"), index=False)
    corr = pd.DataFrame(correct)
    if not corr.empty:
        corr = corr[["setting", "model", "chunks", "top1_pct", "top3_pct",
                     "mean_regret_pct", "median_regret_pct"]]
        corr.to_csv(os.path.join(a.out, f"correctness{tag}.csv"), index=False,
                    float_format="%.4f")

    # ---- rendering -------------------------------------------------------
    def cell(setting, model, metric, field):
        field = "smape" if field == "mape" else field
        r = long[(long.setting == setting) & (long.model == model)
                 & (long.metric == metric)]
        if r.empty:
            return "--"
        v = r.iloc[0][field]
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.1f}" if field != "r2" else f"{v:.3f}"

    order = [lbl for _, lbl in SETTINGS if lbl in set(long.setting)]
    head = ["Model", "Comp time sMAPE", "Decomp time sMAPE", "Ratio sMAPE",
            "Cost sMAPE", "R2 comp", "R2 decomp", "R2 ratio", "R2 cost"]
    capnote = ("sMAPE = 100*mean(|p-a|/(|p|+|a|)), bounded to [0,100] by "
               "construction; raw MAPE"
               + (f" (per-row cap {a.mape_cap:g}%)" if a.mape_cap > 0 else "")
               + " and the uncapped value are columns in the CSV")
    md = [f"# Prediction accuracy ({'policy clamps applied' if clamp else 'no policy clamps'}"
          f"; {capnote})", ""]
    for s in order:
        c = cov[cov.setting == s].iloc[0]
        md += [f"## {s} — {c.chunks} chunks, {c.rows} rows, {c.configs} configurations",
               "", "| " + " | ".join(head) + " |",
               "|" + "|".join(["---"] * len(head)) + "|"]
        for m in MODELS:
            md.append("| " + " | ".join([
                m,
                cell(s, m, "comp_time", "mape"), cell(s, m, "decomp_time", "mape"),
                cell(s, m, "ratio", "mape"), cell(s, m, "cost", "mape"),
                cell(s, m, "comp_time", "r2"), cell(s, m, "decomp_time", "r2"),
                cell(s, m, "ratio", "r2"), cell(s, m, "cost", "r2")]) + " |")
        md.append("")

    if not corr.empty:
        md += ["## Prediction correctness — does the model pick the configuration "
               "measurement says is cheapest?", "",
               "| Setting | Model | Chunks | Top-1 % | Top-3 % | Mean regret % | Median regret % |",
               "|---|---|---|---|---|---|---|"]
        for s_ in order:
            for m in MODELS:
                r = corr[(corr.setting == s_) & (corr.model == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                md.append(f"| {s_} | {m} | {int(r.chunks)} | {r.top1_pct:.1f} | "
                          f"{r.top3_pct:.1f} | {r.mean_regret_pct:.1f} | "
                          f"{r.median_regret_pct:.2f} |")
        md.append("")
    open(os.path.join(a.out, f"accuracy_table{tag}.md"), "w").write("\n".join(md))

    tex = [r"% Prediction accuracy. MAPE in %, over rows where the metric was measured.",
           f"% {capnote}. The uncapped value is in accuracy_long{tag}.csv.",
           r"% Cost is the deployed cost model: max(1ms,ct)+max(1ms,dt)+bytes/(min(ratio,100)*5e6).",
           r"% Requires booktabs.",
           r"\begin{tabular}{l rrrr rrrr}", r"\toprule",
           r"Model & \multicolumn{4}{c}{sMAPE (\%)} & \multicolumn{4}{c}{$R^2$} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r" & Comp.\ time & Decomp.\ time & Ratio & Cost & Comp.\ & Decomp.\ & Ratio & Cost \\"]
    for s in order:
        c = cov[cov.setting == s].iloc[0]
        tex += [r"\midrule",
                rf"\multicolumn{{9}}{{l}}{{\emph{{{s}}} — {c.chunks} chunks, "
                rf"{c.rows} rows}} \\"]
        for m in MODELS:
            name = m.replace("+", r"$+$").replace("_", r"\_")
            tex.append(" & ".join([
                name,
                cell(s, m, "comp_time", "mape"), cell(s, m, "decomp_time", "mape"),
                cell(s, m, "ratio", "mape"), cell(s, m, "cost", "mape"),
                cell(s, m, "comp_time", "r2"), cell(s, m, "decomp_time", "r2"),
                cell(s, m, "ratio", "r2"), cell(s, m, "cost", "r2")]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, f"accuracy_table{tag}.tex"), "w").write("\n".join(tex) + "\n")

    print("\n".join(md))
    print("coverage per setting (rows with a MEASURED value for each metric):")
    print(cov.to_string(index=False))
    print(f"\nwrote accuracy_long{tag}.csv, accuracy_table{tag}.tex, "
          f"accuracy_table{tag}.md, coverage{tag}.csv, correctness{tag}.csv to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

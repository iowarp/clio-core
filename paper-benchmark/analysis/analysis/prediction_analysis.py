#!/usr/bin/env python3
"""Predicted vs measured, and what the predictor systematically misunderstands.

THREE PAIRS, and only three, each with its own caveat drawn from the writer:

  pred_ratio  vs ratio          -- both well defined on every row.
  pred_ct_ms  vs ct_ms          -- ct_ms is CUDA-event kernel time. Explored
                                   candidates are measured while up to
                                   CLIO_NEUROPRESS_EXPLORE_STREAMS of them run
                                   concurrently; the primary runs alone. Some
                                   of the "prediction error" on alt rows is
                                   device contention, so alt and primary rows
                                   are reported separately and never pooled.
  pred_dt_ms  vs dt_ms          -- dt_ms is -1 unless the sweep measured it.
                                   pred_dt_ms is always real (the dt head runs
                                   for every ranked action), so the pair is
                                   available exactly when dt was measured.

ERROR IN LOG SPACE. Ratio spans three orders of magnitude in a single sweep,
so a mean absolute error is dominated by whichever chunk compressed 1000x and
says nothing about the other forty. The headline figure is therefore the
log-ratio error log10(pred/actual), whose zero is "exactly right", whose sign
is the direction of the bias, and whose magnitude is scale free. MAE and RMSE
in native units are reported beside it because they are what a cost model
actually pays.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .statistics import _corr, MIN_SUPPORT

PAIRS = [("ratio", "pred_ratio", "ratio"),
         ("ct", "pred_ct_ms", "ct_ms"),
         ("dt", "pred_dt_ms", "dt_ms_measured")]


def prediction_errors(rows: pd.DataFrame, metric: str, pred: str,
                      actual: str) -> pd.DataFrame:
    """Per-row error table for one predicted metric."""
    if pred not in rows or actual not in rows:
        return pd.DataFrame()
    keep = [c for c in ("chunk_uid", "blob", "workload", "field", "timestep",
                        "role", "rank", "lib_name", "preset", "quantize",
                        "shuffle", "error_bound", "chunk_bytes", "entropy",
                        "mad", "second_deriv", "adopted") if c in rows]
    d = rows[keep + [pred, actual]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[pred, actual])
    if d.empty:
        return d
    d["metric"] = metric
    d["predicted"] = d[pred]
    d["actual"] = d[actual]
    d["abs_error"] = d["predicted"] - d["actual"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["rel_error"] = np.where(np.abs(d["actual"]) > 0,
                                  d["abs_error"] / d["actual"], np.nan)
        # Scale-free, symmetric in over/under, and the only version that can
        # be averaged across a three-decade range without one chunk owning it.
        pos = (d["predicted"] > 0) & (d["actual"] > 0)
        d["log10_ratio_error"] = np.where(
            pos, np.log10(d["predicted"].where(pos, 1.0)
                          / d["actual"].where(pos, 1.0)), np.nan)
    d["over_predicted"] = d["abs_error"] > 0
    # Upstream's MAPE, clamped exactly as gpucompress does before comparing:
    # ratio capped at 100, times floored at 1 ms. Reported so the figure is
    # comparable with the runtime's own cost_model_error_pct.
    if metric == "ratio":
        p, a = np.minimum(d["predicted"], 100.0), np.minimum(d["actual"], 100.0)
    else:
        p, a = np.maximum(d["predicted"], 1.0), np.maximum(d["actual"], 1.0)
    d["clamped_ape_pct"] = 100.0 * np.abs(p - a) / np.abs(a)
    return d


def detect_clamp(rows: pd.DataFrame, pred: str,
                 actual: str) -> Dict[str, object]:
    """Is this prediction column FLOORED or CAPPED, and does it matter?

    The NN's time heads never emit below 1 ms and its ratio head never emits
    above 100 -- the same clamps the cost model applies (`max(1, ct)`,
    `min(cap, ratio)`). When the actuals cross those limits, the "prediction
    error" on those rows is the clamp, not the model: a codec whose real
    compression takes 0.11 ms cannot be predicted better than 9x wrong by a
    head that bottoms out at 1 ms.

    So the clamp is detected from the data (many rows sharing an exact
    extremum) and the fold errors are reported twice -- over everything, and
    over only the rows where the clamp cannot bite.
    """
    out: Dict[str, object] = {"metric": pred}
    if pred not in rows or actual not in rows:
        return out
    p = pd.to_numeric(rows[pred], errors="coerce")
    a = pd.to_numeric(rows[actual], errors="coerce")
    ok = p.notna() & a.notna()
    p, a = p[ok], a[ok]
    if p.empty:
        return out
    lo, hi = float(p.min()), float(p.max())
    n_lo, n_hi = int((p == lo).sum()), int((p == hi).sum())
    # A clamp shows up as many rows at an exact, round extremum. One row at
    # the minimum is just the smallest prediction.
    out["floor"] = lo if n_lo >= max(5, 0.02 * len(p)) else None
    out["cap"] = hi if n_hi >= max(5, 0.02 * len(p)) else None
    out["n_at_floor"], out["n_at_cap"] = n_lo, n_hi
    unaffected = pd.Series(True, index=p.index)
    if out["floor"] is not None:
        out["pct_actual_below_floor"] = float(100.0 * (a < lo).mean())
        unaffected &= a >= lo
    if out["cap"] is not None:
        out["pct_actual_above_cap"] = float(100.0 * (a > hi).mean())
        unaffected &= a <= hi
    out["n_rows"] = int(len(p))
    out["n_rows_clamp_cannot_bite"] = int(unaffected.sum())
    both = (p > 0) & (a > 0)
    if both.any():
        lg = np.log10(p[both] / a[both])
        out["median_fold_error_all"] = float(10 ** lg.abs().median())
        out["pct_over_predicted_all"] = float(100.0 * (lg > 0).mean())
    m = both & unaffected
    if m.sum() >= 10:
        lg2 = np.log10(p[m] / a[m])
        out["median_fold_error_unclamped"] = float(10 ** lg2.abs().median())
        out["pct_over_predicted_unclamped"] = float(100.0 * (lg2 > 0).mean())
    return out


def summarize_errors(err: pd.DataFrame,
                     by: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if err.empty:
        return pd.DataFrame()
    keys = ["metric"] + list(by or [])
    keys = [k for k in keys if k in err.columns]
    recs: List[dict] = []
    for key, g in err.groupby(keys, dropna=False):
        keys_t = key if isinstance(key, tuple) else (key,)
        rec: Dict[str, object] = dict(zip(keys, keys_t))
        lg = g["log10_ratio_error"].replace([np.inf, -np.inf], np.nan).dropna()
        rec.update({
            "n": int(len(g)), "n_chunks": int(g["chunk_uid"].nunique()),
            "bias_mean_abs_error": float(g["abs_error"].mean()),
            "mae": float(g["abs_error"].abs().mean()),
            "rmse": float(np.sqrt(np.mean(g["abs_error"] ** 2))),
            "median_rel_error": float(g["rel_error"].median()),
            "mape_pct": float(100.0 * g["rel_error"].abs().median()),
            "clamped_mape_pct": float(g["clamped_ape_pct"].mean()),
            "pct_over_predicted": float(100.0 * g["over_predicted"].mean()),
        })
        if lg.size:
            rec.update({
                "log10_bias_median": float(lg.median()),
                "log10_bias_mean": float(lg.mean()),
                "median_fold_error": float(10 ** lg.abs().median()),
                "p90_fold_error": float(10 ** lg.abs().quantile(0.90)),
                "max_fold_error": float(10 ** lg.abs().max()),
            })
        c = _corr(g["predicted"].to_numpy(dtype=float),
                  g["actual"].to_numpy(dtype=float))
        rec["pred_actual_spearman"] = c["spearman_rho"]
        rec["pred_actual_pearson"] = c["pearson_r"]
        recs.append(rec)
    return pd.DataFrame(recs)


def error_drivers(err: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Does the prediction error depend on the data, or only on the codec?

    Correlates the SIGNED log error against each intrinsic feature, inside a
    fixed configuration so the chunk is the unit. A feature that correlates
    with the error is a feature the predictor has not learned to use.
    """
    if err.empty:
        return pd.DataFrame()
    recs: List[dict] = []
    keys = [k for k in ("metric", "lib_name", "quantize", "shuffle")
            if k in err.columns]
    for key, g in err.groupby(keys, dropna=False):
        keys_t = key if isinstance(key, tuple) else (key,)
        g1 = g.groupby("chunk_uid", as_index=False).first()
        if len(g1) < MIN_SUPPORT:
            continue
        for f in features:
            if f not in g1:
                continue
            c = _corr(g1[f].to_numpy(dtype=float),
                      g1["log10_ratio_error"].to_numpy(dtype=float))
            rec = dict(zip(keys, keys_t))
            rec.update({"feature": f, "n_chunks": c["n"],
                        "spearman_rho": c["spearman_rho"],
                        "pearson_r": c["pearson_r"], "p": c["pearson_p"]})
            recs.append(rec)
    return pd.DataFrame(recs)


def ranking_quality(rows: pd.DataFrame) -> pd.DataFrame:
    """Does the predictor ORDER the candidates correctly, per chunk?

    The cost model consumes an ordering, not a magnitude, so a predictor that
    is 3x off on every candidate but ranks them perfectly costs nothing, and
    one that is 5% off but inverts the top two costs a chunk. Reported as
    Spearman between predicted and measured ratio WITHIN each chunk, plus the
    measured rank of the candidate the model ranked first.
    """
    recs: List[dict] = []
    for uid, g in rows.groupby("chunk_uid"):
        g = g[np.isfinite(g["pred_ratio"]) & np.isfinite(g["ratio"])]
        if len(g) < 4:
            continue
        c = _corr(g["pred_ratio"].to_numpy(dtype=float),
                  g["ratio"].to_numpy(dtype=float))
        alt = g[g["role"] == "alt"].sort_values("rank")
        rec: Dict[str, object] = {
            "chunk_uid": uid, "n_candidates": int(len(g)),
            "pred_vs_actual_spearman": c["spearman_rho"]}
        if len(alt):
            top = alt.iloc[0]
            better = int((alt["ratio"] > top["ratio"]).sum())
            rec.update({
                "top_ranked_lib": top["lib_name"],
                "top_ranked_ratio": float(top["ratio"]),
                "best_available_ratio": float(alt["ratio"].max()),
                "n_candidates_beating_top_rank": better,
                "top_rank_ratio_shortfall": float(
                    (alt["ratio"].max() - top["ratio"]) / alt["ratio"].max()),
            })
        recs.append(rec)
    return pd.DataFrame(recs)

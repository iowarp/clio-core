#!/usr/bin/env python3
"""The cost model, the per-chunk winners, and how far the adopted choice sits
from the best one available.

WHY THE COST MODEL IS INFERRED AND NOT ASSUMED. neuropress_cost.h defines

    cost = w_ct*max(1,ct) + w_dt*max(1,dt) + w_io*bytes/(min(cap,ratio)*bw)

and every one of w_ct, w_dt, w_io, bw and cap is overridable at runtime
(CLIO_NEUROPRESS_COST_W_CT / _DT / _IO / _BW, CLIO_NEUROPRESS_RATIO_CAP). The
log does not record them. Assuming the defaults on a run that overrode them
would make every regret figure wrong in a way nothing else in the output would
contradict -- so this module recovers them by least squares from the `cost`
column itself and reports the residual, which is the evidence that the
recovery is right.

WHICH `dt` THE COST USED. Not the row's own. The sweep passes each candidate
its MEASURED dt when CLIO_NEUROPRESS_EXPLORE_MEASURE_DT is on, and otherwise
the PRIMARY's PREDICTED dt held constant across candidates (upstream's
behaviour, so dt cancels out of the ranking entirely). Both are tried below
and the better fit wins, which also tells the report which regime the run was
in.

THE RATIO CAP IS A TIE GENERATOR. With cap = 100, every candidate whose ratio
exceeds 100 gets the same io term. On highly compressible data that is most of
the sweep, and the ranking then turns on ct and dt alone -- or, if w_ct and
w_dt are zero, on nothing at all. `ties_at_cap` below counts it, because a
"wrong" selection among tied candidates is not a model failure.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Recovering the weights
# --------------------------------------------------------------------------

def _design(df: pd.DataFrame, cap: float, dt: np.ndarray) -> np.ndarray:
    ct = np.maximum(1.0, df["ct_ms"].to_numpy(dtype=float))
    dtc = np.maximum(1.0, dt)
    ratio = df["ratio"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        io = df["chunk_bytes"].to_numpy(dtype=float) / np.minimum(cap, ratio)
    return np.c_[ct, dtc, io]


def infer_cost_model(rows: pd.DataFrame) -> Dict[str, object]:
    """Recover (w_ct, w_dt, w_io/bw, cap) and which dt the cost used.

    w_io and bw are not separately identifiable from the log -- they enter
    only as the product w_io/bw -- so that ratio is what is reported. It is
    the only combination the cost actually depends on.
    """
    d = rows[rows["cost_valid"] & np.isfinite(rows["cost"])
             & (rows["ratio"] > 0)].copy()
    if len(d) < 8:
        return {"identified": False,
                "reason": f"only {len(d)} usable rows"}

    y = d["cost"].to_numpy(dtype=float)
    # Candidate dt vectors: the row's own measurement, or the chunk-constant
    # value the default path substitutes (recovered as the chunk's primary
    # pred_dt_ms).
    prim = (d[d["role"] == "primary"].groupby("chunk_uid")["pred_dt_ms"]
            .first())
    dt_const = d["chunk_uid"].map(prim).to_numpy(dtype=float)
    dt_opts = {"per-row measured dt_ms": d["dt_ms"].to_numpy(dtype=float),
               "chunk-constant primary pred_dt_ms": dt_const,
               "per-row pred_dt_ms": d["pred_dt_ms"].to_numpy(dtype=float)}

    best = None
    for cap in (100.0, 1e300):
        for dt_name, dt in dt_opts.items():
            if not np.isfinite(dt).all():
                continue
            A = _design(d, cap, dt)
            if not np.isfinite(A).all():
                continue
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ w
            denom = np.where(np.abs(y) > 0, np.abs(y), 1.0)
            rel = float(np.max(np.abs(pred - y) / denom))
            cand = {"cap": cap, "dt_source": dt_name,
                    "w_ct": float(w[0]), "w_dt": float(w[1]),
                    "w_io_over_bw": float(w[2]), "max_rel_error": rel,
                    "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
            if best is None or rel < best["max_rel_error"]:
                best = cand
    assert best is not None
    best["identified"] = best["max_rel_error"] < 1e-6
    best["n_rows_fit"] = int(len(d))
    if best["identified"]:
        # bw is recoverable only if w_io is assumed 1, which is its default
        # and the only value that makes the product interpretable as a
        # bandwidth. Reported as such, not as a measurement.
        best["implied_bandwidth_bytes_per_ms_if_w_io_1"] = (
            1.0 / best["w_io_over_bw"] if best["w_io_over_bw"] > 0 else None)
        ct_w, dt_w = best["w_ct"], best["w_dt"]
        tol = 1e-9 * max(1.0, abs(best["w_io_over_bw"]))
        # With w_dt = 0 every dt vector fits identically, so whichever one the
        # search happened to land on says nothing. Do not report a source the
        # data cannot distinguish.
        if abs(dt_w) <= tol:
            best["dt_source"] = ("unidentifiable -- w_dt is 0, so the cost "
                                 "does not depend on any dt at all")
        best["scores_on"] = (
            "capped ratio ONLY (w_ct = w_dt = 0): the cost is a monotone "
            "function of min(ratio, cap), so selection is pure ratio "
            "maximisation and compression time is not traded against it at all"
            if abs(ct_w) <= tol and abs(dt_w) <= tol else
            "ratio and time jointly")
    else:
        best["scores_on"] = "unknown -- the cost column does not fit the model"
    return best


def cost_diagnostics(rows: pd.DataFrame, model: Dict[str, object]) -> dict:
    """What the recovered cost model implies about the selection problem."""
    out: Dict[str, object] = {}
    cap = float(model.get("cap", 100.0))
    d = rows[rows["cost_valid"]]
    out["ratio_cap"] = cap
    out["pct_rows_at_or_above_cap"] = float(100.0 * (d["ratio"] >= cap).mean())
    # A chunk is "decided at the cap" when its best cost is shared by more than
    # one distinct configuration -- i.e. the cost model cannot tell them apart.
    ties, n = 0, 0
    for _, g in d.groupby("chunk_uid"):
        n += 1
        best = g["cost"].min()
        if int(np.isclose(g["cost"], best, rtol=1e-12).sum()) > 1:
            ties += 1
    out["chunks_with_tied_best_cost"] = ties
    out["chunks_total"] = n
    out["pct_chunks_tied_at_best"] = 100.0 * ties / n if n else 0.0
    return out


# --------------------------------------------------------------------------
# Winners
# --------------------------------------------------------------------------

#: The five senses in which a configuration can "win" a chunk.
WINNER_SPECS = [
    ("best_ratio", "ratio", "max"),
    ("fastest_ct", "ct_ms", "min"),
    ("fastest_dt", "dt_ms_measured", "min"),
    ("lowest_cost", "cost", "min"),
]


def winners(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per chunk: the winning configuration under each objective, plus
    the configuration actually adopted.

    Ties are broken DETERMINISTICALLY by (rank, lib_name, quantize, shuffle) so
    repeated runs agree, and the number of tied candidates is carried alongside
    so a tie is never mistaken for a decision.
    """
    out: List[dict] = []
    order = ["rank", "lib_name", "quantize", "shuffle"]
    for uid, g in rows.groupby("chunk_uid", sort=True):
        rec: Dict[str, object] = {"chunk_uid": uid,
                                  "n_candidates": int(len(g))}
        for label, col, how in WINNER_SPECS:
            gv = g[np.isfinite(g[col])] if col in g else g.iloc[:0]
            if col == "cost":
                gv = gv[gv["cost_valid"]]
            if not len(gv):
                rec[f"{label}_lib"] = None
                rec[f"{label}_value"] = np.nan
                rec[f"{label}_nties"] = 0
                continue
            best = gv[col].max() if how == "max" else gv[col].min()
            tied = gv[np.isclose(gv[col], best, rtol=1e-12, atol=0.0)]
            pick = tied.sort_values(order).iloc[0]
            rec[f"{label}_lib"] = pick["lib_name"]
            rec[f"{label}_quantize"] = int(pick["quantize"])
            rec[f"{label}_shuffle"] = int(pick["shuffle"])
            rec[f"{label}_config"] = pick["config_id"]
            rec[f"{label}_value"] = float(best)
            rec[f"{label}_nties"] = int(len(tied))

        ad = g[g["adopted"] == 1]
        if len(ad):
            a = ad.iloc[0]
            rec.update({
                "adopted_lib": a["lib_name"],
                "adopted_quantize": int(a["quantize"]),
                "adopted_shuffle": int(a["shuffle"]),
                "adopted_config": a["config_id"],
                "adopted_role": a["role"],
                "adopted_rank": int(a["rank"]),
                "adopted_ratio": float(a["ratio"]),
                "adopted_ct_ms": float(a["ct_ms"]),
                "adopted_dt_ms": float(a["dt_ms_measured"]),
                "adopted_cost": float(a["cost"]),
                "n_adopted_rows": int(len(ad)),
            })
        else:
            rec["n_adopted_rows"] = 0
        prim = g[g["role"] == "primary"]
        if len(prim):
            p = prim.iloc[0]
            rec.update({"primary_lib": p["lib_name"],
                        "primary_quantize": int(p["quantize"]),
                        "primary_shuffle": int(p["shuffle"]),
                        "primary_config": p["config_id"],
                        "primary_ratio": float(p["ratio"]),
                        "primary_cost": float(p["cost"])})
        rec["primary_cost_baseline"] = float(g["primary_cost"].iloc[0])
        out.append(rec)
    return pd.DataFrame(out)


def selection_regret(rows: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """How much worse the adopted configuration is than the best available.

    DIRECTION. `cost` is lower-is-better (neuropress_cost.h), so

        regret = (adopted_cost - best_cost) / best_cost   >= 0

    matching the runtime's own definition of regret in neuropress_chunk_diag.h
    ((primary_cost - best_explored_cost) / best_explored_cost). A regret of 0
    means the sweep adopted a cost-optimal candidate -- which it does by
    construction whenever the sweep ran, since it adopts the argmin. The
    interesting quantities are therefore the OTHER two:

      * `oracle_ratio_regret` -- how much compression ratio the cost-optimal
        choice gave up against the highest-ratio candidate. This is where the
        cost model's own preferences show up.
      * `primary_regret` -- how much worse the MODEL'S OWN PICK was than the
        best measured candidate. This is the one that measures the predictor,
        and it is defined for every chunk whether or not exploration adopted.
    """
    r = win.copy()
    best = r["lowest_cost_value"]
    r["adopted_regret"] = (r["adopted_cost"] - best) / best.replace(0, np.nan)
    r["primary_regret"] = (
        (r["primary_cost_baseline"] - best) / best.replace(0, np.nan))
    r["adopted_is_cost_optimal"] = np.isclose(
        r["adopted_cost"], best, rtol=1e-9)
    r["primary_is_cost_optimal"] = np.isclose(
        r["primary_cost_baseline"], best, rtol=1e-9)
    r["exploration_changed_choice"] = (r["adopted_role"] == "alt")

    # Ratio given up by scoring on cost instead of ratio.
    r["oracle_ratio_regret"] = (
        (r["best_ratio_value"] - r["adopted_ratio"]) / r["best_ratio_value"])
    r["primary_ratio_regret"] = (
        (r["best_ratio_value"] - r["primary_ratio"]) / r["best_ratio_value"])
    r["adopted_matches_best_ratio_lib"] = (
        r["adopted_lib"] == r["best_ratio_lib"])
    r["primary_matches_best_ratio_lib"] = (
        r["primary_lib"] == r["best_ratio_lib"])
    r["primary_matches_lowest_cost_lib"] = (
        r["primary_lib"] == r["lowest_cost_lib"])
    return r


def summarize_selection(reg: pd.DataFrame) -> dict:
    """Headline selection numbers, with the caveat that makes them readable."""
    n = len(reg)
    if not n:
        return {}
    def pct(mask) -> float:
        return float(100.0 * pd.Series(mask).mean())
    return {
        "n_chunks": int(n),
        "pct_exploration_changed_choice":
            pct(reg["exploration_changed_choice"]),
        "pct_adopted_cost_optimal": pct(reg["adopted_is_cost_optimal"]),
        "pct_primary_cost_optimal": pct(reg["primary_is_cost_optimal"]),
        "primary_regret_mean": float(reg["primary_regret"].mean()),
        "primary_regret_median": float(reg["primary_regret"].median()),
        "primary_regret_p90": float(reg["primary_regret"].quantile(0.90)),
        "primary_regret_max": float(reg["primary_regret"].max()),
        "pct_primary_picks_best_ratio_codec":
            pct(reg["primary_matches_best_ratio_lib"]),
        "pct_primary_picks_lowest_cost_codec":
            pct(reg["primary_matches_lowest_cost_lib"]),
        "adopted_ratio_regret_mean": float(reg["oracle_ratio_regret"].mean()),
        "adopted_ratio_regret_median":
            float(reg["oracle_ratio_regret"].median()),
        "pct_adopted_is_best_ratio_codec":
            pct(reg["adopted_matches_best_ratio_lib"]),
        "note": ("adopted_regret is ~0 by construction: the sweep adopts the "
                 "argmin of the same cost column. primary_regret is the "
                 "figure that measures the PREDICTOR, and "
                 "oracle_ratio_regret the compression ratio the cost model "
                 "chose to give up."),
    }

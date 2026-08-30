#!/usr/bin/env python3
"""One profile per codec: what it does, and what kind of data it does it on.

Codecs are DISCOVERED from the log (`lib_name`), never enumerated. The action
space is a runtime property -- eight nvcomp codecs today, a different set the
moment KnownCompressors() changes -- and a hard-coded list would silently drop
a codec rather than report it.

TWO SENSES OF "WINS", kept apart because they disagree constantly:

  * `best_ratio`  -- the codec with the highest measured ratio on the chunk.
    A property of the data and the codec, nothing else.
  * `lowest_cost` -- the codec the cost model would pick. Depends on weights
    that are set at runtime, and under a ratio-only cost with a cap it is
    frequently a TIE among everything above the cap. Every win-rate figure
    here carries the tie count, because a codec that "wins" a 14-way tie by
    the deterministic tiebreak has not won anything.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .statistics import _corr, describe


def codec_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behaviour per codec, split by quantize and shuffle.

    Split rather than pooled: pooling a codec's lossless and lossy rows
    averages two different transforms of the data and produces a ratio that
    describes neither.
    """
    recs: List[dict] = []
    keys = [k for k in ("lib_name", "quantize", "shuffle") if k in rows]
    for key, g in rows.groupby(keys, dropna=False):
        keys_t = key if isinstance(key, tuple) else (key,)
        rec: Dict[str, object] = dict(zip(keys, keys_t))
        rec["n_rows"] = int(len(g))
        rec["n_chunks"] = int(g["chunk_uid"].nunique())
        for col, pre in (("ratio", "ratio"), ("ct_ms", "ct"),
                         ("dt_ms_measured", "dt"), ("cost", "cost")):
            if col not in g:
                continue
            d = describe(g[col])
            for k, v in d.items():
                if k != "n":
                    rec[f"{pre}_{k}"] = v
            rec[f"{pre}_n"] = d.get("n", 0)
        rec["pct_expanded"] = float(100.0 * (g["ratio"] < 1.0).mean())
        # Effective compression throughput, the quantity a system integrator
        # actually budgets. Bytes IN per ms of kernel time.
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = g["chunk_bytes"] / g["ct_ms"] / 1e3   # MB/s given ms and B
        rec["ct_throughput_MBps_median"] = float(
            pd.Series(tp).replace([np.inf, -np.inf], np.nan).median())
        recs.append(rec)
    return pd.DataFrame(recs)


def win_rates(rows: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """How often each codec takes each kind of win, and how contested it was."""
    recs: List[dict] = []
    n = len(win)
    if not n:
        return pd.DataFrame()
    for kind in ("best_ratio", "fastest_ct", "fastest_dt", "lowest_cost"):
        col = f"{kind}_lib"
        if col not in win:
            continue
        for lib, g in win.groupby(col, dropna=True):
            recs.append({
                "objective": kind, "lib_name": lib,
                "n_wins": int(len(g)), "pct_wins": float(100.0 * len(g) / n),
                "median_nties": float(g[f"{kind}_nties"].median())
                if f"{kind}_nties" in g else np.nan,
                "pct_uncontested": float(
                    100.0 * (g[f"{kind}_nties"] == 1).mean())
                if f"{kind}_nties" in g else np.nan,
            })
    if "adopted_lib" in win:
        for lib, g in win.groupby("adopted_lib", dropna=True):
            recs.append({"objective": "adopted", "lib_name": lib,
                         "n_wins": int(len(g)),
                         "pct_wins": float(100.0 * len(g) / n),
                         "median_nties": np.nan, "pct_uncontested": np.nan})
    if "primary_lib" in win:
        for lib, g in win.groupby("primary_lib", dropna=True):
            recs.append({"objective": "model_pick", "lib_name": lib,
                         "n_wins": int(len(g)),
                         "pct_wins": float(100.0 * len(g) / n),
                         "median_nties": np.nan, "pct_uncontested": np.nan})
    return pd.DataFrame(recs)


def winner_property_profile(chunks: pd.DataFrame, win: pd.DataFrame,
                            features: Sequence[str],
                            objective: str = "best_ratio") -> pd.DataFrame:
    """The property region each codec wins in, in feature units.

    Only chunks where the win was UNCONTESTED (`nties == 1`) are counted: a
    codec that shares a 14-way tie tells you nothing about which data it
    prefers, and including those pulls every codec's profile toward the global
    mean until they are indistinguishable.
    """
    col, tie = f"{objective}_lib", f"{objective}_nties"
    if col not in win:
        return pd.DataFrame()
    m = win[["chunk_uid", col] + ([tie] if tie in win else [])].merge(
        chunks, on="chunk_uid", how="left")
    if tie in m:
        contested = m[m[tie] > 1]
        m = m[m[tie] == 1]
    else:
        contested = m.iloc[:0]
    recs: List[dict] = []
    for lib, g in m.groupby(col, dropna=True):
        rec: Dict[str, object] = {"objective": objective, "lib_name": lib,
                                  "n_uncontested_wins": int(len(g))}
        for f in features:
            if f not in g:
                continue
            d = describe(g[f])
            rec[f"{f}_median"] = d.get("median", np.nan)
            rec[f"{f}_p10"] = d.get("p10", np.nan)
            rec[f"{f}_p90"] = d.get("p90", np.nan)
            rec[f"{f}_min"] = d.get("min", np.nan)
            rec[f"{f}_max"] = d.get("max", np.nan)
        recs.append(rec)
    out = pd.DataFrame(recs)
    if not out.empty:
        out.attrs["n_contested_excluded"] = int(len(contested))
    return out


def codec_feature_sensitivity(rows: pd.DataFrame,
                              features: Sequence[str]) -> pd.DataFrame:
    """How strongly each codec's ratio tracks each feature, one row per chunk.

    This is the "which codecs are most sensitive to entropy" question, and it
    is only answerable inside a fixed (codec, quantize, shuffle) cell -- across
    codecs the between-codec variance dominates and the pooled correlation
    describes the codec ranking rather than the data.
    """
    recs: List[dict] = []
    for key, g in rows.groupby(["lib_name", "quantize", "shuffle"],
                               dropna=False):
        lib, q, sh = key
        g1 = g.groupby("chunk_uid", as_index=False).first()
        if len(g1) < 8:
            continue
        for f in features:
            if f not in g1:
                continue
            for target in ("ratio", "ct_ms", "dt_ms_measured"):
                if target not in g1:
                    continue
                c = _corr(g1[f].to_numpy(dtype=float),
                          g1[target].to_numpy(dtype=float))
                recs.append({"lib_name": lib, "quantize": int(q),
                             "shuffle": int(sh), "feature": f,
                             "target": target, "n_chunks": c["n"],
                             "pearson_r": c["pearson_r"],
                             "spearman_rho": c["spearman_rho"],
                             "p": c["pearson_p"],
                             "r2": c["r2_linear"]})
    return pd.DataFrame(recs)


def build_profiles(rows: pd.DataFrame, win: pd.DataFrame,
                   chunks: pd.DataFrame, features: Sequence[str],
                   shuffle_pairs: pd.DataFrame,
                   quant_pairs: pd.DataFrame) -> Dict[str, dict]:
    """A readable per-codec profile, assembled from the tables above.

    Only claims with support behind them are emitted: a codec with no
    uncontested win contributes no property range, and a treatment measured on
    fewer than 8 pairs contributes no sensitivity verdict.
    """
    profiles: Dict[str, dict] = {}
    prof = winner_property_profile(chunks, win, features, "best_ratio")
    summ = codec_summary(rows)
    sens = codec_feature_sensitivity(rows, features)
    wr = win_rates(rows, win)

    for lib in sorted(rows["lib_name"].dropna().unique()):
        p: Dict[str, object] = {"codec": lib}
        s = summ[summ["lib_name"] == lib]
        for label, mask in (("lossless_noshuffle",
                             (s["quantize"] == 0) & (s["shuffle"] == 0)),
                            ("lossless_shuffled",
                             (s["quantize"] == 0) & (s["shuffle"] > 0)),
                            ("lossy_noshuffle",
                             (s["quantize"] == 1) & (s["shuffle"] == 0)),
                            ("lossy_shuffled",
                             (s["quantize"] == 1) & (s["shuffle"] > 0))):
            sub = s[mask]
            if sub.empty:
                continue
            p[label] = {
                "n_chunks": int(sub["n_chunks"].sum()),
                "ratio_median": float(sub["ratio_median"].median()),
                "ratio_p10": float(sub["ratio_p10"].min()),
                "ratio_p90": float(sub["ratio_p90"].max()),
                "ct_ms_median": float(sub["ct_median"].median()),
                "dt_ms_median": (float(sub["dt_median"].median())
                                 if "dt_median" in sub else None),
                "pct_expanded": float(sub["pct_expanded"].max()),
            }
        pr = prof[prof["lib_name"] == lib] if not prof.empty else prof
        if not pr.empty and int(pr["n_uncontested_wins"].iloc[0]) >= 3:
            r = pr.iloc[0]
            p["wins_on_data_with"] = {
                f: {"median": float(r.get(f"{f}_median", np.nan)),
                    "p10": float(r.get(f"{f}_p10", np.nan)),
                    "p90": float(r.get(f"{f}_p90", np.nan))}
                for f in features if f"{f}_median" in pr.columns}
            p["n_uncontested_best_ratio_wins"] = int(r["n_uncontested_wins"])
        else:
            p["wins_on_data_with"] = None
            p["n_uncontested_best_ratio_wins"] = (
                int(pr["n_uncontested_wins"].iloc[0]) if not pr.empty else 0)
        if not wr.empty:
            w = wr[(wr["lib_name"] == lib)]
            p["win_rates_pct"] = {
                str(row["objective"]): round(float(row["pct_wins"]), 2)
                for _, row in w.iterrows()}
        # `sens` is column-less when the log carries no usable features, so
        # the emptiness test has to come before the column reference.
        ss = (sens[(sens["lib_name"] == lib) & (sens["target"] == "ratio")]
              if not sens.empty else sens)
        if not ss.empty:
            p["ratio_sensitivity"] = {
                f"{r['feature']}|q{int(r['quantize'])}s{int(r['shuffle'])}":
                    round(float(r["spearman_rho"]), 3)
                for _, r in ss.iterrows() if np.isfinite(r["spearman_rho"])}
        p["shuffle"] = _treatment_verdict(shuffle_pairs, lib)
        p["quantization"] = _treatment_verdict(quant_pairs, lib)
        profiles[lib] = p
    return profiles


def _treatment_verdict(pairs: pd.DataFrame, lib: str,
                       min_pairs: int = 8) -> Optional[dict]:
    if pairs is None or pairs.empty or "lib_name" not in pairs.columns:
        return None
    g = pairs[pairs["lib_name"] == lib]
    if len(g) < min_pairs:
        return None
    d = pd.to_numeric(g.get("d_ratio"), errors="coerce").dropna()
    if d.empty:
        return None
    rel = pd.to_numeric(g.get("rel_ratio"), errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    pct = float(100.0 * g["helps_ratio"].mean())
    verdict = ("strongly beneficial" if pct >= 80 else
               "beneficial" if pct >= 55 else
               "neutral" if pct > 45 else
               "mildly harmful" if pct > 20 else "harmful")
    return {"n_pairs": int(len(g)), "pct_pairs_ratio_improves": pct,
            "median_delta_ratio": float(d.median()),
            "median_rel_ratio": float(rel.median()) if rel.size else None,
            "verdict": verdict}

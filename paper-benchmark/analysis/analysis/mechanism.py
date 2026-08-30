#!/usr/bin/env python3
"""WHY the ratio and the timings come out the way they do, given the stats.

Everything else in this pipeline measures ASSOCIATION -- how much of the
outcome moves with a feature. This module asks the different question: through
what mechanism, and how much of the outcome does that mechanism account for.
Three of them are testable from the log alone, and each makes a prediction
sharp enough to fail.

1. THE ENTROPY BOUND, and what lives above it.
   `entropy` is the Shannon entropy of the 256-bin BYTE histogram, in
   bits/byte (data_stats_gpu_kernels.cu). A coder that assigns one codeword
   per byte symbol and reads no order cannot beat

       ratio <= 8 / H

   That is not a fitted relationship; it is the source coding theorem applied
   to the exact quantity the feature measures. So the achieved ratio splits
   into two parts with two different causes:

       ratio  =  (8 / H)          x  excess
                 ^^^^^^^^            ^^^^^^
                 the histogram       everything ORDER contributes
                 (entropy knows      (entropy cannot know it, by
                  this exactly)       construction)

   The split is what makes the rest of the analysis interpretable. On the Nyx
   log this was developed against, `nvcomp-ans` -- an entropy coder -- sits at
   0.86 of its bound and exceeds it on 0 of 44 chunks, exactly as the theorem
   requires; `nvcomp-zstd` sits at 2.59x the bound and exceeds it on 44 of 44.
   So entropy predicts the ratio QUANTITATIVELY for one codec and explains
   under half of it for another, and the difference is not noise or model
   quality -- it is which of the two parts that codec can reach.

   Codecs are classified by what they DO here, not by a hard-coded list: a
   codec whose measured ratio essentially never exceeds 8/H is behaving as an
   order-blind coder on this data, whatever its name.

2. WHERE THE TRANSFORMS ACT -- and what this log can and cannot say about it.
   The two transforms are not symmetric here, and the difference decides which
   question is answerable:

     * BYTE SHUFFLE IS A PERMUTATION of the byte stream. It regroups bytes by
       position within the word and moves not one of them out of the buffer,
       so the byte multiset -- hence the 256-bin histogram, hence H, hence
       8/H -- is mathematically INVARIANT under it. Every bit of shuffle's
       gain is therefore contributed by order, and that is a theorem about the
       transform rather than a result measured here.
     * QUANTIZATION CHANGES THE ALPHABET. It maps floats onto a coarser grid,
       so it certainly does move the histogram -- and this log does not record
       by how much, because `entropy`, `mad` and `second_deriv` are computed
       ONCE PER CHUNK on the ORIGINAL buffer and repeated unchanged on all 32
       of that chunk's rows. The bound after quantization is unmeasured, so
       the split of quantization's gain into histogram and order is NOT
       IDENTIFIABLE from this log, and this module refuses to report one.
       (An earlier version of it did, and produced "quantization acts 100% on
       order" for every codec -- which was reading the logging convention, not
       the data. Logging the stats of the buffer the codec actually receives
       would make the split identifiable and is the concrete fix.)

   For the same reason the 8/H bound itself is only a bound on LOSSLESS rows.
   On a quantized row the relevant entropy is the post-quantization one, so
   those cells are computed but flagged rather than compared.

   The shuffle half is still worth the trouble, because it makes a sharp
   prediction with a control group built in: a coder that reads only the
   GLOBAL byte histogram cannot gain anything from a permutation. Where such a
   codec gains anyway, it is not modelling the global histogram -- it is
   modelling a local one, and shuffle has made the local blocks purer without
   touching the global distribution.

3. THE TIMINGS ARE DOWNSTREAM OF THE RATIO.
   Compression and decompression times correlate with the data properties, and
   it would be easy to report that as "rough data is slow to compress". The
   mediation test asks whether the correlation survives holding the OUTPUT SIZE
   constant. On the development log it does not: every codec's time tracks
   bytes/ratio more strongly than it tracks any feature, so the properties act
   on time THROUGH the ratio rather than beside it. That distinction changes
   what a scheduler should predict -- predict the ratio, and the times follow.

Direction of inference, stated once: (1) is a bound, so a violation is a
measurement or definition error rather than a discovery; (2) and (3) are
observational and are reported as mediation and interaction, never as proof of
cause. Where a mechanism is only consistent with the data rather than shown by
it, the emitted table says so in a `verdict` column.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

#: Bits in the byte the histogram is taken over.
BITS_PER_BYTE = 8.0

#: A codec that exceeds its entropy bound on no more than this fraction of
#: rows is behaving as an order-blind coder on this data.
ORDER_BLIND_MAX_EXCEED = 0.05

#: Between order-blind and order-sensitive there is a real third case: a codec
#: that sits ON its bound, crossing it about as often as not. Calling that
#: "order-sensitive" produces the self-contradicting verdict "beats the bound
#: on 50% of rows, by a median factor of 0.99", so it gets its own class.
MARGINAL_EXCESS = 1.2

#: Below this many rows a per-codec cell is not reported.
MIN_ROWS = 12


def _spear(x: pd.Series, y: pd.Series) -> float:
    """Spearman rho, nan when it is not defined rather than a warning."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return float("nan")
    return float(sps.spearmanr(x[m], y[m]).correlation)


def _prep(rows: pd.DataFrame) -> pd.DataFrame:
    d = rows.replace([np.inf, -np.inf], np.nan)
    d = d[(d["ratio"] > 0) & d["ratio"].notna()].copy()
    if "chunk_bytes" in d.columns:
        # The size the codec actually emitted. Everything downstream of the
        # compressor -- write time, read time, the I/O term -- scales with it,
        # and it is the ratio expressed in the units the hardware sees.
        d["compressed_bytes"] = d["chunk_bytes"] / d["ratio"]
    return d


# ------------------------------------------------------- 1. the bound
def entropy_bound(rows: pd.DataFrame) -> pd.DataFrame:
    """Achieved ratio against 8/H, per codec and configuration.

    `excess_over_bound` is ratio / (8/H): at most 1.0 for a coder that reads
    only the byte histogram, and unbounded above for one that reads order.
    """
    if "entropy" not in rows.columns:
        return pd.DataFrame()
    d = _prep(rows)
    d = d[d["entropy"] > 0]
    if d.empty:
        return pd.DataFrame()
    d["entropy_bound_ratio"] = BITS_PER_BYTE / d["entropy"]
    d["excess_over_bound"] = d["ratio"] / d["entropy_bound_ratio"]
    keys = [k for k in ("lib_name", "quantize", "shuffle") if k in d.columns]
    # 8/H bounds the ORIGINAL byte stream. Shuffle permutes it and so leaves
    # the bound exact; quantization replaces it and so invalidates the bound,
    # because the log never records the post-quantization entropy.
    quantized = (d["quantize"] == 1) if "quantize" in d.columns else False
    recs: List[dict] = []
    for gk, g in d.groupby(keys, dropna=False, sort=True):
        if len(g) < MIN_ROWS:
            continue
        gk = gk if isinstance(gk, tuple) else (gk,)
        over = float((g["excess_over_bound"] > 1.0).mean())
        cell_q = bool(np.any(quantized.loc[g.index])) \
            if hasattr(quantized, "loc") else False
        recs.append({
            **{k: v for k, v in zip(keys, gk)},
            "bound_valid": not cell_q,
            "bound_note": (
                "the 8/H bound applies to the ORIGINAL byte stream; this cell "
                "is quantized and its post-quantization entropy is not "
                "logged, "
                "so the excess here mixes the quantization gain with order"
                if cell_q else
                "bound exact: entropy is measured on this cell's own byte "
                "stream (shuffle is a permutation and cannot change it)"),
            "n_rows": int(len(g)),
            "n_chunks": int(g["chunk_uid"].nunique())
            if "chunk_uid" in g.columns else None,
            "median_entropy": float(g["entropy"].median()),
            "median_entropy_bound_ratio":
                float(g["entropy_bound_ratio"].median()),
            "median_ratio": float(g["ratio"].median()),
            "median_excess_over_bound":
                float(g["excess_over_bound"].median()),
            "min_excess_over_bound": float(g["excess_over_bound"].min()),
            "max_excess_over_bound": float(g["excess_over_bound"].max()),
            "pct_rows_above_bound": float(100.0 * over),
            # How much of the achieved ratio the histogram alone accounts
            # for, in log space -- the natural scale, since the two parts
            # MULTIPLY rather than add.
            "pct_of_log_ratio_from_entropy": float(
                100.0 * np.median(
                    np.log(g["entropy_bound_ratio"]) / np.log(g["ratio"]))
            ) if (g["ratio"] > 1).all() else np.nan,
        })
    return pd.DataFrame(recs)


def codec_order_sensitivity(bound: pd.DataFrame) -> pd.DataFrame:
    """Classify each codec by whether it can read order, from its own numbers.

    The classification is empirical and per-log: it says how the codec behaved
    on THIS data, not what its documentation claims.
    """
    if bound.empty or "lib_name" not in bound.columns:
        return pd.DataFrame()
    recs: List[dict] = []
    for lib, g in bound.groupby("lib_name", sort=True):
        # The lossless, unshuffled cell is the cleanest read of the codec
        # itself; fall back to everything if the log has no such cell.
        raw = g
        if {"quantize", "shuffle"} <= set(g.columns):
            sel = g[(g["quantize"] == 0) & (g["shuffle"] == 0)]
            raw = sel if not sel.empty else g
        pct_over = float(raw["pct_rows_above_bound"].max())
        med_ex = float(raw["median_excess_over_bound"].median())
        blind = pct_over <= 100.0 * ORDER_BLIND_MAX_EXCEED
        marginal = (not blind) and med_ex < MARGINAL_EXCESS
        cls = ("order-blind" if blind
               else "at the bound" if marginal else "order-sensitive")
        recs.append({
            "lib_name": lib,
            "pct_rows_above_entropy_bound": pct_over,
            "median_excess_over_bound": med_ex,
            "max_excess_over_bound": float(raw["max_excess_over_bound"].max()),
            "class": cls,
            "behaves_order_blind": bool(blind),
            # Kept short on purpose: the table truncates long cells, and the
            # numbers beside it already carry the evidence.
            "verdict": (
                f"never beats 8/H; reaches {med_ex:.2f} of it, so its ratio "
                f"is a function of the byte histogram alone"
                if blind else
                f"sits ON its bound -- median {med_ex:.2f}, crossing it on "
                f"{pct_over:.0f}% of rows; order buys it almost nothing here"
                if marginal else
                f"beats 8/H on {pct_over:.0f}% of rows by a median "
                f"{med_ex:.2f}x; that excess is repeated byte SEQUENCES, "
                f"which no current feature measures"),
        })
    return pd.DataFrame(recs).sort_values(
        "median_excess_over_bound").reset_index(drop=True)


# ------------------------------------------- 2. where the transforms act
def stats_are_pre_transform(rows: pd.DataFrame,
                            features: Sequence[str]) -> bool:
    """Are the features recomputed per configuration, or logged once per chunk?

    Everything about what the transform analysis may claim turns on this, so it
    is tested rather than assumed. If a chunk carries a single entropy value
    across all its configurations, the stats describe the buffer BEFORE the
    transform and no post-transform histogram claim is identifiable.
    """
    feats = [f for f in features if f in rows.columns]
    if not feats or "chunk_uid" not in rows.columns:
        return True
    n = rows.groupby("chunk_uid")[feats].nunique()
    return bool((n <= 1).all().all())


def shuffle_decomposition(rows: pd.DataFrame) -> pd.DataFrame:
    """Shuffle's gain, all of which is order -- by construction, not by fit.

    Byte shuffle permutes the byte stream, so the 256-bin histogram and every
    statistic derived from it are invariant under it. The entropy bound 8/H is
    therefore EXACTLY the same before and after, and the whole of the measured
    ratio change is contributed by structure the histogram cannot see.

    That turns the table below into a test rather than a description. A codec
    that models the global byte histogram must score a fold of 1.0 here. One
    that gains is modelling something local, and the size of the gain says how
    much of its ratio was coming from locality all along.

    Restricted to LOSSLESS rows: on a quantized row the reference histogram is
    not the one the log recorded.
    """
    if "entropy" not in rows.columns:
        return pd.DataFrame()
    d = _prep(rows)
    need = {"chunk_uid", "lib_name", "quantize", "shuffle"}
    if d.empty or not need <= set(d.columns):
        return pd.DataFrame()
    d = d[(d["quantize"] == 0) & (d["entropy"] > 0)]
    if d.empty:
        return pd.DataFrame()
    keys = ["chunk_uid", "lib_name"] + (
        ["preset"] if "preset" in d.columns else [])
    off = d[d["shuffle"] == 0].groupby(keys, dropna=False).agg(
        ratio_off=("ratio", "median")).reset_index()
    on = d[d["shuffle"] > 0].groupby(keys, dropna=False).agg(
        ratio_on=("ratio", "median")).reset_index()
    m = off.merge(on, on=keys, how="inner")
    if m.empty:
        return pd.DataFrame()
    recs: List[dict] = []
    for lib, g in m.groupby("lib_name", sort=True):
        if len(g) < 4:
            continue
        fold = float((g["ratio_on"] / g["ratio_off"]).median())
        recs.append({
            "lib_name": lib, "n_pairs": int(len(g)),
            "n_chunks": int(g["chunk_uid"].nunique()),
            "median_ratio_fold": fold,
            "entropy_bound_fold": 1.0,
            "pct_of_gain_from_order": 100.0,
            "gain_source": (
                "none -- shuffle does not help this codec"
                if fold <= 1.02 else
                "ORDER, in full: the byte histogram is provably unchanged "
                "by a "
                "permutation, so every bit of this fold came from structure"),
        })
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values("median_ratio_fold",
                           ascending=False).reset_index(drop=True)


def locality_probe(shuf: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    """The sharp test: does an order-blind codec still gain from a permutation?

    It cannot, if it is coding the global byte histogram -- the histogram is
    identical on both sides. So a codec classified order-blind by the bound
    test, and yet gaining from shuffle, is coding a LOCAL histogram: shuffle
    concentrates similar bytes into the same block and makes each block's
    distribution far more skewed than the global one, without changing the
    global one at all.

    This is the only place in the pipeline where two independent measurements
    combine into a statement about how a codec works internally, so it is kept
    separate and labelled as an inference rather than folded into either table.
    """
    if shuf.empty or order.empty or "class" not in order.columns:
        return pd.DataFrame()
    m = shuf.merge(order[["lib_name", "class", "median_excess_over_bound",
                          "pct_rows_above_entropy_bound"]],
                   on="lib_name", how="inner")
    if m.empty:
        return pd.DataFrame()
    recs: List[dict] = []
    for _, r in m.iterrows():
        blind = r["class"] == "order-blind"
        gains = float(r["median_ratio_fold"]) > 1.02
        if blind and gains:
            verdict = (
                f"models a LOCAL histogram. It never beats the GLOBAL bound "
                f"({float(r['pct_rows_above_entropy_bound']):.0f}% of rows) "
                f"yet gains {float(r['median_ratio_fold']):.2f}x from a "
                f"permutation that cannot change that global histogram -- so "
                f"what it codes is per-block, and shuffle purifies the blocks")
        elif blind:
            verdict = ("models the GLOBAL histogram: it neither beats the "
                       "bound nor gains from a permutation, which is what a "
                       "pure global entropy coder must do")
        elif gains:
            verdict = (f"reads order, and shuffle gives it "
                       f"{float(r['median_ratio_fold']):.2f}x more of it")
        else:
            verdict = ("reads order, but shuffle does not give it more on "
                       "this data")
        recs.append({
            "lib_name": r["lib_name"],
            "class_from_entropy_bound": r["class"],
            "shuffle_ratio_fold": float(r["median_ratio_fold"]),
            "n_chunks": int(r["n_chunks"]),
            "inference": verdict,
        })
    return pd.DataFrame(recs).sort_values(
        "shuffle_ratio_fold", ascending=False).reset_index(drop=True)


# ------------------------------------------ 3. timings are downstream
def timing_mediation(rows: pd.DataFrame, features: Sequence[str],
                     metrics: Sequence[str] = ("ct_ms", "dt_ms_measured")
                     ) -> pd.DataFrame:
    """Does a feature act on TIME directly, or only by changing the ratio?

    For each (codec, feature, time metric): the raw rank correlation with the
    feature, the rank correlation with the compressed size, and the feature's
    correlation once the compressed size is held constant. If the third
    collapses toward zero while the second is large, the feature's effect on
    time is MEDIATED by the ratio -- the data changes how much output there
    is, and the output size sets the time.

    Rank-based throughout: the relationships are monotone but not linear, and
    a Pearson r on raw milliseconds would be dominated by the slowest rows.
    """
    d = _prep(rows)
    feats = [f for f in features if f in d.columns]
    if d.empty or not feats or "compressed_bytes" not in d.columns:
        return pd.DataFrame()
    recs: List[dict] = []
    for metric in metrics:
        if metric not in d.columns:
            continue
        for lib, g0 in d.groupby("lib_name", sort=True):
            g0 = g0[np.isfinite(g0[metric])]
            # A sentinel (-1) is not a fast decompression.
            g0 = g0[g0[metric] >= 0]
            if len(g0) < MIN_ROWS:
                continue
            r_size = _spear(g0["compressed_bytes"], g0[metric])
            for f in feats:
                g = g0[np.isfinite(g0[f])]
                if len(g) < MIN_ROWS:
                    continue
                r_raw = _spear(g[f], g[metric])
                # Partial Spearman: rank everything, then residualise both on
                # the ranked mediator. Ranks keep it robust to the heavy tail
                # in both time and size.
                rk = g[[f, metric, "compressed_bytes"]].rank()
                z = rk["compressed_bytes"].to_numpy()
                Z = np.column_stack([np.ones_like(z), z])
                try:
                    resid = {}
                    for col in (f, metric):
                        y = rk[col].to_numpy()
                        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
                        resid[col] = y - Z @ beta
                    r_part = _spear(pd.Series(resid[f]),
                                    pd.Series(resid[metric]))
                except np.linalg.LinAlgError:
                    r_part = float("nan")
                mediated = bool(
                    np.isfinite(r_raw) and np.isfinite(r_part)
                    and np.isfinite(r_size)
                    and abs(r_size) > abs(r_raw)
                    and abs(r_part) < 0.5 * abs(r_raw))
                recs.append({
                    "metric": metric, "lib_name": lib, "feature": f,
                    "n_rows": int(len(g)),
                    "n_chunks": int(g["chunk_uid"].nunique())
                    if "chunk_uid" in g.columns else None,
                    "rho_time_vs_feature": r_raw,
                    "rho_time_vs_compressed_bytes": r_size,
                    "partial_rho_time_vs_feature_given_size": r_part,
                    "attenuation": (
                        float(1.0 - abs(r_part) / abs(r_raw))
                        if np.isfinite(r_raw) and abs(r_raw) > 1e-9
                        and np.isfinite(r_part) else np.nan),
                    "mediated_by_output_size": mediated,
                })
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values(
        ["metric", "lib_name", "feature"]).reset_index(drop=True)


def throughput(rows: pd.DataFrame) -> pd.DataFrame:
    """Bytes per millisecond, in and out, per codec.

    The unit a scheduler actually needs, and the one that makes a time
    comparison across chunk sizes meaningful at all.
    """
    d = _prep(rows)
    if d.empty or "chunk_bytes" not in d.columns:
        return pd.DataFrame()
    recs: List[dict] = []
    for lib, g in d.groupby("lib_name", sort=True):
        rec: Dict[str, object] = {"lib_name": lib, "n_rows": int(len(g))}
        ct = g[np.isfinite(g.get("ct_ms", np.nan)) & (g.get("ct_ms", -1) > 0)]
        if len(ct) >= MIN_ROWS:
            rec["median_compress_MBps"] = float(
                (ct["chunk_bytes"] / ct["ct_ms"] / 1e3).median())
            rec["median_ct_ms"] = float(ct["ct_ms"].median())
        dtc = "dt_ms_measured" if "dt_ms_measured" in g.columns else None
        if dtc:
            dt = g[np.isfinite(g[dtc]) & (g[dtc] > 0)]
            if len(dt) >= MIN_ROWS:
                # Decompression reads the COMPRESSED bytes and writes the
                # original, so its natural rate is over the output.
                rec["median_decompress_MBps"] = float(
                    (dt["chunk_bytes"] / dt[dtc] / 1e3).median())
                rec["median_dt_ms"] = float(dt[dtc].median())
        recs.append(rec)
    return pd.DataFrame(recs)


def summarize(bound: pd.DataFrame, order: pd.DataFrame,
              shuf: pd.DataFrame, loc: pd.DataFrame, med: pd.DataFrame,
              pre_transform: bool = True) -> Dict[str, object]:
    """The mechanism verdict, in the form the report and answers both read."""
    out: Dict[str, object] = {"available": not bound.empty}
    if bound.empty:
        out["reason"] = ("the log carries no entropy column, so the ratio "
                         "cannot be compared against its entropy bound")
        return out
    if not order.empty:
        blind = order[order["behaves_order_blind"]]
        seeing = order[~order["behaves_order_blind"]]
        out["n_codecs"] = int(len(order))
        out["order_blind_codecs"] = blind["lib_name"].tolist()
        out["order_sensitive_codecs"] = seeing["lib_name"].tolist()
        if "class" in order.columns:
            out["codec_classes"] = {
                str(r["lib_name"]): str(r["class"])
                for _, r in order.iterrows()}
        if not blind.empty:
            # The order-blind codec CLOSEST to its bound is the one that
            # demonstrates the bound is real and tight. The lowest one only
            # demonstrates that it is far away, which shows nothing.
            b = blind.sort_values("median_excess_over_bound").iloc[-1]
            out["entropy_bound_holds"] = (
                f"{b['lib_name']} exceeds 8/entropy on "
                f"{float(b['pct_rows_above_entropy_bound']):.0f}% of rows and "
                f"sits at {float(b['median_excess_over_bound']):.2f} of its "
                f"bound, so for it the ratio IS a function of the byte "
                f"histogram and entropy predicts it in closed form")
        if "class" in seeing.columns:
            seeing = seeing[seeing["class"] == "order-sensitive"]
        if not seeing.empty:
            s = seeing.sort_values("median_excess_over_bound").iloc[-1]
            out["order_contribution"] = (
                f"{s['lib_name']} reaches "
                f"{float(s['median_excess_over_bound']):.2f}x its entropy "
                f"bound at the median and up to "
                f"{float(s['max_excess_over_bound']):.0f}x, and every bit of "
                f"that excess comes from repeated byte SEQUENCES that no "
                f"current feature measures")
    out["stats_measured_pre_transform"] = bool(pre_transform)
    if pre_transform:
        out["quantization_split_identifiable"] = False
        out["quantization_split_reason"] = (
            "the data properties are logged once per chunk, computed on the "
            "ORIGINAL buffer, and repeated unchanged on every "
            "configuration of that chunk. Quantization changes the value "
            "alphabet and therefore "
            "the byte histogram, but the post-quantization entropy is never "
            "recorded, so how much of quantization's gain is a smaller "
            "alphabet and how much is longer runs cannot be separated from "
            "this log. Recording the stats of the buffer the codec actually "
            "receives would make it identifiable.")
    if not shuf.empty:
        best = shuf.iloc[0]
        out["shuffle_max_fold"] = float(best["median_ratio_fold"])
        out["shuffle_gain_is_pure_order"] = (
            f"byte shuffle permutes the byte stream, so the 256-bin histogram "
            f"and 8/H are invariant under it -- the whole of its gain is "
            f"order, up to {float(best['median_ratio_fold']):.2f}x on "
            f"{best['lib_name']}. This is a property of the transform, not a "
            f"fit to these rows.")
    if not loc.empty:
        hits = loc[loc["inference"].str.startswith("models a LOCAL")]
        out["n_codecs_modelling_local_histogram"] = int(len(hits))
        if not hits.empty:
            h = hits.iloc[0]
            out["locality_finding"] = (
                f"{h['lib_name']} never beats the global entropy bound yet "
                f"gains {float(h['shuffle_ratio_fold']):.2f}x from a "
                f"permutation that cannot change it, so it is coding a "
                f"per-block histogram rather than the global one")
    if not med.empty:
        out["pct_timing_cells_mediated_by_output_size"] = float(
            100.0 * med["mediated_by_output_size"].mean())
        out["n_timing_cells"] = int(len(med))
        strong = med[med["rho_time_vs_compressed_bytes"].abs()
                     > med["rho_time_vs_feature"].abs()]
        out["pct_timing_cells_size_beats_feature"] = float(
            100.0 * len(strong) / len(med))
    return out

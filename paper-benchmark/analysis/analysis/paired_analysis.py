#!/usr/bin/env python3
"""Controlled paired comparisons for shuffle, quantization and error bound.

WHY PAIRED AND NOT AVERAGED. Comparing mean(ratio | shuffle on) against
mean(ratio | shuffle off) answers a different question than the one that gets
asked. The sweep is a full factorial per chunk, so the two means are taken over
the SAME chunks and the SAME codecs -- but the moment any candidate fails, or
the sweep is truncated, or one codec declines the shuffle width, the two sides
stop being over the same population and the difference picks up whichever
codec happens to be missing from one of them. Pairing on

    (chunk, codec, preset, quantize, error_bound)

removes the chunk effect, the codec effect and the configuration effect
exactly, so what is left is the treatment. It also lets the sign be counted
per pair, which a difference of means cannot do: "shuffle raises the mean
ratio" and "shuffle helps on 40% of pairs" are both true of the same data and
only the second one is actionable.

A REAL CONFOUND THIS CANNOT REMOVE, recorded here because it belongs with the
numbers rather than in a caveats section. Candidates are measured while up to
CLIO_NEUROPRESS_EXPLORE_STREAMS of them (default 4) run CONCURRENTLY on the
device (compressor_runtime.cc, the batched drain), while the primary is
measured alone. Compression and decompression TIME deltas therefore carry a
contention term that depends on what else was in flight, which the log does
not record. Ratio deltas are unaffected -- the codec output size does not
depend on scheduling -- so ratio conclusions are safe and timing conclusions
from paired rows are marked as such wherever they appear.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Outcomes a treatment can move, and the direction that counts as "helps".
DELTA_TARGETS = [
    ("ratio", "higher_is_better"),
    ("ct_ms", "lower_is_better"),
    ("dt_ms_measured", "lower_is_better"),
    ("cost", "lower_is_better"),
]
QUALITY_TARGETS = ["meas_rmse_ok", "meas_max_error_ok", "meas_psnr_db_ok",
                   "ssim_deviation"]


def _pair(rows: pd.DataFrame, treat_col: str, control_value, treat_value,
          hold: Sequence[str]) -> pd.DataFrame:
    """Inner join of the control and treatment arms on `hold`.

    validate="one_to_one" is load-bearing: if the held-constant columns do not
    uniquely identify a row within each arm, the join would silently produce a
    cross product and every delta after it would be an average over unrelated
    pairs. A failure here means the key is wrong, and it should stop the run.
    """
    hold = [h for h in hold if h in rows.columns]
    a = rows[rows[treat_col] == control_value]
    b = rows[rows[treat_col] == treat_value]
    if a.empty or b.empty:
        return pd.DataFrame()
    keep = list(dict.fromkeys(
        hold + ["chunk_uid", "field", "timestep", "workload", "entropy",
                "mad", "second_deriv", "chunk_bytes"]))
    keep = [k for k in keep if k in rows.columns]
    vals = [c for c, _ in DELTA_TARGETS] + [
        q for q in QUALITY_TARGETS if q in rows.columns]
    vals = [v for v in vals if v in rows.columns]
    a = a[keep + vals].copy()
    b = b[keep + vals].copy()
    try:
        m = a.merge(b, on=hold, suffixes=("_off", "_on"),
                    validate="one_to_one")
    except pd.errors.MergeError as e:
        raise AssertionError(
            f"paired join on {hold} is not one-to-one for {treat_col}: {e}. "
            f"The hold-constant key does not identify a single row per arm, "
            f"so the log contains repeated measurements of one configuration."
        ) from e
    return m


def _deltas(m: pd.DataFrame, label: str, treat_col: str,
            control_value, treat_value) -> pd.DataFrame:
    if m.empty:
        return m
    out = m.copy()
    out["treatment"] = label
    out["treat_col"] = treat_col
    out["control_value"] = control_value
    out["treat_value"] = treat_value
    for col, direction in DELTA_TARGETS:
        on, off = f"{col}_on", f"{col}_off"
        if on not in out or off not in out:
            continue
        out[f"d_{col}"] = out[on] - out[off]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"rel_{col}"] = np.where(
                np.abs(out[off]) > 0, out[f"d_{col}"] / out[off], np.nan)
        sign = 1.0 if direction == "higher_is_better" else -1.0
        out[f"helps_{col}"] = sign * out[f"d_{col}"] > 0
    for q in QUALITY_TARGETS:
        on, off = f"{q}_on", f"{q}_off"
        if on in out and off in out:
            out[f"d_{q}"] = out[on] - out[off]
    # Carry the chunk properties under their plain names -- they are identical
    # on both arms by construction (same chunk), so the _off copy is canonical.
    for f in ("entropy", "mad", "second_deriv", "chunk_bytes", "field",
              "timestep", "workload"):
        if f"{f}_off" in out:
            out[f] = out[f"{f}_off"]
        elif f in m:
            out[f] = m[f]
    return out


# --------------------------------------------------------------------------
# The three treatments
# --------------------------------------------------------------------------

def shuffle_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """shuffle OFF vs each shuffle WIDTH, holding chunk/codec/quant/eb/preset.

    `shuffle` is a WIDTH, not a flag (neuropress_telemetry.cc says so
    explicitly), so widths 2, 4 and 8 are compared against 0 separately rather
    than collapsed -- collapsing made a stride-8 run indistinguishable from a
    stride-4 one, which is the exact bug the writer's comment warns about.
    """
    if "shuffle" not in rows.columns:
        return pd.DataFrame()
    widths = sorted(int(w) for w in rows["shuffle"].dropna().unique()
                    if int(w) != 0)
    hold = ["chunk_uid", "lib_name", "preset", "quantize", "error_bound"]
    parts = []
    for w in widths:
        m = _pair(rows, "shuffle", 0, w, hold)
        if m.empty:
            continue
        parts.append(_deltas(m, f"shuffle_{w}", "shuffle", 0, w))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def quantization_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """quantize 0 vs 1, holding chunk/codec/shuffle/preset.

    The error bound cannot be held constant: it does not exist on the lossless
    arm at all (eb_encoded carries the 1e-7 model sentinel there, which is not
    a bound). But it must not be pooled over either -- "quantizing helps by
    2.4x" averaged over a 1e-3 and a 1e-2 bound describes neither treatment.

    So the lossless arm is paired against EACH bound separately, exactly as
    the shuffle widths are, and the treatment label carries the bound. With a
    single bound in the sweep -- the usual case -- this reduces to one
    comparison and the distinction is invisible; with several it is the
    difference between a one-to-one join and a cross product.
    """
    if "quantize" not in rows.columns:
        return pd.DataFrame()
    hold = ["chunk_uid", "lib_name", "preset", "shuffle"]
    lossy = rows[rows["quantize"] == 1]
    bounds = sorted(b for b in lossy["error_bound"].dropna().unique()) \
        if "error_bound" in lossy.columns else []
    if len(bounds) <= 1:
        m = _pair(rows, "quantize", 0, 1, hold)
        if m.empty:
            return pd.DataFrame()
        out = _deltas(m, "quantize", "quantize", 0, 1)
        if "error_bound_on" in out.columns:
            out["error_bound"] = out["error_bound_on"]
        return out

    parts = []
    for b in bounds:
        # One bound's lossy rows, plus every lossless row, so the arms are
        # one-to-one on the held key again.
        sub = rows[(rows["quantize"] == 0)
                   | ((rows["quantize"] == 1)
                      & (rows["error_bound"] == b))]
        m = _pair(sub, "quantize", 0, 1, hold)
        if m.empty:
            continue
        d = _deltas(m, f"quantize_eb_{b:g}", "quantize", 0, 1)
        d["error_bound"] = b
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def error_bound_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """Every ordered pair of error bounds among the LOSSY rows, holding
    chunk/codec/shuffle/preset. Tightest bound is the reference arm.

    Empty when the sweep explored a single bound, which is the common case and
    is reported as an unavailable analysis rather than an absent finding.
    """
    lossy = rows[rows["quantize"] == 1]
    if lossy.empty or "error_bound" not in lossy.columns:
        return pd.DataFrame()
    bounds = sorted(b for b in lossy["error_bound"].dropna().unique())
    if len(bounds) < 2:
        return pd.DataFrame()
    hold = ["chunk_uid", "lib_name", "preset", "shuffle"]
    ref = bounds[0]
    parts = []
    for b in bounds[1:]:
        m = _pair(lossy, "error_bound", ref, b, hold)
        if m.empty:
            continue
        parts.append(_deltas(m, f"eb_{ref:g}_to_{b:g}", "error_bound", ref, b))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def summarize_pairs(pairs: pd.DataFrame,
                    by: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-treatment (and optionally per-codec) effect size and win rate.

    Reports the MEDIAN delta rather than the mean as the headline: ratio
    deltas are heavy tailed (one chunk going from 3x to 300x moves a mean by
    more than the other forty chunks combined) and the sign test below is
    about the typical pair, not the loudest one.
    """
    if pairs.empty:
        return pd.DataFrame()
    keys = ["treatment"] + list(by or [])
    keys = [k for k in keys if k in pairs.columns]
    recs: List[dict] = []
    for key, g in pairs.groupby(keys, dropna=False):
        keys_t = key if isinstance(key, tuple) else (key,)
        rec: Dict[str, object] = dict(zip(keys, keys_t))
        rec["n_pairs"] = int(len(g))
        rec["n_chunks"] = int(g["chunk_uid"].nunique())
        for col, direction in DELTA_TARGETS:
            d = g.get(f"d_{col}")
            if d is None:
                continue
            d = pd.to_numeric(d, errors="coerce").replace(
                [np.inf, -np.inf], np.nan).dropna()
            if d.empty:
                continue
            rel = pd.to_numeric(g.get(f"rel_{col}"), errors="coerce").replace(
                [np.inf, -np.inf], np.nan).dropna()
            helps = g.get(f"helps_{col}")
            rec[f"{col}_n"] = int(d.size)
            rec[f"{col}_median_delta"] = float(d.median())
            rec[f"{col}_mean_delta"] = float(d.mean())
            rec[f"{col}_median_rel"] = float(rel.median()) if rel.size else np.nan
            rec[f"{col}_pct_helps"] = (float(100.0 * helps.mean())
                                       if helps is not None else np.nan)
            rec[f"{col}_p10_delta"] = float(d.quantile(0.10))
            rec[f"{col}_p90_delta"] = float(d.quantile(0.90))
            # Sign test against "the treatment does nothing" -- run on
            # CHUNKS, not on pairs.
            #
            # A pair is not an independent observation. With a full factorial
            # sweep, one chunk contributes a pair per (codec, quantize,
            # shuffle) combination, so a per-codec cell of "88 pairs" is 44
            # chunks seen twice and a pooled "704 pairs" is 44 chunks seen
            # sixteen times. Testing over pairs shrinks the p-value by orders
            # of magnitude for no added information: on the development log
            # it turned a shuffle effect that is p = 0.10 across chunks into
            # p = 8e-4 across pairs, and the report called it significant.
            #
            # Each chunk therefore votes once, with the sign of its MEDIAN
            # delta over its own pairs. Ties are dropped rather than counted
            # as successes, which is the conservative direction.
            if helps is not None and "chunk_uid" in g.columns:
                per_chunk = g.groupby("chunk_uid")[f"d_{col}"].median()
                per_chunk = per_chunk.replace(
                    [np.inf, -np.inf], np.nan).dropna()
                moved = per_chunk[per_chunk != 0]
                rec[f"{col}_pct_chunks_helped"] = (
                    float(100.0 * ((per_chunk > 0)
                                   if direction == "higher_is_better"
                                   else (per_chunk < 0)).mean())
                    if per_chunk.size else np.nan)
                if moved.size >= 5:
                    from scipy import stats as sps
                    k = int(np.sum(
                        (moved > 0) if direction == "higher_is_better"
                        else (moved < 0)))
                    rec[f"{col}_sign_test_p"] = float(
                        sps.binomtest(k, int(moved.size), 0.5).pvalue)
                    # Both numbers, so the printed p is reproducible from the
                    # printed counts -- they used to have different
                    # denominators (pairs for the percentage, moved-pairs for
                    # the test) and could not be reconciled by a reader.
                    rec[f"{col}_sign_test_n_chunks"] = int(moved.size)
                    rec[f"{col}_sign_test_k_chunks"] = k
        recs.append(rec)
    return pd.DataFrame(recs)


def benefit_regimes(pairs: pd.DataFrame, features: Sequence[str],
                    target: str = "ratio", nbins: int = 5) -> pd.DataFrame:
    """Where in property space does the treatment help, and where does it hurt?

    Bins each feature by quantile within a treatment and reports the median
    delta and win rate per bin, which is the form the report needs to say
    "shuffle helps when X" with a number attached.
    """
    if pairs.empty:
        return pd.DataFrame()
    col = f"d_{target}"
    if col not in pairs.columns:
        return pd.DataFrame()
    recs: List[pd.DataFrame] = []
    for treat, g in pairs.groupby("treatment"):
        for f in features:
            if f not in g.columns:
                continue
            d = g[[f, col, f"helps_{target}", "chunk_uid"]].replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(d) < nbins * 3 or d[f].nunique() < 2:
                continue
            try:
                b = pd.qcut(d[f], q=min(nbins, d[f].nunique()),
                            duplicates="drop")
            except ValueError:
                continue
            gb = d.groupby(b, observed=True)
            t = pd.DataFrame({
                "treatment": treat, "feature": f, "target": target,
                "bin": [str(i) for i in gb[col].median().index],
                "bin_left": [float(i.left) for i in gb[col].median().index],
                "bin_right": [float(i.right) for i in gb[col].median().index],
                "n_pairs": gb.size().to_numpy(),
                "n_chunks": gb["chunk_uid"].nunique().to_numpy(),
                "median_delta": gb[col].median().to_numpy(),
                "pct_helps": (100.0 * gb[f"helps_{target}"].mean()).to_numpy(),
            })
            t["verdict"] = np.select(
                [t["pct_helps"] >= 80, t["pct_helps"] >= 55,
                 t["pct_helps"] > 45, t["pct_helps"] > 20],
                ["strongly beneficial", "beneficial", "neutral",
                 "mildly harmful"], default="harmful")
            recs.append(t)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()

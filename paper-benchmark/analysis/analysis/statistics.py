#!/usr/bin/env python3
"""Correlations, binned/quantile trends, joint regimes and partial effects --
all of them computed at the level the question actually lives at.

THE REPLICATION TRAP, stated once and enforced everywhere below. A chunk
contributes one row per candidate configuration -- 32 in a full sweep -- and
all of them carry the SAME entropy/mad/second_deriv. A correlation computed
over rows therefore has n = 32 * n_chunks but only n_chunks independent draws
of the predictor, so its p-value is wrong by more than an order of magnitude.
Every function here either

  (a) aggregates to one row per chunk first, or
  (b) conditions on a fixed configuration -- one codec, one quantize, one
      shuffle -- which leaves exactly one row per chunk by construction,

and every result carries `n_chunks` alongside `n`, so the effective sample
size is visible at the point of use rather than in a footnote.

THE UNITS TRAP. `entropy` is bits/byte, bounded in [0,8] and comparable across
everything. `mad` and `second_deriv` are in RAW DATA UNITS
(data_stats_gpu_kernels.cu: mean|x - mean(x)|, and the mean |second
difference|), so a density field and a momentum field produce MAD values that
cannot be compared at all. Pooling them and correlating against ratio measures
the field labels, not the physics. Every pooled figure below is therefore
accompanied by its WITHIN-FIELD counterpart, and `simpson_flag` marks the
cases where the two disagree in sign.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

MIN_SUPPORT = 8   # below this a correlation or a bin is reported, never used


# --------------------------------------------------------------------------
# Correlation, done at chunk level
# --------------------------------------------------------------------------

def _corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    out = {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
           "spearman_rho": np.nan, "spearman_p": np.nan, "r2_linear": np.nan}
    # Uniqueness, not std: scipy tests for a literally constant input, and a
    # two-valued array with a tiny gap has std > 0 while still tripping its
    # ConstantInputWarning and returning NaN.
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return out
    pr, pp = sps.pearsonr(x, y)
    sr, sp = sps.spearmanr(x, y)
    # scipy's t-approximation diverges at |rho| = 1: it reports p ~ 1e-24 for
    # a perfect rank correlation over five points, where the exact two-sided
    # value under the null is 2/n!. Small-n perfect monotonicity is common
    # here -- a field's entropy rising over eight timesteps is exactly that --
    # so the approximation would turn a real but modest result into a
    # spectacular fake one.
    if n <= 12 and np.isfinite(sr) and abs(abs(float(sr)) - 1.0) < 1e-12:
        sp = min(1.0, 2.0 / float(math.factorial(n)))
    if n <= 12 and np.isfinite(pp) and pp < 1e-12:
        # Same guard for Pearson: a tiny p from a dozen points is the
        # approximation talking, not the data.
        sp_floor = 2.0 / float(math.factorial(n))
        pp = max(float(pp), min(1.0, sp_floor))
    out.update({"pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_rho": float(sr), "spearman_p": float(sp),
                "r2_linear": float(pr) ** 2})
    return out


def correlation_table(df: pd.DataFrame, features: Sequence[str],
                      targets: Sequence[str],
                      group_cols: Optional[Sequence[str]] = None,
                      label: str = "all") -> pd.DataFrame:
    """Feature x target correlations over the frame as given.

    The caller is responsible for having reduced `df` to one row per chunk per
    stratum; `n_chunks` is reported alongside `n` so a caller who did not is
    visible in the output rather than hidden by it.
    """
    rows: List[dict] = []
    groups = [((), df)] if not group_cols else list(
        df.groupby(list(group_cols), dropna=False))
    for key, g in groups:
        keys = key if isinstance(key, tuple) else (key,)
        for f, t in itertools.product(features, targets):
            if f not in g or t not in g:
                continue
            r = _corr(g[f].to_numpy(dtype=float), g[t].to_numpy(dtype=float))
            rec: Dict[str, object] = {"stratum": label, "feature": f,
                                      "target": t}
            if group_cols:
                rec.update(dict(zip(group_cols, keys)))
            rec.update(r)
            rec["n_chunks"] = (int(g["chunk_uid"].nunique())
                               if "chunk_uid" in g else r["n"])
            rec["usable"] = r["n"] >= MIN_SUPPORT
            rows.append(rec)
    return pd.DataFrame(rows)


def per_config_correlations(rows: pd.DataFrame, features: Sequence[str],
                            targets: Sequence[str],
                            min_chunks: int = MIN_SUPPORT) -> pd.DataFrame:
    """Correlations CONDITIONED on a single configuration.

    This is the clean version: fixing (lib_name, quantize, shuffle, preset)
    leaves exactly one row per chunk, so n == n_chunks and the p-value means
    what it says. It is also the only form in which "how sensitive is THIS
    codec to entropy" is a well-posed question -- pooled across codecs, the
    variance between codecs swamps the variance the feature explains.
    """
    out: List[pd.DataFrame] = []
    keys = [k for k in ("lib_name", "quantize", "shuffle", "preset")
            if k in rows.columns]
    for key, g in rows.groupby(keys, dropna=False):
        if g["chunk_uid"].nunique() < min_chunks:
            continue
        if len(g) != g["chunk_uid"].nunique():
            g = g.groupby("chunk_uid", as_index=False).first()
        t = correlation_table(g, features, targets, label="per_config")
        for k, v in zip(keys, key if isinstance(key, tuple) else (key,)):
            t[k] = v
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def within_field_correlation(rows: pd.DataFrame, features: Sequence[str],
                             targets: Sequence[str],
                             min_chunks: int = 5) -> pd.DataFrame:
    """Correlations computed INSIDE each physical field, then pooled.

    The pooled figure is a sample-size-weighted mean of the per-field Fisher
    z-transforms, back-transformed. This is the correlation that survives the
    units problem. A large naive |r| that the field-wise version does not
    reproduce is Simpson's paradox, and `simpson_flag` marks it.
    """
    recs: List[dict] = []
    grp = [c for c in ("lib_name", "quantize", "shuffle") if c in rows.columns]
    for key, g in rows.groupby(grp, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        for f, t in itertools.product(features, targets):
            if f not in g or t not in g:
                continue
            zs, ws, rs, flds = [], [], [], []
            n_perfect = 0
            for fld, gf in g.groupby("field", dropna=False):
                gf = gf.groupby("chunk_uid", as_index=False).first()
                if len(gf) < min_chunks:
                    continue
                c = _corr(gf[f].to_numpy(dtype=float),
                          gf[t].to_numpy(dtype=float))
                if not np.isfinite(c["pearson_r"]):
                    continue
                # A perfect |r| = 1 has infinite Fisher z. Dropping the field
                # silently biased the pooled estimate toward the weaker
                # fields, which is the wrong direction and invisible; clamp it
                # to the strongest representable correlation instead and count
                # how often that happened.
                r_i = float(c["pearson_r"])
                if abs(r_i) >= 1.0:
                    n_perfect += 1
                    r_i = np.sign(r_i) * (1.0 - 1e-6)
                zs.append(np.arctanh(r_i))
                ws.append(max(1, c["n"] - 3))
                rs.append(c["pearson_r"])
                flds.append(str(fld))
            if len(zs) < 2:
                continue
            pooled = float(np.tanh(float(np.average(zs, weights=ws))))
            gg = g.groupby("chunk_uid", as_index=False).first()
            naive = _corr(gg[f].to_numpy(dtype=float),
                          gg[t].to_numpy(dtype=float))
            rec = dict(zip(grp, keys))
            rec.update({
                "feature": f, "target": t,
                "pooled_within_field_r": pooled,
                "naive_pooled_r": naive["pearson_r"],
                "n_fields": len(zs), "fields": "|".join(flds),
                "n_fields_with_perfect_r": n_perfect,
                "n_fields_agreeing_sign": int(sum(
                    1 for r in rs if np.sign(r) == np.sign(pooled))),
                "min_field_r": float(np.min(rs)),
                "max_field_r": float(np.max(rs)),
                "n_chunks": int(gg["chunk_uid"].nunique()),
                "simpson_flag": bool(
                    np.isfinite(naive["pearson_r"])
                    and np.sign(naive["pearson_r"]) != np.sign(pooled)
                    and abs(naive["pearson_r"]) > 0.2),
            })
            recs.append(rec)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# Binned / quantile trends
# --------------------------------------------------------------------------

def binned_trend(df: pd.DataFrame, feature: str, target: str,
                 nbins: int = 8, by_quantile: bool = True,
                 min_support: int = 3) -> pd.DataFrame:
    """Median target per feature bin, with support.

    Quantile bins by default: entropy and especially MAD are strongly skewed
    on real field data, and equal-width bins put almost every chunk in one of
    them and then report a "trend" across bins holding 1 sample each.
    """
    if feature not in df or target not in df:
        return pd.DataFrame()
    d = df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < max(nbins, 4):
        return pd.DataFrame()
    try:
        if by_quantile:
            b = pd.qcut(d[feature], q=min(nbins, d[feature].nunique()),
                        duplicates="drop")
        else:
            b = pd.cut(d[feature], bins=nbins)
    except ValueError:
        return pd.DataFrame()
    g = d.groupby(b, observed=True)[target]
    idx = list(g.median().index)
    if len(idx) < 2:
        return pd.DataFrame()
    out = pd.DataFrame({
        "feature": feature, "target": target,
        "bin": [str(i) for i in idx],
        "bin_left": [float(i.left) for i in idx],
        "bin_right": [float(i.right) for i in idx],
        "n": g.size().to_numpy(),
        "median": g.median().to_numpy(),
        "mean": g.mean().to_numpy(),
        "p10": g.quantile(0.10).to_numpy(),
        "p90": g.quantile(0.90).to_numpy(),
    })
    out["usable"] = out["n"] >= min_support
    return out


def monotonicity(trend: pd.DataFrame) -> dict:
    """Is the binned trend monotone, and where does it flatten?

    `saturation_bin_left` is the feature value from which the median stops
    moving by more than 5% of the total span -- the operational answer to
    "where does this feature stop being predictive".
    """
    if trend.empty or int(trend["usable"].sum()) < 3:
        return {}
    t = trend[trend["usable"]]
    med = t["median"].to_numpy(dtype=float)
    if not np.isfinite(med).all() or med.size < 3:
        return {}
    span = float(np.nanmax(med) - np.nanmin(med))
    diffs = np.diff(med)
    out: Dict[str, object] = {
        "n_bins": int(med.size),
        "monotone_decreasing": bool(np.all(diffs <= 0)),
        "monotone_increasing": bool(np.all(diffs >= 0)),
        "sign_changes": int(np.sum(np.diff(np.sign(diffs)) != 0)),
        "span": span,
        "first_bin_median": float(med[0]), "last_bin_median": float(med[-1]),
    }
    if span > 0:
        rel = np.abs(diffs) / span
        for i in range(len(rel)):
            if bool(np.all(rel[i:] < 0.05)):
                out["saturation_bin_index"] = int(i)
                out["saturation_bin_left"] = float(t["bin_left"].iloc[i])
                break
    return out


# --------------------------------------------------------------------------
# Joint regimes and partial effects
# --------------------------------------------------------------------------

def joint_regimes(chunks: pd.DataFrame, features: Sequence[str], target: str,
                  n_per: int = 2, min_support: int = 4) -> pd.DataFrame:
    """Each feature split into levels, crossed, with the target inside each
    cell. Descriptive, not a model: it exists so the report can say what a
    regime IS, in feature units, rather than only that one exists."""
    feats = [f for f in features if f in chunks.columns]
    d = chunks.dropna(subset=feats + [target]).copy()
    if d.empty:
        return pd.DataFrame()
    if len(d) < (n_per ** len(feats)) * min_support:
        n_per = 2
    names = {2: ["low", "high"], 3: ["low", "mid", "high"]}[n_per]
    labs = []
    for f in feats:
        try:
            d[f + "_lvl"] = pd.qcut(d[f], q=n_per, labels=names,
                                    duplicates="drop")
        except ValueError:
            d[f + "_lvl"] = "all"
        labs.append(f + "_lvl")
    g = d.groupby(labs, observed=True)[target]
    out = g.agg(["size", "median", "mean", "std"]).reset_index()
    out = out.rename(columns={"size": "n"})
    out["usable"] = out["n"] >= min_support
    out["target"] = target
    return out


def conditional_gain(chunks: pd.DataFrame, base: Sequence[str], extra: str,
                     target: str) -> dict:
    """Does `extra` add anything to `base` for `target`, at chunk level?

    The semi-partial correlation of `extra` with `target` after both are
    residualised on `base` by OLS, plus the R^2 the extra feature buys. This
    is the linear, interpretable half of the ablation question; the nonlinear
    half is modeling.feature_ablation, and the two are reported together
    because a feature can add nothing linearly and a great deal in a tree.
    """
    base = [b for b in base if b in chunks.columns]
    cols = list(base) + [extra, target]
    if extra not in chunks.columns or target not in chunks.columns:
        return {"insufficient": True, "reason": "column absent"}
    d = chunks[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < MIN_SUPPORT + len(base) + 1:
        return {"n": int(len(d)), "insufficient": True}
    X = np.c_[np.ones(len(d)), d[base].to_numpy(dtype=float)] if base \
        else np.ones((len(d), 1))
    y = d[target].to_numpy(dtype=float)
    xe = d[extra].to_numpy(dtype=float)

    def resid(v):
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ beta

    ry, rx = resid(y), resid(xe)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"n": int(len(d)), "base": base, "extra": extra,
                "target": target, "partial_r": 0.0, "partial_p": 1.0,
                "delta_r2": 0.0, "note": "no residual variance"}
    r, p = sps.pearsonr(rx, ry)
    full = np.c_[X, xe]
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    b1, *_ = np.linalg.lstsq(full, y, rcond=None)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2_0 = 1 - float(np.sum((y - X @ b0) ** 2)) / sst if sst else np.nan
    r2_1 = 1 - float(np.sum((y - full @ b1) ** 2)) / sst if sst else np.nan
    return {"n": int(len(d)), "base": base, "extra": extra, "target": target,
            "partial_r": float(r), "partial_p": float(p),
            "r2_base": float(r2_0), "r2_with_extra": float(r2_1),
            "delta_r2": float(r2_1 - r2_0)}


def describe(series: pd.Series) -> dict:
    v = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(v.median()), "std": float(v.std()),
            "min": float(v.min()), "p10": float(v.quantile(0.10)),
            "p90": float(v.quantile(0.90)), "max": float(v.max())}


# --------------------------------------------------------------------------
# The property x outcome matrix
# --------------------------------------------------------------------------

#: What the simulation hands the compressor, per chunk, before any transform.
PROPERTY_COLS = ["entropy", "mad", "second_deriv", "data_range", "timestep"]
#: What came out. Lossless and quantized ratio are kept APART: they answer
#: different questions and mixing them lets the bound masquerade as a data
#: property.
OUTCOME_COLS = ["best_lossless_ratio", "best_quantized_ratio",
                "fastest_ct_ms", "fastest_dt_ms", "quantized_ssim"]

_MATRIX_LABELS = {
    "entropy": "byte entropy", "mad": "MAD", "second_deriv": "2nd difference",
    "data_range": "data range", "timestep": "timestep",
    "best_lossless_ratio": "best lossless ratio",
    "best_quantized_ratio": "best quantized ratio",
    "fastest_ct_ms": "fastest compress (ms)",
    "fastest_dt_ms": "fastest decompress (ms)",
    "quantized_ssim": "SSIM (quantized)",
}


def property_outcome_frame(chunks: pd.DataFrame,
                           rows: pd.DataFrame) -> pd.DataFrame:
    """One row per chunk: every property beside every outcome.

    Outcomes come from the ROW frame so lossless and quantized can be split:
    `best_ratio_value` on the chunk frame is the best over all 32 candidates
    and would let a quantized 6000x stand in for the data's compressibility.
    """
    if "chunk_uid" not in chunks.columns or "chunk_uid" not in rows.columns:
        return pd.DataFrame()
    props = [c for c in PROPERTY_COLS if c in chunks.columns]
    d = chunks[["chunk_uid"] + props].copy()
    r = rows.replace([np.inf, -np.inf], np.nan)
    if "quantize" in r.columns and "ratio" in r.columns:
        ll = r[r["quantize"] == 0].groupby("chunk_uid")["ratio"].max()
        qz = r[r["quantize"] == 1].groupby("chunk_uid")["ratio"].max()
        d["best_lossless_ratio"] = d["chunk_uid"].map(ll)
        d["best_quantized_ratio"] = d["chunk_uid"].map(qz)
    for src, dst in (("fastest_ct_value", "fastest_ct_ms"),
                     ("fastest_dt_value", "fastest_dt_ms")):
        if src in chunks.columns:
            d[dst] = chunks[src].to_numpy()
    if "meas_ssim" in r.columns and "quantize" in r.columns:
        ss = pd.to_numeric(r.loc[r["quantize"] == 1, "meas_ssim"],
                           errors="coerce")
        d["quantized_ssim"] = d["chunk_uid"].map(
            ss.groupby(r.loc[ss.index, "chunk_uid"]).median())
    return d


def property_outcome_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Spearman over every pair, long form, with the support of each pair.

    Spearman throughout: every one of these relationships is monotone and
    heavy-tailed (ratio spans four decades, MAD seven), so a Pearson r would
    be set by the handful of extreme chunks and read as a fit to outliers.
    """
    cols = [c for c in PROPERTY_COLS + OUTCOME_COLS if c in frame.columns]
    recs = []
    for i, a in enumerate(cols):
        for b in cols[i:]:
            x = pd.to_numeric(frame[a], errors="coerce")
            y = pd.to_numeric(frame[b], errors="coerce")
            m = x.notna() & y.notna()
            n = int(m.sum())
            if a == b:
                rho, p = 1.0, 0.0
            elif n < 8 or x[m].nunique() < 2 or y[m].nunique() < 2:
                rho, p = float("nan"), float("nan")
            else:
                res = sps.spearmanr(x[m], y[m])
                rho, p = float(res.correlation), float(res.pvalue)
            recs.append({"a": a, "b": b, "a_label": _MATRIX_LABELS.get(a, a),
                         "b_label": _MATRIX_LABELS.get(b, b),
                         "a_kind": "property" if a in PROPERTY_COLS else "outcome",
                         "b_kind": "property" if b in PROPERTY_COLS else "outcome",
                         "spearman_rho": rho, "p": p, "n": n})
    return pd.DataFrame(recs)

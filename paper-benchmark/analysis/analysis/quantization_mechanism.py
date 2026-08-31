#!/usr/bin/env python3
"""WHY a quantized chunk compresses -- and degrades -- the way it does.

The entropy-bound section (mechanism.py) explains the LOSSLESS rows in closed
form and then stops, because the three logged features describe the ORIGINAL
buffer and quantization replaces it. That leaves the quantized half of every
sweep -- the half the compression story is actually about -- with a
correlation and no cause.

This module supplies the cause, from the quantizer's own arithmetic. The
device quantizer (data_stats_gpu_kernels.cu, QuantizeDevice) maps every value
onto a grid of spacing 2*eb_eff anchored at data_min, so the number of
distinct values a chunk can still take after quantization is

    L  =  data_range / (2 * eb_eff)              "levels"

and L is a closed-form function of two things the log records: the chunk's
range and the configured bound. Nothing about the codec enters it. L then sets
both outcomes at once, in opposite directions:

  * RATIO. A stream drawn from an alphabet of L symbols carries at most
    log2(L) bits per element; the original carried 32. So an ideal
    order-blind entropy coder reaches AT LEAST

        alphabet_floor  =  32 / log2(L)

    from the alphabet alone, before any order is read -- and that floor is
    a lower bound because equiprobable levels maximise entropy and real
    fields are peaked. The quantizer also picks the storage width from L
    (int8 below ~115 levels, int16 below ~29,800, else int32), so

        width_gain  =  4 / width_bytes

    is an exact, separately attributable part of the ratio.

  * QUALITY. With L levels the reconstruction can distinguish at most L
    states of the field. Below ~10 the field's structure no longer fits
    between the rungs, SSIM collapses towards 0, and the chunk is stored as
    something close to a constant whatever the codec. Above a few hundred
    the loss is invisible at SSIM's resolution. The transition is sharp and
    it is the same L that drove the ratio, which is what "ratio bought by
    discarding physics" means quantitatively.

The regime table binds the two: chunks are binned by L, and each bin reports
its ratio, its SSIM and its dominant codec together with the one-line
mechanism that produced them. That table is the answer to "why does THIS data
with THESE properties compress THIS way" for the lossy rows -- the question the
rest of the report can only answer for the lossless ones.

L counts the rungs the chunk's RANGE spans. The ratio, though, is set by how
many rungs the BULK of the values occupy, and on a field whose range is made
by a rare extreme -- a blast hot-spot in a cold box -- those are very
different numbers. The log carries the bulk's spread too, and the fix is one
substitution:

    L_bulk  =  L * (mad / data_range)  =  mad / (2 * eb_eff)    "bulk levels"

the mean absolute deviation measured in grid steps: how many rungs the
typical value sits from the mean. It is still closed-form, still codec-free,
and it no longer needs the range at all. Measured on Nyx, L_bulk tracks the
ratio at rho = -0.90 where L alone reaches -0.75, and it turns the
high-L tail -- where L is large but the ratio climbs BACK up because almost
every value sits on the same few rungs near data_min -- from a contradiction
(rho +0.52 within that band) into the same mechanism (rho -0.96 against the
concentration mad/range). L still owns what the range decides: the storage
width, the PSNR, and the regime boundaries.

PSNR is reported but is not evidence: the quantizer's max error saturates at
0.95*eb (measured: median 0.950, p95 0.950), so rmse is ~constant * eb and
psnr = 20*log10(range / rmse) is a function of L by construction. Its
correlation with L is a check that the arithmetic is right, and nothing more.
SSIM is the informative quality number.

Two honesty checks are built in, because the arithmetic above assumes the
bound the caller asked for is the bound the quantizer used:

  * BOUND RELAXED. QuantizeDevice widens eb when float32 cannot represent it
    (eb - max|v|*2.4e-7 - 0.05*eb <= 0). The log does not carry the effective
    bound, but it carries the MEASURED max error, which exceeds the requested
    eb exactly when relaxation happened. Those chunks are flagged, their L is
    reported as an UPPER bound (the real grid is coarser), and the fraction is
    surfaced -- on a WarpX log it is 60%, which is a finding about the bound,
    not about the compressor.
  * RANGE SOURCE. data_range comes from selection.csv.quality when it is
    beside the log. Where it is absent, the identity psnr = 20*log10(range /
    rmse) recovers it from any measured row with 0 < psnr < 120 (verified to
    5e-8 dB against the kernel on 50k rows). The source is recorded per chunk.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

#: Below this many levels a field's structure cannot be represented; SSIM
#: collapses. Measured on VPIC at eb=0.05: median SSIM 0.25 with 93.7% of
#: chunks under 10 levels, against 0.9998 at eb=1e-3 with 18.8% under.
FEW_LEVELS = 10.0
#: The quantizer's own precision thresholds (quantization.h:
#: ComputeRequiredPrecision), including its 10% safety margin.
_PREC_MARGIN = 1.1
_INT8_MAX_BINS = 127.0
_INT16_MAX_BINS = 32767.0
#: Effective bound as a fraction of the requested one when float32 can
#: represent it: QuantizeDevice subtracts a 5% safety margin.
_EFF_FRACTION = 0.95
#: A measured max error this far above the requested bound means the
#: quantizer relaxed it. 1.001, not 1.0: the kernel's max error saturates at
#: 0.95*eb, so anything over eb at all is relaxation, and the tolerance only
#: guards float printing.
_RELAX_TOL = 1.001
MIN_ROWS = 8

#: L bands, lower-inclusive, with the mechanism each one carries. Labels are
#: what the report prints, so they are written for a reader, not a parser.
REGIMES: List[tuple] = [
    (0.0, 1.5, "constant",
     "every value lands on one grid point: the chunk is stored as a single "
     "level and the ratio is whatever the codec's run-length path reaches "
     "(the cap). No information about the field survives, but there was "
     "none in the chunk to begin with -- its whole range is under the bound."),
    (1.5, FEW_LEVELS, "few levels",
     "the field is reduced to a handful of values. The alphabet floor "
     "alone is >10x, so the ratio is high regardless of codec -- and SSIM "
     "collapses, because a wave cannot be drawn with three heights. This is "
     "the regime where ratio is bought by discarding the physics."),
    (FEW_LEVELS, 256.0, "byte alphabet",
     "L fits in one byte, so the quantizer emits int8 and width_gain is an "
     "exact 4x before the codec runs. Structure survives at SSIM's "
     "resolution; the remaining ratio is alphabet skew plus order."),
    (256.0, np.inf, "fine",
     "L needs int16 or int32, so the width gain is 2x or none and the "
     "alphabet floor from L is small -- yet the ratio is often HIGH here, "
     "and that is not a contradiction: this regime is where the range is "
     "made by a rare extreme (a blast hot-spot in a cold box), so L counts "
     "rungs the bulk never visits. Read L_bulk instead; it is small in "
     "exactly these chunks and the ratio follows it. Quality is effectively "
     "lossless at SSIM's resolution because the bulk sits within a few grid "
     "steps of its mean."),
]


def _spear(x: pd.Series, y: pd.Series) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return float("nan")
    return float(sps.spearmanr(x[m], y[m]).correlation)


def _width_bytes(levels: np.ndarray) -> np.ndarray:
    """ComputeRequiredPrecision, vectorised: bins = L * 1.1."""
    nb = levels * _PREC_MARGIN
    return np.where(nb <= _INT8_MAX_BINS, 1,
                    np.where(nb <= _INT16_MAX_BINS, 2, 4)).astype(int)


def per_chunk(rows: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    """One row per (chunk, error bound): L and everything derived from it.

    Uses `data_range` on `chunks` when the loader joined it from the quality
    sidecar; otherwise recovers it from psnr/rmse on the chunk's own measured
    rows and says so in `range_source`.
    """
    need = {"quantize", "ratio", "chunk_uid"}
    if not need <= set(rows.columns) or "error_bound" not in rows.columns:
        return pd.DataFrame()
    q = rows[(rows["quantize"] == 1) & rows["error_bound"].notna()
             & (rows["error_bound"] > 0)].copy()
    if q.empty:
        return pd.DataFrame()

    # ---- data_range: sidecar first, psnr/rmse identity second ------------
    rng = pd.Series(np.nan, index=q.index, dtype=float)
    src = pd.Series("none", index=q.index, dtype=object)
    if "data_range" in chunks.columns:
        m = chunks.set_index("chunk_uid")["data_range"]
        rng = q["chunk_uid"].map(m).astype(float)
        src = np.where(rng.notna(), "sidecar", "none")
        src = pd.Series(src, index=q.index)
    if {"meas_psnr_db", "meas_rmse"} <= set(q.columns):
        p = pd.to_numeric(q["meas_psnr_db"], errors="coerce")
        r = pd.to_numeric(q["meas_rmse"], errors="coerce")
        ok = rng.isna() & p.notna() & r.notna() & (r > 0) & (p > 0) & (p < 120)
        rng = rng.where(~ok, r * np.power(10.0, p / 20.0))
        src = src.where(~ok, "recovered from psnr/rmse")
    q["data_range"] = rng
    q["range_source"] = src
    q = q[q["data_range"].notna() & (q["data_range"] >= 0)]
    if q.empty:
        return pd.DataFrame()

    # ---- the closed form ---------------------------------------------------
    eb = q["error_bound"].astype(float)
    eb_eff = _EFF_FRACTION * eb
    q["levels"] = np.maximum(q["data_range"] / (2.0 * eb_eff), 1.0)
    # The bulk's spread in grid steps. mad is a chunk property the loader
    # carries on `chunks`; a log without it gets NaN here and the L-only
    # analysis still runs.
    if "mad" in chunks.columns:
        madm = chunks.set_index("chunk_uid")["mad"]
        mad = pd.to_numeric(q["chunk_uid"].map(madm), errors="coerce")
        q["concentration"] = np.where(q["data_range"] > 0,
                                      mad / q["data_range"], np.nan)
        q["bulk_levels"] = np.maximum(mad / (2.0 * eb_eff), 1e-3)
    else:
        q["concentration"] = np.nan
        q["bulk_levels"] = np.nan
    q["width_bytes"] = _width_bytes(q["levels"].to_numpy())
    q["width_gain"] = 4.0 / q["width_bytes"]
    bits = np.log2(q["levels"])
    with np.errstate(divide="ignore"):
        q["alphabet_floor_ratio"] = np.where(bits > 0, 32.0 / bits, np.inf)
    q["ratio_over_alphabet_floor"] = np.where(
        np.isfinite(q["alphabet_floor_ratio"]),
        q["ratio"] / q["alphabet_floor_ratio"], np.nan)

    # ---- was the bound honoured? -----------------------------------------
    if "meas_max_error" in q.columns:
        me = pd.to_numeric(q["meas_max_error"], errors="coerce")
        q["bound_relaxed"] = (me > _RELAX_TOL * eb).fillna(False)
    else:
        q["bound_relaxed"] = False

    # One row per (chunk, bound): the codec-independent quantities are
    # identical across the chunk's 16 quantized candidates by construction,
    # and the outcomes are summarised.
    keys = ["chunk_uid", "error_bound"]
    agg: Dict[str, object] = {
        "data_range": "first", "range_source": "first", "levels": "first",
        "concentration": "first", "bulk_levels": "first",
        "width_bytes": "first", "width_gain": "first",
        "alphabet_floor_ratio": "first", "bound_relaxed": "max",
        "ratio": "max",
    }
    for c in ("meas_ssim", "meas_psnr_db", "meas_max_error", "field",
              "timestep", "blob"):
        if c in q.columns:
            agg[c] = "first" if c in ("field", "timestep", "blob") else "median"
    g = q.groupby(keys, as_index=False).agg(agg)
    g = g.rename(columns={"ratio": "best_quantized_ratio"})
    g["best_ratio_over_alphabet_floor"] = np.where(
        np.isfinite(g["alphabet_floor_ratio"]),
        g["best_quantized_ratio"] / g["alphabet_floor_ratio"], np.nan)
    # Which codec reached that best ratio.
    idx = q.groupby(keys)["ratio"].idxmax()
    g["best_lib"] = q.loc[idx.to_numpy(), "lib_name"].to_numpy() \
        if "lib_name" in q.columns else None
    g["regime"] = pd.cut(
        g["levels"], bins=[r[0] for r in REGIMES] + [np.inf],
        labels=[r[2] for r in REGIMES], right=False,
        include_lowest=True).astype(str)
    return g


def regime_table(pc: pd.DataFrame) -> pd.DataFrame:
    """The regimes, each with its outcomes and its mechanism in one row."""
    if pc.empty:
        return pd.DataFrame()
    mech = {r[2]: r[3] for r in REGIMES}
    recs = []
    n_all = len(pc)
    for lo, hi, name, _ in REGIMES:
        g = pc[pc["regime"] == name]
        if g.empty:
            continue
        rec = {
            "regime": name,
            "levels_from": lo, "levels_to": hi,
            "n_chunks": int(len(g)),
            "pct_of_chunks": float(100.0 * len(g) / n_all),
            "median_levels": float(g["levels"].median()),
            "median_width_bytes": float(g["width_bytes"].median()),
            "median_alphabet_floor": float(
                g["alphabet_floor_ratio"].replace(np.inf, np.nan).median()),
            "median_best_ratio": float(g["best_quantized_ratio"].median()),
            "pct_bound_relaxed": float(100.0 * g["bound_relaxed"].mean()),
        }
        for c in ("meas_ssim", "meas_psnr_db"):
            if c in g.columns:
                rec[f"median_{c}"] = float(g[c].median())
        if "best_lib" in g.columns and g["best_lib"].notna().any():
            vc = g["best_lib"].value_counts()
            rec["dominant_codec"] = str(vc.index[0])
            rec["dominant_codec_share"] = float(100.0 * vc.iloc[0] / len(g))
        rec["mechanism"] = mech[name]
        recs.append(rec)
    return pd.DataFrame(recs)


def by_field(pc: pd.DataFrame) -> pd.DataFrame:
    """Median L per field at each bound: the table that says which fields a
    bound destroys, before any codec is consulted."""
    if pc.empty or "field" not in pc.columns:
        return pd.DataFrame()
    recs = []
    for (f, eb), g in pc.groupby(["field", "error_bound"], sort=True):
        rec = {
            "field": f, "error_bound": float(eb), "n_chunks": int(len(g)),
            "median_data_range": float(g["data_range"].median()),
            "median_levels": float(g["levels"].median()),
            "pct_under_few_levels": float(
                100.0 * (g["levels"] < FEW_LEVELS).mean()),
            "median_best_ratio": float(g["best_quantized_ratio"].median()),
            "pct_bound_relaxed": float(100.0 * g["bound_relaxed"].mean()),
        }
        for c in ("meas_ssim", "meas_psnr_db"):
            if c in g.columns:
                rec[f"median_{c}"] = float(g[c].median())
        recs.append(rec)
    out = pd.DataFrame(recs)
    return out.sort_values(["error_bound", "median_levels"]) \
        if not out.empty else out


def levels_drive_outcomes(pc: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """How tightly L alone determines each outcome, overall and per codec.

    Spearman, because the relationships are monotone and heavy-tailed. A
    strong negative rho with ratio and a strong positive one with SSIM is the
    mechanism confirmed: one closed-form number moves both outcomes.
    """
    if pc.empty:
        return pd.DataFrame()
    recs = []
    base = pc[~pc["bound_relaxed"]] if "bound_relaxed" in pc.columns else pc
    has_bulk = "bulk_levels" in base.columns and base["bulk_levels"].notna().any()
    for target, col in (("best quantized ratio", "best_quantized_ratio"),
                        ("SSIM", "meas_ssim"),
                        ("PSNR (dB) -- follows L by construction",
                         "meas_psnr_db")):
        if col not in base.columns:
            continue
        rec = {"scope": "all codecs", "lib_name": "(best per chunk)",
               "target": target, "n_chunks": int(base[col].notna().sum()),
               "spearman_rho_vs_levels":
                   _spear(np.log10(base["levels"]), base[col])}
        if has_bulk:
            rec["spearman_rho_vs_bulk_levels"] = _spear(
                np.log10(base["bulk_levels"]), base[col])
        recs.append(rec)
    # Per codec, on the rows themselves.
    q = rows[(rows["quantize"] == 1)].copy() if "quantize" in rows else pd.DataFrame()
    if not q.empty and "lib_name" in q.columns:
        idx = pc.set_index(["chunk_uid", "error_bound"])
        key = list(zip(q["chunk_uid"], q["error_bound"]))
        q["levels"] = [idx["levels"].get(k, np.nan) for k in key]
        q["bound_relaxed"] = [bool(idx["bound_relaxed"].get(k, False))
                              for k in key]
        if has_bulk:
            q["bulk_levels"] = [idx["bulk_levels"].get(k, np.nan) for k in key]
        q = q[q["levels"].notna() & ~q["bound_relaxed"]]
        for lib, g in q.groupby("lib_name"):
            if len(g) < MIN_ROWS:
                continue
            rec = {"scope": "per codec", "lib_name": lib,
                   "target": "ratio", "n_chunks": int(len(g)),
                   "spearman_rho_vs_levels":
                       _spear(np.log10(g["levels"]), g["ratio"])}
            if has_bulk:
                rec["spearman_rho_vs_bulk_levels"] = _spear(
                    np.log10(g["bulk_levels"]), g["ratio"])
            recs.append(rec)
    return pd.DataFrame(recs)


def summarize(pc: pd.DataFrame, reg: pd.DataFrame, drv: pd.DataFrame,
              fld: pd.DataFrame) -> Dict[str, object]:
    """The verdict, in the form the report and the answers both read."""
    out: Dict[str, object] = {"available": not pc.empty}
    if pc.empty:
        out["reason"] = ("no quantized rows with a recoverable data range: "
                         "the quality sidecar is absent and no measured row "
                         "carries a finite psnr/rmse pair to recover it from")
        return out
    out["n_chunks"] = int(len(pc))
    out["bounds"] = sorted(float(b) for b in pc["error_bound"].unique())
    out["pct_range_from_sidecar"] = float(
        100.0 * (pc["range_source"] == "sidecar").mean())
    out["pct_bound_relaxed"] = float(100.0 * pc["bound_relaxed"].mean())
    # Descriptive numbers over the chunks whose bound was HONOURED: a relaxed
    # chunk's L is an upper bound computed from a bound the quantizer did not
    # use, and on a WarpX log it put the median at 2.6e11.
    hon = pc[~pc["bound_relaxed"]] if pc["bound_relaxed"].any() else pc
    out["n_chunks_bound_honoured"] = int(len(hon))
    out["median_levels"] = float(hon["levels"].median())
    out["pct_under_few_levels"] = float(
        100.0 * (hon["levels"] < FEW_LEVELS).mean())
    out["pct_constant"] = float(100.0 * (hon["regime"] == "constant").mean())
    if not drv.empty:
        for _, r in drv[drv["scope"] == "all codecs"].iterrows():
            t = str(r["target"])
            key = ("ratio" if t.startswith("best") else
                   "ssim" if t == "SSIM" else "psnr")
            out[f"rho_levels_vs_{key}"] = float(r["spearman_rho_vs_levels"])
            if "spearman_rho_vs_bulk_levels" in r and np.isfinite(
                    r["spearman_rho_vs_bulk_levels"]):
                out[f"rho_bulk_levels_vs_{key}"] = float(
                    r["spearman_rho_vs_bulk_levels"])
    if not reg.empty:
        worst = reg.sort_values("median_levels").iloc[0]
        out["regimes"] = reg[["regime", "n_chunks", "pct_of_chunks",
                              "median_levels", "median_best_ratio"]
                             + [c for c in ("median_meas_ssim",)
                                if c in reg.columns]].to_dict("records")
    if not fld.empty:
        destroyed = fld[fld["pct_under_few_levels"] >= 50.0]
        out["fields_destroyed"] = destroyed["field"].astype(str).tolist()
        out["n_fields"] = int(fld["field"].nunique())
    # The one-paragraph reading.
    rho_r = out.get("rho_levels_vs_ratio")
    rho_s = out.get("rho_levels_vs_ssim")
    brho_r = out.get("rho_bulk_levels_vs_ratio")
    brho_s = out.get("rho_bulk_levels_vs_ssim")
    parts = [
        f"L = data_range / (2 * 0.95 * eb) is a closed-form function of the "
        f"chunk's range and the configured bound, with no codec in it. "
        f"Across {out['n_chunks']} quantized chunks it"]
    if rho_r is not None and np.isfinite(rho_r):
        parts.append(f" tracks the best achievable ratio at Spearman rho = "
                     f"{rho_r:+.2f}")
    if rho_s is not None and np.isfinite(rho_s):
        parts.append(f" and SSIM at rho = {rho_s:+.2f}")
    if brho_r is not None and np.isfinite(brho_r):
        parts.append(f". Counting only the rungs the BULK occupies -- "
                     f"L_bulk = mad / (2 * 0.95 * eb), the mean absolute "
                     f"deviation in grid steps -- tightens that to rho = "
                     f"{brho_r:+.2f} on the ratio"
                     + (f" and {brho_s:+.2f} on SSIM" if brho_s is not None
                        and np.isfinite(brho_s) else "")
                     + ": a rare extreme widens the range without filling "
                       "the grid, and the codec only sees the filled part")
    parts.append(f". {out['pct_under_few_levels']:.0f}% of chunks are left "
                 f"with fewer than {FEW_LEVELS:.0f} levels")
    if out.get("fields_destroyed"):
        parts.append(f", and in {len(out['fields_destroyed'])} of "
                     f"{out.get('n_fields', '?')} fields that is the majority "
                     f"({', '.join(out['fields_destroyed'][:6])}"
                     f"{', ...' if len(out['fields_destroyed']) > 6 else ''})")
    parts.append(".")
    if out["pct_bound_relaxed"] > 0:
        parts.append(f" On {out['pct_bound_relaxed']:.0f}% of chunks the "
                     f"measured error exceeds the requested bound: the "
                     f"quantizer relaxed it because float32 cannot represent "
                     f"the bound at the field's magnitude, so for those L is "
                     f"an upper bound and the stored error is not the one "
                     f"asked for.")
    out["verdict"] = "".join(parts)
    return out

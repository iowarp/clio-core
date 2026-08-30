#!/usr/bin/env python3
"""Column contract for a Clio-NeuroPress exploration log, and the audit that
checks a CSV against it.

Every definition here was read out of the code that WRITES the log, not
inferred from the column name:

  writer          context-transfer-engine/compressor/src/neuropress_telemetry.cc
                    (LogNeuroPressExplore, the header string)
  call site       context-transfer-engine/compressor/src/compressor_runtime.cc
                    (the exploration sweep, ~L1900-2420)
  cost model      .../include/clio_cte/compressor/models/neuropress_cost.h
  data statistics context-transport-primitives/src/compress/preprocess/
                    data_stats_gpu_kernels.cu

Two things in that list are load-bearing and are the reason this file exists
rather than a dict of dtypes:

  * `cost` is LOWER-IS-BETTER and its weights are RUNTIME-CONFIGURABLE
    (CLIO_NEUROPRESS_COST_W_*), so no analysis may assume the defaults. See
    cost_analysis.infer_cost_model, which recovers them from the log itself.
  * `eb_encoded` is the MODEL'S INPUT, not the configured error bound: it
    carries a 1e-7 sentinel on every lossless row. Reading it as an error
    bound invents a 1e-7 bound for lossless data.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    name: str
    kind: str            # "id" | "property" | "config" | "outcome" | "prediction"
    required: bool
    dtype: str
    meaning: str
    #: value that means "this was not measured", distinct from a measured 0
    sentinel: Optional[float] = None


COLUMNS: List[Column] = [
    # ---- identity -------------------------------------------------------
    Column("seq", "id", True, "int64",
           "Completion order of the row, not chunk order. With online "
           "learning on it is the order the model was updated in."),
    Column("blob", "id", True, "string",
           "Runtime blob key. Identifies the CHUNK: every row sharing a blob "
           "describes the SAME bytes under a different configuration."),
    Column("chunk_bytes", "property", True, "int64",
           "Uncompressed size of the chunk in bytes."),
    Column("role", "id", True, "string",
           "'primary' = the model's own pick (rank -1, never a swept slot); "
           "'alt' = a measured sweep candidate."),
    Column("rank", "id", True, "int64",
           "Sweep rank among SUCCESSFUL candidates (0 = model's first "
           "alternative). -1 on the primary row."),

    # ---- configuration --------------------------------------------------
    Column("lib_name", "config", True, "string", "Codec library name."),
    Column("algo_idx", "config", False, "int64",
           "NeuroPress's 0-7 action index, recovered through the ML base id. "
           "-1 when the codec is outside the model's action space."),
    Column("preset", "config", True, "int64",
           "Codec preset id (low byte of the packed preset word)."),
    Column("quantize", "config", True, "int64",
           "1 when linear quantization was APPLIED (not merely requested)."),
    Column("shuffle", "config", True, "int64",
           "Byte-shuffle WIDTH actually applied (0, 2, 4 or 8) -- a width, "
           "not a boolean. 0 means no shuffle ran."),
    Column("eb_encoded", "config", False, "float64",
           "The model's INPUT 3, not the configured error bound: "
           "`quantize ? error_bound : 1e-7`. The 1e-7 is a training-time "
           "sentinel (configs.py: eb_val = eb if quant else 1e-7). A real "
           "error bound is recoverable only from rows with quantize == 1."),

    # ---- intrinsic data properties (repeat across a chunk's candidates) --
    Column("entropy", "property", False, "float64",
           "Shannon entropy of the BYTE histogram, bits/byte in [0, 8], over "
           "num_elements*sizeof(T) bytes. float64 chunks are DOWNCAST to "
           "float32 before measurement (the model was normalised on the "
           "four-byte distribution), so this is a float32 byte entropy."),
    Column("mad", "property", False, "float64",
           "mean(|x - mean(x)|) over elements. RAW DATA UNITS -- not "
           "normalised by range, despite the header comment upstream. Not "
           "comparable across physical fields."),
    Column("second_deriv", "property", False, "float64",
           "mean(|x[i+1] - 2x[i] + x[i-1]|) over i in [1, n-2], divided by "
           "(n-2). RAW DATA UNITS, and computed on the FLATTENED buffer -- a "
           "1-D stencil over a 3-D field, so it mixes in-row curvature with "
           "row-wrap discontinuities."),

    # ---- predictions (the NN's own outputs for THIS action) -------------
    Column("pred_ratio", "prediction", True, "float64",
           "NN-predicted compression ratio for this action."),
    Column("pred_ct_ms", "prediction", True, "float64",
           "NN-predicted compression time, ms."),
    Column("pred_dt_ms", "prediction", True, "float64",
           "NN-predicted decompression time, ms. Always a real prediction: "
           "the dt head runs for every action in the ranking inference."),

    # ---- measured outcomes ----------------------------------------------
    Column("ratio", "outcome", True, "float64",
           "chunk_bytes / codec_output_bytes. The Clio 24-byte header is "
           "EXCLUDED, matching upstream's actual_ratio. Values below 1 are "
           "real (codec expanded the chunk)."),
    Column("ct_ms", "outcome", True, "float64",
           "MEASURED compression time, CUDA-event kernel time (falls back to "
           "wall clock only if the event bracket failed). Alternatives are "
           "measured while up to CLIO_NEUROPRESS_EXPLORE_STREAMS (default 4) "
           "candidates run CONCURRENTLY; the primary is measured alone."),
    Column("dt_ms", "outcome", True, "float64",
           "MEASURED decompression time, or -1 when the sweep did not take "
           "one (the default; CLIO_NEUROPRESS_EXPLORE_MEASURE_DT enables it). "
           "-1 is 'not measured', NOT a fast decompression.", sentinel=-1.0),
    Column("psnr_db", "outcome", True, "float64",
           "ANALYTICAL PSNR from (data_range, effective_error_bound), capped "
           "at 120 dB; -1 on a lossless row. Structurally unable to see a "
           "bound violation -- use meas_psnr_db for that.", sentinel=-1.0),

    # ---- selection ------------------------------------------------------
    Column("cost", "outcome", True, "float64",
           "LOWER IS BETTER. w_ct*max(1,ct) + w_dt*max(1,dt) + "
           "w_io*chunk_bytes/(min(cap,ratio)*bw); 1e30 when ratio <= 0. The "
           "four weights and the cap are env-overridable, so they are "
           "INFERRED from the log rather than assumed."),
    Column("primary_cost", "outcome", True, "float64",
           "The baseline every row of this chunk was ranked against "
           "(primary_rank_cost). Constant within a chunk. Not the primary "
           "row's `actual_cost` -- it uses the MEASURED primary dt."),
    Column("adopted", "outcome", True, "int64",
           "1 on the single row per chunk whose bytes were stored. The "
           "primary is adopted unless an alternative beat it AND fit."),

    # ---- measured reconstruction quality (opt-in) ------------------------
    Column("quality_measured", "outcome", False, "int64",
           "1 when the candidate was reconstructed and compared against the "
           "original. Disambiguates the -1 sentinels below, which SSIM's "
           "valid range ([-1,1]) could not do on its own."),
    Column("meas_rmse", "outcome", False, "float64",
           "MEASURED RMSE against the pre-transform original.", sentinel=-1.0),
    Column("meas_max_error", "outcome", False, "float64",
           "MEASURED max absolute error -- the one that can witness an error "
           "bound violation.", sentinel=-1.0),
    Column("meas_psnr_db", "outcome", False, "float64",
           "MEASURED PSNR, still carrying upstream's 120 dB cap.",
           sentinel=-1.0),
    Column("meas_ssim", "outcome", False, "float64",
           "MEASURED SSIM. Saturates at 1 for any good reconstruction.",
           sentinel=-1.0),
    Column("meas_ssim_deviation", "outcome", False, "float64",
           "1 - ssim computed WITHOUT the subtraction, so the information "
           "that saturates out of `meas_ssim` survives. Prefer this over "
           "1 - meas_ssim, which loses it to cancellation.", sentinel=-1.0),
]

BY_NAME: Dict[str, Column] = {c.name: c for c in COLUMNS}
REQUIRED = [c.name for c in COLUMNS if c.required]
OPTIONAL = [c.name for c in COLUMNS if not c.required]

#: Intrinsic per-chunk properties. Constant across a chunk's candidate rows.
CHUNK_PROPERTY_COLS = ["chunk_bytes", "entropy", "mad", "second_deriv"]
#: The three features whose sufficiency is the research question.
FEATURES = ["entropy", "mad", "second_deriv"]
#: What identifies a compression configuration.
CONFIG_COLS = ["lib_name", "preset", "quantize", "shuffle", "error_bound"]
#: Measured outcomes.
OUTCOME_COLS = ["ratio", "ct_ms", "dt_ms", "cost"]

#: The lossless sentinel the model is fed on input 3 (configs.py:44).
EB_LOSSLESS_SENTINEL = 1e-7


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

@dataclass
class Audit:
    """Everything questionable about one CSV, recorded rather than fixed."""
    path: str = ""
    n_rows: int = 0
    missing_required: List[str] = _dc_field(default_factory=list)
    missing_optional: List[str] = _dc_field(default_factory=list)
    unknown_columns: List[str] = _dc_field(default_factory=list)
    notes: List[str] = _dc_field(default_factory=list)
    warnings: List[str] = _dc_field(default_factory=list)
    #: every row we dropped, and why. Never silent.
    filters: List[dict] = _dc_field(default_factory=list)
    counts: Dict[str, object] = _dc_field(default_factory=dict)
    nan_counts: Dict[str, int] = _dc_field(default_factory=dict)
    inf_counts: Dict[str, int] = _dc_field(default_factory=dict)
    sentinel_counts: Dict[str, int] = _dc_field(default_factory=dict)
    outliers: List[dict] = _dc_field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def filtered(self, n: int, reason: str, action: str = "dropped") -> None:
        if n:
            self.filters.append({"rows": int(n), "reason": reason,
                                 "action": action})

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "n_rows": self.n_rows,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "unknown_columns": self.unknown_columns,
            "counts": self.counts,
            "nan_counts": self.nan_counts,
            "inf_counts": self.inf_counts,
            "sentinel_counts": self.sentinel_counts,
            "filters": self.filters,
            "outliers": self.outliers,
            "notes": self.notes,
            "warnings": self.warnings,
        }


def audit_frame(df: pd.DataFrame, path: str) -> Audit:
    """Check a raw exploration frame against the contract.

    Reports; does not repair. The only rows the loader later drops are ones
    that cannot be interpreted at all, and each drop lands in `filters`.
    """
    a = Audit(path=path, n_rows=int(len(df)))
    present = set(df.columns)

    a.missing_required = [c for c in REQUIRED if c not in present]
    a.missing_optional = [c for c in OPTIONAL if c not in present]
    a.unknown_columns = sorted(present - set(BY_NAME))

    for name in df.columns:
        s = df[name]
        if not pd.api.types.is_numeric_dtype(s):
            n_nan = int(s.isna().sum())
            if n_nan:
                a.nan_counts[name] = n_nan
            continue
        v = s.to_numpy(dtype="float64", copy=False, na_value=np.nan) \
            if hasattr(s, "to_numpy") else np.asarray(s, dtype="float64")
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        if n_nan:
            a.nan_counts[name] = n_nan
        if n_inf:
            a.inf_counts[name] = n_inf
        col = BY_NAME.get(name)
        if col is not None and col.sentinel is not None:
            n_sent = int(np.sum(v == col.sentinel))
            if n_sent:
                a.sentinel_counts[name] = n_sent

    # ---- duplicates -----------------------------------------------------
    if "seq" in present:
        dup_seq = int(df["seq"].duplicated().sum())
        if dup_seq:
            a.warn(f"{dup_seq} duplicate seq values -- two logs concatenated, "
                   f"or a run that reset its counter. seq is not a key.")
    key = [c for c in ("blob", "lib_name", "preset", "quantize", "shuffle",
                       "role", "rank") if c in present]
    if key:
        n_dup = int(df.duplicated(subset=key).sum())
        if n_dup:
            a.warn(f"{n_dup} rows duplicate the configuration key "
                   f"{'+'.join(key)}. Repeated measurements of one "
                   f"configuration, or a concatenated log.")

    # ---- structural invariants the writer guarantees ---------------------
    if {"blob", "adopted"} <= present:
        per = df.groupby("blob")["adopted"].sum()
        bad = per[per != 1]
        if len(bad):
            a.warn(f"{len(bad)} of {len(per)} chunks do not carry exactly one "
                   f"adopted row (counts seen: "
                   f"{sorted(bad.unique().tolist())[:5]}). The writer emits "
                   f"exactly one; a violation means a truncated or merged log.")
    if {"blob", "primary_cost"} <= present:
        nun = df.groupby("blob")["primary_cost"].nunique()
        if (nun > 1).any():
            a.warn(f"{int((nun > 1).sum())} chunks carry more than one "
                   f"primary_cost. It is the single ranking baseline for a "
                   f"chunk, so this means rows from two sweeps of one blob.")
    if {"blob", "role"} <= present:
        nprim = df[df["role"] == "primary"].groupby("blob").size()
        many = int((nprim > 1).sum())
        if many:
            a.warn(f"{many} chunks have more than one primary row.")

    # ---- values that are suspicious but real -----------------------------
    if "ratio" in present:
        n_exp = int((df["ratio"] < 1.0).sum())
        if n_exp:
            a.note(f"{n_exp} rows have ratio < 1 (the codec EXPANDED the "
                   f"chunk). Real, and kept: those candidates simply lose.")
        n_bad = int((df["ratio"] <= 0).sum())
        if n_bad:
            a.warn(f"{n_bad} rows have ratio <= 0, which the cost model maps "
                   f"to its 1e30 gate sentinel.")
    if "cost" in present:
        n_gate = int((df["cost"] >= 1e29).sum())
        if n_gate:
            a.note(f"{n_gate} rows carry the 1e30 cost gate sentinel "
                   f"(ratio <= 0); excluded from cost statistics.")
    if "dt_ms" in present:
        n_unmeas = int((df["dt_ms"] < 0).sum())
        if n_unmeas == len(df):
            a.note("dt_ms is -1 on every row: the sweep did not measure "
                   "decompression (CLIO_NEUROPRESS_EXPLORE_MEASURE_DT off). "
                   "Every dt-dependent analysis is therefore unavailable, and "
                   "`cost` used the PRIMARY's predicted dt held constant "
                   "across candidates -- so dt cancels out of the ranking.")
        elif n_unmeas:
            a.note(f"{n_unmeas} of {len(df)} rows have dt_ms = -1 (not "
                   f"measured); they are excluded from dt analyses rather "
                   f"than read as 0.")

    # ---- inventory ------------------------------------------------------
    def inv(col):
        if col not in present:
            return None
        u = df[col].dropna().unique()
        return sorted(u.tolist())

    a.counts = {
        "rows": int(len(df)),
        "unique_chunks": int(df["blob"].nunique()) if "blob" in present else 0,
        "codecs": inv("lib_name"),
        "presets": inv("preset"),
        "quantize_modes": inv("quantize"),
        "shuffle_widths": inv("shuffle"),
        "roles": inv("role"),
        "chunk_bytes": inv("chunk_bytes"),
    }
    if "eb_encoded" in present and "quantize" in present:
        q = df[df["quantize"] == 1]["eb_encoded"].dropna().unique()
        a.counts["error_bounds_lossy"] = sorted(float(x) for x in q)
        lossless = df[df["quantize"] == 0]["eb_encoded"].dropna().unique()
        off = [float(x) for x in lossless if not np.isclose(
            x, EB_LOSSLESS_SENTINEL, rtol=1e-6)]
        if off:
            a.warn(f"lossless rows carry eb_encoded values other than the "
                   f"1e-7 sentinel: {sorted(set(off))[:5]}. Input 3 is "
                   f"`quantize ? eb : 1e-7`, so this is unexpected.")
    return a


def outlier_report(df: pd.DataFrame, cols: List[str],
                   z: float = 6.0) -> List[dict]:
    """Robust (median / MAD) outlier counts. Reported, never removed.

    Uses the median absolute deviation rather than a standard deviation
    because the columns of interest -- ratio above all -- are heavy tailed by
    construction, and a mean-based z would call the tail the outlier.
    """
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if v.size < 20:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad <= 0:
            continue
        scaled = 0.6745 * (v - med) / mad
        n = int(np.sum(np.abs(scaled) > z))
        if n:
            out.append({
                "column": c, "median": med, "mad": mad, "threshold_z": z,
                "n_outliers": n, "pct": 100.0 * n / v.size,
                "max_abs_z": float(np.max(np.abs(scaled))),
                "min_value": float(v.min()), "max_value": float(v.max()),
            })
    return out

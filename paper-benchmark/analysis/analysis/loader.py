#!/usr/bin/env python3
"""Load one exploration CSV into the three-level model the analysis needs, and
record everything that had to be assumed to get there.

THE UNIT PROBLEM, which everything downstream rests on. A physical chunk
appears once per candidate configuration -- 32 times in a full 8-codec x
{quant} x {shuffle} sweep. So a row is a (chunk, configuration) MEASUREMENT,
not a scientific-data sample. This module produces both levels explicitly:

    rows    one per (chunk, configuration)   -- outcomes live here
    chunks  one per blob                     -- intrinsic properties live here

Any correlation between an intrinsic property and an outcome that is computed
over `rows` is inflated by the 32x replication of the property. Every such
analysis downstream either aggregates to `chunks` first or reports its
effective sample size as the chunk count.

SIDECAR JOIN. Older exploration logs (before the commit that added the model's
own inputs to the sweep log) have no entropy/mad/second_deriv columns, but the
selection log written beside them does, keyed by the same blob. When the
features are missing this module looks for that sidecar and joins it, and says
so loudly in the provenance -- an analysis of data properties that silently
found no data properties would otherwise report "no relationship".
"""
from __future__ import annotations

import glob
import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field as _dc_field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema
from .chunk_parser import attach_metadata


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit(start: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(start)),
             "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def dependency_versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "missing"
    return out


def provenance(inputs: List[str], argv: List[str], seed: int) -> dict:
    return {
        "inputs": [{"path": os.path.abspath(p), "sha256": file_sha256(p),
                    "bytes": os.path.getsize(p)} for p in inputs],
        "analysis_git_commit": _git_commit(__file__),
        "cli_args": list(argv),
        "random_seed": seed,
        "dependencies": dependency_versions(),
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    """One workload's exploration log, at both levels, plus its audit."""
    name: str
    path: str
    rows: pd.DataFrame          # one row per (chunk, configuration)
    chunks: pd.DataFrame        # one row per blob
    audit: schema.Audit
    features: List[str] = _dc_field(default_factory=list)   # usable of FEATURES
    sidecars: Dict[str, str] = _dc_field(default_factory=dict)
    cost_model: Dict[str, object] = _dc_field(default_factory=dict)
    #: Whether the explored chunks are a representative sample of the run.
    cohort: Dict[str, object] = _dc_field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        return int(self.rows["chunk_uid"].nunique())

    @property
    def has_features(self) -> bool:
        return len(self.features) > 0

    def lossless(self) -> pd.DataFrame:
        return self.rows[self.rows["quantize"] == 0]

    def lossy(self) -> pd.DataFrame:
        return self.rows[self.rows["quantize"] == 1]


def _find_sidecar(explore_path: str, kind: str) -> Optional[str]:
    """Look for a selection / quality / evolution log beside the sweep log."""
    d = os.path.dirname(os.path.abspath(explore_path)) or "."
    pats = {
        "selection": ["selection.csv", "*selection*.csv", "*.selection.csv"],
        "quality": ["selection.csv.quality", "*.quality"],
        "blobs": ["blobs.csv"],
        "evolution": ["blocks.csv", "*.blocks.csv", "*.blocks.csv.gz"],
    }[kind]
    for pat in pats:
        hits = sorted(glob.glob(os.path.join(d, pat)))
        hits = [h for h in hits
                if os.path.abspath(h) != os.path.abspath(explore_path)]
        if hits:
            return hits[0]
    return None


def _join_features_from_selection(df: pd.DataFrame, path: str,
                                  audit: schema.Audit) -> Tuple[pd.DataFrame,
                                                                Optional[str]]:
    """Recover entropy/mad/second_deriv from the selection log beside us.

    The selection log carries the same three features under the same names,
    keyed by blob, because both are written from the same
    (neuropress_entropy, neuropress_mad, neuropress_second_deriv) locals in the
    compress path. So the join is exact, not approximate.
    """
    missing = [c for c in schema.FEATURES if c not in df.columns]
    if not missing:
        return df, None
    side = _find_sidecar(path, "selection")
    if side is None:
        audit.warn(
            "entropy/mad/second_deriv are absent from this log and no "
            "selection.csv was found beside it. This log predates the commit "
            "that added the model's own inputs to the sweep log. Every "
            "data-property analysis is UNAVAILABLE for this input.")
        return df, None
    try:
        sel = pd.read_csv(side)
    except Exception as e:
        audit.warn(f"found {side} but could not read it: {e}")
        return df, None
    have = [c for c in missing if c in sel.columns]
    if "blob" not in sel.columns or not have:
        audit.warn(f"{os.path.basename(side)} carries no usable feature "
                   f"columns; data-property analyses stay unavailable.")
        return df, None

    # One row per blob. The selection log has a `primary` row and possibly an
    # `adopted` row per blob; the features describe the CHUNK, so they are
    # identical on both -- verified here rather than assumed.
    sub = sel[["blob"] + have].copy()
    nun = sub.groupby("blob")[have].nunique().max()
    if (nun > 1).any():
        audit.warn(
            "the selection log gives a blob more than one value for "
            f"{[c for c in have if nun[c] > 1]}; taking the first. These are "
            "chunk properties and should not vary within a blob.")
    sub = sub.groupby("blob", as_index=False).first()

    before = len(df)
    df = df.merge(sub, on="blob", how="left", validate="many_to_one")
    n_matched = int(df[have[0]].notna().sum())
    assert len(df) == before, "feature join changed the row count"
    audit.note(
        f"entropy/mad/second_deriv were NOT in the sweep log; joined from "
        f"{os.path.basename(side)} on `blob` ({n_matched}/{len(df)} rows "
        f"matched, {df.loc[df[have[0]].notna(), 'blob'].nunique()} chunks). "
        f"This log predates those columns; the values are the same locals the "
        f"sweep would have written.")
    if n_matched < len(df):
        audit.warn(f"{len(df) - n_matched} rows had no feature match in the "
                   f"selection log and are excluded from property analyses.")
    return df, side


def cohort_audit(rows: pd.DataFrame, selection_path: Optional[str],
                 audit: schema.Audit) -> Dict[str, object]:
    """Are the chunks in this sweep log a REPRESENTATIVE sample of the run?

    They usually are not, and the reason is structural. Exploration is
    triggered per chunk -- upstream fires it when the model's own prediction
    misses badly enough -- so a sweep log contains the chunks where the
    PREDICTOR ALREADY FAILED and omits the ones it got right. Every statistic
    about prediction error and cost-model regret computed on such a log is
    conditioned on that trigger, and reads worse than the run as a whole.

    The selection log beside the sweep lists every chunk that was written,
    explored or not, which is what makes the comparison possible. When it is
    absent the bias cannot be measured -- and that is reported too, because an
    unmeasurable bias is not the same as an absent one.
    """
    out: Dict[str, object] = {"measurable": False}
    if not selection_path:
        audit.warn(
            "no selection.csv beside this log, so it cannot be checked "
            "whether the explored chunks are a representative sample of the "
            "run. Exploration is trigger-gated upstream, so they usually are "
            "NOT: prediction-error and regret figures below may be "
            "conditioned on the trigger.")
        return out
    try:
        sel = pd.read_csv(selection_path)
    except Exception:
        return out
    if "blob" not in sel.columns:
        return out
    prim = sel[sel["role"] == "primary"] if "role" in sel.columns else sel
    prim = prim.drop_duplicates("blob")
    explored = set(rows["blob"].unique())
    missing = prim[~prim["blob"].isin(explored)]
    kept = prim[prim["blob"].isin(explored)]
    out.update({
        "measurable": True,
        "chunks_in_selection_log": int(len(prim)),
        "chunks_explored": int(len(kept)),
        "chunks_not_explored": int(len(missing)),
        "pct_explored": (100.0 * len(kept) / len(prim)) if len(prim) else 0.0,
    })
    if missing.empty:
        audit.note("every chunk in the selection log was also explored, so "
                   "the sweep is a complete sample of the run.")
        return out

    # Characterise the two groups on whatever the selection log carries. The
    # point is not to guess the trigger's exact rule but to show WHICH WAY the
    # omitted chunks differ, which is what tells a reader how the conditioning
    # moves each conclusion.
    diffs = {}
    for col in ("pred_ratio", "actual_ratio", "entropy", "mad",
                "second_deriv", "actual_ct_ms"):
        if col not in prim.columns:
            continue
        a = pd.to_numeric(missing[col], errors="coerce").dropna()
        b = pd.to_numeric(kept[col], errors="coerce").dropna()
        if a.empty or b.empty:
            continue
        diffs[col] = {"not_explored_median": float(a.median()),
                      "explored_median": float(b.median())}
    out["group_differences"] = diffs
    if {"pred_ratio", "actual_ratio"} <= set(prim.columns):
        cap = float(prim["pred_ratio"].max())
        at_cap = missing["pred_ratio"] >= cap - 1e-9
        over = missing["actual_ratio"] > cap
        out["pct_not_explored_with_pred_at_cap"] = float(100.0 * at_cap.mean())
        out["pct_not_explored_underpredicted"] = float(100.0 * over.mean())
        out["pred_ratio_cap_observed"] = cap
        if at_cap.mean() > 0.9 and over.mean() > 0.9:
            out["inferred_trigger"] = (
                f"every unexplored chunk has pred_ratio at the {cap:g} cap "
                f"AND an actual ratio above it, i.e. the predictor "
                f"UNDER-predicted them and exploration did not fire. The "
                f"explored subset is therefore enriched in chunks the model "
                f"OVER-predicted, which biases every over/under-prediction "
                f"figure in this report toward over-prediction.")
    audit.warn(
        f"this sweep log covers {len(kept)} of the {len(prim)} chunks in the "
        f"selection log beside it ({100.0 * len(kept) / len(prim):.0f}%). "
        f"Exploration is trigger-gated, so these chunks are a CONDITIONED "
        f"sample, not a random one -- prediction-error, over/under-prediction "
        f"and cost-model regret figures describe the explored subset and do "
        f"NOT generalise to the whole run. "
        + str(out.get("inferred_trigger", "")))
    return out


def load_exploration(path: str, name: Optional[str] = None) -> Dataset:
    """Read one exploration CSV into a Dataset. Nothing is silently dropped."""
    raw = pd.read_csv(path)
    audit = schema.audit_frame(raw, path)
    name = name or os.path.splitext(os.path.basename(path))[0]

    if audit.missing_required:
        raise ValueError(
            f"{path}: missing required columns {audit.missing_required}. "
            f"This does not look like a Clio-NeuroPress exploration log.")

    df, sidecar = _join_features_from_selection(raw, path, audit)
    sidecars = {}
    # _join_features_from_selection returns the selection log ONLY when it had
    # to join features out of it. A log that already carries entropy/mad/
    # second_deriv -- every log written since the commit that added the model's
    # own inputs to the sweep -- takes its early return and reports no sidecar,
    # which used to leave cohort_audit() with selection_path=None and made it
    # announce "no selection.csv beside this log". The newest and most complete
    # logs were the ones whose sample bias was declared unmeasurable. Look the
    # sidecar up on its own account instead.
    selection = sidecar or _find_sidecar(path, "selection")
    if selection:
        sidecars["selection"] = selection
    for kind in ("quality", "blobs", "evolution"):
        s = _find_sidecar(path, kind)
        if s:
            sidecars[kind] = s

    df = attach_metadata(df)
    df["workload"] = name

    # ---- derived configuration ------------------------------------------
    df["shuffle_on"] = (df["shuffle"] > 0).astype(int)
    df["lossless"] = (df["quantize"] == 0)
    # The CONFIGURED error bound, recoverable only where quantization ran.
    # eb_encoded carries the model's 1e-7 sentinel on lossless rows, so
    # reading it directly would invent a 1e-7 bound for lossless data.
    if "eb_encoded" in df.columns:
        df["error_bound"] = np.where(df["quantize"] == 1,
                                     df["eb_encoded"], np.nan)
    else:
        df["error_bound"] = np.nan
        audit.note("eb_encoded is absent; error_bound is unknown for every "
                   "row. Lossy rows are still identifiable by quantize == 1, "
                   "but per-error-bound analysis is unavailable.")
    df["config_id"] = (
        df["lib_name"].astype(str) + "|p" + df["preset"].astype(str)
        + "|q" + df["quantize"].astype(str)
        + "|s" + df["shuffle"].astype(str)
        + "|eb" + df["error_bound"].map(
            lambda x: "none" if pd.isna(x) else f"{x:.6g}"))

    # ---- sentinels made explicit ----------------------------------------
    # -1 means NOT MEASURED. Kept as NaN in the analysis columns so that no
    # mean, correlation or model can read it as a fast decompression or a
    # perfect reconstruction; the raw columns are preserved untouched.
    df["dt_ms_measured"] = df["dt_ms"].where(df["dt_ms"] >= 0)
    df["psnr_db_analytical"] = df["psnr_db"].where(df["psnr_db"] >= 0)

    # Current logs write NA where a measured quality is not defined, which
    # arrives here as NaN already; older ones write a -1 sentinel. Both must
    # end as NaN in the _ok columns, so the >= 0 test is kept for the old form
    # and NaN falls through it. quality_measured == 1 is the authority in
    # either case, and it is itself NA on lossless rows in current logs.
    qm = df["quality_measured"] if "quality_measured" in df.columns else None
    for col in ("meas_rmse", "meas_max_error", "meas_psnr_db", "meas_ssim",
                "meas_ssim_deviation"):
        if col not in df.columns:
            continue
        valid = df[col] >= 0 if col != "meas_ssim" else df[col] > -1.0
        if qm is not None:
            valid &= (qm == 1)
        df[col + "_ok"] = df[col].where(valid)

    # SSIM saturates at 1, so the deviation is where the information is. Use
    # the writer's own ssim_deviation when present; otherwise reconstruct it,
    # and flag that the reconstruction has already lost precision to
    # cancellation -- which is exactly why the writer emits it separately.
    if "meas_ssim_deviation_ok" in df.columns:
        df["ssim_deviation"] = df["meas_ssim_deviation_ok"]
    elif "meas_ssim_ok" in df.columns:
        df["ssim_deviation"] = 1.0 - df["meas_ssim_ok"]
        audit.note(
            "meas_ssim_deviation is absent; ssim_deviation was reconstructed "
            "as 1 - meas_ssim. The log is written with 17 significant digits "
            "for exactly this reason, but the subtraction still cancels most "
            "of them -- treat small deviations as order-of-magnitude only.")
    else:
        df["ssim_deviation"] = np.nan

    # ---- rows that cannot be interpreted at all --------------------------
    n0 = len(df)
    gate = df["cost"] >= 1e29
    if gate.any():
        audit.filtered(int(gate.sum()),
                       "cost == 1e30 gate sentinel (ratio <= 0); the "
                       "candidate produced no usable output",
                       action="excluded from cost/selection analyses only")
    df["cost_valid"] = ~gate
    bad_ratio = ~np.isfinite(df["ratio"])
    if bad_ratio.any():
        audit.filtered(int(bad_ratio.sum()), "non-finite ratio", "dropped")
        df = df[~bad_ratio].copy()
    assert len(df) == n0 - int(bad_ratio.sum())

    features = [c for c in schema.FEATURES
                if c in df.columns and df[c].notna().any()]
    for f in features:
        if df[f].nunique(dropna=True) <= 1:
            audit.warn(f"`{f}` takes a single value across the whole log "
                       f"({df[f].dropna().iloc[0] if df[f].notna().any() else 'NaN'}); "
                       f"it cannot explain anything here and is excluded from "
                       f"correlations and models.")
    features = [f for f in features if df[f].nunique(dropna=True) > 1]

    chunks = build_chunk_table(df, features)
    audit.counts["unique_chunks"] = int(len(chunks))
    audit.counts["timesteps"] = sorted(
        int(t) for t in chunks["timestep"].dropna().unique())
    audit.counts["fields"] = sorted(
        str(f) for f in chunks["field"].dropna().unique())
    audit.counts["usable_features"] = features
    audit.outliers = schema.outlier_report(
        df, ["ratio", "ct_ms", "dt_ms_measured", "cost", "pred_ratio"])

    ds = Dataset(name=name, path=path, rows=df, chunks=chunks, audit=audit,
                 features=features, sidecars=sidecars)
    ds.cohort = cohort_audit(df, sidecars.get("selection"), audit)
    return ds


def build_chunk_table(rows: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """One row per chunk: intrinsic properties and metadata only.

    Outcomes are deliberately NOT aggregated in here -- a chunk has 32 of them
    and which one matters depends on the question. cost_analysis.winners()
    builds the outcome side separately and joins on chunk_uid.
    """
    keep = ["chunk_uid", "blob", "workload", "chunk_bytes", "timestep",
            "plotfile", "fab", "component", "field", "field_path", "chunk_id",
            "parse_ok"] + features
    keep = [c for c in keep if c in rows.columns]
    g = rows.groupby("chunk_uid", as_index=False)[keep].first()

    # A property that varies within a chunk is a contradiction: the writer
    # emits the same locals on every row. Check rather than trust.
    for f in features + ["chunk_bytes"]:
        if f not in rows.columns:
            continue
        nun = rows.groupby("chunk_uid")[f].nunique(dropna=True)
        if (nun > 1).any():
            g.attrs.setdefault("inconsistent", []).append(f)
    g["n_candidates"] = rows.groupby("chunk_uid").size().reindex(
        g["chunk_uid"]).to_numpy()
    return g


def load_evolution(path: str) -> Optional[pd.DataFrame]:
    """Read a paper-benchmark evolution blocks.csv (optionally .gz).

    Schema (paper-benchmark/evolution.py):
        step_from,step_to,field,block,evolution,nonfinite,pct_cells_same
    where `evolution` is E(B_t, B_t+dt) = ||B2-B1|| / (||B1||+||B2||+eps).
    """
    try:
        ev = pd.read_csv(path)
    except Exception:
        return None
    need = {"step_to", "field", "block", "evolution"}
    if not need <= set(ev.columns):
        return None
    ev = ev[ev.get("nonfinite", 0) == 0] if "nonfinite" in ev.columns else ev
    return ev

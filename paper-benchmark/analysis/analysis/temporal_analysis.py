#!/usr/bin/env python3
"""How the data's properties, and its compressibility, move as the simulation
runs -- and whether the block-evolution metric explains the movement.

THE EVOLUTION METRIC, when a paper-benchmark blocks.csv is found beside the
log (paper-benchmark/evolution.py):

    E(B_t, B_t+dt) = ||B_t+dt - B_t||_2 / (||B_t+dt||_2 + ||B_t||_2 + eps)

Two caveats from that file's own header are enforced here rather than
inherited silently, because both invert the interpretation of a correlation:

  * ZERO-BACKGROUND FIELDS. E is relative, so a block whose background is
    exactly zero saturates at E = 1 the moment anything appears in it, however
    small. On Nyx Sedov momentum the ranking by E is very nearly the REVERSE
    of the ranking by absolute change. So `evolution_saturated` marks
    E > 0.99, and any correlation computed over a field where those dominate
    is flagged rather than reported as physics.
  * PARTICLE POSITIONS. A wrapped coordinate jumps a full box length on a
    neighbour rebuild. Nothing here can detect that from the CSV, so it is
    carried into the report as a standing caveat for particle workloads.

Everything else in this module needs only a parsed `timestep`, and degrades to
"unavailable" without one rather than inventing an ordering.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from .statistics import _corr, MIN_SUPPORT


def has_time(chunks: pd.DataFrame, min_steps: int = 3) -> bool:
    return ("timestep" in chunks
            and chunks["timestep"].notna().sum() > 0
            and chunks["timestep"].nunique(dropna=True) >= min_steps)


def property_series(chunks: pd.DataFrame,
                    features: Sequence[str]) -> pd.DataFrame:
    """Median of each intrinsic property per (field, timestep)."""
    if not has_time(chunks):
        return pd.DataFrame()
    feats = [f for f in features if f in chunks.columns]
    g = chunks.dropna(subset=["timestep"]).groupby(
        ["field", "timestep"], dropna=False)
    out = g[feats].median().reset_index()
    out["n_chunks"] = g.size().to_numpy()
    return out.sort_values(["field", "timestep"])


def outcome_series(rows: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """Per (field, timestep): best achievable ratio, adopted ratio, and the
    codec that took each win. One row per timestep per field, so the series is
    over chunks and not over the 32 candidates of each chunk."""
    if "timestep" not in rows.columns:
        return pd.DataFrame()
    w = win.merge(
        rows.groupby("chunk_uid", as_index=False)[
            [c for c in ("field", "timestep", "workload") if c in rows]
        ].first(), on="chunk_uid", how="left")
    if "timestep" not in w or w["timestep"].isna().all():
        return pd.DataFrame()
    recs: List[dict] = []
    for (fld, t), g in w.dropna(subset=["timestep"]).groupby(
            ["field", "timestep"], dropna=False):
        rec: Dict[str, object] = {
            "field": fld, "timestep": t, "n_chunks": int(len(g)),
            "best_ratio_median": float(g["best_ratio_value"].median()),
            "adopted_ratio_median": float(g["adopted_ratio"].median()),
            "adopted_cost_median": float(g["adopted_cost"].median()),
        }
        for col, name in (("best_ratio_lib", "modal_best_ratio_lib"),
                          ("adopted_lib", "modal_adopted_lib"),
                          ("primary_lib", "modal_model_pick_lib")):
            if col in g and g[col].notna().any():
                m = g[col].mode()
                rec[name] = str(m.iloc[0]) if len(m) else None
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["field", "timestep"])


def codec_switches(series: pd.DataFrame,
                   col: str = "modal_best_ratio_lib") -> pd.DataFrame:
    """Where along the run the winning codec changes, per field."""
    if series.empty or col not in series:
        return pd.DataFrame()
    recs: List[dict] = []
    for fld, g in series.sort_values("timestep").groupby("field"):
        prev = None
        for _, r in g.iterrows():
            cur = r[col]
            if prev is not None and cur != prev:
                recs.append({"field": fld, "timestep": r["timestep"],
                             "from": prev, "to": cur, "which": col})
            prev = cur
    return pd.DataFrame(recs)


def temporal_trends(series: pd.DataFrame,
                    cols: Sequence[str]) -> pd.DataFrame:
    """Spearman of each series against timestep, per field.

    Spearman, not a linear slope: the question is whether compressibility
    moves monotonically as the simulation evolves, and a blast wave's entropy
    does not move linearly.
    """
    if series.empty:
        return pd.DataFrame()
    recs: List[dict] = []
    for fld, g in series.groupby("field", dropna=False):
        if len(g) < 4:
            continue
        for c in cols:
            if c not in g or g[c].isna().all():
                continue
            r = _corr(g["timestep"].to_numpy(dtype=float),
                      g[c].to_numpy(dtype=float))
            recs.append({"field": fld, "series": c, "n_timesteps": r["n"],
                         "spearman_rho": r["spearman_rho"],
                         "p": r["spearman_p"],
                         # A perfect rho over five timesteps is a real
                         # observation and a weak test; both facts travel
                         # together so the report cannot quote one alone.
                         "usable": bool(r["n"] >= 5),
                         "first": float(g.sort_values("timestep")[c].iloc[0]),
                         "last": float(g.sort_values("timestep")[c].iloc[-1])})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# Block-evolution join
# --------------------------------------------------------------------------

def join_evolution(chunks: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Attach E(B_t, B_t+dt) to each chunk, on (field, timestep, chunk id).

    The evolution CSV keys a block by (field, block index) and an interval by
    (step_from, step_to); a chunk is keyed by (field, timestep, chunk_id). The
    join is therefore chunk_id <-> block and timestep <-> step_to, i.e. each
    chunk gets the evolution of the interval that ENDED at its own timestep.
    That direction is the meaningful one: it asks whether the change that
    produced this frame explains how this frame compresses.

    Returns an empty frame -- not a wrong one -- when the two files disagree
    about field names or step numbering, which is normal when the evolution
    pass was run at a different sampling stride than the sweep.
    """
    if ev is None or ev.empty or "chunk_id" not in chunks.columns:
        return pd.DataFrame()
    e = ev.rename(columns={"block": "chunk_id", "step_to": "timestep"}).copy()
    e["chunk_id"] = pd.to_numeric(e["chunk_id"], errors="coerce")
    e["timestep"] = pd.to_numeric(e["timestep"], errors="coerce")
    e = e.groupby(["field", "timestep", "chunk_id"], as_index=False).agg(
        evolution=("evolution", "mean"),
        pct_cells_same=("pct_cells_same", "mean")
        if "pct_cells_same" in e else ("evolution", "size"))
    c = chunks.copy()
    c["chunk_id"] = pd.to_numeric(c["chunk_id"], errors="coerce")
    c["timestep"] = pd.to_numeric(c["timestep"], errors="coerce")
    m = c.merge(e, on=["field", "timestep", "chunk_id"], how="inner")
    if m.empty:
        return m
    # E saturates at 1 on a zero-background field the instant anything appears
    # (evolution.py's own Nyx-momentum example), so mark it rather than
    # correlate against it blind.
    m["evolution_saturated"] = m["evolution"] > 0.99
    return m


def evolution_effects(joined: pd.DataFrame, features: Sequence[str],
                      win: pd.DataFrame) -> pd.DataFrame:
    """Evolution score against each property and against compressibility."""
    if joined.empty:
        return pd.DataFrame()
    d = joined.merge(
        win[["chunk_uid", "best_ratio_value", "adopted_ratio",
             "best_ratio_lib"]], on="chunk_uid", how="left")
    targets = [f for f in list(features) + ["best_ratio_value",
                                            "adopted_ratio"] if f in d]
    recs: List[dict] = []
    for label, sub in (("all", d),
                       ("unsaturated", d[~d["evolution_saturated"]])):
        if len(sub) < MIN_SUPPORT:
            continue
        for t in targets:
            r = _corr(sub["evolution"].to_numpy(dtype=float),
                      sub[t].to_numpy(dtype=float))
            recs.append({"subset": label, "target": t, "n_chunks": r["n"],
                         "spearman_rho": r["spearman_rho"],
                         "pearson_r": r["pearson_r"], "p": r["spearman_p"],
                         "pct_saturated": float(
                             100.0 * d["evolution_saturated"].mean())})
    return pd.DataFrame(recs)

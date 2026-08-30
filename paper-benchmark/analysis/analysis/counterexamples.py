#!/usr/bin/env python3
"""Cases that contradict the dominant trends, collected on purpose.

Every finder here returns rows in one shape -- kind, why, chunk, the features,
and the numbers that make the case -- so they concatenate into a single
counterexamples.csv that can be sorted by severity. Nothing is suppressed: a
finding that survives only because its counterexamples were filtered out is
not a finding, and the count of counterexamples is quoted in the report beside
each claim it bears on.

The nearest-neighbour finders below deliberately search in STANDARDISED
feature space, and separately WITHIN each physical field. MAD and the second
derivative are in raw data units, so "two chunks with similar MAD" across a
density field and a momentum field is a statement about unit magnitudes, not
about the data -- and the pairs it produces are all spurious.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _rows(kind: str, why: str, frame: pd.DataFrame,
          cols: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[[c for c in cols if c in frame.columns]].copy()
    out.insert(0, "kind", kind)
    out.insert(1, "why", why)
    return out


BASE_COLS = ["chunk_uid", "workload", "field", "timestep", "entropy", "mad",
             "second_deriv", "chunk_bytes"]


def extreme_ratio_for_property(chunks: pd.DataFrame, win: pd.DataFrame,
                               features: Sequence[str],
                               q: float = 0.20) -> pd.DataFrame:
    """Low entropy yet poor compression, and high entropy yet excellent.

    "Poor" and "excellent" are defined against the OBSERVED distribution of
    the best achievable ratio in this log, not against an absolute number: a
    ratio of 20 is dismal on Nyx Sedov and outstanding on a particle dump.
    """
    if "entropy" not in chunks.columns:
        return pd.DataFrame()
    d = chunks.merge(win[["chunk_uid", "best_ratio_value", "best_ratio_lib",
                          "adopted_ratio"]], on="chunk_uid", how="inner")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["entropy", "best_ratio_value"])
    if len(d) < 10:
        return pd.DataFrame()
    e_lo, e_hi = d["entropy"].quantile([q, 1 - q])
    r_lo, r_hi = d["best_ratio_value"].quantile([q, 1 - q])
    cols = BASE_COLS + ["best_ratio_value", "best_ratio_lib", "adopted_ratio"]
    parts = [
        _rows("low_entropy_poor_compression",
              f"entropy <= p{int(q*100)} ({e_lo:.3g}) yet best ratio "
              f"<= p{int(q*100)} ({r_lo:.3g})",
              d[(d["entropy"] <= e_lo) & (d["best_ratio_value"] <= r_lo)],
              cols),
        _rows("high_entropy_good_compression",
              f"entropy >= p{int((1-q)*100)} ({e_hi:.3g}) yet best ratio "
              f">= p{int((1-q)*100)} ({r_hi:.3g})",
              d[(d["entropy"] >= e_hi) & (d["best_ratio_value"] >= r_hi)],
              cols),
    ]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True) \
        if any(not p.empty for p in parts) else pd.DataFrame()


def similar_properties_different_outcome(
        chunks: pd.DataFrame, win: pd.DataFrame, features: Sequence[str],
        prop_tol: float = 0.15, ratio_fold: float = 2.0,
        within_field: bool = True, max_pairs: int = 200,
        near_quantile: float = 0.05,
        max_relative_diff: float = 0.01) -> pd.DataFrame:
    """Chunks that look alike in every feature and compress very differently.

    These are the cases the three features CANNOT explain, and they are the
    direct evidence for or against feature sufficiency.

    "Alike" is ADAPTIVE. A fixed distance in standardised feature space finds
    nothing when a field has eight chunks spread across the whole space, and
    finds everything when it has eighty in a cluster -- in both cases saying
    more about the sample size than about the data. So the threshold is the
    larger of `prop_tol` and the `near_quantile` quantile of the observed
    pairwise distances within the group, and the distance that was actually
    used travels with each row.
    """
    feats = [f for f in features if f in chunks.columns]
    if not feats:
        return pd.DataFrame()
    d = chunks.merge(win[["chunk_uid", "best_ratio_value", "best_ratio_lib"]],
                     on="chunk_uid", how="inner")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=feats + ["best_ratio_value"])
    if len(d) < 6:
        return pd.DataFrame()

    groups = d.groupby("field", dropna=False) if within_field and \
        "field" in d.columns else [("all", d)]
    recs: List[dict] = []
    for fld, g in groups:
        if len(g) < 4:
            continue
        Z = g[feats].to_numpy(dtype=float)
        sd = Z.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (Z - Z.mean(axis=0)) / sd
        r = g["best_ratio_value"].to_numpy(dtype=float)
        uid = g["chunk_uid"].to_numpy()
        lib = g["best_ratio_lib"].to_numpy()
        dmat = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
        iu = np.triu_indices(len(g), k=1)
        tol = max(prop_tol, float(np.quantile(dmat[iu], near_quantile))) \
            if iu[0].size else prop_tol
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                dist = float(dmat[i, j])
                if dist > tol:
                    continue
                hi, lo = max(r[i], r[j]), min(r[i], r[j])
                if lo <= 0 or hi / lo < ratio_fold:
                    continue
                # For the ACROSS-field claim the distance must additionally be
                # small in each feature's OWN units, not merely in
                # standardised ones. Standardising divides by a global SD that
                # one outlier cluster can inflate several-fold -- on the
                # development log the SD of second_deriv was 7x larger with
                # six extreme chunks than without -- which let pairs differing
                # by ~30% in a feature look "identical". A relative test
                # cannot be gamed that way, and the claim being made ("no
                # model on these features can separate them") is only true at
                # genuine numerical identity.
                if not within_field:
                    rel = max(
                        abs(float(g[f].iloc[i]) - float(g[f].iloc[j]))
                        / max(abs(float(g[f].iloc[i])),
                              abs(float(g[f].iloc[j])), 1e-30)
                        for f in feats)
                    if rel > max_relative_diff:
                        continue
                recs.append({
                    "kind": ("similar_properties_different_ratio"
                             if within_field else
                             "identical_properties_different_ratio"),
                    "why": (f"the two chunks are {dist:.4f} standardised "
                            f"units apart in ({'+'.join(feats)}) "
                            + (f"within field '{fld}'"
                               if within_field else
                               "-- numerically indistinguishable to any model "
                               "built on those three features") +
                            f", yet their best achievable ratios differ "
                            f"{hi / lo:.2f}x"),
                    "chunk_uid": uid[i], "chunk_uid_b": uid[j],
                    "field": fld, "feature_distance": dist,
                    "distance_threshold_used": tol,
                    "max_relative_feature_diff": (
                        max(abs(float(g[f].iloc[i]) - float(g[f].iloc[j]))
                            / max(abs(float(g[f].iloc[i])),
                                  abs(float(g[f].iloc[j])), 1e-30)
                            for f in feats)),
                    "ratio_a": float(r[i]), "ratio_b": float(r[j]),
                    "ratio_fold": float(hi / lo),
                    "best_lib_a": lib[i], "best_lib_b": lib[j],
                    **{f: float(g[f].iloc[i]) for f in feats},
                })
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values("ratio_fold", ascending=False).head(max_pairs)


def similar_properties_different_codec(
        chunks: pd.DataFrame, win: pd.DataFrame, features: Sequence[str],
        prop_tol: float = 0.25, max_pairs: int = 200) -> pd.DataFrame:
    """Near-identical properties, different winning codec, and the win was not
    a tie on either side. A tie makes the "different winner" meaningless -- it
    is the deterministic tiebreak talking -- so both must be uncontested."""
    feats = [f for f in features if f in chunks.columns]
    need = {"best_ratio_lib", "best_ratio_nties"}
    if not feats or not need <= set(win.columns):
        return pd.DataFrame()
    d = chunks.merge(win[["chunk_uid", "best_ratio_lib", "best_ratio_nties",
                          "best_ratio_value"]], on="chunk_uid", how="inner")
    d = d[d["best_ratio_nties"] == 1].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=feats)
    if d["best_ratio_lib"].nunique() < 2 or len(d) < 4:
        return pd.DataFrame()
    Z = d[feats].to_numpy(dtype=float)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Z - Z.mean(axis=0)) / sd
    uid = d["chunk_uid"].to_numpy()
    lib = d["best_ratio_lib"].to_numpy()
    dmat = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    iu = np.triu_indices(len(d), k=1)
    # Adaptive for the same reason as above: a fixed radius reports the
    # sample density rather than the feature space.
    tol = max(prop_tol, float(np.quantile(dmat[iu], 0.05))) if iu[0].size \
        else prop_tol
    recs: List[dict] = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if lib[i] == lib[j]:
                continue
            dist = float(dmat[i, j])
            if dist > tol:
                continue
            recs.append({
                "kind": "similar_properties_different_winner",
                "why": (f"{dist:.3f} standardised units apart in "
                        f"({'+'.join(feats)}) but '{lib[i]}' and '{lib[j]}' "
                        f"win, both uncontested"),
                "chunk_uid": uid[i], "chunk_uid_b": uid[j],
                "feature_distance": dist,
                "best_lib_a": lib[i], "best_lib_b": lib[j],
                "field": d["field"].iloc[i] if "field" in d else None,
                **{f: float(d[f].iloc[i]) for f in feats},
            })
    out = pd.DataFrame(recs)
    return out.sort_values("feature_distance").head(max_pairs) \
        if not out.empty else out


def harmful_treatments(pairs: pd.DataFrame, label: str,
                       rel_threshold: float = -0.05,
                       max_rows: int = 300) -> pd.DataFrame:
    """Pairs where the treatment made compression materially WORSE."""
    if pairs.empty or "rel_ratio" not in pairs.columns:
        return pd.DataFrame()
    bad = pairs[pairs["rel_ratio"] <= rel_threshold]
    if bad.empty:
        return pd.DataFrame()
    cols = BASE_COLS + ["lib_name", "shuffle_on", "shuffle_off",
                        "quantize_on", "quantize_off", "error_bound",
                        "ratio_off", "ratio_on", "d_ratio", "rel_ratio"]
    return _rows(f"{label}_harmful",
                 f"{label} reduced the compression ratio by more than "
                 f"{abs(rel_threshold) * 100:.0f}%",
                 bad.sort_values("rel_ratio").head(max_rows), cols)


def large_prediction_errors(err: pd.DataFrame, metric: str,
                            fold: float = 3.0,
                            max_rows: int = 300) -> pd.DataFrame:
    """Rows the predictor got wrong by more than `fold`x in either direction."""
    if err.empty or "log10_ratio_error" not in err.columns:
        return pd.DataFrame()
    bad = err[err["log10_ratio_error"].abs() >= np.log10(fold)].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["fold_error"] = 10 ** bad["log10_ratio_error"].abs()
    cols = BASE_COLS + ["lib_name", "quantize", "shuffle", "role", "rank",
                        "predicted", "actual", "rel_error", "fold_error"]
    return _rows(f"{metric}_prediction_error",
                 f"the NN's {metric} prediction is off by at least "
                 f"{fold:g}x on this candidate",
                 bad.sort_values("fold_error", ascending=False).head(max_rows),
                 cols)


def large_regret(reg: pd.DataFrame, chunks: pd.DataFrame,
                 threshold: float = 0.25, max_rows: int = 300) -> pd.DataFrame:
    """Chunks where the model's own pick was materially worse than the best
    measured candidate."""
    if reg.empty or "primary_regret" not in reg.columns:
        return pd.DataFrame()
    bad = reg[reg["primary_regret"] >= threshold]
    if bad.empty:
        return pd.DataFrame()
    d = bad.merge(chunks, on="chunk_uid", how="left")
    cols = BASE_COLS + ["primary_lib", "lowest_cost_lib", "best_ratio_lib",
                        "primary_regret", "primary_ratio", "best_ratio_value",
                        "adopted_lib", "adopted_ratio"]
    return _rows("cost_model_regret",
                 f"the model's own pick cost at least "
                 f"{threshold * 100:.0f}% more than the best measured "
                 f"candidate for this chunk",
                 d.sort_values("primary_regret", ascending=False).head(max_rows),
                 cols)


def bound_violations(rows: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    """Lossy candidates whose MEASURED max error exceeds the requested bound.

    Only meas_max_error can witness this: psnr_db is analytical, derived from
    the bound itself, and is structurally unable to see a violation. The
    runtime warns when it cannot achieve a requested bound and quantizes to an
    effective one instead, so a hit here is expected to be rare and is worth
    surfacing when it is not.
    """
    if "meas_max_error_ok" not in rows.columns or "error_bound" not in rows:
        return pd.DataFrame()
    d = rows[(rows["quantize"] == 1) & rows["meas_max_error_ok"].notna()
             & rows["error_bound"].notna()]
    if d.empty:
        return pd.DataFrame()
    bad = d[d["meas_max_error_ok"] > d["error_bound"] * 1.001]
    if bad.empty:
        return pd.DataFrame()
    cols = BASE_COLS + ["lib_name", "shuffle", "error_bound",
                        "meas_max_error_ok", "meas_rmse_ok", "meas_psnr_db_ok",
                        "psnr_db"]
    return _rows("error_bound_violation",
                 "the MEASURED max error exceeds the requested error bound; "
                 "the analytical psnr_db column cannot see this",
                 bad.head(max_rows), cols)


def collect(chunks: pd.DataFrame, win: pd.DataFrame, reg: pd.DataFrame,
            rows: pd.DataFrame, features: Sequence[str],
            shuffle_pairs: pd.DataFrame, quant_pairs: pd.DataFrame,
            errors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [
        extreme_ratio_for_property(chunks, win, features),
        # Within field: controls for the units problem, so a hit means two
        # genuinely comparable chunks of the same variable disagree.
        similar_properties_different_outcome(chunks, win, features,
                                             within_field=True),
        # Across fields, at NEAR-ZERO distance only. The units caveat does not
        # bite here and this is the strongest form of the insufficiency
        # evidence: two chunks whose three features are numerically identical
        # cannot be told apart by any model built on those three features, so
        # whatever separates their ratios is information the features do not
        # carry. The tight radius is what keeps it an identity claim rather
        # than a similarity claim across incompatible units.
        similar_properties_different_outcome(
            chunks, win, features, within_field=False, prop_tol=0.02,
            ratio_fold=1.5, near_quantile=0.0),
        similar_properties_different_codec(chunks, win, features),
        harmful_treatments(shuffle_pairs, "shuffle"),
        harmful_treatments(quant_pairs, "quantization"),
        large_regret(reg, chunks),
        bound_violations(rows),
    ]
    for metric, e in errors.items():
        parts.append(large_prediction_errors(e, metric))
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["kind", "why"])
    return pd.concat(parts, ignore_index=True, sort=False)

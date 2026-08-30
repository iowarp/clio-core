#!/usr/bin/env python3
"""Comparison across several workloads: which relationships transfer.

The question this exists for is Q11 -- whether there are general
scientific-data compressibility indicators, or only per-workload ones. Three
tests, in increasing strength:

  1. Do the per-feature correlations AGREE IN SIGN across workloads? A
     relationship that reverses sign is not a general indicator whatever its
     magnitude.
  2. Do the discovered thresholds land in the same place? Entropy is in
     bits/byte and is directly comparable; MAD and the second derivative are
     in raw data units and are NOT, so their thresholds are compared only
     after per-workload standardisation, and the report says which is which.
  3. Does a model trained on n-1 workloads predict the held-out one? This is
     the only test that can fail for the right reason, and it is the one the
     answer leans on.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from .pipeline import Analysis, CHUNK_TARGET


def pooled_chunks(analyses: Sequence[Analysis]) -> pd.DataFrame:
    """One frame of every workload's chunks, with a `workload` column.

    Also carries per-workload z-scores of each feature, because MAD and the
    second derivative are in raw data units and a pooled model on them would
    be sorting workloads by the physical scale of their variables.
    """
    parts = []
    for a in analyses:
        d = a.tables.get("chunk_properties", pd.DataFrame()).copy()
        if d.empty:
            continue
        d["workload"] = a.ds.name
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    feats = sorted({f for a in analyses for f in a.ds.features})
    for f in feats:
        if f not in out:
            continue
        g = out.groupby("workload")[f]
        sd = g.transform("std").replace(0, np.nan)
        out[f + "_z"] = (out[f] - g.transform("mean")) / sd
    return out


def sign_agreement(analyses: Sequence[Analysis]) -> pd.DataFrame:
    """Per (feature, codec, config) correlation with ratio, side by side."""
    parts = []
    for a in analyses:
        d = a.tables.get("correlations_within_field", pd.DataFrame()).copy()
        if d.empty:
            continue
        d["workload"] = a.ds.name
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    all_ = pd.concat(parts, ignore_index=True)
    keys = [k for k in ("lib_name", "quantize", "shuffle", "feature")
            if k in all_.columns]
    recs: List[dict] = []
    for key, g in all_.groupby(keys, dropna=False):
        if g["workload"].nunique() < 2:
            continue
        r = g["pooled_within_field_r"].dropna()
        if r.empty:
            continue
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update({
            "n_workloads": int(g["workload"].nunique()),
            "median_r": float(r.median()),
            "min_r": float(r.min()), "max_r": float(r.max()),
            "all_same_sign": bool(np.all(np.sign(r) == np.sign(r.iloc[0]))),
            "workloads": "|".join(sorted(g["workload"].unique())),
        })
        recs.append(rec)
    return pd.DataFrame(recs)


def feature_importance_across(analyses: Sequence[Analysis]) -> pd.DataFrame:
    parts = []
    for a in analyses:
        d = a.tables.get("feature_ablation_marginal", pd.DataFrame()).copy()
        if d.empty:
            continue
        d["workload"] = a.ds.name
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def leave_one_workload_out(pooled: pd.DataFrame, features: Sequence[str],
                           use_z: bool = False,
                           seed: int = 0) -> pd.DataFrame:
    """Train on every workload but one, test on the held-out one.

    Run twice by the caller: on raw features, and on per-workload z-scored
    features. The gap between the two is informative -- if only the z-scored
    version transfers, the relationship is about a variable's RELATIVE
    variability within its own workload, not about an absolute MAD value, and
    a deployed selector would need the normalisation.
    """
    feats = [f + ("_z" if use_z else "") for f in features]
    feats = [f for f in feats if f in pooled.columns]
    if not feats or "workload" not in pooled or CHUNK_TARGET not in pooled:
        return pd.DataFrame()
    d = pooled[feats + [CHUNK_TARGET, "workload", "chunk_uid"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    d = d[d[CHUNK_TARGET] > 0]
    if d["workload"].nunique() < 2:
        return pd.DataFrame()
    X = d[feats].to_numpy(dtype=float)
    y = np.log10(d[CHUNK_TARGET].to_numpy(dtype=float))
    recs: List[dict] = []
    for held in sorted(d["workload"].unique()):
        te = (d["workload"] == held).to_numpy()
        tr = ~te
        if tr.sum() < 12 or te.sum() < 4:
            continue
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=seed, n_jobs=1)
        m.fit(X[tr], y[tr])
        recs.append({
            "held_out_workload": held, "features": "+".join(feats),
            "standardised_per_workload": use_z,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_test_chunks": int(d.loc[te, "chunk_uid"].nunique()),
            "r2": float(r2_score(y[te], m.predict(X[te]))),
            "train_r2": float(r2_score(y[tr], m.predict(X[tr]))),
        })
    return pd.DataFrame(recs)


def codec_agreement(analyses: Sequence[Analysis]) -> pd.DataFrame:
    parts = []
    for a in analyses:
        d = a.tables.get("codec_win_rates", pd.DataFrame()).copy()
        if d.empty:
            continue
        d["workload"] = a.ds.name
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    all_ = pd.concat(parts, ignore_index=True)
    return all_.pivot_table(index=["objective", "lib_name"],
                            columns="workload", values="pct_wins",
                            aggfunc="first").reset_index()


def treatment_agreement(analyses: Sequence[Analysis],
                        key: str) -> pd.DataFrame:
    parts = []
    for a in analyses:
        d = a.tables.get(key, pd.DataFrame()).copy()
        if d.empty:
            continue
        d["workload"] = a.ds.name
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def anisotropy_agreement(analyses: Sequence[Analysis]) -> pd.DataFrame:
    """Does the matched-control result reproduce in every run?

    This is the strongest evidence the cross-workload stage can produce, and it
    is why several logs are worth having. A single run's matched-control
    finding is immune to the timestep confound but not to that run: some
    accident of one simulation could produce it. Independent runs agreeing on
    WHICH component compresses better, at every moment of every run, cannot be
    such an accident.
    """
    recs: List[dict] = []
    for a in analyses:
        an = a.facts.get("anisotropy", {})
        if not an.get("available"):
            continue
        recs.append({
            "workload": a.ds.name,
            "n_matched_pairs": an.get("n_matched_pairs"),
            "n_moments": an.get("n_moments"),
            "tightest_tolerance": an.get("tightest_tolerance"),
            "tightest_n_pairs": an.get("tightest_n_pairs"),
            "tightest_max_ratio_fold": an.get("tightest_max_fold"),
            "max_achievable_oof_r2": an.get("max_achievable_oof_r2"),
            "n_stable_component_orderings":
                an.get("n_stable_component_orderings"),
            "field_residual_kruskal_p": an.get("field_residual_kruskal_p"),
            "anisotropic": an.get("anisotropic"),
        })
    return pd.DataFrame(recs)


def component_ordering_replication(
        analyses: Sequence[Analysis]) -> pd.DataFrame:
    """One row per (family, axis pair): which way each run ordered them.

    `n_workloads_agreeing` against `n_workloads` is the whole point. A
    direction that holds in every run is a property of the data layout; one
    that flips between runs was a property of a run.
    """
    parts: List[pd.DataFrame] = []
    for a in analyses:
        od = a.tables.get("component_ordering", pd.DataFrame())
        if od is None or od.empty:
            continue
        d = od.copy()
        d["workload"] = a.ds.name
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    allod = pd.concat(parts, ignore_index=True)
    recs: List[dict] = []
    for (fam, ax_a, ax_b), g in allod.groupby(["family", "axis_a", "axis_b"],
                                              sort=True):
        winners = g["more_compressible_axis"].value_counts()
        top = winners.index[0]
        recs.append({
            "family": fam, "axis_a": ax_a, "axis_b": ax_b,
            "n_workloads": int(len(g)),
            "n_workloads_agreeing": int(winners.iloc[0]),
            "more_compressible_axis": top,
            "unanimous": bool(len(winners) == 1 and len(g) > 1),
            "median_ratio_fold": float(g["median_ratio_fold"].median()),
            "min_ratio_fold": float(g["median_ratio_fold"].min()),
            "max_ratio_fold": float(g["median_ratio_fold"].max()),
            "total_moments": int(g["n_moments"].sum()),
            "max_sign_test_p": float(g["sign_test_p"].max()),
            # The replication test, which is the point of having several runs:
            # each RUN votes once for a direction. It is independent of the
            # per-run tests -- a direction can be weak inside every run and
            # still be unmistakable across them, which is exactly the case a
            # single log cannot distinguish from an accident of that log.
            "replication_sign_test_p": (
                float(sps.binomtest(int(winners.iloc[0]), int(len(g)),
                                    0.5).pvalue) if len(g) > 1 else np.nan),
            "workloads": ", ".join(sorted(g["workload"].astype(str))),
        })
    out = pd.DataFrame(recs)
    return out.sort_values("median_ratio_fold",
                           ascending=False).reset_index(drop=True)


def run(analyses: Sequence[Analysis], outdir: str,
        seed: int = 0) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)
    pooled = pooled_chunks(analyses)
    feats = sorted({f for a in analyses for f in a.ds.features})
    tables: Dict[str, pd.DataFrame] = {
        "pooled_chunk_properties": pooled,
        "correlation_sign_agreement": sign_agreement(analyses),
        "feature_marginal_gains_by_workload":
            feature_importance_across(analyses),
        "codec_win_rate_by_workload": codec_agreement(analyses),
        "shuffle_summary_by_workload":
            treatment_agreement(analyses, "shuffle_summary"),
        "quantization_summary_by_workload":
            treatment_agreement(analyses, "quantization_summary"),
        "anisotropy_by_workload": anisotropy_agreement(analyses),
        "component_ordering_replication":
            component_ordering_replication(analyses),
    }
    lo = [leave_one_workload_out(pooled, feats, use_z=False, seed=seed),
          leave_one_workload_out(pooled, feats, use_z=True, seed=seed)]
    lo = [x for x in lo if not x.empty]
    tables["leave_one_workload_out"] = (
        pd.concat(lo, ignore_index=True) if lo else pd.DataFrame())
    for name, df in tables.items():
        if df is not None and not df.empty:
            df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)

    sa = tables["correlation_sign_agreement"]
    lw = tables["leave_one_workload_out"]
    facts: Dict[str, object] = {
        "workloads": [a.ds.name for a in analyses],
        "n_chunks_total": int(len(pooled)),
        "features_common_to_all": [
            f for f in feats
            if all(f in a.ds.features for a in analyses)],
        "pct_correlations_agreeing_in_sign": (
            float(100.0 * sa["all_same_sign"].mean()) if not sa.empty
            else None),
        "transfer_r2_raw_features": (
            lw[~lw["standardised_per_workload"]]["r2"].to_dict()
            if not lw.empty else {}),
        "transfer_r2_median_raw": (
            float(lw[~lw["standardised_per_workload"]]["r2"].median())
            if not lw.empty and (~lw["standardised_per_workload"]).any()
            else None),
        "transfer_r2_median_standardised": (
            float(lw[lw["standardised_per_workload"]]["r2"].median())
            if not lw.empty and lw["standardised_per_workload"].any()
            else None),
    }
    rep = tables["component_ordering_replication"]
    ani = tables["anisotropy_by_workload"]
    facts["n_workloads_anisotropic"] = (
        int(ani["anisotropic"].sum()) if not ani.empty else 0)
    facts["n_component_orderings_unanimous"] = (
        int(rep["unanimous"].sum()) if not rep.empty else 0)
    if not rep.empty and bool(rep["unanimous"].any()):
        u = rep[rep["unanimous"]].iloc[0]
        facts["replicated_ordering"] = (
            f"{u['family']}: the {u['more_compressible_axis']} component "
            f"compresses better in all {int(u['n_workloads'])} runs "
            f"({int(u['total_moments'])} moments in total), by a median of "
            f"{float(u['median_ratio_fold']):.2f}x "
            f"(range {float(u['min_ratio_fold']):.2f}-"
            f"{float(u['max_ratio_fold']):.2f}x); "
            f"replication sign test over runs p = "
            f"{float(u['replication_sign_test_p']):.3g}")
    _report(analyses, tables, facts, outdir)
    return {"tables": tables, "facts": facts}


def _report(analyses, tables, facts, outdir: str) -> None:
    from .reporting import _md_table
    L: List[str] = []
    w = L.append
    names = ", ".join(f"`{a.ds.name}`" for a in analyses)
    w("# Cross-workload comparison\n")
    w(f"Workloads: {names} -- {facts['n_chunks_total']} chunks in total.\n")
    w("The question is whether any of the relationships found per workload "
      "are general scientific-data compressibility indicators, or whether "
      "each workload has its own.\n")
    w("\n> **Units caveat, and it governs everything below.** `entropy` is "
      "bits/byte and is directly comparable across workloads. `mad` and "
      "`second_deriv` are in RAW DATA UNITS, so a threshold on them from one "
      "workload has no meaning in another whose variables carry different "
      "physical scales. Transfer is therefore evaluated twice -- on raw "
      "features and on per-workload standardised ones -- and the gap between "
      "the two is itself the finding.\n")

    w("\n## 1. Do the correlations agree in sign?\n")
    sa = tables["correlation_sign_agreement"]
    if sa.empty:
        w("_Not computable: fewer than two workloads produced within-field "
          "correlations._\n")
    else:
        w(f"{facts['pct_correlations_agreeing_in_sign']:.0f}% of "
          f"(feature x configuration) cells keep the same sign across "
          f"workloads.\n")
        w(_md_table(sa.sort_values(["feature", "all_same_sign"]),
                    ["feature", "lib_name", "quantize", "shuffle",
                     "n_workloads", "median_r", "min_r", "max_r",
                     "all_same_sign"], n=30))

    w("\n## 2. Does a model transfer to an unseen workload?\n")
    lw = tables["leave_one_workload_out"]
    if lw.empty:
        w("_Not computable: at least two workloads with chunk-level targets "
          "are required._\n")
    else:
        w("Train on every workload but one, test on the held-out one. "
          "R^2 on log10(best achievable ratio).\n")
        w(_md_table(lw, ["held_out_workload", "standardised_per_workload",
                         "features", "n_train", "n_test", "n_test_chunks",
                         "train_r2", "r2"], n=30))
        raw = facts.get("transfer_r2_median_raw")
        std = facts.get("transfer_r2_median_standardised")
        if raw is not None and std is not None:
            w(f"\nMedian transfer R^2: **{raw:.2f}** on raw features, "
              f"**{std:.2f}** on per-workload standardised features. "
              f"*(If only the standardised version transfers, the general "
              f"indicator is a variable's RELATIVE variability inside its own "
              f"workload, not an absolute MAD value -- and a deployed "
              f"selector would need that normalisation, which the current "
              f"feature pipeline does not compute.)*\n")

    w("\n## 3. Which codec wins where\n")
    w(_md_table(tables["codec_win_rate_by_workload"], n=40))

    w("\n## 4. Do the transforms behave the same way?\n")
    for label, key in (("Shuffle", "shuffle_summary_by_workload"),
                       ("Quantization", "quantization_summary_by_workload")):
        d = tables[key]
        w(f"\n### {label}\n")
        if d.empty:
            w("_No paired comparison available in any workload._\n")
        else:
            w(_md_table(d, ["workload", "treatment", "lib_name", "n_pairs",
                            "ratio_median_rel", "ratio_pct_helps"], n=40))

    w("\n## 5. Does the matched-control result replicate?\n")
    ani = tables["anisotropy_by_workload"]
    rep = tables["component_ordering_replication"]
    if ani.empty:
        w("_No workload produced matched controls (two chunks sharing a run, "
          "a timestep and a size)._\n")
    else:
        from .reporting import section_ref
        w(f"{section_ref('Same moment, different field')} of each "
          "per-workload report compares chunks written at "
          "the SAME timestep, which is the one comparison the simulation-clock "
          "confound cannot reach. It is still, in a single log, one run -- an "
          "accident of that simulation could produce it. Independent runs "
          "agreeing is what rules that out.\n")
        w(_md_table(ani, ["workload", "n_matched_pairs", "n_moments",
                          "tightest_n_pairs", "tightest_max_ratio_fold",
                          "max_achievable_oof_r2",
                          "n_stable_component_orderings",
                          "field_residual_kruskal_p", "anisotropic"], n=40))
        w(f"\n**{facts.get('n_workloads_anisotropic', 0)} of {len(ani)} "
          f"runs** show chunks with matching statistics and different "
          f"compressibility.\n")
    if not rep.empty:
        w("\n### Which component compresses better, run by run\n")
        w("`unanimous` means every run that measured this pair ordered it the "
          "same way. A direction that survives independent runs is a property "
          "of how the data is laid out, not of one simulation.\n")
        w("`replication_sign_test_p` is over RUNS -- each run votes once for "
          "a direction -- and is independent of the per-run tests: a "
          "direction can be weak inside every run and still be unmistakable "
          "across them, which is the case a single log cannot tell apart "
          "from an accident of that log.\n")
        w(_md_table(rep, ["family", "axis_a", "axis_b", "n_workloads",
                          "n_workloads_agreeing", "more_compressible_axis",
                          "unanimous", "median_ratio_fold", "min_ratio_fold",
                          "max_ratio_fold", "total_moments",
                          "max_sign_test_p", "replication_sign_test_p"],
                    n=30))
        if facts.get("replicated_ordering"):
            w(f"\n> **Replicated:** {facts['replicated_ordering']}.\n")

    w("\n## 6. Feature importance by workload\n")
    w(_md_table(tables["feature_marginal_gains_by_workload"],
                ["workload", "model", "feature",
                 "mean_marginal_oof_r2_gain"], n=40))

    w("\n## Q11: do these relationships generalise?\n")
    raw = facts.get("transfer_r2_median_raw")
    if raw is None:
        w("**Not answerable from this run** -- pass more than one exploration "
          "CSV.\n")
    else:
        agree = facts.get("pct_correlations_agreeing_in_sign")
        w(f"Sign agreement across workloads: "
          f"{agree:.0f}% of cells. Median leave-one-workload-out R^2: "
          f"{raw:.2f} (raw), "
          f"{facts.get('transfer_r2_median_standardised', float('nan')):.2f} "
          f"(standardised). Read the per-workload rows above rather than the "
          f"median alone: a single workload that fails to transfer is the "
          f"interesting case, and the median hides it.\n")

    with open(os.path.join(outdir, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    import json
    from .reporting import _jsonable
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(_jsonable({"facts": facts,
                             "tables": {k: v for k, v in tables.items()
                                        if k != "pooled_chunk_properties"}}),
                  fh, indent=1)

#!/usr/bin/env python3
"""Is it the data, or is it the clock?

This module exists because of a failure the rest of the pipeline could not
see. On a single-run sweep of an evolving simulation, the intrinsic data
properties are very nearly a FUNCTION OF THE TIMESTEP -- on the Nyx Sedov log
this was developed against, corr(entropy, timestep) is between +0.983 and
+1.0000 inside every one of the six fields. Compressibility falls over the
same run. So entropy and ratio correlate strongly, and essentially none of
that correlation is evidence that entropy causes anything: both ride the
blast wave spreading out.

Two consequences that motivated every function below:

  * `statistics.within_field_correlation` CANNOT catch this. It stratifies on
    `field`, and within a field the entropy series IS the time series, so the
    "controlled" correlation is computed inside the confound rather than
    across it.
  * A model scored by group-holdout on chunks cannot catch it either. Chunks
    at nearby timesteps in different fields are near-duplicates, so holding
    out a chunk still leaves its neighbours in training. Grouping on
    `chunk_uid` prevents memorising a chunk; it does nothing about memorising
    a moment.

The test is deliberately blunt and hard to argue with: give a model NOTHING
but the confounder -- one integer -- and see whether it beats the three
features. If it does, the features have not been shown to carry information
about compressibility that the clock does not already carry.

None of this says entropy is irrelevant. It says THIS LOG cannot separate the
two, and what would: a sweep at fixed simulation time across many chunks, or
several runs whose evolution differs.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .modeling import grouped_cv_r2

#: Metadata that is not a data property but may explain the outcome anyway.
CANDIDATES = ["timestep", "field", "chunk_bytes", "chunk_id"]


def _usable(cw: pd.DataFrame, col: str) -> bool:
    if col not in cw.columns:
        return False
    s = cw[col].dropna()
    return len(s) >= 12 and s.nunique() >= 2


def _numeric_codes(cw: pd.DataFrame, col: str) -> np.ndarray:
    """Categorical metadata as codes, so it can enter a tree.

    Only ever used inside a TREE, never in a correlation: the codes carry no
    order and a Pearson r over them would be meaningless.
    """
    if pd.api.types.is_numeric_dtype(cw[col]):
        return cw[col].to_numpy(dtype=float)
    return cw[col].astype("category").cat.codes.to_numpy(dtype=float)


def confounder_power(cw: pd.DataFrame, features: Sequence[str], target: str,
                     seed: int = 0) -> pd.DataFrame:
    """Out-of-fold R^2 of the features, of each confounder ALONE, and of both.

    A confounder that matches or beats the feature set on its own is the
    headline result; `verdict` says so in words so a reader cannot miss it.
    """
    feats = [f for f in features if f in cw.columns]
    if not feats or target not in cw.columns:
        return pd.DataFrame()
    base = grouped_cv_r2(cw, feats, target, "forest", True, seed=seed)
    if not base.get("available"):
        return pd.DataFrame()
    recs: List[dict] = [{
        "predictor": "+".join(feats), "kind": "data properties",
        "oof_r2": base["oof_r2"], "n_chunks": base["n_chunks"],
        "verdict": "the data-property baseline",
    }]
    d = cw.copy()
    for c in CANDIDATES:
        if not _usable(d, c):
            continue
        code = f"__{c}_code"
        d[code] = _numeric_codes(d, c)
        alone = grouped_cv_r2(d, [code], target, "forest", True, seed=seed)
        if not alone.get("available"):
            continue
        both = grouped_cv_r2(d, list(feats) + [code], target, "forest", True,
                             seed=seed)
        gain = (both["oof_r2"] - alone["oof_r2"]) if both.get("available") \
            else np.nan
        if alone["oof_r2"] >= base["oof_r2"]:
            verdict = (f"`{c}` ALONE predicts the outcome at least as well as "
                       f"all the data properties together -- the property "
                       f"effect is not separable from it in this log")
        elif alone["oof_r2"] >= 0.6 * base["oof_r2"]:
            verdict = (f"`{c}` alone recovers most of the fit; the properties "
                       f"add little the metadata does not already carry")
        else:
            verdict = f"`{c}` alone explains substantially less"
        recs.append({
            "predictor": c, "kind": "metadata (NOT a data property)",
            "oof_r2": alone["oof_r2"], "n_chunks": alone["n_chunks"],
            "oof_r2_with_features_too": (both["oof_r2"]
                                         if both.get("available") else np.nan),
            "features_gain_over_metadata_alone": gain,
            "verdict": verdict,
        })
    return pd.DataFrame(recs)


def collinearity(cw: pd.DataFrame, features: Sequence[str],
                 by: str = "field", confounder: str = "timestep",
                 min_n: int = 4) -> pd.DataFrame:
    """How tightly each feature tracks the confounder INSIDE each stratum.

    This is the number that decides whether a within-`by` correlation is a
    control at all. If a feature is collinear with the confounder inside every
    stratum, then "within field" and "within one time series" are the same
    thing and the stratification controls for nothing.
    """
    if confounder not in cw.columns or by not in cw.columns:
        return pd.DataFrame()
    recs: List[dict] = []
    for f in [x for x in features if x in cw.columns]:
        for stratum, g in cw.groupby(by, dropna=False):
            g = g[[f, confounder]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(g) < min_n or g[f].nunique() < 2 \
                    or g[confounder].nunique() < 2:
                continue
            r, p = sps.pearsonr(g[f].to_numpy(dtype=float),
                                g[confounder].to_numpy(dtype=float))
            recs.append({"feature": f, "stratum_column": by,
                         "stratum": str(stratum), "confounder": confounder,
                         "n": int(len(g)), "pearson_r": float(r),
                         "p": float(p), "collinear": bool(abs(r) >= 0.95)})
    out = pd.DataFrame(recs)
    if not out.empty:
        out.attrs["pct_collinear"] = float(100.0 * out["collinear"].mean())
    return out


def partial_correlations(cw: pd.DataFrame, features: Sequence[str],
                         target: str, confounder: str = "timestep",
                         log_target: bool = True) -> pd.DataFrame:
    """Feature vs target, raw and after removing the confounder from both.

    Residualisation is by OLS on the confounder (and its square, so a
    monotone-but-curved trend is removed rather than merely tilted). A SIGN
    FLIP between the two columns is the finding: it means the raw correlation
    was carried by the confounder and points the other way once it is gone.
    """
    feats = [f for f in features if f in cw.columns]
    need = feats + [target, confounder]
    if any(c not in cw.columns for c in need):
        return pd.DataFrame()
    d = cw[need].replace([np.inf, -np.inf], np.nan).dropna()
    if log_target:
        d = d[d[target] > 0]
    if len(d) < 12 or d[confounder].nunique() < 3:
        return pd.DataFrame()
    y = np.log10(d[target].to_numpy(dtype=float)) if log_target \
        else d[target].to_numpy(dtype=float)
    c = d[confounder].to_numpy(dtype=float)
    Z = np.c_[np.ones(len(c)), c, c ** 2]

    def resid(v):
        b, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b

    ry = resid(y)
    recs: List[dict] = []
    for f in feats:
        x = d[f].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            continue
        r0, p0 = sps.pearsonr(x, y)
        rx = resid(x)
        if np.std(rx) == 0 or np.std(ry) == 0:
            continue
        r1, p1 = sps.pearsonr(rx, ry)
        recs.append({
            "feature": f, "target": target, "confounder": confounder,
            "n_chunks": int(len(d)),
            "raw_r": float(r0), "raw_p": float(p0),
            "partial_r": float(r1), "partial_p": float(p1),
            "sign_flipped": bool(np.sign(r0) != np.sign(r1)),
            "attenuation": float(abs(r0) - abs(r1)),
        })
    return pd.DataFrame(recs)


def stratified_holdout(cw: pd.DataFrame, features: Sequence[str],
                       target: str, by: str = "timestep",
                       seed: int = 0) -> Dict[str, object]:
    """Re-score the model with the CONFOUNDER as the grouping variable.

    Grouping on `chunk_uid` stops a chunk being memorised. Grouping on the
    timestep stops a MOMENT being memorised -- which is the leak that matters
    when six fields at one timestep are near-duplicates of one another. The
    drop between the two is the size of that leak.
    """
    feats = [f for f in features if f in cw.columns]
    if by not in cw.columns or not feats:
        return {"available": False, "reason": f"`{by}` not usable"}
    d = cw.copy()
    d["__grp"] = d[by].astype(str)
    if d["__grp"].nunique() < 4:
        return {"available": False,
                "reason": f"only {d['__grp'].nunique()} distinct {by} values"}
    by_chunk = grouped_cv_r2(d, feats, target, "forest", True, seed=seed)
    d2 = d.copy()
    d2["chunk_uid"] = d2["__grp"]          # group by the confounder instead
    by_conf = grouped_cv_r2(d2, feats, target, "forest", True, seed=seed)
    if not (by_chunk.get("available") and by_conf.get("available")):
        return {"available": False, "reason": "could not cross-validate"}
    return {
        "available": True, "confounder": by, "features": feats,
        "oof_r2_grouped_by_chunk": by_chunk["oof_r2"],
        "oof_r2_grouped_by_confounder": by_conf["oof_r2"],
        "drop": by_chunk["oof_r2"] - by_conf["oof_r2"],
        "n_groups": int(d["__grp"].nunique()),
    }


def nearest_neighbour_structure(cw: pd.DataFrame, features: Sequence[str],
                                by: str = "field") -> Dict[str, object]:
    """Where does each chunk's nearest neighbour in feature space live?

    If most chunks' nearest neighbour is a DIFFERENT field at the SAME
    timestep, the sample is a set of moments observed several times over, not
    a set of independent chunks, and every effective sample size in the report
    is smaller than its printed n.
    """
    feats = [f for f in features if f in cw.columns]
    if not feats or by not in cw.columns or "timestep" not in cw.columns:
        return {}
    d = cw[feats + [by, "timestep", "chunk_uid"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(d) < 6:
        return {}
    Z = d[feats].to_numpy(dtype=float)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Z - Z.mean(axis=0)) / sd
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = np.argmin(D, axis=1)
    same_step = (d["timestep"].to_numpy()[nn] == d["timestep"].to_numpy())
    same_by = (d[by].to_numpy()[nn] == d[by].to_numpy())
    return {
        "n_chunks": int(len(d)),
        "pct_nn_same_timestep_different_field": float(
            100.0 * np.mean(same_step & ~same_by)),
        "pct_nn_same_field": float(100.0 * np.mean(same_by)),
        "median_nn_distance": float(np.median(np.min(D, axis=1))),
        "pct_nn_closer_than_0.05": float(
            100.0 * np.mean(np.min(D, axis=1) < 0.05)),
    }


def summarize(power: pd.DataFrame, collin: pd.DataFrame,
              partial: pd.DataFrame, strat: Dict[str, object],
              nn: Dict[str, object]) -> Dict[str, object]:
    """One verdict, plus the evidence for it."""
    out: Dict[str, object] = {"confounded": False, "reasons": []}
    if not power.empty:
        base = power[power["kind"] == "data properties"]["oof_r2"]
        meta = power[power["kind"] != "data properties"]
        if len(base) and not meta.empty:
            b = float(base.iloc[0])
            beats = meta[meta["oof_r2"] >= b]
            out["feature_oof_r2"] = b
            out["best_metadata_predictor"] = (
                str(meta.loc[meta["oof_r2"].idxmax(), "predictor"]))
            out["best_metadata_oof_r2"] = float(meta["oof_r2"].max())
            if len(beats):
                out["confounded"] = True
                out["reasons"].append(
                    "metadata alone (" + ", ".join(
                        f"`{r}`" for r in beats["predictor"]) +
                    ") predicts the outcome at least as well as all three "
                    "data properties together")
    if not collin.empty:
        pct = float(100.0 * collin["collinear"].mean())
        out["pct_feature_stratum_pairs_collinear_with_confounder"] = pct
        if pct >= 30:
            out["confounded"] = True
            out["reasons"].append(
                f"{pct:.0f}% of (feature, stratum) pairs are collinear "
                f"(|r| >= 0.95) with the confounder inside the stratum, so "
                f"stratifying on it does not control for the confounder")
    if not partial.empty:
        flips = partial[partial["sign_flipped"]]
        out["n_features_whose_correlation_flips_sign"] = int(len(flips))
        out["features_flipping"] = flips["feature"].tolist()
        if len(flips):
            out["confounded"] = True
            out["reasons"].append(
                "the correlation of " + ", ".join(
                    f"`{f}`" for f in flips["feature"]) +
                " with the outcome REVERSES sign once the confounder is "
                "removed from both")
    if strat.get("available"):
        out["oof_r2_grouped_by_chunk"] = strat["oof_r2_grouped_by_chunk"]
        out["oof_r2_grouped_by_confounder"] = \
            strat["oof_r2_grouped_by_confounder"]
        out["holdout_drop"] = strat["drop"]
        if strat["drop"] >= 0.25:
            out["confounded"] = True
            out["reasons"].append(
                f"regrouping the cross-validation on `{strat['confounder']}` "
                f"instead of the chunk drops out-of-fold R^2 from "
                f"{strat['oof_r2_grouped_by_chunk']:.2f} to "
                f"{strat['oof_r2_grouped_by_confounder']:.2f}")
    if nn:
        out.update(nn)
        if nn.get("pct_nn_same_timestep_different_field", 0) >= 40:
            out["reasons"].append(
                f"{nn['pct_nn_same_timestep_different_field']:.0f}% of chunks "
                f"have their nearest neighbour in feature space in a "
                f"different field at the SAME timestep, so the effective "
                f"sample is moments, not chunks")
    return out

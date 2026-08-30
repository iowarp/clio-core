#!/usr/bin/env python3
"""Same moment, different field: the one clean control a single log contains.

The confound section (`confounds.py`) establishes the problem. On one run of
an evolving simulation, entropy, MAD and the second derivative are very nearly
FUNCTIONS OF THE TIMESTEP, and compressibility falls over the same run, so a
correlation between a data property and the ratio is shared drift and carries
no mechanism.
That objection has exactly one answer available inside a single log: compare
chunks written AT THE SAME TIMESTEP. The clock is then held exactly -- not
statistically -- constant, and whatever difference in compressibility remains
is a difference between the data and nothing else.

This module builds those matched groups and asks the question the rest of the
pipeline cannot ask cleanly:

    given two chunks whose statistics agree, does the ratio agree?

On the Nyx Sedov log this was developed against, it does not, and the failure
is structured rather than noisy. The blast is spherically symmetric, so its
three momentum components are the same field rotated: at every shared timestep
their MAD agrees to 3e-9 relative and their second derivative to 2e-8, and
`ymom` and `zmom` are indistinguishable in all three features. Their ratios are
not. zmom compresses 1.76-1.80x better than ymom at all six shared timesteps,
and xmom -- which the features rank as the ROUGHEST of the three, with a 58%
larger second derivative -- compresses best of all, up to 2.45x above ymom.
Within any one field, a rising second derivative tracks a falling ratio. Across
components at fixed time it tracks the opposite. That is a sign reversal at the
most tightly controlled comparison the log admits.

The conclusion is narrow and does not depend on a model: all three features are
invariant to something the compressor is not. Byte-histogram entropy is
permutation invariant; MAD is permutation invariant; the second derivative is a
lag-1 stencil along the FLATTENED buffer, so it sees one axis of a
three-dimensional field. Compression ratio is invariant to none of those. Where
a field's orientation relative to the memory layout differs, the features are
identical by construction and the ratio is free to differ -- and here it does,
by up to 2.45x.

What is measured here, and what is only hypothesised, kept apart:

  * MEASURED: matched pairs, their feature agreement in each feature's own
    units, their ratio disagreement, and whether the direction is stable across
    independent timesteps (a sign test over moments, not over rows).
  * MEASURED: the ceiling. Chunks that agree in every feature must receive the
    same prediction from ANY model built on those features, so the spread
    inside a feature-equivalence class is irreducible error. That converts to
    an upper bound on out-of-fold R^2 which no amount of model tuning can pass.
  * MEASURED: whether the residuals of a fitted model are systematically
    ordered by field once the timestep is pinned -- the general form of the
    finding, for workloads with no vector fields in them.
  * HYPOTHESISED, and labelled as such wherever it is printed: that the driver
    is the interaction between a field's dominant variation axis and the
    row-major flattening, which sets the match lengths the LZ stage can use.
    This module cannot test that -- it never sees the buffer. It is stated
    because it is falsifiable and cheap to falsify: add second-derivative
    stencils at lag nx and lag nx*ny and see whether they separate the
    components. If they do not, the hypothesis is wrong and the finding stands
    regardless.

Nothing here is specific to Nyx. Vector families are DETECTED from the field
names, matched groups are built from whatever grouping columns the log carries,
and every output says how many moments it rests on.
"""
from __future__ import annotations

import itertools
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from .modeling import oof_predictions

#: "The same statistics", tested as a RELATIVE difference in each feature's own
#: units. A standardised distance is unusable for an identity claim: it divides
#: by a global SD that one outlier cluster can inflate several-fold, which lets
#: pairs differing by tens of percent look identical. See counterexamples.py.
TOLERANCES: Tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)

#: Below this, a tolerance row is reporting an accident rather than a rate.
MIN_PAIRS = 3

#: What makes two chunks a matched control: same run, same moment, same size.
#: Only the field -- and, where a log has several, the block -- may vary.
GROUP_KEYS: Tuple[str, ...] = ("workload", "timestep", "chunk_bytes")

#: A ratio disagreement below this is not worth calling a disagreement.
INTERESTING_FOLD = 1.25

# `E_x`, `B-y`, `velocity_z`, `Ez`
_AXIS_SUFFIX = re.compile(r"^(?P<base>.+?)[_-]?(?P<axis>[xyz])$")
# `xmom`, `y_velocity`, `zmom`
_AXIS_PREFIX = re.compile(r"^(?P<axis>[xyz])[_-]?(?P<base>.+)$")


def _rel(a: float, b: float) -> float:
    """Relative difference, symmetric, safe at zero."""
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _group_keys(cw: pd.DataFrame) -> List[str]:
    return [k for k in GROUP_KEYS if k in cw.columns and cw[k].notna().any()]


# ---------------------------------------------------------------- matching
def matched_pairs(cw: pd.DataFrame, features: Sequence[str],
                  target: str) -> pd.DataFrame:
    """Every pair of chunks from the same run, moment and size.

    One row per pair, carrying each feature's value on both sides, the largest
    relative disagreement across the features, and the ratio fold. This is the
    raw material for everything below and is written out whole, because the
    interesting rows are the ones a summary would average away.
    """
    feats = [f for f in features if f in cw.columns]
    keys = _group_keys(cw)
    if not feats or target not in cw.columns or "timestep" not in keys:
        return pd.DataFrame()
    cols = list(dict.fromkeys(
        keys + feats + [target, "chunk_uid", "field", "chunk_id",
                        "best_ratio_lib"]))
    d = cw[[c for c in cols if c in cw.columns]].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=feats + [target, "timestep"])
    d = d[d[target] > 0]
    if len(d) < 4:
        return pd.DataFrame()

    recs: List[dict] = []
    for gk, g in d.groupby(keys, dropna=False, sort=True):
        if len(g) < 2:
            continue
        g = g.sort_values("chunk_uid")
        gk = gk if isinstance(gk, tuple) else (gk,)
        gid = "|".join(f"{k}={v}" for k, v in zip(keys, gk))
        for i, j in itertools.combinations(range(len(g)), 2):
            a, b = g.iloc[i], g.iloc[j]
            rel = {f: _rel(float(a[f]), float(b[f])) for f in feats}
            ra, rb = float(a[target]), float(b[target])
            hi, lo = max(ra, rb), min(ra, rb)
            recs.append({
                "group": gid,
                **{k: v for k, v in zip(keys, gk)},
                "chunk_uid_a": a["chunk_uid"], "chunk_uid_b": b["chunk_uid"],
                "field_a": a.get("field"), "field_b": b.get("field"),
                "same_field": bool(a.get("field") == b.get("field")),
                "ratio_a": ra, "ratio_b": rb,
                "ratio_fold": float(hi / lo) if lo > 0 else np.nan,
                # Signed and in the modelling target's units, so it can be
                # summed and sign-tested without re-deriving a direction.
                "log10_ratio_gap": float(np.log10(ra) - np.log10(rb)),
                "max_rel_feature_diff": float(max(rel.values())),
                **{f"{f}_a": float(a[f]) for f in feats},
                **{f"{f}_b": float(b[f]) for f in feats},
                **{f"{f}_rel_diff": rel[f] for f in feats},
                "best_lib_a": a.get("best_ratio_lib"),
                "best_lib_b": b.get("best_ratio_lib"),
                "same_best_codec": bool(
                    a.get("best_ratio_lib") == b.get("best_ratio_lib")),
            })
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values(
        ["max_rel_feature_diff", "ratio_fold"],
        ascending=[True, False]).reset_index(drop=True)


def feature_blind_spread(pairs: pd.DataFrame,
                         tolerances: Sequence[float] = TOLERANCES
                         ) -> pd.DataFrame:
    """How far apart the ratio can be when the statistics are this close.

    Read down the tolerance column: if the fold does not collapse toward 1.0 as
    the feature agreement tightens, the features are not the thing setting the
    ratio. A row is emitted only where enough pairs support it.
    """
    if pairs.empty or "max_rel_feature_diff" not in pairs:
        return pd.DataFrame()
    recs: List[dict] = []
    for tol in sorted(tolerances):
        s = pairs[pairs["max_rel_feature_diff"] <= tol]
        s = s[np.isfinite(s["ratio_fold"])]
        if len(s) < MIN_PAIRS:
            continue
        worst = s.loc[s["ratio_fold"].idxmax()]
        uids = pd.unique(pd.concat([s["chunk_uid_a"], s["chunk_uid_b"]]))
        recs.append({
            "max_rel_feature_diff_at_most": tol,
            "n_pairs": int(len(s)), "n_chunks": int(len(uids)),
            "n_moments": int(s["group"].nunique()),
            "median_ratio_fold": float(s["ratio_fold"].median()),
            "p90_ratio_fold": float(s["ratio_fold"].quantile(0.9)),
            "max_ratio_fold": float(s["ratio_fold"].max()),
            "pct_pairs_above_1.25x": float(
                100.0 * (s["ratio_fold"] >= INTERESTING_FOLD).mean()),
            "pct_pairs_different_field": float(
                100.0 * (~s["same_field"]).mean()),
            "worst_pair": f"{worst['chunk_uid_a']} vs {worst['chunk_uid_b']}",
        })
    return pd.DataFrame(recs)


# ------------------------------------------------------------- the ceiling
def _classes(g: pd.DataFrame, feats: Sequence[str], tol: float) -> np.ndarray:
    """Single-linkage components of "agrees within `tol` in every feature"."""
    n = len(g)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    vals = {f: g[f].to_numpy(dtype=float) for f in feats}
    for i, j in itertools.combinations(range(n), 2):
        if all(_rel(vals[f][i], vals[f][j]) <= tol for f in feats):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def explainable_ceiling(cw: pd.DataFrame, features: Sequence[str],
                        target: str,
                        tolerances: Sequence[float] = TOLERANCES
                        ) -> pd.DataFrame:
    """An upper bound on R^2 that no model on these features can pass.

    Chunks whose feature vectors are identical are, to any function of those
    features, the SAME INPUT: the model must give them one number, so the
    spread inside such a class is irreducible. Summing that spread over the
    classes gives residual sum of squares >= SS_within, hence

        R^2 <= 1 - SS_within / SS_total

    computed on log10(target), the same scale the models are scored on.

    Two deliberate conservatisms, both of which push the bound UP (so a low
    bound is a strong statement):

      * classes are formed only INSIDE a matched group, so two chunks with
        identical features at different timesteps are never merged even though
        a model could not tell them apart either;
      * at tolerance 0 the bound is exact; at a loose tolerance it assumes a
        model cannot resolve differences below `tol`, which is an assumption
        and is flagged as one by `near_exact`.
    """
    feats = [f for f in features if f in cw.columns]
    keys = _group_keys(cw)
    if not feats or target not in cw.columns or not keys:
        return pd.DataFrame()
    d = cw[[c for c in dict.fromkeys(keys + feats + [target, "chunk_uid"])
            if c in cw.columns]].replace([np.inf, -np.inf], np.nan).dropna(
        subset=feats + [target])
    d = d[d[target] > 0]
    if len(d) < 8:
        return pd.DataFrame()
    y = np.log10(d[target].to_numpy(dtype=float))
    ss_total = float(np.sum((y - y.mean()) ** 2))
    if ss_total <= 0:
        return pd.DataFrame()

    recs: List[dict] = []
    for tol in sorted(tolerances):
        ss_within, n_multi, n_in_multi, n_classes = 0.0, 0, 0, 0
        for _, g in d.groupby(keys, dropna=False, sort=True):
            lab = _classes(g, feats, tol) if len(g) > 1 else np.zeros(len(g))
            yy = np.log10(g[target].to_numpy(dtype=float))
            for c in np.unique(lab):
                m = lab == c
                n_classes += 1
                if m.sum() > 1:
                    n_multi += 1
                    n_in_multi += int(m.sum())
                    ss_within += float(np.sum((yy[m] - yy[m].mean()) ** 2))
        if n_multi == 0:
            continue
        recs.append({
            "max_rel_feature_diff_at_most": tol,
            "near_exact": bool(tol <= 1e-4),
            "n_classes": n_classes,
            "n_classes_with_more_than_one_chunk": n_multi,
            "n_chunks_in_those_classes": n_in_multi,
            "n_chunks": int(len(d)),
            "irreducible_ss_fraction": float(ss_within / ss_total),
            "max_achievable_oof_r2": float(1.0 - ss_within / ss_total),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------- vector-field families
def _split_axis(field: str) -> Optional[Tuple[str, str]]:
    for rx in (_AXIS_SUFFIX, _AXIS_PREFIX):
        m = rx.match(str(field))
        if m:
            return m.group("base"), m.group("axis").lower()
    return None


def component_families(cw: pd.DataFrame) -> pd.DataFrame:
    """Fields that are components of one vector, detected from their names.

    A family is only accepted when at least TWO distinct axes share a base,
    which is what keeps `density` (ends in 'y') and `flux` (ends in 'x') from
    being mistaken for components. Nothing here is workload-specific: it finds
    `xmom/ymom/zmom` and `E_x/E_y/E_z` by the same rule.
    """
    if "field" not in cw.columns:
        return pd.DataFrame()
    recs: List[dict] = []
    for fld in sorted(cw["field"].dropna().astype(str).unique()):
        sp = _split_axis(fld)
        if sp:
            recs.append({"base": sp[0], "axis": sp[1], "field": fld,
                         "n_chunks": int((cw["field"] == fld).sum())})
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    keep = out.groupby("base")["axis"].nunique()
    out = out[out["base"].isin(keep[keep >= 2].index)]
    return out.sort_values(["base", "axis"]).reset_index(drop=True)


def component_ordering(pairs: pd.DataFrame, fams: pd.DataFrame,
                       features: Sequence[str]) -> pd.DataFrame:
    """Is one component of a vector reliably more compressible than another?

    By symmetry the components of an isotropic vector field carry the same
    distribution, and the features say so. If the RATIO nevertheless orders
    them the same way at moment after moment, the ordering is a property of how
    the field sits in memory, not of its statistics.

    The sign test is over MOMENTS -- each timestep votes once. Two components
    at the same timestep are one observation, not one per candidate row.
    """
    if pairs.empty or fams.empty:
        return pd.DataFrame()
    ax = dict(zip(fams["field"].astype(str), fams["axis"]))
    base = dict(zip(fams["field"].astype(str), fams["base"]))
    p = pairs[(~pairs["same_field"])].copy()
    p["fa"] = p["field_a"].astype(str)
    p["fb"] = p["field_b"].astype(str)
    p = p[p["fa"].isin(ax) & p["fb"].isin(ax)]
    p = p[[base.get(a) == base.get(b) for a, b in zip(p["fa"], p["fb"])]]
    if p.empty:
        return pd.DataFrame()

    recs: List[dict] = []
    for (bs, lo_ax, hi_ax), g in p.assign(
            _b=[base[f] for f in p["fa"]],
            _lo=[min(ax[a], ax[b]) for a, b in zip(p["fa"], p["fb"])],
            _hi=[max(ax[a], ax[b]) for a, b in zip(p["fa"], p["fb"])],
    ).groupby(["_b", "_lo", "_hi"], sort=True):
        # Orient every row the same way: positive means the alphabetically
        # first axis compressed better, so the votes are commensurable.
        sign = np.where([ax[f] == lo_ax for f in g["fa"]], 1.0, -1.0)
        delta = g["log10_ratio_gap"].to_numpy(dtype=float) * sign
        by_moment = pd.Series(delta, index=g["group"].to_numpy()).groupby(
            level=0).median()
        moved = by_moment[by_moment != 0]
        n = int(moved.size)
        if n < 2:
            continue
        k = int((moved > 0).sum())
        winner = lo_ax if k * 2 > n else hi_ax
        recs.append({
            "family": bs, "axis_a": lo_ax, "axis_b": hi_ax,
            "n_moments": n,
            "n_moments_axis_a_better": k,
            "more_compressible_axis": winner,
            "median_ratio_fold": float(np.median(10.0 ** np.abs(moved))),
            "max_ratio_fold": float(np.max(10.0 ** np.abs(moved))),
            "sign_test_p": float(sps.binomtest(k, n, 0.5).pvalue),
            "median_max_rel_feature_diff": float(
                g["max_rel_feature_diff"].median()),
            "features_agree": bool(g["max_rel_feature_diff"].median() <= 1e-3),
            "features": "+".join(features),
        })
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values("median_ratio_fold", ascending=False).reset_index(
        drop=True)


# ------------------------------------------------- the general form of it
def residual_by_field(cw: pd.DataFrame, features: Sequence[str], target: str,
                      seed: int = 0) -> pd.DataFrame:
    """After the features have had their say, is the error ordered by field?

    Fit out of fold on the features, then compare each chunk's residual to the
    MEAN RESIDUAL OF ITS MOMENT. Centring inside the matched group removes the
    timestep entirely, so what is left is the part of the outcome that depends
    on which field a chunk came from and that the features did not supply.

    This is the workload-independent version of the component finding: it needs
    no vector fields and no naming convention, only two fields at one timestep.
    """
    feats = [f for f in features if f in cw.columns]
    keys = _group_keys(cw)
    if not feats or "field" not in cw.columns or not keys:
        return pd.DataFrame()
    oof = oof_predictions(cw, feats, target, "forest", True, seed=seed)
    if oof.empty:
        return pd.DataFrame()
    meta = cw[[c for c in dict.fromkeys(["chunk_uid", "field"] + keys)
               if c in cw.columns]]
    d = oof.merge(meta, on="chunk_uid", how="left").dropna(subset=["field"])
    if d.empty:
        return pd.DataFrame()
    d["_gm"] = d.groupby(keys, dropna=False)["residual"].transform("mean")
    d["_n_in_moment"] = d.groupby(keys, dropna=False)["residual"].transform(
        "size")
    d = d[d["_n_in_moment"] >= 2]
    if d.empty:
        return pd.DataFrame()
    d["residual_vs_moment"] = d["residual"] - d["_gm"]

    recs: List[dict] = []
    for fld, g in d.groupby("field", sort=True):
        v = g["residual_vs_moment"].to_numpy(dtype=float)
        moved = v[v != 0]
        n, k = int(moved.size), int((moved > 0).sum())
        recs.append({
            "field": fld, "n_chunks_in_shared_moments": int(len(g)),
            "median_residual_log10": float(np.median(v)),
            # 10**residual: >1 means the features UNDER-rate this field, i.e.
            # it compresses better than they say.
            "median_fold_vs_moment": float(10.0 ** np.median(v)),
            "n_sign_test": n, "k_sign_test": k,
            "sign_test_p": float(sps.binomtest(k, n, 0.5).pvalue)
            if n >= 2 else np.nan,
            "direction": ("compresses BETTER than the features predict"
                          if np.median(v) > 0 else
                          "compresses WORSE than the features predict"),
        })
    out = pd.DataFrame(recs)
    if len(out) >= 2:
        groups = [g["residual_vs_moment"].to_numpy(dtype=float)
                  for _, g in d.groupby("field", sort=True) if len(g) >= 2]
        if len(groups) >= 2:
            try:
                h = sps.kruskal(*groups)
                out.attrs["kruskal_h"] = float(h.statistic)
                out.attrs["kruskal_p"] = float(h.pvalue)
            except ValueError:
                pass          # all residuals identical: no structure to find
    return out.sort_values("median_residual_log10",
                           ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- verdict
def summarize(pairs: pd.DataFrame, spread: pd.DataFrame,
              ceiling: pd.DataFrame, ordering: pd.DataFrame,
              resid: pd.DataFrame) -> Dict[str, object]:
    """One dict the report and the answers both read, with the verdict."""
    out: Dict[str, object] = {
        "available": not pairs.empty,
        "n_matched_pairs": int(len(pairs)),
        "n_moments": int(pairs["group"].nunique()) if not pairs.empty else 0,
        "n_cross_field_pairs":
            int((~pairs["same_field"]).sum()) if not pairs.empty else 0,
    }
    if pairs.empty:
        out["reason"] = ("no two chunks in this log share a workload, "
                         "timestep and size, so there is no comparison in "
                         "which the simulation clock is held constant")
        return out

    reasons: List[str] = []
    tight = spread[spread["max_rel_feature_diff_at_most"] <= 1e-3] \
        if not spread.empty else pd.DataFrame()
    if not tight.empty:
        r = tight.iloc[0]
        out.update({
            "tightest_tolerance": float(r["max_rel_feature_diff_at_most"]),
            "tightest_n_pairs": int(r["n_pairs"]),
            "tightest_n_moments": int(r["n_moments"]),
            "tightest_median_fold": float(r["median_ratio_fold"]),
            "tightest_max_fold": float(r["max_ratio_fold"]),
        })
        if float(r["max_ratio_fold"]) >= INTERESTING_FOLD:
            reasons.append(
                f"{int(r['n_pairs'])} pairs of chunks from the same moment "
                f"agree in every feature to "
                f"{float(r['max_rel_feature_diff_at_most']):g} relative, yet "
                f"their ratios differ by up to "
                f"{float(r['max_ratio_fold']):.2f}x "
                f"(median {float(r['median_ratio_fold']):.2f}x)")

    if not ceiling.empty:
        ex = ceiling[ceiling["near_exact"]]
        # The tightest tolerance is the most defensible bound but often merges
        # one lucky pair, which makes it near 1.0 and says nothing. Among the
        # tolerances tight enough to still mean "identical", take the one
        # resting on the most classes.
        support = ex[ex["n_classes_with_more_than_one_chunk"] >= 3] \
            if not ex.empty else ex
        row = (support.iloc[-1] if not support.empty
               else (ex.iloc[0] if not ex.empty else ceiling.iloc[0]))
        out["max_achievable_oof_r2"] = float(row["max_achievable_oof_r2"])
        out["ceiling_tolerance"] = float(row["max_rel_feature_diff_at_most"])
        out["ceiling_n_classes"] = int(
            row["n_classes_with_more_than_one_chunk"])
        out["ceiling_n_chunks_with_a_twin"] = int(
            row["n_chunks_in_those_classes"])
        out["ceiling_n_chunks"] = int(row["n_chunks"])
        # The bound only sees chunks that HAVE a numerically identical twin.
        # Most do not, and their irreducible error is invisible here -- so the
        # bound is loose, and loose in the direction of optimism.
        out["ceiling_is_weak"] = bool(
            row["n_chunks_in_those_classes"] < 0.5 * row["n_chunks"])
        if float(row["max_achievable_oof_r2"]) < 0.98:
            reasons.append(
                f"no model built on these features can exceed out-of-fold "
                f"R^2 = {float(row['max_achievable_oof_r2']):.3f} on this "
                f"log, because "
                f"{int(row['n_chunks_in_those_classes'])} chunks fall into "
                f"{int(row['n_classes_with_more_than_one_chunk'])} classes "
                f"that are numerically identical in every feature and do not "
                f"share a ratio")

    stable = ordering[(ordering["sign_test_p"] < 0.05)] \
        if not ordering.empty else pd.DataFrame()
    out["n_stable_component_orderings"] = int(len(stable))
    if not stable.empty:
        r = stable.iloc[0]
        win = r["more_compressible_axis"]
        lost = r["axis_b"] if win == r["axis_a"] else r["axis_a"]
        n_tot = int(r["n_moments"])
        k_a = int(r["n_moments_axis_a_better"])
        n_win = k_a if win == r["axis_a"] else n_tot - k_a
        out["headline_ordering"] = (
            f"{r['family']}: the {win} component compresses "
            f"{float(r['median_ratio_fold']):.2f}x better than {lost} at "
            f"{n_win} of {n_tot} moments "
            f"(p = {float(r['sign_test_p']):.3g})")
        agree = stable[stable["features_agree"]]
        if not agree.empty:
            tol = float(agree["median_max_rel_feature_diff"].max())
            reasons.append(
                f"{len(agree)} pair(s) of vector components whose features "
                f"agree to {tol:.1g} relative are nevertheless ordered the "
                f"same way by compressibility at every moment")

    if not resid.empty and "kruskal_p" in resid.attrs:
        out["field_residual_kruskal_p"] = float(resid.attrs["kruskal_p"])
        sig = resid[resid["sign_test_p"] < 0.05] if "sign_test_p" in resid \
            else pd.DataFrame()
        out["n_fields_with_systematic_residual"] = int(len(sig))
        if float(resid.attrs["kruskal_p"]) < 0.05:
            reasons.append(
                f"with the timestep pinned, the out-of-fold residuals are "
                f"still ordered by field (Kruskal-Wallis p = "
                f"{float(resid.attrs['kruskal_p']):.3g}), so the features are "
                f"missing something that varies between fields at one moment")

    out["anisotropic"] = bool(reasons)
    out["reasons"] = reasons
    out["hypothesis"] = (
        "HYPOTHESIS, not a measurement: the missing quantity is orientation. "
        "Entropy is a byte histogram and MAD is a mean, so both are invariant "
        "to any permutation of the buffer; second_deriv is a lag-1 stencil on "
        "the FLATTENED buffer, so it sees only the fastest-varying axis. A "
        "field whose structure runs along a different axis therefore has the "
        "same three numbers and different match lengths for the codec. This "
        "module cannot test that -- it never sees the buffer. The cheap test "
        "is to add second-derivative stencils at lag nx and lag nx*ny and "
        "check whether they separate the components that these three do not.")
    return out

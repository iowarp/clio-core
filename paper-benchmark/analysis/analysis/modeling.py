#!/usr/bin/env python3
"""Interpretable models, the feature ablation, and threshold discovery.

This is not an accuracy competition. Every model here exists to answer one of
three questions -- how much of the outcome the three features explain, which
of them carries the information, and where in feature space the behaviour
changes -- and each is reported with the evidence that it generalises rather
than only the number it scored.

LEAKAGE, which is the whole reason this module is careful. A chunk contributes
32 rows. A random row split therefore puts the SAME chunk on both sides of the
train/test boundary, with the same entropy, the same MAD, the same second
derivative and a highly correlated ratio. A model can then reach an R^2 near 1
by memorising chunks, and nothing in the score reveals it. Every split below is
GROUP-AWARE on `chunk_uid`, and three progressively harder generalisation
tests are run on top of it:

    chunk holdout   -- unseen chunks of the same fields and timesteps
    time holdout    -- train on early timesteps, test on later ones
    field holdout   -- leave one physical field out entirely
    workload holdout-- (cross-workload only) leave one workload out

They fail in that order on real data, and the gap between them is the finding:
a model that survives chunk holdout and collapses on field holdout has learned
the fields, not the physics.

TARGET IN LOG SPACE. Ratio spans 1x to 1400x in a single sweep. Fitting it in
native units makes the loss almost entirely about the handful of chunks above
100x, and the R^2 then reports how well those few were fitted. log10(ratio) is
fitted instead, and the report says so wherever an R^2 appears.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

MIN_TRAIN = 12
MIN_TEST = 4


# --------------------------------------------------------------------------
# Group-aware evaluation
# --------------------------------------------------------------------------

def _prep(df: pd.DataFrame, features: Sequence[str], target: str,
          log_target: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                     pd.DataFrame]:
    cols = list(features) + [target, "chunk_uid"]
    d = df[[c for c in cols if c in df.columns]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if log_target:
        d = d[d[target] > 0]
    X = d[list(features)].to_numpy(dtype=float)
    y = d[target].to_numpy(dtype=float)
    if log_target:
        y = np.log10(y)
    g = d["chunk_uid"].to_numpy()
    return X, y, g, d


def grouped_cv_r2(df: pd.DataFrame, features: Sequence[str], target: str,
                  model_name: str = "tree", log_target: bool = True,
                  n_splits: int = 5, seed: int = 0) -> Dict[str, object]:
    """Group-k-fold R^2 with `chunk_uid` as the group.

    Reports the out-of-fold R^2, the in-fold R^2 beside it (the gap is the
    memorisation the grouping prevented from being scored), and the R^2 of a
    mean-predicting baseline so a small positive number is readable as such.
    """
    feats = [f for f in features if f in df.columns]
    if not feats:
        return {"available": False, "reason": "no usable features"}
    X, y, g, d = _prep(df, feats, target, log_target)
    n_groups = int(pd.unique(g).size)
    if len(y) < MIN_TRAIN or n_groups < 4:
        return {"available": False, "n": int(len(y)), "n_groups": n_groups,
                "reason": "too few chunks to cross-validate"}
    k = int(min(n_splits, n_groups))
    cv = GroupKFold(n_splits=k)
    oof = np.full(len(y), np.nan)
    infold = []
    for tr, te in cv.split(X, y, groups=g):
        m = _make_regressor(model_name, seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        infold.append(r2_score(y[tr], m.predict(X[tr])))
    ok = np.isfinite(oof)
    base = DummyRegressor(strategy="mean").fit(X, y)
    return {
        "available": True, "model": model_name, "target": target,
        "log_target": log_target, "features": list(feats),
        "n": int(len(y)), "n_chunks": n_groups, "n_splits": k,
        "oof_r2": float(r2_score(y[ok], oof[ok])),
        "in_fold_r2_mean": float(np.mean(infold)),
        "baseline_r2": float(r2_score(y, base.predict(X))),
        "oof_rmse": float(np.sqrt(np.mean((y[ok] - oof[ok]) ** 2))),
        "memorisation_gap": float(np.mean(infold) - r2_score(y[ok], oof[ok])),
    }


def oof_predictions(df: pd.DataFrame, features: Sequence[str], target: str,
                    model_name: str = "forest", log_target: bool = True,
                    n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Per-chunk out-of-fold prediction, under the same group discipline as
    `grouped_cv_r2`.

    Exists so a caller can look at the RESIDUALS rather than only the score.
    What a model gets systematically wrong -- and whether those errors line up
    with something the features do not carry -- is a different question from
    how well it scores, and answering it needs the predictions themselves.
    """
    feats = [f for f in features if f in df.columns]
    if not feats or target not in df.columns:
        return pd.DataFrame()
    X, y, g, d = _prep(df, feats, target, log_target)
    n_groups = int(pd.unique(g).size)
    if len(y) < MIN_TRAIN or n_groups < 4:
        return pd.DataFrame()
    cv = GroupKFold(n_splits=int(min(n_splits, n_groups)))
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups=g):
        m = _make_regressor(model_name, seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    out = d.copy()
    out["y"] = y
    out["oof_pred"] = oof
    out["residual"] = y - oof
    return out.dropna(subset=["oof_pred"]).reset_index(drop=True)


def _make_regressor(name: str, seed: int, n_estimators: int = 300):
    if name == "linear":
        return LinearRegression()
    if name == "forest_small":
        # Only for the stability sweep, which fits ~150 forests: the question
        # there is whether a DELTA changes sign across partitions, and that is
        # answered at a fraction of the trees the reported scores use.
        return RandomForestRegressor(n_estimators=80, min_samples_leaf=2,
                                     random_state=seed, n_jobs=1)
    if name == "tree":
        # Depth 3: the point is a rule a person can read, and a deeper tree
        # stops being one long before it stops improving.
        return DecisionTreeRegressor(max_depth=3, min_samples_leaf=5,
                                     random_state=seed)
    if name == "forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, n_jobs=1)
    raise ValueError(name)


def _make_classifier(name: str, seed: int):
    if name == "tree":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                      random_state=seed)
    if name == "forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      random_state=seed, n_jobs=1)
    raise ValueError(name)


def holdout_by(df: pd.DataFrame, features: Sequence[str], target: str,
               by: str, model_name: str = "forest", log_target: bool = True,
               seed: int = 0) -> pd.DataFrame:
    """Leave-one-`by`-out generalisation (by = 'field', 'workload', ...).

    For `by == 'timestep'` this becomes a forward split instead: train on the
    early half of the run and test on the later half, which is the question
    that actually gets asked of a temporal model.
    """
    feats = [f for f in features if f in df.columns]
    if by not in df.columns or not feats:
        return pd.DataFrame()
    X, y, g, d = _prep(df.assign(**{by: df[by]}), feats, target, log_target)
    d = d.join(df[[by]], how="left") if by not in d.columns else d
    if by not in d.columns:
        return pd.DataFrame()
    recs: List[dict] = []
    if by == "timestep":
        ts = pd.to_numeric(d[by], errors="coerce")
        if ts.notna().sum() < MIN_TRAIN or ts.nunique() < 4:
            return pd.DataFrame()
        cut = float(ts.quantile(0.6))
        tr, te = (ts <= cut).to_numpy(), (ts > cut).to_numpy()
        if tr.sum() < MIN_TRAIN or te.sum() < MIN_TEST:
            return pd.DataFrame()
        m = _make_regressor(model_name, seed).fit(X[tr], y[tr])
        recs.append({"holdout": "later timesteps", "by": by,
                     "held": f"timestep > {cut:g}",
                     "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "r2": float(r2_score(y[te], m.predict(X[te]))),
                     "train_r2": float(r2_score(y[tr], m.predict(X[tr])))})
        return pd.DataFrame(recs)

    for held, sub in d.groupby(by, dropna=False):
        te = (d[by] == held).to_numpy()
        tr = ~te
        if tr.sum() < MIN_TRAIN or te.sum() < MIN_TEST:
            continue
        m = _make_regressor(model_name, seed).fit(X[tr], y[tr])
        recs.append({"holdout": f"leave-one-{by}-out", "by": by,
                     "held": str(held),
                     "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "n_test_chunks": int(pd.unique(g[te]).size),
                     "r2": float(r2_score(y[te], m.predict(X[te]))),
                     "train_r2": float(r2_score(y[tr], m.predict(X[tr])))})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------

def _random_group_folds(groups: np.ndarray, n_splits: int,
                        rng: np.random.Generator) -> List[np.ndarray]:
    """Assign whole GROUPS to folds at random.

    `GroupKFold` is deterministic: it sorts groups by size and deals them out,
    so with one row per group it returns the SAME partition however the rows
    are ordered. A "stability sweep" built on permuting rows therefore
    re-measures one partition and reports a spread of zero -- robustness that
    was never tested. This deals the groups out randomly instead, which is
    what makes the repeats independent draws.
    """
    uniq = pd.unique(groups)
    order = rng.permutation(len(uniq))
    assign = {u: int(order[i] % n_splits) for i, u in enumerate(uniq)}
    fold_of = np.array([assign[g] for g in groups])
    return [np.flatnonzero(fold_of == k) for k in range(n_splits)]


def _oof_r2_random_folds(df: pd.DataFrame, features: Sequence[str],
                         target: str, model_name: str, log_target: bool,
                         rng: np.random.Generator, n_splits: int,
                         seed: int) -> Optional[float]:
    X, y, g, _ = _prep(df, features, target, log_target)
    if len(y) < MIN_TRAIN or pd.unique(g).size < n_splits:
        return None
    oof = np.full(len(y), np.nan)
    for te in _random_group_folds(g, n_splits, rng):
        if te.size == 0 or te.size >= len(y) - 2:
            continue
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = _make_regressor(model_name, seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    ok = np.isfinite(oof)
    if ok.sum() < 5:
        return None
    return float(r2_score(y[ok], oof[ok]))


def ablation_stability(df: pd.DataFrame, features: Sequence[str],
                       target: str, base: Sequence[str], extra: str,
                       model_name: str = "forest_small",
                       log_target: bool = True,
                       n_repeats: int = 12,
                       n_splits: int = 5) -> Dict[str, object]:
    """How stable is "does `extra` help on top of `base`" across re-folding?

    A single out-of-fold delta is one draw from one fold partition, and at a
    few dozen chunks the draw-to-draw spread can exceed the delta itself -- so
    "no, it adds noise" can become "yes, it helps" on a different partition of
    the same data with nothing else changed. Each repeat deals the CHUNKS out
    to folds afresh (see `_random_group_folds`; sklearn's GroupKFold cannot do
    this, and using it here would silently measure one partition many times).

    Reports the spread and, decisively, how often the sign flips.
    """
    base = [b for b in base if b in df.columns]
    if extra not in df.columns or not base:
        return {"available": False}
    deltas = []
    for i in range(n_repeats):
        r0 = _oof_r2_random_folds(df, base, target, model_name, log_target,
                                  np.random.default_rng(1000 + i), n_splits,
                                  seed=i)
        # The SAME partition for both arms, so the delta isolates the feature
        # rather than the re-fold.
        r1 = _oof_r2_random_folds(df, list(base) + [extra], target,
                                  model_name, log_target,
                                  np.random.default_rng(1000 + i), n_splits,
                                  seed=i)
        if r0 is not None and r1 is not None:
            deltas.append(r1 - r0)
    if len(deltas) < 3:
        return {"available": False}
    a = np.asarray(deltas, dtype=float)
    med = float(np.median(a))
    n_flip = int(np.sum(np.sign(a) != np.sign(med)))
    return {
        "available": True, "base": base, "extra": extra, "target": target,
        "model": model_name, "n_repeats": int(a.size),
        "median_delta_r2": med,
        "mean_delta_r2": float(a.mean()), "sd_delta_r2": float(a.std()),
        "min_delta_r2": float(a.min()), "max_delta_r2": float(a.max()),
        "n_sign_flips": n_flip,
        "stable": bool(n_flip == 0 and abs(med) > a.std()),
    }


def feature_ablation(df: pd.DataFrame, features: Sequence[str], target: str,
                     model_name: str = "forest", log_target: bool = True,
                     seed: int = 0) -> pd.DataFrame:
    """Every non-empty subset of the features, scored out-of-fold on chunks.

    This is the direct answer to "does MAD add anything beyond entropy" and
    "does second_deriv add anything beyond entropy + MAD": read the delta
    between the corresponding rows. Reported for both a linear model and a
    forest, because a feature can add nothing linearly and a great deal in a
    tree -- and on this data some of them do.
    """
    feats = [f for f in features if f in df.columns]
    recs: List[dict] = []
    for k in range(1, len(feats) + 1):
        for combo in itertools.combinations(feats, k):
            for mn in (model_name, "linear"):
                r = grouped_cv_r2(df, combo, target, mn, log_target, seed=seed)
                if not r.get("available"):
                    continue
                recs.append({
                    "features": "+".join(combo), "n_features": k,
                    "model": mn, "target": target, "log_target": log_target,
                    "oof_r2": r["oof_r2"], "baseline_r2": r["baseline_r2"],
                    "in_fold_r2": r["in_fold_r2_mean"],
                    "memorisation_gap": r["memorisation_gap"],
                    "n": r["n"], "n_chunks": r["n_chunks"]})
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    # The marginal contribution of each feature: the mean improvement it makes
    # when added to every subset that does not already contain it. This is the
    # Shapley-style reading of the table and is what the report quotes.
    marg: List[dict] = []
    for mn, gm in out.groupby("model"):
        idx = {frozenset(r["features"].split("+")): r["oof_r2"]
               for _, r in gm.iterrows()}
        base_r2 = float(gm["baseline_r2"].iloc[0])
        for f in feats:
            gains = []
            for s, v in idx.items():
                if f in s:
                    prev = idx.get(s - {f}, base_r2) if len(s) > 1 else base_r2
                    gains.append(v - prev)
            if gains:
                marg.append({"model": mn, "feature": f,
                             "mean_marginal_oof_r2_gain": float(np.mean(gains)),
                             "max_marginal_gain": float(np.max(gains)),
                             "min_marginal_gain": float(np.min(gains)),
                             "n_subsets": len(gains), "target": target})
    out.attrs["marginal"] = pd.DataFrame(marg)
    return out


def importances(df: pd.DataFrame, features: Sequence[str], target: str,
                log_target: bool = True, seed: int = 0) -> pd.DataFrame:
    """Impurity and permutation importance from a group-held-out forest.

    Permutation importance is computed on HELD-OUT chunks, not on the training
    set: measured in-sample it rewards a feature for the memorisation the
    grouping was introduced to prevent.
    """
    feats = [f for f in features if f in df.columns]
    if not feats:
        return pd.DataFrame()
    X, y, g, _ = _prep(df, feats, target, log_target)
    n_groups = int(pd.unique(g).size)
    if len(y) < MIN_TRAIN or n_groups < 4:
        return pd.DataFrame()
    cv = GroupKFold(n_splits=int(min(5, n_groups)))
    tr, te = next(iter(cv.split(X, y, groups=g)))
    m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                              random_state=seed, n_jobs=1).fit(X[tr], y[tr])
    perm = permutation_importance(m, X[te], y[te], n_repeats=20,
                                  random_state=seed, n_jobs=1)
    return pd.DataFrame({
        "feature": feats, "target": target,
        "impurity_importance": m.feature_importances_,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
        "n_train": int(tr.size), "n_test": int(te.size),
        "test_r2": float(r2_score(y[te], m.predict(X[te]))),
    })


# --------------------------------------------------------------------------
# Codec classification
# --------------------------------------------------------------------------

def codec_classifier(chunks: pd.DataFrame, features: Sequence[str],
                     label_col: str, seed: int = 0,
                     min_per_class: int = 5) -> Dict[str, object]:
    """Can the three features predict which codec wins?

    The majority-class baseline is reported first and deliberately: on a
    single workload one codec often wins nearly every chunk, and an accuracy
    of 0.95 against a 0.95 baseline is not a finding. `degenerate` marks that
    case explicitly so the report cannot claim a prediction that is really a
    constant.
    """
    feats = [f for f in features if f in chunks.columns]
    if label_col not in chunks.columns or not feats:
        return {"available": False, "reason": "missing columns"}
    d = chunks[feats + [label_col, "chunk_uid"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    counts = d[label_col].value_counts()
    major = float(counts.iloc[0] / counts.sum()) if len(counts) else 1.0
    keep = counts[counts >= min_per_class].index
    d = d[d[label_col].isin(keep)]
    if d[label_col].nunique() < 2 or len(d) < MIN_TRAIN:
        return {"available": False, "degenerate": True,
                "n": int(len(d)), "n_classes": int(counts.size),
                "majority_class": str(counts.index[0]) if len(counts) else None,
                "majority_share": major,
                "reason": ("one codec wins essentially every chunk in this "
                           "log, so there is nothing to classify. This is a "
                           "property of the workload, not of the features.")}
    X = d[feats].to_numpy(dtype=float)
    y = d[label_col].to_numpy()
    g = d["chunk_uid"].to_numpy()
    n_groups = int(pd.unique(g).size)
    cv = GroupKFold(n_splits=int(min(5, n_groups)))
    oof = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups=g):
        if pd.unique(y[tr]).size < 2:
            oof[te] = pd.Series(y[tr]).mode().iloc[0]
            continue
        m = _make_classifier("forest", seed).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    tree = _make_classifier("tree", seed).fit(X, y)
    return {
        "available": True, "n": int(len(y)),
        "n_classes": int(pd.unique(y).size),
        "classes": sorted(str(c) for c in pd.unique(y)),
        "oof_accuracy": float(accuracy_score(y, oof)),
        "majority_baseline_accuracy": float(
            accuracy_score(y, dummy.predict(X))),
        "lift_over_baseline": float(
            accuracy_score(y, oof) - accuracy_score(y, dummy.predict(X))),
        "degenerate": bool(major >= 0.9),
        "majority_share": major,
        "tree_rules": export_text(tree, feature_names=list(feats)),
    }


# --------------------------------------------------------------------------
# Threshold discovery
# --------------------------------------------------------------------------

def discover_thresholds(chunks: pd.DataFrame, features: Sequence[str],
                        label_col: str, min_support: int = 20,
                        min_purity: float = 0.65,
                        seed: int = 0) -> List[dict]:
    """Shallow-tree rules, kept only when they have support AND beat the base
    rate.

    A leaf is emitted only when it covers at least `min_support` chunks, is at
    least `min_purity` pure, AND its purity exceeds the class's overall
    prevalence by 10 points -- without that last test a rule that "predicts"
    the codec winning 95% of chunks would be emitted for every leaf, which is
    exactly the "do not invent thresholds if sample support is weak"
    requirement.
    """
    feats = [f for f in features if f in chunks.columns]
    if label_col not in chunks.columns or not feats:
        return []
    d = chunks[feats + [label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < min_support * 2 or d[label_col].nunique() < 2:
        return []
    X = d[feats].to_numpy(dtype=float)
    y = d[label_col].astype(str).to_numpy()
    prevalence = pd.Series(y).value_counts(normalize=True).to_dict()
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_support,
                                  random_state=seed).fit(X, y)
    t = tree.tree_
    rules: List[dict] = []

    def walk(node: int, conds: List[str]) -> None:
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            total = float(counts.sum())
            i = int(np.argmax(counts))
            cls = str(tree.classes_[i])
            purity = float(counts[i] / total) if total else 0.0
            if total >= min_support and purity >= min_purity and \
                    purity - prevalence.get(cls, 0.0) >= 0.10:
                rules.append({
                    "label_column": label_col, "predicted": cls,
                    "rule": " and ".join(conds) if conds else "(always)",
                    "win_probability": round(purity, 4),
                    "sample_count": int(total),
                    "base_rate": round(prevalence.get(cls, 0.0), 4),
                    "lift": round(purity - prevalence.get(cls, 0.0), 4),
                })
            return
        f = feats[t.feature[node]]
        thr = float(t.threshold[node])
        walk(t.children_left[node], conds + [f"{f} <= {thr:.6g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.6g}"])

    walk(0, [])
    return sorted(rules, key=lambda r: (-r["lift"], -r["sample_count"]))


def discover_regression_thresholds(df: pd.DataFrame, features: Sequence[str],
                                   target: str, log_target: bool = True,
                                   min_support: int = 20,
                                   seed: int = 0) -> List[dict]:
    """The same, for a continuous target: shallow-tree leaves as regimes.

    Emitted only when the leaf's median differs from the global median by more
    than half the interquartile range -- otherwise a leaf is a partition of
    the data, not a regime in it.
    """
    feats = [f for f in features if f in df.columns]
    if target not in df.columns or not feats:
        return []
    X, y, g, d = _prep(df, feats, target, log_target)
    if len(y) < min_support * 2:
        return []
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=min_support,
                                 random_state=seed).fit(X, y)
    t = tree.tree_
    gmed, iqr = float(np.median(y)), float(np.subtract(*np.percentile(y, [75, 25])))
    rules: List[dict] = []

    def walk(node: int, conds: List[str], mask: np.ndarray) -> None:
        if t.children_left[node] == -1:
            n = int(mask.sum())
            if n < min_support:
                return
            med = float(np.median(y[mask]))
            if iqr > 0 and abs(med - gmed) < 0.5 * iqr:
                return
            rules.append({
                "target": target, "log_target": log_target,
                "rule": " and ".join(conds) if conds else "(always)",
                "median": med, "global_median": gmed,
                "median_in_native_units": float(10 ** med) if log_target else med,
                "n_chunks": int(pd.unique(g[mask]).size),
                "sample_count": n,
                "delta_vs_global_median": med - gmed,
            })
            return
        f = feats[t.feature[node]]
        thr = float(t.threshold[node])
        col = X[:, t.feature[node]]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.6g}"],
             mask & (col <= thr))
        walk(t.children_right[node], conds + [f"{f} > {thr:.6g}"],
             mask & (col > thr))

    walk(0, [], np.ones(len(y), dtype=bool))
    return sorted(rules, key=lambda r: -abs(r["delta_vs_global_median"]))

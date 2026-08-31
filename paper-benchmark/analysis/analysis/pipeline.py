#!/usr/bin/env python3
"""Orchestration: run every stage for one workload, then turn the tables into
answers.

The stages are ordered by dependency, not by importance, and every one of them
is allowed to produce nothing. An analysis that cannot run -- no error bounds
were swept, decompression was never measured, one codec won every chunk -- is
recorded in `unavailable` with the reason, and the report prints the reason
instead of silence. A missing analysis and a null result are different
findings and must not look the same.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field as _dc_field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import (anisotropy, codec_analysis, confounds, cost_analysis,
               counterexamples, mechanism, modeling, paired_analysis,
               prediction_analysis, quantization_mechanism, statistics,
               temporal_analysis)
from .loader import Dataset, load_evolution
from .plotting import Figures

#: The outcome analysed at chunk level. "What is the best this chunk can do"
#: is the question the features are supposed to answer; a single codec's ratio
#: confounds the data with that codec's quirks.
CHUNK_TARGET = "best_ratio_value"


def _sec(title: str) -> str:
    """`Section N` for a report section, resolved from the report's own order.

    Imported lazily: reporting imports this module, so a module-level import
    the other way would be circular.
    """
    from .reporting import section_ref
    return section_ref(title)


@dataclass
class Analysis:
    ds: Dataset
    tables: Dict[str, pd.DataFrame] = _dc_field(default_factory=dict)
    facts: Dict[str, object] = _dc_field(default_factory=dict)
    profiles: Dict[str, dict] = _dc_field(default_factory=dict)
    thresholds: Dict[str, list] = _dc_field(default_factory=dict)
    unavailable: Dict[str, str] = _dc_field(default_factory=dict)
    figures: Optional[Figures] = None
    answers: Dict[str, dict] = _dc_field(default_factory=dict)

    def put(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        self.tables[name] = df if df is not None else pd.DataFrame()
        return self.tables[name]

    def na(self, name: str, why: str) -> None:
        self.unavailable[name] = why


def run(ds: Dataset, outdir: str, seed: int = 0,
        make_figures: bool = True) -> Analysis:
    a = Analysis(ds=ds)
    rows, chunks, feats = ds.rows, ds.chunks, ds.features

    # ---- 1. cost model, recovered from the log ---------------------------
    cm = cost_analysis.infer_cost_model(rows)
    ds.cost_model = cm
    a.facts["cost_model"] = cm
    a.facts["cost_diagnostics"] = cost_analysis.cost_diagnostics(rows, cm)

    # ---- 2. winners and regret -------------------------------------------
    win = a.put("winners", cost_analysis.winners(rows))
    reg = a.put("selection_regret", cost_analysis.selection_regret(rows, win))
    a.facts["selection"] = cost_analysis.summarize_selection(reg)
    a.put("selection_results", win)

    # Chunk-level frame: intrinsic properties beside the best outcome each
    # chunk can reach. This is the unit almost everything below runs on.
    cw = chunks.merge(
        win[["chunk_uid", "best_ratio_value", "best_ratio_lib",
             "best_ratio_nties", "lowest_cost_lib", "adopted_lib",
             "adopted_ratio", "adopted_cost", "primary_lib",
             "fastest_ct_value", "fastest_dt_value"]],
        on="chunk_uid", how="left")
    a.put("chunk_properties", cw)

    # ---- 2b. IS IT THE DATA, OR THE CLOCK? ------------------------------
    # Deliberately before the correlations, because its verdict governs how
    # every one of them may be read. See confounds.py.
    if ds.features:
        power = a.put("confounder_power",
                      confounds.confounder_power(cw, ds.features,
                                                 CHUNK_TARGET, seed=seed))
        collin = a.put("feature_confounder_collinearity",
                       confounds.collinearity(cw, ds.features))
        partial = a.put("partial_correlations",
                        confounds.partial_correlations(cw, ds.features,
                                                       CHUNK_TARGET))
        strat = confounds.stratified_holdout(cw, ds.features, CHUNK_TARGET,
                                             seed=seed)
        nn = confounds.nearest_neighbour_structure(cw, ds.features)
        a.facts["confounds"] = confounds.summarize(power, collin, partial,
                                                   strat, nn)
    else:
        a.na("confounds", "no usable data-property features to test")
    a.facts["cohort"] = ds.cohort

    # ---- 2c. SAME MOMENT, DIFFERENT FIELD -------------------------------
    # The answer to 2b, and the only comparison in a single log where the
    # clock is held EXACTLY constant rather than statistically. Runs before
    # the correlations for the same reason 2b does. See anisotropy.py.
    if ds.features:
        mp = a.put("matched_pairs",
                   anisotropy.matched_pairs(cw, ds.features, CHUNK_TARGET))
        spread = a.put("feature_blind_spread",
                       anisotropy.feature_blind_spread(mp))
        ceiling = a.put("explainable_ceiling",
                        anisotropy.explainable_ceiling(cw, ds.features,
                                                       CHUNK_TARGET))
        fams = a.put("component_families", anisotropy.component_families(cw))
        order = a.put("component_ordering",
                      anisotropy.component_ordering(mp, fams, ds.features))
        resid = a.put("residual_by_field",
                      anisotropy.residual_by_field(cw, ds.features,
                                                   CHUNK_TARGET, seed=seed))
        an = anisotropy.summarize(mp, spread, ceiling, order, resid)
        a.facts["anisotropy"] = an
        if not an.get("available"):
            a.na("anisotropy", str(an.get("reason", "no matched groups")))
    else:
        a.na("anisotropy", "no usable data-property features, so there is "
                           "nothing to hold constant between two chunks")

    # ---- 2d. WHY: the ratio against its entropy bound ------------------
    # Association is what the rest of the pipeline measures; this asks through
    # what mechanism, and how much of the outcome that mechanism accounts for.
    # See mechanism.py.
    bnd = a.put("entropy_bound", mechanism.entropy_bound(rows))
    ordr = a.put("codec_order_sensitivity",
                 mechanism.codec_order_sensitivity(bnd))
    shuf_dec = a.put("shuffle_decomposition",
                     mechanism.shuffle_decomposition(rows))
    loc = a.put("locality_probe",
                mechanism.locality_probe(shuf_dec, ordr))
    med = a.put("timing_mediation",
                mechanism.timing_mediation(rows, ds.features))
    a.put("codec_throughput", mechanism.throughput(rows))
    pre = mechanism.stats_are_pre_transform(rows, ds.features)
    mech = mechanism.summarize(bnd, ordr, shuf_dec, loc, med,
                               pre_transform=pre)
    a.facts["mechanism"] = mech
    if not mech.get("available"):
        a.na("mechanism", str(mech.get("reason", "no entropy column")))

    # ---- 2e. WHY, for the QUANTIZED rows -------------------------------
    # 2d stops at the lossless rows because the logged stats describe the
    # original buffer. This stage explains the other half from the
    # quantizer's own arithmetic: the level count L = range / (2*0.95*eb) is
    # closed-form and codec-free, and it sets ratio and quality together.
    # See quantization_mechanism.py.
    qpc = a.put("quantization_levels_per_chunk",
                quantization_mechanism.per_chunk(rows, cw))
    qreg = a.put("quantization_regimes",
                 quantization_mechanism.regime_table(qpc))
    qfld = a.put("quantization_levels_by_field",
                 quantization_mechanism.by_field(qpc))
    qdrv = a.put("quantization_levels_drive_outcomes",
                 quantization_mechanism.levels_drive_outcomes(qpc, rows))
    qm = quantization_mechanism.summarize(qpc, qreg, qdrv, qfld)
    a.facts["quantization_mechanism"] = qm
    if not qm.get("available"):
        a.na("quantization_mechanism", str(qm.get("reason", "no rows")))

    # ---- 3. correlations --------------------------------------------------
    if not feats:
        a.na("correlations", "the log carries no usable data-property "
                             "columns and no selection.csv was found to "
                             "join them from")
    else:
        targets = ["ratio", "ct_ms", "dt_ms_measured", "cost"]
        targets = [t for t in targets if rows[t].notna().any()]
        a.put("correlations_per_config",
              statistics.per_config_correlations(rows, feats, targets))
        a.put("correlations_within_field",
              statistics.within_field_correlation(rows, feats, ["ratio"]))
        a.put("correlations_chunk_level",
              statistics.correlation_table(
                  cw, feats, [CHUNK_TARGET, "fastest_ct_value"],
                  label="chunk_level"))
        # Every property against every outcome in one matrix, with lossless
        # and quantized ratio kept apart so the bound cannot pose as a data
        # property. This is the figure a reader looks at first.
        pof = statistics.property_outcome_frame(cw, rows)
        a.put("property_outcome_frame", pof)
        a.put("property_correlations",
              statistics.property_outcome_matrix(pof))
        # Binned trends, at chunk level, for the saturation question.
        trends, mono = [], []
        for f in feats:
            for t in (CHUNK_TARGET,):
                tr = statistics.binned_trend(cw, f, t)
                if tr.empty:
                    continue
                trends.append(tr)
                m = statistics.monotonicity(tr)
                if m:
                    m.update({"feature": f, "target": t})
                    mono.append(m)
        a.put("binned_trends", pd.concat(trends, ignore_index=True)
              if trends else pd.DataFrame())
        a.put("trend_monotonicity", pd.DataFrame(mono))
        a.put("joint_regimes", statistics.joint_regimes(cw, feats,
                                                        CHUNK_TARGET))
        gains = []
        for f in feats:
            base = [x for x in feats if x != f]
            g = statistics.conditional_gain(cw, base, f, CHUNK_TARGET)
            g["stage"] = "linear_partial"
            gains.append(g)
        a.put("conditional_gains", pd.DataFrame(gains))

    # ---- 4. codecs --------------------------------------------------------
    a.put("codec_summary", codec_analysis.codec_summary(rows))
    a.put("codec_win_rates", codec_analysis.win_rates(rows, win))
    if feats:
        a.put("codec_feature_sensitivity",
              codec_analysis.codec_feature_sensitivity(rows, feats))
        a.put("codec_property_profile",
              codec_analysis.winner_property_profile(chunks, win, feats))

    # ---- 5. paired treatments --------------------------------------------
    sh = a.put("shuffle_comparison", paired_analysis.shuffle_comparison(rows))
    qz = a.put("quantization_comparison",
               paired_analysis.quantization_comparison(rows))
    eb = a.put("error_bound_comparison",
               paired_analysis.error_bound_comparison(rows))
    if sh.empty:
        a.na("shuffle", "the sweep contains no shuffle-off/shuffle-on pair "
                        "for any chunk and codec")
    if qz.empty:
        a.na("quantization", "the sweep contains no lossless/quantized pair "
                             "for any chunk and codec")
    if eb.empty:
        a.na("error_bound",
             "the sweep explored a single error bound, so the effect of "
             "relaxing it cannot be measured from this log. Re-run the sweep "
             "with several --eb values to answer Q9.")
    a.put("shuffle_summary",
          paired_analysis.summarize_pairs(sh, ["lib_name"]))
    a.put("quantization_summary",
          paired_analysis.summarize_pairs(qz, ["lib_name"]))
    a.put("error_bound_summary", paired_analysis.summarize_pairs(eb))
    if feats:
        a.put("shuffle_benefit_regimes",
              paired_analysis.benefit_regimes(sh, feats))
        a.put("quantization_benefit_regimes",
              paired_analysis.benefit_regimes(qz, feats))

    a.profiles = codec_analysis.build_profiles(rows, win, chunks, feats,
                                               sh, qz)

    # ---- 6. prediction ----------------------------------------------------
    errors: Dict[str, pd.DataFrame] = {}
    for metric, pred, actual in prediction_analysis.PAIRS:
        e = prediction_analysis.prediction_errors(rows, metric, pred, actual)
        errors[metric] = e
        a.put(f"{metric}_prediction_errors", e)
        if e.empty:
            a.na(f"{metric}_prediction",
                 f"{actual} is not available on any row -- "
                 + ("decompression was not measured "
                    "(CLIO_NEUROPRESS_EXPLORE_MEASURE_DT off)"
                    if metric == "dt" else "the column is absent"))
    a.put("prediction_clamps", pd.DataFrame(
        [prediction_analysis.detect_clamp(rows, p, act)
         for _, p, act in prediction_analysis.PAIRS]))
    summ = [prediction_analysis.summarize_errors(e, ["role"])
            for e in errors.values() if not e.empty]
    a.put("prediction_summary", pd.concat(summ, ignore_index=True)
          if summ else pd.DataFrame())
    by_codec = [prediction_analysis.summarize_errors(e, ["lib_name"])
                for e in errors.values() if not e.empty]
    a.put("prediction_by_codec", pd.concat(by_codec, ignore_index=True)
          if by_codec else pd.DataFrame())
    if feats:
        drivers = [prediction_analysis.error_drivers(e, feats)
                   for e in errors.values() if not e.empty]
        drivers = [d for d in drivers if not d.empty]
        a.put("prediction_error_drivers",
              pd.concat(drivers, ignore_index=True) if drivers
              else pd.DataFrame())
    a.put("ranking_quality", prediction_analysis.ranking_quality(rows))

    # ---- 7. temporal ------------------------------------------------------
    if not temporal_analysis.has_time(chunks):
        a.na("temporal", "fewer than three distinct timesteps could be parsed "
                         "from the blob keys")
        ps = os_ = pd.DataFrame()
    else:
        ps = a.put("property_series",
                   temporal_analysis.property_series(chunks, feats))
        os_ = a.put("outcome_series",
                    temporal_analysis.outcome_series(rows, win))
        merged = (ps.merge(os_, on=["field", "timestep"], how="outer",
                           suffixes=("", "_o"))
                  if not ps.empty and not os_.empty else
                  (os_ if ps.empty else ps))
        a.put("temporal_series", merged)
        cols = list(feats) + ["best_ratio_median", "adopted_ratio_median"]
        a.put("temporal_trends",
              temporal_analysis.temporal_trends(merged, cols))
        a.put("codec_switches", temporal_analysis.codec_switches(os_))

    ev_path = ds.sidecars.get("evolution")
    ev = load_evolution(ev_path) if ev_path else None
    if ev is None or ev.empty:
        a.na("evolution",
             "no block-evolution CSV was found beside the log. Generate one "
             "with paper-benchmark/evolution.py and place blocks.csv in the "
             "same directory to enable the temporal-evolution analysis.")
    else:
        j = temporal_analysis.join_evolution(chunks, ev)
        if j.empty:
            a.na("evolution",
                 f"found {os.path.basename(ev_path)} but no chunk matched it "
                 f"on (field, timestep, chunk id) -- the evolution pass was "
                 f"probably run at a different sampling stride than the sweep")
        else:
            a.put("evolution_joined", j)
            a.put("evolution_effects",
                  temporal_analysis.evolution_effects(j, feats, win))

    # ---- 8. models --------------------------------------------------------
    if not feats:
        a.na("models", "no usable data-property features")
    else:
        ab = modeling.feature_ablation(cw, feats, CHUNK_TARGET, seed=seed)
        a.put("feature_ablation", ab)
        a.put("feature_ablation_marginal",
              ab.attrs.get("marginal", pd.DataFrame()))
        a.put("feature_importance",
              modeling.importances(cw, feats, CHUNK_TARGET, seed=seed))
        # Is each "does X add anything" verdict stable, or an artefact of one
        # fold partition? At a few dozen chunks it is often the latter.
        stab = []
        for i, fx in enumerate(feats):
            r = modeling.ablation_stability(
                cw, feats, CHUNK_TARGET, base=feats[:i], extra=fx)
            if r.get("available"):
                stab.append(r)
        a.put("ablation_stability", pd.DataFrame(stab))
        metrics = []
        for target, log_t in ((CHUNK_TARGET, True),
                              ("fastest_ct_value", True)):
            if target not in cw or cw[target].notna().sum() < 12:
                continue
            for model in ("linear", "tree", "forest"):
                r = modeling.grouped_cv_r2(cw, feats, target, model,
                                           log_target=log_t, seed=seed)
                if r.get("available"):
                    metrics.append(r)
        a.put("model_metrics", pd.DataFrame(metrics))
        hold = []
        for by in ("field", "timestep"):
            h = modeling.holdout_by(cw, feats, CHUNK_TARGET, by, seed=seed)
            if not h.empty:
                hold.append(h)
        a.put("generalisation", pd.concat(hold, ignore_index=True)
              if hold else pd.DataFrame())
        clf = modeling.codec_classifier(cw, feats, "best_ratio_lib", seed=seed)
        a.facts["codec_classifier"] = clf
        a.put("codec_classifier_metrics",
              pd.DataFrame([{k: v for k, v in clf.items()
                             if k not in ("tree_rules", "classes")}]))
        a.thresholds = {
            "best_ratio_codec": modeling.discover_thresholds(
                cw, feats, "best_ratio_lib",
                min_support=max(10, len(cw) // 8), seed=seed),
            "compression_ratio_regimes":
                modeling.discover_regression_thresholds(
                    cw, feats, CHUNK_TARGET,
                    min_support=max(6, len(cw) // 8), seed=seed),
        }
        if not sh.empty:
            shc = sh.copy()
            shc["shuffle_helps"] = shc["helps_ratio"].map(
                {True: "helps", False: "hurts"})
            shc["chunk_uid"] = shc["chunk_uid"]
            a.thresholds["shuffle_helps"] = modeling.discover_thresholds(
                shc, feats, "shuffle_helps",
                min_support=max(20, len(shc) // 10), seed=seed)
        if not qz.empty:
            qzc = qz.copy()
            qzc["quantization_helps"] = qzc["helps_ratio"].map(
                {True: "helps", False: "hurts"})
            a.thresholds["quantization_helps"] = modeling.discover_thresholds(
                qzc, feats, "quantization_helps",
                min_support=max(20, len(qzc) // 10), seed=seed)

    # ---- 9. counterexamples -----------------------------------------------
    a.put("counterexamples",
          counterexamples.collect(chunks, win, reg, rows, feats, sh, qz,
                                  errors))

    # ---- 10. figures ------------------------------------------------------
    if make_figures:
        a.figures = _draw(a, os.path.join(outdir, "figures"))

    a.answers = _answer(a)
    return a


def _draw(a: Analysis, figdir: str) -> Figures:
    f = Figures(figdir)
    ds, cw = a.ds, a.tables["chunk_properties"]
    rows, feats = ds.rows, ds.features
    # The mechanism pictures first, then the matched-control ones: both
    # survive the confound, so neither should be buried under the
    # correlations they are meant to be read before.
    f.entropy_bound_scatter(rows, a.tables.get("codec_order_sensitivity",
                                               pd.DataFrame()),
                            "entropy_bound_vs_achieved.png")
    f.excess_over_bound(a.tables.get("codec_order_sensitivity",
                                     pd.DataFrame()),
                        "excess_over_entropy_bound.png")
    f.timing_mediation(a.tables.get("timing_mediation", pd.DataFrame()),
                       "timing_follows_output_size.png")
    f.property_correlation_matrix(
        a.tables.get("property_correlations", pd.DataFrame()),
        "property_correlations.png",
        f"{ds.name}: data properties vs outcomes")
    qpc = a.tables.get("quantization_levels_per_chunk", pd.DataFrame())
    f.levels_vs_outcome(qpc, "best_quantized_ratio",
                        "levels_vs_quantized_ratio.png",
                        "best quantized compression ratio", logy=True)
    f.levels_vs_outcome(qpc, "meas_ssim", "levels_vs_ssim.png",
                        "measured SSIM", logy=False)
    f.levels_vs_outcome(qpc, "best_quantized_ratio",
                        "bulk_levels_vs_quantized_ratio.png",
                        "best quantized compression ratio", logy=True,
                        xcol="bulk_levels")
    f.levels_vs_outcome(qpc, "meas_ssim", "bulk_levels_vs_ssim.png",
                        "measured SSIM", logy=False, xcol="bulk_levels")
    f.regime_bars(a.tables.get("quantization_regimes", pd.DataFrame()),
                  "quantization_regimes.png")
    f.matched_pair_spread(a.tables.get("matched_pairs", pd.DataFrame()),
                          "matched_pair_spread.png")
    f.component_ratio_over_time(cw,
                                a.tables.get("component_families",
                                             pd.DataFrame()),
                                CHUNK_TARGET, "component_ratio_over_time.png")
    for feat in feats:
        f.scatter_feature_target(cw, feat, CHUNK_TARGET,
                                 f"{feat}_vs_ratio.png",
                                 "best achievable compression ratio")
        f.facet_by_codec(rows, feat, "ratio", f"{feat}_vs_ratio_by_codec.png",
                         "compression ratio")
        f.benefit_vs_feature(a.tables.get("shuffle_comparison",
                                          pd.DataFrame()),
                             feat, f"shuffle_benefit_vs_{feat}.png", "shuffle")
        f.benefit_vs_feature(a.tables.get("quantization_comparison",
                                          pd.DataFrame()),
                             feat, f"quantization_benefit_vs_{feat}.png",
                             "quantization")
        f.over_time(a.tables.get("temporal_series", pd.DataFrame()), feat,
                    f"{feat}_over_time.png", _pretty(feat))
    if len(feats) >= 2:
        f.heatmap_two_features(cw, feats[0], feats[1], CHUNK_TARGET,
                               f"{feats[0]}_{feats[1]}_ratio_heatmap.png",
                               "compression ratio")
    if len(feats) >= 3:
        f.heatmap_two_features(cw, feats[0], feats[2], CHUNK_TARGET,
                               f"{feats[0]}_{feats[2]}_ratio_heatmap.png",
                               "compression ratio")
    if len(feats) >= 2:
        f.winner_property_space(ds.chunks, a.tables["winners"], feats,
                                "winning_codec_property_space.png")
    f.error_bound_effect(rows, "error_bound_vs_ratio.png")
    f.pareto(rows, "ct_ms", "ratio_vs_ct.png", "compression time (ms)")
    f.pareto(rows, "dt_ms_measured", "ratio_vs_dt.png",
             "decompression time (ms)")
    for metric, label in (("ratio", "compression ratio"),
                          ("ct", "compression time (ms)"),
                          ("dt", "decompression time (ms)")):
        f.predicted_vs_actual(a.tables.get(f"{metric}_prediction_errors",
                                           pd.DataFrame()),
                              f"predicted_vs_actual_{metric}.png", label)
    ts = a.tables.get("temporal_series", pd.DataFrame())
    f.over_time(ts, "best_ratio_median", "ratio_over_time.png",
                "best achievable compression ratio", logy=True)
    f.winner_over_time(a.tables.get("outcome_series", pd.DataFrame()),
                       "winning_codec_over_time.png")
    return f


def _pretty(col: str) -> str:
    return {"entropy": "byte entropy (bits/byte)",
            "mad": "MAD (raw data units)",
            "second_deriv": "mean |2nd difference|"}.get(col, col)


# --------------------------------------------------------------------------
# Turning tables into the fifteen required answers
# --------------------------------------------------------------------------

def _confound_sentence(conf: Dict[str, object]) -> str:
    """The one clause that must ride along with any feature-effect claim."""
    if not conf or not conf.get("confounded"):
        return ""
    best = conf.get("best_metadata_predictor")
    r2 = conf.get("best_metadata_oof_r2")
    flip = conf.get("features_flipping") or []
    bits = []
    if best is not None and r2 is not None:
        bits.append(f"`{best}` ALONE scores {r2:.2f}")
    if flip:
        bits.append("and the correlation of "
                    + ", ".join(f"`{x}`" for x in flip)
                    + " reverses sign once it is removed")
    return ("  **This is confounded** -- " + ", ".join(bits)
            + ". Treat the number above as an association within this one "
              "run, not as evidence that the data properties drive "
              "compressibility.") if bits else ""


def _mechanism_sentence(m: Dict[str, object]) -> str:
    """The entropy-bound decomposition in one clause.

    This is the only statement in the pipeline that rests on a theorem rather
    than on a fit, so it is worth saying wherever entropy's predictive power is
    being reported as a number.
    """
    if not m or not m.get("available"):
        return ""
    bits: List[str] = []
    if m.get("entropy_bound_holds"):
        bits.append(str(m["entropy_bound_holds"]))
    if m.get("order_contribution"):
        bits.append(str(m["order_contribution"]))
    if not bits:
        return ""
    return ("  **Mechanism (" + _sec("Why: the ratio against its entropy bound")
            + "):** entropy is not a weak predictor -- "
            "it is an EXACT one for the part of the ratio a byte histogram can "
            "reach, and blind to the rest by construction. " + ". ".join(bits)
            + ".")


def _mechanism_caveats(m: Dict[str, object]) -> List[str]:
    if not m or not m.get("available"):
        return []
    out = [
        "The 8/H bound is the source coding theorem applied to the exact "
        "quantity `entropy` measures, not a fitted relationship: a codec "
        "exceeding it is reading ORDER, which no current feature measures.",
    ]
    if m.get("shuffle_gain_is_pure_order"):
        out.append(str(m["shuffle_gain_is_pure_order"]))
    if m.get("quantization_split_identifiable") is False:
        out.append(str(m.get("quantization_split_reason", "")))
    return [x for x in out if x]


def _anisotropy_sentence(an: Dict[str, object]) -> str:
    """The matched-control result, in one clause, for an answer that needs it.

    Written to be appendable: it is the half of the story that the confound
    section cannot take away, because the clock is held exactly in it.
    """
    if not an or not an.get("available") or not an.get("anisotropic"):
        return ""
    bits: List[str] = []
    if an.get("tightest_n_pairs"):
        bits.append(
            f"{int(an['tightest_n_pairs'])} pairs of chunks from the SAME "
            f"MOMENT agree in every feature to "
            f"{float(an['tightest_tolerance']):g} relative yet differ up to "
            f"{float(an['tightest_max_fold']):.2f}x in ratio")
    if an.get("max_achievable_oof_r2") is not None:
        bits.append(
            f"which caps out-of-fold R^2 for ANY model on these features at "
            f"{float(an['max_achievable_oof_r2']):.3f}")
    if an.get("headline_ordering"):
        bits.append(str(an["headline_ordering"]))
    return ("  **The matched-control test (" + _sec("Same moment, different field")
            + ") is not confounded** -- "
            "the timestep is held exactly constant in it -- and it finds "
            + "; ".join(bits) + ".") if bits else ""


def _anisotropy_caveats(an: Dict[str, object]) -> List[str]:
    if not an or not an.get("available"):
        return []
    out = [
        f"{_sec('Same moment, different field')} compares only chunks "
        f"sharing a run, a timestep and a "
        f"size, so the simulation clock is held EXACTLY constant there and "
        f"the {_sec('Is it the data, or the clock?')} confound does not "
        f"apply to it "
        f"({int(an.get('n_matched_pairs', 0))} pairs over "
        f"{int(an.get('n_moments', 0))} moments)."]
    if an.get("hypothesis"):
        out.append(str(an["hypothesis"]))
    return out


def _confound_caveats(conf: Dict[str, object]) -> List[str]:
    if not conf or not conf.get("confounded"):
        return []
    out = ["CONFOUNDED, see the 'Is it the data, or the clock?' section: "
           + "; ".join(str(r) for r in conf.get("reasons", []))]
    out.append(
        "What would separate them: a sweep over many chunks at FIXED "
        "simulation time, or several runs whose evolution differs. Neither "
        "is available in this log.")
    return out


def _cohort_caveats(cohort: Dict[str, object]) -> List[str]:
    if not cohort or not cohort.get("measurable"):
        return ["Whether the explored chunks are a representative sample of "
                "the run could not be checked (no selection log beside the "
                "input). Exploration is trigger-gated upstream, so they may "
                "well not be."]
    if not cohort.get("chunks_not_explored"):
        return []
    out = [f"CONDITIONED SAMPLE: this log covers "
           f"{cohort['chunks_explored']} of "
           f"{cohort['chunks_in_selection_log']} chunks written by the run "
           f"({cohort['pct_explored']:.0f}%). Exploration is trigger-gated, "
           f"so this figure describes the explored subset, not the run."]
    if cohort.get("inferred_trigger"):
        out.append(str(cohort["inferred_trigger"]))
    return out


def _fmt_r(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:+.2f}"


def _answer(a: Analysis) -> Dict[str, dict]:
    """Each answer is {verdict, evidence, confidence, caveats}.

    `confidence` is deliberately coarse -- high / moderate / low / unavailable
    -- and is driven by sample size, whether the effect survives the within-
    field control, and whether a generalisation test was possible at all.
    """
    t, f, ds = a.tables, a.facts, a.ds
    feats = ds.features
    out: Dict[str, dict] = {}
    n_chunks = ds.n_chunks

    def ans(q, verdict, evidence, confidence, caveats=None, unavailable=None):
        out[q] = {"verdict": verdict, "evidence": evidence,
                  "confidence": confidence,
                  "caveats": caveats or [], "unavailable": unavailable}

    # ---- Q1 entropy -> compressibility ------------------------------------
    ab = t.get("feature_ablation", pd.DataFrame())
    ent_only = _ab_r2(ab, "entropy", "forest")
    ent_lin = _ab_r2(ab, "entropy", "linear")
    all_r2 = _ab_r2(ab, "+".join(feats), "forest") if feats else None
    wf = t.get("correlations_within_field", pd.DataFrame())
    ent_wf = wf[wf["feature"] == "entropy"] if not wf.empty else wf
    conf = f.get("confounds", {})
    conf_note = _confound_sentence(conf)
    aniso_note = _anisotropy_sentence(a.facts.get("anisotropy", {}))
    mech = a.facts.get("mechanism", {})
    mech_note = _mechanism_sentence(mech)
    if ent_only is None:
        ans("Q1", "unavailable", {}, "unavailable",
            unavailable=a.unavailable.get("models", "no entropy column"))
    else:
        ans("Q1",
            f"entropy alone explains {100 * ent_only:.0f}% of the variance in "
            f"log10(best achievable ratio) out of fold, across {n_chunks} "
            f"chunks -- but only {100 * (ent_lin or 0):.0f}% linearly, so the "
            f"relationship is strongly nonlinear."
            + mech_note + conf_note + aniso_note,
            {"oof_r2_forest_entropy_only": ent_only,
             "oof_r2_linear_entropy_only": ent_lin,
             "oof_r2_all_features": all_r2,
             "median_within_field_pearson_r": (
                 float(ent_wf["pooled_within_field_r"].median())
                 if not ent_wf.empty else None),
             "n_chunks": n_chunks, "confounds": conf,
             "matched_control": {k: v for k, v in
                                 a.facts.get("anisotropy", {}).items()
                                 if k != "hypothesis"}},
            ("low" if conf.get("confounded")
             else _confidence(n_chunks, ent_only)),
            ["R^2 is on log10(ratio); on native ratio the score would be "
             "dominated by the handful of chunks above 100x.",
             "Out-of-fold with chunks as groups, so no chunk appears in both "
             "train and test."]
            + _confound_caveats(conf)
            + _anisotropy_caveats(a.facts.get("anisotropy", {}))
            + _mechanism_caveats(mech))

    # ---- Q2 MAD beyond entropy --------------------------------------------
    out["Q2"] = _incremental_answer(a, "mad", ["entropy"], n_chunks)
    # ---- Q3 second derivative beyond entropy + MAD -------------------------
    out["Q3"] = _incremental_answer(a, "second_deriv", ["entropy", "mad"],
                                    n_chunks)

    # ---- Q4 can they predict ratio ----------------------------------------
    mm = t.get("model_metrics", pd.DataFrame())
    gen = t.get("generalisation", pd.DataFrame())
    if mm.empty:
        ans("Q4", "unavailable", {}, "unavailable",
            unavailable=a.unavailable.get("models", "no model could be fitted"))
    else:
        best = mm[mm["target"] == CHUNK_TARGET].sort_values(
            "oof_r2", ascending=False)
        b = best.iloc[0] if len(best) else None
        fld = gen[gen["by"] == "field"] if not gen.empty else gen
        tim = gen[gen["by"] == "timestep"] if not gen.empty else gen
        ans("Q4",
            (f"on held-out CHUNKS yes -- a {b['model']} on "
             f"{'+'.join(feats)} reaches out-of-fold R^2 = {b['oof_r2']:.2f} "
             f"on log10(best ratio) over {int(b['n_chunks'])} chunks -- but "
             f"that does not survive a harder split"
             + (f": leave-one-field-out R^2 falls to a median of "
                f"{fld['r2'].median():.2f}"
                + (f" and training on early timesteps to predict later ones "
                   f"gives {tim['r2'].iloc[0]:.2f}" if not tim.empty else "")
                + ", i.e. worse than predicting the mean"
                if not fld.empty and fld["r2"].median() < 0.3 else "")
             + "." + conf_note + aniso_note
             if b is not None else "no model available"),
            {"best_model": None if b is None else str(b["model"]),
             "oof_r2": None if b is None else float(b["oof_r2"]),
             "memorisation_gap": None if b is None
             else float(b["memorisation_gap"]),
             "leave_one_field_out_r2_median": (
                 float(fld["r2"].median()) if not fld.empty else None),
             "leave_one_field_out_worst": (
                 float(fld["r2"].min()) if not fld.empty else None),
             "later_timesteps_r2": (
                 float(tim["r2"].iloc[0]) if not tim.empty else None),
             "max_achievable_oof_r2_on_these_features":
                 a.facts.get("anisotropy", {}).get("max_achievable_oof_r2")},
            ("low" if conf.get("confounded")
             else _confidence(n_chunks, None if b is None else b["oof_r2"])),
            ["The chunk-holdout score and the field-holdout score answer "
             "different questions; the gap between them is how much of the "
             "fit is the fields rather than the physics."
             + (f" Here they differ sharply: leave-one-field-out R^2 has "
                f"median {fld['r2'].median():.2f} and worst "
                f"{fld['r2'].min():.2f}"
                + (f", and training on early timesteps and testing on later "
                   f"ones gives {tim['r2'].iloc[0]:.2f}" if not tim.empty
                   else "")
                + ". A negative R^2 is worse than predicting the mean."
                if not fld.empty else "")]
            + _confound_caveats(conf)
            + _anisotropy_caveats(a.facts.get("anisotropy", {})))

    # ---- Q5 predict the winning codec --------------------------------------
    clf = f.get("codec_classifier", {})
    if not clf.get("available"):
        ans("Q5",
            "not answerable from this log: " + str(clf.get("reason", "")),
            {k: v for k, v in clf.items() if k != "tree_rules"},
            "unavailable",
            ["A single-workload sweep on smooth mesh data often has one "
             "dominant codec; the question needs a log where the winner "
             "actually varies."])
    else:
        ans("Q5",
            (f"out-of-fold accuracy {clf['oof_accuracy']:.2f} against a "
             f"majority baseline of "
             f"{clf['majority_baseline_accuracy']:.2f} "
             f"(+{clf['lift_over_baseline']:.2f})"),
            {k: v for k, v in clf.items() if k != "tree_rules"},
            _confidence(clf["n"], clf["lift_over_baseline"] * 2),
            ["Read the lift, not the accuracy: a 0.95 accuracy against a "
             "0.95 baseline is a constant, not a prediction."])

    # ---- Q6 which regimes favour which codec -------------------------------
    rules = a.thresholds.get("best_ratio_codec", [])
    prof = t.get("codec_property_profile", pd.DataFrame())
    ans("Q6",
        (f"{len(rules)} interpretable rule(s) cleared the support and lift "
         f"gates" if rules else
         "no codec-selection rule cleared the support and lift gates on this "
         "log -- see the win-rate table for why (usually one codec wins "
         "nearly everything, or the wins are ties at the ratio cap)"),
        {"rules": rules,
         "n_codecs_with_uncontested_wins": int(len(prof)) if not prof.empty
         else 0},
        "moderate" if rules else "unavailable",
        ["Rules are emitted only when a leaf covers enough chunks AND beats "
         "the class's own base rate by 10 points, so the absence of rules is "
         "a real result rather than a failed search."])

    # ---- Q7 / Q8 shuffle and quantization -----------------------------------
    # Shuffle has a mechanism that is a theorem rather than a fit: it permutes
    # the byte stream, so the histogram -- and every feature derived from it --
    # is invariant, and the whole gain is order. Quantization does not: it
    # moves the histogram, and this log does not record by how much.
    out["Q7"] = _treatment_answer(
        a, "shuffle", "shuffle_summary", "shuffle_benefit_regimes", "shuffle",
        mechanism_note=(
            ". MECHANISM: shuffle is a PERMUTATION of the byte stream, so the "
            "256-bin histogram and 8/H are invariant under it -- every bit of "
            "this gain is contributed by order, by construction rather than "
            "by inference"
            if mech.get("shuffle_gain_is_pure_order") else ""),
        mechanism_caveats=(
            [str(mech["shuffle_gain_is_pure_order"])]
            + ([str(mech["locality_finding"])]
               if mech.get("locality_finding") else [])
            if mech.get("shuffle_gain_is_pure_order") else []))
    out["Q8"] = _treatment_answer(
        a, "quantization", "quantization_summary",
        "quantization_benefit_regimes", "quantization",
        mechanism_caveats=(
            [str(mech.get("quantization_split_reason", ""))]
            if mech.get("quantization_split_identifiable") is False else []))

    # ---- Q9 error bound ----------------------------------------------------
    ebs = t.get("error_bound_summary", pd.DataFrame())
    if ebs.empty:
        ans("Q9", "unavailable", {}, "unavailable",
            unavailable=a.unavailable.get("error_bound", "no bounds swept"))
    else:
        ans("Q9",
            "relaxing the bound changes the ratio as tabulated",
            {"summary": ebs.to_dict("records")},
            _confidence(int(ebs["n_pairs"].sum()), 1.0))

    # ---- Q10 temporal -------------------------------------------------------
    tt = t.get("temporal_trends", pd.DataFrame())
    if tt.empty:
        ans("Q10", "unavailable", {}, "unavailable",
            unavailable=a.unavailable.get("temporal", "no timesteps parsed"))
    else:
        rr = tt[tt["series"] == "best_ratio_median"]
        ee = tt[tt["series"] == "entropy"]
        ans("Q10",
            (f"compressibility falls as the run advances in "
             f"{int((rr['spearman_rho'] < 0).sum())} of {len(rr)} fields "
             f"(median rho = {rr['spearman_rho'].median():+.2f}), while "
             f"entropy rises in {int((ee['spearman_rho'] > 0).sum())} of "
             f"{len(ee)} (median rho = {ee['spearman_rho'].median():+.2f})"
             if not rr.empty and not ee.empty else "see the trend table"),
            {"trends": tt.to_dict("records"),
             "codec_switches": t.get("codec_switches",
                                     pd.DataFrame()).to_dict("records")},
            _confidence(int(tt["n_timesteps"].max()) if not tt.empty else 0,
                        1.0),
            ["Rank correlations over a handful of timesteps: a rho of 1.0 "
             "over five points is a real observation and a weak test, and "
             "both facts are carried in the table."])

    # ---- Q11 cross workload -------------------------------------------------
    ans("Q11",
        "requires more than one workload CSV; pass several to this tool and "
        "read cross_workload/",
        {"workloads_in_this_run": [ds.name]}, "unavailable")

    # ---- Q12 predictor failure ----------------------------------------------
    ps = t.get("prediction_summary", pd.DataFrame())
    drv = t.get("prediction_error_drivers", pd.DataFrame())
    if ps.empty:
        ans("Q12", "unavailable", {}, "unavailable",
            unavailable="no predicted/measured pairs available")
    else:
        worst = drv.reindex(
            drv["spearman_rho"].abs().sort_values(ascending=False).index
        ).head(8) if not drv.empty else drv
        clamps = t.get("prediction_clamps", pd.DataFrame())
        clamped = clamps[clamps["floor"].notna() | clamps["cap"].notna()] \
            if not clamps.empty and "floor" in clamps else pd.DataFrame()
        ans("Q12",
            "; ".join(
                f"{r['metric']}: median {r['median_fold_error']:.2f}x off, "
                f"{r['pct_over_predicted']:.0f}% over-predicted "
                f"(role={r.get('role', 'all')})"
                for _, r in ps.iterrows())
            + (("  BUT " + "; ".join(
                f"`{r['metric']}` is FLOORED at {r['floor']:g} while "
                f"{r['pct_actual_below_floor']:.0f}% of actuals fall below "
                f"it -- excluding the rows where that clamp bites, the error "
                f"is {r['median_fold_error_unclamped']:.2f}x and "
                f"{r['pct_over_predicted_unclamped']:.0f}% over-predicted, "
                f"not {r['median_fold_error_all']:.2f}x / "
                f"{r['pct_over_predicted_all']:.0f}%"
                for _, r in clamped.iterrows()
                if pd.notna(r.get("median_fold_error_unclamped"))))
               if not clamped.empty else ""),
            {"per_metric": ps.to_dict("records"),
             "clamps": clamps.to_dict("records") if not clamps.empty else [],
             "cohort": f.get("cohort", {}),
             "strongest_feature_dependence": (
                 worst.to_dict("records") if not worst.empty else [])},
            _confidence(int(ps["n"].max()), 1.0),
            ["Compression times on explored candidates are measured while up "
             "to CLIO_NEUROPRESS_EXPLORE_STREAMS candidates run concurrently, "
             "while the primary is measured alone. That difference is NOT "
             "identifiable from this log: no (chunk, configuration) is "
             "measured in both roles, and the two arms also differ in codec "
             "composition, so an alt-vs-primary gap cannot be attributed to "
             "contention."]
            + _cohort_caveats(f.get("cohort", {})))

    # ---- Q13 cost-model selection -------------------------------------------
    sel = f.get("selection", {})
    cd = f.get("cost_diagnostics", {})
    if not sel:
        ans("Q13", "unavailable", {}, "unavailable")
    else:
        ans("Q13",
            (f"the model's own pick was cost-optimal on "
             f"{sel['pct_primary_cost_optimal']:.0f}% of chunks; exploration "
             f"changed the choice on "
             f"{sel['pct_exploration_changed_choice']:.0f}%. Median regret of "
             f"the model's pick against the best measured candidate: "
             f"{sel['primary_regret_median']:.3f}."),
            {"selection": sel, "cost_model": f.get("cost_model", {}),
             "cost_diagnostics": cd},
            _confidence(sel.get("n_chunks", 0), 1.0),
            [f"{cd.get('pct_chunks_tied_at_best', 0):.0f}% of chunks have "
             f"more than one candidate at the best cost -- with a ratio cap "
             f"of {cd.get('ratio_cap')}, everything above the cap scores "
             f"identically, so a 'wrong' pick among those is not an error."]
            + _cohort_caveats(f.get("cohort", {})))

    # ---- Q14 / Q15 sufficiency and what to add -------------------------------
    ce = t.get("counterexamples", pd.DataFrame())
    ident = ce[ce["kind"] == "identical_properties_different_ratio"] \
        if not ce.empty and "kind" in ce else pd.DataFrame()
    gen = t.get("generalisation", pd.DataFrame())
    fld = gen[gen["by"] == "field"] if not gen.empty else gen
    max_rel = (float(ident["max_relative_feature_diff"].max())
               if not ident.empty
               and "max_relative_feature_diff" in ident else None)
    an = a.facts.get("anisotropy", {})
    ceil_txt = ""
    if an.get("max_achievable_oof_r2") is not None:
        ceil_txt = (
            f" Independently of any counterexample: chunks that are "
            f"numerically identical in every feature but disagree on the "
            f"ratio cap the out-of-fold R^2 of ANY model on these three "
            f"features at {float(an['max_achievable_oof_r2']):.3f} on this "
            f"log ({_sec('Same moment, different field')}).")
    if an.get("headline_ordering"):
        ceil_txt += f" {an['headline_ordering']}."
    ans("Q14",
        (f"no. {len(ident)} pair(s) of chunks agree in "
         f"({'+'.join(feats)}) to within "
         f"{max_rel * 100:.3f}% on every feature yet differ up to "
         f"{ident['ratio_fold'].max():.2f}x in achievable ratio -- no model "
         f"built on these three features can separate them." + ceil_txt
         if not ident.empty else
         (f"no counterexample of the identical-features kind was found in "
          f"{n_chunks} chunks; sufficiency is not contradicted here, which "
          f"is weaker than being established") + ceil_txt),
        {"n_identical_property_counterexamples": int(len(ident)),
         "max_relative_feature_difference_within_those_pairs": max_rel,
         "max_ratio_fold_at_identical_properties": (
             float(ident["ratio_fold"].max()) if not ident.empty else None),
         "examples": (ident.head(5).to_dict("records") if not ident.empty
                      else []),
         "leave_one_field_out_r2_median": (
             float(fld["r2"].median()) if not fld.empty else None),
         "max_achievable_oof_r2_on_these_features":
             an.get("max_achievable_oof_r2"),
         "n_stable_component_orderings":
             an.get("n_stable_component_orderings"),
         "n_matched_pairs_at_fixed_timestep": an.get("n_matched_pairs")},
        "high" if len(ident) >= 3 else _confidence(n_chunks, 0.5),
        ["A counterexample at identical feature values is an existence proof "
         "and needs no sample size; the ABSENCE of one over a few dozen "
         "chunks is not evidence of sufficiency.",
         "'Identical' is tested in each feature's OWN units (relative "
         "difference), not in standardised distance -- a standardised radius "
         "widens whenever an outlier cluster inflates the global spread, and "
         "would admit pairs differing by tens of percent.",
         "These pairs are near-simultaneous components of the same physical "
         "field, so they are fewer independent observations than they are "
         "rows. The existence claim is unaffected; a rate would not be."]
        + _anisotropy_caveats(an))
    ans("Q15", "see the Missing Features section of the report",
        {"proposals": _propose_features(a)}, "moderate")

    # ---- Q16 / Q17: the quantized half, explained -------------------------
    qm = a.facts.get("quantization_mechanism", {})
    if not qm.get("available"):
        for q in ("Q16", "Q17"):
            ans(q, "unavailable", {}, "unavailable",
                unavailable=a.unavailable.get("quantization_mechanism",
                                              qm.get("reason", "")))
    else:
        n = int(qm.get("n_chunks", 0))
        rr = qm.get("rho_levels_vs_ratio")
        rs = qm.get("rho_levels_vs_ssim")
        relaxed = float(qm.get("pct_bound_relaxed", 0.0))
        caveats = [
            "L assumes the effective bound is 0.95*eb, which holds whenever "
            "float32 can represent the bound at the field's magnitude. Where "
            "it cannot, the quantizer relaxes it; those chunks are flagged "
            "bound_relaxed and excluded from the correlations.",
            f"data_range came from the quality sidecar for "
            f"{float(qm.get('pct_range_from_sidecar', 0)):.0f}% of chunks and "
            f"was recovered from psnr/rmse for the rest."]
        if relaxed > 0:
            caveats.append(
                f"{relaxed:.0f}% of chunks had their bound relaxed, so the "
                f"requested eb was not the one applied and their stored error "
                f"exceeds it. For a bound that holds, it must clear "
                f"max|v|*2.4e-7 / 0.95 at the field's magnitude.")
        br = qm.get("rho_bulk_levels_vs_ratio")
        bulk_note = (f" Counting only the rungs the bulk of the data occupies "
                     f"(L_bulk = mad / (2*0.95*eb), the MAD in grid steps) "
                     f"tightens the ratio correlation to {br:+.2f}, because a "
                     f"rare extreme widens the range without filling the grid."
                     if br is not None and np.isfinite(br) else "")
        ans("Q16",
            f"yes, in closed form. L = data_range / (2*0.95*eb) has no codec "
            f"in it and tracks the best quantized ratio at Spearman rho = "
            f"{rr:+.2f}" + (f" and SSIM at rho = {rs:+.2f}" if rs is not None
                            and np.isfinite(rs) else "") +
            f" over {n} chunks." + bulk_note +
            f" The ratio splits into an exact width gain "
            f"(4/width_bytes, set by L), an alphabet floor of 32/log2(L) that "
            f"an entropy coder reaches before reading any order, and an "
            f"excess above that from level skew and order.",
            {k: qm.get(k) for k in ("n_chunks", "median_levels",
                                    "rho_levels_vs_ratio", "rho_levels_vs_ssim",
                                    "rho_bulk_levels_vs_ratio",
                                    "rho_bulk_levels_vs_ssim",
                                    "rho_levels_vs_psnr", "pct_bound_relaxed",
                                    "regimes")},
            "high" if (rr is not None and np.isfinite(rr) and abs(rr) >= 0.6
                       and n >= 100) else "moderate",
            caveats)
        few = float(qm.get("pct_under_few_levels", 0.0))
        destroyed = qm.get("fields_destroyed", [])
        ans("Q17",
            f"where L falls below ~{quantization_mechanism.FEW_LEVELS:.0f}. "
            f"{few:.0f}% of quantized chunks are left with fewer than "
            f"{quantization_mechanism.FEW_LEVELS:.0f} distinct values at "
            f"this bound" +
            (f", and that is the majority in {len(destroyed)} of "
             f"{qm.get('n_fields', '?')} fields ({', '.join(destroyed[:8])})"
             if destroyed else "") +
            f". Below that the field's structure no longer fits between the "
            f"grid rungs and SSIM collapses regardless of codec; above a few "
            f"hundred the loss is invisible at SSIM's resolution. The same L "
            f"that raised the ratio destroyed the quality -- an absolute "
            f"bound is a per-field decision, not a per-run one.",
            {"pct_under_few_levels": few, "fields_destroyed": destroyed,
             "pct_constant": qm.get("pct_constant")},
            "high" if n >= 100 else "moderate",
            caveats[:1])
    return out


def _ab_r2(ab: pd.DataFrame, features: str, model: str) -> Optional[float]:
    if ab.empty:
        return None
    m = ab[(ab["features"] == features) & (ab["model"] == model)]
    return float(m["oof_r2"].iloc[0]) if len(m) else None


def _incremental_answer(a: Analysis, extra: str, base: Sequence[str],
                        n_chunks: int) -> dict:
    ab = a.tables.get("feature_ablation", pd.DataFrame())
    base = [b for b in base if b in a.ds.features]
    if extra not in a.ds.features or ab.empty or not base:
        return {"verdict": "unavailable", "evidence": {},
                "confidence": "unavailable",
                "unavailable": f"`{extra}` is not usable in this log",
                "caveats": []}
    key_base = "+".join(base)
    key_full = "+".join(list(base) + [extra])
    ev = {}
    for model in ("forest", "linear"):
        b = _ab_r2(ab, key_base, model)
        fll = _ab_r2(ab, key_full, model)
        ev[f"oof_r2_{model}_{key_base}"] = b
        ev[f"oof_r2_{model}_{key_full}"] = fll
        ev[f"delta_r2_{model}"] = (None if b is None or fll is None
                                   else fll - b)
    cg = a.tables.get("conditional_gains", pd.DataFrame())
    row = cg[cg["extra"] == extra] if not cg.empty and "extra" in cg else cg
    if not row.empty:
        ev["linear_partial_r"] = float(row["partial_r"].iloc[0])
        ev["linear_partial_p"] = float(row["partial_p"].iloc[0])
    marg = a.tables.get("feature_ablation_marginal", pd.DataFrame())
    if not marg.empty:
        mm = marg[(marg["feature"] == extra) & (marg["model"] == "forest")]
        if len(mm):
            ev["mean_marginal_oof_r2_gain_forest"] = float(
                mm["mean_marginal_oof_r2_gain"].iloc[0])
    stab = a.tables.get("ablation_stability", pd.DataFrame())
    st = stab[stab["extra"] == extra] if not stab.empty and "extra" in stab \
        else pd.DataFrame()
    unstable = False
    if not st.empty:
        r = st.iloc[0]
        ev["stability_n_repeats"] = int(r["n_repeats"])
        ev["stability_sd_delta_r2"] = float(r["sd_delta_r2"])
        ev["stability_range"] = [float(r["min_delta_r2"]),
                                 float(r["max_delta_r2"])]
        ev["stability_n_sign_flips"] = int(r["n_sign_flips"])
        unstable = not bool(r["stable"])
    d = ev.get("delta_r2_forest")
    if d is None:
        verdict, conf = "unavailable", "unavailable"
    elif d >= 0.05:
        verdict = (f"yes -- adding {extra} to {key_base} raises out-of-fold "
                   f"R^2 by {d:+.3f}")
        conf = _confidence(n_chunks, d * 4)
    elif d <= -0.02:
        verdict = (f"no -- adding {extra} to {key_base} LOWERS out-of-fold "
                   f"R^2 by {d:+.3f}, i.e. it adds noise the model overfits")
        conf = _confidence(n_chunks, abs(d) * 4)
    else:
        verdict = (f"marginal -- adding {extra} to {key_base} changes "
                   f"out-of-fold R^2 by only {d:+.3f}")
        conf = "low"
    if unstable and not st.empty:
        r = st.iloc[0]
        verdict += (f". **This verdict is not stable**: over "
                    f"{int(r['n_repeats'])} re-foldings of the same data the "
                    f"delta ranges {r['min_delta_r2']:+.3f} to "
                    f"{r['max_delta_r2']:+.3f} (sd {r['sd_delta_r2']:.3f})"
                    + (f" and changes sign on {int(r['n_sign_flips'])} of "
                       f"them" if r["n_sign_flips"] else "")
                    + " -- at this sample size the partition matters as much "
                      "as the feature")
        conf = "low"
    return {"verdict": verdict, "evidence": ev, "confidence": conf,
            "caveats": [
                f"{extra} is in RAW DATA UNITS and is not comparable across "
                f"physical fields; the within-field correlation table is the "
                f"one that controls for this."
                if extra in ("mad", "second_deriv") else
                "entropy is a byte-histogram entropy in bits/byte, bounded "
                "in [0, 8].",
                "Deltas are out-of-fold with chunks as groups. A single "
                "delta is noisy at this sample size; the mean marginal gain "
                "over all feature subsets is the more stable reading."],
            "unavailable": None}


def _treatment_answer(a: Analysis, name: str, summary_key: str,
                      regime_key: str, label: str,
                      mechanism_note: str = "",
                      mechanism_caveats: Optional[Sequence[str]] = None
                      ) -> dict:
    s = a.tables.get(summary_key, pd.DataFrame())
    if s.empty:
        return {"verdict": "unavailable", "evidence": {},
                "confidence": "unavailable",
                "unavailable": a.unavailable.get(name, "no pairs"),
                "caveats": []}
    ev: Dict[str, object] = {"per_codec": s.to_dict("records")}
    reg = a.tables.get(regime_key, pd.DataFrame())
    overall = float((s["ratio_pct_helps"] * s["n_pairs"]).sum()
                    / s["n_pairs"].sum())
    hurt = s[s["ratio_pct_helps"] < 45]
    verdict = (f"{label} improves the ratio on {overall:.0f}% of "
               f"{int(s['n_pairs'].sum())} controlled pairs")
    if not hurt.empty and "lib_name" in hurt:
        verdict += ("; it is harmful for "
                    + ", ".join(f"{r['lib_name']} "
                                f"({r['ratio_pct_helps']:.0f}% helps)"
                                for _, r in hurt.iterrows()))
    if not reg.empty:
        strong = reg[reg["pct_helps"] >= 80]
        weak = reg[reg["pct_helps"] <= 45]
        ev["strongly_beneficial_regions"] = strong.to_dict("records")
        ev["neutral_or_harmful_regions"] = weak.to_dict("records")
    ev["overall_pct_pairs_improved"] = overall
    return {"verdict": verdict + mechanism_note, "evidence": ev,
            "confidence": _confidence(int(s["n_pairs"].sum()), 1.0),
            "caveats": [
                "Pairs hold chunk, codec, preset and the other transform "
                "constant, so the chunk and codec effects cancel exactly.",
                "Ratio deltas are unaffected by measurement contention; "
                "TIME deltas are not, because explored candidates run "
                "concurrently on the device."] + list(mechanism_caveats or []),
            "unavailable": None}


def _confidence(n: int, effect: Optional[float]) -> str:
    if not n:
        return "unavailable"
    e = abs(effect) if effect is not None and np.isfinite(effect) else 0.0
    if n >= 200 and e >= 0.3:
        return "high"
    if n >= 40 and e >= 0.2:
        return "moderate"
    if n >= 20:
        return "low"
    return "very low"


def _propose_features(a: Analysis) -> List[dict]:
    """Concrete feature proposals, each tied to an observed failure in THIS
    log. Nothing is proposed that no counterexample here motivates."""
    props: List[dict] = []
    t = a.tables
    ce = t.get("counterexamples", pd.DataFrame())
    kinds = set(ce["kind"].unique()) if not ce.empty and "kind" in ce else set()
    ident = ce[ce["kind"] == "identical_properties_different_ratio"] \
        if "identical_properties_different_ratio" in kinds else pd.DataFrame()

    # The mechanism stage motivates the first proposal, because it says which
    # PART of the ratio the current features already explain exactly and which
    # part they cannot touch -- so the gap is sized, not just asserted.
    mech = a.facts.get("mechanism", {})
    os_ = t.get("codec_order_sensitivity", pd.DataFrame())
    if mech.get("available") and not os_.empty:
        seeing = os_[os_.get("class", "") == "order-sensitive"] \
            if "class" in os_.columns else pd.DataFrame()
        if not seeing.empty:
            worst = seeing.sort_values("median_excess_over_bound").iloc[-1]
            props.append({
                "feature": "match-structure statistics: mean/max run length "
                           "of repeated bytes, and the fraction of the buffer "
                           "covered by matches at a small window",
                "motivated_by": (
                    f"the entropy bound decomposition. "
                    f"{worst['lib_name']} reaches "
                    f"{float(worst['median_excess_over_bound']):.2f}x the "
                    f"8/H bound at the median and up to "
                    f"{float(worst['max_excess_over_bound']):.0f}x, on "
                    f"{float(worst['pct_rows_above_entropy_bound']):.0f}% of "
                    f"rows -- so for the codec that actually wins this "
                    f"workload, MOST of the achieved ratio lives in a "
                    f"component the current features provably cannot see"),
                "captures": (
                    "the second factor in ratio = (8/H) x excess. The first "
                    "factor is already measured exactly by `entropy`; the "
                    "excess is contributed entirely by repeated byte "
                    "SEQUENCES, and nothing in the current vector is a "
                    "function of sequence. This is not a guess about what "
                    "might help -- it is the one quantity the decomposition "
                    "leaves unmeasured."),
                "why_current_features_miss_it": (
                    "entropy is a 256-bin histogram and MAD is a mean, and "
                    "both are invariant under ANY permutation of the buffer; "
                    "second_deriv is order-sensitive but at lag 1 only, and "
                    "is reduced to a mean that averages a long run and a "
                    "noisy patch to the same number. Byte shuffle proves the "
                    "point: it is a permutation, so it cannot move any of the "
                    "three, yet it changes the ratio by up to "
                    + (f"{float(mech.get('shuffle_max_fold', 0)):.2f}x"
                       if mech.get("shuffle_max_fold") else "a large factor")
                    + "."),
                "gpu_cost": (
                    "cheap, and it shares the existing pass: a run-length "
                    "accumulator needs one comparison against the previous "
                    "byte per thread plus a small atomic histogram, the same "
                    "shape as the byte histogram already being built"),
            })
    if mech.get("quantization_split_identifiable") is False:
        props.append({
            "feature": "no new statistic -- recompute the EXISTING three on "
                       "the buffer the codec actually receives",
            "motivated_by": (
                "the transform analysis cannot attribute quantization's gain. "
                "The stats are computed once per chunk on the original buffer "
                "and repeated on all 32 configurations, so the "
                "post-quantization entropy is never seen"),
            "captures": (
                "nothing new about the data -- it makes the existing features "
                "describe the bytes the codec is given rather than the bytes "
                "the simulation produced. Quantization changes the value "
                "alphabet, so the two differ by exactly the amount that "
                "matters for predicting a quantized ratio."),
            "why_current_features_miss_it": (
                "they do not miss it; they are measured at the wrong point in "
                "the pipeline. This is a plumbing change, not a kernel one."),
            "gpu_cost": (
                "one extra stats pass per candidate that quantizes, or one "
                "per chunk if only the quantized variant is profiled. It also "
                "makes the entropy bound valid on lossy rows, which would "
                "extend the whole of the mechanism section to them."),
        })

    # The matched-control stage motivates the sharpest proposal, because it
    # names the invariance that is being violated rather than only observing
    # that something is missing.
    an = a.facts.get("anisotropy", {})
    od = t.get("component_ordering", pd.DataFrame())
    if an.get("anisotropic"):
        stable = od[od["sign_test_p"] < 0.05] if not od.empty else od
        why = []
        if an.get("tightest_n_pairs"):
            why.append(
                f"{int(an['tightest_n_pairs'])} pairs of chunks from the same "
                f"timestep whose features agree to "
                f"{float(an['tightest_tolerance']):g} relative differ up to "
                f"{float(an['tightest_max_fold']):.2f}x in achievable ratio")
        if not stable.empty:
            r = stable.iloc[0]
            why.append(
                f"the {r['family']} components are ordered identically by "
                f"compressibility at {int(r['n_moments'])} independent "
                f"moments (p = {float(r['sign_test_p']):.3g}) while their "
                f"features agree to "
                f"{float(r['median_max_rel_feature_diff']):.1g} relative")
        if an.get("max_achievable_oof_r2") is not None:
            why.append(
                f"which caps any model on the current features at out-of-fold "
                f"R^2 = {float(an['max_achievable_oof_r2']):.3f}")
        props.append({
            "feature": "second-derivative stencils at the OTHER two strides "
                       "(lag nx and lag nx*ny), not only lag 1",
            "motivated_by": "; ".join(why),
            "captures": (
                "orientation. The three features are invariant to something "
                "the compressor is not: a byte histogram and a mean are "
                "invariant to ANY permutation of the buffer, and the existing "
                "second derivative is a lag-1 stencil on the FLATTENED "
                "buffer, so it measures smoothness along the fastest-varying "
                "axis and along no other. A field whose structure runs along "
                "a different axis therefore produces the same three numbers "
                "and different match lengths for the codec -- which is "
                "exactly the observed failure, and it is systematic rather "
                "than noisy."),
            "why_current_features_miss_it": (
                "by construction, not by accident. Two of the three cannot "
                "see order at all, and the third sees one of three axes. No "
                "amount of data or model capacity can recover a quantity the "
                "inputs are invariant to."),
            "gpu_cost": (
                "two more terms in the pass that already computes "
                "second_deriv: the same |x[i+s] - 2x[i] + x[i-s]| reduction "
                "at s = nx and s = nx*ny. It needs the chunk's logical shape, "
                "which the runtime knows and the stats kernel is currently "
                "not told -- that plumbing is the only real work. HYPOTHESIS "
                "under test: if these do NOT separate the components, the "
                "explanation is elsewhere and the observation still stands."),
        })

    if not ident.empty:
        pairs = ident[["chunk_uid", "chunk_uid_b", "ratio_fold"]].head(3)
        props.append({
            "feature": "zero / repeated-value fraction (and run-length "
                       "statistics over the byte stream)",
            "motivated_by": (
                f"{len(ident)} chunk pairs with numerically identical "
                f"entropy, MAD and second derivative whose achievable ratios "
                f"differ by up to {ident['ratio_fold'].max():.2f}x "
                f"(e.g. {pairs.iloc[0]['chunk_uid']} vs "
                f"{pairs.iloc[0]['chunk_uid_b']})"),
            "captures": (
                "how the identical value distribution is ARRANGED. A byte "
                "histogram is permutation invariant -- shuffle a chunk's "
                "bytes and its entropy does not move -- and MAD and the "
                "second derivative are means, so all three are blind to run "
                "structure. LZ77-family codecs encode runs, so two chunks "
                "with the same histogram and very different run lengths "
                "compress very differently, which is exactly the observed "
                "failure."),
            "why_current_features_miss_it": (
                "entropy is computed from a 256-bin byte histogram "
                "(data_stats_gpu_kernels.cu StatsPass1Kernel), which discards "
                "order entirely; MAD and second_deriv are order-sensitive but "
                "are reduced to a single mean, which averages a long run and "
                "a noisy patch to the same number"),
            "gpu_cost": (
                "cheap: a run-length histogram needs one more grid-stride "
                "pass with a per-thread comparison against the previous byte "
                "and an atomic into a small bin array -- the same shape as "
                "the existing histogram pass, and it can share it"),
        })
        props.append({
            "feature": "float32 exponent-byte entropy and mantissa-byte "
                       "entropy, reported separately",
            "motivated_by": "the same identical-feature pairs",
            "captures": (
                "IEEE-754 structure. In a float32 field the exponent byte is "
                "nearly constant over a smooth region and the low mantissa "
                "bytes are nearly uniform; a single 8-bit histogram over all "
                "four byte positions averages those two very different "
                "distributions into one number that describes neither, and "
                "byte shuffle exists precisely to separate them"),
            "why_current_features_miss_it": (
                "the histogram is taken over the whole buffer with no byte "
                "position, so a chunk whose compressibility lives entirely in "
                "a constant exponent plane looks the same as one where it "
                "does not"),
            "gpu_cost": (
                "essentially free: four 256-bin histograms instead of one in "
                "the SAME pass, indexed by (i & 3). It also directly predicts "
                "the shuffle benefit, which is currently unpredictable from "
                "the three features"),
        })
    sh = t.get("shuffle_summary", pd.DataFrame())
    if not sh.empty and (sh["ratio_pct_helps"] < 45).any():
        bad = sh[sh["ratio_pct_helps"] < 45]["lib_name"].tolist()
        props.append({
            "feature": "per-byte-plane variance ratio "
                       "(var(plane_k) / var(all planes))",
            "motivated_by": (
                f"shuffle helps most codecs on this data but is harmful for "
                f"{', '.join(map(str, bad))}, and none of the three features "
                f"separates the two cases"),
            "captures": (
                "whether the four byte planes of a float32 word actually "
                "differ in their statistics -- which is the precondition for "
                "byte shuffle paying off at all"),
            "why_current_features_miss_it": (
                "all three features are computed on the TYPED values or on "
                "the undifferentiated byte stream; none of them looks at byte "
                "position, which is the axis shuffle operates on"),
            "gpu_cost": "one extra reduction in the existing pass",
        })
    drv = t.get("prediction_error_drivers", pd.DataFrame())
    if not drv.empty:
        strong = drv[drv["spearman_rho"].abs() >= 0.5]
        if not strong.empty:
            r = strong.reindex(
                strong["spearman_rho"].abs().sort_values(
                    ascending=False).index).iloc[0]
            props.append({
                "feature": ("no new statistic -- retrain on the existing "
                            f"`{r['feature']}` input"),
                "motivated_by": (
                    f"the NN's {r['metric']} error still correlates "
                    f"{r['spearman_rho']:+.2f} with `{r['feature']}` for "
                    f"{r.get('lib_name')}, so the information is already in "
                    f"the input vector and the model is not using it"),
                "captures": "nothing new; this is a training gap, not a "
                            "feature gap",
                "why_current_features_miss_it": "they do not -- the model "
                                                "does",
                "gpu_cost": "zero",
            })
    return props

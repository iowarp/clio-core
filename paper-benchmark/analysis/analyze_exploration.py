#!/usr/bin/env python3
"""Analyse one or more Clio-NeuroPress exploration logs.

    ./analyze_exploration.py explore.csv
    ./analyze_exploration.py --input explore.csv --output analysis_results/
    ./analyze_exploration.py warpx.csv vpic.csv nyx.csv lammps.csv

Each input is analysed independently into its own directory; with more than
one, a cross-workload comparison is written beside them. The default command
runs the complete pipeline -- the flags below only turn work OFF.

What the tool assumes about its input, and what it does when the assumption
fails:

  * The five columns it cannot work without are seq, blob, chunk_bytes, role,
    rank, plus the codec/config/outcome columns. A missing one is a hard error
    naming the column.
  * entropy/mad/second_deriv are optional in the log. Older sweeps predate
    them; when they are absent the tool joins them from a `selection.csv`
    found beside the input, and says so in the report. With neither, every
    data-property analysis is reported as unavailable rather than as a null
    result.
  * Everything else -- measured decompression time, measured quality, several
    error bounds, several timesteps, several workloads -- is optional, and
    each analysis that needs one says why it could not run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import traceback
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import crossworkload, pipeline, reporting          # noqa: E402
from analysis.loader import load_exploration, provenance         # noqa: E402

#: Where each produced table lands, relative to the workload's directory.
#: Names follow the requested layout; extras land beside their nearest kin.
LAYOUT: Dict[str, str] = {
    # summary/
    "codec_summary": "summary/codec_summary.csv",
    "correlations_per_config": "summary/feature_correlations.csv",
    "correlations_within_field": "summary/feature_correlations_within_field.csv",
    "correlations_chunk_level": "summary/feature_correlations_chunk_level.csv",
    "property_correlations": "summary/property_correlations.csv",
    "property_outcome_frame": "processed/property_outcome_frame.csv",
    "feature_ablation": "summary/feature_ablation.csv",
    "feature_ablation_marginal": "summary/feature_ablation_marginal.csv",
    "feature_importance": "summary/feature_importance.csv",
    "ablation_stability": "summary/ablation_stability.csv",
    "winners": "summary/winners.csv",
    "codec_win_rates": "summary/codec_win_rates.csv",
    "codec_feature_sensitivity": "summary/codec_feature_sensitivity.csv",
    "codec_property_profile": "summary/codec_property_profile.csv",
    "binned_trends": "summary/binned_trends.csv",
    "trend_monotonicity": "summary/trend_monotonicity.csv",
    "joint_regimes": "summary/joint_regimes.csv",
    "conditional_gains": "summary/conditional_gains.csv",
    "confounder_power": "confounds/confounder_power.csv",
    # mechanism/ -- WHY, not how much; see analysis/mechanism.py
    "entropy_bound": "mechanism/entropy_bound.csv",
    "quantization_levels_per_chunk":
        "mechanism/quantization_levels_per_chunk.csv",
    "quantization_regimes": "mechanism/quantization_regimes.csv",
    "quantization_levels_by_field":
        "mechanism/quantization_levels_by_field.csv",
    "quantization_levels_drive_outcomes":
        "mechanism/quantization_levels_drive_outcomes.csv",
    "codec_order_sensitivity": "mechanism/codec_order_sensitivity.csv",
    "shuffle_decomposition": "mechanism/shuffle_decomposition.csv",
    "locality_probe": "mechanism/locality_probe.csv",
    "timing_mediation": "mechanism/timing_mediation.csv",
    "codec_throughput": "mechanism/codec_throughput.csv",
    # anisotropy/ -- the matched-control stage, see analysis/anisotropy.py
    "matched_pairs": "anisotropy/matched_pairs.csv",
    "feature_blind_spread": "anisotropy/feature_blind_spread.csv",
    "explainable_ceiling": "anisotropy/explainable_ceiling.csv",
    "component_families": "anisotropy/component_families.csv",
    "component_ordering": "anisotropy/component_ordering.csv",
    "residual_by_field": "anisotropy/residual_by_field.csv",
    "partial_correlations": "confounds/partial_correlations.csv",
    "feature_confounder_collinearity":
        "confounds/feature_confounder_collinearity.csv",
    "prediction_clamps": "prediction/prediction_clamps.csv",
    # paired/
    "shuffle_comparison": "paired/shuffle_comparison.csv",
    "quantization_comparison": "paired/quantization_comparison.csv",
    "error_bound_comparison": "paired/error_bound_comparison.csv",
    "shuffle_summary": "paired/shuffle_summary.csv",
    "quantization_summary": "paired/quantization_summary.csv",
    "error_bound_summary": "paired/error_bound_summary.csv",
    "shuffle_benefit_regimes": "paired/shuffle_benefit_regimes.csv",
    "quantization_benefit_regimes": "paired/quantization_benefit_regimes.csv",
    # cost_model/
    "selection_results": "cost_model/selection_results.csv",
    "selection_regret": "cost_model/selection_regret.csv",
    # prediction/
    "ratio_prediction_errors": "prediction/ratio_prediction_errors.csv",
    "ct_prediction_errors": "prediction/ct_prediction_errors.csv",
    "dt_prediction_errors": "prediction/dt_prediction_errors.csv",
    "prediction_summary": "prediction/prediction_summary.csv",
    "prediction_by_codec": "prediction/prediction_by_codec.csv",
    "prediction_error_drivers": "prediction/prediction_error_drivers.csv",
    "ranking_quality": "prediction/ranking_quality.csv",
    # models/
    "model_metrics": "models/model_metrics.csv",
    "codec_classifier_metrics": "models/codec_classifier_metrics.csv",
    "generalisation": "models/generalisation.csv",
    # temporal
    "property_series": "temporal/property_series.csv",
    "outcome_series": "temporal/outcome_series.csv",
    "temporal_series": "temporal/temporal_series.csv",
    "temporal_trends": "temporal/temporal_trends.csv",
    "codec_switches": "temporal/codec_switches.csv",
    "evolution_joined": "temporal/evolution_joined.csv",
    "evolution_effects": "temporal/evolution_effects.csv",
    # counterexamples/ and processed/
    "counterexamples": "counterexamples/counterexamples.csv",
    "chunk_properties": "processed/chunk_properties.csv",
}


def _write_tables(a: pipeline.Analysis, outdir: str) -> List[str]:
    written: List[str] = []
    for name, rel in LAYOUT.items():
        df = a.tables.get(name)
        if df is None or len(df) == 0:
            continue
        path = os.path.join(outdir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        written.append(rel)
    # The full row-level frame, with everything derived attached, so a reader
    # can re-run any of this by hand from one file.
    p = os.path.join(outdir, "processed/exploration_results.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    a.ds.rows.to_csv(p, index=False)
    written.append("processed/exploration_results.csv")

    # A one-row dataset summary, which the requested layout asks for by name.
    c = a.ds.audit.counts
    ds_summary = pd.DataFrame([{
        "workload": a.ds.name, "path": a.ds.path,
        "rows": c.get("rows"), "unique_chunks": a.ds.n_chunks,
        "n_codecs": len(c.get("codecs") or []),
        "codecs": "|".join(map(str, c.get("codecs") or [])),
        "n_fields": len(c.get("fields") or []),
        "fields": "|".join(map(str, c.get("fields") or [])),
        "n_timesteps": len(c.get("timesteps") or []),
        "shuffle_widths": "|".join(map(str, c.get("shuffle_widths") or [])),
        "quantize_modes": "|".join(map(str, c.get("quantize_modes") or [])),
        "error_bounds_lossy": "|".join(
            map(str, c.get("error_bounds_lossy") or [])),
        "usable_features": "|".join(a.ds.features),
        "n_warnings": len(a.ds.audit.warnings),
        "n_unavailable_analyses": len(a.unavailable),
    }])
    q = os.path.join(outdir, "summary/dataset_summary.csv")
    os.makedirs(os.path.dirname(q), exist_ok=True)
    ds_summary.to_csv(q, index=False)
    written.append("summary/dataset_summary.csv")

    import json
    from analysis.reporting import _jsonable
    r = os.path.join(outdir, "models/discovered_thresholds.json")
    os.makedirs(os.path.dirname(r), exist_ok=True)
    with open(r, "w") as fh:
        json.dump(_jsonable(a.thresholds), fh, indent=1)
    written.append("models/discovered_thresholds.json")
    return written


def analyse_one(path: str, outroot: str, seed: int, figures: bool,
                argv: List[str], name: Optional[str] = None
                ) -> Optional[pipeline.Analysis]:
    label = name or os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join(outroot, label)
    os.makedirs(outdir, exist_ok=True)
    print(f"== {label}: reading {path}", flush=True)
    ds = load_exploration(path, label)
    print(f"   {len(ds.rows):,} rows, {ds.n_chunks} chunks, "
          f"features: {ds.features or 'NONE'}", flush=True)
    for wmsg in ds.audit.warnings:
        print(f"   ! {wmsg}", flush=True)

    a = pipeline.run(ds, outdir, seed=seed, make_figures=figures)
    prov = provenance([path], argv, seed)
    prov["timestamp"] = _dt.datetime.now().astimezone().isoformat(
        timespec="seconds")
    written = _write_tables(a, outdir)
    reporting.write_summary(a, outdir, prov)
    reporting.write_report(a, outdir, prov)
    print(f"   wrote {len(written)} tables, "
          f"{len(a.figures.made) if a.figures else 0} figures, "
          f"REPORT.md and summary.json -> {outdir}", flush=True)
    if a.unavailable:
        print(f"   {len(a.unavailable)} analysis stage(s) unavailable: "
              f"{', '.join(sorted(a.unavailable))}", flush=True)
    return a


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="*", help="exploration CSV(s) to analyse")
    p.add_argument("--input", "-i", action="append", default=[],
                   help="an exploration CSV (repeatable; same as positional)")
    p.add_argument("--output", "-o", default="analysis_results",
                   help="output directory (default: analysis_results/)")
    p.add_argument("--name", action="append", default=[],
                   help="label for the corresponding input, in order "
                        "(default: the file stem)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for every model (default: 0)")
    p.add_argument("--no-figures", action="store_true",
                   help="skip figure generation")
    p.add_argument("--no-cross-workload", action="store_true",
                   help="skip the cross-workload comparison")
    a = p.parse_args(argv)

    inputs = list(a.csv) + list(a.input)
    if not inputs:
        p.error("no input CSV given")
    missing = [x for x in inputs if not os.path.isfile(x)]
    if missing:
        p.error(f"input not found: {', '.join(missing)}")
    os.makedirs(a.output, exist_ok=True)

    analyses: List[pipeline.Analysis] = []
    failures: List[str] = []
    for i, path in enumerate(inputs):
        label = a.name[i] if i < len(a.name) else None
        try:
            res = analyse_one(path, a.output, a.seed, not a.no_figures,
                              ["analyze_exploration.py"] + argv, label)
            if res is not None:
                analyses.append(res)
        except Exception as e:                       # noqa: BLE001
            # One unreadable input must not lose the analyses of the others.
            failures.append(f"{path}: {e}")
            print(f"   FAILED: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if len(analyses) > 1 and not a.no_cross_workload:
        out = os.path.join(a.output, "cross_workload")
        print(f"== cross-workload comparison over "
              f"{len(analyses)} workloads", flush=True)
        crossworkload.run(analyses, out, seed=a.seed)
        print(f"   wrote {out}/REPORT.md and summary.json", flush=True)
    elif len(analyses) == 1 and len(inputs) > 1:
        print("== cross-workload comparison skipped: only one input "
              "analysed successfully", flush=True)

    if failures:
        print(f"\n{len(failures)} input(s) failed:", file=sys.stderr)
        for fmsg in failures:
            print(f"  - {fmsg}", file=sys.stderr)
        return 1
    print(f"\nDone. {len(analyses)} workload(s) -> {a.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

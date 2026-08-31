#!/usr/bin/env python3
"""REPORT.md and summary.json.

Two rules the whole renderer follows:

  * Nothing is claimed without its sample count beside it. Every headline
    sentence carries n, and every table prints support.
  * An analysis that could not run prints WHY it could not run, in the place
    where its result would have gone. "Shuffle helps 78% of the time" and
    "shuffle could not be measured" must never be indistinguishable, and
    neither may be silently absent.

Observation, correlation, hypothesis and mechanism are labelled explicitly
wherever they appear, because the interesting conclusions in this analysis sit
in different tiers and the reader cannot tell them apart from the numbers.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .pipeline import Analysis

Q_TEXT = {
    "Q1": "How strongly does entropy determine compressibility?",
    "Q2": "Does MAD provide useful information beyond entropy?",
    "Q3": "Does the second derivative provide useful information beyond "
          "entropy and MAD?",
    "Q4": "Can these features predict compression ratio?",
    "Q5": "Can they predict the winning codec?",
    "Q6": "What data-property regimes favour each codec?",
    "Q7": "What determines whether shuffle helps?",
    "Q8": "What determines whether quantization helps?",
    "Q9": "How does the error bound interact with intrinsic data properties?",
    "Q10": "How does compressibility change as the simulation evolves?",
    "Q11": "Do these relationships generalise across workloads?",
    "Q12": "When does the NeuroPress predictor fail?",
    "Q13": "When does the cost model select the wrong configuration?",
    "Q14": "Are entropy, MAD and second derivative sufficient?",
    "Q15": "What additional inexpensive statistics should we collect?",
    "Q16": "Why does a quantized chunk reach the ratio it reaches?",
    "Q17": "Why does quality collapse where it collapses?",
}


#: The report's running order. Section numbers and every cross-reference are
#: DERIVED from this list, so reordering the report is a one-line edit here and
#: no number in the prose can fall out of step with the heading it names.
#: Ordered by what answers the question the analysis is for -- what the data's
#: properties do to the ratio and the timings, and why -- with the machinery
#: needed to read those answers, but not itself an answer, kept behind them.
SECTIONS: List[str] = [
    "Dataset overview",
    "Is it the data, or the clock?",
    "Why: the ratio against its entropy bound",
    "Same moment, different field",
    "Data-property findings",
    "Codec behaviour",
    "Shuffle behaviour",
    "Quantization behaviour",
    "Error-bound behaviour",
    "Temporal behaviour",
    "Prediction accuracy",
    "Counterexamples",
    "Missing features",
    "The fifteen questions, answered",
    "Figures",
    "Appendix A: what the cost column actually is",
    "Appendix B: cost-model performance",
    "Appendix C: reproducibility",
]
SECTION_NO = {t: i + 1 for i, t in enumerate(SECTIONS)}


def section_ref(title: str) -> str:
    """`Section N` for a title in SECTIONS, so prose cannot name a stale one."""
    return f"Section {SECTION_NO[title]}"


#: Display names for the paired-analysis columns. The internal names stay as
#: they are -- pipeline.py, crossworkload.py and counterexamples.py select on
#: them -- and only the RENDERED header changes, so a reader never has to know
#: that "treatment/control" is A/B vocabulary or that "delta" means a
#: subtraction. Applied by _md_table, so every table in every report and the
#: cross-workload section pick it up from one place.
#:
#: A paired row holds (chunk, codec, preset, quantize, error bound) fixed and
#: varies ONE knob, so every quantity below is "same chunk, setting on, minus
#: same chunk, setting off".
FRIENDLY_COLS = {
    "treatment": "setting changed",
    "control_value": "value when off",
    "treat_value": "value when on",
    "n_pairs": "pairs compared",
    "n_chunks": "chunks",
}
#: Per-outcome suffixes, expanded for each measured column (ratio, ct_ms, ...).
FRIENDLY_SUFFIX = {
    "_median_delta": " change (median)",
    "_mean_delta": " change (mean)",
    "_median_rel": " change % (median)",
    "_pct_helps": " % of pairs improved",
    "_pct_chunks_helped": " % of chunks improved",
    "_p10_delta": " change (10th pct)",
    "_p90_delta": " change (90th pct)",
    "_sign_test_k_chunks": " chunks improved (k)",
    "_sign_test_n_chunks": " chunks tested (n)",
    "_sign_test_p": " sign-test p",
    "_n": " n",
}


def friendly(col: str) -> str:
    """Header text for one column. Unknown columns pass through unchanged."""
    if col in FRIENDLY_COLS:
        return FRIENDLY_COLS[col]
    for suf, nice in FRIENDLY_SUFFIX.items():
        if col.endswith(suf):
            return col[:-len(suf)] + nice
    return col


def _md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None,
              n: int = 20, floatfmt: str = "{:.4g}",
              drop_empty: bool = False) -> str:
    """Render a frame as a Markdown table.

    `drop_empty` removes columns that are entirely null in the SHOWN rows. It
    is off by default -- an all-null column is usually itself a finding -- and
    on for the counterexample sections, where the column list is shared by
    finders that fill different subsets of it.
    """
    if df is None or df.empty:
        return "_(no rows)_\n"
    d = df[[c for c in (cols or df.columns) if c in df.columns]].head(n)
    if drop_empty:
        keep = [c for c in d.columns if d[c].notna().any()]
        d = d[keep] if keep else d
    if d.empty or not len(d.columns):
        return "_(no rows)_\n"

    def cell(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else floatfmt.format(v)
        s = str(v)
        return s.replace("|", "\\|")[:120]

    head = "| " + " | ".join(friendly(str(c)) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join("| " + " | ".join(cell(v) for v in row) + " |"
                     for row in d.itertuples(index=False))
    more = (f"\n\n_({len(df)} rows total; {n} shown -- the full table is in "
            f"the CSV beside this report.)_\n" if len(df) > n else "\n")
    return head + "\n" + rule + "\n" + body + more


def _unavailable(a: Analysis, key: str) -> str:
    why = a.unavailable.get(key)
    return (f"> **Not available in this run.** {why}\n" if why else "")


def write_report(a: Analysis, outdir: str, prov: dict) -> str:
    ds, t, f = a.ds, a.tables, a.facts
    L: List[str] = []
    w = L.append

    def sec(title: str) -> None:
        """Emit a heading numbered from SECTIONS, never from a literal."""
        w(f"\n## {SECTION_NO[title]}. {title}\n")

    w(f"# Compressibility analysis: `{ds.name}`\n")
    w(f"Generated by `analyze_exploration.py` from `{os.path.basename(ds.path)}` "
      f"at {prov['timestamp']}.\n")
    w("This report reads a Clio-NeuroPress exploration log and asks what "
      "makes a scientific-data chunk compress the way it does. Every column "
      "it interprets was checked against the code that writes it "
      "(`neuropress_telemetry.cc`, `compressor_runtime.cc`, "
      "`neuropress_cost.h`, `data_stats_gpu_kernels.cu`); the definitions "
      "that matter are restated where they change a conclusion.\n")

    # ---------------------------------------------------------------- audit
    sec("Dataset overview")
    c = ds.audit.counts
    w(f"- **{c.get('rows', 0):,} rows** describing **{ds.n_chunks} unique "
      f"chunks** -- a row is a *(chunk, configuration)* measurement, not an "
      f"independent data sample. Each chunk was measured under "
      f"{int(ds.chunks['n_candidates'].median())} configurations "
      f"(median), so any statistic computed over rows has an effective "
      f"sample size of {ds.n_chunks}, not {c.get('rows', 0):,}.")
    w(f"- **Chunk size:** {', '.join(f'{int(b):,} B' for b in c.get('chunk_bytes', []))}")
    w(f"- **Codecs ({len(c.get('codecs') or [])}):** "
      f"{', '.join(map(str, c.get('codecs') or []))}")
    w(f"- **Presets:** {c.get('presets')} · **quantize:** "
      f"{c.get('quantize_modes')} · **shuffle widths:** "
      f"{c.get('shuffle_widths')}")
    eb = c.get("error_bounds_lossy")
    w(f"- **Error bounds (lossy rows):** "
      f"{eb if eb else 'not recorded in this log'}")
    w(f"- **Physical fields ({len(c.get('fields') or [])}):** "
      f"{', '.join(map(str, c.get('fields') or [])) or 'none parsed'}")
    ts = c.get("timesteps") or []
    w(f"- **Timesteps ({len(ts)}):** "
      f"{f'{min(ts)}..{max(ts)}' if ts else 'none parsed'}")
    w(f"- **Usable data-property features:** "
      f"{', '.join(ds.features) if ds.features else 'NONE'}\n")

    coh = f.get("cohort", {})
    if coh.get("measurable") and coh.get("chunks_not_explored"):
        w(f"\n> :rotating_light: **This is a CONDITIONED sample, not a random "
          f"one.** The selection log beside this sweep lists "
          f"{coh['chunks_in_selection_log']} chunks; only "
          f"{coh['chunks_explored']} of them ({coh['pct_explored']:.0f}%) "
          f"were explored. Exploration is trigger-gated upstream, so the "
          f"chunks in this file are the ones that fired the trigger.\n")
        if coh.get("inferred_trigger"):
            w(f"> {coh['inferred_trigger']}\n")
        gd = coh.get("group_differences") or {}
        if gd:
            w("> How the omitted chunks differ (medians):\n>")
            w(_md_table(pd.DataFrame([
                {"quantity": k,
                 "explored": v["explored_median"],
                 "NOT explored": v["not_explored_median"]}
                for k, v in gd.items()])))
        w("\nEvery prediction-error, over/under-prediction and cost-model "
          "regret figure below therefore describes **the explored subset**, "
          "and does not generalise to the whole run.\n")
    elif coh.get("measurable"):
        w("\nEvery chunk the run wrote was explored, so this sweep is a "
          "complete sample of it.\n")

    if ds.sidecars:
        w("**Sidecar files found beside the log:** "
          + ", ".join(f"`{os.path.basename(v)}` ({k})"
                      for k, v in ds.sidecars.items()) + "\n")

    if ds.audit.notes:
        w("### Data-handling decisions\n")
        for n in ds.audit.notes:
            w(f"- {n}")
        w("")
    if ds.audit.warnings:
        w("### Warnings\n")
        for x in ds.audit.warnings:
            w(f"- :warning: {x}")
        w("")
    if ds.audit.filters:
        w("### Rows filtered\n")
        w(_md_table(pd.DataFrame(ds.audit.filters)))
    if ds.audit.outliers:
        w("### Robust outlier scan (reported, never removed)\n")
        w(_md_table(pd.DataFrame(ds.audit.outliers)))

    # ------------------------------------------------------ the cost model
    sec("Is it the data, or the clock?")
    conf = f.get("confounds", {})
    if "confounds" in a.unavailable:
        w(_unavailable(a, "confounds"))
    elif not conf:
        w("_Not computed._\n")
    else:
        w("On a single run of an evolving simulation the intrinsic data "
          "properties are largely a **function of the timestep**, and "
          "compressibility falls over the same run. Any correlation between "
          "them is then shared drift, not evidence of a mechanism. This "
          "section tests that before "
          + section_ref("Data-property findings")
          + " is allowed to be read.\n")
        if conf.get("confounded"):
            w("> :rotating_light: **The data-property findings in this log "
              "are confounded.** Specifically:\n>")
            for r in conf.get("reasons", []):
                w(f"> - {r}")
            w("\n> What this does and does not mean: it does **not** show "
              "that entropy is irrelevant to compression. It shows that "
              "*this log* cannot separate a data-property effect from the "
              "simulation clock, because the two move together. Separating "
              "them needs a sweep over many chunks at **fixed simulation "
              "time**, or several runs whose evolution differs.\n")
        else:
            w("No confounder tested here reproduces the data-property fit, "
              "no correlation reverses sign under control, and the "
              "cross-validation survives regrouping. The "
              + section_ref("Data-property findings") + " findings "
              "are not explained away by the metadata.\n")
        pw = t.get("confounder_power", pd.DataFrame())
        if not pw.empty:
            w("\n### Give a model nothing but the metadata\n")
            w("Out-of-fold R^2 on log10(best achievable ratio). A metadata "
              "column is **not a data property** -- if one of them alone "
              "matches the three features, the features have not been shown "
              "to carry information the metadata does not.\n")
            w(_md_table(pw.sort_values("oof_r2", ascending=False),
                        ["predictor", "kind", "oof_r2",
                         "oof_r2_with_features_too",
                         "features_gain_over_metadata_alone", "n_chunks",
                         "verdict"], n=10))
        pc = t.get("partial_correlations", pd.DataFrame())
        if not pc.empty:
            w("\n### Correlation before and after removing the confounder\n")
            w("Both variables residualised on the confounder (and its "
              "square). A **sign flip** means the raw correlation was "
              "carried entirely by the confounder and points the other way "
              "once it is gone.\n")
            w(_md_table(pc, ["feature", "confounder", "n_chunks", "raw_r",
                             "raw_p", "partial_r", "partial_p",
                             "sign_flipped", "attenuation"]))
        cl = t.get("feature_confounder_collinearity", pd.DataFrame())
        if not cl.empty:
            n_col = int(cl["collinear"].sum())
            w("\n### Does stratifying by field control for anything?\n")
            w(f"{n_col} of {len(cl)} (feature, field) pairs are collinear "
              f"(|r| >= 0.95) with the confounder **inside the field**. Where "
              f"they are, 'within field' and 'within one time series' are the "
              f"same thing, and the within-field correlation in "
              f"{section_ref('Data-property findings')} is "
              f"computed *inside* the confound rather than across it.\n")
            w(_md_table(cl.sort_values("pearson_r", key=abs,
                                       ascending=False),
                        ["feature", "stratum", "confounder", "n", "pearson_r",
                         "collinear"], n=12))
        if "oof_r2_grouped_by_confounder" in conf:
            w("\n### Regrouping the cross-validation\n")
            w(f"Grouping folds on `chunk_uid` stops a **chunk** being "
              f"memorised; grouping them on the confounder stops a "
              f"**moment** being memorised. Out-of-fold R^2 falls from "
              f"**{conf['oof_r2_grouped_by_chunk']:.3f}** (by chunk) to "
              f"**{conf['oof_r2_grouped_by_confounder']:.3f}** (by "
              f"confounder). "
              + (f"{conf.get('pct_nn_same_timestep_different_field', 0):.0f}% "
                 f"of chunks have their nearest neighbour in feature space in "
                 f"a *different field at the same timestep*, which is why."
                 if conf.get("pct_nn_same_timestep_different_field")
                 else "") + "\n")

    # --------------------------------------------------------- mechanism
    sec("Why: the ratio against its entropy bound")
    mech = f.get("mechanism", {})
    if "mechanism" in a.unavailable:
        w(_unavailable(a, "mechanism"))
    elif not mech.get("available"):
        w("_Not computed._\n")
    else:
        w("Everything after this section measures **how much** an outcome "
          "moves with a feature. This one asks **through what**, and it can, "
          "because one of the three features has a closed-form consequence "
          "rather than only a correlation.\n")
        w("`entropy` is the Shannon entropy of the 256-bin byte histogram, in "
          "bits/byte. A coder that assigns one codeword per byte symbol and "
          "reads no order cannot beat **ratio <= 8 / H**. That is the source "
          "coding theorem applied to the exact quantity the feature measures "
          "-- not a fit -- so the achieved ratio splits into two parts with "
          "two different causes:\n")
        w("```\n"
          "ratio  =  (8 / H)                x  excess\n"
          "          the byte histogram        everything ORDER contributes\n"
          "          entropy knows this        entropy cannot know it,\n"
          "          exactly                   by construction\n"
          "```\n")
        w("The split is what makes the rest of this report interpretable: a "
          "feature that predicts one part well and the other not at all will "
          "look like a mediocre predictor overall, and the fix is not a "
          "better model.\n")

        os_ = t.get("codec_order_sensitivity", pd.DataFrame())
        if not os_.empty:
            w("\n### Which codecs can read order, measured rather than "
              "assumed\n")
            w("Classified from each codec's own numbers on the lossless, "
              "unshuffled cell -- how often it beats its own entropy bound. "
              "A codec that essentially never does is behaving as an "
              "order-blind coder on this data, whatever its name.\n")
            w(_md_table(os_, ["lib_name", "class",
                              "pct_rows_above_entropy_bound",
                              "median_excess_over_bound",
                              "max_excess_over_bound", "verdict"], n=20))
            w("\n![ratio against the entropy bound]"
              "(figures/entropy_bound_vs_achieved.png)\n")
            w("*Every point above the dashed line is ratio a byte histogram "
              "cannot account for. Lossless, unshuffled rows only.*\n")
            if mech.get("entropy_bound_holds"):
                w(f"\n- **The bound holds where it must.** "
                  f"{mech['entropy_bound_holds']}.")
            if mech.get("order_contribution"):
                w(f"- **And it is beaten where it can be.** "
                  f"{mech['order_contribution']}.")
            w("\nSo `entropy` is not a weak predictor of compressibility. It "
              "is an *exact* predictor of one component and a blind one for "
              "the other, and which codec you ask decides how much of the "
              "answer it gives you. That is the mechanism behind the "
              "per-codec spread in " + section_ref("Codec behaviour")
              + ", and it is why no amount of "
              "model capacity closes the gap.\n")

        bnd = t.get("entropy_bound", pd.DataFrame())
        if not bnd.empty:
            w("\n### The bound and the excess, per codec and configuration\n")
            w("`median_excess_over_bound` is the achieved ratio divided by "
              "8/H. At most 1.0 for an order-blind coder; unbounded above for "
              "one that reads sequences.\n")
            w("![how far past the bound each codec reaches]"
              "(figures/excess_over_entropy_bound.png)\n")
            cols = [c for c in ("lib_name", "quantize", "shuffle", "n_rows",
                                "median_entropy", "median_entropy_bound_ratio",
                                "median_ratio", "median_excess_over_bound",
                                "pct_rows_above_bound") if c in bnd.columns]
            w(_md_table(bnd.sort_values("median_excess_over_bound",
                                        ascending=False), cols, n=24))

        w("\n### Where quantization and shuffle act -- and what is knowable\n")
        if mech.get("stats_measured_pre_transform"):
            w("> :warning: **The stats are measured on the ORIGINAL buffer.** "
              "`entropy`, `mad` and `second_deriv` are logged once per chunk "
              "and repeated unchanged on all of that chunk's configurations, "
              "so they describe the bytes *before* any transform. Two "
              "consequences, and they point in opposite directions:\n>")
            w("> - **Quantization: not identifiable.** It changes the value "
              "alphabet and therefore the byte histogram, but the "
              "post-quantization entropy is never recorded. How much of its "
              "gain is a smaller alphabet and how much is longer runs cannot "
              "be separated from this log, and no split is reported here. "
              "*(An earlier version of this section did report one, and got "
              "\"100% from order\" for every codec -- which was reading the "
              "logging convention rather than the data.)* Recording the stats "
              "of the buffer the codec actually receives would fix it.")
            w("> - **Shuffle: fully identifiable, and for free.** Byte shuffle "
              "*permutes* the byte stream. The 256-bin histogram is a multiset "
              "count, so it -- and H, and 8/H -- are **invariant** under any "
              "permutation. Every bit of shuffle's gain is therefore "
              "contributed by order. That is a property of the transform, not "
              "an inference from these rows.\n")
        sh = t.get("shuffle_decomposition", pd.DataFrame())
        if not sh.empty:
            w("\nShuffle gain at a provably fixed byte histogram, paired "
              "within *(chunk, codec, preset)* on lossless rows only:\n")
            w(_md_table(sh, ["lib_name", "n_pairs", "n_chunks",
                             "median_ratio_fold", "entropy_bound_fold",
                             "pct_of_gain_from_order", "gain_source"], n=20))
        lp = t.get("locality_probe", pd.DataFrame())
        if not lp.empty:
            w("\n### What that reveals about how each codec works\n")
            w("Two independent measurements combine into one statement here, "
              "so it is kept apart from both. A codec that codes the "
              "**global** byte histogram cannot gain from a permutation -- "
              "the histogram is identical on both sides. So a codec that never "
              "beats the global bound *and yet gains from shuffle* is not "
              "coding the global histogram at all: it is coding a **local** "
              "one, and shuffle concentrates like bytes into the same block, "
              "making each block far more skewed than the whole buffer while "
              "leaving the whole buffer's distribution untouched.\n")
            w(_md_table(lp, ["lib_name", "class_from_entropy_bound",
                             "shuffle_ratio_fold", "n_chunks", "inference"],
                        n=20))
            if mech.get("locality_finding"):
                w(f"\n> **{mech['locality_finding']}.** This is an inference "
                  f"from two measurements, not a direct observation of the "
                  f"codec's internals -- but a global-histogram coder has no "
                  f"way to produce this pair of numbers.\n")

        med = t.get("timing_mediation", pd.DataFrame())
        if not med.empty:
            w("\n### The timings are downstream of the ratio\n")
            w("Compression and decompression time both correlate with the "
              "data properties, and it would be easy to report that as "
              "\"rough data is slow\". The test below holds the **output "
              "size** constant instead. If a feature's correlation with time "
              "collapses once `compressed_bytes = chunk_bytes / ratio` is "
              "controlled, then the feature acts on time *through* the ratio "
              "rather than beside it -- the data decides how much output "
              "there is, and the amount of output decides the time.\n")
            w("Rank-based throughout: the relationships are monotone but not "
              "linear, and a Pearson r on raw milliseconds would be set by "
              "the slowest rows.\n")
            w("\n![time against the data properties and against the output "
              "size](figures/timing_follows_output_size.png)\n")
            w("*Each codec is one dumbbell: its strongest data feature "
              "against the compressed size. Where the orange dot sits to the "
              "right, the output size is the better account of the time.*\n")
            w(_md_table(med, ["metric", "lib_name", "feature", "n_rows",
                              "rho_time_vs_feature",
                              "rho_time_vs_compressed_bytes",
                              "partial_rho_time_vs_feature_given_size",
                              "attenuation", "mediated_by_output_size"],
                        n=24))
            if mech.get("pct_timing_cells_size_beats_feature") is not None:
                w(f"\nIn "
                  f"**{float(mech['pct_timing_cells_size_beats_feature']):.0f}%** "
                  f"of {int(mech.get('n_timing_cells', 0))} "
                  f"(codec x feature x metric) cells the compressed size "
                  f"tracks the time more strongly than the feature does, and "
                  f"**{float(mech.get('pct_timing_cells_mediated_by_output_size', 0)):.0f}%** "
                  f"meet the full mediation test (size beats the feature AND "
                  f"the feature's partial correlation falls below half its "
                  f"raw value).\n")
            w("\n*Read this as mediation, not proof of cause: it is an "
              "observational test on one log, and a common driver of both "
              "would produce the same pattern. What it does rule out is the "
              "reading that the properties set the time directly -- under "
              "that reading the partial correlation would survive, and it "
              "does not.*\n")

        tp = t.get("codec_throughput", pd.DataFrame())
        if not tp.empty:
            w("\n### Throughput, for scale\n")
            w(_md_table(tp, n=20))

    # ----------------------------------------------- matched controls
    # ------------------------------------------- quantization mechanism
    # Part of the WHY section rather than its own: it is the same question
    # asked of the other half of the rows, and the reader should meet the two
    # halves together.
    qm = f.get("quantization_mechanism", {})
    w("\n### Why, for the quantized rows: the level count\n")
    if "quantization_mechanism" in a.unavailable:
        w(_unavailable(a, "quantization_mechanism"))
    elif not qm.get("available"):
        w("_Not computed._\n")
    else:
        w("The bound above stops at the lossless rows, because the logged "
          "stats describe the ORIGINAL buffer and quantization replaces it. "
          "The quantized rows have a mechanism of their own, and it needs no "
          "new measurement: the quantizer snaps every value onto a grid of "
          "spacing `2 * eb_eff` anchored at the chunk's minimum "
          "(`QuantizeDevice`, data_stats_gpu_kernels.cu), so the number of "
          "distinct values a chunk can still take afterwards is\n")
        w("```\n"
          "L  =  data_range / (2 * 0.95 * eb)        levels after quantization\n"
          "```\n")
        w("`L` is a closed-form function of the chunk's range and the "
          "configured bound. **No codec appears in it**, and it sets both "
          "outcomes at once, in opposite directions:\n")
        w("- **ratio.** An alphabet of `L` symbols carries at most `log2(L)` "
          "bits per element against the original 32, so an ideal entropy "
          "coder reaches **at least** `32 / log2(L)` from the alphabet alone "
          "(a floor: equiprobable levels maximise entropy, and real fields "
          "are peaked). The quantizer also chooses the storage width from `L` "
          "-- int8 below ~115 levels, int16 below ~29,800, else int32 -- so "
          "`4 / width_bytes` is an exact, separately attributable part of "
          "the ratio.\n"
          "- **quality.** With `L` levels the reconstruction can distinguish "
          "at most `L` states of the field. Below ~10 a wave cannot be drawn "
          "between the rungs and SSIM collapses whatever the codec; above a "
          "few hundred the loss is below SSIM's resolution.\n")
        w("`L` counts the rungs the chunk's **range** spans. The ratio is set "
          "by how many rungs the **bulk** of the values occupy, and on a field "
          "whose range is made by a rare extreme those differ by orders of "
          "magnitude. The log carries the bulk's spread too, so a second "
          "quantity falls out of the same arithmetic:\n")
        w("```\n"
          "L_bulk  =  L * (mad / range)  =  mad / (2 * 0.95 * eb)     "
          "bulk levels: the MAD in grid steps\n"
          "```\n")
        w("Still closed-form, still codec-free. `L` owns what the range "
          "decides -- the storage width, the PSNR, the regime edges -- and "
          "`L_bulk` owns the ratio.\n")
        w(f"\n{qm.get('verdict', '')}\n")
        drv = t.get("quantization_levels_drive_outcomes", pd.DataFrame())
        if not drv.empty:
            w("\n**How tightly each quantity sets each outcome** (Spearman, "
              "on chunks whose bound was honoured):\n")
            cols = [c for c in ("scope", "lib_name", "target", "n_chunks",
                                "spearman_rho_vs_levels",
                                "spearman_rho_vs_bulk_levels")
                    if c in drv.columns]
            w(_md_table(drv, cols, n=16))
            w("\n*PSNR is listed for completeness, not as evidence. The "
              "quantizer's max error saturates at 0.95·eb (measured: median "
              "0.950, p95 0.950), so rmse is ~constant·eb and "
              "psnr = 20·log10(range/rmse) is a function of `L` by "
              "construction. Its correlation checks the arithmetic; SSIM is "
              "the informative quality number.*\n")
        w("\n![best quantized ratio against levels]"
          "(figures/levels_vs_quantized_ratio.png)\n")
        w("![SSIM against levels](figures/levels_vs_ssim.png)\n")
        w("*The same x-axis in both: one number, computed before any codec "
          "runs, and the two outcomes it moves in opposite directions. Hollow "
          "points had their bound relaxed and sit off-curve by construction. "
          "Where the ratio climbs back UP at high `L`, the range is being set "
          "by a rare extreme while the bulk sits on a few rungs -- the next "
          "pair of figures, on `L_bulk`, straightens that tail out.*\n")
        w("\n![best quantized ratio against bulk levels]"
          "(figures/bulk_levels_vs_quantized_ratio.png)\n")
        w("![SSIM against bulk levels](figures/bulk_levels_vs_ssim.png)\n")
        reg = t.get("quantization_regimes", pd.DataFrame())
        if not reg.empty:
            w("\n#### The regimes, with their mechanism\n")
            w("Chunks binned by `L`. Each row is a population, its outcomes, "
              "and the one sentence that produces them.\n")
            w("![regimes](figures/quantization_regimes.png)\n")
            cols = [c for c in ("regime", "levels_from", "levels_to",
                                "n_chunks", "pct_of_chunks", "median_levels",
                                "median_width_bytes", "median_alphabet_floor",
                                "median_best_ratio", "median_meas_ssim",
                                "dominant_codec", "dominant_codec_share",
                                "pct_bound_relaxed") if c in reg.columns]
            w(_md_table(reg, cols, n=8))
            for _, r in reg.iterrows():
                w(f"- **{r['regime']}** ({int(r['n_chunks'])} chunks, "
                  f"{float(r['pct_of_chunks']):.0f}%): {r['mechanism']}")
        fld = t.get("quantization_levels_by_field", pd.DataFrame())
        if not fld.empty:
            w("\n#### Which fields the bound destroys, before any codec runs\n")
            w("Median `L` per field. A field whose median is under 10 is being "
              "stored as a few plateaus; its ratio measures the bound, not "
              "the compressor. This table is computable from the data and "
              "the bound alone, so it is the check to run BEFORE a sweep.\n")
            cols = [c for c in ("field", "error_bound", "n_chunks",
                                "median_data_range", "median_levels",
                                "pct_under_few_levels", "median_best_ratio",
                                "median_meas_ssim", "pct_bound_relaxed")
                    if c in fld.columns]
            w(_md_table(fld, cols, n=40))
        if float(qm.get("pct_bound_relaxed", 0)) > 0:
            w(f"\n> :warning: **The bound was relaxed on "
              f"{float(qm['pct_bound_relaxed']):.0f}% of chunks.** The "
              f"quantizer widens `eb` when float32 cannot represent it at "
              f"the field's magnitude (`eb - max|v|*2.4e-7 - 0.05*eb <= 0`), "
              f"and the log records only the requested value. Those chunks "
              f"are detected from the MEASURED error exceeding the request, "
              f"drawn hollow above, excluded from the correlations, and their "
              f"`L` is an upper bound. Their stored error is not the one that "
              f"was asked for.\n")

    sec("Same moment, different field")
    an = f.get("anisotropy", {})
    if "anisotropy" in a.unavailable:
        w(_unavailable(a, "anisotropy"))
    elif not an.get("available"):
        w("_Not computed._\n")
    else:
        w(f"{section_ref('Is it the data, or the clock?')} leaves one way "
          f"out. If the trouble is that the data "
          "properties and the clock move together, then compare chunks "
          "written **at the same timestep** -- the clock is held exactly, not "
          "statistically, and any difference that remains is a difference "
          "between the data. This section is that comparison, and it is the "
          "only one in a single log that the confound cannot reach.\n")
        w(f"There are **{an['n_matched_pairs']} such pairs** across "
          f"**{an['n_moments']} moments**, "
          f"{an['n_cross_field_pairs']} of them between different physical "
          f"fields.\n")
        if an.get("anisotropic"):
            w("> :rotating_light: **Chunks with the same statistics do not "
              "have the same compressibility.** Specifically:\n>")
            for r in an.get("reasons", []):
                w(f"> - {r}")
            w("")

        sp = t.get("feature_blind_spread", pd.DataFrame())
        if not sp.empty:
            w("\n### How close the statistics get, and how far the ratio stays\n")
            w("Each row restricts to pairs agreeing within the stated "
              "**relative** difference in *every* feature, in that feature's "
              "own units. Read downward: if the ratio fold does not collapse "
              "toward 1.00 as the agreement tightens, the features are not "
              "what sets the ratio.\n")
            w(_md_table(sp, ["max_rel_feature_diff_at_most", "n_pairs",
                             "n_chunks", "n_moments", "median_ratio_fold",
                             "p90_ratio_fold", "max_ratio_fold",
                             "pct_pairs_above_1.25x", "worst_pair"], n=12))

        ce = t.get("explainable_ceiling", pd.DataFrame())
        if not ce.empty:
            w("\n### The ceiling: how good a model on these features could "
              "possibly get\n")
            w("Chunks that are numerically identical in every feature are, to "
              "*any* function of those features, the **same input** -- the "
              "model must return one number for both, so the spread inside "
              "such a class is irreducible. That converts directly into an "
              "upper bound on out-of-fold R^2 that no model, no architecture "
              "and no amount of tuning can pass. Classes are formed only "
              "*within* a moment, which makes the bound conservative (it is "
              "an over-estimate of what is reachable); `near_exact` marks the "
              "rows where the tolerance is tight enough that the bound is "
              "effectively exact rather than an assumption.\n")
            w(_md_table(ce, ["max_rel_feature_diff_at_most", "near_exact",
                             "n_classes_with_more_than_one_chunk",
                             "n_chunks_in_those_classes", "n_chunks",
                             "irreducible_ss_fraction",
                             "max_achievable_oof_r2"], n=12))
            mm = t.get("model_metrics", pd.DataFrame())
            cap = an.get("max_achievable_oof_r2")
            if not mm.empty and "oof_r2" in mm.columns and cap is not None:
                best = float(mm["oof_r2"].max())
                w(f"\nAt a tolerance of "
                  f"{float(an.get('ceiling_tolerance', 0)):g} the bound is "
                  f"**R^2 <= {float(cap):.4f}**, and the best model in "
                  f"{section_ref('Data-property findings')} reaches **{best:.3f}**. "
                  + ("The bound is therefore not yet binding: most of what "
                     "the current model is missing is fit and sample size, "
                     "not feature information. "
                     if best < float(cap) - 0.02 else
                     "The model is already at the bound: the remaining error "
                     "is in the features, and no better fit can remove it. "))
                if an.get("ceiling_is_weak"):
                    w(f"But read the bound as a floor on the damage, not a "
                      f"measurement of it. It can only see chunks that HAVE a "
                      f"numerically identical twin in this log -- "
                      f"{int(an.get('ceiling_n_chunks_with_a_twin', 0))} of "
                      f"{int(an.get('ceiling_n_chunks', 0))} here. Every "
                      f"chunk without a twin contributes zero to it however "
                      f"badly the features describe it, so the true "
                      f"irreducible error is larger than this and the true "
                      f"ceiling is lower.\n")
                else:
                    w("")

        fam = t.get("component_families", pd.DataFrame())
        od = t.get("component_ordering", pd.DataFrame())
        if not od.empty:
            w("\n### Components of one vector field\n")
            w("By symmetry the components of a vector field carry the same "
              "distribution, and the features agree that they do. If the "
              "*ratio* nevertheless orders them the same way moment after "
              "moment, the ordering is a property of how each component sits "
              "in memory rather than of its statistics. The sign test is over "
              "**moments** -- each timestep votes once -- because two "
              "components at one timestep are a single observation.\n")
            if not fam.empty:
                w("Families detected from the field names: "
                  + ", ".join(
                      f"`{b}` ({'/'.join(sorted(g['axis']))})"
                      for b, g in fam.groupby("base", sort=True)) + ".\n")
            w(_md_table(od, ["family", "axis_a", "axis_b", "n_moments",
                             "more_compressible_axis", "median_ratio_fold",
                             "max_ratio_fold", "sign_test_p",
                             "median_max_rel_feature_diff",
                             "features_agree"], n=12))

        rb = t.get("residual_by_field", pd.DataFrame())
        if not rb.empty:
            w("\n### The general form: residuals with the clock pinned\n")
            w("This is the same question without needing vector fields or a "
              "naming convention. Fit out of fold on the features, then "
              "compare each chunk's residual to the mean residual **of its "
              "own moment**. Centring inside the moment removes the timestep "
              "entirely, so what is left is the part of compressibility that "
              "depends on which field a chunk came from and that the features "
              "did not supply. `median_fold_vs_moment` above 1.0 means the "
              "features UNDER-rate that field -- it compresses better than "
              "they say.\n")
            w(_md_table(rb, ["field", "n_chunks_in_shared_moments",
                             "median_fold_vs_moment", "n_sign_test",
                             "k_sign_test", "sign_test_p", "direction"],
                        n=12))
            if "field_residual_kruskal_p" in an:
                kp = float(an["field_residual_kruskal_p"])
                w(f"\nKruskal-Wallis across fields on the moment-centred "
                  f"residuals: **p = {kp:.3g}**. "
                  + ("The residuals are ordered by field, so a field-level "
                     "property the features do not carry is still driving the "
                     "outcome after they have had their say."
                     if kp < 0.05 else
                     "No field-level structure survives in the residuals.")
                  + " Chunks within one field are serially correlated in "
                    "time, so this p-value is anticonservative; the "
                    "moment-level sign tests above are the stricter reading.\n")

        if an.get("hypothesis"):
            w("\n### What would explain it\n")
            w(f"> {an['hypothesis']}\n")

    # ---------------------------------------------------- data properties
    sec("Data-property findings")
    w("**Definitions, from `data_stats_gpu_kernels.cu`:** `entropy` is the "
      "Shannon entropy of the 256-bin BYTE histogram, in bits/byte, bounded "
      "in [0,8] (float64 chunks are downcast to float32 first). `mad` is "
      "`mean|x - mean(x)|` and `second_deriv` is "
      "`mean|x[i+1] - 2x[i] + x[i-1]|`, **both in raw data units** and both "
      "computed on the FLATTENED buffer. The last point governs how they may "
      "be read: MAD in a density field and MAD in a momentum field are "
      "different physical quantities, so a correlation pooled across fields "
      "measures the field labels. The within-field table below is the "
      "controlled one.\n")
    _q(w, a, "Q1")
    _q(w, a, "Q2")
    _q(w, a, "Q3")

    ab = t.get("feature_ablation", pd.DataFrame())
    if not ab.empty:
        w("\n### Feature ablation (out-of-fold, chunks as groups)\n")
        w("`oof_r2` is on log10(best achievable ratio). `memorisation_gap` is "
          "in-fold minus out-of-fold: it is the amount of fit the chunk-level "
          "grouping refused to score, and a random row split would have "
          "counted it as accuracy.\n")
        w(_md_table(ab.sort_values(["model", "oof_r2"], ascending=[True, False]),
                    ["model", "features", "oof_r2", "in_fold_r2",
                     "memorisation_gap", "n_chunks"], n=30))
    marg = t.get("feature_ablation_marginal", pd.DataFrame())
    if not marg.empty:
        w("### Mean marginal contribution of each feature\n")
        w("The average out-of-fold R^2 a feature adds over every subset that "
          "does not already contain it.\n")
        w(_md_table(marg, ["model", "feature", "mean_marginal_oof_r2_gain",
                           "min_marginal_gain", "max_marginal_gain",
                           "n_subsets"]))
    stab = t.get("ablation_stability", pd.DataFrame())
    if not stab.empty:
        w("\n### Is each ablation verdict stable across re-folding?\n")
        w("The same data, repartitioned into folds. At a few dozen chunks a "
          "single out-of-fold delta is one draw, and a verdict that changes "
          "sign across draws is a property of the partition, not of the "
          "feature.\n")
        w(_md_table(stab, ["extra", "base", "n_repeats", "median_delta_r2",
                           "sd_delta_r2", "min_delta_r2", "max_delta_r2",
                           "n_sign_flips", "stable"]))

    gen = t.get("generalisation", pd.DataFrame())
    if not gen.empty:
        w("\n### Does the fit generalise? (the harder splits)\n")
        w("Chunk holdout is the easiest test and the one Q4's headline uses. "
          "Leaving out a whole physical field, or training on early "
          "timesteps to predict later ones, is what a deployed selector "
          "actually faces. **A negative R^2 is worse than predicting the "
          "mean**, and the gap between these rows and the chunk-holdout "
          "score is how much of that score was the fields and the clock "
          "rather than the physics.\n")
        w(_md_table(gen.sort_values("r2"),
                    ["holdout", "by", "held", "n_train", "n_test",
                     "n_test_chunks", "train_r2", "r2"], n=20))
        neg = int((gen["r2"] < 0).sum())
        if neg:
            w(f"\n:warning: **{neg} of {len(gen)} generalisation scores are "
              f"negative** -- on those splits the model is worse than a "
              f"constant.\n")

    imp = t.get("feature_importance", pd.DataFrame())
    if not imp.empty:
        w("### Permutation importance (measured on held-out chunks)\n")
        w(f"One group-held-out fold, n_test = "
          f"{int(imp['n_test'].iloc[0])} chunks.\n")
        w(_md_table(imp, ["feature", "impurity_importance",
                          "permutation_importance_mean",
                          "permutation_importance_std", "test_r2"]))

    wf = t.get("correlations_within_field", pd.DataFrame())
    if not wf.empty:
        simp = wf[wf["simpson_flag"]]
        w("\n### Simpson's-paradox check: pooled vs within-field correlation\n")
        if len(simp):
            w(f":warning: **{len(simp)} of {len(wf)} "
              f"(feature x configuration) cells reverse sign** between the "
              f"naive pooled correlation and the within-field pooled one. "
              f"*(Mechanism, not speculation: MAD and the second derivative "
              f"are in raw data units, so pooling fields with different "
              f"physical scales sorts the chunks by field before it sorts "
              f"them by data. Read the within-field column.)*\n")
            w(_md_table(simp, ["lib_name", "quantize", "shuffle", "feature",
                               "naive_pooled_r", "pooled_within_field_r",
                               "n_fields", "n_chunks"], n=15))
        else:
            w("No cell reverses sign between the pooled and within-field "
              "correlations.\n")
        w("\nFull within-field table (all configurations):\n")
        w(_md_table(wf, ["lib_name", "quantize", "shuffle", "feature",
                         "pooled_within_field_r", "naive_pooled_r",
                         "n_fields", "n_fields_agreeing_sign", "n_chunks"],
                    n=24))

    mono = t.get("trend_monotonicity", pd.DataFrame())
    if not mono.empty:
        w("\n### Where each feature stops being predictive\n")
        w("`saturation_bin_left` is the feature value beyond which the binned "
          "median moves by less than 5% of its total span.\n")
        w(_md_table(mono, ["feature", "target", "n_bins",
                           "monotone_decreasing", "monotone_increasing",
                           "sign_changes", "first_bin_median",
                           "last_bin_median", "saturation_bin_left"]))
    jr = t.get("joint_regimes", pd.DataFrame())
    if not jr.empty:
        w("\n### Joint regimes (feature levels crossed)\n")
        w(_md_table(jr[jr["usable"]] if "usable" in jr else jr, n=16))

    # ---------------------------------------------------------- codecs
    sec("Codec behaviour")
    _q(w, a, "Q5")
    _q(w, a, "Q6")
    wr = t.get("codec_win_rates", pd.DataFrame())
    if not wr.empty:
        w("\n### Win rates by objective\n")
        w("`median_nties` is how many candidates shared the win. A win with "
          "many ties was decided by the deterministic tiebreak, not by the "
          "data.\n")
        w(_md_table(wr.sort_values(["objective", "pct_wins"],
                                   ascending=[True, False]),
                    ["objective", "lib_name", "n_wins", "pct_wins",
                     "median_nties", "pct_uncontested"], n=40))
    if a.profiles:
        w("\n### Per-codec profiles\n")
        for lib, p in a.profiles.items():
            w(f"\n#### `{lib}`\n")
            for arm in ("lossless_noshuffle", "lossless_shuffled",
                        "lossy_noshuffle", "lossy_shuffled"):
                if arm not in p:
                    continue
                v = p[arm]
                dt = ("" if v.get("dt_ms_median") is None
                      or not np.isfinite(v["dt_ms_median"])
                      else f", dt {v['dt_ms_median']:.3g} ms")
                w(f"- **{arm.replace('_', ', ')}** "
                  f"({v['n_chunks']} chunks): ratio median "
                  f"{v['ratio_median']:.3g} (p10 {v['ratio_p10']:.3g}, p90 "
                  f"{v['ratio_p90']:.3g}), ct {v['ct_ms_median']:.3g} ms{dt}"
                  + (f", expanded on {v['pct_expanded']:.0f}% of rows"
                     if v.get("pct_expanded") else ""))
            wo = p.get("wins_on_data_with")
            if wo:
                w("- **Wins (highest ratio, uncontested) on data with:** "
                  + "; ".join(
                      f"{k} median {v['median']:.4g} "
                      f"[p10 {v['p10']:.4g}, p90 {v['p90']:.4g}]"
                      for k, v in wo.items())
                  + f"  ({p['n_uncontested_best_ratio_wins']} chunks)")
            else:
                w(f"- **Uncontested highest-ratio wins:** "
                  f"{p.get('n_uncontested_best_ratio_wins', 0)} -- too few to "
                  f"describe a property region.")
            for name, key in (("Shuffle", "shuffle"),
                              ("Quantization", "quantization")):
                v = p.get(key)
                if v:
                    rel = ("" if v.get("median_rel_ratio") is None
                           else f", median {v['median_rel_ratio'] * 100:+.0f}% "
                                f"ratio change")
                    w(f"- **{name}:** {v['verdict']} -- improves the ratio on "
                      f"{v['pct_pairs_ratio_improves']:.0f}% of "
                      f"{v['n_pairs']} controlled pairs{rel}")
                else:
                    w(f"- **{name}:** not measurable for this codec "
                      f"(too few paired rows).")
            rs = p.get("ratio_sensitivity")
            if rs:
                w("- **Ratio sensitivity (Spearman, per configuration):** "
                  + ", ".join(f"`{k}` {v:+.2f}" for k, v in rs.items()))

    # ------------------------------------------------------- transforms
    sec("Shuffle behaviour")
    w(_unavailable(a, "shuffle"))
    _q(w, a, "Q7")
    ss = t.get("shuffle_summary", pd.DataFrame())
    if not ss.empty:
        w("\nEach row compares the SAME chunk with the setting on and with it "
          "off, holding *(chunk, codec, preset, quantize, error bound)* "
          "constant and varying only the shuffle width. A `change` column "
          "is therefore `with - without` on one chunk, so the chunk's own "
          "difficulty cancels and what is left is the setting's effect. "
          "`shuffle` is a WIDTH, not a flag, so each width is compared "
          "against 0 separately.\n")
        w("`ratio % of pairs improved` counts PAIRS; the sign test counts "
          "CHUNKS "
          "(each chunk votes once, with the sign of its median change), "
          "because a chunk contributes one pair per codec and quantize "
          "level and those are not independent. `k`/`n` are the sign test's "
          "own counts, so the printed p is reproducible from them.\n")
        w(_md_table(ss, ["treatment", "lib_name", "n_pairs",
                         "ratio_median_delta", "ratio_median_rel",
                         "ratio_pct_helps", "ratio_pct_chunks_helped",
                         "ratio_sign_test_k_chunks",
                         "ratio_sign_test_n_chunks", "ratio_sign_test_p",
                         "ct_ms_median_delta"], n=30))
    sbr = t.get("shuffle_benefit_regimes", pd.DataFrame())
    if not sbr.empty:
        w("\n### Where shuffle helps, in property space\n")
        w(_md_table(sbr, ["treatment", "feature", "bin", "n_pairs",
                          "n_chunks", "median_delta", "pct_helps", "verdict"],
                    n=30))

    sec("Quantization behaviour")
    w(_unavailable(a, "quantization"))
    _q(w, a, "Q8")
    qs = t.get("quantization_summary", pd.DataFrame())
    if not qs.empty:
        w("\n`ratio % of pairs improved` counts pairs; the sign test counts "
          "chunks "
          "-- see the note in the shuffle section.\n")
        w(_md_table(qs, ["treatment", "lib_name", "n_pairs",
                         "ratio_median_delta", "ratio_median_rel",
                         "ratio_pct_helps", "ratio_pct_chunks_helped",
                         "ratio_sign_test_k_chunks",
                         "ratio_sign_test_n_chunks", "ratio_sign_test_p",
                         "ct_ms_median_delta"], n=30))
    qbr = t.get("quantization_benefit_regimes", pd.DataFrame())
    if not qbr.empty:
        w("\n### Where quantization helps, in property space\n")
        w(_md_table(qbr, ["treatment", "feature", "bin", "n_pairs",
                          "median_delta", "pct_helps", "verdict"], n=24))

    sec("Error-bound behaviour")
    w(_unavailable(a, "error_bound"))
    _q(w, a, "Q9")
    ebs = t.get("error_bound_summary", pd.DataFrame())
    if not ebs.empty:
        w("\n" + _md_table(ebs, n=20))

    # ------------------------------------------------------- prediction
    sec("Temporal behaviour")
    w(_unavailable(a, "temporal"))
    _q(w, a, "Q10")
    tt = t.get("temporal_trends", pd.DataFrame())
    if not tt.empty:
        w("\n" + _md_table(tt, ["field", "series", "n_timesteps",
                                "spearman_rho", "p", "usable", "first",
                                "last"], n=40))
    cs = t.get("codec_switches", pd.DataFrame())
    if not cs.empty:
        w("\n### Codec switches over the run\n")
        w(_md_table(cs, n=20))
    elif "temporal" not in a.unavailable:
        w("\nThe highest-ratio codec did not change for any field over the "
          "run.\n")
    w("\n### Block-evolution metric\n")
    w(_unavailable(a, "evolution"))
    ee = t.get("evolution_effects", pd.DataFrame())
    if not ee.empty:
        w("`E = ||B2-B1|| / (||B1||+||B2||+eps)`. The `unsaturated` subset "
          "excludes blocks at E > 0.99: on a zero-background field "
          "(momenta, velocities) E saturates at 1 the instant anything "
          "appears, so the saturated rows rank almost inversely to the real "
          "change -- `evolution.py`'s own Nyx example.\n")
        w(_md_table(ee, n=20))

    # -------------------------------------------------- counterexamples
    sec("Prediction accuracy")
    _q(w, a, "Q12")
    ps = t.get("prediction_summary", pd.DataFrame())
    if not ps.empty:
        w("\n`median_fold_error` is 10^median|log10(pred/actual)| -- a "
          "scale-free 'how many times off'. It is the headline because ratio "
          "spans three decades in one sweep, and a native-unit MAE would be "
          "owned by the few chunks above 100x. `clamped_mape_pct` applies "
          "the runtime's own clamps first (ratio capped at 100, times "
          "floored at 1 ms) so it is comparable with "
          "`cost_model_error_pct`.\n")
        w(_md_table(ps, ["metric", "role", "n", "n_chunks",
                         "median_fold_error", "p90_fold_error",
                         "log10_bias_median", "pct_over_predicted",
                         "clamped_mape_pct", "pred_actual_spearman"], n=20))
    cl = t.get("prediction_clamps", pd.DataFrame())
    if not cl.empty and "floor" in cl.columns:
        clamped = cl[cl["floor"].notna() | cl["cap"].notna()]
        if not clamped.empty:
            w("\n### The predictor's output is clamped -- read the error "
              "twice\n")
            w("The NN's heads do not emit values outside the range the cost "
              "model itself clamps to (`max(1, ct)`, `min(cap, ratio)`). "
              "Where the measured value falls outside that range, the "
              "\"prediction error\" is the clamp, not the model: a codec "
              "that really compresses in 0.11 ms cannot be predicted better "
              "than ~9x wrong by a head that bottoms out at 1 ms. The "
              "right-hand columns restrict to rows where the clamp cannot "
              "bite.\n")
            w(_md_table(clamped, ["metric", "floor", "cap", "n_at_floor",
                                  "pct_actual_below_floor", "n_rows",
                                  "n_rows_clamp_cannot_bite",
                                  "median_fold_error_all",
                                  "pct_over_predicted_all",
                                  "median_fold_error_unclamped",
                                  "pct_over_predicted_unclamped"]))
            w("\n*(This is why the by-codec table below shows the fastest "
              "codecs as the most badly mispredicted: they are the ones "
              "whose real times sit under the floor.)*\n")

    pc = t.get("prediction_by_codec", pd.DataFrame())
    if not pc.empty:
        w("\n### By codec\n")
        w(_md_table(pc, ["metric", "lib_name", "n", "median_fold_error",
                         "log10_bias_median", "pct_over_predicted",
                         "pred_actual_spearman"], n=30))
    drv = t.get("prediction_error_drivers", pd.DataFrame())
    if not drv.empty:
        top = drv.reindex(
            drv["spearman_rho"].abs().sort_values(ascending=False).index)
        w("\n### Does the error depend on the data?\n")
        w("Signed log-error against each intrinsic feature, inside a fixed "
          "configuration so the chunk is the unit. A strong correlation means "
          "the predictor has information available that it is not using.\n")
        w(_md_table(top, ["metric", "lib_name", "quantize", "shuffle",
                          "feature", "spearman_rho", "n_chunks", "p"], n=20))
    rq = t.get("ranking_quality", pd.DataFrame())
    if not rq.empty and "n_candidates_beating_top_rank" in rq:
        beat = rq["n_candidates_beating_top_rank"].dropna()
        w(f"\n**Ranking, which is what the selector actually consumes:** the "
          f"model's top-ranked alternative was beaten on measured ratio by a "
          f"median of {beat.median():.0f} other candidates "
          f"(mean {beat.mean():.1f}) across {len(beat)} chunks; the median "
          f"within-chunk Spearman between predicted and measured ratio is "
          f"{rq['pred_vs_actual_spearman'].median():+.2f}. *(A predictor can "
          f"be badly calibrated and still rank correctly; only the ranking "
          f"costs anything here.)*\n")

    # ------------------------------------------------------- cost model
    sec("Counterexamples")
    ce = t.get("counterexamples", pd.DataFrame())
    if ce.empty:
        w("None found by any finder.\n")
    else:
        w(f"{len(ce)} cases contradicting or unexplained by the dominant "
          f"trends, all of them in `counterexamples/counterexamples.csv`.\n")
        w(_md_table(ce["kind"].value_counts().rename_axis("kind")
                    .reset_index(name="n"), n=20))
        ident = ce[ce["kind"] == "identical_properties_different_ratio"] \
            if "kind" in ce else pd.DataFrame()
        if not ident.empty:
            w("\n### The decisive ones: identical properties, different "
              "compressibility\n")
            w("These chunk pairs agree on every feature to within "
              "`max_relative_feature_diff` -- a relative test in each "
              "feature's own units, not a standardised distance, so an "
              "outlier cluster inflating the global spread cannot widen it. "
              "**No model built on those three features can separate them**, "
              "whatever its form, so whatever explains the ratio gap is "
              "information the current feature set does not carry. This is "
              "an existence proof and needs no sample size -- though the "
              "pairs below are near-simultaneous components of one field, so "
              "they are fewer independent observations than rows.\n")
            w(_md_table(ident, ["chunk_uid", "chunk_uid_b",
                                "max_relative_feature_diff", "ratio_a",
                                "ratio_b", "ratio_fold", "best_lib_a",
                                "best_lib_b"], n=15))
        for kind in ("shuffle_harmful", "quantization_harmful",
                     "error_bound_violation", "cost_model_regret"):
            sub = ce[ce["kind"] == kind] if "kind" in ce else pd.DataFrame()
            if sub.empty:
                continue
            w(f"\n### {kind.replace('_', ' ')} ({len(sub)})\n")
            cols = [c for c in ("chunk_uid", "field", "lib_name", "entropy",
                                "mad", "second_deriv", "rel_ratio",
                                "meas_max_error_ok", "error_bound",
                                "primary_regret", "primary_lib",
                                "best_ratio_lib") if c in sub.columns]
            w(_md_table(sub, cols, n=10, drop_empty=True))

    # ------------------------------------------------- missing features
    sec("Missing features")
    _q(w, a, "Q14")
    props = a.answers.get("Q15", {}).get("evidence", {}).get("proposals", [])
    if not props:
        w("\nNo counterexample in this log motivates a specific new "
          "statistic. That is a weaker statement than sufficiency: see Q14.\n")
    else:
        w("\nEach proposal below is tied to a failure observed *in this log*. "
          "Statistics that nothing here motivates are deliberately not "
          "proposed.\n")
        for i, p in enumerate(props, 1):
            w(f"\n### {i}. {p['feature']}\n")
            w(f"- **Motivated by:** {p['motivated_by']}")
            w(f"- **Captures:** {p['captures']}")
            w(f"- **Why the current three miss it:** "
              f"{p['why_current_features_miss_it']}")
            w(f"- **GPU cost:** {p['gpu_cost']}")

    # ---------------------------------------------------- the questions
    sec("The fifteen questions, answered")
    for q in [f"Q{i}" for i in range(1, 18)]:
        ansd = a.answers.get(q, {})
        w(f"\n**{q}. {Q_TEXT[q]}**\n")
        if ansd.get("unavailable"):
            w(f"> **Not answerable from this log.** {ansd['unavailable']}\n")
        else:
            w(f"{ansd.get('verdict', '_(no answer produced)_')}  ")
            w(f"_Confidence: {ansd.get('confidence', 'unknown')}._\n")
        for cav in ansd.get("caveats", []):
            w(f"  - _{cav}_")

    # -------------------------------------------------------- figures
    # Emitted unconditionally: the section numbers are derived from SECTIONS,
    # and a section that sometimes disappears would leave a gap in them.
    sec("Figures")
    if not a.figures:
        w("_Figure generation was disabled for this run (`--no-figures`)._\n")
    else:
        w(f"{len(a.figures.made)} written to `figures/`:\n")
        for m in sorted(a.figures.made):
            w(f"- `{m}`")
        if a.figures.skipped:
            w("\nSkipped, with reasons -- an absent figure is a result too:\n")
            for s in a.figures.skipped:
                w(f"- `{s['figure']}`: {s['reason']}")

    # ---------------------------------------------------- reproducibility
    sec("Appendix A: what the cost column actually is")
    cm = f.get("cost_model", {})
    cd = f.get("cost_diagnostics", {})
    w("`neuropress_cost.h` defines cost as "
      "`w_ct*max(1,ct) + w_dt*max(1,dt) + w_io*bytes/(min(cap,ratio)*bw)`, "
      "**lower is better**, and every weight is overridable at runtime. The "
      "log does not record them, so they are recovered here by least squares "
      "from the `cost` column itself -- the residual is the proof:\n")
    if cm.get("identified"):
        w(f"- Recovered: `w_ct = {cm['w_ct']:.3g}`, `w_dt = {cm['w_dt']:.3g}`, "
          f"`w_io/bw = {cm['w_io_over_bw']:.4g}`, `ratio_cap = "
          f"{cm['cap']:g}` -- reproducing every one of "
          f"{cm['n_rows_fit']:,} rows to a maximum relative error of "
          f"{cm['max_rel_error']:.2g}.")
        w(f"- **This run scored on {cm['scores_on']}.**")
        if cm.get("implied_bandwidth_bytes_per_ms_if_w_io_1"):
            w(f"- Implied bandwidth (if `w_io = 1`): "
              f"{cm['implied_bandwidth_bytes_per_ms_if_w_io_1']:,.0f} "
              f"bytes/ms.")
        w(f"- `dt` source in the cost: {cm.get('dt_source')}.")
    else:
        w(f"- :warning: the cost column does **not** fit the documented model "
          f"(best max relative error {cm.get('max_rel_error', float('nan')):.3g}). "
          f"Treat every cost-based conclusion below as unverified.")
    w(f"- **{cd.get('pct_rows_at_or_above_cap', 0):.1f}%** of rows have "
      f"ratio at or above the cap of {cd.get('ratio_cap')}, and "
      f"**{cd.get('pct_chunks_tied_at_best', 0):.0f}%** of chunks "
      f"({cd.get('chunks_with_tied_best_cost')}/"
      f"{cd.get('chunks_total')}) have more than one candidate tied at the "
      f"best cost. *(Mechanism: above the cap the io term is identical, so "
      f"the cost model cannot distinguish those candidates at all. A "
      f"'wrong' selection among them is not a model failure.)*\n")

    # ------------------------------------------- the confound, up front
    sec("Appendix B: cost-model performance")
    _q(w, a, "Q13")
    sel = f.get("selection", {})
    if sel:
        w("")
        for k in ("n_chunks", "pct_exploration_changed_choice",
                  "pct_adopted_cost_optimal", "pct_primary_cost_optimal",
                  "primary_regret_median", "primary_regret_p90",
                  "primary_regret_max",
                  "pct_primary_picks_best_ratio_codec",
                  "adopted_ratio_regret_median",
                  "pct_adopted_is_best_ratio_codec"):
            if k in sel:
                w(f"- `{k}`: {sel[k]:.4g}" if isinstance(sel[k], float)
                  else f"- `{k}`: {sel[k]}")
        w(f"\n_{sel.get('note', '')}_\n")

    # --------------------------------------------------------- temporal
    sec("Appendix C: reproducibility")
    w(f"- Input: `{prov['inputs'][0]['path']}`")
    w(f"- SHA-256: `{prov['inputs'][0]['sha256']}`")
    w(f"- Bytes: {prov['inputs'][0]['bytes']:,}")
    w(f"- Rows: {ds.audit.counts.get('rows')}, unique chunks: {ds.n_chunks}")
    w(f"- Analysis git commit: `{prov.get('analysis_git_commit')}`")
    w(f"- Timestamp: {prov['timestamp']}")
    w(f"- Random seed: {prov['random_seed']}")
    w(f"- CLI: `{' '.join(prov['cli_args'])}`")
    w("- Dependencies: "
      + ", ".join(f"{k} {v}" for k, v in prov["dependencies"].items()))
    w("\nRe-running this command on the same CSV reproduces every number: all "
      "models are seeded, all tie-breaks are deterministic orderings, and no "
      "sampling is used anywhere.\n")

    text = "\n".join(L) + "\n"
    path = os.path.join(outdir, "REPORT.md")
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _q(w, a: Analysis, q: str) -> None:
    d = a.answers.get(q, {})
    w(f"\n**{q}. {Q_TEXT[q]}**\n")
    if d.get("unavailable"):
        w(f"> **Not answerable from this log.** {d['unavailable']}\n")
        return
    w(f"{d.get('verdict', '')}  _(confidence: "
      f"{d.get('confidence', 'unknown')})_\n")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict("records"))
    if o is None or isinstance(o, (str, int, bool)):
        return o
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def write_summary(a: Analysis, outdir: str, prov: dict) -> str:
    ds, t = a.ds, a.tables
    ce = t.get("counterexamples", pd.DataFrame())
    summary = {
        "schema_version": 1,
        "workload": ds.name,
        "provenance": prov,
        "dataset": {
            "rows": int(ds.audit.counts.get("rows", 0)),
            "unique_chunks": ds.n_chunks,
            "codecs": ds.audit.counts.get("codecs"),
            "fields": ds.audit.counts.get("fields"),
            "timesteps": ds.audit.counts.get("timesteps"),
            "presets": ds.audit.counts.get("presets"),
            "shuffle_widths": ds.audit.counts.get("shuffle_widths"),
            "quantize_modes": ds.audit.counts.get("quantize_modes"),
            "error_bounds_lossy": ds.audit.counts.get("error_bounds_lossy"),
            "usable_features": ds.features,
            "chunk_bytes": ds.audit.counts.get("chunk_bytes"),
        },
        "audit": ds.audit.as_dict(),
        "cohort": a.facts.get("cohort"),
        "confounds": a.facts.get("confounds"),
        "anisotropy": a.facts.get("anisotropy"),
        "feature_blind_spread": (
            t.get("feature_blind_spread", pd.DataFrame()).to_dict("records")),
        "explainable_ceiling": (
            t.get("explainable_ceiling", pd.DataFrame()).to_dict("records")),
        "component_ordering": (
            t.get("component_ordering", pd.DataFrame()).to_dict("records")),
        "residual_by_field": (
            t.get("residual_by_field", pd.DataFrame()).to_dict("records")),
        "confounder_power": (
            t.get("confounder_power", pd.DataFrame()).to_dict("records")),
        "partial_correlations": (
            t.get("partial_correlations", pd.DataFrame()).to_dict("records")),
        "prediction_clamps": (
            t.get("prediction_clamps", pd.DataFrame()).to_dict("records")),
        "cost_model": a.facts.get("cost_model"),
        "cost_diagnostics": a.facts.get("cost_diagnostics"),
        "selection": a.facts.get("selection"),
        "codec_classifier": {
            k: v for k, v in (a.facts.get("codec_classifier") or {}).items()
            if k != "tree_rules"},
        "feature_ablation": (
            t.get("feature_ablation", pd.DataFrame()).to_dict("records")),
        "feature_marginal_gains": (
            t.get("feature_ablation_marginal",
                  pd.DataFrame()).to_dict("records")),
        "feature_importance": (
            t.get("feature_importance", pd.DataFrame()).to_dict("records")),
        "generalisation": (
            t.get("generalisation", pd.DataFrame()).to_dict("records")),
        "shuffle_summary": (
            t.get("shuffle_summary", pd.DataFrame()).to_dict("records")),
        "quantization_summary": (
            t.get("quantization_summary", pd.DataFrame()).to_dict("records")),
        "error_bound_summary": (
            t.get("error_bound_summary", pd.DataFrame()).to_dict("records")),
        "prediction_summary": (
            t.get("prediction_summary", pd.DataFrame()).to_dict("records")),
        "temporal_trends": (
            t.get("temporal_trends", pd.DataFrame()).to_dict("records")),
        "codec_profiles": a.profiles,
        "discovered_thresholds": a.thresholds,
        "counterexample_counts": (
            ce["kind"].value_counts().to_dict()
            if not ce.empty and "kind" in ce else {}),
        "unavailable_analyses": a.unavailable,
        "answers": a.answers,
        "figures": {
            "written": sorted(a.figures.made) if a.figures else [],
            "skipped": a.figures.skipped if a.figures else [],
        },
    }
    path = os.path.join(outdir, "summary.json")
    with open(path, "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1, sort_keys=False)
    return path

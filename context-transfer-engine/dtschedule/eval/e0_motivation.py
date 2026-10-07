#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# All rights reserved.
#
# This file is part of IOWarp Core.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
E0 motivation (EVAL_PLAN section 2, reviewer B3): decision-flip heatmap.

For every (entropy level x producer load x target tier) cell the figure
marks the best codec by the ranker's cost of every scored candidate
(``reason == ok``; measured under ``ccm: oracle``) and the location where
dtschedule compressed (P = producer, C = consumer, from the decision's
``chosen_scenario``).  Entropy, producer load and tier come from the
decision row of the same (tag, blob).  The title states how far the best
single static codec is from the per-cell optimum.

Inputs: ``--candidates '<trace_path>.cand.*.csv'`` and ``--traces
'<trace_path>.*.csv'`` of the compress-bench runs (one run per target
tier / load level).  Output: ``e0_motivation.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd

import dtlib

LOAD_LEVELS = [0, 50, 100]
N_ENTROPY_BINS = 4
LOSS_THRESHOLD = 0.10
DECISION_FIELDS = ["entropy", "producer_cpu", "chosen_tier", "chosen_scenario"]


def join_features(cand: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    """Attach the decision-level fields to scored candidate rows.

    :param cand: Candidate rows (:func:`dtlib.load_candidates`).
    :param trace: Decision rows (:func:`dtlib.load_traces`).
    :return: ``reason == ok`` candidates with :data:`DECISION_FIELDS`
        joined on (tag, blob); rows without a decision are dropped.
    """
    ok = dtlib.ok_candidates(cand)
    dec = trace[trace["kind"] == "decision"]
    feats = dec.drop_duplicates(["tag", "blob"])[["tag", "blob"] + DECISION_FIELDS]
    merged = ok.merge(feats, on=["tag", "blob"], how="left")
    missing = merged["entropy"].isna().sum()
    if missing:
        print(f"e0: {missing} candidate rows without a matching decision row "
              "(no entropy); dropped", file=sys.stderr)
    return merged.dropna(subset=["entropy"])


def bin_cells(cand: pd.DataFrame) -> pd.DataFrame:
    """Assign each candidate row to an (entropy, load, tier) cell.

    :param cand: Output of :func:`join_features`.
    :return: Rows with ``ent_bin`` (quartile label), ``load`` (nearest of
        0/50/100 %), ``tier``, ``location`` (P or C) and ``cost`` columns.
    """
    df = cand.copy()
    edges = np.unique(np.quantile(df["entropy"], np.linspace(0, 1, N_ENTROPY_BINS + 1)))
    if len(edges) < 3:
        edges = np.array([df["entropy"].min() - 1e-9, df["entropy"].max() + 1e-9])
    labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
    df["ent_bin"] = pd.cut(df["entropy"], edges, labels=labels, include_lowest=True)
    loads = np.asarray(LOAD_LEVELS)
    cpu = df["producer_cpu"].fillna(0).to_numpy()
    df["load"] = loads[np.abs(cpu[:, None] - loads[None, :]).argmin(axis=1)]
    scen = pd.to_numeric(df["chosen_scenario"], errors="coerce").fillna(1).astype(int)
    df["location"] = np.where(scen == 3, "C", "P")
    df["tier"] = df["chosen_tier"].fillna("").astype(str).replace("", "ram")
    df["cost"] = df["cost_ms"]
    return df


def best_per_cell(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find the best codec per cell and the best static codec.

    Codec cost in a cell = mean over blobs of the codec's cheapest preset.
    The static codec minimises the summed cost over every cell it appears
    in.  The cell's location is the majority ``location`` of its decisions.

    :param df: Output of :func:`bin_cells`.
    :return: ``(cells, policies)``: one row per cell with ``lib``,
        ``location``, ``best_cost``, ``static_cost``, ``loss``; and the
        per-(cell, codec) cost table.
    """
    keys = ["ent_bin", "load", "tier"]
    per_blob = (df.groupby(keys + ["lib", "tag", "blob"], observed=True)["cost"]
                .min().reset_index())
    policies = per_blob.groupby(keys + ["lib"], observed=True)["cost"].mean().reset_index()
    idx = policies.groupby(keys, observed=True)["cost"].idxmin()
    cells = policies.loc[idx].rename(columns={"cost": "best_cost"})
    static_lib = policies.groupby("lib")["cost"].sum().idxmin()
    static = policies[policies["lib"] == static_lib][keys + ["cost"]].rename(
        columns={"cost": "static_cost"})
    cells = cells.merge(static, on=keys, how="left")
    cells["loss"] = cells["static_cost"] / cells["best_cost"] - 1.0
    loc = (df.drop_duplicates(keys + ["tag", "blob"]).groupby(keys, observed=True)
           ["location"].agg(lambda s: s.mode().iloc[0]).reset_index())
    cells = cells.merge(loc, on=keys, how="left")
    cells.attrs["static"] = static_lib
    return cells, policies


def plot(cells: pd.DataFrame, demo: bool, out_dir: str) -> list:
    """Draw the codec x location grid and save it.

    :param cells: Output of :func:`best_per_cell`.
    :param demo: Whether the data is synthetic (title prefix).
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    import matplotlib.patches as mpatches
    tiers = [t for t in dtlib.TIER_ORDER if t in set(cells["tier"])]
    tiers += sorted(set(cells["tier"]) - set(tiers))
    ent_bins = list(cells["ent_bin"].cat.categories) if hasattr(
        cells["ent_bin"], "cat") else sorted(cells["ent_bin"].unique())
    cols = [(t, l) for t in tiers for l in LOAD_LEVELS]
    fig, ax = dtlib.figure(height=0.55 * len(ent_bins) + 1.15)
    ax.grid(False)
    for i, ent in enumerate(ent_bins):
        for j, (tier, load) in enumerate(cols):
            sel = cells[(cells["ent_bin"] == ent) & (cells["tier"] == tier)
                        & (cells["load"] == load)]
            if sel.empty:
                continue
            row = sel.iloc[0]
            color = dtlib.codec_color(row["lib"])
            ax.add_patch(mpatches.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92,
                                            facecolor=color, edgecolor=dtlib.SURFACE,
                                            linewidth=0.8))
            lum = sum(int(color[k:k + 2], 16) for k in (1, 3, 5)) / 3.0
            ax.text(j + 0.5, i + 0.5, row["location"], ha="center", va="center",
                    fontsize=7, color="white" if lum < 150 else dtlib.INK,
                    fontweight="bold")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(ent_bins), 0)
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels([str(l) for _, l in cols], fontsize=6)
    ax.set_xlabel("producer load (%) per target tier", labelpad=22)
    for k, tier in enumerate(tiers):
        ax.text(k * len(LOAD_LEVELS) + len(LOAD_LEVELS) / 2.0, len(ent_bins) + 0.38,
                tier.upper(), ha="center", va="top", fontsize=6.5, color=dtlib.INK2,
                clip_on=False)
    ax.set_yticks([i + 0.5 for i in range(len(ent_bins))])
    ax.set_yticklabels(ent_bins, fontsize=6)
    ax.set_ylabel("entropy (bits/B)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    libs = list(dict.fromkeys(cells["lib"]))
    handles = [mpatches.Patch(color=dtlib.codec_color(l), label=l) for l in libs]
    handles.append(mpatches.Patch(facecolor="none", edgecolor="none",
                                  label="P/C = compressed at producer / consumer"))
    dtlib.legend_outside(ax, ncol=min(4, len(handles)), handles=handles, fontsize=6)
    frac = float((cells["loss"] >= LOSS_THRESHOLD).mean()) * 100.0
    dtlib.demo_title(ax, f"best static codec ({cells.attrs['static']}) is "
                     f">={LOSS_THRESHOLD:.0%} off in {frac:.0f}% of cells", demo)
    return dtlib.savefig(fig, "e0_motivation", out_dir)


def make_demo() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesise candidate and decision rows (``--demo`` only).

    :return: ``(candidates, trace)`` shaped like the loaders' output.
    """
    rng = np.random.default_rng(0)
    cand_rows, dec_rows = [], []
    blob = 0
    for ent in np.linspace(1.0, 7.5, 16):
        for load in LOAD_LEVELS:
            for tier, bw in (("ram", 8.0), ("nvme", 1.0), ("ssd", 0.2)):
                mult = 1.0 + load / 50.0
                for lib, speed, ratio in (("zstd", 1.0, 3.0), ("lz4", 0.3, 1.8),
                                          ("sz3", 2.5, 8.0), ("zlib", 2.0, 3.2)):
                    # lossless ratios collapse with entropy; lossy sz3 does not
                    r = ratio if lib == "sz3" else 1.0 + (ratio - 1.0) * (8.0 - ent) / 7.0
                    for preset in (0, 1):
                        cost = speed * (1 + 0.3 * preset) * mult + (1.0 / r) / bw
                        cand_rows.append({"tag": "t", "blob": f"b{blob}", "lib": lib,
                                          "preset": preset, "pred_ctime_ms": speed,
                                          "pred_dtime_ms": speed * 0.4, "pred_ratio": r,
                                          "cost_ms": cost * rng.lognormal(0, 0.05),
                                          "reason": "ok"})
                cand_rows.append({"tag": "t", "blob": f"b{blob}", "lib": "zfp", "preset": 0,
                                  "pred_ctime_ms": np.nan, "pred_dtime_ms": np.nan,
                                  "pred_ratio": np.nan, "cost_ms": np.nan,
                                  "reason": "qos_error_bound"})
                dec_rows.append({"tag": "t", "blob": f"b{blob}", "entropy": ent,
                                 "producer_cpu": load, "chosen_tier": tier,
                                 "chosen_scenario": 3 if load >= 50 else 1})
                blob += 1
    cand = pd.DataFrame(cand_rows)
    trace = dtlib.demo_trace(len(dec_rows), seed=1)
    dec = trace["kind"] == "decision"
    for col in ["tag", "blob"] + DECISION_FIELDS:
        trace.loc[dec, col] = pd.DataFrame(dec_rows)[col].to_numpy()
    return cand, trace


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "unused for E0 (inputs are traces)")
    parser.add_argument("--candidates", help="glob of *.cand.*.csv files")
    parser.add_argument("--traces", help="glob of decision trace CSVs")
    args = parser.parse_args(argv)
    if args.demo:
        cand, trace = make_demo()
    else:
        if not args.candidates or not args.traces:
            parser.error("--candidates and --traces are required without --demo")
        cand = dtlib.load_candidates(args.candidates)
        trace = dtlib.load_traces(args.traces)
    cells, _ = best_per_cell(bin_cells(join_features(cand, trace)))
    dtlib.print_table(cells, "E0 best codec and compression location per cell")
    for path in plot(cells, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

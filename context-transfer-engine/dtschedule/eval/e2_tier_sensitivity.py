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
E2 tier sensitivity (EVAL_PLAN section 2, reviewers A2 / B4).

Same workload on tier sets {RAM}, {NVMe}, {SSD}, {NFS}, {RAM+NVMe},
{RAM+NVMe+SSD}, {all+NFS}, capacity-constrained (fast tier holds 30 % of
the data) and unconstrained.  Three stacked rows: makespan per tier set,
bytes landed per tier, and the codec share per tier.

Inputs: ``--results <exp dir>``.  Each run must carry the knobs
``tier_set`` (e.g. ``ram+nvme``) and ``capacity`` (``constrained`` /
``unconstrained``); when they are not overrides, the run name is parsed as
``<tier_set>[__cap]``.  Bytes per tier and codec per tier come from the
run's traces (``<exp>/traces/<run>/*.csv``); without traces the
``dtschedule.tier.<name>`` counts stand in.  Output: ``e2_tier_sensitivity.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import dtlib

TIER_SETS = ["ram", "nvme", "ssd", "nfs", "ram+nvme", "ram+nvme+ssd", "all+nfs"]
CAPACITY = ["unconstrained", "constrained"]
CAP_COLORS = {"unconstrained": dtlib.CAT[0], "constrained": dtlib.CAT[1]}
CAP_HATCH = {"unconstrained": None, "constrained": dtlib.HATCH}


def label_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``tier_set`` and ``capacity`` columns exist.

    :param results: Rows of :func:`dtlib.load_results`.
    :return: Copy with both columns filled (parsed from the run name when
        the knobs were not overrides).
    """
    df = results.copy()
    if "tier_set" not in df.columns:
        df["tier_set"] = df["run"].str.split("__").str[0].str.replace(
            r"[._-]r\d+$", "", regex=True)
    if "capacity" not in df.columns:
        if "capacity_constrained" in df.columns:
            df["capacity"] = np.where(df["capacity_constrained"].astype(bool),
                                      "constrained", "unconstrained")
        else:
            df["capacity"] = np.where(df["run"].str.contains("__cap"),
                                      "constrained", "unconstrained")
    df["tier_set"] = df["tier_set"].astype(str).str.lower()
    order = [t for t in TIER_SETS if t in set(df["tier_set"])]
    order += sorted(set(df["tier_set"]) - set(order))
    df["tier_set"] = pd.Categorical(df["tier_set"], order, ordered=True)
    return df


def per_tier_tables(results: pd.DataFrame, traces: Dict[str, pd.DataFrame]):
    """Bytes stored per tier and codec share per tier for every run.

    :param results: Labelled result rows.
    :param traces: ``{run: trace DataFrame}`` (may be empty per run).
    :return: ``(bytes_df, codec_df)``: long tables with columns
        ``tier_set, capacity, tier, bytes`` and
        ``tier_set, capacity, tier, lib, count``.
    """
    byte_rows, codec_rows = [], []
    for _, row in results.iterrows():
        key = (row["tier_set"], row["capacity"])
        trace = traces.get(row["run"])
        if trace is not None and not trace.empty:
            dec = trace[trace["kind"] == "decision"].copy()
            dec["stored"] = dec["size"] / dec["obs_ratio"].fillna(1.0).clip(lower=1e-9)
            for tier, grp in dec.groupby("chosen_tier"):
                byte_rows.append({"tier_set": key[0], "capacity": key[1],
                                  "tier": tier or "ram", "bytes": grp["stored"].sum()})
                for lib, cnt in grp["chosen_lib"].value_counts().items():
                    codec_rows.append({"tier_set": key[0], "capacity": key[1],
                                       "tier": tier or "ram", "lib": lib, "count": cnt})
        else:
            counts = dtlib.stat_prefix(row, f"{dtlib.DT_PKG}.tier.")
            total = sum(counts.values()) or 1.0
            bytes_in = float(row.get(f"{dtlib.DT_PKG}.bytes_in", np.nan))
            ratio = float(row.get(f"{dtlib.DT_PKG}.mean_ratio", np.nan)) or np.nan
            for tier, cnt in counts.items():
                byte_rows.append({"tier_set": key[0], "capacity": key[1], "tier": tier,
                                  "bytes": bytes_in / ratio * cnt / total})
    bytes_df = pd.DataFrame(byte_rows, columns=["tier_set", "capacity", "tier", "bytes"])
    codec_df = pd.DataFrame(codec_rows, columns=["tier_set", "capacity", "tier",
                                                 "lib", "count"])
    return bytes_df, codec_df


def _plot_makespan(ax, agg: pd.DataFrame, order: List[str]) -> None:
    """Grouped makespan bars (one bar per capacity mode) with CIs.

    :param ax: Axes.
    :param agg: Output of :func:`dtlib.aggregate` by (tier_set, capacity).
    :param order: Tier-set order on the x axis.
    """
    width = 0.38
    caps = [c for c in CAPACITY if c in set(agg["capacity"])]
    for k, cap in enumerate(caps):
        sub = agg[agg["capacity"] == cap].set_index("tier_set").reindex(order)
        x = np.arange(len(order)) + (k - (len(caps) - 1) / 2.0) * width
        est = sub["makespan_ms"].to_numpy() / 1000.0
        lo = sub["makespan_ms_lo"].to_numpy() / 1000.0
        hi = sub["makespan_ms_hi"].to_numpy() / 1000.0
        ax.bar(x, est, width, label=cap, **dtlib.bar_kwargs(CAP_COLORS[cap],
                                                             CAP_HATCH[cap]))
        ax.errorbar(x, est, yerr=np.clip([est - lo, hi - est], 0, None),
                    fmt="none", ecolor=dtlib.INK2, elinewidth=0.6, capsize=1.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=6)
    ax.set_ylabel("makespan (s)")
    ax.set_ylim(0, None)
    if len(caps) > 1:
        dtlib.legend_outside(ax, ncol=2)


def _plot_bytes(ax, bytes_df: pd.DataFrame, order: List[str], cap: str) -> None:
    """Stacked bytes-per-tier bars for one capacity mode.

    :param ax: Axes.
    :param bytes_df: Long bytes table.
    :param order: Tier-set order.
    :param cap: Capacity mode to show.
    """
    sub = bytes_df[bytes_df["capacity"] == cap]
    piv = (sub.pivot_table(index="tier_set", columns="tier", values="bytes",
                           aggfunc="mean", observed=True).reindex(order).fillna(0.0))
    tiers = [t for t in dtlib.TIER_ORDER if t in piv.columns] + [
        t for t in piv.columns if t not in dtlib.TIER_ORDER]
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for tier in tiers:
        vals = piv[tier].to_numpy() / 2 ** 30
        ax.bar(x, vals, 0.6, bottom=bottom, label=tier,
               **dtlib.bar_kwargs(dtlib.TIER_COLORS.get(tier, dtlib.OTHER)))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=6)
    ax.set_ylabel(f"bytes stored (GiB)\n{cap}")
    ax.set_ylim(0, None)
    dtlib.legend_outside(ax, ncol=min(4, len(tiers)))


def _plot_codecs(ax, codec_df: pd.DataFrame, order: List[str], cap: str) -> None:
    """Codec share per (tier set, tier) as 100 % stacked bars.

    :param ax: Axes.
    :param codec_df: Long codec-count table.
    :param order: Tier-set order.
    :param cap: Capacity mode to show.
    """
    sub = codec_df[codec_df["capacity"] == cap]
    if sub.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "codec per tier needs per-run traces", ha="center",
                va="center", color=dtlib.MUTED, fontsize=7)
        return
    sub = sub.assign(col=sub["tier_set"].astype(str) + "/" + sub["tier"])
    cols = [f"{ts}/{t}" for ts in order for t in dtlib.TIER_ORDER
            if f"{ts}/{t}" in set(sub["col"])]
    piv = sub.pivot_table(index="col", columns="lib", values="count",
                          aggfunc="sum").reindex(cols).fillna(0.0)
    share = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100
    libs = list(share.sum().sort_values(ascending=False).index)
    bottom = np.zeros(len(cols))
    x = np.arange(len(cols))
    for lib in libs:
        ax.bar(x, share[lib].to_numpy(), 0.7, bottom=bottom, label=lib,
               **dtlib.bar_kwargs(dtlib.codec_color(lib)))
        bottom += share[lib].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=5.5)
    ax.set_ylabel(f"codec share (%)\n{cap}")
    ax.set_ylim(0, 100)
    dtlib.legend_outside(ax, ncol=min(5, len(libs)))


def plot(results: pd.DataFrame, bytes_df: pd.DataFrame, codec_df: pd.DataFrame,
         demo: bool, out_dir: str) -> List[str]:
    """Draw the three-row figure.

    :param results: Labelled result rows.
    :param bytes_df: Bytes per tier table.
    :param codec_df: Codec per tier table.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    order = list(results["tier_set"].cat.categories)
    agg = dtlib.aggregate(results, ["tier_set", "capacity"], "makespan_ms")
    cap = "constrained" if "constrained" in set(results["capacity"]) else \
        results["capacity"].iloc[0]
    fig, axes = dtlib.figure(3, 1, height=5.2, gridspec_kw={"hspace": 0.95})
    _plot_makespan(axes[0], agg, order)
    dtlib.demo_title(axes[0], "makespan per tier set", demo)
    _plot_bytes(axes[1], bytes_df, order, cap)
    _plot_codecs(axes[2], codec_df, order, cap)
    return dtlib.savefig(fig, "e2_tier_sensitivity", out_dir)


def make_demo():
    """Synthesise a tier-set sweep with traces (``--demo`` only).

    :return: ``(results, traces)``.
    """
    speed = {"ram": 1.0, "nvme": 1.15, "ssd": 1.5, "nfs": 2.4, "ram+nvme": 1.05,
             "ram+nvme+ssd": 1.08, "all+nfs": 1.12}
    settings = [{"tier_set": ts, "capacity": cap} for ts in TIER_SETS for cap in CAPACITY]
    results = dtlib.demo_results(
        settings, effect=lambda s: speed[s["tier_set"]]
        * (1.2 if s["capacity"] == "constrained" and "+" in s["tier_set"] else 1.0),
        repeats=2)
    traces = {}
    for i, row in results.iterrows():
        tiers = row["tier_set"].replace("all+nfs", "ram+nvme+ssd+nfs").split("+")
        traces[row["run"]] = dtlib.demo_trace(120, seed=i, tiers=tiers)
    return results, traces


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir of the tier-set sweep")
    args = parser.parse_args(argv)
    if args.demo:
        results, traces = make_demo()
    else:
        if not args.results:
            parser.error("--results is required without --demo")
        results = dtlib.load_results(args.results)
        traces = {r["run"]: dtlib.load_traces(r["trace_glob"])
                  for _, r in results.iterrows() if r["trace_glob"]}
    results = label_runs(results)
    bytes_df, codec_df = per_tier_tables(results, traces)
    agg = dtlib.aggregate(results, ["tier_set", "capacity"], "makespan_ms")
    dtlib.print_table(agg, "E2 makespan per tier set (ms)")
    dtlib.print_table(bytes_df.groupby(["tier_set", "capacity", "tier"], observed=True)
                      ["bytes"].mean().reset_index(), "E2 bytes landed per tier")
    for path in plot(results, bytes_df, codec_df, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
E5 scenario ablation (EVAL_PLAN section 2, reviewers B6 / B1).

Heatmap of makespan normalised to the best *fixed* scenario per grid cell
(rows S1, S2, S3, auto; columns = consumer load x producer load x network
cells), plus a stacked bar of the scenarios ``auto`` chose in each cell.

Inputs: ``--results <parent dir>`` holding one ``sweep.py --exp`` directory
per grid cell (sub-directory name = cell label, e.g. ``c50_p0_n25``) with
runs over ``force_scenario`` in {auto, 1, 2, 3}; a flat directory whose
runs carry a ``cell`` override also works.  The scenario distribution
comes from the auto run's traces (``<cell>/traces/<run>/*.csv``) or, without
traces, from its ``dtschedule.scenario.<k>`` stats.
Output: ``e5_scenario_ablation.svg/pdf`` + the normalised table.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import dtlib

SCENARIOS = ["1", "2", "3", "auto"]
ROW_LABELS = {"1": "S1", "2": "S2", "3": "S3", "auto": "auto"}
BEST_FIXED_TOL = 0.05


def prepare(results: pd.DataFrame) -> pd.DataFrame:
    """Normalise makespan to the best fixed scenario within each cell.

    :param results: Rows of :func:`dtlib.load_results` with
        ``force_scenario`` and ``cell``.
    :return: Aggregated frame: one row per (cell, scenario) with
        ``makespan_ms``, ``norm`` (= makespan / min over S1..S3 means).
    """
    df = results.copy()
    df["force_scenario"] = df["force_scenario"].astype(str).str.lower()
    if (df["cell"] == "").all() and "cell_label" in df.columns:
        df["cell"] = df["cell_label"].astype(str)
    df["cell"] = df["cell"].replace("", "all")
    agg = dtlib.aggregate(df, ["cell", "force_scenario"], "makespan_ms")
    fixed = agg[agg["force_scenario"] != "auto"]
    best = fixed.groupby("cell")["makespan_ms"].min().rename("best_fixed")
    agg = agg.merge(best.reset_index(), on="cell", how="left")
    for suffix in ("", "_lo", "_hi"):
        agg[f"norm{suffix}"] = agg[f"makespan_ms{suffix}"] / agg["best_fixed"]
    return agg


def scenario_mix(results: pd.DataFrame) -> pd.DataFrame:
    """Scenario distribution chosen by the ``auto`` run per cell.

    :param results: Result rows.
    :return: Long frame ``cell, scenario, share`` (shares sum to 1 per cell).
    """
    rows = []
    auto = results[results["force_scenario"].astype(str).str.lower() == "auto"]
    for cell, grp in auto.groupby(auto["cell"].replace("", "all")):
        counts: Dict[str, float] = {}
        for _, row in grp.iterrows():
            if row.get("trace_glob"):
                trace = dtlib.load_traces(row["trace_glob"])
                dec = trace[trace["kind"] == "decision"]
                vc = dec["chosen_scenario"].dropna().astype(int).astype(str).value_counts()
                part = vc.to_dict()
            else:
                part = dtlib.stat_prefix(row, f"{dtlib.DT_PKG}.scenario.")
            for k, v in part.items():
                counts[str(k)] = counts.get(str(k), 0.0) + float(v)
        total = sum(counts.values()) or 1.0
        for k, v in counts.items():
            rows.append({"cell": cell, "scenario": k, "share": v / total})
    return pd.DataFrame(rows, columns=["cell", "scenario", "share"])


def _heatmap(ax, agg: pd.DataFrame, cells: List[str]) -> None:
    """Draw the normalised-makespan heatmap with cell annotations.

    :param ax: Axes.
    :param agg: Output of :func:`prepare`.
    :param cells: Column order.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", dtlib.DIVERGING)
    piv = agg.pivot(index="force_scenario", columns="cell", values="norm")
    piv = piv.reindex(index=[s for s in SCENARIOS if s in piv.index], columns=cells)
    vmax = float(np.nanmax(piv.to_numpy())) if piv.size else 1.5
    vmax = max(1.15, min(vmax, 2.0))
    norm = TwoSlopeNorm(vmin=2.0 - vmax, vcenter=1.0, vmax=vmax)
    ax.imshow(piv.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iat[i, j]
            if np.isnan(val):
                continue
            dark = abs(val - 1.0) > 0.6 * (vmax - 1.0)
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if dark else dtlib.INK)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c.replace("_", "/") for c in cells], rotation=35,
                       ha="right", fontsize=6)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([ROW_LABELS.get(s, s) for s in piv.index])
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("forced scenario")


def _mix_bars(ax, mix: pd.DataFrame, cells: List[str]) -> None:
    """Stacked bars of the scenarios auto chose per cell.

    :param ax: Axes.
    :param mix: Output of :func:`scenario_mix`.
    :param cells: Column order.
    """
    piv = mix.pivot(index="cell", columns="scenario", values="share").reindex(cells)
    piv = piv.fillna(0.0) * 100.0
    x = np.arange(len(cells))
    bottom = np.zeros(len(cells))
    for scen in [s for s in SCENARIOS[:-1] if s in piv.columns]:
        vals = piv[scen].to_numpy()
        ax.bar(x, vals, 0.65, bottom=bottom, label=ROW_LABELS[scen],
               **dtlib.bar_kwargs(dtlib.SCENARIO_COLORS[scen]))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "/") for c in cells], rotation=35,
                       ha="right", fontsize=6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("auto's choices (%)")
    dtlib.legend_outside(ax, ncol=3)


def plot(agg: pd.DataFrame, mix: pd.DataFrame, demo: bool, out_dir: str) -> List[str]:
    """Draw heatmap + scenario-mix bars and save.

    :param agg: Output of :func:`prepare`.
    :param mix: Output of :func:`scenario_mix`.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    cells = sorted(agg["cell"].unique())
    fig, (ax1, ax2) = dtlib.figure(2, 1, height=3.6, gridspec_kw={
        "height_ratios": [1.3, 1.0], "hspace": 0.9})
    _heatmap(ax1, agg, cells)
    auto = agg[agg["force_scenario"] == "auto"]
    within = float((auto["norm"] <= 1.0 + BEST_FIXED_TOL).mean()) * 100 if len(auto) else np.nan
    dtlib.demo_title(ax1, f"makespan / best fixed; auto within {BEST_FIXED_TOL:.0%} "
                     f"in {within:.0f}% of cells", demo)
    _mix_bars(ax2, mix, cells)
    return dtlib.savefig(fig, "e5_scenario_ablation", out_dir)


def make_demo() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesise an E5 grid (``--demo`` only).

    :return: ``(results, mix)``.
    """
    cells = [f"c{c}_p{p}_n{n}" for c in (0, 50, 100) for p in (0, 50) for n in (25, 5)]
    cost = {"1": lambda c, p, n: 1.0 + p / 100 + (0.3 if n == 5 else 0),
            "2": lambda c, p, n: 1.05 + p / 100 + (0.5 if n == 5 else 0),
            "3": lambda c, p, n: 1.1 + c / 100 + (0.5 if n == 5 else 0)}
    settings = []
    for cell in cells:
        c, p, n = (int(x[1:]) for x in cell.split("_"))
        for scen in SCENARIOS:
            settings.append({"cell": cell, "force_scenario": scen, "_cpn": (c, p, n)})

    def effect(s):
        """Makespan multiplier of one synthetic setting.

        :param s: Setting dict with ``_cpn`` and ``force_scenario``.
        :return: Multiplier.
        """
        c, p, n = s["_cpn"]
        best = min(f(c, p, n) for f in cost.values())
        return best * 1.03 if s["force_scenario"] == "auto" else cost[s["force_scenario"]](c, p, n)

    results = dtlib.demo_results(settings, effect=effect, repeats=2).drop(columns="_cpn")
    rng = np.random.default_rng(5)
    rows = []
    for cell in cells:
        c, p, n = (int(x[1:]) for x in cell.split("_"))
        w = np.array([1.0 + c / 50, 0.5, 1.0 + p / 50]) + rng.uniform(0, 0.3, 3)
        for scen, share in zip(SCENARIOS[:-1], w / w.sum()):
            rows.append({"cell": cell, "scenario": scen, "share": share})
    return results, pd.DataFrame(rows)


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "parent dir of the per-cell sweep dirs")
    args = parser.parse_args(argv)
    if args.demo:
        results, mix = make_demo()
    else:
        if not args.results:
            parser.error("--results is required without --demo")
        results = dtlib.load_results(args.results)
        mix = scenario_mix(results)
    agg = prepare(results)
    dtlib.print_table(agg[["cell", "force_scenario", "makespan_ms", "norm",
                           "norm_lo", "norm_hi", "n"]],
                      "E5 makespan normalised to best fixed scenario")
    dtlib.print_table(mix, "E5 scenarios chosen by auto")
    for path in plot(agg, mix, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

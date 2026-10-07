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
E6 component ablation (EVAL_PLAN section 2, reviewers C1 / B6).

Leave-one-out and cumulative-add over five switches -- routing (+w),
prefetch, CCM selection, load-aware offload (+l), lossy family -- as
stacked compute / I/O makespan bars per workflow.  The I/O share is the
trace-measured time inside dtschedule (compress + store on writes,
decompress on reads) divided by the node count; compute is the remainder
of the makespan.

Inputs: ``--results <exp dir>`` of ``e6_component_ablation.yaml`` (one
sub-directory per workflow when several were run: ``<exp>/<workflow>/``).
Each run's component set is derived from its knobs: ``workflow_aware !=
none`` -> routing, ``prefetch`` (default on), ``ccm`` not ``fixed:*`` ->
ccm, ``load_aware`` -> load, ``qos_max_error > 0`` -> lossy.
Cache hit rate and remote bytes are printed when the stats carry
``dtschedule.cache_hit_rate`` / ``dtschedule.remote_read_bytes``.
Output: ``e6_component_ablation.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

import dtlib

COMPONENTS = ["routing", "prefetch", "ccm", "load", "lossy"]
COMP_SHORT = {"routing": "w", "prefetch": "pf", "ccm": "ccm", "load": "l",
              "lossy": "lossy"}
CUMULATIVE_ORDER = ["routing", "ccm", "load", "prefetch", "lossy"]


def _truthy(value) -> bool:
    """Interpret a knob value (bool / str / number) as a boolean.

    :param value: Knob value.
    :return: Boolean.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "on", "yes")
    try:
        return bool(value) and not (isinstance(value, float) and np.isnan(value))
    except (TypeError, ValueError):
        return False


def component_set(row: pd.Series) -> frozenset:
    """Derive the enabled component set of one run from its knobs.

    :param row: Result row.
    :return: Frozenset of :data:`COMPONENTS` names that are on.
    """
    on = set()
    if str(row.get("workflow_aware", "dag")).lower() not in ("none", "nan", ""):
        on.add("routing")
    if "prefetch" not in row.index or _truthy(row["prefetch"]):
        on.add("prefetch")
    if not str(row.get("ccm", "qtable")).lower().startswith("fixed"):
        on.add("ccm")
    if _truthy(row.get("load_aware", True)):
        on.add("load")
    try:
        if float(row.get("qos_max_error", 0.0)) > 0:
            on.add("lossy")
    except (TypeError, ValueError):
        pass
    return frozenset(on)


def label_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Attach ``components``, ``label``, ``panel`` and the I/O split.

    :param results: Rows of :func:`dtlib.load_results`.
    :return: Copy with ``components`` (frozenset), ``label`` (``full``,
        ``-ccm``, ``+w+ccm`` ...), ``panel`` (``loo`` / ``cum``), ``io_ms``,
        ``compute_ms`` and ``workflow`` (the ``cell`` sub-directory).
    """
    df = results.copy()
    df["components"] = [component_set(r) for _, r in df.iterrows()]
    labels, panels = [], []
    for comp in df["components"]:
        missing = [c for c in COMPONENTS if c not in comp]
        if not missing:
            labels.append("full")
            panels.append("both")
        elif len(missing) == 1:
            labels.append("-" + COMP_SHORT[missing[0]])
            panels.append("loo")
        else:
            on = [c for c in CUMULATIVE_ORDER if c in comp]
            labels.append("none" if not on else "".join("+" + COMP_SHORT[c] for c in on))
            panels.append("cum")
    df["label"] = labels
    df["panel"] = panels
    df["workflow"] = df["cell"].replace("", "workflow")
    puts = df.get(f"{dtlib.DT_PKG}.puts", pd.Series(0, index=df.index)).fillna(0)
    gets = df.get(f"{dtlib.DT_PKG}.gets", pd.Series(0, index=df.index)).fillna(0)
    ct = df.get(f"{dtlib.DT_PKG}.mean_obs_ctime_ms", pd.Series(0.0, index=df.index)).fillna(0)
    dt = df.get(f"{dtlib.DT_PKG}.mean_obs_dtime_ms", pd.Series(0.0, index=df.index)).fillna(0)
    nodes = df.get("n_nodes", pd.Series(1, index=df.index)).fillna(1).clip(lower=1)
    df["io_ms"] = (puts * ct + gets * dt) / nodes
    df["io_ms"] = df["io_ms"].clip(upper=df["makespan_ms"])
    df["compute_ms"] = df["makespan_ms"] - df["io_ms"]
    return df


def panel_table(df: pd.DataFrame, panel: str) -> pd.DataFrame:
    """Aggregate one panel (leave-one-out or cumulative) per workflow.

    :param df: Labelled rows.
    :param panel: ``"loo"`` or ``"cum"``.
    :return: Rows ``workflow, label, order, makespan_ms(+ci), io_ms, compute_ms``.
    """
    sub = df[df["panel"].isin([panel, "both"])]
    if panel == "loo":
        order = ["full"] + ["-" + COMP_SHORT[c] for c in COMPONENTS]
    else:
        order = ["none"] + ["".join("+" + COMP_SHORT[c] for c in CUMULATIVE_ORDER[:k])
                            for k in range(1, len(CUMULATIVE_ORDER))] + ["full"]
    mk = dtlib.aggregate(sub, ["workflow", "label"], "makespan_ms")
    io = sub.groupby(["workflow", "label"])[["io_ms", "compute_ms"]].mean().reset_index()
    tab = mk.merge(io, on=["workflow", "label"])
    tab["order"] = tab["label"].map({l: i for i, l in enumerate(order)})
    tab = tab[tab["order"].notna()].sort_values(["workflow", "order"])
    return tab


def _stacked(ax, tab: pd.DataFrame, title: str, demo: bool) -> None:
    """Draw stacked compute / I/O bars, grouped by workflow.

    :param ax: Axes.
    :param tab: Output of :func:`panel_table`.
    :param title: Panel title.
    :param demo: Synthetic-data flag.
    """
    workflows = list(dict.fromkeys(tab["workflow"]))
    labels = list(dict.fromkeys(tab.sort_values("order")["label"]))
    n_wf, n_lab = len(workflows), len(labels)
    width = 0.8 / max(n_lab, 1)
    ticks, tick_labels = [], []
    for w, wf in enumerate(workflows):
        sub = tab[tab["workflow"] == wf].set_index("label").reindex(labels)
        x = w + (np.arange(n_lab) - (n_lab - 1) / 2.0) * width
        comp = sub["compute_ms"].to_numpy() / 1000.0
        io = sub["io_ms"].to_numpy() / 1000.0
        ax.bar(x, comp, width, label="compute" if w == 0 else None,
               **dtlib.bar_kwargs(dtlib.CAT[0]))
        ax.bar(x, io, width, bottom=comp, label="I/O (dtschedule)" if w == 0 else None,
               **dtlib.bar_kwargs(dtlib.CAT[1], dtlib.HATCH))
        total = sub["makespan_ms"].to_numpy() / 1000.0
        err = np.vstack([total - sub["makespan_ms_lo"] / 1000.0,
                         sub["makespan_ms_hi"] / 1000.0 - total])
        ax.errorbar(x, total, yerr=np.clip(err, 0, None), fmt="none",
                    ecolor=dtlib.INK2, elinewidth=0.5, capsize=1)
        ticks.extend(x.tolist())
        tick_labels.extend(labels)
    axes_pt = ax.get_position().height * ax.get_figure().get_figheight() * 72.0
    name_y = -(10.0 + 4.4 * max(len(l) for l in labels)) / axes_pt
    for w, wf in enumerate(workflows):
        ax.text(w, name_y, wf, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=6.5, color=dtlib.INK2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("makespan (s)")
    ax.set_ylim(0, None)
    dtlib.legend_outside(ax, ncol=2)
    dtlib.demo_title(ax, title, demo)


def plot(loo: pd.DataFrame, cum: pd.DataFrame, demo: bool, out_dir: str) -> List[str]:
    """Draw leave-one-out (top) and cumulative (bottom) panels.

    :param loo: Leave-one-out table.
    :param cum: Cumulative table.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    fig, (ax1, ax2) = dtlib.figure(2, 1, height=4.6, gridspec_kw={"hspace": 0.9})
    _stacked(ax1, loo, "leave one out (full, then -component)", demo)
    _stacked(ax2, cum, "cumulative add (none -> full)", demo)
    return dtlib.savefig(fig, "e6_component_ablation", out_dir)


def make_demo() -> pd.DataFrame:
    """Synthesise leave-one-out and cumulative runs for two workflows.

    :return: Result rows shaped like :func:`dtlib.load_results`.
    """
    gain = {"routing": 0.18, "prefetch": 0.06, "ccm": 0.12, "load": 0.09, "lossy": 0.15}
    settings = []
    for wf in ("montage", "genome"):
        sets = [frozenset(COMPONENTS)] + [frozenset(set(COMPONENTS) - {c}) for c in COMPONENTS]
        sets += [frozenset(CUMULATIVE_ORDER[:k]) for k in range(0, len(CUMULATIVE_ORDER))]
        for comp in sets:
            settings.append({"cell": wf, "workflow_aware": "dag" if "routing" in comp else "none",
                             "prefetch": "prefetch" in comp,
                             "ccm": "qtable" if "ccm" in comp else "fixed:zstd:balanced",
                             "load_aware": "load" in comp,
                             "qos_max_error": 1e-3 if "lossy" in comp else 0.0})

    def effect(s):
        """Makespan multiplier: each missing component costs its gain.

        :param s: Setting dict (knobs + ``cell``).
        :return: Multiplier.
        """
        comp = component_set(pd.Series(s))
        base = 1.0 if s["cell"] == "montage" else 1.6
        return base * float(np.prod([1.0 + gain[c] for c in COMPONENTS if c not in comp]))

    results = dtlib.demo_results(settings, effect=effect, repeats=2)
    results[f"{dtlib.DT_PKG}.puts"] = 12000
    results[f"{dtlib.DT_PKG}.gets"] = 12000
    results[f"{dtlib.DT_PKG}.mean_obs_ctime_ms"] = 6.0
    return results


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir of e6_component_ablation")
    args = parser.parse_args(argv)
    results = make_demo() if args.demo else None
    if results is None:
        if not args.results:
            parser.error("--results is required without --demo")
        results = dtlib.load_results(args.results)
    df = label_runs(results)
    loo, cum = panel_table(df, "loo"), panel_table(df, "cum")
    dtlib.print_table(loo, "E6 leave-one-out (ms)")
    dtlib.print_table(cum, "E6 cumulative (ms)")
    extra = [c for c in (f"{dtlib.DT_PKG}.cache_hit_rate",
                         f"{dtlib.DT_PKG}.remote_read_bytes") if c in df.columns]
    if extra:
        dtlib.print_table(df.groupby(["workflow", "label"])[extra].mean().reset_index(),
                          "E6 cache hit rate / remote bytes")
    for path in plot(loo, cum, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

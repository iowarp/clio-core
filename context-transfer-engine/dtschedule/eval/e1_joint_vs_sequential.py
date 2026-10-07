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
E1 joint vs sequential decision (EVAL_PLAN section 2, reviewers A1 / B1).

Offline part: replay the joint run's candidate file.  A codec-first
decider (DeepPress / PACM style) picks the codec from data-only cost
(``pred_ctime_ms + pred_dtime_ms``), the joint decider by the ranker's
``cost_ms`` at the decision's placement; the replay counts how often the
two differ and the predicted cost penalty.  Placement (scenario, tier) is
decision-level, so tier-first cannot be replayed offline -- it is measured
online: the makespan of runs with ``decision_order`` in {joint,
codec_first, tier_first}, normalised to joint.

Inputs: ``--candidates '<trace_path>.cand.*.csv'`` of the joint run,
``--traces`` (optional, adds the flip rate per producer-load bin) and
``--results <exp dir>`` of the ``decision_order`` sweep.
Output: ``e1_joint_vs_sequential.svg/pdf`` + the flip table.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

import dtlib

ORDERS = ["joint", "codec_first", "tier_first"]
ORDER_LABELS = {"joint": "joint", "codec_first": "codec→tier",
                "tier_first": "tier→codec"}
ORDER_COLORS = {"joint": dtlib.CAT[0], "codec_first": dtlib.CAT[1],
                "tier_first": dtlib.CAT[2]}
LOAD_BINS = [0, 25, 75, 101]
LOAD_LABELS = ["idle", "50 %", "100 %"]


def decide_blob(grp: pd.DataFrame) -> dict:
    """Apply the joint and codec-first deciders to one blob's candidates.

    :param grp: ``reason == ok`` candidate rows of one (tag, blob).
    :return: ``joint`` / ``codec_first`` choices ``(lib, preset)`` and
        their ``cost_ms``.
    """
    joint = grp.loc[grp["cost_ms"].idxmin()]
    data_cost = grp["pred_ctime_ms"].fillna(0) + grp["pred_dtime_ms"].fillna(0)
    lib = data_cost.groupby(grp["lib"]).mean().idxmin()
    sub = grp[grp["lib"] == lib]
    pick = sub.loc[data_cost[sub.index].idxmin()]
    return {"joint": (joint["lib"], joint["preset"]),
            "joint_cost": float(joint["cost_ms"]),
            "codec_first": (pick["lib"], pick["preset"]),
            "codec_first_cost": float(pick["cost_ms"])}


def replay(cand: pd.DataFrame, trace: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Replay every blob and tabulate codec-first flips, overall and per load bin.

    :param cand: Candidate rows (:func:`dtlib.load_candidates`).
    :param trace: Decision rows for the producer-load bin (optional).
    :return: Rows ``group, codec_flip, cost_penalty, n_blobs`` where
        ``group`` is ``all`` plus one row per producer-load bin.
    """
    ok = dtlib.ok_candidates(cand)
    records = []
    for (tag, blob), grp in ok.groupby(["tag", "blob"], sort=False):
        d = decide_blob(grp)
        records.append({"tag": tag, "blob": blob,
                        "flip": d["codec_first"] != d["joint"],
                        "penalty": d["codec_first_cost"] / d["joint_cost"] - 1.0})
    if not records:
        return pd.DataFrame(columns=["group", "codec_flip", "cost_penalty", "n_blobs"])
    df = pd.DataFrame(records)
    df["group"] = "all"
    if trace is not None and not trace.empty:
        dec = trace[trace["kind"] == "decision"].drop_duplicates(["tag", "blob"])
        load = df.merge(dec[["tag", "blob", "producer_cpu"]], on=["tag", "blob"],
                        how="left")["producer_cpu"]
        binned = df.assign(group=pd.cut(load, LOAD_BINS, labels=LOAD_LABELS,
                                        right=False).astype(str))
        df = pd.concat([df, binned[binned["group"] != "nan"]], ignore_index=True)
    out = df.groupby("group").agg(codec_flip=("flip", "mean"),
                                  cost_penalty=("penalty", "mean"),
                                  n_blobs=("flip", "size"))
    order = [g for g in ["all"] + LOAD_LABELS if g in out.index]
    return out.reindex(order).reset_index()


def online_makespan(results: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Aggregate the sweep's makespan per ``decision_order``, normalised to joint.

    :param results: Rows of :func:`dtlib.load_results` or None.
    :return: Aggregated frame with ``makespan_ms_norm`` columns, or None.
    """
    if results is None or results.empty or "decision_order" not in results.columns:
        return None
    agg = dtlib.aggregate(results, ["decision_order"], "makespan_ms")
    agg = dtlib.normalize_to_baseline(agg, "makespan_ms",
                                      agg["decision_order"] == "joint")
    agg["order"] = pd.Categorical(agg["decision_order"], ORDERS, ordered=True)
    return agg.sort_values("order")


def _plot_offline(ax, flips: pd.DataFrame, demo: bool) -> None:
    """Codec-first flip rate and cost penalty per producer-load group.

    :param ax: Axes.
    :param flips: Output of :func:`replay`.
    :param demo: Synthetic-data flag.
    """
    x = np.arange(len(flips))
    width = 0.36
    series = [("codec_flip", "codec differs", dtlib.CAT[1], None),
              ("cost_penalty", "pred. cost penalty", dtlib.CAT[3], dtlib.HATCH)]
    for k, (col, label, color, hatch) in enumerate(series):
        vals = flips[col].to_numpy() * 100.0
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=label,
                      **dtlib.bar_kwargs(color, hatch))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=6, color=dtlib.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(flips["group"], fontsize=6.5)
    ax.set_xlabel("producer load")
    ax.set_ylabel("codec→tier vs joint (%)")
    ax.set_ylim(0, max(5.0, float(flips[["codec_flip", "cost_penalty"]].max().max())
                       * 100.0 * 1.3))
    dtlib.legend_outside(ax, ncol=1)
    dtlib.demo_title(ax, "offline replay", demo)


def _plot_online(ax, online: Optional[pd.DataFrame], demo: bool) -> None:
    """Normalised makespan per decision order.

    :param ax: Axes.
    :param online: Output of :func:`online_makespan` or None.
    :param demo: Synthetic-data flag.
    """
    if online is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "no --results", ha="center", va="center",
                color=dtlib.MUTED, fontsize=7)
        return
    xs = np.arange(len(online))
    est = online["makespan_ms_norm"].to_numpy()
    err = np.vstack([est - online["makespan_ms_norm_lo"],
                     online["makespan_ms_norm_hi"] - est])
    colors = [ORDER_COLORS.get(o, dtlib.OTHER) for o in online["decision_order"]]
    ax.bar(xs, est, 0.6, color=colors, edgecolor=dtlib.SURFACE, linewidth=0.8)
    ax.errorbar(xs, est, yerr=np.clip(err, 0, None), fmt="none",
                ecolor=dtlib.INK2, elinewidth=0.6, capsize=1.5)
    ax.axhline(1.0, color=dtlib.AXIS, linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([ORDER_LABELS.get(o, o) for o in online["decision_order"]],
                       rotation=20)
    ax.set_ylabel("makespan / joint")
    ax.set_ylim(min(0.9, float(np.nanmin(est)) * 0.95),
                max(1.1, float(np.nanmax(online["makespan_ms_norm_hi"])) * 1.05))
    dtlib.demo_title(ax, "online", demo)


def plot(flips: pd.DataFrame, online: Optional[pd.DataFrame], demo: bool,
         out_dir: str) -> List[str]:
    """Draw flip-rate bars (left) and normalised makespan (right).

    :param flips: Output of :func:`replay`.
    :param online: Output of :func:`online_makespan` or None.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    fig, (ax1, ax2) = dtlib.figure(1, 2, height=1.9, gridspec_kw={"wspace": 0.55})
    _plot_offline(ax1, flips, demo)
    _plot_online(ax2, online, demo)
    return dtlib.savefig(fig, "e1_joint_vs_sequential", out_dir)


def make_demo():
    """Synthesise candidates, a trace and a decision_order sweep (``--demo``).

    :return: ``(candidates, trace, results)``.
    """
    rng = np.random.default_rng(1)
    n_blobs = 120
    rows = []
    trace = dtlib.demo_trace(n_blobs, seed=2)
    dec = trace["kind"] == "decision"
    trace.loc[dec, "blob"] = [f"b{b}" for b in range(n_blobs)]
    trace.loc[dec, "tag"] = "t"
    loads = rng.choice([0, 50, 100], n_blobs)
    trace.loc[dec, "producer_cpu"] = loads
    for b in range(n_blobs):
        bw = 0.25 if rng.random() < 0.4 else 4.0  # slow tier when the fast one is full
        mult = 1 + loads[b] / 50.0
        for lib, speed, ratio in (("zstd", 1.0, 3.0), ("lz4", 0.3, 1.6), ("sz3", 2.5, 8.0)):
            for preset in (0, 1):
                ctime = speed * (1 + 0.3 * preset) * rng.lognormal(0, 0.1)
                rows.append({"tag": "t", "blob": f"b{b}", "lib": lib, "preset": preset,
                             "pred_ctime_ms": ctime, "pred_dtime_ms": speed * 0.4,
                             "pred_ratio": ratio * (1 + 0.1 * preset),
                             "cost_ms": ctime * mult + (1.0 / ratio) / bw
                             + rng.normal(0, 0.05), "reason": "ok"})
        rows.append({"tag": "t", "blob": f"b{b}", "lib": "zfp", "preset": 0,
                     "pred_ctime_ms": np.nan, "pred_dtime_ms": np.nan,
                     "pred_ratio": np.nan, "cost_ms": np.nan, "reason": "qos_error_bound"})
    cand = pd.DataFrame(rows)
    eff = {"joint": 1.0, "codec_first": 1.12, "tier_first": 1.07}
    results = dtlib.demo_results([{"decision_order": o} for o in ORDERS],
                                 effect=lambda s: eff[s["decision_order"]])
    return cand, trace, results


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir of the decision_order sweep")
    parser.add_argument("--candidates", help="glob of *.cand.*.csv of the joint run")
    parser.add_argument("--traces", help="glob of the joint run's decision traces")
    args = parser.parse_args(argv)
    if args.demo:
        cand, trace, results = make_demo()
    else:
        if not args.candidates and not args.results:
            parser.error("--candidates and/or --results is required without --demo")
        cand = dtlib.load_candidates(args.candidates) if args.candidates else pd.DataFrame(
            columns=dtlib.CANDIDATE_COLUMNS)
        trace = dtlib.load_traces(args.traces) if args.traces else None
        results = dtlib.load_results(args.results) if args.results else None
    flips = replay(cand, trace)
    online = online_makespan(results)
    dtlib.print_table(flips, "E1 codec-first vs joint (offline replay)")
    if online is not None:
        dtlib.print_table(online[["decision_order", "makespan_ms", "makespan_ms_norm",
                                  "makespan_ms_norm_lo", "makespan_ms_norm_hi", "n"]],
                          "E1 online makespan (normalised to joint)")
    for path in plot(flips, online, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

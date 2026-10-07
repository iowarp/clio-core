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
E10 load-response timeline (EVAL_PLAN section 2, reviewer C6).

Four aligned rows over wall time: producer and consumer CPU % (the trace's
``producer_cpu`` / ``consumer_cpu`` columns, i.e. what dtschedule saw),
chosen scenario (compression location), chosen codec, and per-write latency
(``obs_ctime_ms + store_ms``).  Vertical lines mark the injected load steps
(``load_steps.sh``: consumer at 120 s, release at 240 s, producer at 300 s).

Inputs: ``--traces '<trace_path>.*.csv'`` of the E10 run (or ``--results`` of
the e10 sweep, whose single run's traces are used).  ``--steps`` overrides
the step times and labels (``t:label,...``).  Output: ``e10_load_timeline.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

import dtlib

DEFAULT_STEPS = "120:consumer +load,240:consumer release,300:producer +load"


def parse_steps(spec: str) -> List[Tuple[float, str]]:
    """Parse ``t:label,t:label`` into a list.

    :param spec: Step spec.
    :return: ``[(t_s, label), ...]``.
    """
    steps = []
    for item in spec.split(","):
        if not item.strip():
            continue
        t, _, label = item.partition(":")
        steps.append((float(t), label.strip() or f"{float(t):g} s"))
    return steps


def _mark_steps(ax, steps: List[Tuple[float, str]], label: bool) -> None:
    """Draw the load-step markers on one row.

    :param ax: Axes.
    :param steps: Output of :func:`parse_steps`.
    :param label: Whether to write the step labels (top row only).
    """
    for k, (t, text) in enumerate(steps):
        ax.axvline(t, color=dtlib.MUTED, linewidth=0.6, linestyle=":")
        if label:  # alternate heights so neighbouring labels do not collide
            ax.text(t + 3, 0.97 - 0.2 * (k % 2), text,
                    transform=ax.get_xaxis_transform(), ha="left", va="top",
                    fontsize=5.5, color=dtlib.INK2,
                    bbox={"facecolor": dtlib.SURFACE, "edgecolor": "none", "pad": 0.5})


def _row_cpu(ax, dec: pd.DataFrame) -> None:
    """Producer / consumer CPU row.

    :param ax: Axes.
    :param dec: Decision rows.
    """
    have = dec[["producer_cpu", "consumer_cpu"]].notna().any()
    if not have.any():
        ax.text(0.5, 0.5, "trace has no producer_cpu / consumer_cpu samples",
                transform=ax.transAxes, ha="center", va="center", fontsize=6.5,
                color=dtlib.MUTED)
    for col, label, color in (("producer_cpu", "producer", dtlib.CAT[0]),
                              ("consumer_cpu", "consumer", dtlib.CAT[1])):
        if have[col]:
            ax.plot(dec["t_s"], dec[col], color=color, label=label, linewidth=1.2)
    ax.set_ylabel("CPU (%)")
    ax.set_ylim(0, 105)
    dtlib.legend_outside(ax, ncol=2)


def _row_categorical(ax, dec: pd.DataFrame, col: str, order: List[str],
                     colors, ylabel: str) -> None:
    """Scatter of a categorical choice over time (scenario or codec).

    :param ax: Axes.
    :param dec: Decision rows.
    :param col: Column to plot.
    :param order: Category order on the y axis (bottom to top).
    :param colors: ``category -> colour`` mapping or callable.
    :param ylabel: Y label.
    """
    vals = dec[col].astype(str).replace("nan", "")
    cats = [c for c in order if c in set(vals)] + sorted(set(vals) - set(order) - {""})
    y = vals.map({c: i for i, c in enumerate(cats)})
    for i, cat in enumerate(cats):
        sel = y == i
        color = colors(cat) if callable(colors) else colors.get(cat, dtlib.OTHER)
        ax.scatter(dec.loc[sel, "t_s"], y[sel], s=5, color=color, linewidths=0)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=6)
    ax.set_ylim(-0.6, max(len(cats) - 0.4, 0.6))
    ax.set_ylabel(ylabel)
    ax.grid(False)


def plot(trace: pd.DataFrame, steps: List[Tuple[float, str]], demo: bool,
         out_dir: str) -> List[str]:
    """Draw the four aligned rows.

    :param trace: Trace rows (:func:`dtlib.load_traces`).
    :param steps: Load steps.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    dec = trace[trace["kind"] == "decision"].sort_values("t_s")
    dec = dec.assign(scenario=dec["chosen_scenario"].map(
        lambda v: f"{int(v)}" if pd.notna(v) else ""),
        latency=dec["obs_ctime_ms"].fillna(0) + dec["store_ms"].fillna(0))
    fig, axes = dtlib.figure(4, 1, height=4.6, sharex=True,
                             gridspec_kw={"hspace": 0.22,
                                          "height_ratios": [1.1, 0.8, 1.0, 1.0]})
    _row_cpu(axes[0], dec)
    _row_categorical(axes[1], dec, "scenario", ["1", "2", "3"],
                     dtlib.SCENARIO_COLORS, "scenario")
    axes[1].set_yticklabels([{"1": "S1 @P", "2": "S2 @P", "3": "S3 @C"}.get(t.get_text(), t.get_text())
                             for t in axes[1].get_yticklabels()], fontsize=6)
    codecs = list(dtlib.CODEC_COLORS)
    _row_categorical(axes[2], dec, "chosen_lib", codecs, dtlib.codec_color, "codec")
    axes[3].plot(dec["t_s"], dec["latency"], color=dtlib.CAT[0], linewidth=0.6)
    axes[3].set_ylabel("write latency (ms)")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("time (s)")
    for k, ax in enumerate(axes):
        _mark_steps(ax, steps, label=(k == 0))
    dtlib.demo_title(axes[0], "decisions follow injected load steps", demo)
    return dtlib.savefig(fig, "e10_load_timeline", out_dir)


def make_demo() -> pd.DataFrame:
    """Synthesise a 10-minute trace reacting to load steps (``--demo`` only).

    :return: Trace rows.
    """
    trace = dtlib.demo_trace(900, seed=7, duration_s=600.0)
    dec = trace["kind"] == "decision"
    t = trace.loc[dec, "t_s"].to_numpy()
    rng = np.random.default_rng(8)
    cons = np.where((t > 120) & (t < 240), 95, 8) + rng.normal(0, 3, t.size)
    prod = np.where(t > 300, 90, 25) + rng.normal(0, 3, t.size)
    trace.loc[dec, "consumer_cpu"] = np.clip(cons, 0, 100)
    trace.loc[dec, "producer_cpu"] = np.clip(prod, 0, 100)
    scen = np.where(t > 300, "3", np.where((t > 120) & (t < 240), "1", "2"))
    flip = rng.random(t.size) < 0.1
    scen[flip] = rng.choice(["1", "2", "3"], flip.sum())
    trace.loc[dec, "chosen_scenario"] = scen.astype(int)
    lib = np.where(t > 300, "lz4", "zstd").astype(object)
    lib[flip] = rng.choice(["zstd", "lz4", "sz3"], flip.sum())
    trace.loc[dec, "chosen_lib"] = lib
    base = trace.loc[dec, "obs_ctime_ms"].to_numpy()
    trace.loc[dec, "obs_ctime_ms"] = base * np.where(t > 300, 0.5, 1.0) * np.where(
        (t > 120) & (t < 240), 1.4, 1.0)
    return trace


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir of the e10 run (traces/ used)")
    parser.add_argument("--traces", help="glob of the E10 decision trace CSVs")
    parser.add_argument("--steps", default=DEFAULT_STEPS,
                        help="load steps as t_s:label,... (default: EVAL_PLAN E10)")
    args = parser.parse_args(argv)
    if args.demo:
        trace = make_demo()
    elif args.traces:
        trace = dtlib.load_traces(args.traces)
    elif args.results:
        results = dtlib.load_results(args.results)
        globs = [g for g in results["trace_glob"] if g]
        trace = dtlib.load_traces(globs)
    else:
        parser.error("--traces or --results is required without --demo")
    if trace.empty:
        parser.error("no trace rows found")
    dec = trace[trace["kind"] == "decision"]
    dtlib.print_table(dec.groupby(pd.cut(dec["t_s"], 10), observed=True)
                      .agg(n=("blob", "size"), producer_cpu=("producer_cpu", "mean"),
                           consumer_cpu=("consumer_cpu", "mean"),
                           s3_share=("chosen_scenario", lambda s: (s == 3).mean()))
                      .reset_index(), "E10 per-window summary")
    for path in plot(trace, parse_steps(args.steps), args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

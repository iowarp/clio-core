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
E8 predictor ablation under drift (EVAL_PLAN section 2, reviewers C4 / C2).

Gray-Scott, 40 timesteps, one run per predictor (``ccm`` in {qtable, ema,
xgboost, oracle}).  Per timestep the figure shows the MAPE of the
predicted compression time against the observed one (from the trace's
``pred_ctime_ms`` / ``obs_ctime_ms``) and the regret: observed cost
(compress + store + decompress) of the predictor's choices relative to the
oracle run's choices at the same timestep.  When the oracle run's
candidate file is given, regret is instead computed per blob against the
cheapest measured candidate.

Inputs: ``--results <exp dir>`` whose runs carry ``ccm`` and per-run traces
(``<exp>/traces/<run>/*.csv``), or ``--traces 'pattern with {ccm}'`` (e.g.
``${HOME}/jarvis-runs/e8_{ccm}/dtschedule_trace.*.csv``) plus optional
``--candidates`` of the oracle run.  Output: ``e8_predictor_drift.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import dtlib

PREDICTORS = ["qtable", "ema", "xgboost", "oracle"]
MARKERS = {"qtable": "o", "ema": "s", "xgboost": "^", "oracle": "D"}


def per_timestep(trace: pd.DataFrame) -> pd.DataFrame:
    """Reduce one predictor's trace to per-timestep MAPE and observed cost.

    :param trace: Decision + decompress rows of one run.
    :return: Rows ``step, mape_ctime, mape_ratio, obs_cost_ms, n``.
    """
    dec = trace[trace["kind"] == "decision"].copy()
    dec["step"] = dtlib.timestep_index(dec)
    dtime = (trace[trace["kind"] == "decompress"]
             .groupby(["tag", "blob"])["obs_dtime_ms"].median().rename("dtime"))
    dec = dec.merge(dtime.reset_index(), on=["tag", "blob"], how="left")
    dec["ape_ctime"] = ((dec["pred_ctime_ms"] - dec["obs_ctime_ms"]).abs()
                        / dec["obs_ctime_ms"].replace(0, np.nan))
    dec["ape_ratio"] = ((dec["pred_ratio"] - dec["obs_ratio"]).abs()
                        / dec["obs_ratio"].replace(0, np.nan))
    dec["obs_cost"] = (dec["obs_ctime_ms"].fillna(0) + dec["store_ms"].fillna(0)
                       + dec["dtime"].fillna(0))
    out = dec.groupby("step").agg(mape_ctime=("ape_ctime", "mean"),
                                  mape_ratio=("ape_ratio", "mean"),
                                  obs_cost_ms=("obs_cost", "sum"),
                                  n=("obs_cost", "size")).reset_index()
    return out


def regret_from_candidates(trace: pd.DataFrame, cand: pd.DataFrame) -> pd.Series:
    """Per-timestep regret against the cheapest measured candidate per blob.

    Under ``ccm: oracle`` every candidate is run, so the oracle candidate
    file's ``cost_ms`` (``reason == ok``) is the measured cost.

    :param trace: The predictor's decision rows.
    :param cand: Oracle candidate rows.
    :return: Series ``step -> regret`` (relative cost excess, mean per step).
    """
    dec = trace[trace["kind"] == "decision"].copy()
    dec["step"] = dtlib.timestep_index(dec)
    cand = dtlib.ok_candidates(cand)
    cand["obs_cost"] = cand["cost_ms"]
    best = cand.groupby(["tag", "blob"])["obs_cost"].min().rename("best")
    chosen = cand.merge(dec[["tag", "blob", "chosen_lib", "chosen_preset", "step"]],
                        left_on=["tag", "blob", "lib", "preset"],
                        right_on=["tag", "blob", "chosen_lib", "chosen_preset"])
    chosen = chosen.merge(best.reset_index(), on=["tag", "blob"])
    chosen["regret"] = chosen["obs_cost"] / chosen["best"].replace(0, np.nan) - 1.0
    return chosen.groupby("step")["regret"].mean()


def build_table(traces: Dict[str, pd.DataFrame],
                cand: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Combine all predictors into one long table with regret.

    :param traces: ``{predictor: trace}``.
    :param cand: Oracle candidates or None.
    :return: Rows ``predictor, step, mape_ctime, mape_ratio, regret``.
    """
    parts = []
    for pred, trace in traces.items():
        tab = per_timestep(trace)
        tab["predictor"] = pred
        parts.append(tab)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["predictor", "step", "mape_ctime", "mape_ratio", "obs_cost_ms"])
    if cand is not None and not cand.empty:
        reg = pd.concat([regret_from_candidates(t, cand).rename("regret").reset_index()
                         .assign(predictor=p) for p, t in traces.items()])
        df = df.merge(reg, on=["predictor", "step"], how="left")
    elif "oracle" in traces:
        base = df[df["predictor"] == "oracle"].set_index("step")["obs_cost_ms"]
        df["regret"] = df["obs_cost_ms"] / df["step"].map(base).replace(0, np.nan) - 1.0
    else:
        df["regret"] = np.nan
    return df


def plot(df: pd.DataFrame, demo: bool, out_dir: str) -> List[str]:
    """Draw MAPE (top) and regret (bottom) per timestep.

    :param df: Output of :func:`build_table`.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    fig, (ax1, ax2) = dtlib.figure(2, 1, height=3.4, sharex=True,
                                   gridspec_kw={"hspace": 0.25})
    preds = [p for p in PREDICTORS if p in set(df["predictor"])]
    preds += sorted(set(df["predictor"]) - set(preds))
    for pred in preds:
        sub = df[df["predictor"] == pred].sort_values("step")
        color = dtlib.PREDICTOR_COLORS.get(pred, dtlib.OTHER)
        every = max(1, len(sub) // 8)
        ax1.plot(sub["step"], sub["mape_ctime"] * 100.0, color=color, label=pred,
                 marker=MARKERS.get(pred, "o"), markevery=every)
        ax2.plot(sub["step"], sub["regret"] * 100.0, color=color, label=pred,
                 marker=MARKERS.get(pred, "o"), markevery=every)
    ax1.set_ylabel("MAPE of pred. compress\ntime (%)")
    ax1.set_ylim(0, None)
    ax2.set_ylabel("regret vs oracle (%)")
    ax2.axhline(0, color=dtlib.AXIS, linewidth=0.6)
    ax2.set_xlabel("Gray-Scott timestep")
    dtlib.legend_outside(ax1, ncol=min(4, len(preds)))
    dtlib.demo_title(ax1, "predictor accuracy and regret under drift", demo)
    return dtlib.savefig(fig, "e8_predictor_drift", out_dir)


def make_demo() -> Dict[str, pd.DataFrame]:
    """Synthesise one drifting trace per predictor (``--demo`` only).

    :return: ``{predictor: trace}``.
    """
    base = dtlib.demo_trace(320, seed=0)
    dec = base["kind"] == "decision"
    step = dtlib.timestep_index(base[dec]).to_numpy()
    drift = 1.0 + 0.6 * (step > 20)
    true_cost = base.loc[dec, "obs_ctime_ms"].to_numpy() * drift
    traces = {}
    for k, pred in enumerate(PREDICTORS):
        trace = base.copy()
        err = {"qtable": 0.12, "ema": 0.30, "xgboost": 0.18, "oracle": 0.0}[pred]
        rng = np.random.default_rng(10 + k)
        # Unadapted predictors keep predicting the pre-drift cost.
        adapted = {"oracle": step >= 0, "qtable": step > 28,
                   "xgboost": step > 32, "ema": step > 24}[pred]
        pred_cost = np.where(adapted, true_cost, true_cost / drift)
        trace.loc[dec, "pred_ctime_ms"] = pred_cost * rng.lognormal(0, err, step.size)
        # A worse predictor picks worse codecs: its observed cost is higher.
        penalty = 1.0 + 0.8 * err + 0.5 * (~adapted & (drift > 1.0))
        trace.loc[dec, "obs_ctime_ms"] = true_cost * penalty
        traces[pred] = trace
    return traces


def load_inputs(args) -> Dict[str, pd.DataFrame]:
    """Resolve per-predictor traces from ``--results`` or ``--traces``.

    :param args: Parsed arguments.
    :return: ``{predictor: trace}``.
    """
    traces: Dict[str, pd.DataFrame] = {}
    if args.results:
        results = dtlib.load_results(args.results)
        for _, row in results.iterrows():
            if row["trace_glob"] and "ccm" in row.index:
                pred = str(row["ccm"]).split(":")[0]
                part = dtlib.load_traces(row["trace_glob"])
                traces[pred] = pd.concat([traces.get(pred, pd.DataFrame()), part],
                                         ignore_index=True)
    if args.traces:
        for pred in PREDICTORS:
            part = dtlib.load_traces(args.traces.replace("{ccm}", pred))
            if not part.empty:
                traces[pred] = part
    return traces


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir of the ccm sweep (with traces/)")
    parser.add_argument("--traces", help="trace glob with a {ccm} placeholder")
    parser.add_argument("--candidates", help="oracle run's *.cand.csv glob")
    args = parser.parse_args(argv)
    if args.demo:
        traces, cand = make_demo(), None
    else:
        traces = load_inputs(args)
        if not traces:
            parser.error("no traces found; give --results or --traces")
        cand = dtlib.load_candidates(args.candidates) if args.candidates else None
    df = build_table(traces, cand)
    summary = df.groupby("predictor")[["mape_ctime", "mape_ratio", "regret"]].mean()
    dtlib.print_table(summary.reset_index(), "E8 mean over timesteps")
    for path in plot(df, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

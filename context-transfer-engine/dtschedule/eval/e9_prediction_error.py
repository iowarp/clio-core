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
E9 prediction-error sensitivity (EVAL_PLAN section 2, reviewer C5).

Multiplicative log-normal noise on the predicted ratio
(``ratio_noise_sigma`` in {0, 0.1, 0.25, 0.5, 1.0}) and the ``oracle_ratio``
substitute.  Left: decision-flip rate against the sigma = 0 run (same
(tag, blob): different codec / preset / tier), from per-run traces; without
traces the flip rate falls back to half the L1 distance between the codec
histograms (``dtschedule.lib.*``).  Right: makespan normalised to sigma = 0
with bootstrap intervals; the oracle-ratio run is the dashed reference.

Inputs: ``--results <exp dir>`` of the ``ratio_noise_sigma`` sweep (runs
with ``ccm: oracle_ratio`` are the oracle reference).
Output: ``e9_prediction_error.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import dtlib

CHOICE = ["chosen_lib", "chosen_preset", "chosen_tier"]


def label_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Add ``sigma`` (float) and ``is_oracle`` columns.

    :param results: Rows of :func:`dtlib.load_results`.
    :return: Copy with both columns.
    """
    df = results.copy()
    ccm = df.get("ccm", pd.Series("", index=df.index)).astype(str).str.lower()
    df["is_oracle"] = ccm.str.contains("oracle")
    df["sigma"] = pd.to_numeric(df.get("ratio_noise_sigma", 0.0), errors="coerce").fillna(0.0)
    return df


def flip_rate_traces(base: pd.DataFrame, other: pd.DataFrame) -> float:
    """Fraction of blobs whose choice differs between two traces.

    :param base: sigma = 0 trace.
    :param other: Noisy trace.
    :return: Flip fraction over blobs present in both (NaN when none).
    """
    cols = ["tag", "blob"] + CHOICE
    b = base[base["kind"] == "decision"][cols].drop_duplicates(["tag", "blob"])
    o = other[other["kind"] == "decision"][cols].drop_duplicates(["tag", "blob"])
    m = b.merge(o, on=["tag", "blob"], suffixes=("_b", "_o"))
    if m.empty:
        return np.nan
    diff = np.zeros(len(m), dtype=bool)
    for col in CHOICE:
        diff |= (m[f"{col}_b"].astype(str) != m[f"{col}_o"].astype(str)).to_numpy()
    return float(diff.mean())


def flip_rate_stats(base: pd.Series, other: pd.Series) -> float:
    """Codec-histogram distance as a flip-rate proxy (no traces).

    :param base: sigma = 0 result row.
    :param other: Noisy result row.
    :return: ``0.5 * L1`` distance between normalised codec histograms.
    """
    hb = dtlib.stat_prefix(base, f"{dtlib.DT_PKG}.lib.")
    ho = dtlib.stat_prefix(other, f"{dtlib.DT_PKG}.lib.")
    if not hb or not ho:
        return np.nan
    libs = set(hb) | set(ho)
    tb, to = sum(hb.values()), sum(ho.values())
    return 0.5 * sum(abs(hb.get(l, 0) / tb - ho.get(l, 0) / to) for l in libs)


def flip_table(df: pd.DataFrame, traces: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flip rate per run against the first sigma = 0 run.

    :param df: Labelled result rows.
    :param traces: ``{run: trace}`` (possibly empty).
    :return: Rows ``sigma, is_oracle, flip`` (one per run) with ``source``.
    """
    base_rows = df[(df["sigma"] == 0) & ~df["is_oracle"]]
    if base_rows.empty:
        return pd.DataFrame(columns=["sigma", "is_oracle", "flip", "source"])
    base = base_rows.iloc[0]
    base_trace = traces.get(base["run"])
    rows = []
    for _, row in df.iterrows():
        trace = traces.get(row["run"])
        if base_trace is not None and trace is not None and not trace.empty:
            flip, src = flip_rate_traces(base_trace, trace), "trace"
        else:
            flip, src = flip_rate_stats(base, row), "codec-hist"
        rows.append({"sigma": row["sigma"], "is_oracle": row["is_oracle"],
                     "flip": flip, "source": src, "run": row["run"]})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, flips: pd.DataFrame, demo: bool, out_dir: str) -> List[str]:
    """Flip rate (left) and normalised makespan (right) versus sigma.

    :param df: Labelled result rows.
    :param flips: Output of :func:`flip_table`.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    noisy = df[~df["is_oracle"]]
    mk = dtlib.aggregate(noisy, ["sigma"], "makespan_ms").sort_values("sigma")
    mk = dtlib.normalize_to_baseline(mk, "makespan_ms", mk["sigma"] == 0)
    fl = (flips[~flips["is_oracle"]].groupby("sigma")["flip"].mean().reset_index()
          .sort_values("sigma"))
    fig, (ax1, ax2) = dtlib.figure(1, 2, height=1.8, gridspec_kw={"wspace": 0.6})
    xs = np.arange(len(fl))
    ax1.plot(xs, fl["flip"] * 100.0, color=dtlib.CAT[0], marker="o")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{s:g}" for s in fl["sigma"]])
    ax1.set_ylabel("decisions flipped vs σ=0 (%)")
    ax1.set_ylim(0, None)
    ax1.set_xlabel("ratio noise σ")
    src = ",".join(sorted(set(flips["source"]))) if len(flips) else "n/a"
    dtlib.demo_title(ax1, f"flip rate ({src})", demo)
    xs = np.arange(len(mk))
    est = mk["makespan_ms_norm"].to_numpy()
    err = np.vstack([est - mk["makespan_ms_norm_lo"], mk["makespan_ms_norm_hi"] - est])
    ax2.errorbar(xs, est, yerr=np.clip(err, 0, None), color=dtlib.CAT[0], marker="o",
                 elinewidth=0.6, capsize=1.5, ecolor=dtlib.INK2, label="noisy ratio")
    oracle = df[df["is_oracle"]]
    if not oracle.empty:
        ref = oracle["makespan_ms"].mean() / noisy[noisy["sigma"] == 0]["makespan_ms"].mean()
        ax2.axhline(ref, color=dtlib.CAT[5], linestyle="--", linewidth=1.2,
                    label="oracle ratio")
        ax2.legend(loc="upper left", fontsize=6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{s:g}" for s in mk["sigma"]])
    ax2.set_xlabel("ratio noise σ")
    ax2.set_ylabel("makespan / σ=0")
    ax2.axhline(1.0, color=dtlib.AXIS, linewidth=0.6)
    dtlib.demo_title(ax2, "makespan", demo)
    return dtlib.savefig(fig, "e9_prediction_error", out_dir)


def make_demo():
    """Synthesise a sigma sweep with traces (``--demo`` only).

    :return: ``(results, traces)``.
    """
    sigmas = [0.0, 0.1, 0.25, 0.5, 1.0]
    settings = [{"ratio_noise_sigma": s, "ccm": "qtable"} for s in sigmas]
    settings.append({"ratio_noise_sigma": 0.0, "ccm": "oracle_ratio"})
    results = dtlib.demo_results(
        settings, effect=lambda s: 0.97 if "oracle" in s["ccm"]
        else 1.0 + 0.25 * s["ratio_noise_sigma"], repeats=3)
    base = dtlib.demo_trace(200, seed=0)
    traces = {}
    rng = np.random.default_rng(3)
    for _, row in results.iterrows():
        t = base.copy()
        dec = t["kind"] == "decision"
        p = min(0.9, 0.6 * row["ratio_noise_sigma"])
        flip = rng.random(dec.sum()) < p
        libs = t.loc[dec, "chosen_lib"].to_numpy().copy()
        libs[flip] = rng.choice(dtlib.DEMO_CODECS, flip.sum())
        t.loc[dec, "chosen_lib"] = libs
        traces[row["run"]] = t
    return results, traces


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir of the ratio_noise_sigma sweep")
    args = parser.parse_args(argv)
    if args.demo:
        results, traces = make_demo()
    else:
        if not args.results:
            parser.error("--results is required without --demo")
        results = dtlib.load_results(args.results)
        traces = {r["run"]: dtlib.load_traces(r["trace_glob"])
                  for _, r in results.iterrows() if r["trace_glob"]}
    df = label_runs(results)
    flips = flip_table(df, traces)
    dtlib.print_table(flips, "E9 flip rate per run")
    dtlib.print_table(dtlib.aggregate(df, ["sigma", "is_oracle"], "makespan_ms"),
                      "E9 makespan (ms)")
    for path in plot(df, flips, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

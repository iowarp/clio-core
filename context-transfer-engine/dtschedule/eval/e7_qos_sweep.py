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
E7 QoS sweep (EVAL_PLAN section 2, reviewers C3 / A3 / B5).

Columns: ``max_error`` in {lossless, 1e-2, 1e-3, 1e-4, 1e-5}, the "no QoS"
run (allowlist ``.*``, bound 1.0) and the per-stage run; two objectives
(performance, ratio).  Four stacked rows: makespan, bytes stored per
tier, codec histogram, and the downstream analysis error (L-inf of the
consumer's PDF against the lossless run) versus ``max_error``.

Inputs: ``--results <exp dir>`` of ``e7_qos_sweep.yaml``.  Bytes per tier and
codec counts come from each run's traces (``<exp>/traces/<run>/*.csv``) or,
without them, from the ``dtschedule.tier.*`` / ``dtschedule.lib.*`` stats
and ``bytes_in / mean_ratio``.  The downstream error is read from the stat
``analysis.linf_error`` (or a ``--errors CSV`` with columns ``run,linf_error``).
Output: ``e7_qos_sweep.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import dtlib

OBJECTIVES = ["performance", "ratio"]
OBJ_COLORS = {"performance": dtlib.CAT[0], "ratio": dtlib.CAT[1]}
OBJ_HATCH = {"performance": None, "ratio": dtlib.HATCH}
ERROR_STAT = "analysis.linf_error"


def column_label(row: pd.Series) -> str:
    """Name the QoS column a run belongs to.

    :param row: Result row with ``qos_max_error``, ``qos_lossy_allowlist``,
        ``qos_stages``.
    :return: ``lossless``, ``1e-3`` ..., ``no QoS`` or ``per-stage``.
    """
    stages = str(row.get("qos_stages", "") or "")
    if stages not in ("", "[]", "nan", "None"):
        return "per-stage"
    allow = str(row.get("qos_lossy_allowlist", "") or "")
    err = float(row.get("qos_max_error", 0.0) or 0.0)
    if allow.strip("[]'\" ") == ".*" or err >= 1.0:
        return "no QoS"
    if err <= 0.0:
        return "lossless"
    return f"{err:.0e}".replace("e-0", "e-")


def column_order(labels: List[str]) -> List[str]:
    """Order QoS columns: lossless, descending bounds, no QoS, per-stage.

    :param labels: Column labels present.
    :return: Ordered unique labels.
    """
    bounds = sorted({l for l in labels if l[0].isdigit()},
                    key=lambda s: -float(s))
    order = ["lossless"] + bounds + ["no QoS", "per-stage"]
    return [l for l in order if l in set(labels)] + sorted(set(labels) - set(order))


def per_run_tables(results: pd.DataFrame, traces: Dict[str, pd.DataFrame]):
    """Bytes per tier and codec counts per run.

    :param results: Labelled result rows.
    :param traces: ``{run: trace}``.
    :return: ``(bytes_df, codec_df)`` long tables keyed by (col, objective).
    """
    byte_rows, codec_rows = [], []
    for _, row in results.iterrows():
        key = {"col": row["col"], "objective": row["qos_objective"]}
        trace = traces.get(row["run"])
        if trace is not None and not trace.empty:
            dec = trace[trace["kind"] == "decision"]
            stored = dec["size"] / dec["obs_ratio"].fillna(1.0).clip(lower=1e-9)
            for tier, val in stored.groupby(dec["chosen_tier"].replace("", "ram")).sum().items():
                byte_rows.append({**key, "tier": tier, "bytes": val})
            for lib, cnt in dec["chosen_lib"].value_counts().items():
                codec_rows.append({**key, "lib": lib, "count": cnt})
            continue
        tiers = dtlib.stat_prefix(row, f"{dtlib.DT_PKG}.tier.") or {"all": 1.0}
        total = sum(tiers.values())
        bytes_in = float(row.get(f"{dtlib.DT_PKG}.bytes_in", np.nan))
        ratio = float(row.get(f"{dtlib.DT_PKG}.mean_ratio", np.nan))
        for tier, cnt in tiers.items():
            byte_rows.append({**key, "tier": tier, "bytes": bytes_in / ratio * cnt / total})
        for lib, cnt in dtlib.stat_prefix(row, f"{dtlib.DT_PKG}.lib.").items():
            codec_rows.append({**key, "lib": lib, "count": cnt})
    return (pd.DataFrame(byte_rows, columns=["col", "objective", "tier", "bytes"]),
            pd.DataFrame(codec_rows, columns=["col", "objective", "lib", "count"]))


def _grouped(ax, tab: pd.DataFrame, cols: List[str], value: str, ylabel: str,
             scale: float = 1.0, ci: bool = False) -> None:
    """Grouped bars per column, one bar per objective.

    :param ax: Axes.
    :param tab: Aggregated rows with ``col``, ``objective`` and ``value``.
    :param cols: Column order.
    :param value: Value column.
    :param ylabel: Y label.
    :param scale: Divide values by this.
    :param ci: Draw ``<value>_lo/_hi`` error bars.
    """
    objs = [o for o in OBJECTIVES if o in set(tab["objective"])]
    width = 0.8 / max(len(objs), 1)
    for k, obj in enumerate(objs):
        sub = tab[tab["objective"] == obj].set_index("col").reindex(cols)
        x = np.arange(len(cols)) + (k - (len(objs) - 1) / 2.0) * width
        est = sub[value].to_numpy() / scale
        ax.bar(x, est, width, label=obj, **dtlib.bar_kwargs(OBJ_COLORS[obj], OBJ_HATCH[obj]))
        if ci:
            err = np.vstack([est - sub[f"{value}_lo"] / scale, sub[f"{value}_hi"] / scale - est])
            ax.errorbar(x, est, yerr=np.clip(err, 0, None), fmt="none",
                        ecolor=dtlib.INK2, elinewidth=0.5, capsize=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=6, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)
    if len(objs) > 1:
        dtlib.legend_outside(ax, ncol=2)


def _codec_hist(ax, codec_df: pd.DataFrame, cols: List[str]) -> None:
    """100 % stacked codec share per column (objectives side by side).

    :param ax: Axes.
    :param codec_df: Long codec-count table.
    :param cols: Column order.
    """
    objs = [o for o in OBJECTIVES if o in set(codec_df["objective"])]
    width = 0.8 / max(len(objs), 1)
    libs = list(codec_df.groupby("lib")["count"].sum().sort_values(ascending=False).index)
    for k, obj in enumerate(objs):
        piv = (codec_df[codec_df["objective"] == obj]
               .pivot_table(index="col", columns="lib", values="count", aggfunc="sum")
               .reindex(cols).fillna(0.0))
        share = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        x = np.arange(len(cols)) + (k - (len(objs) - 1) / 2.0) * width
        bottom = np.zeros(len(cols))
        for lib in libs:
            vals = share[lib].to_numpy() if lib in share.columns else np.zeros(len(cols))
            ax.bar(x, vals, width, bottom=bottom, label=lib if k == 0 else None,
                   **dtlib.bar_kwargs(dtlib.codec_color(lib), OBJ_HATCH[obj]))
            bottom += vals
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=6, rotation=20, ha="right")
    ax.set_ylabel("codec share (%)")
    ax.set_ylim(0, 100)
    dtlib.legend_outside(ax, ncol=min(5, len(libs)))


def _bytes(ax, bytes_df: pd.DataFrame, cols: List[str]) -> None:
    """Stacked bytes per tier, objectives side by side.

    :param ax: Axes.
    :param bytes_df: Long bytes table.
    :param cols: Column order.
    """
    objs = [o for o in OBJECTIVES if o in set(bytes_df["objective"])]
    width = 0.8 / max(len(objs), 1)
    tiers = [t for t in dtlib.TIER_ORDER if t in set(bytes_df["tier"])]
    tiers += sorted(set(bytes_df["tier"]) - set(tiers))
    for k, obj in enumerate(objs):
        piv = (bytes_df[bytes_df["objective"] == obj]
               .pivot_table(index="col", columns="tier", values="bytes", aggfunc="mean")
               .reindex(cols).fillna(0.0))
        x = np.arange(len(cols)) + (k - (len(objs) - 1) / 2.0) * width
        bottom = np.zeros(len(cols))
        for tier in tiers:
            vals = piv[tier].to_numpy() / 2 ** 30 if tier in piv.columns else np.zeros(len(cols))
            ax.bar(x, vals, width, bottom=bottom, label=tier if k == 0 else None,
                   **dtlib.bar_kwargs(dtlib.TIER_COLORS.get(tier, dtlib.OTHER), OBJ_HATCH[obj]))
            bottom += vals
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=6, rotation=20, ha="right")
    ax.set_ylabel("bytes stored (GiB)")
    ax.set_ylim(0, None)
    dtlib.legend_outside(ax, ncol=min(4, len(tiers)))


def plot(results: pd.DataFrame, bytes_df: pd.DataFrame, codec_df: pd.DataFrame,
         demo: bool, out_dir: str) -> List[str]:
    """Draw the four-row QoS figure.

    :param results: Labelled rows (``col``, ``qos_objective``, ``linf_error``).
    :param bytes_df: Bytes per tier table.
    :param codec_df: Codec count table.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    cols = column_order(list(results["col"]))
    mk = dtlib.aggregate(results, ["col", "qos_objective"], "makespan_ms").rename(
        columns={"qos_objective": "objective"})
    fig, axes = dtlib.figure(4, 1, height=6.6, gridspec_kw={"hspace": 0.9})
    _grouped(axes[0], mk, cols, "makespan_ms", "makespan (s)", 1000.0, ci=True)
    dtlib.demo_title(axes[0], "QoS sweep: max_error x objective", demo)
    _bytes(axes[1], bytes_df, cols)
    _codec_hist(axes[2], codec_df, cols)
    err = results.dropna(subset=["linf_error"])
    if err.empty:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, f"no downstream error ({ERROR_STAT} / --errors)",
                     ha="center", va="center", color=dtlib.MUTED, fontsize=7)
    else:
        agg = err.groupby(["col", "qos_objective"])["linf_error"].mean().reset_index()
        agg = agg.rename(columns={"qos_objective": "objective"})
        _grouped(axes[3], agg, cols, "linf_error", "PDF L-inf error\nvs lossless")
        axes[3].set_yscale("symlog", linthresh=1e-6)
        axes[3].set_ylim(0, None)
    return dtlib.savefig(fig, "e7_qos_sweep", out_dir)


def load_errors(results: pd.DataFrame, path: str | None) -> pd.DataFrame:
    """Attach the downstream L-inf error per run.

    :param results: Result rows.
    :param path: Optional CSV ``run,linf_error``.
    :return: Rows with a ``linf_error`` column (NaN when unknown).
    """
    df = results.copy()
    df["linf_error"] = pd.to_numeric(df.get(ERROR_STAT, np.nan), errors="coerce")
    if path:
        ext = pd.read_csv(path).set_index("run")["linf_error"]
        df["linf_error"] = df["linf_error"].fillna(df["run"].map(ext))
    return df


def make_demo():
    """Synthesise a QoS sweep with traces (``--demo`` only).

    :return: ``(results, traces)``.
    """
    bounds = [0.0, 1e-2, 1e-3, 1e-4, 1e-5, 1.0, 1e-2]
    allow = ["float"] * 5 + [".*", "float"]
    stages = [""] * 6 + ["[{match: stage2, max_error: 0}]"]
    settings = [{"qos_max_error": b, "qos_lossy_allowlist": a, "qos_stages": s,
                 "qos_objective": o}
                for b, a, s in zip(bounds, allow, stages) for o in OBJECTIVES]

    def effect(s):
        """Makespan multiplier: looser bounds run faster, ratio objective slower.

        :param s: Setting dict.
        :return: Multiplier.
        """
        e = s["qos_max_error"]
        base = 1.0 if e == 0 else 0.75 + 0.05 * -np.log10(max(e, 1e-5)) / 5
        return base * (1.08 if s["qos_objective"] == "ratio" else 1.0)

    results = dtlib.demo_results(settings, effect=effect, repeats=2)
    results[ERROR_STAT] = [0.0 if r["qos_max_error"] == 0 else
                           min(r["qos_max_error"], 0.05) * 0.6 for _, r in results.iterrows()]
    traces = {}
    for i, row in results.iterrows():
        codecs = ["zstd", "lz4"] if row["qos_max_error"] == 0 else ["zstd", "sz3", "zfp"]
        traces[row["run"]] = dtlib.demo_trace(100, seed=i, codecs=codecs)
    return results, traces


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir of e7_qos_sweep")
    parser.add_argument("--errors", help="CSV run,linf_error with the downstream error")
    args = parser.parse_args(argv)
    if args.demo:
        results, traces = make_demo()
    else:
        if not args.results:
            parser.error("--results is required without --demo")
        results = dtlib.load_results(args.results)
        traces = {r["run"]: dtlib.load_traces(r["trace_glob"])
                  for _, r in results.iterrows() if r["trace_glob"]}
    results["col"] = [column_label(r) for _, r in results.iterrows()]
    results["qos_objective"] = results["qos_objective"].astype(str)
    results = load_errors(results, args.errors)
    bytes_df, codec_df = per_run_tables(results, traces)
    dtlib.print_table(dtlib.aggregate(results, ["col", "qos_objective"], "makespan_ms"),
                      "E7 makespan (ms)")
    dtlib.print_table(results.groupby(["col", "qos_objective"])["linf_error"].mean()
                      .reset_index(), "E7 downstream L-inf error")
    for path in plot(results, bytes_df, codec_df, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

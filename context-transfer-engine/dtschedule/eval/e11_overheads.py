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
E11 overheads at several chunk sizes (EVAL_PLAN section 2, reviewer C7).

Stacked per-phase latency per chunk size (64 KiB, 1 MiB, 16 MiB by
default): selection (the trace's ``select_ms`` column, SelectCodec wall
time; reported as not measured when absent), compression (``obs_ctime_ms``),
store (``store_ms``) and, as a separate bar, decompression (``obs_dtime_ms``
of the matching GetBlob rows).  Medians, with the p95 of the write total as
a whisker, so the "10 ms vs 800 us" question is answered per size.

Inputs: ``--traces '<trace_path>.*.csv'`` of a Gray-Scott run that wrote
every size (or ``--results`` with per-run traces).  ``--sizes`` picks the
bins (``dtlib.size_bin_label`` names).  Output: ``e11_overheads.svg/pdf``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

import dtlib

DEFAULT_SIZES = "64 KiB,1 MiB,16 MiB"
PHASES = [("select_ms", "selection", dtlib.CAT[3]),
          ("obs_ctime_ms", "compress", dtlib.CAT[0]),
          ("store_ms", "store", dtlib.CAT[2])]


def phase_table(trace: pd.DataFrame, sizes: List[str]) -> pd.DataFrame:
    """Median / p95 per phase and size bin.

    :param trace: Trace rows.
    :param sizes: Size-bin labels to keep (in order).
    :return: Rows ``size, phase, median_ms, p95_ms, n``; the write ``total``
        and ``decompress`` appear as phases too.
    """
    dec = trace[trace["kind"] == "decision"].copy()
    dec["size_bin"] = dec["size"].map(dtlib.size_bin_label)
    dtime = (trace[trace["kind"] == "decompress"]
             .groupby(["tag", "blob"])["obs_dtime_ms"].median().rename("decomp"))
    dec = dec.merge(dtime.reset_index(), on=["tag", "blob"], how="left")
    if "select_ms" not in dec.columns:
        dec["select_ms"] = np.nan
    dec["total"] = sum(dec[c].fillna(0) for c, _, _ in PHASES)
    rows = []
    for size in sizes:
        sub = dec[dec["size_bin"] == size]
        for col, name, _ in PHASES + [("total", "write total", None),
                                      ("decomp", "decompress", None)]:
            vals = sub[col].dropna()
            rows.append({"size": size, "phase": name,
                         "median_ms": vals.median() if len(vals) else np.nan,
                         "p95_ms": vals.quantile(0.95) if len(vals) else np.nan,
                         "n": int(len(vals))})
    return pd.DataFrame(rows)


def plot(tab: pd.DataFrame, sizes: List[str], demo: bool, out_dir: str) -> List[str]:
    """Horizontal stacked bars per size, decompression as a hatched bar.

    :param tab: Output of :func:`phase_table`.
    :param sizes: Size order.
    :param demo: Synthetic-data flag.
    :param out_dir: Figure directory.
    :return: Written paths.
    """
    fig, ax = dtlib.figure(height=0.55 * len(sizes) + 1.0)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    y = np.arange(len(sizes)) * 1.0
    h = 0.34
    left = np.zeros(len(sizes))
    piv = tab.pivot(index="size", columns="phase", values="median_ms").reindex(sizes)
    for col, name, color in PHASES:
        vals = piv[name].fillna(0).to_numpy()
        if not np.any(vals > 0):
            continue
        ax.barh(y + h / 2, vals, h, left=left, label=name, **dtlib.bar_kwargs(color))
        left += vals
    total = piv["write total"].to_numpy()
    p95 = tab[tab["phase"] == "write total"].set_index("size").reindex(sizes)["p95_ms"]
    ax.errorbar(total, y + h / 2, xerr=np.clip([np.zeros(len(sizes)), p95 - total], 0, None),
                fmt="none", ecolor=dtlib.INK2, elinewidth=0.6, capsize=1.5)
    for yi, tot in zip(y + h / 2, total):
        if np.isfinite(tot):
            ax.text(tot, yi, f" {tot:.2f} ms", va="center", ha="left", fontsize=5.5,
                    color=dtlib.INK2)
    dec = piv["decompress"].fillna(0).to_numpy()
    ax.barh(y - h / 2, dec, h, label="decompress (read)",
            **dtlib.bar_kwargs(dtlib.CAT[1], dtlib.HATCH))
    for yi, val in zip(y - h / 2, dec):
        if val > 0:
            ax.text(val, yi, f" {val:.2f} ms", va="center", ha="left", fontsize=5.5,
                    color=dtlib.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(sizes)
    ax.set_xlabel("latency (ms), median; whisker = p95 of write total")
    ax.set_xlim(0, float(np.nanmax(np.concatenate([p95.to_numpy(), dec]))) * 1.3)
    ax.invert_yaxis()
    dtlib.legend_outside(ax, ncol=4)
    note = "" if (piv.get("selection", pd.Series(dtype=float)).fillna(0) > 0).any() \
        else " (selection not in trace)"
    dtlib.demo_title(ax, "per-phase latency per chunk size" + note, demo)
    return dtlib.savefig(fig, "e11_overheads", out_dir)


def make_demo() -> pd.DataFrame:
    """Synthesise a trace with the three sizes (``select_ms`` included).

    :return: Trace rows.
    """
    return dtlib.demo_trace(600, seed=11)


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dtlib.add_common_args(parser, "results dir whose traces/ hold the run")
    parser.add_argument("--traces", help="glob of decision trace CSVs")
    parser.add_argument("--sizes", default=DEFAULT_SIZES,
                        help="comma-separated size bins (default 64 KiB,1 MiB,16 MiB)")
    args = parser.parse_args(argv)
    if args.demo:
        trace = make_demo()
    elif args.traces:
        trace = dtlib.load_traces(args.traces)
    elif args.results:
        results = dtlib.load_results(args.results)
        trace = dtlib.load_traces([g for g in results["trace_glob"] if g])
    else:
        parser.error("--traces or --results is required without --demo")
    if trace.empty:
        parser.error("no trace rows found")
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    tab = phase_table(trace, sizes)
    dtlib.print_table(tab, "E11 per-phase latency (ms)")
    for path in plot(tab, sizes, args.demo, dtlib.resolve_out(args)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

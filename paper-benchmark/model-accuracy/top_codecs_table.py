#!/usr/bin/env python3
"""Table of the top 3 compressors per workload: for each option's measured run (NeuroPress, the
cost oracle, the time oracle, HCompress), the three settings it used most, the share of the
chunks (= share of the data: every chunk is 4 MiB) each one compressed and the share of the run's
time spent on those chunks (each chunk's share of the measured write window + its timed reads' wall
time, phases.csv), plus how many distinct settings it used. The best static uses one setting for 100 %
of the chunks.

    top_codecs_table.py RUNS_CSV [--out PREFIX]

RUNS_CSV: a run table of final_check.py's schema (e.g. ipdps-results/option_a_1x1_runs.csv); each run's
choice per chunk is read from its folder: v2_measured.csv (one 'primary' row per compressed chunk)
and blobs.csv (a chunk with no row was stored raw: 'store'). Writes PREFIX.csv (one row per
workload, option and rank), PREFIX.pdf / .png (the table for the paper, booktabs style) and PREFIX.tex
(the same table in LaTeX).
"""
import argparse
import os
import textwrap

import matplotlib
import pandas as pd

import compare_parallel_runs as cp
import final_config as fc
import ground_truth as gt
from plot_workload_summary import GRID, INK, INK_2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "/mnt/nvme0/v2-work/runs"
OPTIONS = ("best static", "NeuroPress", "time oracle", "cost oracle", "HCompress", "XGBoost")


def choices(run):
    """@return the setting name per chunk of one run (raw-stored chunks: 'store')."""
    b = pd.read_csv(os.path.join(run, "blobs.csv"), usecols=["blob"])
    m = pd.read_csv(os.path.join(run, "v2_measured.csv"), usecols=["blob", "spec", "role"])
    m = m[m.role == "primary"].drop_duplicates("blob").set_index("blob").spec
    return b.blob.map(m).fillna("store")


def chunk_time(run):
    """@return each chunk's measured time in s, by chunk name: its share of the write window + the
    wall time of its timed reads (the untimed bit-exact check excluded). The per-chunk write wall
    times overlap (a chunk's I/O runs while the next chunk is prepared), so they are scaled to add up
    to the run's measured write window; the timed reads do not overlap and are taken as measured."""
    w, rd = gt.phases(run)
    x = cp.parse_proc(run)
    window_ms = (x["end_ns"] - x["start_ns"]) / 1e6
    write = w.set_index("chunk_id").wall_ms
    write = write * (window_ms / write.sum())
    t = write.add(rd.groupby("chunk_id").wall_ms.sum(), fill_value=0)
    return t / 1e3


def table(runs_csv, top=3):
    """@return one row per workload, option and rank (1..top, then 'other')."""
    r = pd.read_csv(runs_csv)
    rows = []
    for wl in fc.WORKLOADS:
        for opt in OPTIONS:
            x = r[(r.workload == wl) & (r.option == opt)]
            if x.empty:
                continue
            run = os.path.join(RUNS, x.run.iloc[0])
            b = pd.read_csv(os.path.join(run, "blobs.csv"), usecols=["blob"]).blob
            c = pd.Series(choices(run).to_numpy(), index=b)
            secs = chunk_time(run).reindex(c.index)
            share = c.value_counts(normalize=True) * 100
            tsum = secs.groupby(c).sum()
            tpct = tsum / secs.sum() * 100
            base = {"workload": wl, "option": opt, "settings_used": len(share), "chunks": len(c),
                    "run_chunk_time_s": secs.sum()}
            for k, (name, pct) in enumerate(share.head(top).items(), 1):
                rows.append({**base, "rank": k, "setting": name, "chunks_pct": pct, "time_pct": tpct[name],
                             "time_s": tsum[name]})
            if len(share) > top:
                rest = share.index[top:]
                rows.append({**base, "rank": "other", "setting": f"{len(share) - top} others",
                             "chunks_pct": share.iloc[top:].sum(), "time_pct": tpct[rest].sum(),
                             "time_s": tsum[rest].sum()})
    return pd.DataFrame(rows)


# the methods of the paper table (the best static is one setting: it goes into the workload label)
METHODS = ("NeuroPress", "time oracle", "HCompress", "XGBoost")
HEAD = {"NeuroPress": "NeuroPress", "time oracle": "Time oracle", "cost oracle": "Cost oracle", "HCompress": "HCompress",
        "XGBoost": "XGBoost"}
FAMILY_COLOR = {"ans": "#d1495b", "ndzip": "#2a7bd5", "spratio": "#2e8b57", "spspeed": "#8cc5a4", "lz4": "#e39b3a",
                "snappy": "#c9a227", "bitcomp": "#8f86d0", "cascaded": "#5b5a55", "store": "#c4c3be", "zstd": "#7a1f2c",
                "deflate": "#a07d5a", "gdeflate": "#a07d5a", "gpulz": "#3fa7a0"}


def short_name(spec, sep="\u00b7"):
    """@return a compact codec name for the paper: the codec, then '<sep>byte' / '<sep>bit' for a byte /
    bit shuffle, e.g. 'ans shuffle=byte' -> 'ans\u00b7byte'; 'store' -> 'raw' (stored uncompressed)."""
    if spec == "store":
        return "raw"
    parts = spec.split()
    name, tags = parts[0], []
    for p in parts[1:]:
        k, v = p.split("=", 1)
        if k == "shuffle":
            tags.append(v)
        elif k == "bitshuffle":
            tags.append("bitshuf")
        elif k == "algo":
            name += v
        elif k == "rle" and v == "1":
            tags.append("RLE")
        elif k == "type" and v == "float16":
            tags.append("fp16")
    return sep.join([name] + tags)


def _rule(ax, y, x1, lw):
    ax.plot([0, x1], [y, y], color=INK, lw=lw, solid_capstyle="butt")


def plot(t, out, statics):
    """The paper table as a figure (booktabs style): one block per workload, one column group per
    method with the codec, its share of the chunks and its share of the time; out without suffix
    (writes .pdf and .png). @param statics {workload: best static setting}"""
    wl_w, grp_w, row_h, gap = 1.2, 1.6, 0.165, 0.07
    wls = list(dict.fromkeys(t.workload))
    methods = [m for m in METHODS if m in set(t.option)]
    width = wl_w + grp_w * len(methods)
    height = 0.55 + len(wls) * (3 * row_h + gap) + 0.05
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(-height, 0)
    ax.axis("off")
    y = -0.04
    _rule(ax, y, width, 1.0)
    for j, m in enumerate(methods):
        x = wl_w + j * grp_w
        ax.text(x + grp_w / 2, y - 0.12, HEAD[m], ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.plot([x + 0.06, x + grp_w - 0.06], [y - 0.22] * 2, color=INK, lw=0.5)
        for xx, lab, ha in ((x + 0.06, "codec", "left"), (x + grp_w - 0.42, "chunks", "right"), (x + grp_w - 0.06, "time", "right")):
            ax.text(xx, y - 0.32, lab, ha=ha, va="center", fontsize=6.8, color=INK_2, style="italic")
    ax.text(0.04, y - 0.32, "workload", ha="left", va="center", fontsize=6.8, color=INK_2, style="italic")
    y -= 0.42
    _rule(ax, y, width, 0.6)
    for wi, wl in enumerate(wls):
        top = y - gap / 2 - row_h / 2
        ds, w, bw = fc.WORKLOADS[wl]
        ax.text(0.04, top, wl, ha="left", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(0.04, top - row_h, f"best static: {short_name(statics[wl])}", ha="left", va="center", fontsize=6.5,
                color=INK_2)
        ax.text(0.04, top - 2 * row_h, f"cost model {fc.model_name(w)}", ha="left", va="center", fontsize=6.5, color=INK_2)
        for j, m in enumerate(methods):
            x = wl_w + j * grp_w
            s = t[(t.workload == wl) & (t.option == m) & (t["rank"] != "other")]
            for i, (_, r) in enumerate(s.iterrows()):
                yy = top - i * row_h
                fam = r.setting.split()[0]
                ax.plot(x + 0.09, yy, "o", ms=3.6, color=FAMILY_COLOR.get(fam, INK_2), mec="none")
                ax.text(x + 0.16, yy, short_name(r.setting), ha="left", va="center", fontsize=7, color=INK)
                for xx, v in ((x + grp_w - 0.42, r.chunks_pct), (x + grp_w - 0.06, r.time_pct)):
                    ax.text(xx, yy, f"{v:.0f}%" if v >= 0.5 or v == 0 else "<1%", ha="right", va="center", fontsize=7,
                            color=INK, fontweight="bold" if i == 0 else "normal")
        y -= 3 * row_h + gap
        if wi < len(wls) - 1:
            ax.plot([0, width], [y, y], color=GRID, lw=0.6)
    _rule(ax, y, width, 1.0)
    ax.text(0.04, y - 0.06, textwrap.fill(
        "chunks: share of the chunks (= of the data) the selector compressed with the codec; time: share of the run's "
        "end-to-end time spent on those chunks. \u00b7byte / \u00b7bit: byte / bit shuffle before the codec; raw: "
        "stored uncompressed.", int(width * 19)), ha="left", va="top", fontsize=6, color=INK_2, linespacing=1.4)
    ax.set_ylim(y - 0.32, 0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def latex(t, out, statics):
    """The same table in LaTeX (booktabs, table*): the top 3 codecs of every method per workload."""
    methods = [m for m in METHODS if m in set(t.option)]
    cols = " ".join(["l"] + ["lrr"] * len(methods))
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{HEAD[m]}}}" for m in methods)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 3 * j}-{4 + 3 * j}}}" for j in range(len(methods)))
    sub = " & ".join(["Codec & Chunks & Time"] * len(methods))
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             rf"Workload & {head} \\", cmid, rf" & {sub} \\", r"\midrule"]
    pct = lambda v: f"{v:.0f}\\%" if v >= 0.5 or v == 0 else r"$<$1\%"
    for wi, wl in enumerate(dict.fromkeys(t.workload)):
        label = [rf"\textbf{{{wl}}}", rf"\scriptsize static: {short_name(statics[wl], r'$\cdot$')}",
                 rf"\scriptsize cost model {fc.model_name(fc.WORKLOADS[wl][1])}"]
        for i in range(3):
            cells = [label[i]]
            for m in methods:
                s = t[(t.workload == wl) & (t.option == m) & (t["rank"] != "other")]
                if i < len(s):
                    r = s.iloc[i]
                    cells += [short_name(r.setting, r"$\cdot$"), pct(r.chunks_pct), pct(r.time_pct)]
                else:
                    cells += ["", "", ""]
            lines.append(" & ".join(cells) + r" \\")
        if wi < t.workload.nunique() - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Top three codecs per workload for each selector, measured in Clio (1 process, 1 chunk in "
              r"flight, 1 write + 10 reads with k-means, page cache dropped before each read). Chunks: share of the "
              r"chunks (= share of the data; every chunk is 4\,MiB) compressed with the codec; Time: share of the run's "
              r"end-to-end time spent on those chunks. $\cdot$byte / $\cdot$bit: byte / bit shuffle before the codec; raw: "
              r"stored uncompressed. The best static codec compresses every chunk with one setting.}",
              r"\label{tab:top-codecs}", r"\end{table*}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_csv")
    ap.add_argument("--out", default=None, help="output prefix (default: next to RUNS_CSV, <name>_top_codecs)")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.runs_csv)[0].replace("_runs", "") + "_top_codecs"
    t = table(a.runs_csv)
    t.to_csv(out + ".csv", index=False)
    st = t[t.option == "best static"].set_index("workload").setting.to_dict()
    plot(t, out, st)
    latex(t, out + ".tex", st)
    print("wrote", out + ".csv", out + ".pdf", out + ".png", out + ".tex")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(t.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

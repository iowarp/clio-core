#!/usr/bin/env python3
"""Figure 8, panel (d): the same two per-chunk metrics for every model.

  plot_fig8_models.py --chunks CSV [--out DIR]

Panels (a) and (b) show NeuroPress's online model converging, one line per
workload. This panel asks the same question of the baselines: one row per
workload, one line per model, regret on the left and cost MAPE on the right.

Input is `model-accuracy/fig8_model_chunks.py`'s CSV
(workload, chunk_index, model, regret_pct, cost_mape_pct). Colours and model
order match `plot_accuracy.py`, so a reader moving between the two figures
follows the same model by the same colour.

AXES. Regret is linear to 10% and logarithmic above; cost MAPE is logarithmic
throughout. Neither is capped. Figure 8's own panels cap instead, which is
right when every line is within a factor of a few of the others -- here the
static baselines' cost MAPE runs to four digits, and a cap would draw them as
one flat line at the ceiling and hide that they differ by 3x.
"""
import argparse, os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_fig8 import (FS_AXIS, FS_TICK, FS_LEG, LINE_W, GRID, DEFAULT_BIN,   # noqa: E402
                       PREVIEW_DPI, PER_CHUNK_W, REGRET_LINTHRESH,
                       binned, set_fonts, ytick_step)

#: Legend order, colours (matching plot_accuracy.py's MODELS), line style and
#: draw order. Two models can pick identically for a whole workload -- XGBoost
#: and HCompress seed both collapse to bitcomp|q0|s0 on VPIC and LAMMPS -- so a
#: hidden line would otherwise read as a missing one. The static models are
#: dashed, and the deployed model is drawn last, on top of everything.
MODELS = [
    ("NeuroPress (online)",   "#17becf", "-",  5),
    ("NeuroPress (static)",   "#1f77b4", "--", 4),
    ("XGBoost",               "#ff7f0e", "--", 3),
    ("HCompress CCP (+fb)",   "#d62728", "-",  2),
    ("HCompress CCP (seed)",  "#8c564b", ":",  1),
]
WORKLOADS = [("vpic", "VPIC"), ("nyx", "Nyx"), ("lammps", "LAMMPS"),
             ("warpx", "WarpX"), ("ai", "AI")]
ROW_H = 1.45
SUMMARY_LAST_FRAC = 0.20
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "figures", "fig8", "full")


#: Both axes are symmetric-log: linear up to the threshold, logarithmic above.
#: A plain log axis cannot draw an exact zero, and a bin whose every chunk was
#: predicted perfectly is a real value, not a missing one.
MAPE_LINTHRESH = 1.0


def symlog_axis(ax, y, linthresh):
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.6)
    top = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else linthresh
    ax.set_ylim(0, max(linthresh, top * 1.4))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def draw(data, bin_n, stat, models, floor=None):
    rows = [(k, lbl) for k, lbl in WORKLOADS if k in data]
    fig, axes = plt.subplots(len(rows), 2, squeeze=False,
                             figsize=(PER_CHUNK_W, ROW_H * len(rows) + 0.7),
                             gridspec_kw=dict(hspace=0.70, wspace=0.22))
    panels = (("regret_pct", "Regret (%)"), ("cost_mape_pct", "Cost MAPE (%)"))
    for r, (k, lbl) in enumerate(rows):
        d = data[k]
        for c, (col, ylabel) in enumerate(panels):
            ax, allv = axes[r][c], []
            for name, colour, style, z in models:
                s = d[d.model == name]
                if s.empty:
                    continue
                x, y = binned(s, col, bin_n, None, stat)
                if len(x) == 0:
                    continue
                ax.plot(x, y, color=colour, linewidth=LINE_W, linestyle=style,
                        zorder=z, solid_capstyle="round")
                allv.append(y)
            allv = np.concatenate(allv) if allv else np.array([1.0])
            symlog_axis(ax, allv,
                        REGRET_LINTHRESH if col == "regret_pct" else MAPE_LINTHRESH)
            ax.set_xlim(0, max(1.0, float(d["chunk_index"].max())))
            ax.set_ylabel(ylabel, fontsize=FS_AXIS)
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(**GRID)
            ax.set_axisbelow(True)
            if r == len(rows) - 1:
                ax.set_xlabel("Chunk index", fontsize=FS_AXIS)
        axes[r][0].set_title(f"{lbl}  ({d.chunk_index.nunique()} chunks)",
                             loc="left", fontsize=FS_AXIS)
    handles = [Line2D([], [], color=c, linewidth=LINE_W, linestyle=st, label=n)
               for n, c, st, _ in models]
    # Anchored to the bottom ROW, not to the figure: the figure's bottom margin
    # scales with the row count, so a fixed figure-fraction leaves a growing gap.
    h = fig.get_figheight()
    y0 = min(ax.get_position().y0 for ax in axes[-1])
    fig.legend(handles=handles, loc="upper center", ncol=len(models), fontsize=FS_LEG,
               frameon=False, handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.5, y0 - 0.52 / h))
    note = (f"{stat.capitalize()} of {bin_n} chunks.  Both axes are linear to "
            f"{REGRET_LINTHRESH:g}% (regret) and {MAPE_LINTHRESH:g}% (cost MAPE) and "
            "logarithmic above.  Neither is capped.")
    if floor is not None:
        note = f"Reported at a {floor:g} ms time floor.  " + note
    fig.text(0.5, y0 - 0.80 / h, note, ha="center", va="top",
             fontsize=FS_LEG - 0.5, color="0.35")
    return fig


def summary(data, models):
    w = 22
    head = (f"{'workload':<9}{'model':<{w}}{'chunks':>7}{'regret first 200':>18}"
            f"{'regret last 20%':>17}{'cost MAPE last 20%':>20}{'top-1 pick':>12}")
    lines = [head]
    for k, lbl in WORKLOADS:
        if k not in data:
            continue
        d = data[k]
        for name, *_ in models:
            s = d[d.model == name].sort_values("chunk_index")
            if s.empty:
                continue
            tail = s.iloc[int(np.floor(len(s) * (1.0 - SUMMARY_LAST_FRAC))):]
            first = s[s.chunk_index < s.chunk_index.min() + 200]
            top1 = 100.0 * float((s.regret_pct <= 1e-7).mean())
            lines.append(f"{lbl:<9}{name:<{w}}{len(s):>7}{first.regret_pct.mean():>17.2f}%"
                         f"{tail.regret_pct.mean():>16.2f}%{tail.cost_mape_pct.mean():>19.2f}%"
                         f"{top1:>11.1f}%")
        lines.append("")
    out = "\n".join(lines).rstrip()
    print(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", required=True, help="fig8_model_chunks.py output")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--bin", type=int, default=DEFAULT_BIN)
    ap.add_argument("--stat", default="mean", choices=["mean", "median"])
    ap.add_argument("--models", default="", help="comma-separated subset, in draw order")
    ap.add_argument("--name", default="fig8d_models", help="output basename")
    a = ap.parse_args()

    df = pd.read_csv(a.chunks)
    models = MODELS
    if a.models:
        want = [m.strip() for m in a.models.split(",")]
        known = {n: (n, c, st, z) for n, c, st, z in MODELS}
        missing = [m for m in want if m not in known]
        if missing:
            raise SystemExit(f"unknown model(s): {', '.join(missing)}\n"
                             f"known: {', '.join(n for n, _ in MODELS)}")
        models = [known[m] for m in want]
    have = set(df.model.unique())
    models = [m for m in models if m[0] in have]
    if not models:
        raise SystemExit("none of the requested models are in the CSV")

    data = {k: df[df.workload == k].copy() for k, _ in WORKLOADS if (df.workload == k).any()}
    if not data:
        raise SystemExit("no known workload in the CSV")

    os.makedirs(a.out, exist_ok=True)
    set_fonts()
    floor = float(df.floor_ms.iloc[0]) if "floor_ms" in df and df.floor_ms.nunique() == 1 else None
    fig = draw(data, a.bin, a.stat, models, floor)
    png = os.path.join(a.out, f"{a.name}.png")
    fig.savefig(png, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)
    txt = os.path.join(a.out, f"{a.name}_summary.txt")
    with open(txt, "w") as f:
        f.write(summary(data, models) + "\n")
    print(f"\nwrote {png} and {txt}")


if __name__ == "__main__":
    main()

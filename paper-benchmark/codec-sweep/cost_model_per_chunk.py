#!/usr/bin/env python3
"""Per-chunk codec selection vs every single codec, per tier and cost model.

For each storage tier (bandwidth from probe_tiers.sh) and each NeuroPress
cost-model setting

  balanced              cost = compress + decompress + I/O     (w = 1, 1, 1)
  compression-focused   cost = I/O = compressed bytes / bw     (w = 0, 0, 1)
  speed-focused         cost = compress + decompress           (w = 1, 1, 0)

this scores, on that setting's OWN cost in ms per GiB of input,
  - every codec used alone for all chunks, and
  - an emulated per-chunk selector that takes, for every 4 MiB chunk, the codec
    with the lowest cost. It reads the measured values, so it is an oracle:
    the most any per-chunk selector can gain.
The gap between the oracle and the best single codec is that ceiling.

Writes figures/cost-model-tiers/per-chunk/<tier>_<model>.png (12 figures): one
panel per workload (bars sorted by cost, the oracle's codec mix listed) and a
summary panel of the ceiling per workload; and codec_mix_by_tier.png, which
codecs the per-chunk oracle chose, per tier and cost model. No time floor, no
ratio cap.

  cost_model_per_chunk.py PROBE_CSV [--sweep DIR] [--out DIR]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cost_model_tiers as cm  # noqa: E402
from selection_headroom import CODEC_COLOR  # noqa: E402  same hue per codec

DEFAULT_OUT = os.path.join(cm.DEFAULT_OUT, "per-chunk")
NEUTRAL = "#bdbbb2"  # single codecs that are not the best one
ORACLE = cm.ORANGE
BEST = cm.BLUE
# model -> (title, what its cost counts, (w_ct, w_dt, w_io))
OBJECTIVES = {
    "balanced": ("Balanced", "compress + decompress + I/O", (1, 1, 1)),
    "compression": ("Compression-focused",
                    "I/O only = compressed bytes / bandwidth", (0, 0, 1)),
    "speed": ("Speed-focused", "compress + decompress", (1, 1, 0)),
}
ORDER = {c: i for i, c in enumerate(cm.CODECS)}


def evaluate(df, bw_gbs, model):
    """Score every single codec and the per-chunk oracle for one setting.

    @param df     per-chunk measurements (cost_model_tiers.load_workload)
    @param bw_gbs bandwidth in GB/s: one number for a single tier, or a Series
                  aligned with df giving each row its chunk's tier bandwidth
    @param model  key of OBJECTIVES
    @return dict: fixed (codec -> ms/GiB), oracle (ms/GiB), mix (codec ->
            share of chunks the oracle gives it), raw (ms/GiB for no
            compression, None when the cost ignores I/O), best codec, gain
            (fraction the oracle saves against the best codec)
    """
    w_ct, w_dt, w_io = OBJECTIVES[model][2]
    d = df.assign(bw=bw_gbs)
    io_ms = d["comp_bytes"] / (d["bw"] * 1e6)  # GB/s = 1e6 bytes/ms
    d = d.assign(score=w_ct * d["comp_ms"] + w_dt * d["decomp_ms"]
                 + w_io * io_ms, order=d["codec"].map(ORDER))
    per_chunk = d.groupby(["file", "chunk"])["codec"].nunique()
    if (per_chunk != len(cm.CODECS)).any():
        sys.exit("a chunk is missing a codec; the oracle would be unfair")
    chunks = d.groupby(["file", "chunk"])[["bytes", "bw"]].first()
    gib = chunks["bytes"].sum() / 2**30
    fixed = d.groupby("codec")["score"].sum() / gib
    pick = (d.sort_values(["file", "chunk", "score", "order"])
            .groupby(["file", "chunk"]).head(1))
    oracle = pick["score"].sum() / gib
    best = fixed.idxmin()
    return {"fixed": fixed, "oracle": oracle,
            "mix": pick["codec"].value_counts(normalize=True),
            "raw": (w_io * (chunks["bytes"] / (chunks["bw"] * 1e6)).sum() / gib
                    if w_io else None),
            "best": best, "gain": max(0.0, 1.0 - oracle / fixed[best])}


def fmt_ms(v):
    """Milliseconds: whole numbers from 100 up, three significant digits below."""
    return f"{v:,.0f} ms" if v >= 100 else f"{v:.3g} ms"


def mix_text(mix, keep=3):
    """The oracle's codec mix: the largest shares, then the rest summed."""
    top = mix.iloc[:keep]
    parts = [f"{cm.CODECS[c]} {100 * s:.0f}%" for c, s in top.items()
             if s >= 0.005]
    rest = mix.iloc[keep:].sum()
    if rest >= 0.005:
        parts.append(f"other {100 * rest:.0f}%")
    return "picks: " + ", ".join(parts)


def plot_panel(ax, wl, r):
    """One workload: single codecs and the oracle, cheapest at the top.

    @param ax matplotlib axes
    @param wl workload key
    @param r  output of evaluate
    """
    bars = [(cm.CODECS[c], v, BEST if c == r["best"] else NEUTRAL)
            for c, v in r["fixed"].items()]
    bars.append(("Per-chunk optimal", r["oracle"], ORACLE))
    bars.sort(key=lambda b: b[1])
    y = np.arange(len(bars))[::-1]
    vals = [b[1] for b in bars]
    ax.barh(y, vals, height=0.74, color=[b[2] for b in bars],
            edgecolor=cm.SURFACE, linewidth=1)
    cm.style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", which="major", color=cm.GRID, linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    for yi, (name, v, color) in zip(y, bars):
        if color != NEUTRAL:
            ax.annotate(fmt_ms(v), (v, yi), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=cm.INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.1", fc=cm.SURFACE,
                                  ec="none"))
    if r["raw"] is not None:
        ax.axvline(r["raw"], color=cm.MUTED, linewidth=1.3,
                   linestyle=(0, (4, 2)), zorder=4)
    lo = min(vals + ([r["raw"]] if r["raw"] else []))
    hi = max(vals + ([r["raw"]] if r["raw"] else []))
    ax.set_xlim(lo / 1.5, hi * 3.5)
    # 1-2-5 ticks crowd past ~1.5 decades (5000 meets 10000); use 1-3 there.
    wide = np.log10(hi * 3.5 / (lo / 1.5)) > 1.5
    ax.xaxis.set_major_locator(LogLocator(base=10,
                                          subs=(1, 3) if wide else (1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(f"{cm.WORKLOADS[wl]}: per-chunk {100 * r['gain']:.2f}% below "
                 f"{cm.CODECS[r['best']]}", loc="left", fontsize=9.5,
                 color=cm.INK, pad=16)
    ax.text(0, 1.012, mix_text(r["mix"]), transform=ax.transAxes, fontsize=7.5,
            color=cm.INK2, va="bottom")


def plot_summary(ax, res):
    """The ceiling per workload: how far below the best codec the oracle gets.

    @param ax  matplotlib axes
    @param res workload -> output of evaluate
    """
    wls = [w for w in cm.WORKLOADS if w in res][::-1]
    gains = [100 * res[w]["gain"] for w in wls]
    y = np.arange(len(wls))
    ax.barh(y, gains, height=0.6, color=ORACLE, edgecolor=cm.SURFACE)
    cm.style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=cm.GRID, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([cm.WORKLOADS[w] for w in wls], fontsize=8.5)
    for yi, g, w in zip(y, gains, wls):
        ax.annotate(f"{g:.2f}% vs {cm.CODECS[res[w]['best']]}", (g, yi),
                    xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=cm.INK)
    ax.set_xlim(0, max(max(gains), 0.1) * 1.9)
    ax.set_xlabel("Cost saved by per-chunk selection (%)", color=cm.INK2,
                  fontsize=8.5)
    ax.set_title("Ceiling of per-chunk selection", loc="left", fontsize=9.5,
                 color=cm.INK, pad=16)
    ax.text(0, 1.012, "cost saved vs the best single codec", fontsize=7.5,
            transform=ax.transAxes, color=cm.INK2, va="bottom")


OTHER_COLOR = "#c3c2b7"  # codecs that never reach MIX_MIN anywhere
MIX_MIN = 0.01


def ink_on(hex_color):
    """Text colour readable on a fill: near-black on light, white on dark."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return cm.INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def mix_table(res):
    """Oracle codec shares per workload plus an equal-weight "All" row.

    @param res workload -> output of evaluate
    @return DataFrame, rows = workload display names (All last), cols = codecs
    """
    t = pd.DataFrame({cm.WORKLOADS[w]: res[w]["mix"] for w in cm.WORKLOADS
                      if w in res}).T.fillna(0.0)
    t.loc["All (equal weight)"] = t.mean()
    return t


# Short column names for the mix figure
TIER_SHORT = {"dram": "DRAM (GPU to host memory)", "nvme": "Node-local NVMe",
              "burst_buffer": "Burst buffer (/work/nvme)",
              "lustre": "Lustre PFS (/work/hdd)"}


def plot_mix_panel(ax, t, shown):
    """One (tier, cost model) panel: a 100% stacked bar per workload.

    @param ax    matplotlib axes
    @param t     mix_table output
    @param shown codecs drawn in their own colour; the rest fold into "other"
    """
    shares = {c: t[c].to_numpy() if c in t.columns else np.zeros(len(t))
              for c in shown}
    shares["other"] = t.drop(columns=[c for c in shown
                                      if c in t.columns]).sum(axis=1).to_numpy()
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for c, frac in shares.items():
        v = 100 * frac
        color = CODEC_COLOR.get(c, OTHER_COLOR)
        ax.barh(y, v, left=left, height=0.72, color=color,
                edgecolor=cm.SURFACE, linewidth=1.2)
        for yi, l, w in zip(y, left, v):
            if w >= 12:
                ax.text(l + w / 2, yi, f"{w:.0f}", ha="center", va="center",
                        fontsize=7.5, color=ink_on(color))
        left += v
    cm.style_axes(ax)
    ax.grid(False)
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=8.5)
    ax.axhline(y[-1] + 0.5, color=cm.AXIS, linewidth=0.8)


def plot_mix(mixes, bw, out):
    """Which codecs the per-chunk oracle chose, per tier (columns) and cost
    model (rows), as 100% stacked bars per workload.

    @param mixes (tier, model) -> mix_table output
    @param bw    tier -> GB/s
    @param out   PNG path
    """
    shown = [c for c in CODEC_COLOR
             if any(m.get(c, pd.Series(0)).max() >= MIX_MIN
                    for m in mixes.values())]
    tiers, models = list(bw), list(OBJECTIVES)
    fig, axes = plt.subplots(len(models), len(tiers), figsize=(17, 11),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.055, top=0.845,
                        wspace=0.06, hspace=0.24)
    for i, model in enumerate(models):
        for j, tier in enumerate(tiers):
            plot_mix_panel(axes[i, j], mixes[(tier, model)], shown)
            if i == 0:
                axes[i, j].set_title(f"{TIER_SHORT[tier]}\n{bw[tier]:.2f} GB/s",
                                     fontsize=9.5, color=cm.INK, loc="left")
        box = axes[i, 0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2,
                 f"{OBJECTIVES[model][0]}\n({OBJECTIVES[model][1]})",
                 rotation=90, ha="center", va="center", fontsize=10.5,
                 color=cm.INK, linespacing=1.4)
    for ax in axes[-1]:
        ax.set_xlabel("Share of chunks (%)", color=cm.INK2, fontsize=8.5)
    handles = [Patch(color=CODEC_COLOR[c], label=cm.CODECS[c]) for c in shown]
    handles.append(Patch(color=OTHER_COLOR,
                         label=f"Other (each < {100 * MIX_MIN:.0f}%)"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles),
               frameon=False, fontsize=9.5, labelcolor=cm.INK2,
               bbox_to_anchor=(0.005, 0.925))
    fig.suptitle("Which codec the per-chunk optimal selector chooses, by tier "
                 "(columns) and cost model (rows)", x=0.01, ha="left",
                 fontsize=13, color=cm.INK, y=0.995)
    fig.text(0.01, 0.945, "Share of 4 MiB chunks given to each codec when every "
             "chunk takes its cheapest codec under the row's cost (measured A100 "
             "values, 12 lossless codecs).\nOnly the balanced row can change "
             "with the tier: bandwidth is absent from the speed-focused cost and "
             "only scales the compression-focused one.", fontsize=9,
             color=cm.INK2, va="bottom", linespacing=1.4)
    fig.savefig(out, dpi=170, facecolor=cm.SURFACE)
    plt.close(fig)


def plot_figure(title, note, model, res, out):
    """One cost-model figure: seven workload panels and the summary.

    @param title figure title
    @param note  sentence appended to the cost definition line ("" for none)
    @param model key of OBJECTIVES
    @param res   workload -> output of evaluate
    @param out   PNG path
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 10.5))
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.83,
                        wspace=0.42, hspace=0.36)
    wls = [w for w in cm.WORKLOADS if w in res]
    for ax, wl in zip(axes.flat, wls):
        plot_panel(ax, wl, res[wl])
    for ax in axes[1, :3]:
        ax.set_xlabel("ms per GiB of input (log)", color=cm.INK2, fontsize=8.5)
    plot_summary(axes.flat[len(wls)], res)
    what = OBJECTIVES[model][1]
    handles = [Patch(color=ORACLE, label="Per-chunk optimal (oracle)"),
               Patch(color=BEST, label="Best single codec"),
               Patch(color=NEUTRAL, label="Other single codecs")]
    if model != "speed":
        handles.append(Line2D([], [], color=cm.MUTED, linewidth=1.3,
                              linestyle=(0, (4, 2)), label="No compression"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles),
               frameon=False, fontsize=9.5, labelcolor=cm.INK2,
               bbox_to_anchor=(0.005, 0.925))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=cm.INK, y=0.99)
    fig.text(0.01, 0.945, f"Cost = {what}, in ms per GiB of input. Per-chunk "
             "optimal = the cheapest codec for each 4 MiB chunk, from measured "
             "A100 values (an oracle: the most per-chunk selection can gain)."
             + note, fontsize=9, color=cm.INK2)
    fig.savefig(out, dpi=170, facecolor=cm.SURFACE)
    plt.close(fig)


def main():
    """Load the sweep and the tier probe, draw 12 figures, print the ceilings."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe_csv", help="tier_bw.csv from probe_tiers.sh")
    ap.add_argument("--sweep", default=cm.DEFAULT_SWEEP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    bw = cm.tier_bandwidths(a.probe_csv)
    data = {w: cm.load_workload(a.sweep, w) for w in cm.WORKLOADS}
    os.makedirs(a.out, exist_ok=True)
    rows, mixes = [], {}
    for tier, gbs in bw.items():
        for model in OBJECTIVES:
            res = {w: evaluate(d, gbs, model) for w, d in data.items()}
            mixes[(tier, model)] = mix_table(res)
            title = (f"{OBJECTIVES[model][0]} cost model on {cm.TIERS[tier][0]}, "
                     f"{gbs:.2f} GB/s measured: every codec alone vs per-chunk "
                     "selection")
            note = (" Bandwidth is not in this cost: same figure on every tier."
                    if model == "speed" else
                    " Bandwidth only scales this cost: same ranking on every "
                    "tier." if model == "compression" else "")
            plot_figure(title, note, model, res,
                        os.path.join(a.out, f"{tier}_{model}.png"))
            for w, r in res.items():
                rows.append({"tier": tier, "model": model, "workload": w,
                             "best": cm.CODECS[r["best"]],
                             "best_ms": r["fixed"][r["best"]],
                             "oracle_ms": r["oracle"],
                             "gain_pct": 100 * r["gain"],
                             "mix": mix_text(r["mix"])[7:]})
    with pd.option_context("display.width", 220, "display.precision", 3,
                           "display.max_rows", None,
                           "display.max_colwidth", 60):
        print(pd.DataFrame(rows).to_string(index=False))
    plot_mix(mixes, bw, os.path.join(a.out, "codec_mix_by_tier.png"))
    print(f"wrote {len(rows) // len(data)} figures and codec_mix_by_tier.png "
          f"to {a.out}")


if __name__ == "__main__":
    main()

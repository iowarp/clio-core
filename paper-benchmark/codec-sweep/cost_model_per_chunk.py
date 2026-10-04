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
RATIO_COLOR = "#7b3fa0"  # compression-ratio overlay, apart from the bar hues
STORE_COLOR = "#6b6960"  # the measured "store uncompressed" candidate
STORE = "store"
CPU_NEUTRAL = "#e3d3b1"  # CPU codecs (cpu_corpus_sweep) outside the top 3
CPU = "cpu-"
STORES = (STORE, CPU + STORE)  # GPU and CPU "store uncompressed"
ORACLE = cm.ORANGE
BEST = cm.BLUE
RANK_COLORS = [BEST, "#7aaee8", "#bcd4f2"]  # 1st, 2nd, 3rd single codec
# model -> (title, what its cost counts, (w_ct, w_dt, w_io))
OBJECTIVES = {
    "balanced": ("Balanced", "compress + decompress + I/O", (1, 1, 1)),
    "compression": ("Compression-focused",
                    "I/O only = compressed bytes / bandwidth", (0, 0, 1)),
    "speed": ("Speed-focused", "compress + decompress", (1, 1, 0)),
}
ORDER = {c: i for i, c in enumerate(cm.CODECS)}
SHUF = "|shuf"  # variant suffix: the setting runs a byte or bit pre-shuffle


def variants(df):
    """The bar each row belongs to: its codec, split by pre-shuffle.

    @param df per-chunk rows; with a "config" column (float32 sweep) a
              codec's settings with shuffle=byte|bit form a second variant
              (LZ4's own bitshuffle= is not a pre-shuffle)
    @return Series aligned with df: codec key, plus SHUF when shuffled
    """
    if "config" not in df:
        return df["codec"]
    shuffled = df["config"].str.contains(r"(?<!bit)shuffle=(?:byte|bit)")
    return df["codec"] + np.where(shuffled, SHUF, "")


def label(key):
    """@return the display name of a variant key, e.g. "SPratio + shuffle"."""
    base, _, rest = key.partition("|")
    return cm.CODECS[base] + (" + shuffle" if rest else "")


def share(pick, col):
    """Share of chunks the oracle gives each value of col, largest first.

    @param pick the oracle's rows; with a "weight" column (weighted
                placement) a row counts as that fraction of a chunk
    @param col  "codec" or "variant"
    """
    if "weight" not in pick:
        return pick[col].value_counts(normalize=True)
    s = pick.groupby(col)["weight"].sum()
    return (s / s.sum()).sort_values(ascending=False)


def evaluate(df, bw_gbs, model):
    """Score every single codec and the per-chunk oracle for one setting.

    With a "config" column (cost_model_tiers.load_float32) each setting is a
    candidate: a codec's bar is its best single setting used for every chunk,
    and the oracle picks among every setting per chunk.

    @param df     per-chunk measurements (cost_model_tiers.load_workload or
                  load_float32)
    @param bw_gbs bandwidth in GB/s: one number for a single tier, or a Series
                  aligned with df giving each row its chunk's tier bandwidth
    @param model  key of OBJECTIVES
    @return dict: fixed (codec -> ms/GiB of its best setting), oracle
            (ms/GiB), mix (codec -> share of chunks the oracle gives it), raw
            (ms/GiB for no compression, None when the cost ignores I/O), best
            codec, best_config (its setting), gain (fraction the oracle saves
            against the best codec), ratio (codec -> compression ratio of the
            setting its bar uses), oracle_ratio (ratio of the oracle's picks)
    """
    w_ct, w_dt, w_io = OBJECTIVES[model][2]
    key = "config" if "config" in df else "codec"
    d = df.assign(bw=bw_gbs)
    io_ms = d["comp_bytes"] / (d["bw"] * 1e6)  # GB/s = 1e6 bytes/ms
    d = d.assign(score=w_ct * d["comp_ms"] + w_dt * d["decomp_ms"]
                 + w_io * io_ms, order=d["codec"].map(ORDER),
                 variant=variants(d))
    per_chunk = d.groupby(["file", "chunk"])[key].nunique()
    if (per_chunk != d[key].nunique()).any():
        sys.exit("a chunk is missing a candidate; the oracle would be unfair")
    chunks = d.groupby(["file", "chunk"])[["bytes", "bw"]].first()
    gib = chunks["bytes"].sum() / 2**30
    per_cand = d.groupby(key)["score"].sum() / gib
    family = d.groupby(key)["variant"].first()
    fixed = per_cand.groupby(family).min()
    pick = (d.sort_values(["file", "chunk", "score", "order"])
            .groupby(["file", "chunk"]).head(1))
    oracle = pick["score"].sum() / gib
    best = fixed.idxmin()
    sums = d.groupby(key)[["bytes", "comp_bytes"]].sum()
    best_cfg = per_cand.groupby(family).idxmin()
    ratio = pd.Series((sums["bytes"] / sums["comp_bytes"])[best_cfg].values,
                      index=best_cfg.index)
    return {"fixed": fixed, "oracle": oracle, "ratio": ratio,
            "oracle_ratio": pick["bytes"].sum() / pick["comp_bytes"].sum(),
            "mix": share(pick, "codec"),
            "mix_variant": share(pick, "variant"),
            "raw": (w_io * (chunks["bytes"] / (chunks["bw"] * 1e6)).sum() / gib
                    if w_io else None),
            "best": best, "model": model,
            "best_config": per_cand.groupby(family).idxmin()[best],
            "gain": max(0.0, 1.0 - oracle / fixed[best])}


def fmt_ms(v):
    """Milliseconds: whole numbers from 100 up, three significant digits below."""
    return f"{v:,.0f} ms" if v >= 100 else f"{v:.3g} ms"


def mix_text(mix, keep=3):
    """The oracle's codec mix: the largest shares, then the rest summed."""
    top = mix.iloc[:keep]
    parts = [f"{label(c)} {100 * s:.0f}%" for c, s in top.items()
             if s >= 0.005]
    rest = mix.iloc[keep:].sum()
    if rest >= 0.005:
        parts.append(f"other {100 * rest:.0f}%")
    return "picks: " + ", ".join(parts)


def plot_ratio_line(ax, y, ratios):
    """Overlay each bar's compression ratio on a twin axis along the top.

    @param ax     the panel's axes (cost on x, one bar per y)
    @param y      bar positions
    @param ratios compression ratio of each bar, aligned with y
    """
    tw = ax.twiny()
    tw.plot(ratios, y, color=RATIO_COLOR, linewidth=1.2, marker="o",
            markersize=3.2, markerfacecolor=cm.SURFACE, markeredgewidth=1.1,
            zorder=6)
    lo, hi = min(min(ratios), 1.0), max(ratios)
    if hi / lo > 20:
        tw.set_xscale("log")
        tw.set_xlim(lo / 1.3, hi * 1.3)
        tw.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        tw.xaxis.set_minor_formatter(NullFormatter())
    else:
        pad = (hi - lo) * 0.08 or 0.05
        tw.set_xlim(lo - pad, hi + pad)
        tw.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}×"))
    tw.tick_params(axis="x", labelsize=7, colors=RATIO_COLOR, length=2.5,
                   pad=1.5)
    tw.spines["top"].set_color(RATIO_COLOR)
    for s in ("left", "right", "bottom"):
        tw.spines[s].set_visible(False)


def plot_panel(ax, wl, r, ratio_line=False):
    """One workload: single codecs and the oracle, cheapest at the top.

    @param ax         matplotlib axes
    @param wl         workload key
    @param r          output of evaluate
    @param ratio_line overlay each bar's compression ratio on a top axis
    """
    ranked = r["fixed"].sort_values()
    best_v = ranked.iloc[0]
    bars = [(label(c), v, RANK_COLORS[i] if i < len(RANK_COLORS)
             else STORE_COLOR if c in STORES
             else CPU_NEUTRAL if c.startswith(CPU) else NEUTRAL, i,
             r["ratio"][c])
            for i, (c, v) in enumerate(ranked.items())]
    bars.append(("Per-chunk optimal", r["oracle"], ORACLE, -1,
                 r["oracle_ratio"]))
    bars.sort(key=lambda b: b[1])
    y = np.arange(len(bars))[::-1]
    vals = [b[1] for b in bars]
    drawn = ax.barh(y, vals, height=0.74, color=[b[2] for b in bars],
                    edgecolor=cm.SURFACE, linewidth=1)
    for patch, b in zip(drawn, bars):
        if b[0] in {label(c) for c in STORES}:
            patch.set_hatch("////")
    cm.style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", which="major", color=cm.GRID, linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    won = r["mix_variant"].get(r["best"], 0.0)
    ratio = r.get("model") == "compression"  # cost = raw / ratio
    for yi, (name, v, color, rank, _) in zip(y, bars):
        plain = color in (NEUTRAL, CPU_NEUTRAL)
        if not plain or ratio:
            text = fmt_ms(v)
            if rank == 0:
                text += f"  · per-chunk pick on {100 * won:.0f}%"
            elif rank > 0:
                text = (f"+{100 * (v / best_v - 1):.2f}% vs 1st"
                        if not plain else "")
            if ratio:
                text = f"{r['raw'] / v:.2f}×" + (f"  {text}" if text else "")
            ax.annotate(text, (v, yi), xytext=(3, 0),
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
    title = (f"{cm.WORKLOADS[wl]}: per-chunk {100 * r['gain']:.2f}% below "
             f"{label(r['best'])}")
    picks = mix_text(r["mix_variant"], keep=2)
    lift = 0
    if ratio_line:
        plot_ratio_line(ax, y, [b[4] for b in bars])
        lift = 12  # clear the ratio axis's tick labels
    # In points, so the stack holds at any panel height.
    for text, dy, size, color in ((picks, 3 + lift, 7.5, cm.INK2),
                                  (title, 14 + lift, 9.5, cm.INK)):
        ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(0, dy),
                    textcoords="offset points", fontsize=size, color=color,
                    va="bottom", ha="left")


def plot_summary(ax, res):
    """The ceiling per workload: how far below the best codec the oracle gets.

    @param ax  matplotlib axes
    @param res workload -> output of evaluate
    """
    wls = [w for w in cm.WORKLOADS if w in res][::-1]
    gains = [100 * res[w]["gain"] for w in wls]
    # how much worse the 2nd- and 3rd-best single codecs are than the 1st
    gaps = {k: [] for k in (1, 2)}
    for w in wls:
        f = res[w]["fixed"].sort_values()
        for k in gaps:
            gaps[k].append(100 * (f.iloc[k] / f.iloc[0] - 1) if len(f) > k
                           else 0.0)
    y = np.arange(len(wls))
    h, off = 0.26, 0.27
    # Scale to the oracle and 2nd place; a longer 3rd-place bar is clipped
    # at the edge (its label keeps the true value).
    lim = max(max(gains + gaps[1]), 0.1) * 1.6
    edge = lim * 0.72
    ax.barh(y + off, gains, height=h, color=ORACLE, edgecolor=cm.SURFACE)
    for k, dy in ((1, 0.0), (2, -off)):
        ax.barh(y + dy, [min(v, edge) for v in gaps[k]], height=h,
                color=RANK_COLORS[k], edgecolor=cm.SURFACE)
    cm.style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=cm.GRID, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([cm.WORKLOADS[w] for w in wls], fontsize=8.5)
    for i, (yi, w) in enumerate(zip(y, wls)):
        ax.annotate(f"{gains[i]:.2f}% vs {label(res[w]['best'])}",
                    (gains[i], yi + off), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color=cm.INK)
        for k, dy, name in ((1, 0.0, "2nd"), (2, -off, "3rd")):
            v = gaps[k][i]
            ax.annotate(f"{name} +{v:.2f}%" + (" →" if v > edge else ""),
                        (min(v, edge), yi + dy), xytext=(3, 0),
                        textcoords="offset points", va="center",
                        fontsize=6.5, color=cm.INK2)
    ax.set_xlim(0, lim)
    ax.set_xlabel("% of the best single codec's cost", color=cm.INK2,
                  fontsize=8.5)
    ax.set_title("Ceiling of per-chunk selection", loc="left", fontsize=9.5,
                 color=cm.INK, pad=16)
    ax.text(0, 1.012, "orange: oracle saves vs 1st; blues: 2nd / 3rd worse by",
            fontsize=7.5, transform=ax.transAxes, color=cm.INK2, va="bottom")


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
                name = TIER_SHORT.get(tier) or cm.FIXED_TIERS[tier][0]
                axes[i, j].set_title(f"{name}\n{bw[tier]:g} GB/s",
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
             "values; each codec's settings folded into it).\nOnly the balanced row can change "
             "with the tier: bandwidth is absent from the speed-focused cost and "
             "only scales the compression-focused one.", fontsize=9,
             color=cm.INK2, va="bottom", linespacing=1.4)
    fig.savefig(out, dpi=170, facecolor=cm.SURFACE)
    plt.close(fig)


def plot_figure(title, note, model, res, out, ratio_line=False):
    """One cost-model figure: a panel per workload and the summary, 4 across.

    @param title      figure title
    @param note       sentence appended to the cost definition line ("" for
                      none)
    @param model      key of OBJECTIVES
    @param res        workload -> output of evaluate
    @param out        PNG path
    @param ratio_line overlay each bar's compression ratio on a top axis
    """
    wls = [w for w in cm.WORKLOADS if w in res]
    ncols = 4
    nrows = -(-(len(wls) + 1) // ncols)
    nbars = max(len(r["fixed"]) for r in res.values()) + 1
    row = 4.95 * max(1.0, nbars / 13)  # 13 bars fit the original 4.95 in
    height = 0.6 + row * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, height), squeeze=False)
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.63 / height,
                        top=1 - 1.785 / height, wspace=0.42,
                        hspace=(0.46 if ratio_line else 0.36) * 4.95 / row)
    flat = list(axes.flat)
    for i, (ax, wl) in enumerate(zip(flat, wls)):
        plot_panel(ax, wl, res[wl], ratio_line)
        if i + ncols >= len(wls) + 1:  # nothing below it
            ax.set_xlabel("ms per GiB of input (log)", color=cm.INK2,
                          fontsize=8.5)
    plot_summary(flat[len(wls)], res)
    for ax in flat[len(wls) + 1:]:
        ax.set_visible(False)
    what = OBJECTIVES[model][1]
    handles = [Patch(color=ORACLE, label="Per-chunk optimal (oracle)"),
               Patch(color=BEST, label="Best single codec (1st)"),
               Patch(color=RANK_COLORS[1], label="2nd"),
               Patch(color=RANK_COLORS[2], label="3rd"),
               Patch(color=NEUTRAL, label="Other single codecs")]
    is_cpu = [c.startswith(CPU) for r in res.values() for c in r["fixed"].index
              if c not in STORES]
    if all(is_cpu):
        handles[-1] = Patch(color=CPU_NEUTRAL, label="Other single codecs")
    elif any(is_cpu):
        handles[-1] = Patch(color=NEUTRAL, label="Other GPU codecs")
        handles.append(Patch(color=CPU_NEUTRAL, label="Other CPU codecs"))
    if any(c in r["fixed"] for c in STORES for r in res.values()):
        handles.append(Patch(facecolor=STORE_COLOR, edgecolor=cm.SURFACE,
                             hatch="////", label="Uncompressed (measured)"))
    if model != "speed":
        handles.append(Line2D([], [], color=cm.MUTED, linewidth=1.3,
                              linestyle=(0, (4, 2)), label="No compression"))
    if ratio_line:
        handles.append(Line2D([], [], color=RATIO_COLOR, linewidth=1.2,
                              marker="o", markersize=4,
                              markerfacecolor=cm.SURFACE,
                              label="Compression ratio (top axis)"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles),
               frameon=False, fontsize=9.5, labelcolor=cm.INK2,
               bbox_to_anchor=(0.005, 1 - 0.7875 / height))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=cm.INK,
                 y=1 - 0.105 / height)
    fig.text(0.01, 1 - 0.5775 / height, f"Cost = {what}, in ms per GiB of input. Per-chunk "
             "optimal = the cheapest codec for each 4 MiB chunk, from measured "
             "values (an oracle: the most per-chunk selection can gain)."
             + note, fontsize=9, color=cm.INK2, wrap=True)
    fig.savefig(out, dpi=170, facecolor=cm.SURFACE)
    plt.close(fig)


def load_data(a):
    """Per-chunk measurements of every workload to draw.

    @param a parsed arguments: --float32 DIR (complete workloads of a float32
             sweep, every setting a candidate) or --sweep DIR (legacy layout);
             optional --cpu DIR (a cpu_corpus_sweep run on the same chunks)
             and --engines gpu|cpu|both (which candidates to keep)
    @return dict workload -> DataFrame for evaluate
    """
    engines = getattr(a, "engines", "gpu")
    cpu_dir = getattr(a, "cpu", None)
    if engines != "gpu" and not cpu_dir:
        sys.exit(f"--engines {engines} needs --cpu DIR")
    if a.float32 or cpu_dir:
        gpu = cm.load_float32(a.float32) if a.float32 and engines != "cpu" else {}
        cpu = cm.load_float32(cpu_dir, "cpu") if engines != "gpu" else {}
        if engines == "both":
            wls = [w for w in gpu if w in cpu]
            data = {w: pd.concat([gpu[w], cpu[w]], ignore_index=True)
                    for w in wls}
        else:
            data = gpu or cpu
        if not data:
            sys.exit("no complete workload to draw")
        return data
    return {w: cm.load_workload(a.sweep, w) for w in cm.LEGACY_WORKLOADS}


def main():
    """Load the sweep and the tier bandwidths, draw the figures, print the
    ceilings."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", help="tier_bw.csv from probe_tiers.sh; "
                    "default: cost_model_tiers.FIXED_TIERS")
    ap.add_argument("--sweep", default=cm.DEFAULT_SWEEP)
    ap.add_argument("--float32", help="run_float32_sweep.sh results dir")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.probe:
        bw = cm.tier_bandwidths(a.probe)
    else:
        bw = {t: gbs for t, (_, gbs) in cm.FIXED_TIERS.items()}
    data = load_data(a)
    os.makedirs(a.out, exist_ok=True)
    rows, mixes = [], {}
    for tier, gbs in bw.items():
        for model in OBJECTIVES:
            res = {w: evaluate(d, gbs, model) for w, d in data.items()}
            mixes[(tier, model)] = mix_table(res)
            where = (f"{cm.TIERS[tier][0]}, {gbs:.2f} GB/s measured" if a.probe
                     else f"{cm.FIXED_TIERS[tier][0]}, {gbs:g} GB/s")
            title = (f"{OBJECTIVES[model][0]} cost model on {where}: "
                     f"every codec alone vs per-chunk selection")
            note = (" Bandwidth is not in this cost: same figure on every tier."
                    if model == "speed" else
                    " Bandwidth only scales this cost: same ranking on every "
                    "tier." if model == "compression" else "")
            if a.float32:
                note += " Codec bar = its best single setting (incl. shuffle)."
            plot_figure(title, note, model, res,
                        os.path.join(a.out, f"{tier}_{model}.png"))
            for w, r in res.items():
                rows.append({"tier": tier, "model": model, "workload": w,
                             "best": label(r["best"]),
                             "best_config": r["best_config"],
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

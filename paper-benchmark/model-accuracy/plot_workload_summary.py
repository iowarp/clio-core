#!/usr/bin/env python3
"""Executive summary: how much NeuroPress gains over the best single codec on
each workload, next to HCompress and the best possible (oracle), from the
compare CSVs of run_kmeans_parallel.sh (no run).

    plot_workload_summary.py --out PNG [--config p1i1]
    plot_workload_summary.py --final FIG_ROOT [--summary CSV] [--out PNG]

--final: the final setup (final_config.py, final_check.py's FIG_ROOT/final_summary.csv):
NeuroPress with min-max bars over the iterations, the time oracle (best per chunk by
end-to-end time) as the light track and the cost oracle (best per chunk by the cost model) as a
marker on NeuroPress's bar; default output FIG_ROOT/summary_1x1_final.png. --summary: another
summary CSV of final_check.py's schema (default FIG_ROOT/final_summary.csv). The setup, the cost
model of each workload and the run count are written on the figure.

Left: application time saved (first timed write to last k-means iteration);
right: compression-ratio gain; both against each workload's best single codec
(the 0 line), so more is better in both. NeuroPress is the highlighted bar,
HCompress the muted one, the oracle a light track behind NeuroPress's bar. Also writes the numbers
to the PNG's name with .csv.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = "/mnt/nvme0/v2-work/runs"
# (label, description, dataset, tag prefix, weights as in the tag, cost model)
WORKLOADS = (
    ("Nyx", "cosmology hydrodynamics, 51 GB", "nyx-multiphase-50g", "km10b1g", "1-10-10",
     "1/10/10 at 1 GB/s"),
    ("VPIC", "plasma particle-in-cell, 26 GB", "vpic-slabs", "km10b1g", "1-10-10",
     "1/10/10 at 1 GB/s"),
    ("WarpX", "laser-wakefield accelerator, 25 GB", "ref-warpx-tune-final25g", "km10b3g",
     "1-20-11", "1/20/11 at 3 GB/s"),
    ("incflo", "incompressible fluid mixing, 25 GiB", "ref-incflo-tune-final25g",
     "km10b0.5g", "1-40-2.8", "1/40/2.8 at 0.5 GB/s"),
)
NP_COLOR, HC_COLOR = "#d1495b", "#b9b8b3"
# error bars: a darker shade of their bar's colour
NP_ERR, HC_ERR = "#7a1f2c", "#5f5e5a"
CLIP_LO = -12.0   # a bar below this is drawn as a short stub with its value written in it
STUB = -10.0      # the stub's length (the axis then starts here)
TRACK_COLOR, TRACK_INK = "#f4d3d8", "#a8505d"   # the oracle's track: a tint of NP_COLOR
COST_ORACLE = "#4a3aa7"                          # the cost oracle's marker (--final)
INK, INK_2, GRID = "#1a1a19", "#52514e", "#e6e5e1"


def load(config, extras=("",)):
    """@return one row per (workload, option, run) of the given configuration.

    @param extras tag parts after the prefix, one per repeated run (e.g. dc1,
                  dc2, dc3); "" is the untagged run
    """
    rows = []
    for (label, desc, ds, prefix, w, model), extra in (
            (wl, e) for wl in WORKLOADS for e in extras):
        f = os.path.join(RUNS, f"{ds}_{prefix}{extra}{config}w{w}_compare.csv")
        if not os.path.exists(f):
            continue
        t = pd.read_csv(f)
        own = t[t.config == t.config.iloc[-1]]   # the file's own configuration
        best = own[own["mode"] == "fixed"].iloc[0]
        for _, r in own.iterrows():
            rows.append({"workload": label, "description": desc, "dataset": ds,
                         "cost_model": model, "config": r.config, "mode": r["mode"],
                         "app_s": r.app_s, "ratio": r.ratio,
                         "best_single_app_s": best.app_s, "best_single_ratio": best.ratio,
                         "time_saved_pct": -r.app_s_vs_best_single_pct,
                         "ratio_gain_pct": r.ratio_vs_best_single_pct,
                         "bit_exact": r.digest_ok, "run": extra})
    return pd.DataFrame(rows)


def load_final(path):
    """@return a summary CSV of final_check.py's schema in the schema of aggregate()."""
    import final_config as fc
    s = pd.read_csv(path)
    mode = {"best static": "fixed", "NeuroPress": "learn", "cost oracle": "oracle", "time oracle": "timeoracle",
            "HCompress": "hcompress"}
    desc = {w[0]: w[1] for w in WORKLOADS}
    rows = []
    for _, r in s.iterrows():
        ds, w, bw = fc.WORKLOADS[r.workload]
        rows.append({"workload": r.workload, "description": desc.get(r.workload, ""), "dataset": ds,
                     "cost_model": f"cost model {fc.model_name(w)} at {fc.bw_label(bw)} GB/s",
                     "config": "1 process, 1 chunk in flight", "mode": mode[r.option],
                     "app_s": r.app_s, "ratio": r.ratio,
                     "time_saved_pct": -r.time_vs_static_pct, "ratio_gain_pct": r.ratio_vs_static_pct,
                     "time_saved_pct_min": -r.time_vs_static_max, "time_saved_pct_max": -r.time_vs_static_min,
                     "ratio_gain_pct_min": r.ratio_vs_static_min, "ratio_gain_pct_max": r.ratio_vs_static_max,
                     "runs": r.runs, "bit_exact": True})
    return pd.DataFrame(rows)


def aggregate(t):
    """@return one row per (workload, option): the mean over the runs, with the
    min, max and standard deviation of the two benefit columns and the run count."""
    keys = ["workload", "description", "dataset", "cost_model", "config", "mode"]
    g = t.groupby(keys, sort=False)
    out = g[["app_s", "ratio", "best_single_app_s", "best_single_ratio", "time_saved_pct",
             "ratio_gain_pct"]].mean()
    for col in ("time_saved_pct", "ratio_gain_pct"):
        out[f"{col}_min"] = g[col].min()
        out[f"{col}_max"] = g[col].max()
        out[f"{col}_std"] = g[col].std()
    out["runs"] = g.size()
    out["bit_exact"] = g["bit_exact"].all()
    return out.reset_index()


def pct(v):
    """@return v as a percentage, signed only when negative."""
    return f"{v:.0f}%" if v >= 0 else f"\u2212{-v:.0f}%"


def rng(t, label, mode, col):
    """@return (min, max) over the repeated runs, or None for a single run."""
    if f"{col}_min" not in t:
        return None
    s = t[(t.workload == label) & (t["mode"] == mode)]
    if not len(s) or s.runs.iloc[0] < 2:
        return None
    lo, hi = float(s[f"{col}_min"].iloc[0]), float(s[f"{col}_max"].iloc[0])
    return (lo, hi) if hi - lo >= 0.05 else None   # no bar for a range of ~0 (e.g. the ratio)


def whisker(a, r, y, h, color):
    """A min-max error bar at height y: a thin line with short rounded end caps,
    in a darker shade of its bar's colour."""
    a.plot([r[0], r[1]], [y, y], color=color, lw=1.0, alpha=0.9, zorder=4,
           solid_capstyle="round")
    for x in r:
        a.plot([x, x], [y - h * 0.22, y + h * 0.22], color=color, lw=1.0, alpha=0.9,
               zorder=4, solid_capstyle="round")


def value(t, label, mode, col):
    """@return one value, or NaN when that option was not run."""
    s = t[(t.workload == label) & (t["mode"] == mode)][col]
    return float(s.iloc[0]) if len(s) else np.nan


def panel(a, t, labels, col, title, missing="run pending", hc=False):
    """Horizontal bars per workload: NeuroPress, HCompress; the oracle as a marker.

    @param hc keep HCompress's row even without HCompress runs (a 'missing' placeholder)"""
    n = len(labels)
    track = "timeoracle" if "timeoracle" in set(t["mode"]) else "oracle"   # the light track
    vals = t[t["mode"].isin(["learn", "hcompress", "oracle", "timeoracle"])][col]
    inside = vals[vals >= CLIP_LO]
    # the error bars of the bars that fit must fit too
    lows = [inside.min()]
    for lab in labels:
        for m in ("learn", "hcompress"):
            r = rng(t, lab, m, col)
            if r is not None and value(t, lab, m, col) >= CLIP_LO:
                lows.append(r[0])
    # lower axis limit: the lowest bar that fits, or a short stub for the cut ones
    lo_lim = min(min(lows), 0) if (vals < CLIP_LO).sum() == 0 else min(min(lows), STUB)
    span = max(vals.max(), 0) - lo_lim + 1e-9
    min_inside = 0.09 * span   # room for the bold label inside the bar
    has_hc = hc or "hcompress" in set(t["mode"])
    yo = 0.17 if has_hc else 0.0   # NeuroPress's row offset; centered without HCompress
    for i, lab in enumerate(labels):
        y = n - 1 - i
        np_v, hc_v, or_v = (value(t, lab, m, col) for m in ("learn", "hcompress", track))
        co_v = value(t, lab, "oracle", col) if track == "timeoracle" else np.nan
        # The oracle is a light track from 0 to the best possible value, with
        # NeuroPress's bar on top of it (a bullet chart): how much of the
        # possible gain NeuroPress takes is read off directly.
        pad = 0.015 * span
        if np.isfinite(or_v):
            a.barh(y + yo, or_v, height=0.40, color=TRACK_COLOR, zorder=1)
        a.barh(y + yo, np_v, height=0.28, color=NP_COLOR, zorder=2)
        if np.isfinite(co_v):   # the cost oracle: a marker on NeuroPress's row
            a.plot(co_v, y + yo + 0.24, "v", ms=7, color=COST_ORACLE, mec="white", mew=0.8, zorder=5)
        np_rng = rng(t, lab, "learn", col)
        if np_rng is not None:
            whisker(a, np_rng, y + yo, 0.28, NP_ERR)
        label_w = 0.12 * span   # the bold value label's width
        # inside the bar when it fits: at the bar's end, or at its start when an
        # error bar is there, and only if the error bar leaves room for it
        fits = np_v >= min_inside and (np_rng is None or np_rng[0] >= label_w + 0.024 * span)
        if fits:
            x, ha = ((0.012 * span, "left") if np_rng is not None
                     else (np_v - 0.012 * span, "right"))
            a.text(x, y + yo, pct(np_v), va="center", ha=ha,
                   fontsize=12, fontweight="bold", color="white", zorder=3)
            end = max(np_v, np_rng[1]) if np_rng is not None else np_v
        else:   # after the bar (and its error bar)
            x = max(np_v, np_rng[1] if np_rng is not None else np_v) + pad
            a.text(x, y + yo, pct(np_v), va="center", ha="left",
                   fontsize=12, fontweight="bold", color=INK, zorder=3)
            end = x + label_w
        if np.isfinite(or_v):
            a.text(max(end, or_v) + pad, y + yo, f"{'time oracle' if track == 'timeoracle' else 'oracle'} {pct(or_v)}", va="center",
                   ha="left", fontsize=9.5, color=TRACK_INK, zorder=3)
        if np.isfinite(hc_v) and hc_v < CLIP_LO:
            # far below the scale: a short stub with one white break; its real
            # value is written inside it
            a.barh(y - 0.21, lo_lim, height=0.22, color=HC_COLOR, zorder=2)
            xb = lo_lim + 0.012 * span
            a.plot([xb + 0.006 * span, xb - 0.006 * span], [y - 0.33, y - 0.09],
                   color="white", lw=2.0, zorder=3)
            a.text(-pad * 0.6, y - 0.21, pct(hc_v), va="center", ha="right", fontsize=10,
                   color=INK, zorder=4)
        elif np.isfinite(hc_v):
            a.barh(y - 0.21, hc_v, height=0.22, color=HC_COLOR, zorder=2)
            hc_rng = rng(t, lab, "hcompress", col)
            if hc_rng is not None:
                whisker(a, hc_rng, y - 0.21, 0.22, HC_ERR)
                x = max(hc_v, hc_rng[1]) + pad if hc_v >= 0 else min(hc_v, hc_rng[0]) - pad
            else:
                x = hc_v + pad if hc_v >= 0 else hc_v - pad
            a.text(x, y - 0.21, pct(hc_v), va="center", ha="left" if hc_v >= 0 else "right",
                   fontsize=10, color=INK_2)
        elif has_hc:
            a.text(pad, y - 0.21, f"HCompress: {missing}", va="center", ha="left",
                   fontsize=9, color=INK_2, style="italic")
    a.axvline(0, color=INK, lw=1.2)
    a.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: "0%" if abs(v) < 1e-9 else pct(v)))
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(axis="y", length=0)
    a.tick_params(axis="x", colors=INK_2, labelsize=9.5)
    lo, hi = a.get_xlim()
    # room left of the lowest bar for its value label when that bar is drawn in full
    left = 0.15 * span if min(lows) < 0 else 0.02 * span
    a.set_xlim(lo_lim - left if lo_lim < 0 else -0.04 * span, hi + 0.12 * span)
    a.set_ylim(-0.62 if has_hc else -0.5, n - 0.45)   # room for the HCompress row's text
    if inside.min() >= 0:   # only a stub is left of 0: no negative ticks to misread
        a.set_xticks([v for v in a.get_xticks() if 0 <= v <= hi])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--final", default=None, help="FIG_ROOT of the final suite (final_summary.csv)")
    ap.add_argument("--summary", default=None, help="with --final: the summary CSV (default FIG_ROOT/final_summary.csv)")
    ap.add_argument("--config", default="p1i1", help="processes x in flight as in the tag")
    ap.add_argument("--extra", default="",
                    help="repeated runs: tag parts, e.g. dc1,dc2,dc3 (mean and min-max bars)")
    a = ap.parse_args()
    if a.final:
        a.out = a.out or os.path.join(a.final, "summary_1x1_final.png")
        t = load_final(a.summary or os.path.join(a.final, "final_summary.csv"))
        t.to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
    else:
        if not a.out:
            raise SystemExit("--out is needed without --final")
        runs = load(a.config, a.extra.split(",") if a.extra else ("",))
        if runs.empty:
            raise SystemExit("no compare CSV found")
        t = aggregate(runs)
        t.to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
        runs.to_csv(os.path.splitext(a.out)[0] + "_runs.csv", index=False)
    nrun = int(t.runs.max())
    extras = a.extra.split(",") if a.extra else [""]
    dropped = bool(a.final) or any(e.startswith("dc") for e in extras)   # drop-cache runs: dc*, final
    labels = [w[0] for w in WORKLOADS if w[0] in set(t.workload)]
    desc = {w[0]: w[1] for w in WORKLOADS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
    missing = "run pending"
    hc = bool(a.final)   # the final figure always has HCompress's row (pending until it is run)
    panel(ax[0], t, labels, "time_saved_pct", "Application time saved", missing, hc)
    panel(ax[1], t, labels, "ratio_gain_pct", "Compression ratio gain", missing, hc)
    n = len(labels)
    cm = {lab: t[t.workload == lab].cost_model.iloc[0] for lab in labels} if a.final else {}
    ax[0].set_yticks(range(n), [f"{lab}\n{desc[lab]}" + (f"\n{cm[lab]}" if a.final else "")
                                for lab in reversed(labels)])
    for tick in ax[0].get_yticklabels():
        tick.set_fontsize(10.5)
        tick.set_color(INK)
    np_t = t[t["mode"] == "learn"]
    fig.text(0.012, 0.975, "NeuroPress beats the best single codec on every workload",
             fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915,
             f"{np_t.time_saved_pct.min():.0f}–{np_t.time_saved_pct.max():.0f}% less "
             f"application time and {np_t.ratio_gain_pct.min():.0f}–"
             f"{np_t.ratio_gain_pct.max():.0f}% more compression, lossless; "
             "0 = the best single codec of each workload",
             fontsize=11.5, color=INK_2, va="top")
    if a.final:   # the setup in the header, so that the figure explains itself
        import final_config as fc
        fig.text(0.012, 0.872, f"Measured in Clio: {t.config.iloc[0]}; 1 write + {fc.READS} reads, each followed by "
                 f"one k-means iteration ({fc.KMEANS} clusters); page cache dropped before each timed read; "
                 + ("1 run per option" if nrun == 1 else f"mean of {nrun} runs per option, error bars min\u2013max"),
                 fontsize=10.5, color=INK_2, va="top")
    has_hc = hc or "hcompress" in set(t["mode"])
    handles = [matplotlib.patches.Patch(color=NP_COLOR, label="NeuroPress (learning)")]
    if has_hc:
        handles.append(matplotlib.patches.Patch(color=HC_COLOR, label="HCompress"))
    if a.final:
        handles.append(matplotlib.patches.Patch(color=TRACK_COLOR, label="time oracle (best per chunk by end-to-end time)"))
        handles.append(matplotlib.lines.Line2D([], [], ls="", marker="v", ms=7, color=COST_ORACLE,
                                               label="cost oracle (best per chunk by the cost model)"))
    else:
        handles.append(matplotlib.patches.Patch(color=TRACK_COLOR,
                                                label="oracle (best per chunk by the cost model)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.83 if a.final else 0.865),
               ncol=4 if a.final else 3,
               frameon=False, fontsize=10.5, handlelength=1.6, columnspacing=1.8)
    cfg = t.config.iloc[0]
    models = "; ".join(f"{w[0]} {t[t.workload == w[0]].cost_model.iloc[0]}" if a.final else f"{w[0]} {w[5]}"
                       for w in WORKLOADS if w[0] in labels)
    final_note = ("Cost model: cost = w_ct \u00d7 compress ms + w_dt \u00d7 decompress ms + w_io \u00d7 stored bytes / "
                  "bandwidth, weights (w_ct/w_dt/w_io) and bandwidth per workload as in the row labels. Best static (the 0 "
                  "line) = the one setting with the lowest total cost; NeuroPress = online learning, no exploration; cost "
                  "oracle = each chunk's lowest-cost setting; time oracle = each chunk's lowest end-to-end time setting at "
                  "the measured disk speed (from the exhaustive search). Every chunk verified bit-exact.")
    fig.text(0.012, 0.015, final_note if a.final else
             f"{cfg}; 1 write + 10 reads, each read followed by one k-means iteration; every "
             f"chunk verified bit-exact"
             + ("; page cache dropped before each timed read" if dropped else "")
             + (f"; mean of {nrun} runs, error bars min\u2013max" if nrun > 1 else "")
             + ((f" ({nrun} of {len(extras)} runs done so far)" if int(t.runs.min()) == nrun
                 else f" ({int(t.runs.min())}\u2013{nrun} of {len(extras)} runs done so far)")
                if len(extras) > 1 and int(t.runs.min()) < len(extras) and not a.final else "")
             + f". Each option selects by the workload's cost model "
             f"(w_ct/w_dt/w_io at bandwidth): {models}."
             + ("" if has_hc or a.final else " HCompress was run at 1 process x 1 chunk only."),
             fontsize=8.5, color=INK_2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.1 if a.final else 0.06, 1, 0.76 if a.final else 0.80))
    fig.savefig(a.out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    print("wrote", os.path.abspath(a.out))
    with pd.option_context("display.width", 200):
        print(t[["workload", "mode", "app_s", "ratio", "time_saved_pct", "ratio_gain_pct",
                 "bit_exact"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

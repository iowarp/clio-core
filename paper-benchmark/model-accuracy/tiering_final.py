#!/usr/bin/env python3
"""Tiering for one workload of the final setup (offline model from the stored exhaustive
search, no run): the paper's configuration steps with the workload's cost model, and both
oracles.

    tiering_final.py WORKLOAD [--fig-root DIR] [--summary CSV]

Parameters from final_config.py. Single tier: every chunk written to and read from the
SINGLE_TIER (the burst buffer). Tiered: write-through to the burst buffer (100 %), with DRAM and
NVMe copies of TIERED_COPIES % of the chunks (dealt round-robin in write order) serving the
reads; the PFS only receives a background flush. 1 write + READS reads (the k-means read
pattern; k-means compute is not modelled). Options: best static (lowest total cost under the
cost model), NeuroPress (offline replay of Clio's selection and learning, no exploration,
paying its measured selection and training time), cost oracle (each chunk's lowest-cost
candidate setting) and time oracle (each chunk's lowest end-to-end time candidate setting).
Async overlaps codec work with the transfers (perfect overlap), given to every option.
The measured single-tier result (--summary, default FIG_ROOT/final_summary.csv of final_check.py)
is the anchor row. The figure's title and subtitle state the cost model, the tier speeds and the
composition; the CSV's 'setup' column states each row's scenario.

Writes FIG_ROOT/<workload>/tiering_<workload>.{png,csv}.
"""
import argparse
import os
import textwrap

import matplotlib
import numpy as np
import pandas as pd

import eval_v2_workloads as ev
import final_config as fc
import replay_learning as rl
from plot_workload_summary import COST_ORACLE, NP_COLOR, TRACK_COLOR, TRACK_INK

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRAM, NVME = fc.TIER_BW["DRAM"], fc.TIER_BW["NVMe"]
BB = fc.TIER_BW[fc.SINGLE_TIER]
INK, INK_2, GRID = "#1a1a19", "#52514e", "#e6e5e1"
# NeuroPress in red tones and the oracles as in the summary figure (cost oracle: its marker's
# colour; time oracle: the light track)
COL = {"Baseline": "#b9b8b3", "Best static": "#9cc0ec", "NeuroPress": "#eda0aa",
       "Best static + tiering": "#2a78d6", "NeuroPress + tiering": "#e07082",
       "Best static + tiering + async": "#174c8f", "NeuroPress + tiering + async": "#d1495b",
       "Cost oracle + tiering + async": "#4a3aa7", "Time oracle + tiering + async": "#f4d3d8"}


def read_bw(n, slots=20):
    """@return each chunk's read bandwidth: the DRAM / NVMe copies (TIERED_COPIES %), the rest
    the burst buffer, interleaved in write order (as Clio's round-robin tiers)."""
    counts = [round(fc.TIERED_COPIES["DRAM"] * slots / 100), round(fc.TIERED_COPIES["NVMe"] * slots / 100)]
    counts.append(slots - sum(counts))
    bws = (DRAM, NVME, BB)
    pos = sorted(((j + 0.5) / c, t) for t, c in enumerate(counts) for j in range(c))
    pat = np.array([bws[t] for _, t in pos])
    return pat[np.arange(n) % len(pat)]


def stored(meas, nb, store):
    """@return stored bytes of every (chunk, setting); a setting that does not shrink stores raw."""
    r = meas[..., 2]
    kept = r > 1
    st = np.where(kept, nb[:, None] / np.where(kept, r, 1.0), nb[:, None])
    st[:, store] = nb
    return st, kept


def cost_matrix(meas, nb, sbw, w, store):
    """@return the cost of every (chunk, setting) at each chunk's bandwidth sbw, candidates only."""
    st, kept = stored(meas, nb, store)
    c = w[0] * meas[..., 0] + w[1] * np.where(kept, meas[..., 1], 0.0) + w[2] * st / sbw[:, None]
    c[:, store] = w[2] * nb / sbw
    return ev.for_selection(np.where(np.isfinite(c), c, np.nan))


def time_matrix(meas, nb, wbw, rbw, store):
    """@return the end-to-end time of every (chunk, setting): compress + write I/O + READS x
    (decompress + read I/O), candidates only."""
    st, kept = stored(meas, nb, store)
    t = meas[..., 0] + st / wbw[:, None] + fc.READS * (np.where(kept, meas[..., 1], 0.0) + st / rbw[:, None])
    t[:, store] = nb / wbw + fc.READS * nb / rbw
    return ev.for_selection(np.where(np.isfinite(t), t, np.nan))


def parts(meas, nb, pick, store, extra):
    """@return (stored bytes, compress ms incl. extra work, decompress ms) per chunk of one pick."""
    i = np.arange(len(pick))
    ct, dt, r = meas[i, pick].T
    raw = (pick == store) | ~(r > 1)
    st = np.where(raw, nb, nb / np.where(raw, 1.0, r))
    return st, np.where(pick == store, 0.0, ct) + extra, np.where(raw, 0.0, dt)


def e2e(st, ct, dt, wbw, rbw, overlap):
    """@return end-to-end s of 1 write + READS reads (serial, or async with perfect overlap)."""
    w_io, r_io = (st / wbw).sum(), (st / rbw).sum()
    if overlap:
        return (max(ct.sum(), w_io) + fc.READS * max(dt.sum(), r_io)) / 1e3
    return (ct.sum() + w_io + fc.READS * (dt.sum() + r_io)) / 1e3


def neuropress(wl, data, sbw):
    """@return (NeuroPress's pick per chunk, its own work in ms per chunk) at bandwidths sbw."""
    names, store, order, meas, cost = data
    ds, w, _ = fc.WORKLOADS[wl]
    o2 = order.copy()
    o2["tier_bw"] = sbw
    rec, _, _ = rl.replay(ds, "nlms", 0.5, thr=rl.MAPE_THRESHOLD, data=(names, store, o2, meas, cost), w=w, bw=None)
    return rec.pick.to_numpy(), fc.SEL_MS[wl] + fc.TRAIN_MS * rec.trained.to_numpy(float)


def scenario(wl, data, sbw, wbw, rbw, suffix, out):
    """Add the four options of one scenario (single tier or tiered) to out."""
    names, store, order, meas, _ = data
    w = fc.WORKLOADS[wl][1]
    nb = order.bytes.to_numpy(float)
    n = len(nb)
    zero = np.zeros(n)
    c = cost_matrix(meas, nb, sbw, w, store)
    s = int(np.nanargmin(np.nansum(np.where(np.isnan(c), np.inf, c), axis=0)))
    p, extra = neuropress(wl, data, sbw)
    picks = (("Best static", np.full(n, s), zero), ("NeuroPress", p, extra),
             ("Cost oracle", np.nanargmin(c, axis=1), zero),
             ("Time oracle", np.nanargmin(time_matrix(meas, nb, wbw, rbw, store), axis=1), zero))
    for name, pk, ex in picks:
        st, ct, dt = parts(meas, nb, pk, store, ex)
        setting = names[s] if name == "Best static" else ""
        out[name + suffix] = (e2e(st, ct, dt, wbw, rbw, False), nb.sum() / st.sum(), setting)
        if suffix:
            out[name + suffix + " + async"] = (e2e(st, ct, dt, wbw, rbw, True), nb.sum() / st.sum(), "")


def run(wl, data):
    """@return {config: (time s, ratio, best static setting)}."""
    names, store, order, meas, _ = data
    nb = order.bytes.to_numpy(float)
    n = len(nb)
    bb = np.full(n, BB)
    st, ct, dt = parts(meas, nb, np.full(n, store), store, np.zeros(n))
    out = {"Baseline": (e2e(st, ct, dt, bb, bb, False), 1.0, "store")}
    scenario(wl, data, bb, bb, bb, "", out)                       # single tier: the burst buffer
    rbw = read_bw(n)                                              # tiered: write-through to the BB
    sbw = (1 + fc.READS) / (1.0 / BB + fc.READS / rbw)            # what a chunk sees over 1 write + READS reads
    scenario(wl, data, sbw, bb, rbw, " + tiering", out)
    return out


def measured(path, wl):
    """@return ({option: (time saved %, ratio gain %)}, runs per option) of the measured runs in
    the summary CSV at path (final_check.py's schema), or ({}, 0)."""
    if not os.path.exists(path):
        return {}, 0
    s = pd.read_csv(path)
    s = s[s.workload == wl].set_index("option")
    gains = {o: (-s.loc[o, "time_vs_static_pct"], s.loc[o, "ratio_vs_static_pct"]) for o in s.index if o != "best static"}
    return gains, int(s.runs.min()) if len(s) else 0


def style(ax):
    """Light axes: faint x grid, no top / right spines."""
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8.5)


def gain_rows(d, meas):
    """@return the NeuroPress-vs-best-static rows: the measured run (when meas has it) and the
    model's single tier, tiered and tiered + async scenarios, each as (label, NP time saved %,
    NP ratio gain %, cost oracle (time saved, ratio gain), time oracle (time saved, ratio gain)).

    @param d the tiering table indexed by config; @param meas {option: (time saved %, ratio gain %)}"""
    gain = lambda x, y: -100 * (d.loc[x, "time_s"] / d.loc[y, "time_s"] - 1)
    rgain = lambda x, y: 100 * (d.loc[x, "ratio"] / d.loc[y, "ratio"] - 1)
    rows = []
    if "NeuroPress" in meas:
        rows.append(("measured run (1 tier, real disk)", *meas["NeuroPress"], meas.get("cost oracle"), meas.get("time oracle")))
    for label, sfx, ref in (("model: single tier (burst buffer)", "", "Best static"),
                            ("model: tiered", " + tiering", "Best static + tiering"),
                            ("model: tiered + async", " + tiering + async", "Best static + tiering + async")):
        rows.append((label, gain("NeuroPress" + sfx, ref), rgain("NeuroPress" + sfx, ref),
                     (gain("Cost oracle" + sfx, ref), rgain("Cost oracle" + sfx, ref)),
                     (gain("Time oracle" + sfx, ref), rgain("Time oracle" + sfx, ref))))
    return rows


def gain_panel(ax, rows, j, title):
    """One panel of NeuroPress vs best static, as in the summary figure: NeuroPress's bar on the
    time oracle's light track, the cost oracle as a marker. j: 0 = time saved, 1 = ratio gain."""
    vals = [v for x in rows for v in (x[1 + j], (x[3] or (0, 0))[j], (x[4] or (0, 0))[j])]
    lo, hi = min(0, min(vals)), max(vals)
    span = hi - lo + 1e-9
    pad = 0.02 * span
    yy = np.arange(len(rows))[::-1]
    for y, row in zip(yy, rows):
        v, co, to = row[1 + j], row[3], row[4]
        if to is not None:
            ax.barh(y, to[j], 0.5, color=TRACK_COLOR, zorder=1)
        ax.barh(y, v, 0.34, color=NP_COLOR, zorder=2)
        if co is not None:
            ax.plot(co[j], y + 0.3, "v", ms=7, color=COST_ORACLE, mec="white", mew=0.8, zorder=5)
        inside = abs(v) >= 0.3 * span    # room for the bold label inside the bar
        x = v - pad if inside else (v + pad if v >= 0 else v - pad)
        ax.text(x, y, f"{v:+.0f}%", va="center", ha="right" if inside or v < 0 else "left", fontsize=10,
                fontweight="bold", color="white" if inside else INK, zorder=3)
        if to is not None:
            end = max(to[j], v if inside else v + 0.24 * span)
            ax.text(end + pad, y, f"{to[j]:+.0f}%", va="center", ha="left", fontsize=8.5, color=TRACK_INK, zorder=3)
    ax.axhline(yy[0] - 0.5, color=GRID, lw=1, ls="--")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(lo - (0.05 * span if lo < 0 else 0), hi + 0.3 * span)
    ax.set_yticks(yy, [x[0] for x in rows] if j == 0 else [], fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    ax.set_xlabel("% vs best static (positive = better)", color=INK_2, fontsize=8.5)
    style(ax)


def plot(d, wl, meas, runs, out):
    """Left: time per configuration step and the two oracles; right: NeuroPress vs best static
    (time saved, ratio gain) with the oracles encoded as in the summary figure."""
    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.6], wspace=0.42)
    l = fig.add_subplot(gs[0])
    steps = list(COL)
    y = np.arange(len(steps))[::-1]
    base = d.loc["Baseline", "time_s"]
    for yy, s_ in zip(y, steps):
        v = d.loc[s_, "time_s"]
        to = s_.startswith("Time")   # the time oracle: its track colour, hatched so that it stands out
        l.barh(yy, v, 0.68, color=COL[s_], edgecolor=TRACK_INK if to else "none", lw=1, hatch="////" if to else None)
        l.text(v + base * 0.015, yy, f"{v:,.0f} s" if s_ == "Baseline" else f"{v:,.0f} s  {100 * (v / base - 1):+.0f}%",
               va="center", fontsize=8.5, color=INK, fontweight="bold" if s_.startswith("Neuro") else "normal")
    l.set_yticks(y, steps, fontsize=9, color=INK)
    l.set_xlim(0, base * 1.3)
    l.set_title("End-to-end time per configuration", loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    l.set_xlabel(f"end-to-end time (s), 1 write + {fc.READS} reads (offline model)", color=INK_2, fontsize=8.5)
    style(l)
    rows = gain_rows(d, meas)
    sub = gs[1].subgridspec(1, 2, wspace=0.1)
    m, r = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    gain_panel(m, rows, 0, "Time saved")
    gain_panel(r, rows, 1, "Compression ratio gain")
    fig.legend(handles=[matplotlib.patches.Patch(color=NP_COLOR, label="NeuroPress"),
                        matplotlib.patches.Patch(color=TRACK_COLOR, label="time oracle (best per chunk by end-to-end time)"),
                        matplotlib.lines.Line2D([], [], ls="", marker="v", ms=7, color=COST_ORACLE,
                                                label="cost oracle (best per chunk by the cost model)")],
               loc="lower right", bbox_to_anchor=(0.9, 0.925), ncol=3, frameon=False, fontsize=9)
    ds, w, bw = fc.WORKLOADS[wl]
    st1, st2 = d.loc["Best static", "setting"], d.loc["Best static + tiering", "setting"]
    fig.suptitle(f"{wl}: NeuroPress with tiering, cost model w_ct/w_dt/w_io = {fc.model_name(w)}; tier speeds DRAM "
                 f"{DRAM / 1e6:g}, NVMe {NVME / 1e6:g}, burst buffer {BB / 1e6:g}, PFS {fc.TIER_BW['PFS'] / 1e6:g} GB/s",
                 x=0.125, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.17)
    sub = (f"Single tier: every chunk on the burst buffer. Tiered: write-through to the burst buffer (100 %), "
           f"copies of {fc.TIERED_COPIES['DRAM']} % of the chunks on DRAM and {fc.TIERED_COPIES['NVMe']} % on NVMe (dealt "
           f"round-robin), reads from the fastest copy, PFS flush in the background. 1 write + {fc.READS} reads; every "
           f"option selects with each chunk's tier speed.",
           f"Measured row: Clio on the test disk, cost bandwidth {bw / 1e6:g} GB/s, 1 process, 1 chunk in flight, page "
           f"cache dropped before each read, k-means after each read, "
           + ("1 run per option." if runs == 1 else f"mean of {runs} runs per option."))
    fig.text(0.125, 1.03, "\n".join(textwrap.fill(x, 175) for x in sub), fontsize=9.5, color=INK_2, va="bottom")
    fig.text(0.125, -0.1, f"Offline model from the measured exhaustive search, except the measured row. Best static (the 0 "
             f"line) = the one setting with the lowest total cost: {st1} (single tier), {st2} (tiered). NeuroPress = replay "
             f"of Clio's selection and learning (no exploration), paying its measured selection and training time. Cost "
             f"oracle = each chunk's lowest-cost setting; time oracle = each chunk's lowest end-to-end time setting. Async: "
             f"perfect overlap of codec work and transfers, no simulation compute.", fontsize=8, color=INK_2, wrap=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return rows


def setup(config):
    """@return the scenario of one configuration row, in words (the CSV's 'setup' column)."""
    tiers = (f"tiered: write-through to the burst buffer ({BB / 1e6:g} GB/s, 100 %), DRAM ({DRAM / 1e6:g} GB/s) "
           f"copies of {fc.TIERED_COPIES['DRAM']} % and NVMe ({NVME / 1e6:g} GB/s) copies of {fc.TIERED_COPIES['NVMe']} % "
             f"of the chunks, round-robin")
    if "tiering" not in config:
        text = f"single tier: burst buffer {BB / 1e6:g} GB/s" + (", no compression" if config == "Baseline" else "")
    else:
        text = tiers + ("; async (perfect overlap of codec work and transfers)" if "async" in config else "")
    return f"{text}; 1 write + {fc.READS} reads; offline model"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workload", choices=list(fc.WORKLOADS))
    ap.add_argument("--fig-root", default=os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning", "final"))
    ap.add_argument("--summary", default=None, help="measured summary CSV (default FIG_ROOT/final_summary.csv)")
    a = ap.parse_args()
    wl = a.workload
    data = rl.load(fc.WORKLOADS[wl][0])
    res = run(wl, data)
    t = pd.DataFrame([{"workload": wl, "cost_model": fc.model_name(fc.WORKLOADS[wl][1]), "main": True, "config": k,
                       "setup": setup(k), "time_s": v[0], "ratio": v[1], "setting": v[2]} for k, v in res.items()])
    out_dir = os.path.join(a.fig_root, wl.lower())
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"tiering_{wl.lower()}")
    t.to_csv(stem + ".csv", index=False)
    gains, runs = measured(a.summary or os.path.join(a.fig_root, "final_summary.csv"), wl)
    for row in plot(t.set_index("config"), wl, gains, runs, stem + ".png"):
        co = f", cost oracle {row[3][0]:+.1f}%" if row[3] else ""
        to = f", time oracle {row[4][0]:+.1f}%" if row[4] else ""
        print(f"  {row[0]:30s} NP time saved {row[1]:+.1f}%, ratio gain {row[2]:+.1f}%{co}{to}")
    print("wrote", stem + ".png")


if __name__ == "__main__":
    main()

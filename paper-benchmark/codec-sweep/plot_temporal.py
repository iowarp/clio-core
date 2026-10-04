#!/usr/bin/env python3
"""Figures for temporal_sweep results (lossless look-ahead study).

    plot_temporal.py --dir ~/np-temporal/nyx --out DIR [--setting "zstd shuffle=byte"]

Writes three PNGs:
  modes_by_codec.png   bytes relative to independent coding, per codec setting
                       and temporal mode, T=4 and T=8 side by side
  lookahead_frames.png per-frame bytes vs independent under the look-ahead plan
                       (T=8), coloured by how the frame is coded
  phase.png            the same ratio split by where in the run the window sits
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = {  # categorical slots, fixed order (reference palette, light mode)
    "idelta_prev": "#2a78d6", "idelta_la": "#eb6834",
    "xor_prev": "#1baf7a", "xor_la": "#eda100",
}
LABEL = {"idelta_prev": "int delta vs previous", "idelta_la": "int delta, look-ahead",
         "xor_prev": "XOR vs previous", "xor_la": "XOR, look-ahead"}
KIND_COLOR = {"intra": "#52514e", "forward": "#2a78d6", "bidir": "#eb6834"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(d):
    out = {}
    for T in (4, 8):
        p = os.path.join(d, f"temporal_T{T}.csv")
        if not os.path.exists(p):
            continue
        f = pd.read_csv(p)
        f = f[f.ok == 1].copy()
        f["setting"] = (f.algorithm + " " + f.settings.fillna("")).str.strip()
        out[T] = f
    return out


def rel_bytes(f):
    """bytes vs indep per (setting, mode), summed over groups."""
    rows = []
    for st, g in f.groupby("setting"):
        base = g[g["mode"] == "indep"].set_index("group").comp_bytes
        for mode, gm in g.groupby("mode"):
            gm = gm.set_index("group")
            idx = base.index.intersection(gm.index)
            rows.append((st, mode, gm.loc[idx].comp_bytes.sum() / base.loc[idx].sum(),
                         gm.loc[idx].bytes_in.sum() / base.loc[idx].sum()))
    return pd.DataFrame(rows, columns=["setting", "mode", "rel", "indep_ratio"])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_modes(data, out, top=12):
    fig, axes = plt.subplots(1, len(data), figsize=(5.8 * len(data), 6.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (T, f) in zip(axes, sorted(data.items())):
        r = rel_bytes(f)
        order = (r[r["mode"] == "indep"].sort_values("indep_ratio", ascending=False)
                 .setting.head(top).tolist())[::-1]
        modes = [m for m in SERIES if m in set(r["mode"])]
        h = 0.8 / len(modes)
        y = np.arange(len(order))
        for i, m in enumerate(modes):
            v = r[r["mode"] == m].set_index("setting").rel.reindex(order).values
            ax.barh(y + (i - (len(modes) - 1) / 2) * h, (v - 1) * 100, height=h * 0.9,
                    color=SERIES[m], label=LABEL[m], edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=TEXT, linewidth=0.8)
        ax.set_yticks(y)
        ratio = r[r["mode"] == "indep"].set_index("setting").indep_ratio.reindex(order)
        ax.set_yticklabels([f"{s}   ({ratio[s]:.1f}x)" for s in order], fontsize=8, color=TEXT)
        ax.set_xlabel("bytes vs independent coding (%)   <- smaller is better", color=MUTED, fontsize=9)
        ax.set_title(f"Nyx Sedov 256³, {T}-frame windows", fontsize=10, color=TEXT, loc="left")
        style(ax)
        ax.set_xlim(-15, 15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Lossless temporal coding vs independent per-frame coding "
                 "(label: codec setting, its independent ratio)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "modes_by_codec.png"), dpi=160)
    plt.close(fig)


def plan_kinds(T):
    kinds = {0: "intra", T - 1: "forward"}
    def mid(lo, hi):
        if hi - lo < 2:
            return
        m = (lo + hi) // 2
        kinds[m] = "bidir"
        mid(lo, m); mid(m, hi)
    mid(0, T - 1)
    return kinds


def fig_frames(data, out, setting):
    if 8 not in data:
        return
    f = data[8]
    s = f[f.setting == setting]
    fb = lambda r: np.array([int(x) for x in str(r).split(";")])
    base = s[s["mode"] == "indep"].set_index("group").frame_bytes.map(fb)
    B = np.vstack(base.values)
    kinds = plan_kinds(8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, mode in zip(axes, ["idelta_la", "idelta_prev"]):
        m = s[s["mode"] == mode].set_index("group").frame_bytes.map(fb)
        M = np.vstack(m.loc[base.index].values)
        rel = (M.sum(0) / B.sum(0) - 1) * 100
        kk = kinds if mode.endswith("_la") else {t: ("intra" if t == 0 else "forward") for t in range(8)}
        cols = [KIND_COLOR[kk[t]] for t in range(8)]
        ax.bar(range(8), rel, color=cols, width=0.7, edgecolor="white")
        for t in range(8):
            ax.text(t, rel[t] + (0.8 if rel[t] >= 0 else -0.8), f"{rel[t]:+.1f}", ha="center",
                    va="bottom" if rel[t] >= 0 else "top", fontsize=8, color=TEXT)
        ax.axhline(0, color=TEXT, linewidth=0.8)
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"t{t}" for t in range(8)], color=TEXT)
        ax.set_title(LABEL[mode], fontsize=10, color=TEXT, loc="left")
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.grid(axis="x", visible=False)
    lo = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_ylim(min(lo, -8) - 2, ax.get_ylim()[1] + 2)
    axes[0].set_ylabel("bytes vs independent (%)", color=MUTED, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLOR.values()]
    axes[1].legend(handles, ["intra (coded alone)", "forward (from one earlier frame)",
                             "bidirectional (from both sides)"], frameon=False, fontsize=8)
    fig.suptitle(f"Per-frame cost of the look-ahead plan, Nyx 8-frame windows, {setting}",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lookahead_frames.png"), dpi=160)
    plt.close(fig)


def fig_phase(data, out, setting):
    fig, axes = plt.subplots(1, len(data), figsize=(4.8 * len(data), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (T, f) in zip(axes, sorted(data.items())):
        s = f[f.setting == setting].copy()
        s["dump"] = s.group.str.extract(r"@plt(\d+)#").astype(int)
        edges = [-1, 16, 33, 60]
        s["phase"] = pd.cut(s.dump, edges, labels=["early", "middle", "late"])
        base = s[s["mode"] == "indep"].set_index("group").comp_bytes
        x = np.arange(3)
        for m in SERIES:
            vals = []
            for ph in ["early", "middle", "late"]:
                g = s[(s["mode"] == m) & (s.phase == ph)].set_index("group")
                vals.append((g.comp_bytes.sum() / base.loc[g.index].sum() - 1) * 100)
            ax.plot(x, vals, "-o", color=SERIES[m], label=LABEL[m], linewidth=2, markersize=6)
        ind = [s[(s["mode"] == "indep") & (s.phase == ph)].pipe(lambda g: g.bytes_in.sum() / g.comp_bytes.sum())
               for ph in ["early", "middle", "late"]]
        ax.axhline(0, color=TEXT, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n(indep {r:.1f}x)" for p, r in zip(["early", "middle", "late"], ind)],
                           color=TEXT, fontsize=8)
        ax.set_title(f"{T}-frame windows", fontsize=10, color=TEXT, loc="left")
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("bytes vs independent (%)", color=MUTED, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Where in the run temporal coding helps, Nyx, {setting}", fontsize=10,
                 color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", default="zstd shuffle=byte")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(a.dir)
    fig_modes(data, a.out)
    fig_frames(data, a.out, a.setting)
    fig_phase(data, a.out, a.setting)
    print("wrote", os.listdir(a.out))


if __name__ == "__main__":
    main()

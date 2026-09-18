#!/usr/bin/env python3
"""Figure 9: end-to-end wall-clock time per workload, lower is better.

(a) the ablation and (b) NeuroPress against fixed codecs. Every bar is an
independent run; its pale upper segment is that run's own I/O time.

Usage:
  ./plot_fig9.py                            # defaults, writes figures/
  ./plot_fig9.py --csv a.csv [--csv b.csv]  # measured, merged per workload
  ./plot_fig9.py --write-template fig9.csv
"""
import argparse, csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# STYLE CONSTANTS -- everything tweakable lives here
# ----------------------------------------------------------------------------
FIG_W, FIG_H = 7.16, 3.8          # IEEE two-column figure* width, inches
FS_AXIS, FS_TICK, FS_LEG, FS_VAL = 9, 8, 7.5, 7    # nothing below 7 pt
Y_CLIP = 6.5                       # MINUTES; cuSZ/VPIC at 14.2 min must not set this
Y_AUTO_BELOW = 0.5                 # if max(total) < Y_CLIP*this, rescale to the
                                   # data -- a smoke run is ~4 s and would
                                   # otherwise be invisible against a 390 s axis
IO_BLEND = 0.60                    # I/O segment blended this far toward white
BAR_PAD = 0.18                     # fraction of the group width left as gutter

WORKLOADS = ["VPIC", "Nyx", "LAMMPS", "WarpX", "AI"]

# tab10 mapping carried over from the published figure, so colours stay stable
# Untiered arms write straight to the PFS; every +Tier arm spills to NVMe. A
# codec keeps its hue across both, so the pair reads as one codec, two devices.
COLORS_A = {
    "Baseline":              "#7f7f7f",
    "nvCOMP":                "#1f77b4",
    "nvCOMP+Tier":           "#2ca02c",
    "NP only":               "#5d3fd3",
    "NP+Tier":               "#ff7f0e",
    "NP+Tier+Async":         "#9467bd",
    "NP+Tier+Async+Lossy":   "#d62728",
}
COLORS_B = {
    "Best fixed nvCOMP":      "#1f77b4",
    "Best fixed nvCOMP+Tier": "#6baed6",
    "ndzip":                  "#bcbd22",
    "ndzip+Tier":             "#dbdb8d",
    "cuSZp3":                 "#8c564b",
    "cuSZp3+Tier":            "#c49c94",
    "cuSZ":                   "#7f7f7f",
    "cuSZ+Tier":              "#c7c7c7",
    "NeuroPress":             "#d62728",
    "NeuroPress+Tier":        "#ff9896",
}
ORDER_A = list(COLORS_A)
ORDER_B = list(COLORS_B)

# The single figure carries both panels, so every bar needs its own colour: the
# panel (b) palette above deliberately reuses (a)'s hues. Panel (b)'s NeuroPress
# is no longer a duplicate: panel (b) now runs each codec both ways.
# External bars are outlined.
COLORS_B_SINGLE = {"Best fixed nvCOMP": "#08519c", "Best fixed nvCOMP+Tier": "#6baed6",
                   "ndzip": "#b15928", "ndzip+Tier": "#dbdb8d",
                   "cuSZp3": "#006d2c", "cuSZp3+Tier": "#a1d99b",
                   "cuSZ": "#252525", "cuSZ+Tier": "#969696"}
ORDER_SINGLE = ([("a", s) for s in ORDER_A] +
                [("b", s) for s in ORDER_B if s in COLORS_B_SINGLE])
COLORS_SINGLE = {**COLORS_A, **COLORS_B_SINGLE}

# Error bound per arm, as figure_9.sh sets it; the CSV's eb column overrides.
# Every lossy arm now runs at the same bound, 1e-3; the CSV's eb column wins.
EB_DEFAULT = {k: 1e-3 for k in
              ("NP+Tier+Async+Lossy", "Best fixed nvCOMP", "Best fixed nvCOMP+Tier",
               "cuSZp3", "cuSZp3+Tier", "cuSZ", "cuSZ+Tier",
               "NeuroPress", "NeuroPress+Tier")}
LOSSLESS = {"ndzip", "ndzip+Tier"}   # given the bound, but a lossless codec

# Default data, read off the published figure: (total, compute) in minutes.
# AI is an empty placeholder column.
_A = {                     # strategy -> {workload: (total, compute)}
    "Baseline":      {"VPIC": (5.5, 1.10), "Nyx": (5.5, 2.75), "LAMMPS": (6.0, 1.19), "WarpX": (5.0, 2.49)},
    "nvCOMP":        {"VPIC": (4.4, 1.14), "Nyx": (4.9, 2.85), "LAMMPS": (4.8, 1.24), "WarpX": (4.5, 2.60)},
    "nvCOMP+Tier":   {"VPIC": (3.4, 1.13), "Nyx": (4.3, 2.82), "LAMMPS": (3.7, 1.23), "WarpX": (3.9, 2.57)},
    "NP only":       {},   # no data yet
    "NP+Tier":       {"VPIC": (3.0, 1.10), "Nyx": (4.0, 2.75), "LAMMPS": (3.3, 1.19), "WarpX": (3.6, 2.49)},
    "NP+Tier+Async": {"VPIC": (2.5, 1.06), "Nyx": (3.6, 2.66), "LAMMPS": (2.8, 1.16), "WarpX": (3.3, 2.43)},
    "NP+Tier+Async+Lossy": {"VPIC": (2.2, 1.05), "Nyx": (3.4, 2.63), "LAMMPS": (2.5, 1.14), "WarpX": (3.1, 2.40)},
}

# Panel (b) defaults: old VPIC totals with no compute/I-O split (drawn hatched).
_B = {
    "Best fixed nvCOMP": {"VPIC": 3.4},
    "ndzip":             {"VPIC": 3.1},
    "cuSZp3":            {"VPIC": 2.8},
    "cuSZ":              {"VPIC": 14.2},   # clipped; real value printed above
    "NeuroPress":        {"VPIC": 2.2},
}


def default_rows():
    """Defaults as long-format rows, matching the --csv schema exactly."""
    rows = []
    for s in ORDER_A:
        for w in WORKLOADS:
            tc = _A[s].get(w)
            if tc is None:
                rows.append(dict(panel="a", workload=w, strategy=s,
                                 compute_min="", io_min="", total_min="", std_min=""))
            else:
                total, comp = tc
                rows.append(dict(panel="a", workload=w, strategy=s,
                                 compute_min=f"{comp:.2f}",
                                 io_min=f"{total - comp:.2f}",
                                 total_min=f"{total:.2f}", std_min=""))
    for s in ORDER_B:
        for w in WORKLOADS:
            t = _B[s].get(w)
            rows.append(dict(panel="b", workload=w, strategy=s,
                             compute_min="", io_min="",
                             total_min="" if t is None else f"{t:.2f}", std_min=""))
    return rows


FIELDS = ["panel", "workload", "strategy", "compute_min", "io_min", "total_min", "std_min", "ratio", "eb"]


def _f(v):
    """Parse a possibly-empty numeric cell into a float or None."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "tbd", "-"):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def index(rows):
    """rows -> {(panel, strategy, workload): {compute, io, total, std}}"""
    out, warn = {}, []
    for r in rows:
        total = _f(r.get("total_min"))
        comp, io = _f(r.get("compute_min")), _f(r.get("io_min"))
        if total is None and comp is not None and io is not None:
            total = comp + io          # total is optional when both parts given
        key = (r["panel"].strip(), r["strategy"].strip(), r["workload"].strip())
        # When merging CSVs, a TBD row never replaces a measured one.
        if total is None and out.get(key, {}).get("total") is not None:
            continue
        out[key] = dict(compute=comp, io=io, total=total, std=_f(r.get("std_min")),
                        ratio=_f(r.get("ratio")), eb=_f(r.get("eb")))
    return out, warn


def blend_to_white(hexcolor, frac):
    """Opaque blend toward white -- NOT alpha, so the PDF prints cleanly."""
    c = np.array(mpl.colors.to_rgb(hexcolor))
    return tuple(c + (np.array([1.0, 1.0, 1.0]) - c) * frac)


# ----------------------------------------------------------------------------
# SANITY CHECKS -- print to stdout, never raise
# ----------------------------------------------------------------------------
def sanity(D):
    print("== sanity checks ==")
    n = 0
    for (p, s, w), d in sorted(D.items()):
        if d["total"] is None:
            continue
        if d["compute"] is not None and d["io"] is not None:
            if abs(d["compute"] + d["io"] - d["total"]) > 0.05:
                print(f"  [split]  {p}/{w}/{s}: compute+io={d['compute']+d['io']:.3f} "
                      f"!= total={d['total']:.3f}"); n += 1

    # each added layer should not be slower than the one before it
    chains = [("nvCOMP", "nvCOMP+Tier"),
              ("NP only", "NP+Tier"), ("NP+Tier", "NP+Tier+Async"),
              ("NP+Tier+Async", "NP+Tier+Async+Lossy")]
    for w in WORKLOADS:
        for lo, hi in chains:
            a = D.get(("a", lo, w), {}).get("total")
            b = D.get(("a", hi, w), {}).get("total")
            if a is not None and b is not None and not (a >= b):
                print(f"  [order]  a/{w}: {lo} ({a:.2f}) < {hi} ({b:.2f})"); n += 1
    if n == 0:
        print("  all checks passed")
    print()


def reductions(D):
    """The percentages quoted in the caption and body text."""
    def pct(new, old):
        if new is None or old in (None, 0):
            return None
        return (old - new) / old * 100.0

    print("== percentage reductions (positive = faster) ==")
    pairs = [("NP+Tier+Async", "Baseline"), ("NP+Tier+Async", "nvCOMP+Tier"),
             ("NP only", "nvCOMP"), ("NP+Tier", "nvCOMP+Tier"),
             ("NP+Tier+Async+Lossy", "NP+Tier+Async")]
    print(f"{'workload':<9}" + "".join(f"{(a.replace('NP+Tier+Async+', '')[:15]):>17}" for a, _ in pairs))
    print(f"{'':<9}" + "".join(f"{('vs ' + b)[:15]:>17}" for _, b in pairs))
    print(f"{'':<9}" + "".join(f"{'-' * 15:>17}" for _ in pairs))
    for w in WORKLOADS:
        cells = []
        for a, b in pairs:
            r = pct(D.get(("a", a, w), {}).get("total"), D.get(("a", b, w), {}).get("total"))
            cells.append(f"{'--':>17}" if r is None else f"{r:>16.1f}%")
        print(f"{w:<9}" + "".join(cells))

    print("\n== panel (b): NeuroPress vs each external baseline ==")
    for w in WORKLOADS:
        np_t = D.get(("b", "NeuroPress", w), {}).get("total")
        if np_t is None:
            print(f"  {w:<8} --")
            continue
        parts = []
        for s in ORDER_B:
            if s == "NeuroPress":
                continue
            r = pct(np_t, D.get(("b", s, w), {}).get("total"))
            parts.append(f"{s} {'--' if r is None else f'{r:.1f}%'}")
        print(f"  {w:<8} " + ",  ".join(parts))
    print()


# ----------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------
def draw_panel(ax, D, panel, order, colors, warn_total_only, ylim, dec, show_ratio=False):
    nb = len(order)
    # Vertical labels are wider than the bar pitch once the ratio is appended.
    # Spread them over ROWS interleaved baselines so no two neighbours share one:
    # 14 arms per group need 3 rows; 2 still collided.
    # A rotated 7 pt label is ~28 px wide against a ~21 px bar pitch, so
    # neighbours collide however short the text is. Interleave their baselines:
    # 3 rows when the ratio doubles the length, 2 for a time-only label.
    rows = (3 if nb > 10 else 2) if show_ratio else (2 if nb > 6 else 1)
    width = (1.0 - BAR_PAD) / nb
    x0 = np.arange(len(WORKLOADS))

    for i, item in enumerate(order):
        # An entry is a strategy, or (panel, strategy) when one axis carries both.
        p, s = item if isinstance(item, tuple) else (panel, item)
        col = colors[s]
        pale = blend_to_white(col, IO_BLEND)
        # Mark the external codecs when they share an axis with the ablation.
        edge = "black" if isinstance(item, tuple) and p == "b" else "white"
        for j, w in enumerate(WORKLOADS):
            x = x0[j] - (1.0 - BAR_PAD) / 2 + width * (i + 0.5)
            d = D.get((p, s, w), {})
            total = d.get("total")

            # Not measured yet: leave the slot empty but keep its width, so the
            # layout is identical once the number arrives.
            if total is None:
                ax.text(x, ylim * 0.01, "TBD", ha="center", va="bottom",
                        fontsize=FS_VAL - 1, color="0.55", rotation=90)
                continue

            comp, io = d.get("compute"), d.get("io")
            label = f"{total:.{dec}f}"
            r = d.get("ratio")
            if show_ratio and r is not None:
                label += f" ({r:.0f}\u00d7)" if r >= 10 else f" ({r:.1f}\u00d7)"
            drawn = min(total, ylim)        # clipped bars stop at the limit
            clipped = total > ylim

            if comp is None or io is None:
                # Total only -- no split available. Hatch it so the figure never
                # implies a compute/I-O breakdown that was not measured.
                ax.bar(x, drawn, width=width * 0.92, color=col,
                       edgecolor=edge, linewidth=0.4, hatch="////", zorder=3)
                warn_total_only.append(f"{p}/{w}/{s}")
            else:
                cdraw = min(comp, drawn)
                ax.bar(x, cdraw, width=width * 0.92, color=col,
                       edgecolor=edge, linewidth=0.4, zorder=3)
                ax.bar(x, max(drawn - cdraw, 0.0), bottom=cdraw, width=width * 0.92,
                       color=pale, edgecolor=edge, linewidth=0.4, zorder=3)

            if d.get("std") is not None and not clipped:
                ax.errorbar(x, total, yerr=d["std"], ecolor="black",
                            elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=5)

            if clipped:
                # Two diagonal white slashes: the axis-break convention.
                for dy in (ylim * 0.025, ylim * 0.046):
                    ax.plot([x - width * 0.46, x + width * 0.46],
                            [drawn - dy - ylim * 0.011, drawn - dy + ylim * 0.011],
                            color="white", lw=1.1, solid_capstyle="butt",
                            zorder=6, clip_on=False)

            if clipped:
                # Printed inside the axes: a clipped bar already reaches the
                # top, so a label above it would collide with the panel above.
                ax.text(x, ylim * 0.98, label, ha="center", va="top",
                        rotation=90, fontsize=FS_VAL, zorder=7,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white",
                                  ec="none", alpha=0.85))
            else:
                lift = ylim * (0.012 + (0.075 if show_ratio else 0.055) * (i % rows))
                ax.text(x, drawn + lift, label, ha="center",
                        va="bottom", rotation=90, fontsize=FS_VAL, zorder=7)

    ax.set_xticks(x0)
    ax.set_xlim(-0.5, len(WORKLOADS) - 0.5)
    ax.set_ylim(0, ylim)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.75", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append",
                    help="long-format CSV to plot instead of the defaults; repeat "
                         "to merge per-workload runs into one figure")
    ap.add_argument("--write-template", metavar="PATH",
                    help="write the default data as a CSV and exit")
    ap.add_argument("--out", default="figures", help="output directory (default: figures)")
    ap.add_argument("--ylim", type=float, default=None,
                    help=f"y-axis limit in MINUTES (default: {Y_CLIP:g}, auto-rescaled "
                         "when the data is far below it, e.g. a smoke run)")
    args = ap.parse_args()

    if args.write_template:
        os.makedirs(os.path.dirname(os.path.abspath(args.write_template)) or ".", exist_ok=True)
        with open(args.write_template, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=FIELDS)
            wtr.writeheader()
            wtr.writerows(default_rows())
        print(f"wrote template: {args.write_template}")
        return 0

    rows = [r for c in args.csv for r in load(c)] if args.csv else default_rows()
    D, _ = index(rows)
    # The CSVs are already in minutes, and so is the figure -- no conversion.

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    totals = [d["total"] for d in D.values() if d["total"] is not None]
    dmax = max(totals) if totals else Y_CLIP
    if args.ylim is not None:
        ylim = args.ylim
    elif dmax < Y_CLIP * Y_AUTO_BELOW:
        ylim = dmax * 1.52          # smoke-scale data: show it, do not clip it
                                    # (1.52, not 1.28: the ratio makes the
                                    #  vertical bar labels about twice as long,
                                    #  and they are staggered over two rows)
        print(f"note: max total {dmax:.4g} min is far below the {Y_CLIP:g} min "
              f"paper limit; y-axis rescaled to {ylim:.4g}. Pass --ylim to override.\n")
    else:
        # Fit the data rather than clip it: at full scale WarpX reaches 8.6 min,
        # so the 6.5 min paper limit truncated five bars and pushed their labels
        # into the legend. --ylim still forces a fixed limit for the paper.
        ylim = dmax * 1.30          # time-only labels need little headroom
    dec = 1 if ylim >= 2 else (2 if ylim >= 0.3 else 3)

    def eps(e):
        k = math.log10(e)
        return (rf"$\varepsilon = 10^{{{round(k)}}}$" if abs(k - round(k)) < 1e-9
                else rf"$\varepsilon = {e:g}$")

    def legend_label(s):
        if s in LOSSLESS:
            return f"{s} (lossless)"
        seen = [d["eb"] for (p_, w_, s_), d in D.items() if s_ == s and d.get("eb") is not None]
        e = seen[0] if seen else EB_DEFAULT.get(s, 0.0)
        if not e:
            return s
        return f"{s.replace(' (low)', '').replace(' (med)', '').replace(' (high)', '')}, {eps(e)}"

    # One figure per panel, on the same y-axis so the two stay comparable.
    os.makedirs(args.out, exist_ok=True)
    warn_total_only, pngs = [], []
    for panel, order, colors, height, title, name in (
            ("a", ORDER_A, COLORS_A, 2.3,
             "(a) Ablation  --  each bar is a separate run; lossless unless $\\varepsilon$ is shown",
             "fig9a_ablation.png"),
            ("b", ORDER_B, COLORS_B, 2.0, "(b) External baselines",
             "fig9b_baselines.png")):
        fig, ax = plt.subplots(figsize=(FIG_W, height))
        draw_panel(ax, D, panel, order, colors, warn_total_only, ylim, dec)
        ax.set_ylabel("Total wall-clock time (min)", fontsize=FS_AXIS)
        ax.set_xticklabels(WORKLOADS, fontsize=FS_TICK)
        fig.subplots_adjust(top=0.99)
        fig.legend(handles=[Patch(facecolor=colors[s_], label=legend_label(s_)) for s_ in order],
                   loc="lower left", bbox_to_anchor=(0.005, 0.995),
                   ncol=3 if panel == "a" else 5,   # (a): nvCOMP | NP | NP lossy
                   fontsize=FS_LEG, frameon=False, handlelength=1.3, handleheight=0.9,
                   columnspacing=1.0, labelspacing=0.35, title=title,
                   title_fontsize=FS_LEG, alignment="left")
        # Segment key in neutral gray: a convention, not a strategy.
        fig.legend(handles=[Patch(facecolor="#808080", label="Write loop: stats, NN, quantize, codec,\n"
                              "tier put, setup, scheduling"),
                            Patch(facecolor=blend_to_white("#808080", IO_BLEND),
                                  label="Input read + final flush")],
                   loc="lower right", bbox_to_anchor=(0.998, 0.995), ncol=1,
                   fontsize=FS_LEG, frameon=False, handlelength=1.3, handleheight=0.9)
        pngs.append(os.path.join(args.out, name))
        fig.savefig(pngs[-1], dpi=300, bbox_inches="tight")
        plt.close(fig)

    # The same data as one figure: the ablation and the external codecs on one
    # axis, so every arm is read against the same bars.
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    draw_panel(ax, D, None, ORDER_SINGLE, COLORS_SINGLE, warn_total_only, ylim, dec)
    ax.set_ylabel("Total wall-clock time (min)", fontsize=FS_AXIS)
    ax.set_xticklabels(WORKLOADS, fontsize=FS_TICK)
    fig.subplots_adjust(top=0.99)
    fig.legend(handles=[Patch(facecolor=COLORS_SINGLE[s_], label=legend_label(s_),
                              edgecolor="black" if p_ == "b" else "white", linewidth=0.5)
                        for p_, s_ in ORDER_SINGLE],
               loc="lower left", bbox_to_anchor=(0.005, 0.995), ncol=3,
               fontsize=FS_LEG, frameon=False, handlelength=1.3, handleheight=0.9,
               columnspacing=1.0, labelspacing=0.35,
               title="Each bar is a separate run; lossless unless $\\varepsilon$ is shown; "
                     "outlined = external codec\n"
                     "Solid = the measured write loop; light = the input read and final flush.\n"
                     "EXCLUDED: H2D staging, and simulate time for LAMMPS",
               title_fontsize=FS_LEG, alignment="left")
    fig.legend(handles=[Patch(facecolor="#808080", label="Write loop: stats, NN, quantize, codec,\n"
                              "tier put, setup, scheduling"),
                        Patch(facecolor=blend_to_white("#808080", IO_BLEND),
                              label="Input read + final flush")],
               loc="lower right", bbox_to_anchor=(0.998, 0.995), ncol=1,
               fontsize=FS_LEG, frameon=False, handlelength=1.3, handleheight=0.9)
    pngs.append(os.path.join(args.out, "fig9.png"))
    fig.savefig(pngs[-1], dpi=300, bbox_inches="tight")
    plt.close(fig)

    if warn_total_only:
        print(f"WARNING: {len(warn_total_only)} bar(s) had a total but no "
              f"compute/I-O split; drawn hatched: {', '.join(warn_total_only)}\n")
    sanity(D)
    reductions(D)
    print("wrote " + ", ".join(pngs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Figure 9: end-to-end wall-clock time per workload, lower is better.

(a) the ablation ladder and (b) NeuroPress against fixed codecs. Every bar is
one run: the solid segment is its write loop, the pale one the input read and
the final flush (see fig9.md, beside this file, for what that does and does
not mean).

THE ARMS, THE WORKLOADS AND THE ERROR BOUNDS ALL COME FROM THE CSV. This script
carries no data of its own and no fixed arm list -- an earlier version did, and
a campaign whose arm names it did not happen to list lost those bars silently.

Usage:
  ./plot_fig9.py --csv fig9.csv [--csv other.csv]   # merged per workload
  ./plot_fig9.py --write-template fig9.csv          # an empty CSV, header only
"""
import argparse, csv, math, os, re, sys
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

# Preferred left-to-right column order. A workload present in the CSV but not
# named here is appended in the order it first appears, so a new one plots
# without editing this file.
WORKLOAD_ORDER = ["VPIC", "Nyx", "LAMMPS", "WarpX", "AI"]

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
    # The lossy ladder as the published figure drew it: one hue per bound, not
    # three shades of one. Listed rather than derived so a re-plot reproduces
    # the campaign's own colours.
    "NP+Tier+Async+Lossy (low)":  "#d62728",
    "NP+Tier+Async+Lossy (med)":  "#e377c2",
    "NP+Tier+Async+Lossy (high)": "#17becf",
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
# Codecs that are lossless however they were invoked: panel (b) gives ndzip the
# run's error bound, but it ignores it.
LOSSLESS_BASES = {"ndzip"}

# Hues for an arm no palette above names. Deliberately distinct from both.
FALLBACK_CYCLE = ["#17becf", "#e377c2", "#bcbd22", "#8c564b", "#7b4173",
                  "#843c39", "#5254a3", "#637939"]

_LADDER_RE = re.compile(r"\s*\((low|med|high)\)\s*$")
_LADDER_TINT = {"low": 0.0, "med": 0.30, "high": 0.55}


def base_name(strategy):
    """The arm's family: its ladder suffix and a trailing `+Tier` removed.

    `NP+Tier+Async+Lossy (med)` -> `NP+Tier+Async+Lossy`; `cuSZ+Tier` -> `cuSZ`.
    Consulted only for an arm the palettes do not name, so that it lands near
    its family's hue instead of an arbitrary one.

    @param strategy Arm label exactly as the CSV spells it.
    @return The family name, which may equal `strategy`.
    """
    s = _LADDER_RE.sub("", strategy)
    return s[:-len("+Tier")] if s.endswith("+Tier") else s


def build_palette(order, known):
    """Map every arm of one panel to a colour, published hues first.

    An unlisted arm is derived rather than dropped: a `+Tier` row takes its
    codec's hue blended toward white, a `(med)`/`(high)` ladder step a deeper
    blend of its family's, and a name with no family at all takes the next
    fallback hue. Deterministic -- the same CSV always plots the same colours.

    @param order Arm labels for this panel, in plotting order.
    @param known Published {label: colour} for this panel.
    @return {label: colour} covering every entry of `order`.
    """
    out, spare = {}, 0
    for s in order:
        if s in known:
            out[s] = known[s]
            continue
        base = base_name(s)
        if base in known:
            m = _LADDER_RE.search(s)
            tint = _LADDER_TINT.get(m.group(1), 0.0) if m else 0.0
            if s.endswith("+Tier"):
                tint = max(tint, 0.45)
            out[s] = (mpl.colors.to_hex(blend_to_white(known[base], tint))
                      if tint > 0 else known[base])
        else:
            out[s] = FALLBACK_CYCLE[spare % len(FALLBACK_CYCLE)]
            spare += 1
    return out


def panel_order(rows, panel, known):
    """Arms of one panel: the published order first, then any new name.

    The order comes from the DATA. That is what lets a campaign with a
    different arm set -- an extra error-bound step, a tiered variant -- plot
    completely instead of losing the bars this file does not name.

    @param rows Raw CSV rows.
    @param panel "a" or "b".
    @param known Published palette for that panel; its key order is canonical.
    @return Arm labels in plotting order.
    """
    seen = []
    for r in rows:
        if (r.get("panel") or "").strip() != panel:
            continue
        s = (r.get("strategy") or "").strip()
        if s and s not in seen:
            seen.append(s)
    head = [s for s in known if s in seen]
    return head + [s for s in seen if s not in head]


def workload_order(rows):
    """Workloads as columns: the canonical five first, then any other.

    @param rows Raw CSV rows.
    @return Workload names in plotting order.
    """
    seen = []
    for r in rows:
        w = (r.get("workload") or "").strip()
        if w and w not in seen:
            seen.append(w)
    head = [w for w in WORKLOAD_ORDER if w in seen]
    return head + [w for w in seen if w not in head]

# The single figure carries both panels, so every bar needs its own colour: the
# panel (b) palette above deliberately reuses (a)'s hues. Panel (b)'s NeuroPress
# is no longer a duplicate: panel (b) now runs each codec both ways.
# External bars are outlined.
COLORS_B_SINGLE = {"Best fixed nvCOMP": "#08519c", "Best fixed nvCOMP+Tier": "#6baed6",
                   "ndzip": "#b15928", "ndzip+Tier": "#dbdb8d",
                   "cuSZp3": "#006d2c", "cuSZp3+Tier": "#a1d99b",
                   "cuSZ": "#252525", "cuSZ+Tier": "#969696"}
# The error bound of every arm comes from the CSV's `eb` column. It used to
# have a hardcoded fallback table here, which is what the legend actually read:
# the lookup meant to consult the data unpacked the index key in the wrong
# order and never matched, so a re-plot at a different bound still printed
# 1e-3. The column is written by figure_9.sh for every row.

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
def sanity(D, workloads, order_a):
    """Report contradictions in the data; print, never raise.

    @param D Indexed rows from index().
    @param workloads Workload names to check, in plotting order.
    @param order_a Panel (a) arm labels, as the CSV spells them.
    """
    print("== sanity checks ==")
    n = 0
    for (p, s, w), d in sorted(D.items()):
        if d["total"] is None:
            continue
        if d["compute"] is not None and d["io"] is not None:
            if abs(d["compute"] + d["io"] - d["total"]) > 0.05:
                print(f"  [split]  {p}/{w}/{s}: compute+io={d['compute']+d['io']:.3f} "
                      f"!= total={d['total']:.3f}"); n += 1

    # Each added layer should not be slower than the one before it. The last
    # step is built from the data: a campaign may run one lossy arm or a ladder
    # of several, and each is checked against the lossless arm it extends.
    chains = [("nvCOMP", "nvCOMP+Tier"),
              ("NP only", "NP+Tier"), ("NP+Tier", "NP+Tier+Async")]
    chains += [("NP+Tier+Async", s) for s in order_a
               if base_name(s) == "NP+Tier+Async+Lossy"]
    for w in workloads:
        for lo, hi in chains:
            a = D.get(("a", lo, w), {}).get("total")
            b = D.get(("a", hi, w), {}).get("total")
            if a is not None and b is not None and not (a >= b):
                print(f"  [order]  a/{w}: {lo} ({a:.2f}) < {hi} ({b:.2f})"); n += 1
    if n == 0:
        print("  all checks passed")
    print()


def reductions(D, workloads, order_a, order_b):
    """The percentages quoted in the caption and body text.

    @param D Indexed rows from index().
    @param workloads Workload names, in plotting order.
    @param order_a,order_b Arm labels per panel, as the CSV spells them.
    """
    def pct(new, old):
        if new is None or old in (None, 0):
            return None
        return (old - new) / old * 100.0

    print("== percentage reductions (positive = faster) ==")
    # One column per lossy arm the campaign actually ran, not one fixed name:
    # a ladder over three bounds used to report a single empty column, because
    # its arms are spelled `... Lossy (low|med|high)`.
    pairs = [("NP+Tier+Async", "Baseline"), ("NP+Tier+Async", "nvCOMP+Tier"),
             ("NP only", "nvCOMP"), ("NP+Tier", "nvCOMP+Tier")]
    pairs += [(s, "NP+Tier+Async") for s in order_a
              if base_name(s) == "NP+Tier+Async+Lossy"]
    print(f"{'workload':<9}" + "".join(f"{(a.replace('NP+Tier+Async+', '')[:15]):>17}" for a, _ in pairs))
    print(f"{'':<9}" + "".join(f"{('vs ' + b)[:15]:>17}" for _, b in pairs))
    print(f"{'':<9}" + "".join(f"{'-' * 15:>17}" for _ in pairs))
    for w in workloads:
        cells = []
        for a, b in pairs:
            r = pct(D.get(("a", a, w), {}).get("total"), D.get(("a", b, w), {}).get("total"))
            cells.append(f"{'--':>17}" if r is None else f"{r:>16.1f}%")
        print(f"{w:<9}" + "".join(cells))

    print("\n== panel (b): NeuroPress vs each external baseline ==")
    for w in workloads:
        np_t = D.get(("b", "NeuroPress", w), {}).get("total")
        if np_t is None:
            print(f"  {w:<8} --")
            continue
        parts = []
        for s in order_b:
            if s == "NeuroPress":
                continue
            r = pct(np_t, D.get(("b", s, w), {}).get("total"))
            parts.append(f"{s} {'--' if r is None else f'{r:.1f}%'}")
        print(f"  {w:<8} " + ",  ".join(parts))
    print()


# ----------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------
def draw_panel(ax, D, panel, order, colors, warn_total_only, ylim, dec,
               workloads, show_ratio=False):
    """Draw one group of bars per workload, one bar per arm.

    @param workloads Workload names, left to right; one x-tick each.
    @return True if anything was drawn; False for an empty arm list, which a
      `--panel a` run produces for panel (b) -- the CSV then has no rows of
      that panel at all.
    """
    nb = len(order)
    if nb == 0 or not workloads:
        return False
    # Vertical labels are wider than the bar pitch once the ratio is appended.
    # Spread them over ROWS interleaved baselines so no two neighbours share one:
    # 14 arms per group need 3 rows; 2 still collided.
    # A rotated 7 pt label is ~28 px wide against a ~21 px bar pitch, so
    # neighbours collide however short the text is. Interleave their baselines:
    # 3 rows when the ratio doubles the length, 2 for a time-only label.
    # A rotated label is wider than the bar pitch, so neighbours are given
    # interleaved baselines. The combined axis carries both panels -- 17 arms
    # with the external +Tier pairs -- and two rows is not enough there.
    # >13, not >12: the published combined axis is exactly 13 arms (9 ablation
    # + 4 externals) and staggers over 2 rows. Adding the external +Tier pairs
    # takes it to 17, which needs a third row.
    rows = (3 if nb > 10 else 2) if show_ratio else (3 if nb > 13 else
                                                    (2 if nb > 6 else 1))
    width = (1.0 - BAR_PAD) / nb
    x0 = np.arange(len(workloads))

    # A workload with NO measured arm gets ONE centred marker rather than a
    # "TBD" per bar: at 17 arms those overlap into an illegible band, and the
    # message is about the workload, not about each arm of it.
    def _measured(w):
        for it in order:
            pp, ss = it if isinstance(it, tuple) else (panel, it)
            if D.get((pp, ss, w), {}).get("total") is not None:
                return True
        return False
    blank = {w for w in workloads if not _measured(w)}
    for j, w in enumerate(workloads):
        if w in blank:
            ax.text(x0[j], ylim * 0.02, "not measured", ha="center",
                    va="bottom", fontsize=FS_VAL, color="0.55")

    for i, item in enumerate(order):
        # An entry is a strategy, or (panel, strategy) when one axis carries both.
        p, s = item if isinstance(item, tuple) else (panel, item)
        col = colors[s]
        pale = blend_to_white(col, IO_BLEND)
        # Mark the external codecs when they share an axis with the ablation.
        edge = "black" if isinstance(item, tuple) and p == "b" else "white"
        for j, w in enumerate(workloads):
            x = x0[j] - (1.0 - BAR_PAD) / 2 + width * (i + 0.5)
            d = D.get((p, s, w), {})
            total = d.get("total")

            # Not measured yet: leave the slot empty but keep its width, so the
            # layout is identical once the number arrives. A per-bar marker is
            # only drawn when SOME arm of this workload was measured -- a wholly
            # unmeasured workload already carries one centred label above.
            if total is None:
                if w not in blank:
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
    ax.set_xlim(-0.5, len(workloads) - 0.5)
    ax.set_ylim(0, ylim)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.75", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append",
                    help="long-format CSV to plot; repeat to merge per-workload "
                         "runs into one figure. REQUIRED -- this script has no "
                         "data of its own")
    ap.add_argument("--write-template", metavar="PATH",
                    help="write an empty CSV with just the header and exit")
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
        print(f"wrote template: {args.write_template}")
        return 0

    # No measured CSV, no figure. The previous version fell back to a table of
    # numbers read off the published PNG, so a bare invocation produced a
    # complete, plausible Figure 9 out of nothing.
    if not args.csv:
        print("error: --csv is required (pass figure_9.sh's fig9.csv; repeat "
              "the flag to merge one CSV per workload)", file=sys.stderr)
        return 2
    rows = [r for c in args.csv for r in load(c)]
    D, _ = index(rows)
    # The CSVs are already in minutes, and so is the figure -- no conversion.

    # Arms, workloads and colours all come from the rows just read.
    workloads = workload_order(rows)
    order_a = panel_order(rows, "a", COLORS_A)
    order_b = panel_order(rows, "b", COLORS_B)
    colors_a = build_palette(order_a, COLORS_A)
    colors_b = build_palette(order_b, COLORS_B)
    # The combined figure needs a distinct hue per bar, so panel (b) is redrawn
    # from its own palette there; an arm with no entry keeps its panel colour.
    order_single = ([("a", s) for s in order_a] +
                    [("b", s) for s in order_b
                     if s in COLORS_B_SINGLE or base_name(s) in COLORS_B_SINGLE])
    colors_single = {**colors_a,
                     **build_palette([s for pan, s in order_single if pan == "b"],
                                     COLORS_B_SINGLE)}

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
        """The arm's legend text, with the bound it actually ran at.

        The key order is (panel, strategy, workload) -- the previous version
        unpacked it as (panel, workload, strategy), so the match never fired
        and every bound came from a hardcoded table instead of the run.
        """
        if base_name(s) in LOSSLESS_BASES:
            return f"{s} (lossless)"
        seen = [d["eb"] for (p_, s_, w_), d in D.items()
                if s_ == s and d.get("eb")]
        # With the bound appended the (low)/(med)/(high) suffix is redundant --
        # epsilon is what tells the ladder steps apart -- so it is dropped, as
        # the published legend does. Without a bound it is all there is, so it
        # stays.
        return f"{_LADDER_RE.sub('', s)}, {eps(seen[0])}" if seen else s

    # One figure per panel, on the same y-axis so the two stay comparable.
    os.makedirs(args.out, exist_ok=True)
    warn_total_only, pngs = [], []
    for panel, order, colors, height, title, name in (
            ("a", order_a, colors_a, 2.3,
             "(a) Ablation  --  each bar is a separate run; lossless unless $\\varepsilon$ is shown",
             "fig9a_ablation.png"),
            ("b", order_b, colors_b, 2.0, "(b) External baselines",
             "fig9b_baselines.png")):
        if not order:
            # `--panel a` / `--panel b` writes only its own rows, so the other
            # panel has no arms. Skip it instead of drawing an empty axis.
            print(f"note: no panel ({panel}) rows in the CSV; {name} not written")
            continue
        fig, ax = plt.subplots(figsize=(FIG_W, height))
        draw_panel(ax, D, panel, order, colors, warn_total_only, ylim, dec,
                   workloads)
        ax.set_ylabel("Total wall-clock time (min)", fontsize=FS_AXIS)
        ax.set_xticks(np.arange(len(workloads)))
        ax.set_xticklabels(workloads, fontsize=FS_TICK)
        fig.subplots_adjust(top=0.99)
        # At most 4 columns: the arm legend and the segment key below share one
        # line, and a 5-column row of bound-annotated labels ran under the key.
        fig.legend(handles=[Patch(facecolor=colors[s_], label=legend_label(s_)) for s_ in order],
                   loc="lower left", bbox_to_anchor=(0.005, 0.995),
                   ncol=3 if panel == "a" else min(4, len(order)),
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
    if not order_single:
        print("note: nothing to plot on the combined axis")
        sanity(D, workloads, order_a)
        reductions(D, workloads, order_a, order_b)
        print("wrote " + (", ".join(pngs) if pngs else "nothing"))
        return 0
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    draw_panel(ax, D, None, order_single, colors_single, warn_total_only, ylim,
               dec, workloads)
    ax.set_ylabel("Total wall-clock time (min)", fontsize=FS_AXIS)
    ax.set_xticks(np.arange(len(workloads)))
    ax.set_xticklabels(workloads, fontsize=FS_TICK)
    fig.subplots_adjust(top=0.99)
    fig.legend(handles=[Patch(facecolor=colors_single[s_], label=legend_label(s_),
                              edgecolor="black" if p_ == "b" else "white", linewidth=0.5)
                        for p_, s_ in order_single],
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
    sanity(D, workloads, order_a)
    reductions(D, workloads, order_a, order_b)
    print("wrote " + ", ".join(pngs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Figure 9: end-to-end wall-clock time per workload, lower is better.

(a) the ablation ladder and (b) NeuroPress against fixed codecs. Every bar is
one run: the solid segment is compute, the pale one the MEASURED elapsed
device I/O (the bdev's own write timers, union of the per-write intervals)
(see fig9.md, beside this file, for what that does and does not mean; CSVs
from before 2026-09-22 still carry the input read in the pale segment).

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
# STYLE -- everything tweakable lives here
# ----------------------------------------------------------------------------
FIG_W = 7.16                       # IEEE two-column figure* width, inches
PLOT_H = 2.35                      # inches of plot area, axis to axis
FS_TITLE, FS_TICK, FS_LEG, FS_VAL = 8.5, 7, 7, 7   # nothing below 7 pt
# THE CSV IS IN MINUTES, THE CHART IS IN SECONDS. figure_9.sh writes *_min
# columns; index() multiplies every one of them by this on the way in, so the
# axis, the value labels, --ylim, the sanity checks and --deduct-setup all
# speak the same unit and nothing downstream has to convert again.
SEC_PER_MIN = 60.0
Y_CLIP = 6.5 * SEC_PER_MIN         # SECONDS; cuSZ/VPIC at 14.2 min must not set this
Y_AUTO_BELOW = 0.5                 # if max(total) < Y_CLIP*this, rescale to the
                                   # data -- a smoke run is ~4 s and would
                                   # otherwise be invisible against a 390 s axis
HEADROOM = 1.30                    # y limit over the tallest bar: room for its
                                   # upright value label
IO_BLEND = 0.55                    # pale segment: this far from the fill to white
BAR_W = 0.80                       # bar width as a fraction of its slot; the rest
                                   # is the surface gap between neighbours
GROUP_W = 0.92                     # share of each workload's width given to bars;
                                   # the rest is the gutter between groups
AX_LEFT, AX_RIGHT = 0.07, 0.995    # plot area, figure fractions
LEGEND_COLS = 4                    # arm-legend columns above the plot
KEY_H = 0.46                       # inches under the axis: x labels + key line

# Preferred panel order. A workload present in the CSV but not named here is
# appended in the order it first appears, so a new one plots without editing
# this file.
WORKLOAD_ORDER = ["VPIC", "Nyx", "LAMMPS", "WarpX", "AI"]

# Ink and chrome. Text never takes a series colour.
INK, INK_MUTED, GRID = "#2b2b2b", "#6b6a66", "#e7e6e2"

# ARM COLOURS: hue = library, hatching = the +Tier variant, and along the
# NeuroPress ladder a darker step = one more feature. Seven colour classes,
# not fifteen: the previous one-hue-per-arm palette ran past what a reader can
# hold, gave cuSZ+Tier the same grey as Baseline, and marked the external
# codecs with black outlines. NeuroPress is green and ndzip red, as requested;
# cuSZ moved from orange to amber because every red beside an orange failed
# the normal-vision floor. Validated with the dataviz skill's
# validate_palette.js on a light surface:
#   libraries  #d9403f #c98500 #872985 #5089cc #008856 -- all checks pass
#   ordinal    #5089cc -> #1c5cab  and  #31aa76 -> #008856 -> #006435 -- pass
#   neighbours Baseline|ndzip CVD 10.3 / normal 26.5; ndzip|cuSZ 8.3 / 16.0;
#              nvCOMP+Tier|NP only 16.3 / 18.6; Best fixed|NeuroPress 19.0 / 20.1
#   cuSZp3 vs Best fixed nvCOMP (neighbours in panel b): CVD 7.4, normal 18.5;
#   ndzip red vs NeuroPress green (never neighbours): CVD 7.7, normal 30.5 --
#   both inside the 6-8 CVD floor band, legal with secondary encoding: every
#   bar is labelled and the legend follows bar order.
ARM_COLORS = {
    "Baseline":            "#4b4a47",
    "ndzip":               "#d9403f",
    "cuSZ":                "#c98500",
    "cuSZp3":              "#872985",
    "nvCOMP":              "#5089cc",   # lossless: the ablation's fixed codec
    "Best fixed nvCOMP":   "#1c5cab",   # the same library at the run's bound
    # Slowest fixed action in the sweep. Same library, so the same hue family:
    # the lighter step of the validated #5089cc -> #1c5cab ordinal pair, and
    # always drawn beside Best so that pair is the one a reader compares.
    "Worst fixed nvCOMP":  "#5089cc",
    "NP only":             "#31aa76",
    "NP":                  "#31aa76",   # base_name("NP+Tier")
    "NP+Tier+Async":       "#008856",
    "NP+Tier+Async+Lossy": "#006435",
    "NeuroPress":          "#006435",   # panel (b): NeuroPress at the bound
}
TIER_HATCH = "////"
KEY_GREY = "#8c8b87"               # neutral swatch for the segment/hatch key
# Hues for an arm no entry above names, taken in fixed order.
FALLBACK_COLORS = ["#008300", "#e87ba4", "#eda100"]

# Codecs that are lossless however they were invoked: panel (b) gives ndzip the
# run's error bound, but it ignores it.
LOSSLESS_BASES = {"ndzip"}

# Canonical left-to-right arm order within each panel; an arm the CSV has and
# these lists lack is appended in the order it first appears.
ORDER_A = ["Baseline", "nvCOMP", "nvCOMP+Tier", "NP only", "NP+Tier",
           "NP+Tier+Async", "NP+Tier+Async+Lossy"]
ORDER_B = ["ndzip", "ndzip+Tier", "cuSZ", "cuSZ+Tier", "cuSZp3", "cuSZp3+Tier",
           "Best fixed nvCOMP", "Best fixed nvCOMP+Tier",
           "Worst fixed nvCOMP", "Worst fixed nvCOMP+Tier",
           "NeuroPress", "NeuroPress+Tier"]

_LADDER_RE = re.compile(r"\s*\((low|med|high)\)\s*$")

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


def panel_order(rows, panel, known):
    """Arms of one panel: the published order first, then any new name.

    The order comes from the DATA. That is what lets a campaign with a
    different arm set -- an extra error-bound step, a tiered variant -- plot
    completely instead of losing the bars this file does not name.

    @param rows Raw CSV rows.
    @param panel "a" or "b".
    @param known Canonical arm order for that panel (ORDER_A / ORDER_B).
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

# Left-to-right order on the combined figure: Baseline, the external codecs as
# plain/+Tier pairs, then nvCOMP -- the fixed-codec reference NeuroPress is
# measured against -- directly before the NeuroPress arms, which follow in the
# ablation's own order. Each entry is (panel, arm label). Panel (b)'s own
# NeuroPress arms stay off it: panel (a)'s ladder already carries NeuroPress.
SINGLE_ORDER = [("a", "Baseline"),
                ("b", "ndzip"), ("b", "ndzip+Tier"),
                ("b", "cuSZ"), ("b", "cuSZ+Tier"),
                ("b", "cuSZp3"), ("b", "cuSZp3+Tier"),
                ("a", "nvCOMP"), ("a", "nvCOMP+Tier"),
                ("b", "Best fixed nvCOMP"), ("b", "Best fixed nvCOMP+Tier"),
                ("b", "Worst fixed nvCOMP"), ("b", "Worst fixed nvCOMP+Tier")]
# The error bound of every arm comes from the CSV's `eb` column. It used to
# have a hardcoded fallback table here, which is what the legend actually read:
# the lookup meant to consult the data unpacked the index key in the wrong
# order and never matched, so a re-plot at a different bound still printed
# 1e-3. The column is written by figure_9.sh for every row.

FIELDS = ["panel", "workload", "strategy", "compute_min", "io_min", "total_min",
          "std_min", "ratio", "eb", "setup_min"]

# The y-axis caption. A list so --deduct-setup can rewrite it for every axes
# without threading a flag through draw_bars.
YLABEL = ["Total wall-clock time (s)"]


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
        def _sec(name):
            v = _f(r.get(name))
            return None if v is None else v * SEC_PER_MIN
        total = _sec("total_min")
        comp, io = _sec("compute_min"), _sec("io_min")
        if total is None and comp is not None and io is not None:
            total = comp + io          # total is optional when both parts given
        key = (r["panel"].strip(), r["strategy"].strip(), r["workload"].strip())
        # When merging CSVs, a TBD row never replaces a measured one.
        if total is None and out.get(key, {}).get("total") is not None:
            continue
        out[key] = dict(compute=comp, io=io, total=total, std=_sec("std_min"),
                        ratio=_f(r.get("ratio")), eb=_f(r.get("eb")),
                        bound=(r.get("bound") or "").strip(),
                        payload=_f(r.get("payload_mib")),
                        setup=_sec("setup_min"))
    return out, warn


def deduct_setup(D):
    """Take cuSZ's per-chunk resource-manager build out of every bar.

    cuSZ's rev1 C API has no reset and no reuse contract, so the wrapper
    constructs a CUDA stream and a psz_resource -- histogram, Huffman book,
    quantisation buffers -- and destroys both for EVERY chunk. figure_9.sh
    measures exactly those three calls per chunk (cusz.h's SetupLog) and sums
    them into `setup_min`; this removes that sum from the compute segment, so a
    bar shows what the codec would cost if the library let its state be reused.

    NOT the default. The measured figure is the honest one -- this is the
    library a user would actually link -- and only cuSZ has a column to remove,
    so a deducted chart must say so in its axis label. Applied to EVERY arm,
    not just the cuSZ ones: a NeuroPress arm that routes some chunks to cuSZ
    pays the same construction and gets the same credit.

    @param D index() output; modified in place.
    @return seconds removed, summed over every bar that had any.
    """
    removed = 0.0
    for v in D.values():
        u = v.get("setup")
        if not u or v.get("total") is None:
            continue
        u = min(u, v["total"])
        v["total"] -= u
        if v.get("compute") is not None:
            # The manager is host-side construction, so it sits in the solid
            # compute segment; the pale I/O segment is untouched.
            v["compute"] = max(0.0, v["compute"] - u)
        removed += u
    return removed


def payload_text(mib):
    """How much data one bar moved, for the axis label.

    @param mib Payload replayed by that workload's arms, in MiB.
    @return "4 GiB", "512 MiB", or "" when the CSV did not record it.
    """
    if not mib:
        return ""
    return f"{mib / 1024:.3g} GiB" if mib >= 1024 else f"{mib:.3g} MiB"


def workload_labels(D, workloads):
    """Axis labels: the workload, and under it the payload its arms replayed.

    A merged CSV can hold workloads measured at different sizes, and a bar
    means nothing without the bytes behind it.

    @param D Indexed rows from index().
    @param workloads Workload names, left to right.
    @return One label per workload.
    """
    out = []
    for w in workloads:
        sizes = {d.get("payload") for k, d in D.items() if k[2] == w and d.get("payload")}
        size = payload_text(max(sizes)) if sizes else ""
        out.append(f"{w}\n{size}" if size else w)
    return out


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
# PLOTTING -- one chart, the workloads' groups lined up left to right
# ----------------------------------------------------------------------------
def arm_style(s, spare):
    """Fill colour and tier flag of one arm.

    @param s Arm label exactly as the CSV spells it.
    @param spare Fallback assignments shared across calls, so an arm no table
      entry names keeps one colour in every figure of the run.
    @return (fill hex, tiered) -- a tiered arm is drawn hatched.
    """
    tiered = "+Tier" in s
    for key in (s, _LADDER_RE.sub("", s), base_name(s)):
        if key in ARM_COLORS:
            return ARM_COLORS[key], tiered
    if s not in spare:
        spare[s] = FALLBACK_COLORS[len(spare) % len(FALLBACK_COLORS)]
    return spare[s], tiered


def eps_text(e):
    """Mathtext for one error bound: 10^k when it is a power of ten.

    @param e Absolute error bound, > 0.
    @return A mathtext string such as $\\varepsilon = 10^{-3}$.
    """
    k = math.log10(e)
    return (rf"$\varepsilon = 10^{{{round(k)}}}$" if abs(k - round(k)) < 1e-9
            else rf"$\varepsilon = {e:g}$")


def lossy_bounds(D):
    """Every distinct non-zero bound a lossy arm ran at, ascending.

    @param D Indexed rows from index().
    @return Sorted list of bounds; empty when every arm was lossless.
    """
    return sorted({d["eb"] for (_, s, _), d in D.items()
                   if (d.get("eb") or 0) > 0 and base_name(s) not in LOSSLESS_BASES})


def arm_label(s, D):
    """Legend text for one arm.

    One bound across the run: a dagger marks the lossy arms and the key under
    the grid states the bound once. Several bounds (an older ladder campaign):
    each lossy arm carries its own bound instead.

    @param s Arm label as the CSV spells it.
    @param D Indexed rows from index().
    @return The label to print.
    """
    name = _LADDER_RE.sub("", s)
    if base_name(s) in LOSSLESS_BASES:
        return name
    ebs = sorted({d["eb"] for (_, s_, _), d in D.items()
                  if s_ == s and (d.get("eb") or 0) > 0})
    if not ebs:
        return name
    return f"{name} †" if len(lossy_bounds(D)) == 1 else f"{name}, {eps_text(ebs[0])}"


def label_size(nmax, n_groups):
    """Value-label font size that fits one upright label per bar.

    Every group gets 1/n of the axes width and GROUP_W of that for bars, so a
    bar's slot is known before anything is drawn. An upright label is about
    0.8 em thick; it is kept inside 90% of the slot so neighbours never touch.

    @param nmax Slots per group: the most arms any workload in the figure has.
    @param n_groups Number of workload groups on the axis.
    @return Font size in points, at most FS_VAL.
    """
    axes_pt = FIG_W * (AX_RIGHT - AX_LEFT) * 72.0
    slot_pt = axes_pt / n_groups * GROUP_W / nmax
    return min(FS_VAL, math.floor(slot_pt * 0.9 / 0.80 * 2) / 2)


def draw_bars(ax, D, items, workloads, nmax, ylim, dec, fs, spare, warn):
    """Every workload's bars on one axis, one group per workload.

    A group holds only the arms with a row for that workload, packed and
    centred, so an arm it never runs leaves no gap; an arm with a row but no
    total was meant to run and did not, and keeps its slot marked TBD. Each
    value label stands upright on its own bar's centre line, so it cannot
    reach a neighbour however tall the bars are.

    @param ax Axes to draw on.
    @param D Indexed rows from index().
    @param items (panel, arm) pairs in plotting order.
    @param workloads Workload names, left to right.
    @param nmax Slots per group.
    @param ylim Y-axis limit, seconds.
    @param dec Decimals on the value labels.
    @param fs Value-label font size, from label_size().
    @param spare Fallback-colour assignments, shared across figures.
    @param warn Collects arms that had a total but no compute/I-O split.
    """
    slot = GROUP_W / nmax
    for j, w in enumerate(workloads):
        present = [it for it in items if it + (w,) in D]
        start = j - len(present) * slot / 2.0
        for k, (p, s) in enumerate(present):
            d, x = D[(p, s, w)], start + slot * (k + 0.5)
            col, tiered = arm_style(s, spare)
            total, comp = d.get("total"), d.get("compute")
            if total is None:
                ax.text(x, ylim * 0.01, "TBD", ha="center", va="bottom",
                        rotation=90, fontsize=fs, color=INK_MUTED)
                continue
            drawn = min(total, ylim)
            if comp is None:
                warn.append(f"{p}/{w}/{s}")  # no split: one solid segment
            solid = drawn if comp is None else min(comp, drawn)
            ax.bar(x, solid, width=slot * BAR_W, color=col, edgecolor="white",
                   linewidth=0, hatch=TIER_HATCH if tiered else None, zorder=3)
            if drawn > solid:
                ax.bar(x, drawn - solid, bottom=solid, width=slot * BAR_W,
                       linewidth=0, color=blend_to_white(col, IO_BLEND), zorder=3)
            clipped = total > ylim
            top = drawn
            ax.annotate(f"{total:.{dec}f}", (x, top),
                        xytext=(0, -1.5 if clipped else 1.5),
                        textcoords="offset points", ha="center",
                        va="top" if clipped else "bottom", rotation=90,
                        fontsize=fs, color="white" if clipped else INK, zorder=6)
    ax.set_xlim(-0.5, len(workloads) - 0.5)
    ax.set_ylim(0, ylim)
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels(workload_labels(D, workloads), fontsize=FS_TITLE, color=INK)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=FS_TICK, colors=INK_MUTED, width=0.6,
                   length=2.5)
    # RELABELLED when a deduction was applied, so a chart can never be mistaken
    # for the measured one: the reader is told what was taken out of every bar.
    ax.set_ylabel(YLABEL[0], fontsize=FS_TICK, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.6)


def draw_legend(fig, ax, D, items, spare, title):
    """The arm legend, spread across the full width above the plot.

    @param fig The figure.
    @param ax The plot axes; the legend sits on its top edge.
    @param D Indexed rows from index().
    @param items (panel, arm) pairs in plotting order.
    @param spare Fallback-colour assignments, shared across figures.
    @param title Figure name printed over the entries, or None for none.
    """
    handles = []
    for _, s in items:
        col, tiered = arm_style(s, spare)
        handles.append(Patch(facecolor=col, edgecolor="white", linewidth=0,
                             hatch=TIER_HATCH if tiered else None,
                             label=arm_label(s, D)))
    leg = ax.legend(handles=handles, loc="lower left", mode="expand",
                    bbox_to_anchor=(0.0, 1.03, 1.0, 0.0), borderaxespad=0,
                    ncol=min(LEGEND_COLS, len(handles)), frameon=False,
                    fontsize=FS_LEG, handlelength=1.5, handleheight=1.1,
                    columnspacing=1.0, labelspacing=0.45, title=title,
                    title_fontsize=FS_TITLE, alignment="left")
    for t in leg.get_texts():
        t.set_color(INK)
    leg.get_title().set_color(INK)


def draw_key(fig, D):
    """One line under the axis: what a bar's parts, its hatching and the
    dagger mean.

    @param fig The figure.
    @param D Indexed rows from index(), for the error bound.
    """
    # NOT "compute". The solid segment is total minus the measured device I/O,
    # so it also holds scheduling, allocation and per-chunk library setup --
    # on the cuSZ arm 82% of the bar is the resource manager being built and
    # torn down, which a reader would never guess from the word "compute".
    key = [Patch(facecolor=KEY_GREY, linewidth=0, label="non-I/O elapsed"),
           Patch(facecolor=blend_to_white(KEY_GREY, IO_BLEND), linewidth=0,
                 label="device I/O (measured)"),
           Patch(facecolor=KEY_GREY, edgecolor="white", linewidth=0,
                 hatch=TIER_HATCH, label="+Tier: RAM tier over NVMe")]
    ebs = lossy_bounds(D)
    if len(ebs) == 1:
        # Text only: an invisible swatch keeps it spaced like its neighbours.
        key.append(Patch(facecolor="none", edgecolor="none",
                         label=f"\u2020 lossy, {eps_text(ebs[0])}"))
    leg = fig.legend(handles=key, loc="lower left", borderaxespad=0,
                     bbox_to_anchor=(AX_LEFT - 0.01, 0.01), ncol=len(key),
                     frameon=False, fontsize=FS_LEG, handlelength=1.5,
                     handleheight=1.1, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK)


def render_single(path, D, items, workloads, ylim, dec, spare, warn, title):
    """One chart: every workload's group of bars lined up left to right.

    @param path Output PNG path.
    @param D Indexed rows from index().
    @param items (panel, arm) pairs in plotting order.
    @param workloads Workload names, left to right.
    @param ylim Y-axis limit, seconds.
    @param dec Decimals on the value labels.
    @param spare Fallback-colour assignments, shared across figures.
    @param warn Collects arms that had a total but no compute/I-O split.
    @param title Figure name printed over the legend, or None for none.
    @return The value-label font size used.
    """
    nmax = max(sum(1 for it in items if it + (w,) in D) for w in workloads)
    fs = label_size(nmax, len(workloads))
    legend_rows = math.ceil(len(items) / LEGEND_COLS)
    top_h = (0.30 if title else 0.12) + 0.19 * legend_rows
    height = PLOT_H + top_h + KEY_H
    fig, ax = plt.subplots(figsize=(FIG_W, height))
    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, bottom=KEY_H / height,
                        top=1.0 - top_h / height)
    draw_bars(ax, D, items, workloads, nmax, ylim, dec, fs, spare, warn)
    draw_legend(fig, ax, D, items, spare, title)
    draw_key(fig, D)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    return fs


def pick_ylim(D, forced):
    """The shared y limit and the value-label precision.

    @param D Indexed rows from index().
    @param forced --ylim from the command line, or None.
    @return (ylim in seconds, decimals for the value labels).
    """
    totals = [d["total"] for d in D.values() if d["total"] is not None]
    dmax = max(totals) if totals else Y_CLIP
    if forced is not None:
        ylim = forced
    else:
        ylim = dmax * HEADROOM
        if dmax < Y_CLIP * Y_AUTO_BELOW:
            print(f"note: max total {dmax:.4g} s is far below the {Y_CLIP:g} "
                  f"s paper limit; y-axis fitted to {ylim:.4g}. Pass --ylim "
                  "to override.\n")
    # Decimals from the SMALLEST bar: an axis stretched by one slow arm used
    # to print every fast arm as "0.2", hiding the differences between them.
    lo = min(totals) if totals else ylim
    return ylim, (1 if lo >= 2 else (2 if lo >= 0.2 else 3))


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
                    help="y-axis limit in SECONDS (default: fitted to the data)")
    ap.add_argument("--deduct-setup", action="store_true",
                    help="subtract each arm's measured cuSZ resource-manager "
                         "construction (setup_min) from its bar; the chart is "
                         "relabelled to say so (default: plot what was measured)")
    ap.add_argument("--panels", action="store_true",
                    help="also write the per-panel figures fig9a_ablation.png "
                         "and fig9b_baselines.png (default: the single chart only)")
    args = ap.parse_args()

    if args.write_template:
        os.makedirs(os.path.dirname(os.path.abspath(args.write_template)) or ".", exist_ok=True)
        with open(args.write_template, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
        print(f"wrote template: {args.write_template}")
        return 0
    # No measured CSV, no figure. An older version fell back to numbers read
    # off the published PNG, so a bare invocation drew a plausible Figure 9
    # out of nothing.
    if not args.csv:
        print("error: --csv is required (pass figure_9.sh's fig9.csv; repeat "
              "the flag to merge one CSV per workload)", file=sys.stderr)
        return 2
    rows = [r for c in args.csv for r in load(c)]
    D, _ = index(rows)
    if args.deduct_setup:
        cut = deduct_setup(D)
        if cut == 0.0:
            print("note: --deduct-setup had nothing to remove; no row carries a "
                  "setup_min (only runs after 2026-09-22 measure it)",
                  file=sys.stderr)
        else:
            print(f"--deduct-setup: removed {cut:.2f} s of cuSZ "
                  f"resource-manager construction across the chart")
            YLABEL[0] = ("Wall-clock time (s)\n"
                         "excl. cuSZ per-chunk resource-manager build")
    workloads = workload_order(rows)
    order_a = panel_order(rows, "a", ORDER_A)
    order_b = panel_order(rows, "b", ORDER_B)
    # Panel (b)'s NeuroPress arms are normally left off the single chart --
    # panel (a)'s ladder already carries NeuroPress. When a run measured no
    # panel (a) NP arm (an --only campaign, say), dropping them would leave the
    # chart with no NeuroPress bar at all, so they come back in.
    # The duplicate is panel (a)'s LOSSY NP rung, not any NP arm: panel (b)'s
    # NeuroPress runs at the same bound, so the two would draw one measurement
    # twice. A lossless-only campaign has no such rung and keeps them.
    has_np_lossy_a = any((D.get(("a", s, w)) or {}).get("eb")
                         for s in order_a if s.startswith("NP") for w in workloads)
    single = ([("a", s) for s in order_a] +
              [("b", s) for s in order_b
               if not has_np_lossy_a or base_name(s) != "NeuroPress"])
    single = ([x for x in SINGLE_ORDER if x in single] +
              [x for x in single if x not in SINGLE_ORDER])

    mpl.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "hatch.linewidth": 0.7,
                         "hatch.color": "white", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    ylim, dec = pick_ylim(D, args.ylim)
    os.makedirs(args.out, exist_ok=True)
    spare, warn, pngs = {}, [], []
    figures = [(single, None, "fig9.png")]
    if args.panels:
        figures = [([("a", s) for s in order_a], "(a) Ablation", "fig9a_ablation.png"),
                   ([("b", s) for s in order_b], "(b) External baselines",
                    "fig9b_baselines.png")] + figures
    for items, title, name in figures:
        if not items:
            print(f"note: no rows for {name}; not written")
            continue
        pngs.append(os.path.join(args.out, name))
        fs = render_single(pngs[-1], D, items, workloads, ylim, dec, spare, warn, title)
        if fs < FS_VAL:
            print(f"note: {name}: value labels at {fs:g} pt so one fits on every bar")
    if warn:
        print(f"WARNING: {len(warn)} bar(s) had a total but no compute/I-O "
              f"split; drawn as one segment: {', '.join(sorted(set(warn)))}\n")
    sanity(D, workloads, order_a)
    reductions(D, workloads, order_a, order_b)
    print("wrote " + ", ".join(pngs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

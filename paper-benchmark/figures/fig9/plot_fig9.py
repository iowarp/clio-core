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
import argparse, csv, math, os, re, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# STYLE -- everything tweakable lives here
# ----------------------------------------------------------------------------
# EuroSys 2027: the text block is 178 x 229 mm (7 x 9 in), and ">=10-point
# font ... applies to all text, including figures and captions". A point size
# only MEANS 10 pt if the figure is placed at the width it was drawn at, so
# this is the text block exactly and the figure must be included with
# width=\textwidth and NOT scaled. 7.16 was the IEEE figure* width and
# overflowed EuroSys by 0.16 in, which made LaTeX shrink the figure and drag
# every label below its nominal size.
FIG_W = 7.00                       # inches = 178 mm EuroSys text block
# One EuroSys column: two columns in the 7 in block, >= 0.33 in apart. The
# per-workload figures are drawn at this width and placed unscaled.
COL_W = (7.00 - 0.33) / 2          # 3.335 in
COL_PLOT_H = 1.90                  # inches of plot area in a column figure
FS_MIN = 10                        # EuroSys: every text >= 10 pt, figures included
PLOT_H = 2.35                      # inches of plot area, axis to axis
FS_TITLE, FS_TICK, FS_LEG, FS_VAL = 11, 10, 10, 10  # EuroSys floor: 10 pt
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
HEADROOM_FLAT = 1.12               # the same with horizontal labels: one text line tall
IO_BLEND = 0.55                    # pale segment: this far from the fill to white
BAR_W = 0.80                       # bar width as a fraction of its slot; the rest
                                   # is the surface gap between neighbours
GROUP_W = 0.92                     # share of each workload's width given to bars;
                                   # the rest is the gutter between groups
AX_LEFT, AX_RIGHT = 0.07, 0.995    # plot area, figure fractions
# Most arm-legend columns tried above the plot. The columns actually used are
# measured (fit_legend_cols): the fewest rows whose drawn width fits.
MAX_LEGEND_COLS = 6
# Inches under the axis, holding TWO lines of tick label (workload, then
# payload) plus the key row. Every number here was derived at 7 pt type; at the
# EuroSys 10 pt floor a text line is ~14 pt = 0.19 in, so two label lines plus
# a two-row key plus gaps needs ~0.95 in. At the old 0.46 the key row was drawn
# straight through the payload labels.
KEY_H = 0.95
# The payload under each workload name ("40 GiB"): off. Without it the tick
# labels are one line, so the band under the axis loses one 10 pt line.
SHOW_PAYLOAD = False
if not SHOW_PAYLOAD:
    KEY_H -= 0.19

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
    # GPULZ (lossless LZSS) takes cuSZp3's purple: cuSZp3 is out of every
    # figure (DROPPED_CODECS), so the two never share a chart.
    "GPULZ":               "#872985",
    "nvCOMP":              "#5089cc",   # lossless: the ablation's fixed codec
    "Best fixed nvCOMP":   "#1c5cab",   # the same library at the run's bound
    "Best fixed nvCOMP (lossless)":      "#1c5cab",  # the same library, no -q
    "Best fixed nvCOMP+Tier (lossless)": "#1c5cab",
    # Slowest fixed action in the sweep. Same library, so the same hue family:
    # the lighter step of the validated #5089cc -> #1c5cab ordinal pair, and
    # always drawn beside Best so that pair is the one a reader compares.
    "Worst fixed nvCOMP":  "#5089cc",
    "NP only":             "#31aa76",
    "NP":                  "#31aa76",   # base_name("NP+Tier")
    "NP+Tier+Async":       "#008856",
    "NP+Tier+Async+Lossy": "#006435",
    "NeuroPress":          "#006435",   # panel (b): NeuroPress at the bound
    # HCompress (IPDPS 2020) as a selector. Validated against EVERY arm, not
    # just its bar neighbours -- a legend swatch is matched against all bars:
    # the only candidate of five clearing the normal-vision floor everywhere
    # (worst 15.3 vs ndzip; brown failed at 7.6 vs cuSZ, magenta 13.0, cyan
    # 12.5, olive 14.2). Worst CVD is 7.0 vs NP only, in the 6-8 band that is
    # legal only with secondary encoding -- which every bar has: a value
    # label, the +Tier hatch, and a legend that follows bar order.
    "HCompress":           "#e87ba4",
    # Selector baselines. HCompress keeps its pink above: amber is cuSZ's, and
    # the combined chart draws both. The nvCOMP blue never appears in figure
    # (b), so XGB may reuse it.
    "XGB":                 "#5089cc",
    # Figure 9 (b), lossless only: Tableau 10's blue (XGBoost) and orange
    # (HCompress), the user's choice, beside NP only's green.
    "XGB (lossless)":             "#4e79a7",
    "HCompress (lossless)":       "#f28e2b",
    "NeuroPress (lossless)":      "#31aa76",
    "NeuroPress+Tier (lossless)": "#31aa76",
}
TIER_HATCH = "////"
KEY_GREY = "#8c8b87"               # neutral swatch for the segment/hatch key
# Legend names. The legend names each COLOUR once: a tiered arm shares its
# untiered twin's hue, and the +Tier hatch gets one entry of its own, so
# "+Tier" never appears in a name. The dagger already says lossy, so "+Lossy"
# goes too, and "(lossless)" goes because an undaggered name is lossless.
SHORT_NAMES = {
    "Worst fixed nvCOMP":    "Worst nvCOMP",
    "Best fixed nvCOMP":     "Best nvCOMP",
    "Best fixed nvCOMP (lossless)": "Best nvCOMP",
    "NP only":               "NP",
    "NP+Async+Lossy":        "NP+Async",
    "XGB (lossless)":        "XGBoost",
    "HCompress (lossless)":  "HCompress",
    "NeuroPress (lossless)": "NeuroPress",
}
TIER_LABEL = "+Tier"
# NeuroPress's tiering steps keep a legend entry each (see legend_handles).
NP_LADDER = {"NP only": "NP only", "NP+Tier": "NP+Tier", "NP+Tier+Async": "NP+Tier+Async",
             "NP+Tier+Async+Lossy": "NP+Tier+Async"}
# Hues for an arm no entry above names, taken in fixed order.
FALLBACK_COLORS = ["#008300", "#e87ba4", "#eda100"]

# Codecs that are lossless however they were invoked: panel (b) gives ndzip the
# run's error bound, but it ignores it.
LOSSLESS_BASES = {"ndzip", "GPULZ"}

# Canonical left-to-right arm order within each panel; an arm the CSV has and
# these lists lack is appended in the order it first appears.
ORDER_A = ["Baseline", "nvCOMP", "nvCOMP+Tier", "NP only", "NP+Tier",
           "NP+Tier+Async", "NP+Tier+Async+Lossy"]
ORDER_B = ["ndzip", "ndzip+Tier", "GPULZ", "GPULZ+Tier", "cuSZ", "cuSZ+Tier", "cuSZp3", "cuSZp3+Tier",
           "Best fixed nvCOMP", "Best fixed nvCOMP+Tier",
           "Best fixed nvCOMP (lossless)", "Best fixed nvCOMP+Tier (lossless)",
           "Worst fixed nvCOMP", "Worst fixed nvCOMP+Tier",
           "NeuroPress", "NeuroPress+Tier", "XGB", "XGB+Tier",
           "HCompress", "HCompress+Tier"]

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


# ARMS MEASURED BUT NOT DRAWN. panel_order() deliberately appends any arm it
# does not recognise, so a campaign with a new arm never loses bars silently --
# which means removing a name from ORDER_A/ORDER_B does NOT hide it. Anything
# listed here is filtered out instead. The runs still happen and their CSVs are
# kept; only the bar is suppressed.
#
#   nvCOMP, +Tier  panel (a)'s fixed-codec rungs, dropped from the arm set on
#                  2026-09-23; older campaigns still carry their rows.
#   NeuroPress     panel (b)'s untiered NeuroPress at the bound. A choice, not
#                  a dedup: it is NOT a duplicate of any drawn arm (arms.csv:
#                  eb=1e-3 tier=0 async=0, against the lossy rung's tier=1
#                  async=500), and measured 2-3.4x apart from it. Left off to
#                  keep the chart to the ablation ladder plus the external
#                  codecs. NOTE the panel (b) reduction table below still
#                  prints NeuroPress-vs-codec percentages from its rows, so
#                  the text reports a bar the chart does not draw.
HIDDEN_ARMS = {"nvCOMP", "nvCOMP+Tier", "NeuroPress"}


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
    seen = [s for s in seen if base_name(s) not in HIDDEN_ARMS
            and s not in HIDDEN_ARMS]
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
                # The WORST fixed codec sits right after Baseline, untiered
                # then tiered. Both are slower than Baseline on Nyx, so putting
                # them beside it reads as the ceiling the rest of the chart is
                # measured against -- and the pair's own untiered/tiered step is
                # visible before the eye travels along the descending bars.
                ("b", "Worst fixed nvCOMP"), ("b", "Worst fixed nvCOMP+Tier"),
                ("b", "ndzip"), ("b", "ndzip+Tier"),
                ("b", "GPULZ"), ("b", "GPULZ+Tier"),
                ("b", "cuSZ"), ("b", "cuSZ+Tier"),
                ("b", "cuSZp3"), ("b", "cuSZp3+Tier"),
                ("a", "nvCOMP"), ("a", "nvCOMP+Tier"),
                ("b", "Best fixed nvCOMP"), ("b", "Best fixed nvCOMP+Tier"),
                ("b", "Best fixed nvCOMP (lossless)"),
                ("b", "Best fixed nvCOMP+Tier (lossless)"),
                # HCompress last among the panel (b) comparators, directly
                # before NeuroPress's own ladder: it is the other SELECTOR,
                # ranking the same candidates under the same cost model, so
                # it belongs beside NeuroPress rather than among the codecs.
                ("b", "XGB"), ("b", "XGB+Tier"),
                ("b", "HCompress"), ("b", "HCompress+Tier")]
# THE FOUR FIGURE 9 PANELS (2026-09-24), one question each. Items are
# (panel, arm); panels "m", "c" and "l" are derived rows figure_rows() copies
# from measured ones, so one measurement can appear under a clearer name.
#
# (a) Tiering: what the RAM tier and the async flush buy, on lossless
#     NeuroPress -- uncompressed, compressed, +tier, +async flush.
FIG_TIERING = [("a", "Baseline"), ("a", "NP only"), ("a", "NP+Tier"),
               ("a", "NP+Tier+Async")]
# (b) Model performance: three selectors ranking the SAME 32 candidates under
#     the SAME balanced cost model on the SAME storage (untiered, no flush),
#     lossless and at the bound. Only the predictor differs.
FIG_MODELS = [("m", "XGB (lossless)"), ("m", "HCompress (lossless)"),
              ("m", "NeuroPress (lossless)")]   # lossless only (user, 2026-09-29)
# (c) External codecs against NeuroPress at the same bound, each untiered and
#     with every storage feature on (tier, async flush): NeuroPress's matching
#     bars are its untiered lossy arm and NP+Tier+Async+Lossy.
FIG_EXTERNAL = [("c", "Worst fixed nvCOMP"), ("c", "Worst fixed nvCOMP+Tier"),
                ("c", "Best fixed nvCOMP"), ("c", "Best fixed nvCOMP+Tier"),
                ("c", "ndzip"), ("c", "ndzip+Tier"),
                ("c", "GPULZ"), ("c", "GPULZ+Tier"),
                ("c", "cuSZ"), ("c", "cuSZ+Tier"),
                ("c", "cuSZp3"), ("c", "cuSZp3+Tier"),
                ("c", "NeuroPress (lossless)"), ("c", "NeuroPress+Tier (lossless)")]
# (d) Lossless against lossy NeuroPress, untiered and tiered.
FIG_LOSSY = [("l", "NeuroPress (lossless)"), ("l", "NeuroPress"),
             ("l", "NeuroPress+Tier (lossless)"), ("l", "NeuroPress+Tier")]
# Selector-baseline arms: drawn in fig9b_models, never in the combined chart.
MODEL_ARM_PREFIXES = ("HCompress", "XGB")
# Lossy compressors dropped from the comparison (user, 2026-09-28): their arms,
# untiered and +Tier, are kept in the data but drawn in no figure.
DROPPED_CODECS = {"cuSZ", "cuSZp3"}
# Arms that may have no measured row yet; they get a TBD slot until one lands.
PLANNED_ARMS = {"HCompress", "XGB", "HCompress (lossless)", "XGB (lossless)"}

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
# Set by --app-time: bars show the application wall clock, and draw_bars adds
# the part beyond the measured bar as its own band. A list, like YLABEL, so
# main() can set it without threading a flag through the drawing code.
APP_TIME = [False]
# The --app-time overhead band: input read + H2D staging + runtime setup.
# Neutral and identical for every arm, because it is the same replay work
# whichever codec ran; dotted, so it stays distinct from the pale device-I/O
# band in grayscale, where two light tones alone would merge.
OVERHEAD_FILL, OVERHEAD_EDGE, OVERHEAD_HATCH = "#e3e2de", "#8c8b87", "...."
# Arms drawn WITHOUT an error bar even when they have repeated runs (user,
# 2026-09-24). Their bars are still the mean of those runs.
NO_WHISKER_ARMS = {"HCompress", "HCompress+Tier"}
# Set by --job-time: bars show the whole SLURM job as one solid segment. No
# split: the job's clock has no measured compute/I-O breakdown of its own.
JOB_TIME = [False]
# Per figure: draw each bar as ONE solid segment (its total), whatever the
# view -- set for the figure being rendered from its key_kw "solid".
SOLID = [False]
# Per figure: value labels read horizontally when they fit their bar's slot
# (key_kw "horizontal"); upright otherwise, as every other figure draws them.
HORIZONTAL = [False]
# Set by --note: a description of the run printed across the top of every
# figure of this invocation (cost model, learning, data size ...), so a
# figure from a side experiment cannot be mistaken for the paper's.
RUN_NOTE = [None]
NOTE_FS = 9
# Per figure: write each bar's compression ratio inside it (key_kw "ratios";
# only the models figure sets it).
RATIOS = [True]
# Per figure: sub-groups inside each workload, [(label, {arm, ...}), ...], drawn
# as a two-level x axis (sub-group label, then the workload) with a gap between
# sub-groups and a separator between workloads. None: one level, as before.
SPLIT = [None]
SPLIT_GAP = 0.35                   # slots between two sub-groups of one workload
GROUP_W_SPLIT = 0.97               # the separator marks the workload, so the gutter
                                   # shrinks and each bar keeps the >= 14 pt a 10 pt
                                   # ratio chip needs (ratio_font)


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


def load_app_time(csv_path):
    """The APPLICATION wall clock for the arm whose fig9.csv is `csv_path`.

    Read from the arm's sibling app.csv, which holds the replay driver's own
    `time:` line: the whole run, setup included. Kept beside fig9.csv rather
    than added to it, so the figure's CSVs stay byte-identical to the
    campaign's.

    @param csv_path Path of one arm's fig9.csv.
    @return app_wall_s as a string, or None when the arm has no app.csv.
    """
    app = os.path.join(os.path.dirname(csv_path), "app.csv")
    if not os.path.exists(app):
        return None
    rows = load(app)
    return rows[-1].get("app_wall_s") if rows else None


def load_app_std(csv_path):
    """Sample std of the application wall clock over repeated runs, if any.

    @param csv_path Path of one arm's fig9.csv; reads its sibling app.csv.
    @return app_wall_std_s as a string, or None for a single-run arm.
    """
    app = os.path.join(os.path.dirname(csv_path), "app.csv")
    if not os.path.exists(app):
        return None
    rows = load(app)
    return rows[-1].get("app_wall_std_s") if rows else None


def apply_app_time(D):
    """Make every bar the APPLICATION wall clock instead of the measured one.

    The measured bar is the replay driver's total MINUS the input read and the
    host->device staging -- both replay artifacts an in-situ producer never
    pays -- and without the runtime setup the driver itself excludes. This
    puts all three back. compute and io keep their measured values, so the
    difference, total - (compute + io), is drawn by draw_bars as its own band:
    a reader sees the application time AND how much of it is that overhead.

    The overhead is near-constant WITHIN a workload -- the same dump is read
    for every arm -- so this shifts each workload's bars by about the same
    amount and compresses the relative differences between them.

    An arm with no application time becomes TBD rather than keeping its
    measured total: a chart mixing the two would compare different clocks.

    @param D Indexed rows from index(); changed in place.
    @return Arms that had no application time.
    """
    missing = []
    for k, d in D.items():
        if d.get("total") is None:
            continue
        if d.get("app") is None:
            missing.append("/".join(k))
            d["total"] = None
            continue
        d["bar"] = d["total"]
        d["total"] = d["app"]
        # The measured bar's std is a different quantity; use the
        # application time's own, or none for a single run.
        d["std"] = d.get("app_std")
    return missing


def load_job_times(path):
    """Every arm's SLURM job wall clock, less the harness's own work.

    Three parts are taken out, none of them the arm's:
      - after_app_s: the post-run check. The harness checksums every blob
        and, for lossless arms only, reads all of them back through the
        decompressor to prove the round trip bit-exact -- 180-290 s on a
        lossless NeuroPress arm, 40-75 s on a lossy one.
      - warm_etc_s: the pre-run harness (wrapper checks, cleaning leftovers,
        warming the page cache) and the post-run harness (CSVs, the
        harness's quick-look plot, deleting the tier files, teardown).
      - staging_s: copying the input dump from Lustre onto the node, which
        depends on Lustre's load, not on the arm.
      - setup_s: the runtime's one-off start-up (pools, allocators), paid
        once per run whatever the arm.
    What is left is the application's timed loop: input read, host->device
    copy, compute and device I/O.

    @param path job_phases.csv: workload, arm, job_s (sacct Elapsed of the
                job that ran that one arm), after_app_s, warm_etc_s,
                staging_s and setup_s.
    @return {(workload, arm): seconds}.
    """
    out = {}
    for r in load(path):
        parts = [_f(r.get(c)) for c in
                 ("job_s", "after_app_s", "warm_etc_s", "staging_s",
                  "setup_s")]
        out[(r["workload"], r["arm"])] = (
            None if None in parts else parts[0] - sum(parts[1:]))
    return out


def load_job_time_std(path):
    """Sample std of the job-time bar over repeated runs, per arm.

    @param path job_phases.csv; reads its optional job_time_std_s column.
    @return {(workload, arm): seconds}, only for arms that have one.
    """
    out = {}
    for r in load(path):
        v = _f(r.get("job_time_std_s"))
        if v is not None:
            out[(r["workload"], r["arm"])] = v
    return out


def apply_job_time(D, jobs):
    """Make every bar the whole SLURM job instead of the application clock.

    Runs after apply_app_time, so each row still has its application time in
    "app". The bar is drawn solid; load_job_times has already taken out the
    harness's own work, the input staging and the runtime setup, so what is
    left is the application's timed loop.

    An arm with no job time becomes TBD, for the same reason as in
    apply_app_time: one chart, one clock.

    @param D Indexed rows from index(), after apply_app_time; changed in place.
    @param jobs {(workload, arm): seconds} from load_job_times().
    @return Arms that had no job time.
    """
    missing = []
    for (p, s, w), d in D.items():
        if d.get("total") is None:
            continue
        job = jobs.get((w, s))
        if job is None:
            missing.append(f"{p}/{s}/{w}")
            d["total"] = None
            continue
        d["total"] = job
    return missing


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
                        setup=_sec("setup_min"),
                        # Seconds already (app.csv is not in minutes).
                        app=_f(r.get("app_s")),
                        app_std=_f(r.get("app_std_s")))
    return out, warn


# Derived rows: (panel, arm) -> the measured (panel, arm) it copies.
DERIVED = {
    ("m", "XGB (lossless)"): ("b", "XGB (lossless)"),
    ("m", "HCompress (lossless)"): ("b", "HCompress (lossless)"),
    ("m", "NeuroPress (lossless)"): ("a", "NP only"),
    ("m", "XGB"): ("b", "XGB"),
    ("m", "HCompress"): ("b", "HCompress"),
    ("m", "NeuroPress"): ("b", "NeuroPress"),
    ("c", "Worst fixed nvCOMP"): ("b", "Worst fixed nvCOMP"),
    # Lossless paper (user, 2026-09-29): the external figure's Best nvCOMP is
    # the no--q rerun, and NeuroPress is its lossless ladder.
    ("c", "Best fixed nvCOMP"): ("b", "Best fixed nvCOMP (lossless)"),
    ("c", "NeuroPress (lossless)"): ("a", "NP only"),
    ("c", "NeuroPress+Tier (lossless)"): ("a", "NP+Tier+Async"),
    ("c", "ndzip"): ("b", "ndzip"),
    ("c", "GPULZ"): ("b", "GPULZ"),
    ("c", "cuSZ"): ("b", "cuSZ"),
    ("c", "cuSZp3"): ("b", "cuSZp3"),
    ("c", "NeuroPress"): ("b", "NeuroPress"),
    ("c", "Worst fixed nvCOMP+Tier"): ("b", "Worst fixed nvCOMP+Tier"),
    ("c", "Best fixed nvCOMP+Tier"): ("b", "Best fixed nvCOMP+Tier (lossless)"),
    ("c", "ndzip+Tier"): ("b", "ndzip+Tier"),
    ("c", "GPULZ+Tier"): ("b", "GPULZ+Tier"),
    ("c", "cuSZ+Tier"): ("b", "cuSZ+Tier"),
    ("c", "cuSZp3+Tier"): ("b", "cuSZp3+Tier"),
    ("c", "NeuroPress+Tier"): ("a", "NP+Tier+Async+Lossy"),
    ("l", "NeuroPress (lossless)"): ("a", "NP only"),
    ("l", "NeuroPress"): ("b", "NeuroPress"),
    ("l", "NeuroPress+Tier (lossless)"): ("a", "NP+Tier+Async"),
    ("l", "NeuroPress+Tier"): ("a", "NP+Tier+Async+Lossy"),
}


def figure_rows(D, workloads):
    """Build the derived rows figures (b)-(d) draw, in place.

    Each derived row IS its source measurement under the name the figure uses
    (DERIVED). A lossless-only workload (AI) has no lossy NeuroPress rung, so
    its NeuroPress+Tier falls back to NP+Tier+Async. PLANNED_ARMS with no
    measured row get an empty row, drawn TBD.

    @param D Indexed rows from index(); derived rows are added in place.
    @param workloads Workload names.
    @return Workloads whose NeuroPress+Tier bar is the lossless fallback.
    """
    empty = dict(compute=None, io=None, total=None, std=None, ratio=None,
                 eb=None, bound="", payload=None, setup=None)
    lossless = []
    for w in workloads:
        for (p, s), src in DERIVED.items():
            if src + (w,) in D:
                D[(p, s, w)] = D[src + (w,)]
            elif (p, s) == ("c", "NeuroPress+Tier") and ("a", "NP+Tier+Async", w) in D:
                D[(p, s, w)] = D[("a", "NP+Tier+Async", w)]
                lossless.append(w)
            elif s in PLANNED_ARMS:
                D[(p, s, w)] = empty
    return lossless


# Worst fixed nvCOMP ran WITHOUT -q on these workloads (figure_9.sh defaults:
# static-deflate on Nyx, static-deflate-s4 on VPIC and LAMMPS), so the 1e-3 its
# rows record was never applied -- they are lossless; per-chunk ratios confirm
# it (medians 1.18-1.26). WarpX's worst is static-deflate-q-s4: quantized.
WORST_UNQUANTIZED = {"VPIC", "Nyx", "LAMMPS"}


def lossless_rows(D):
    """The rows a lossless-only figure may draw.

    A row is lossy when it ran at a positive bound and its codec applies it:
    every arm except the always-lossless codecs (LOSSLESS_BASES) and the Worst
    fixed nvCOMP rows WORST_UNQUANTIZED marks as unquantized.

    @param D Indexed rows from index(), derived rows included.
    @return A new dict without the lossy rows. An unquantized Worst row is a
      copy carrying eb 0, so no dagger marks it; every other row is shared.
    """
    out = {}
    for (p, s, w), d in D.items():
        if (d.get("eb") or 0) <= 0 or base_name(s) in LOSSLESS_BASES:
            out[(p, s, w)] = d
        elif s.startswith("Worst fixed nvCOMP") and w in WORST_UNQUANTIZED:
            out[(p, s, w)] = dict(d, eb=0.0)
    return out


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
        out.append(w if not SHOW_PAYLOAD else (f"{w}\n{size}" if size else w))
    return out


def luminance(hexcolor):
    """Relative luminance (0 dark .. 1 light) of a hex colour, for text contrast."""
    r, g, b = (int(hexcolor.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ratio_text(ratio):
    """A bar's compression ratio as drawn on it, or None when it has none."""
    if ratio is None or not ratio == ratio or ratio <= 0:
        return None
    return f"{ratio:.2f}\u00d7" if ratio < 10 else f"{ratio:.1f}\u00d7"


def ratio_font(ax, text, fs, bar_height, slot_width, ylim):
    """Font size (pt) that fits `text`, rotated on its chip, inside one bar.

    Starts from the value-label size less one point and shrinks until the
    text's length fits the bar's height and its chip fits the bar's SLOT (bar
    plus its gap): on a narrow bar the chip may overhang the bar's edges, but
    never reaches the neighbouring slot, so two chips cannot touch. Both
    measured on the axes. It is drawn at the 10 pt EuroSys floor or not at
    all (the ratio then joins the time label above the bar).

    @param ax the axes
    @param text the ratio label
    @param fs the value-label font size
    @param bar_height bar height, data units
    @param slot_width one bar's slot, data units
    @param ylim the axis top, data units
    """
    bb = ax.get_window_extent()
    h_pt = bar_height / ylim * bb.height / ax.figure.dpi * 72
    x0, x1 = ax.get_xlim()
    w_pt = slot_width / (x1 - x0) * bb.width / ax.figure.dpi * 72
    size = min(fs, (h_pt - 4.0) / (0.56 * len(text)), (w_pt - 1.0) / 1.3)
    return FS_MIN if size >= FS_MIN else None


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
            # Against the MEASURED bar: under --app-time total is the
            # application clock, which is compute + io + the overhead band.
            meas = d.get("bar", d["total"])
            if abs(d["compute"] + d["io"] - meas) > 0.05:
                print(f"  [split]  {p}/{w}/{s}: compute+io={d['compute']+d['io']:.3f} "
                      f"!= total={meas:.3f}"); n += 1

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
             ("NP+Tier", "nvCOMP+Tier")]
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
    slot_pt = axes_pt / n_groups * (GROUP_W_SPLIT if SPLIT[0] else GROUP_W) / nmax
    return min(FS_VAL, math.floor(slot_pt * 0.9 / 0.80 * 2) / 2)


def draw_split_axis(ax, workloads, split, spans):
    """Two-level x axis: each sub-group's label under its bars, the workload
    under those, and a separator between workloads through both label rows.

    @param ax Axes to draw on.
    @param workloads Workload names, left to right.
    @param split [(label, arms), ...] from SPLIT.
    @param spans {(workload index, sub-group index): [bar centre x, ...]}.
    """
    keys = sorted(spans)
    ax.set_xticks([sum(spans[k]) / len(spans[k]) for k in keys])
    ax.set_xticklabels([split[i][0] for _, i in keys], fontsize=FS_TICK, color=INK)
    for j, w in enumerate(workloads):
        ax.annotate(w, (j, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -22), textcoords="offset points", ha="center",
                    va="top", fontsize=FS_TITLE, color=INK)
    # "|" between workloads, from the axis down past the workload name.
    drop = 38.0 / (ax.get_position().height * ax.figure.get_figheight() * 72.0)
    for j in range(1, len(workloads)):
        ax.plot([j - 0.5, j - 0.5], [0, -drop], transform=ax.get_xaxis_transform(),
                clip_on=False, color=INK_MUTED, linewidth=0.8)


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
    # Limits first: ratio_font() measures bars against the final axes.
    ax.set_xlim(-0.5, len(workloads) - 0.5)
    ax.set_ylim(0, ylim)
    split = SPLIT[0]
    sub_of = {s: i for i, (_, arms) in enumerate(split or []) for s in arms}
    slot = (GROUP_W_SPLIT if split else GROUP_W) / nmax
    spans = {}                   # (workload, sub-group) -> bar centres
    for j, w in enumerate(workloads):
        present = [it for it in items if it + (w,) in D]
        subs = [sub_of.get(s, 0) for _, s in present]
        width = len(present) + (SPLIT_GAP * (len(set(subs)) - 1) if split else 0)
        start = j - width * slot / 2.0
        off = 0.0
        for k, (p, s) in enumerate(present):
            if split and k and subs[k] != subs[k - 1]:
                off += SPLIT_GAP
            d, x = D[(p, s, w)], start + slot * (k + off + 0.5)
            spans.setdefault((j, subs[k]), []).append(x)
            col, tiered = arm_style(s, spare)
            total, comp = d.get("total"), d.get("compute")
            if total is None:
                ax.text(x, ylim * 0.01, "TBD", ha="center", va="bottom",
                        rotation=90, fontsize=fs or FS_MIN, color=INK_MUTED)
                continue
            drawn = min(total, ylim)
            one = JOB_TIME[0] or SOLID[0]
            if comp is None and not one:
                warn.append(f"{p}/{w}/{s}")  # no split: one solid segment
            solid = drawn if comp is None or one else min(comp, drawn)
            ax.bar(x, solid, width=slot * BAR_W, color=col, edgecolor="white",
                   linewidth=0, hatch=TIER_HATCH if tiered else None, zorder=3)
            io = d.get("io")
            if one:
                pass  # one solid bar: the job clock (or this figure) has no split
            elif APP_TIME[0] and comp is not None and io is not None:
                # The measured bar ends at compute + io; the rest of the
                # application clock is the replay overhead, drawn on top.
                meas_top = min(comp + io, drawn)
                if meas_top > solid:
                    ax.bar(x, meas_top - solid, bottom=solid,
                           width=slot * BAR_W, linewidth=0,
                           color=blend_to_white(col, IO_BLEND), zorder=3)
                if drawn > meas_top:
                    ax.bar(x, drawn - meas_top, bottom=meas_top,
                           width=slot * BAR_W, linewidth=0,
                           color=OVERHEAD_FILL, edgecolor=OVERHEAD_EDGE,
                           hatch=OVERHEAD_HATCH, zorder=3)
            elif drawn > solid:
                ax.bar(x, drawn - solid, bottom=solid, width=slot * BAR_W,
                       linewidth=0, color=blend_to_white(col, IO_BLEND), zorder=3)
            clipped = total > ylim
            top = drawn
            std = None if s in NO_WHISKER_ARMS else d.get("std")
            if std and std > 0 and not clipped:
                # +-1 sample std over the arm's repeated runs. Single-run
                # arms carry no std and get no whisker.
                ax.errorbar(x, total, yerr=std, fmt="none", ecolor=INK,
                            elinewidth=0.8, capsize=2.0, capthick=0.8,
                            zorder=5)
                top = min(total + std, ylim)
            # Clear of the whisker's cap, so the cap never reads as a "+".
            gap = 3.5 if (std and std > 0 and not clipped) else 1.5
            if fs is None:
                continue  # no label fits at the 10 pt floor: numbers are in the per-workload figures
            label = f"{total:.{dec}f}"
            rtxt = ratio_text(d.get("ratio")) if RATIOS[0] else None
            rfs = ratio_font(ax, rtxt, fs, drawn, slot, ylim) if rtxt else None
            if rtxt and not rfs and not clipped:
                label = f"{label} \u00b7 {rtxt}"  # bar too short: with the time
            if rfs:
                # The compression ratio INSIDE the bar, at its middle, on a small
                # white chip: legible over any colour, hatch or stacked segment.
                # Horizontal, like the time label, when the chip fits the bar.
                bar_pt = (slot * BAR_W * FIG_W * (AX_RIGHT - AX_LEFT) * 72.0
                          / len(workloads))
                flat_r = HORIZONTAL[0] and 0.52 * rfs * len(rtxt) + 4 <= bar_pt
                ax.annotate(rtxt, (x, drawn / 2), ha="center", va="center",
                            rotation=0 if flat_r else 90, fontsize=rfs,
                            color=INK, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                      edgecolor="none", alpha=0.92))
            # Horizontal only where the text fits the slot (digits ~0.52 em).
            slot_pt = slot * FIG_W * (AX_RIGHT - AX_LEFT) * 72.0 / len(workloads)
            flat = HORIZONTAL[0] and 0.52 * fs * len(label) <= 0.95 * slot_pt
            ax.annotate(label, (x, top),
                        xytext=(0, -1.5 if clipped else gap),
                        textcoords="offset points", ha="center",
                        va="top" if clipped else "bottom",
                        rotation=0 if flat else 90,
                        fontsize=fs, color="white" if clipped else INK, zorder=6)
    ax.set_xlim(-0.5, len(workloads) - 0.5)
    ax.set_ylim(0, ylim)
    if split:
        draw_split_axis(ax, workloads, split, spans)
    else:
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


def legend_handles(D, items, spare, bound=False):
    """Legend entries for one figure: one per colour, then the symbols.

    A tiered arm and its untiered twin share a colour, so they share an entry;
    the hatched swatch says what the hatch means for every arm at once, and
    the dagger's meaning follows it.

    @param D Indexed rows from index().
    @param items (panel, arm) pairs in plotting order.
    @param spare Fallback-colour assignments, shared across figures.
    @param bound True to end with "† lossy, ε = ..." when some arm here is
      lossy and the run has one bound (a column figure prints it instead).
    @return Patch handles in first-appearance order; the +Tier swatch only
      when some arm is tiered, the dagger entry only when asked and lossy.
    """
    handles, seen, any_tier = [], set(), False
    for _, s in items:
        col, tiered = arm_style(s, spare)
        full = _LADDER_RE.sub("", s)
        any_tier |= tiered and full not in NP_LADDER  # ladder entries show their own hatch
        mark = arm_label(s, D)[len(full):]      # " †", ", ε = ..." or ""
        if full in NP_LADDER:
            # NeuroPress's own steps are named in full, each with its own
            # swatch (hatched when tiered): folding NP+Tier into "NP" plus the
            # hatch key left readers unable to match three greens to two
            # entries (user, 2026-09-29).
            label = NP_LADDER[full] + mark
            if (col, label) not in seen:
                seen.add((col, label))
                handles.append(Patch(facecolor=col, edgecolor="white", linewidth=0,
                                     hatch=TIER_HATCH if tiered else None, label=label))
            continue
        name = full.replace("+Tier", "")
        label = SHORT_NAMES.get(name, name) + mark
        if (col, label) in seen:
            continue
        seen.add((col, label))
        handles.append(Patch(facecolor=col, edgecolor="white", linewidth=0,
                             label=label))
    if any_tier:
        handles.append(Patch(facecolor=KEY_GREY, edgecolor="white", linewidth=0,
                             hatch=TIER_HATCH, label=TIER_LABEL))
    ebs = lossy_bounds(D)
    if bound and len(ebs) == 1 and any("†" in h.get_label() for h in handles):
        # Text only: an invisible swatch keeps it aligned with its neighbours.
        handles.append(Patch(facecolor="none", edgecolor="none",
                             label=f"† lossy, {eps_text(ebs[0])}"))
    return handles


WHISKER_TEXT = "whiskers ±1 std over repeated runs"


def has_whiskers(D):
    """True when some bar carries a +-1 std whisker.

    @param D Indexed rows from index().
    @return Whether any arm outside NO_WHISKER_ARMS has a std above zero.
    """
    return any((d.get("std") or 0) > 0 for (_, s, _), d in D.items()
               if s not in NO_WHISKER_ARMS)


def row_major(handles, ncol):
    """Reorder legend entries so they read left to right, row by row.

    Matplotlib fills a legend column by column; this permutation makes that
    fill lay the entries out in reading order.

    @param handles Entries in reading order.
    @param ncol Number of legend columns.
    @return The permuted list.
    """
    n = len(handles)
    return [handles[r * ncol + c] for c in range(ncol)
            for r in range(math.ceil(n / ncol)) if r * ncol + c < n]


def fit_legend_cols(handles, width_in, max_cols, kw):
    """Legend columns for the fewest rows that fit a width, spread evenly.

    Candidates are drawn on a scratch figure and measured, widest first; the
    first that fits sets the row count, and the columns are then the fewest
    that keep it (7 entries in 2 rows are 4 + 3, not 6 + 1).

    @param handles Legend entries.
    @param width_in Width the legend may take, inches.
    @param max_cols Most columns to try.
    @param kw Legend keyword arguments to measure with (font, spacing).
    @return (columns, the legend's height in inches at that column count).
    """
    if not handles:
        return 1, 0.0
    fig = plt.figure(figsize=(width_in + 2.0, 4.0))
    rend = fig.canvas.get_renderer()

    def measure(ncol):
        leg = fig.legend(handles=row_major(handles, ncol), ncol=ncol,
                         frameon=False, borderaxespad=0, **kw)
        box = leg.get_window_extent(rend)
        leg.remove()
        return box.width / fig.dpi, box.height / fig.dpi

    n = len(handles)
    fit = next((c for c in range(min(max_cols, n), 1, -1)
                if measure(c)[0] <= width_in), 1)
    ncol = math.ceil(n / math.ceil(n / fit))
    height = measure(ncol)[1]
    plt.close(fig)
    return ncol, height


# The full-width arm legend's type and spacing, shared by the fit and the draw.
LEG_KW = dict(fontsize=FS_LEG, handlelength=1.4, handleheight=1.0,
              handletextpad=0.5, columnspacing=1.6, labelspacing=0.4)


def draw_legend(fig, ax, handles, ncol, title):
    """The arm legend, centred above the plot at its natural width.

    @param fig The figure.
    @param ax The plot axes; the legend sits on its top edge.
    @param handles Entries from legend_handles(), in reading order.
    @param ncol Columns, from fit_legend_cols().
    @param title Figure name printed over the entries, or None for none.
    """
    # Centred at its natural width: an edge-to-edge legend (mode="expand")
    # left wide gaps between entries and pushed the last one to the margin.
    leg = ax.legend(handles=row_major(handles, ncol), loc="lower center",
                    bbox_to_anchor=(0.5, 1.03), borderaxespad=0, ncol=ncol,
                    frameon=False, title=title, title_fontsize=FS_TITLE,
                    alignment="center", **LEG_KW)
    for t in leg.get_texts():
        t.set_color(INK)
    leg.get_title().set_color(INK)


def draw_key(fig, D, tier_text=None, notes=()):
    """Under the axis: what a bar's parts and its hatching mean, the whisker
    note and any extra notes. The dagger's meaning is in the arm legend above
    the plot (legend_handles), not here.

    @param fig The figure.
    @param D Indexed rows from index(), for the error bound.
    @param tier_text Legend text for the hatched swatch.
    @param notes Extra text-only entries appended after the bound.
    """
    # NOT "compute". The solid segment is total minus the measured device I/O,
    # so it also holds scheduling, allocation and per-chunk library setup --
    # on the cuSZ arm 82% of the bar is the resource manager being built and
    # torn down, which a reader would never guess from the word "compute".
    # A --job-time bar is one solid segment, so there are no parts to name.
    parts = [] if (JOB_TIME[0] or SOLID[0]) else [
        Patch(facecolor=KEY_GREY, linewidth=0, label="non-I/O elapsed"),
        Patch(facecolor=blend_to_white(KEY_GREY, IO_BLEND), linewidth=0,
              label="device I/O (measured)"),
        *([Patch(facecolor=OVERHEAD_FILL, edgecolor=OVERHEAD_EDGE,
                 linewidth=0, hatch=OVERHEAD_HATCH,
                 label="input read + H2D + setup (replay)")]
          if APP_TIME[0] else [])]
    key = [*parts,
           *([Patch(facecolor=KEY_GREY, edgecolor="white", linewidth=0,
                    hatch=TIER_HATCH, label=tier_text)] if tier_text else [])]
    if has_whiskers(D):
        key.append(Patch(facecolor="none", edgecolor="none", label=WHISKER_TEXT))
    for n in notes:
        key.append(Patch(facecolor="none", edgecolor="none", label=n))
    # TWO ROWS, NOT ONE. At 10 pt these four entries measure ~7.5 in on a
    # single line and ran off a 7 in figure, losing the error bound entirely.
    # The labels are not shortened to fit: "non-I/O elapsed" is deliberate
    # wording (see above) and the bound belongs in the figure, not the caption.
    if not key:
        return
    ncol = 2 if FS_LEG >= 9 else len(key)
    leg = fig.legend(handles=key, loc="lower left", borderaxespad=0,
                     bbox_to_anchor=(AX_LEFT - 0.01, 0.005), ncol=ncol,
                     frameon=False, fontsize=FS_LEG, handlelength=1.5,
                     handleheight=1.1, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK)


def render_single(path, D, items, workloads, ylim, dec, spare, warn, title,
                  key_kw=None):
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
    @param key_kw Keyword arguments passed on to draw_key(), or None; its
      show_bound is read here, for the legend's dagger entry.
    @return The value-label font size used.
    """
    key_kw = dict(key_kw or {})
    show_bound = key_kw.pop("show_bound", True)
    nmax = max(sum(1 for it in items if it + (w,) in D) for w in workloads)
    if SPLIT[0]:
        nmax += SPLIT_GAP * (len(SPLIT[0]) - 1)
    fs = label_size(nmax, len(workloads))
    below_floor_ok = key_kw.pop("labels_below_floor", False)
    if fs < FS_MIN and below_floor_ok:
        print(f"note: {os.path.basename(path)}: value labels at {fs:g} pt, UNDER the "
              f"{FS_MIN} pt EuroSys floor (kept on request for this figure)")
    elif fs < FS_MIN:
        print(f"note: {os.path.basename(path)}: value labels would be {fs:g} pt, under "
              f"the {FS_MIN} pt EuroSys floor; omitted (the per-workload figures carry them)")
        fs = None
    # Only arms that draw a bar somewhere get a legend entry.
    drawn = [it for it in items if any(it + (w,) in D for w in workloads)]
    handles = legend_handles(D, drawn, spare, bound=show_bound)
    ncol, leg_in = fit_legend_cols(handles, FIG_W * (AX_RIGHT - AX_LEFT),
                                   MAX_LEGEND_COLS, LEG_KW)
    # The legend's measured height, the 0.03-axes gap under it, a margin over
    # it, and the title line when there is one.
    top_h = (0.26 if title else 0.0) + leg_in + 0.03 * PLOT_H + 0.08
    note = textwrap.wrap(RUN_NOTE[0], 118) if RUN_NOTE[0] else []
    note_h = (0.17 * len(note) + 0.08) if note else 0.0
    top_h += note_h
    # The band under the axis, sized from what it holds: the workload names
    # (one line; two with the payload) and the key, two entries a row.
    n_key = ((0 if (JOB_TIME[0] or SOLID[0]) else (3 if APP_TIME[0] else 2))
             + (1 if key_kw.get("tier_text") else 0)
             + (1 if has_whiskers(D) else 0)
             + len(key_kw.get("notes", ())))
    rows = math.ceil(n_key / 2)
    key_h = 0.12 + 0.22 * (2 if SHOW_PAYLOAD or SPLIT[0] else 1) + 0.27 * rows + (0.06 if rows else 0)
    height = PLOT_H + top_h + key_h
    fig, ax = plt.subplots(figsize=(FIG_W, height))
    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, bottom=key_h / height,
                        top=1.0 - top_h / height)
    draw_bars(ax, D, items, workloads, nmax, ylim, dec, fs, spare, warn)
    draw_legend(fig, ax, handles, ncol, title)
    draw_key(fig, D, **key_kw)
    if note:
        fig.text(0.5, 1.0 - 0.05 / height, "\n".join(note), ha="center", va="top",
                 fontsize=NOTE_FS, color=INK_MUTED, linespacing=1.2)
    # NO bbox_inches="tight". It GROWS the canvas to swallow any artist that
    # overflows the figure, so the saved file was 7.21 in however FIG_W was
    # set -- and the saved width is what LaTeX places, so the EuroSys text
    # block was being exceeded silently. The layout below is explicit
    # (subplots_adjust + a computed height), so the tight box was only hiding
    # overflow that should be fixed instead.
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return fs


def workload_legend(fig, D, items, spare, width_in):
    """The arm legend for a column figure: the fewest rows that fit, measured.

    The same entries as the full-width figure's legend: one per colour, then
    the +Tier hatch.

    @param fig The figure.
    @param D Indexed rows from index().
    @param items (panel, arm) pairs of this workload, in plotting order.
    @param spare Fallback-colour assignments, shared across figures.
    @param width_in Width available to the legend, inches.
    @return The legend, placed at the figure's top-left (caller re-anchors).
    """
    handles = legend_handles(D, items, spare)
    kw = dict(fontsize=FS_MIN, handlelength=1.3, handleheight=1.0,
              handletextpad=0.5, columnspacing=0.9, labelspacing=0.35)
    ncol, _ = fit_legend_cols(handles, width_in, 4, kw)
    leg = fig.legend(handles=row_major(handles, ncol), loc="upper left",
                     ncol=ncol, frameon=False, borderaxespad=0.0,
                     bbox_to_anchor=(0, 1), **kw)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def render_workload(path, D, items, w, spare, key_kw=None):
    """One workload's bars as a single-column EuroSys figure (COL_W wide).

    The bars are named by a legend above the plot, as in the full-width
    figure (one entry per colour plus the +Tier hatch, fewest rows that fit). The y-axis is fitted to
    this workload alone. All text is at the 10 pt floor. Place with
    width=\\columnwidth, unscaled.

    @param path Output PNG path.
    @param D Indexed rows from index().
    @param items (panel, arm) pairs in plotting order.
    @param w The workload.
    @param spare Fallback-colour assignments, shared across figures.
    @param key_kw The figure's draw_key() arguments (show_bound is read).
    @return The path written, or None when the workload has no bar.
    """
    key_kw = key_kw or {}
    items = [it for it in items if it + (w,) in D]
    totals = [D[it + (w,)]["total"] for it in items if D[it + (w,)]["total"] is not None]
    if not items or not totals:
        return None
    ylim = max(totals) * 1.38
    lo = min(totals)
    dec = 1 if lo >= 2 else (2 if lo >= 0.2 else 3)
    left, right, bottom, title_in, gap = 0.62, 0.08, 0.12, 0.30, 0.10
    fig, ax = plt.subplots(figsize=(COL_W, 5.0))
    leg = workload_legend(fig, D, items, spare, COL_W - 0.10)
    fig.canvas.draw()
    leg_in = leg.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    top = title_in + leg_in + gap
    height = COL_PLOT_H + top + bottom
    fig.set_size_inches(COL_W, height)
    fig.subplots_adjust(left=left / COL_W, right=1 - right / COL_W,
                        bottom=bottom / height, top=1 - top / height)
    leg.set_bbox_to_anchor((0.05 / COL_W, 1 - title_in / height), transform=fig.transFigure)
    # label_size() for THIS axes width: one upright label per slot.
    slot_pt = (COL_W - left - right) * 72 * GROUP_W / len(items)
    fs = min(FS_VAL, math.floor(slot_pt * 0.9 / 0.80 * 2) / 2)
    fs = fs if fs >= FS_MIN else None
    split, SPLIT[0] = SPLIT[0], None     # a column figure keeps one axis level
    draw_bars(ax, D, items, [w], len(items), ylim, dec, fs, spare, [])
    SPLIT[0] = split
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=FS_MIN)
    ax.set_ylabel(YLABEL[0], fontsize=FS_MIN, color=INK)
    size = workload_labels(D, [w])[0].replace("\n", ", ")
    fig.text(0.05 / COL_W, 1 - 0.06 / height, size, ha="left", va="top",
             fontsize=FS_MIN, color=INK, fontweight="bold")
    names = [arm_label(s, D) for _, s in items]
    ebs = lossy_bounds(D) if key_kw.get("show_bound", True) else []
    if len(ebs) == 1 and any("\u2020" in n for n in names):
        fig.text(1 - right / COL_W, 1 - 0.06 / height, f"\u2020 lossy, {eps_text(ebs[0])}",
                 ha="right", va="top", fontsize=FS_MIN, color=INK)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


# Figures that are always also written one workload per single-column PNG.
PER_WORKLOAD_STEMS = {"fig9a_tiering", "fig9b_models", "fig9c_external", "fig9d_lossy"}

# Each figure and its per-workload PNGs go in their own directory under --out.
FIG_DIRS = {"fig9a_tiering": "tiering", "fig9b_models": "models",
            "fig9c_external": "external", "fig9d_lossy": "lossy",
            "fig9": "combined", "fig9a_ablation": "panels",
            "fig9b_baselines": "panels"}


def render_figure(args, D, items, workloads, title, stem, key_kw, spare, warn):
    """Write one figure, and with --per-workload one PNG per workload of it.

    The y-axis is fitted to this figure's own bars. The per-workload PNGs
    share it rather than refitting: refitting would draw a 21 s bar and a 72 s
    bar at the same height, which is exactly the comparison a reader makes
    when the five sit side by side. The cost is whitespace above the shorter
    workloads.

    @param args Parsed command line (out, ylim, per_workload).
    @param D Indexed rows from index().
    @param items (panel, arm) pairs in plotting order.
    @param workloads Workload names, left to right.
    @param title Figure name printed over the legend, or None for none.
    @param stem Output file name without extension; FIG_DIRS maps it to its
      directory under args.out.
    @param key_kw Keyword arguments for draw_key().
    @param spare Fallback-colour assignments, shared so an arm keeps one
      colour across every figure.
    @param warn Collects arms that had a total but no compute/I-O split.
    @return Paths of the PNGs written.
    """
    if not any(it + (w,) in D for it in items for w in workloads):
        print(f"note: no rows for {stem}.png; not written")
        return []
    ylim, dec = pick_ylim(D, args.ylim, items)

    out = os.path.join(args.out, FIG_DIRS.get(stem, stem))
    os.makedirs(out, exist_ok=True)
    pngs = [os.path.join(out, stem + ".png")]
    fs = render_single(pngs[0], D, items, workloads, ylim, dec, spare, warn,
                       title, key_kw)

    # The four panel figures always get their single-column per-workload
    # versions; the combined chart only on --per-workload.
    if not (args.per_workload or stem in PER_WORKLOAD_STEMS):
        return pngs
    for w in workloads:
        p = render_workload(os.path.join(out, f"{stem}_{w.lower()}.png"), D, items, w,
                            spare, key_kw)
        if p:
            pngs.append(p)
    return pngs


def pick_ylim(D, forced, items=None):
    """The shared y limit and the value-label precision.

    @param D Indexed rows from index().
    @param forced --ylim from the command line, or None.
    @param items (panel, arm) pairs the figure draws; None fits every row.
    @return (ylim in seconds, decimals for the value labels).
    """
    keep = None if items is None else set(items)
    totals = [d["total"] for k, d in D.items() if d["total"] is not None
              and (keep is None or k[:2] in keep)]
    dmax = max(totals) if totals else Y_CLIP
    if forced is not None:
        ylim = forced
    else:
        ylim = dmax * (HEADROOM_FLAT if HORIZONTAL[0] else HEADROOM)
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
    ap.add_argument("--app-time", action="store_true",
                    help="bar = the APPLICATION's wall clock, from each "
                         "fig9.csv's sibling app.csv: the measured bar plus "
                         "the input read, host->device staging and runtime "
                         "setup, drawn as a separate band on top")
    ap.add_argument("--job-time", nargs="?", metavar="PATH",
                    const=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "data", "job_phases.csv"),
                    help="bar = the SLURM job's wall clock minus the harness's "
                         "own work (pre-run checks and cache warming, post-run "
                         "verification, teardown), the input staging and the "
                         "runtime setup: the application's timed loop, from "
                         "job_phases.csv "
                         "(default: data/job_phases.csv beside this script); "
                         "drawn as one solid bar")
    ap.add_argument("--deduct-setup", action="store_true",
                    help="subtract each arm's measured cuSZ resource-manager "
                         "construction (setup_min) from its bar; the chart is "
                         "relabelled to say so (default: plot what was measured)")
    ap.add_argument("--per-workload", action="store_true",
                    help="also write one figure per workload "
                         "(fig9_<workload>.png). They SHARE the combined "
                         "chart's y-axis, so bar heights stay comparable "
                         "between them; pass --ylim to set it explicitly")
    ap.add_argument("--combined", action="store_true",
                    help="also write the old single chart fig9.png with every "
                         "arm (default: the four figures fig9a_tiering, fig9b_models, fig9c_external, fig9d_lossy)")
    ap.add_argument("--exclude", action="append", default=[], metavar="WORKLOAD",
                    help="leave this workload out of every figure (by its name "
                         "in the CSV, e.g. AI); repeatable")
    ap.add_argument("--panels", action="store_true",
                    help="also write the per-panel figures fig9a_ablation.png "
                         "and fig9b_baselines.png (default: the single chart only)")
    ap.add_argument("--note", default=None, metavar="TEXT",
                    help="describe the run in a line across the top of every "
                         "figure (cost model, learning, data size ...)")
    ap.add_argument("--model-ratios", action="store_true",
                    help="write each bar's compression ratio inside it in the "
                         "models figure (fig9b_models); used for the ratio-only "
                         "cost-model set, where the ratio is what is optimised")
    args = ap.parse_args()
    RUN_NOTE[0] = args.note

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
    if args.job_time:
        args.app_time = True
    rows = []
    for c in args.csv:
        rs = load(c)
        if args.app_time:
            app = load_app_time(c)
            for r in rs:
                r["app_s"] = app
                r["app_std_s"] = load_app_std(c)
        rows.extend(r for r in rs if r.get("workload") not in args.exclude)
    D, _ = index(rows)
    if args.app_time:
        missing = apply_app_time(D)
        if missing:
            print(f"--app-time: {len(missing)} arm(s) have no app.csv and are "
                  f"drawn TBD, not with their measured total: "
                  f"{', '.join(missing)}", file=sys.stderr)
        YLABEL[0] = "Application wall-clock time (s)"
        APP_TIME[0] = True
        # One more key entry, so one more row under the axis.
        global KEY_H
        KEY_H += 0.27
    if args.job_time:
        missing = apply_job_time(D, load_job_times(args.job_time))
        # The job-time bar is a different quantity from the measured one, so
        # its whisker is its own std (repeated runs only), never std_min.
        job_std = load_job_time_std(args.job_time)
        for (p, s, w), d in D.items():
            d["std"] = job_std.get((w, s))
        if missing:
            print(f"--job-time: {len(missing)} arm(s) have no job time and "
                  f"are drawn TBD: {', '.join(missing)}", file=sys.stderr)
        YLABEL[0] = "Application time excl. setup (s)"
        JOB_TIME[0] = True
        # Two key entries (tier, bound) instead of four: two rows fewer.
        KEY_H -= 2 * 0.27
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
    # ONLY `NeuroPress+Tier` IS A DUPLICATE, and only of panel (a)'s
    # `NP+Tier+Async+Lossy`: same config, same bound, same tier, same flush
    # (figure_9.sh's build_arms says so, and no longer emits it).
    #
    # This used to drop every panel (b) arm whose base_name() was NeuroPress,
    # on the reasoning that it "runs at the same bound" as the lossy rung. Same
    # bound, but NOT the same arm: plain `NeuroPress` is untiered with no async
    # (arms.csv: tier=0 async=0) where the rung is tiered with a 500 ms flush,
    # and they measured 2-3.4x apart (WarpX 82.5 s vs 24.3 s). base_name()
    # strips `+Tier`, so a filter aimed at `NeuroPress+Tier` also swallowed
    # plain `NeuroPress` -- and with it the one untiered bar panel (b)'s
    # untiered external codecs are measured against. The reduction table still
    # printed it, so text and chart disagreed.
    single = ([("a", s) for s in order_a] +
              [("b", s) for s in order_b if s != "NeuroPress+Tier"])
    single = ([x for x in SINGLE_ORDER if x in single] +
              [x for x in single if x not in SINGLE_ORDER])
    # The selector baselines (HCompress, XGB) belong to the models figure only.
    single = [x for x in single if not x[1].startswith(MODEL_ARM_PREFIXES)]

    mpl.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "hatch.linewidth": 0.7,
                         "hatch.color": "white", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    lossless = figure_rows(D, workloads)
    os.makedirs(args.out, exist_ok=True)
    spare, warn, pngs = {}, [], []
    algo_notes = ([f"{', '.join(lossless)}: lossless arms only"]
                  if lossless else [])
    # (items, title, file stem, draw_key kwargs)
    figures = [(FIG_TIERING, None, "fig9a_tiering", dict(horizontal=True)),
               (FIG_MODELS, None, "fig9b_models",
                dict(solid=True, horizontal=True, ratios=args.model_ratios)),
               (FIG_EXTERNAL, None, "fig9c_external",
                dict(notes=algo_notes, lossless=True)),
               (FIG_LOSSY, None, "fig9d_lossy",
                {})]
    if args.combined:
        figures.append((single, None, "fig9", {"labels_below_floor": True,
                                                "lossless": True}))
    if args.panels:
        figures += [([("a", s) for s in order_a], "(a) Ablation", "fig9a_ablation", {}),
                    ([("b", s) for s in order_b], "(b) External baselines",
                     "fig9b_baselines", {})]
    for items, title, stem, key_kw in figures:
        items = [it for it in items if base_name(it[1]) not in DROPPED_CODECS]
        key_kw = dict(key_kw)
        FD = lossless_rows(D) if key_kw.pop("lossless", False) else D
        RATIOS[0] = key_kw.pop("ratios", False)
        SPLIT[0] = key_kw.pop("split", None)
        # The dagger entry only where some drawn bar is lossy.
        key_kw["show_bound"] = any((FD.get(it + (w,)) or {}).get("eb") not in (None, 0, 0.0)
                                   for it in items for w in workloads)
        HORIZONTAL[0] = key_kw.pop("horizontal", False)
        SOLID[0] = key_kw.pop("solid", True)  # every figure: one solid bar, no split
        pngs += render_figure(args, FD, items, workloads, title, stem, key_kw,
                              spare, warn)
        SOLID[0], SPLIT[0] = False, None
    if warn:
        print(f"WARNING: {len(warn)} bar(s) had a total but no compute/I-O "
              f"split; drawn as one segment: {', '.join(sorted(set(warn)))}\n")
    sanity(D, workloads, order_a)
    reductions(D, workloads, order_a, order_b)
    print("wrote " + ", ".join(pngs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

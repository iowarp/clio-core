#!/usr/bin/env python3
"""Figure 5: where the per-chunk time goes on the write and read paths.

One figure per workload (fig5_<workload>.png): a write-path and a read-path pie,
each wedge a stage's mean share, drawn at its true size. The overhead share
(stats + NN + choice + factory on write, factory on read) is printed with its
spread.

Input rows (per chunk per path; inapplicable fields empty):
  workload,path,chunk_id,chunk_bytes,stats_ms,nn_ms,nn_batch_chunks,
  choice_ms,factory_ms,compress_ms,decompress_ms,io_ms
as written by CLIO_NEUROPRESS_PHASE_LOG (../figure_5.sh), or per-dump tables
from fig5_timesteps.py.

Usage:
  ./plot_fig5.py --timesteps benchmark_vpic_timesteps.csv --no-defaults --out DIR
  ./plot_fig5.py --csv 'logs/*.csv' --err ci95
  ./plot_fig5.py --write-template fig5.csv
"""
import argparse, csv, glob, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# STYLE CONSTANTS -- everything tweakable lives here
# ----------------------------------------------------------------------------
PIE_W, PIE_H = 1.35, 1.45          # inches per pie
FS_AXIS, FS_TICK, FS_LEG, FS_ANN = 8, 7, 7, 6.5   # final print size, nothing < 6.5
FONT_SERIF = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]

WEDGE_EDGE_LW = 0.6                # white line between wedges
PIE_LABEL_MIN_PCT = 7.0            # wedges at least this big carry their % inside
ANN_GRAY, TBD_GRAY = "0.35", "0.55"
LEG_NCOL = 4

COVERAGE_MIN_PCT = 80.0            # warn when the stages explain less of the wall
CLAIM_PCT = 11.0                   # paper: overhead "under 11%"
REBUTTAL_LO, REBUTTAL_HI = 5.0, 10.0   # rebuttal: "averages 5-10%"
BYTES_PER_MB = 1024 ** 2           # "4 MB" chunk = 4 MiB

ROW_TITLE = {"write": "(a) Write path", "read": "(b) Read path"}

# Bar order is FIXED: the original figure's data point first, then workloads in
# paper order. Never sort by value. (CSV key, tick label)
BARS = [
    ("Synthetic", "Synthetic\n(4 MB periodic)"),
    ("VPIC", "VPIC"),
    ("Nyx", "Nyx"),
    ("LAMMPS", "LAMMPS"),
    ("WarpX", "WarpX"),
    ("AI", "AI"),
]

# Stacked bottom-up, legend in the same order. Names and colours are the original
# donut's. The overhead components are the first OVERHEAD_N entries of each
# list: they sit on the common baseline, and the error bar goes on their top.
WRITE = [  # (legend name, CSV column, colour)
    ("Stats Kernel",     "stats_ms",      "#1f77b4"),
    ("NN Inference",     "nn_ms",         "#ff7f0e"),
    ("Compress Choice",  "choice_ms",     "#2ca02c"),
    ("Compress Factory", "factory_ms",    "#d62728"),
    ("Compress Time",    "compress_ms",   "#9467bd"),
    ("I/O Time",         "io_ms",         "#8c564b"),
]
READ = [
    ("Compress Factory", "factory_ms",    "#d62728"),
    ("Decompress Time",  "decompress_ms", "#e377c2"),
    ("I/O Time",         "io_ms",         "#8c564b"),
]
PATHS = {"write": WRITE, "read": READ}
OVERHEAD_N = {"write": 4, "read": 1}

TIME_UNIT = ""                     # "/dump" when shares are per-dump (--timesteps)

FIELDS = ["workload", "path", "chunk_id", "chunk_bytes", "stats_ms", "nn_ms",
          "nn_batch_chunks", "choice_ms", "factory_ms", "compress_ms",
          "decompress_ms", "io_ms"]
# Optional phase-log columns the stages above do not cover.
EXTRA = ["preproc_ms", "h2d_ms", "wall_ms", "other_ms"]

# Default data: the original figure's one synthetic 4 MB chunk (no spread).
# A --csv group (workload, path) replaces it.
_SYN = str(4 * BYTES_PER_MB)
DEFAULT_ROWS = [
    dict(workload="Synthetic", path="write", chunk_id="0", chunk_bytes=_SYN,
         stats_ms="5", nn_ms="12", nn_batch_chunks="1", choice_ms="0.040",
         factory_ms="0.010", compress_ms="110", decompress_ms="", io_ms="30"),
    dict(workload="Synthetic", path="read", chunk_id="0", chunk_bytes=_SYN,
         stats_ms="", nn_ms="", nn_batch_chunks="", choice_ms="",
         factory_ms="0.010", compress_ms="", decompress_ms="80", io_ms="30"),
]


# ----------------------------------------------------------------------------
# LOADING AND AGGREGATION
# ----------------------------------------------------------------------------
def _num(v):
    """Numeric cell -> float, or None when empty/unparseable. Negatives are kept
    so the sanity check can report them."""
    s = "" if v is None else str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "na", "-"):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def load_csvs(patterns):
    rows = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            print(f"WARNING: --csv {pat!r} matches no file")
        for path in matches:
            with open(path, newline="") as fh:
                for i, r in enumerate(csv.DictReader(fh), start=2):
                    r["_src"] = f"{os.path.basename(path)}:{i}"
                    rows.append(r)
    return rows


_WORKLOAD_KEY = {k.lower(): k for k, _ in BARS}
_WORKLOAD_KEY.update({lab.replace("\n", " ").lower(): k for k, lab in BARS})


def per_chunk(rows):
    """rows -> ({(workload, path): {"ms": [[comp...]...], "bytes": [...]}}, problems)

    A chunk with any applicable component missing or negative is excluded from
    the aggregate (its percentages are undefined) and reported."""
    groups, problems, ignored = {}, {}, set()
    for r in rows:
        src = r.get("_src", "default")
        w = _WORKLOAD_KEY.get((r.get("workload") or "").strip().lower())
        p = (r.get("path") or "").strip().lower()
        if w is None or p not in PATHS:
            ignored.add(f"workload={r.get('workload')!r} path={r.get('path')!r}")
            continue
        comps, bad = [], []
        for name, col, _ in PATHS[p]:
            x = _num(r.get(col))
            if x is None:
                bad.append(f"{col} missing")
            elif x < 0:
                bad.append(f"{col}={x:g} negative")
            if name == "NN Inference" and x is not None:
                b = _num(r.get("nn_batch_chunks"))
                if b is not None and b <= 0:
                    bad.append(f"nn_batch_chunks={b:g} not positive")
                elif b is not None:
                    x = x / b          # batch time -> per-chunk cost
            comps.append(x)
        if not bad and sum(comps) <= 0:
            bad.append("all components are zero")
        if bad:
            problems.setdefault((w, p), []).append(
                f"chunk {r.get('chunk_id', '?')} [{src}]: {', '.join(bad)}")
            continue
        g = groups.setdefault((w, p), dict(ms=[], bytes=[], extra=[], reused=0))
        g["ms"].append(comps)
        g["bytes"].append(_num(r.get("chunk_bytes")))
        g["extra"].append([_num(r.get(c)) for c in EXTRA])
        g["reused"] += (str(r.get("reused", "")).strip() == "1")
    if ignored:
        print(f"WARNING: ignored rows with unknown workload/path (known workloads: "
              f"{', '.join(k for k, _ in BARS)}; paths: write, read): "
              f"{'; '.join(sorted(ignored))}")
    return groups, problems


# Two-sided 97.5% Student t quantiles, df = 1..30. Beyond that 1.96 + 2.5/df is
# within 0.002 of the exact value, so no scipy is needed.
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df):
    return _T975[df - 1] if df <= len(_T975) else 1.96 + 2.5 / df


def summarize(groups, err_mode):
    """Percentages are computed per chunk first, then averaged; the error bar is
    the spread of the per-chunk overhead percentage."""
    S = {}
    for (w, p), g in groups.items():
        ms = np.asarray(g["ms"], dtype=float)          # chunks x components
        total = ms.sum(axis=1)
        share = ms / total[:, None] * 100.0
        k = OVERHEAD_N[p]
        ov_pct = share[:, :k].sum(axis=1)
        n = len(total)
        err = None                                     # one chunk: no spread
        if n > 1:
            sd = float(ov_pct.std(ddof=1))
            err = sd if err_mode == "std" else t975(n - 1) * sd / math.sqrt(n)
        # Coverage, only where every chunk carries a wall time: the named
        # components as a share of it, and what they leave out.
        extra = np.asarray([[np.nan if v is None else v for v in e]
                            for e in g["extra"]], dtype=float)
        wall = extra[:, EXTRA.index("wall_ms")]
        cover = None
        if np.all(np.isfinite(wall)) and np.all(wall > 0):
            cover = {"named": float((total / wall).mean() * 100.0)}
            for c in ("preproc_ms", "h2d_ms", "other_ms"):
                col = np.nan_to_num(extra[:, EXTRA.index(c)])
                cover[c] = float((col / wall).mean() * 100.0)
            cover["wall_ms"] = float(wall.mean())
        S[(w, p)] = dict(
            n=n, share=share.mean(axis=0), comp_ms=ms.mean(axis=0),
            total_ms=float(total.mean()), ov_ms=float(ms[:, :k].sum(axis=1).mean()),
            ov_pct=float(ov_pct.mean()), err=err, cover=cover, reused=g["reused"],
            sizes=sorted({int(b) for b in g["bytes"] if b is not None}),
            no_size=sum(b is None for b in g["bytes"]))
    return S


# ----------------------------------------------------------------------------
# FORMATTING
# ----------------------------------------------------------------------------
def fmt_ms(v):
    return f"{v:.0f} ms" if v >= 100 else (f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms")


def _mb(b):
    mb = b / BYTES_PER_MB
    if mb >= 1:
        return f"{mb:.0f}" if abs(mb - round(mb)) < 1e-9 else f"{mb:.1f}"
    return f"{mb:.2g}"


def fmt_size(sizes):
    if not sizes:
        return "? MB"
    if len(sizes) == 1:
        return f"{_mb(sizes[0])} MB"
    return f"{_mb(sizes[0])}-{_mb(sizes[-1])} MB"     # mixed chunk sizes


def fmt_pct(v):
    if v >= 1:
        return f"{v:.1f}%"
    return f"{v:.2f}%" if v >= 0.01 else "<0.01%"


# ----------------------------------------------------------------------------
# SANITY CHECKS AND SUMMARY -- print to stdout, never raise
# ----------------------------------------------------------------------------
def sanity(S, problems):
    print("== sanity checks ==")
    n = 0
    names_w = [c[0] for c in WRITE]
    i_nn, i_cmp, i_io = (names_w.index(x) for x in ("NN Inference", "Compress Time", "I/O Time"))

    for w, _ in BARS:
        s = S.get((w, "write"))
        if s is None:
            continue
        # 1. the paper's "under 11%"
        if s["ov_pct"] > CLAIM_PCT:
            print(f"  [claim]    {w}/write: overhead {s['ov_pct']:.2f}% exceeds the "
                  f"paper's {CLAIM_PCT:g}%"); n += 1
        # 2. the rebuttal's "selection (incl. inference) averages 5-10%"
        if not (REBUTTAL_LO <= s["ov_pct"] <= REBUTTAL_HI):
            print(f"  [rebuttal] {w}/write: overhead {s['ov_pct']:.2f}% is outside the "
                  f"rebuttal's {REBUTTAL_LO:g}-{REBUTTAL_HI:g}%"); n += 1
        # 3. inference must be amortized
        nn = s["comp_ms"][i_nn]
        work = s["comp_ms"][i_cmp] + s["comp_ms"][i_io]
        if nn > work:
            print(f"  [amortize] {w}/write: mean NN inference {nn:.3f} ms/chunk > "
                  f"mean compress+I/O {work:.3f} ms/chunk -- inference is not amortized"); n += 1

    # 4. chunk sizes the caption must state
    per_wl = {}
    for (w, _p), s in S.items():
        per_wl.setdefault(w, set()).update(s["sizes"])
        if s["no_size"]:
            print(f"  [size]     {w}/{_p}: {s['no_size']} chunk(s) have no chunk_bytes"); n += 1
    distinct = {tuple(sorted(v)) for v in per_wl.values()}
    if len(distinct) > 1 or any(len(v) > 1 for v in per_wl.values()):
        print("  [size]     chunk sizes differ -- state them in the caption:"); n += 1
        for w, _ in BARS:
            if w in per_wl:
                print(f"               {w:<10} {fmt_size(sorted(per_wl[w]))}")

    # 6. the drawn components must explain most of the chunk's wall time
    for (w, p), s in sorted(S.items()):
        c = s["cover"]
        if c is not None and c["named"] < COVERAGE_MIN_PCT:
            print(f"  [coverage] {w}/{p}: the drawn components are {c['named']:.1f}% of the "
                  f"runtime's per-chunk wall ({c['wall_ms']:.3f} ms); see the coverage table"); n += 1

    # 5. missing or negative component times
    for (w, p), msgs in sorted(problems.items()):
        print(f"  [data]     {w}/{p}: {len(msgs)} chunk(s) excluded, component "
              f"times missing or negative"); n += 1
        for m in msgs[:5]:
            print(f"               {m}")
        if len(msgs) > 5:
            print(f"               ... and {len(msgs) - 5} more")

    if n == 0:
        print("  all checks passed")
    print()


def summary(S, err_mode):
    elabel = "+-1sd" if err_mode == "std" else "+-CI95"
    for p in ("write", "read"):
        comps = PATHS[p]
        heads = [c[0].replace("Compress ", "C.").replace(" Time", "").replace(" Kernel", "")
                 .replace(" Inference", "") for c in comps]
        print(f"== summary: {p} path (component shares are means of per-chunk %) ==")
        hdr = (f"{'workload':<10}{'chunk':>9}{'n':>7}{'total ms':>11}{'ovh ms':>10}"
               f"{'ovh %':>8}{elabel:>8}" + "".join(f"{h:>11}" for h in heads))
        print(hdr)
        print("-" * len(hdr))
        for w, _ in BARS:
            s = S.get((w, p))
            if s is None:
                print(f"{w:<10}{'TBD':>9}")
                continue
            err = "--" if s["err"] is None else f"{s['err']:.2f}"
            print(f"{w:<10}{fmt_size(s['sizes']):>9}{s['n']:>7}{s['total_ms']:>11.3f}"
                  f"{s['ov_ms']:>10.3f}{s['ov_pct']:>8.2f}{err:>8}"
                  + "".join(f"{v:>10.3f}%" for v in s["share"]))
        print()

    rows = [(w, p, S[(w, p)]) for w, _ in BARS for p in ("write", "read")
            if (w, p) in S and S[(w, p)]["cover"] is not None]
    if rows:
        print("== coverage: what the bars leave out, as % of the runtime's per-chunk wall ==")
        print(f"{'workload':<10}{'path':>6}{'wall ms':>10}{'named %':>9}{'preproc %':>11}"
              f"{'h2d %':>8}{'other %':>9}{'reused':>8}")
        for w, p, s in rows:
            c = s["cover"]
            print(f"{w:<10}{p:>6}{c['wall_ms']:>10.3f}{c['named']:>9.2f}{c['preproc_ms']:>11.2f}"
                  f"{c['h2d_ms']:>8.2f}{c['other_ms']:>9.2f}"
                  f"{(s['reused'] if p == 'write' else ''):>8}")
        print("  named = the drawn components; preproc = quantize + byte shuffle (upstream's")
        print("  figure leaves it out too); h2d = staging a host chunk up (replay route only);")
        print("  other = the rest of the runtime's wall: allocation, IPC, headers, logging,")
        print("  and waiting behind chunks in flight at the same time (the replay submits a")
        print("  file's chunks together), so a low named % is not by itself missing work.")
        print()


# ----------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------
_SHORT = {"Stats Kernel": "Stats", "NN Inference": "NN", "Compress Choice": "Choice",
          "Compress Factory": "Factory", "Compress Time": "Compress",
          "Decompress Time": "Decompress", "I/O Time": "I/O"}


def _text_color(hexcolor):
    r, g, b = mpl.colors.to_rgb(hexcolor)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else "black"


def draw_pie(ax, s, path):
    """One pie: every wedge at its true share. Wedges too thin to label are
    listed beneath with their exact share, so none of them is lost."""
    ax.set_aspect("equal")
    ax.axis("off")
    if s is None:
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", fontsize=FS_AXIS,
                color=TBD_GRAY, transform=ax.transAxes)
        return
    comps = PATHS[path]
    shares = [max(float(v), 0.0) for v in s["share"]]
    wedges, _ = ax.pie(shares, colors=[c for _, _, c in comps], startangle=90,
                       counterclock=False, radius=1.0,
                       wedgeprops=dict(edgecolor="white", linewidth=WEDGE_EDGE_LW))
    small = []
    for wdg, (name, _col, color), share in zip(wedges, comps, shares):
        if share >= PIE_LABEL_MIN_PCT:
            ang = np.deg2rad((wdg.theta1 + wdg.theta2) / 2.0)
            # One decimal where rounding would claim the whole pie (99.5 -> "100%").
            txt = f"{share:.1f}%" if share >= 99.5 else f"{share:.0f}%"
            ax.text(0.62 * np.cos(ang), 0.62 * np.sin(ang), txt,
                    ha="center", va="center", fontsize=FS_ANN, fontweight="bold",
                    color=_text_color(color))
        else:
            small.append(f"{_SHORT[name]} {fmt_pct(share)}")
    err = f" ± {s['err']:.2f}" if s["err"] is not None else ""
    lines = [f"overhead {fmt_pct(s['ov_pct'])}{err}"]
    # Two short entries per line keeps the list inside the pie's column.
    lines += ["  ".join(small[k:k + 2]) for k in range(0, len(small), 2)]
    ax.text(0.5, -0.04, "\n".join(lines), ha="center", va="top", fontsize=FS_ANN,
            color=ANN_GRAY, transform=ax.transAxes, linespacing=1.15)


def legend_handles():
    """Write-path stages, then the read-path ones not already listed."""
    seen, out = set(), []
    for name, _col, color in WRITE + READ:
        if name not in seen:
            seen.add(name)
            out.append(Patch(facecolor=color, edgecolor="none", label=name))
    return out


def set_fonts():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    serif = [f for f in FONT_SERIF if f in avail] or ["DejaVu Serif"]
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": serif, "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    })
    return serif[0]


SAVE_PAD = 0.02                    # pad_inches for the tight bbox


def save(fig, stem, out):
    png = os.path.join(out, stem + ".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD)
    plt.close(fig)
    return [png]


def timesteps_rows(patterns):
    """Per-dump tables -> one write and one read row per dump."""
    rows = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            print(f"WARNING: --timesteps {pat!r} matches no file")
        for path in matches:
            with open(path, newline="") as fh:
                for i, r in enumerate(csv.DictReader(fh), start=2):
                    src = f"{os.path.basename(path)}:{i}"
                    nw = _num(r.get("chunks_write")) or 0
                    chunk_b = (_num(r.get("orig_mib")) or 0) * BYTES_PER_MB / nw if nw else ""
                    base = dict(workload=r["workload"], chunk_id=r["dump"],
                                chunk_bytes=chunk_b, _src=src)
                    rows.append(dict(base, path="write", nn_batch_chunks="1",
                                     **{c: r[c] for c in ("stats_ms", "nn_ms", "choice_ms",
                                                          "factory_ms", "compress_ms", "io_ms")},
                                     preproc_ms=r.get("preproc_ms"), h2d_ms=r.get("h2d_ms"),
                                     wall_ms=r.get("write_wall_ms"),
                                     other_ms=r.get("write_other_ms")))
                    if (_num(r.get("chunks_read")) or 0) > 0:
                        rows.append(dict(base, path="read",
                                         factory_ms=r["read_factory_ms"],
                                         decompress_ms=r["decompress_ms"],
                                         io_ms=r["read_io_ms"], wall_ms=r.get("read_wall_ms"),
                                         other_ms=r.get("read_other_ms")))
    return rows


def plot_workload(S, w, out):
    """One workload's breakdown: a write-path pie beside a read-path pie."""
    label = dict(BARS)[w].replace("\n", " ")
    fig, axes = plt.subplots(1, 2, figsize=(2 * PIE_W + 0.6, PIE_H + 0.75))
    for ax, path in zip(axes, ("write", "read")):
        s = S.get((w, path))
        sub = f"\n{fmt_ms(s['total_ms'])}{TIME_UNIT}, n={s['n']}" if s is not None else ""
        ax.set_title(f"{ROW_TITLE[path]}{sub}", fontsize=FS_AXIS, pad=3, linespacing=1.1)
        draw_pie(ax, s, path)
    fig.suptitle(label, fontsize=FS_AXIS + 1, y=1.06)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.18, top=0.66, wspace=0.35)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.80),
               ncol=LEG_NCOL, fontsize=FS_LEG, frameon=False, handlelength=1.1,
               handleheight=0.8, handletextpad=0.4, columnspacing=0.9, labelspacing=0.25)
    return save(fig, f"fig5_{w.lower()}", out)


def main():
    global TIME_UNIT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="extend", nargs="+", default=[], metavar="PATH",
                    help="per-chunk timing log(s); repeatable, globs allowed")
    ap.add_argument("--timesteps", action="extend", nargs="+", default=[], metavar="PATH",
                    help="per-dump tables from fig5_timesteps.py: shares are taken per "
                         "dump and averaged across dumps")
    ap.add_argument("--no-defaults", action="store_true",
                    help="drop the embedded synthetic data point, for a measured-only figure")
    ap.add_argument("--write-template", metavar="PATH",
                    help="write an example log (the embedded defaults) and exit")
    ap.add_argument("--out", default="figures", help="output directory (default: figures)")
    ap.add_argument("--err", choices=("std", "ci95"), default="std",
                    help="spread printed with the overhead share: +-1 standard deviation (default) or 95%% CI of the mean")
    args = ap.parse_args()

    if args.write_template:
        d = os.path.dirname(os.path.abspath(args.write_template))
        os.makedirs(d, exist_ok=True)
        with open(args.write_template, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=FIELDS)
            wtr.writeheader()
            wtr.writerows(DEFAULT_ROWS)
        print(f"wrote template: {args.write_template}")
        return 0

    groups, problems = ({}, {}) if args.no_defaults else per_chunk(DEFAULT_ROWS)
    if args.timesteps:
        TIME_UNIT = "/dump"
    if args.csv or args.timesteps:
        g_csv, p_csv = per_chunk(load_csvs(args.csv) + timesteps_rows(args.timesteps))
        for key in set(g_csv) | set(p_csv):
            if key in groups:
                print(f"note: {key[0]}/{key[1]} taken from --csv, replacing the embedded default")
            groups.pop(key, None)
            problems.pop(key, None)
        groups.update(g_csv)
        problems.update(p_csv)

    S = summarize(groups, args.err)
    font = set_fonts()
    os.makedirs(args.out, exist_ok=True)
    written = []
    for w, _ in BARS:
        if (w, "write") in S or (w, "read") in S:
            written += plot_workload(S, w, args.out)

    try:
        sanity(S, problems)
        summary(S, args.err)
    except Exception as e:          # the checks must never cost the figure
        print(f"WARNING: sanity/summary failed: {e!r}\n")
    print(f"font: {font}   spread: {'+-1 sd' if args.err == 'std' else '95% CI'}")
    for f in written:
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

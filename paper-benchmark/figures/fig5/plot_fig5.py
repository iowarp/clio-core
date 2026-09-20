#!/usr/bin/env python3
"""Figure 5: where the per-chunk time goes on the write and read paths.

One figure per workload (fig5_<workload>.png): a write-path and a read-path pie,
each wedge a stage's mean share, drawn at its true size. The overhead share
(stats + NN + choice + factory on write, factory on read) is printed with its
spread.

Input rows (per chunk per path; inapplicable fields empty):
  workload,path,chunk_id,chunk_bytes,stats_ms,nn_ms,nn_batch_chunks,
  choice_ms,factory_ms,compress_ms,decompress_ms,io_ms
as written by CLIO_NEUROPRESS_PHASE_LOG (figure_5.sh, beside this file), or per-dump tables
from fig5_timesteps.py.

THIS DIRECTORY KEEPS ONE PLOTTER, so figure 5's four jobs are four subcommands
of this file rather than four scripts that drift apart. Each one's flags
belong to it alone -- `plot_fig5.py <mode> --help`:

  ./plot_fig5.py stacked [--split]        the paper's plate, from the phase logs
  ./plot_fig5.py anatomy                  the two anatomy plates
  ./plot_fig5.py table PHASE.CSV --workload Nyx --out ts.csv
  ./plot_fig5.py pies --timesteps ts.csv --no-defaults --out DIR

`stacked` and `anatomy` draw into this script's own directory unless --out says
otherwise. The stage colours and the print sizes are shared by all three
renderings, which is the point of having them in one file.
"""
import argparse, collections, csv, glob, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))   # figures/fig5

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


# ============================================================================
# THE PER-DUMP TABLE  (--phase): the runtime's per-chunk phase log, summed per
# dump, which is what --timesteps reads back. It was fig5_timesteps.py until
# this directory went to one plotter.
# ============================================================================
DUMP_RE = re.compile(r"^(?:step_?|plt|diag)(\d+)$")
TS_WRITE = ["stats_ms", "nn_ms", "choice_ms", "factory_ms", "compress_ms", "io_ms"]
TS_READ = {"factory_ms": "read_factory_ms", "decompress_ms": "decompress_ms",
           "io_ms": "read_io_ms"}
TS_COLUMNS = (["workload", "timestep", "dump", "chunks_write", "chunks_read",
               "orig_mib", "stored_mib", "ratio"] + TS_WRITE + list(TS_READ.values()) +
              ["preproc_ms", "h2d_ms", "write_other_ms", "read_other_ms",
               "write_wall_ms", "read_wall_ms", "write_mib_s", "read_mib_s",
               "explorations", "explored_alternatives", "sgd_updates",
               "reused_chunks", "explore_ms", "sgd_ms"])


def ts_num(v):
    """@return a phase-log cell as a float, or None when it is empty or junk."""
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def dump_of(chunk_id):
    """The dump a chunk belongs to, from its blob name.

    @param chunk_id the phase log's chunk_id, e.g. plt00010/field/chunk_3
    @return (name, timestep), or (None, None) when no segment names a dump
    """
    for seg in chunk_id.split("/"):
        m = DUMP_RE.match(seg)
        if m:
            return seg, int(m.group(1))
    return None, None


def phase_dumps(phase_log, warmup_step):
    """Sum a phase log per dump.

    @param phase_log CLIO_NEUROPRESS_PHASE_LOG, one row per chunk per path
    @param warmup_step drop dumps at step <= this
    @return ({(step, name): sums}, excluded counts as (warm, missing, unnamed))
    """
    dumps = collections.OrderedDict()
    unnamed = missing = warm = 0
    with open(phase_log, newline="") as fh:
        for r in csv.DictReader(fh):
            name, step = dump_of(r["chunk_id"])
            if name is None:
                unnamed += 1
                continue
            if step <= warmup_step:
                warm += 1
                continue
            d = dumps.setdefault((step, name), collections.defaultdict(float))
            path = r["path"]
            cols = TS_WRITE if path == "write" else list(TS_READ)
            if any(ts_num(r.get(c)) is None for c in cols):
                missing += 1
                continue
            if path == "write":
                d["chunks_write"] += 1
                for c in TS_WRITE:
                    d[c] += ts_num(r[c])
                d["orig_b"] += ts_num(r["chunk_bytes"]) or 0
                d["stored_b"] += ts_num(r.get("stored_bytes")) or 0
                d["write_wall_ms"] += ts_num(r["wall_ms"]) or 0
                d["write_other_ms"] += ts_num(r["other_ms"]) or 0
                d["preproc_ms"] += ts_num(r.get("preproc_ms")) or 0
                d["h2d_ms"] += ts_num(r.get("h2d_ms")) or 0
                d["explore_ms"] += ts_num(r.get("explore_ms")) or 0
                d["sgd_ms"] += ts_num(r.get("sgd_ms")) or 0
                n_alt = ts_num(r.get("explored")) or 0
                d["explored_alternatives"] += n_alt
                d["explorations"] += 1 if n_alt > 0 else 0
                d["sgd_updates"] += ts_num(r.get("sgd_updates")) or 0
                d["reused_chunks"] += 1 if r.get("reused") == "1" else 0
            elif path == "read":
                d["chunks_read"] += 1
                for c, out in TS_READ.items():
                    d[out] += ts_num(r[c])
                d["read_b"] += ts_num(r["chunk_bytes"]) or 0
                d["read_wall_ms"] += ts_num(r["wall_ms"]) or 0
                d["read_other_ms"] += ts_num(r["other_ms"]) or 0
    return dumps, (warm, missing, unnamed)


def write_timesteps(phase_log, workload, out, warmup_step=-1):
    """Turn a phase log into the per-dump table --timesteps reads.

    Throughput is bytes over the summed chunk wall time, a lower bound.

    @param phase_log the runtime's per-chunk phase log
    @param workload the name to stamp on every row
    @param out the csv to write
    @param warmup_step drop dumps at step <= this (-1 keeps all)
    @return 0 when at least one dump survived, 1 otherwise
    """
    dumps, (warm, missing, unnamed) = phase_dumps(phase_log, warmup_step)
    rows = []
    for ts, ((step, name), d) in enumerate(sorted(dumps.items())):
        mib = d["orig_b"] / 2**20
        rows.append({
            "workload": workload, "timestep": ts, "dump": name,
            "chunks_write": int(d["chunks_write"]), "chunks_read": int(d["chunks_read"]),
            "orig_mib": f"{mib:.3f}", "stored_mib": f"{d['stored_b'] / 2**20:.3f}",
            "ratio": f"{d['orig_b'] / d['stored_b']:.3f}" if d["stored_b"] else "",
            **{c: f"{d[c]:.4f}" for c in TS_WRITE + list(TS_READ.values())},
            **{c: f"{d[c]:.4f}" for c in ("preproc_ms", "h2d_ms", "write_other_ms",
                                          "read_other_ms", "write_wall_ms",
                                          "read_wall_ms", "explore_ms", "sgd_ms")},
            "write_mib_s": f"{mib / (d['write_wall_ms'] / 1e3):.2f}" if d["write_wall_ms"] else "",
            "read_mib_s": (f"{d['read_b'] / 2**20 / (d['read_wall_ms'] / 1e3):.2f}"
                           if d["read_wall_ms"] else ""),
            **{c: int(d[c]) for c in ("explorations", "explored_alternatives",
                                      "sgd_updates", "reused_chunks")},
        })
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TS_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{workload}: {len(rows)} dump(s) -> {out}"
          f"   (excluded: {warm} warmup chunk row(s), {missing} with a missing stage,"
          f" {unnamed} without a dump in the name)")
    return 0 if rows else 1


# ============================================================================
# THE STACKED PLATE  (--stacked): the same measurement in the EBS FAST'24
# Fig.11 idiom, which is what the paper carries. The write panel has a BROKEN
# x-axis: the left sub-axis magnifies 0..ST_BREAK_MS so the selector stages are
# visible (0.004-0.5% of the bar, otherwise sub-pixel); the right runs to the
# full per-chunk time. Say so in the caption -- SIGPLAN asks that zooming "be
# pointed out explicitly". Bars are means over every chunk of that path;
# Choice includes the codec factory.
# ============================================================================
ST_WLS = [("vpic", "VPIC"), ("nyx", "Nyx"), ("lammps", "LAMMPS"), ("warpx", "WarpX")]
ST_SEL = [("Stats", "stats_ms"), ("NN", "nn_ms"), ("Choice", "__choice__")]
ST_WRITE = ST_SEL + [("Compress", "compress_ms"), ("I/O", "io_ms")]
ST_READ = [("Decompress", "decompress_ms"), ("I/O", "io_ms")]
ST_COL = {"Stats": "#2f5597", "NN": "#7c9fd4", "Choice": "#c6d4ec",
          "Compress": "#ffc000", "I/O": "#ededed", "Decompress": "#9ecae1"}
ST_HAT = {"Stats": "....", "NN": "////", "Choice": "xxxx", "Compress": "....",
          "I/O": "", "Decompress": "////"}
ST_BREAK_MS = 0.30
COL_W, TEXT_W = 3.334, 7.0          # acmart sigplan \columnwidth / \textwidth
#: The rc the two bar renderings print at. Applied per draw, not at import, so
#: the pies above keep their own font handling.
BAR_RC = {"font.family": "serif",
          "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
          "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6,
          "hatch.linewidth": 0.45, "font.size": 7}


def default_results():
    """@return where a figure_5.sh run leaves its per-workload phase logs."""
    return os.environ.get("FIG5_RESULTS",
                          os.path.join(HERE, "..", "..", "results", "figure5"))


def st_per_chunk(res, wl, path, stages):
    """Mean milliseconds per stage over one workload's chunks of one path.

    @param res the results directory holding figure_5_<wl>/phase.csv
    @param wl the workload key, e.g. vpic
    @param path write or read
    @param stages [(legend name, column)], __choice__ meaning choice+factory
    @return ({stage: ms}, rows seen)
    """
    rows = [r for r in csv.DictReader(open(f"{res}/figure_5_{wl}/phase.csv"))
            if r["path"] == path]

    def col(c):
        if c == "__choice__":
            return col("choice_ms") + col("factory_ms")
        return float(np.mean([float(r[c]) for r in rows
                              if r.get(c) not in (None, "")] or [0.0]))
    return {n: col(c) for n, c in stages}, len(rows)


def _st_stack(ax, y, d, stages, lw):
    """One workload's bar, stacked stage by stage. @return its total ms."""
    left = 0.0
    for name, _ in stages:
        v = d[name]
        ax.barh(y, v, left=left, height=0.62, color=ST_COL[name], hatch=ST_HAT[name],
                edgecolor="black", linewidth=lw, zorder=3)
        left += v
    return left


def _st_frame(ax, ys, labels=True):
    """Ticks, grid and spines shared by every stacked sub-axis."""
    ax.set_yticks(ys)
    ax.set_yticklabels([l for _, l in ST_WLS] if labels else [], fontsize=FS_TICK)
    ax.set_ylim(-0.6, len(ST_WLS) - 0.4)
    ax.grid(axis="x", color="#cccccc", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


def st_draw_write(res, axL, axR):
    """The write panel, across the axis break.

    @param res the results directory
    @param axL the magnified 0..ST_BREAK_MS sub-axis
    @param axR the full-range sub-axis
    """
    ys = np.arange(len(ST_WLS))[::-1]
    wmax = 0.0
    for y, (wl, _l) in zip(ys, ST_WLS):
        d, _n = st_per_chunk(res, wl, "write", ST_WRITE)
        for ax, lw in ((axL, 0.3), (axR, 0.5)):
            tot = _st_stack(ax, y, d, ST_WRITE, lw)
        wmax = max(wmax, tot)
        axR.text(tot * 1.02, y, f"{tot:.1f} ms", va="center", ha="left",
                 fontsize=FS_ANN, zorder=4)
    axL.set_xlim(0, ST_BREAK_MS)
    axR.set_xlim(ST_BREAK_MS, wmax * 1.26)
    _st_frame(axL, ys, True)
    _st_frame(axR, ys, False)
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)
    axL.set_xticks([0, 0.1, 0.2])
    axR.set_xticks([5, 10, 15, 20, 25])
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=4, linestyle="none",
              color="black", mec="black", mew=0.6, clip_on=False)
    axL.plot([1, 1], [0, 1], transform=axL.transAxes, **kw)
    axR.plot([0, 0], [0, 1], transform=axR.transAxes, **kw)


def st_draw_read(res, ax):
    """The read panel: decompress and I/O, no break needed."""
    ys = np.arange(len(ST_WLS))[::-1]
    rmax = 0.0
    for y, (wl, _l) in zip(ys, ST_WLS):
        d, _n = st_per_chunk(res, wl, "read", ST_READ)
        tot = _st_stack(ax, y, d, ST_READ, 0.5)
        rmax = max(rmax, tot)
        ax.text(tot * 1.02, y, f"{tot:.1f} ms", va="center", ha="left",
                fontsize=FS_ANN, zorder=4)
    ax.set_xlim(0, rmax * 1.30)
    _st_frame(ax, ys, True)


def bar_legend(fig, names, y=1.0, ncol=None, columnspacing=1.0):
    """The stage key both bar renderings carry, in stage order."""
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=ST_COL.get(name, AN_COL.get(name)),
                       hatch=ST_HAT.get(name, AN_HAT.get(name)), edgecolor="black",
                       linewidth=0.5) for name in names]
    fig.legend(h, names, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(names), fontsize=FS_LEG, frameon=False,
               handlelength=1.3, handleheight=0.85, columnspacing=columnspacing,
               handletextpad=0.4)


def emit(fig, out, name):
    """Write one bar figure and close it. @return its path."""
    path = os.path.join(out, name)
    os.makedirs(out, exist_ok=True)
    fig.savefig(path, dpi=300)
    print("wrote", path)
    plt.close(fig)
    return path


def draw_stacked(res, out, split=False, name="fig5_stacked.png"):
    """The stacked plate: both panels in one \\textwidth figure, or one each.

    @param res the results directory holding figure_5_<wl>/phase.csv
    @param out the directory to write into
    @param split True for fig5_write.png and fig5_read.png at \\columnwidth
    @param name the combined figure's file name, when not splitting
    @return 0
    """
    with plt.rc_context(BAR_RC):
        if split:
            fw = plt.figure(figsize=(COL_W, 1.60))
            gs = fw.add_gridspec(1, 2, width_ratios=[1.0, 1.9], wspace=0.06)
            st_draw_write(res, fw.add_subplot(gs[0]), fw.add_subplot(gs[1]))
            bar_legend(fw, ["Stats", "NN", "Choice", "Compress", "I/O"], y=1.02, ncol=5)
            fw.text(0.58, 0.035, "Per-chunk time (ms); axis break at 0.3 ms",
                    ha="center", fontsize=FS_AXIS)
            fw.subplots_adjust(top=0.80, bottom=0.30, left=0.17, right=0.975)
            emit(fw, out, "fig5_write.png")

            fr = plt.figure(figsize=(COL_W, 1.60))
            st_draw_read(res, fr.add_subplot(111))
            bar_legend(fr, ["Decompress", "I/O"], y=1.02, ncol=2)
            fr.text(0.60, 0.035, "Per-chunk time (ms)", ha="center", fontsize=FS_AXIS)
            fr.subplots_adjust(top=0.80, bottom=0.30, left=0.20, right=0.975)
            emit(fr, out, "fig5_read.png")
            return 0
        fig = plt.figure(figsize=(TEXT_W, 2.35))
        outer = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.7], wspace=0.30)
        inner = outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.05)
        axL, axR = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        axRd = fig.add_subplot(outer[0, 1])
        st_draw_write(res, axL, axR)
        st_draw_read(res, axRd)
        axRd.set_xlabel("Per-chunk time (ms)", fontsize=FS_AXIS, labelpad=2)
        axRd.set_title("(b) Read path", fontsize=FS_AXIS, pad=4)
        bar_legend(fig, ["Stats", "NN", "Choice", "Compress", "I/O", "Decompress"], ncol=6)
        fig.text((axL.get_position().x0 + axR.get_position().x1) / 2, 0.055,
                 "Per-chunk time (ms)   [left axis magnified; note the break]",
                 ha="center", fontsize=FS_AXIS)
        fig.text((axL.get_position().x0 + axR.get_position().x1) / 2, 0.775,
                 "(a) Write path", ha="center", fontsize=FS_AXIS)
        fig.subplots_adjust(top=0.72, bottom=0.235, left=0.075, right=0.985)
        emit(fig, out, os.path.basename(os.environ.get("FIG5_OUT", name)))
    return 0


# ============================================================================
# THE ANATOMY PLATES  (--anatomy): the donuts anatomy_write.pdf and
# anatomy_read.pdf redrawn in the same stacked idiom, one \columnwidth figure
# each. The data is DEFAULT_ROWS above -- the synthetic reference point -- and
# the percentages reproduce the donuts exactly:
#   write  3.18 / 7.64 / 0.03 / 0.01 / 70.0 / 19.1   (157.05 ms)
#   read                      0.01 / 72.7 / 27.3     (110.01 ms)
# One continuous bar per path; the stages too small to draw are labelled
# beside it, as the donuts did.
# ============================================================================
AN_COL = dict(ST_COL, Factory="#e8467c")
AN_HAT = dict(ST_HAT, Factory="")
AN_WRITE = [("Stats", 5.0), ("NN", 12.0), ("Choice", 0.040), ("Factory", 0.010),
            ("Compress", 110.0), ("I/O", 30.0)]
AN_READ = [("Factory", 0.010), ("Decompress", 80.0), ("I/O", 30.0)]


def an_bar(ax, stages, total, lw, lo, hi, pct_min):
    """One path as a single continuous bar. @return the bar's total."""
    left = 0.0
    for name, v in stages:
        ax.barh(0, v, left=left, height=0.5, color=AN_COL[name], hatch=AN_HAT[name],
                edgecolor="black", linewidth=lw, zorder=3)
        if lo <= left < hi and v / (hi - lo) > pct_min:
            ax.text(left + v / 2, 0, f"{v:g} ms ({v / total * 100:.1f}%)",
                    ha="center", va="center", fontsize=FS_ANN - 1, color="black",
                    zorder=5, bbox=dict(facecolor="white", alpha=0.82,
                                        edgecolor="none", boxstyle="round,pad=0.18"))
        left += v
    return left


def an_frame(ax, lo, hi, ticks, ytop=1.0):
    """The anatomy bar's axes: no y, a light x grid, left spine drawn by hand.

    @param ytop how much room above the bar the annotations need
    """
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.45, ytop)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.grid(axis="x", color="#cccccc", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


def draw_anatomy_write(out):
    """The write path: the bar, the overhead bracket, the four narrow stages."""
    total = sum(v for _, v in AN_WRITE)
    fw = plt.figure(figsize=(COL_W, 1.95))
    ax = fw.add_subplot(111)
    an_bar(ax, AN_WRITE, total, 0.5, 0, total * 1.12, 0.15)
    ax.text(total * 1.015, 0, f"{total:.1f} ms", va="center", ha="left", fontsize=FS_ANN)
    an_frame(ax, 0, total * 1.24, [0, 25, 50, 75, 100, 125, 150], ytop=1.55)
    ax.spines["left"].set_visible(True)
    # No leaders: a bracket ties the narrow stages to the region they occupy,
    # and their values sit above it in two colour-keyed columns.
    ovh = sum(v for n, v in AN_WRITE if n in ("Stats", "NN", "Choice", "Factory"))
    ax.annotate("", xy=(0, 0.34), xytext=(ovh, 0.34),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#555555",
                                connectionstyle="bar,fraction=0.28"), zorder=6)
    ax.text(ovh + 4, 0.56,
            f"NeuroPress overhead  {ovh:.2f} ms ({ovh / total * 100:.1f}%)",
            ha="left", va="center", fontsize=FS_ANN - 1, color="#333333", zorder=6)
    ink = {"Stats": AN_COL["Stats"], "NN": "#4a6fa5", "Choice": "#33517f",
           "Factory": AN_COL["Factory"]}
    for name, cx, cy in (("Stats", 34, 1.30), ("NN", 34, 1.03),
                         ("Choice", 92, 1.30), ("Factory", 92, 1.03)):
        v = dict(AN_WRITE)[name]
        lab = (f"{name} {v:g} ms ({v / total * 100:.2f}%)" if v >= 1
               else f"{name} {v * 1000:.0f} us ({v / total * 100:.2f}%)")
        ax.add_patch(plt.Rectangle((cx - 4.5, cy - 0.075), 3.2, 0.15,
                                   facecolor=AN_COL[name], edgecolor="black",
                                   linewidth=0.4, zorder=6))
        ax.text(cx, cy, lab, ha="left", va="center", fontsize=FS_ANN - 1.5,
                color=ink[name], zorder=6)
    bar_legend(fw, [n for n, _ in AN_WRITE], y=1.03, ncol=6, columnspacing=0.9)
    fw.text(0.55, 0.04, "Write-path time (ms)", ha="center", fontsize=FS_AXIS)
    fw.subplots_adjust(top=0.84, bottom=0.26, left=0.035, right=0.975)
    os.makedirs(out, exist_ok=True)
    fw.savefig(os.path.join(out, "anatomy_write_stacked.png"), dpi=300)
    plt.close(fw)


def draw_anatomy_read(out):
    """The read path: decompress, I/O, and the factory called out on a leader."""
    total = sum(v for _, v in AN_READ)
    fr = plt.figure(figsize=(COL_W, 1.55))
    ax = fr.add_subplot(111)
    an_bar(ax, AN_READ, total, 0.5, 0, total * 1.12, 0.12)
    ax.text(total * 1.015, 0, f"{total:.1f} ms", va="center", ha="left", fontsize=FS_ANN)
    an_frame(ax, 0, total * 1.14, [0, 25, 50, 75, 100])
    ax.spines["left"].set_visible(True)
    ax.annotate("Factory 10 us (0.01%)", xy=(0.01, 0.26), xytext=(14, 0.66),
                ha="left", va="center", fontsize=FS_ANN - 1.5, color=AN_COL["Factory"],
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#888888",
                                shrinkA=0, shrinkB=1), zorder=6)
    bar_legend(fr, [n for n, _ in AN_READ], y=1.03, ncol=3, columnspacing=0.9)
    fr.text(0.55, 0.05, "Read-path time (ms)", ha="center", fontsize=FS_AXIS)
    fr.subplots_adjust(top=0.72, bottom=0.34, left=0.035, right=0.975)
    os.makedirs(out, exist_ok=True)
    fr.savefig(os.path.join(out, "anatomy_read_stacked.png"), dpi=300)
    plt.close(fr)


def draw_anatomy(out):
    """Both anatomy plates, in the rc the bar renderings print at. @return 0"""
    with plt.rc_context(BAR_RC):
        draw_anatomy_write(out)
        draw_anatomy_read(out)
    print("wrote anatomy_write_stacked.png and anatomy_read_stacked.png")
    return 0


def draw_pies(args):
    """One write/read pie pair per workload, with the sanity checks after it.

    @param args the parsed `pies` arguments
    @return 0
    """
    global TIME_UNIT
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


def main():
    """Four jobs, one per subcommand, so each one's flags stay its own."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_subparsers(dest="mode", required=True, metavar="MODE")

    m = mode.add_parser("stacked", help="THE PAPER'S FIGURE: every workload's "
                                        "write and read bar, from the phase logs")
    m.add_argument("--results", default="", metavar="DIR",
                   help="a figure_5.sh results tree [$FIG5_RESULTS, else "
                        "../../results/figure5]")
    m.add_argument("--out", default=HERE, metavar="DIR",
                   help="where to write [this script's own directory]")
    m.add_argument("--split", action="store_true",
                   help="two \\columnwidth figures (fig5_write.png, "
                        "fig5_read.png) instead of one \\textwidth plate")
    m.add_argument("--name", default="fig5_stacked.png", metavar="FILE",
                   help="the combined plate's file name")

    m = mode.add_parser("anatomy", help="the two anatomy plates, from the "
                                        "embedded synthetic point")
    m.add_argument("--out", default=HERE, metavar="DIR",
                   help="where to write [this script's own directory]")

    m = mode.add_parser("table", help="a run's phase log summed per dump, which "
                                      "is what `pies --timesteps` reads")
    m.add_argument("phase", metavar="PHASE.CSV",
                   help="the runtime's per-chunk phase log")
    m.add_argument("--workload", required=True, metavar="NAME",
                   help="the workload name stamped on every row")
    m.add_argument("--out", required=True, metavar="CSV",
                   help="the per-dump table to write")
    m.add_argument("--warmup-step", type=int, default=-1, metavar="N",
                   help="drop dumps at step <= N")

    m = mode.add_parser("pies", help="one write/read pie pair per workload, "
                                     "each wedge a stage's mean share")
    m.add_argument("--csv", action="extend", nargs="+", default=[], metavar="PATH",
                   help="per-chunk timing log(s); repeatable, globs allowed")
    m.add_argument("--timesteps", action="extend", nargs="+", default=[], metavar="PATH",
                   help="per-dump tables from `table`: shares are taken per "
                        "dump and averaged across dumps")
    m.add_argument("--no-defaults", action="store_true",
                   help="drop the embedded synthetic data point, for a "
                        "measured-only figure")
    m.add_argument("--write-template", metavar="PATH",
                   help="write an example log (the embedded defaults) and exit")
    m.add_argument("--out", default="figures", metavar="DIR",
                   help="where to write [figures/]")
    m.add_argument("--err", choices=("std", "ci95"), default="std",
                   help="spread printed with the overhead share: +-1 standard "
                        "deviation (default) or 95%% CI of the mean")

    args = ap.parse_args()
    if args.mode == "stacked":
        return draw_stacked(args.results or default_results(), args.out,
                            args.split, args.name)
    if args.mode == "anatomy":
        return draw_anatomy(args.out)
    if args.mode == "table":
        return write_timesteps(args.phase, args.workload, args.out, args.warmup_step)
    return draw_pies(args)


if __name__ == "__main__":
    sys.exit(main())

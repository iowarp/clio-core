#!/usr/bin/env python3
"""Figure 8: regret and cost MAPE per chunk, one line per workload.

  (a) Regret (%)     (cost_selected - cost_optimal) / cost_optimal x 100
  (b) Cost MAPE (%)  |cost_predicted - cost_measured| / cost_measured x 100

Both use the trace's real_cost / predicted_cost (see fig8_trace.py);
--mape-raw and --mape-floor rescore the MAPE. Lines are per-bin means.

Inputs: --trace PATH:WORKLOAD (fig8_trace.py output) and/or --chunks PATH
(workload,chunk_index,regret_pct,cost_mape_pct).
Outputs in --out: fig8a_regret.png, fig8b_cost_mape.png, fig8_preview.png,
fig8c_per_chunk.png. --chunks-out DIR also writes DIR/<workload>/chunks.csv
(every chunk's metrics) and DIR/summary.txt.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
try:   # the floor the traces were written with
    from fig8_trace import MIN_TIME_MS as _TF
    TRACE_FLOOR = f"{_TF:g} ms"
except Exception:
    TRACE_FLOOR = "trace"
from matplotlib import font_manager
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# STYLE CONSTANTS -- everything tweakable lives here
# ----------------------------------------------------------------------------
FIG_W = 3.5                           # one IEEE column, inches
FIG_H = {"regret": 2.0, "mape": 1.8}  # per subfigure
FS_AXIS, FS_TICK, FS_LEG = 9, 8, 7.5
FONT_SERIF = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
LINE_W = 1.2
GRID = dict(axis="y", linestyle=":", linewidth=0.5, color="0.75")
DEFAULT_BIN = 25
# Per-chunk ceiling applied BEFORE binning, so one 1000% chunk cannot carry its bin.
# The printed summary stays uncapped, and says so.
MAPE_CLIP = 50.0
REGRET_CLIP = 20.0
YTICK_STEP = 20.0   # fallback step; ytick_step() picks one that suits the cap
YTICK_TARGET = 5    # above ~120%, widen the step instead of drawing 15 gridlines
PREVIEW_DPI = 300
# Plots live in the repo, beside the other paper figures.
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig8")

# Fixed order and colours: legend order, draw order. Never sorted by value.
WORKLOADS = [  # (key, label, colour)
    ("vpic",   "VPIC",   "#1f77b4"),
    ("nyx",    "Nyx",    "#ff7f0e"),
    ("lammps", "LAMMPS", "#2ca02c"),
    ("warpx",  "WarpX",  "#d62728"),
    ("ai",     "AI",     "#9467bd"),
]
KEYS = [k for k, _, _ in WORKLOADS]
LABEL = {k: l for k, l, _ in WORKLOADS}
COLOR = {k: c for k, _, c in WORKLOADS}

SUMMARY_FIRST_N = 200     # "first 200 chunks"
SUMMARY_LAST_FRAC = 0.20  # "last 20% of chunks"

# Per-chunk figure (fig8c_per_chunk.png): one row per workload, regret | cost MAPE.
PER_CHUNK_W = 7.16        # two IEEE columns, inches
PER_CHUNK_ROW_H = 1.35    # per workload row
PER_CHUNK_LW = 0.45
PER_CHUNK_ALPHA = 0.35    # every chunk, faint; the bin mean is drawn solid on top
REGRET_LINTHRESH = 10.0   # regret axis: linear to 10%, logarithmic above


# ----------------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------------
def workload_key(name):
    k = str(name).strip().lower()
    if k not in KEYS:
        raise ValueError(f"unknown workload {name!r} (expected one of {', '.join(KEYS)})")
    return k


def read_chunks(patterns):
    """Per-chunk CSVs -> DataFrame[workload, chunk_index, regret_pct, cost_mape_pct]."""
    frames = []
    for pat in patterns:
        paths = sorted(glob.glob(pat))
        if not paths:
            print(f"WARNING: --chunks {pat!r} matches no file")
        for p in paths:
            d = pd.read_csv(p)
            missing = {"workload", "chunk_index", "regret_pct", "cost_mape_pct"} - set(d.columns)
            if missing:
                raise ValueError(f"{p}: missing column(s) {sorted(missing)}")
            d["workload"] = d["workload"].map(workload_key)
            frames.append(d[["workload", "chunk_index", "regret_pct", "cost_mape_pct"]])
    return frames


def mape_at_floor(sel, floor, bw=5e6, cap=100.0):
    """Cost MAPE at an arbitrary time floor, from a trace's components; None
    when the trace has no component columns."""
    need = {"ct_ms", "dt_ms", "ratio", "pred_ct_ms", "pred_dt_ms", "pred_ratio",
            "chunk_bytes"}
    if not need <= set(sel.columns):
        return None
    io = lambda r: sel["chunk_bytes"] / (bw * np.clip(r, 0.1, cap))
    # An unmeasured decompression is scored at the compress time, as the runtime does.
    dt = sel["dt_ms"].where(sel["dt_ms"] > 0, sel["ct_ms"])
    pred = (sel["pred_ct_ms"].clip(lower=floor) + sel["pred_dt_ms"].clip(lower=floor)
            + io(sel["pred_ratio"]))
    meas = (sel["ct_ms"].clip(lower=floor) + dt.clip(lower=floor) + io(sel["ratio"]))
    return np.where(meas > 0, (pred - meas).abs() / meas * 100.0, np.nan)


PICK_COLS = ["chunk_bytes", "ct_ms", "dt_ms", "ratio", "pred_ct_ms", "pred_dt_ms", "pred_ratio"]


def chunk_extras(t, sel, ids):
    """The pick's and the cheapest configuration's numbers per chunk, for --chunks-out."""
    best = t.loc[t.groupby("chunk_id")["real_cost"].idxmin()].set_index("chunk_id").loc[ids]
    pick = sel.loc[ids]
    out = {"chunk_id": ids}
    if "config_id" in t.columns:
        out["pick_config"] = pick["config_id"].to_numpy()
        out["best_config"] = best["config_id"].to_numpy()
    out["pick_cost"] = pick["real_cost"].to_numpy(dtype=float)
    out["best_cost"] = best["real_cost"].to_numpy(dtype=float)
    if "predicted_cost" in sel.columns:
        out["predicted_cost"] = pick["predicted_cost"].to_numpy(dtype=float)
    if "real_cost_raw" in t.columns:
        raw_best = t.groupby("chunk_id")["real_cost_raw"].min().loc[ids].to_numpy(dtype=float)
        raw = pick["real_cost_raw"].to_numpy(dtype=float)
        out["regret_raw_pct"] = np.where(raw_best > 0, (raw - raw_best) / raw_best * 100.0, np.nan)
    for c in PICK_COLS:
        if c in sel.columns:
            out[c] = pick[c].to_numpy()
    return pd.DataFrame(out)


def read_trace(spec, mape_raw=False, mape_floor=None):
    """One (chunk, configuration) trace -> per-chunk regret and cost MAPE, in
    file order. mape_raw uses the *_raw costs when the trace has them."""
    path, _, wl = spec.rpartition(":")
    if not path:
        raise ValueError(f"--trace expects PATH:WORKLOAD, got {spec!r}")
    wl = workload_key(wl)
    t = pd.read_csv(path)
    need = {"chunk_id", "real_cost", "chosen"}
    if not need <= set(t.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}")
    if "mape_cost" not in t.columns and "predicted_cost" not in t.columns:
        raise ValueError(f"{path}: needs predicted_cost or mape_cost")

    order = pd.unique(t["chunk_id"])
    optimal = t.groupby("chunk_id")["real_cost"].min()
    sel = t[t["chosen"] == 1].drop_duplicates("chunk_id", keep="first").set_index("chunk_id")
    no_choice = len(order) - sel.index.isin(order).sum()
    if no_choice:
        print(f"WARNING {path}: {no_choice} chunk(s) have no chosen==1 row and are skipped")

    ids = [c for c in order if c in sel.index]
    measured = sel.loc[ids, "real_cost"].to_numpy(dtype=float)
    opt = optimal.loc[ids].to_numpy(dtype=float)
    regret = np.where(opt > 0, (measured - opt) / opt * 100.0, np.nan)
    extras = chunk_extras(t, sel, ids)
    at_floor = mape_at_floor(sel.loc[ids], mape_floor) if mape_floor is not None else None
    if at_floor is not None:
        print(f"{os.path.basename(path)} [{wl}]: {len(ids)} chunks, cost MAPE "
              f"re-scored at a {mape_floor:g} ms reporting floor")
        return pd.concat([pd.DataFrame({"workload": wl, "chunk_index": np.arange(len(ids)),
                                        "regret_pct": regret, "cost_mape_pct": at_floor}),
                          extras], axis=1)
    raw = (mape_raw
           and {"real_cost_raw", "predicted_cost_raw"} <= set(sel.columns))
    if "mape_cost" in sel.columns and not raw:
        mape = sel.loc[ids, "mape_cost"].to_numpy(dtype=float)
    else:
        pcol, mcol = (("predicted_cost_raw", "real_cost_raw") if raw
                      else ("predicted_cost", "real_cost"))
        pred = sel.loc[ids, pcol].to_numpy(dtype=float)
        meas = sel.loc[ids, mcol].to_numpy(dtype=float)
        mape = np.where(meas > 0, np.abs(pred - meas) / meas * 100.0, np.nan)
    print(f"{os.path.basename(path)} [{wl}]: {len(ids)} chunks, cost MAPE "
          f"{'unfloored (raw times)' if raw else f'at the {TRACE_FLOOR} floor fig8_trace.py used'}")
    return pd.concat([pd.DataFrame({"workload": wl, "chunk_index": np.arange(len(ids)),
                                    "regret_pct": regret, "cost_mape_pct": mape}),
                      extras], axis=1)


# ----------------------------------------------------------------------------
# PROCESSING
# ----------------------------------------------------------------------------
STAT = "mean"   # --stat: the per-bin statistic for every plotted line


def binned(d, col, bin_n, clip=None, stat=None):
    """`stat` of `col` per bin of `bin_n` chunks, placed at the bin's center."""
    d = d.dropna(subset=[col]).sort_values("chunk_index")
    if d.empty:
        return np.array([]), np.array([])
    v = d[col].clip(upper=clip) if clip is not None else d[col]
    g = d.assign(v=v, b=d["chunk_index"] // bin_n).groupby("b")
    x = ((g["chunk_index"].min() + g["chunk_index"].max()) / 2.0).to_numpy()
    return x, g["v"].agg(stat or STAT).to_numpy()


# ----------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------
def ytick_step(top):
    """The smallest round step giving at most YTICK_TARGET+1 gridlines."""
    if top <= 0:
        return YTICK_STEP
    for base in (1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0):
        if top / base <= YTICK_TARGET + 1:
            return base
    return top / YTICK_TARGET


def set_fonts():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    serif = [f for f in FONT_SERIF if f in avail] or ["DejaVu Serif"]
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": serif, "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return serif[0]


def draw(ax, data, panel, bin_n, regret_clip=REGRET_CLIP, mape_clip=MAPE_CLIP, stat=None):
    """One panel; every workload keeps its legend entry. `stat` may be per panel."""
    col, ylabel = ("regret_pct", "Regret (%)") if panel == "regret" else ("cost_mape_pct", "Cost MAPE (%)")
    if isinstance(stat, dict):
        stat = stat.get(panel)
    raw = mape_clip if panel == "mape" else regret_clip
    clip = raw if raw and raw > 0 else None
    xmax, ymax = 0.0, 0.0
    for k in KEYS:                                  # fixed draw order
        d = data.get(k)
        if d is None or d.empty:
            continue
        x, y = binned(d, col, bin_n, clip, stat)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=COLOR[k], linewidth=LINE_W, solid_capstyle="round")
        xmax = max(xmax, float(d["chunk_index"].max()))
        ymax = max(ymax, float(np.nanmax(y)))

    ax.set_xlim(0, xmax if xmax > 0 else 1)
    if clip is not None:
        ax.set_ylim(0, clip)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytick_step(clip)))
    else:
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
    ax.set_xlabel("Chunk index", fontsize=FS_AXIS)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():                   # full box
        sp.set_visible(True)
    handles = [Line2D([], [], color=COLOR[k], linewidth=LINE_W, label=LABEL[k]) for k in KEYS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=len(KEYS),
              fontsize=FS_LEG, frameon=False, handlelength=1.6, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.0)


def draw_per_chunk(data, bin_n, regret_clip=REGRET_CLIP, mape_clip=MAPE_CLIP, stat=None):
    """Every chunk's regret and cost MAPE, one row per workload; None without data."""
    keys = [k for k in KEYS if data.get(k) is not None and not data[k].empty]
    if not keys:
        return None
    fig, axes = plt.subplots(len(keys), 2, squeeze=False,
                             figsize=(PER_CHUNK_W, PER_CHUNK_ROW_H * len(keys) + 0.55),
                             gridspec_kw=dict(hspace=0.62, wspace=0.24))
    rclip = regret_clip if regret_clip and regret_clip > 0 else None
    mclip = mape_clip if mape_clip and mape_clip > 0 else None
    panels = (("regret_pct", "Regret (%)", rclip), ("cost_mape_pct", "Cost MAPE (%)", mclip))
    for r, k in enumerate(keys):
        d = data[k].dropna(subset=["regret_pct", "cost_mape_pct"], how="all").sort_values("chunk_index")
        for c, (col, ylabel, clip) in enumerate(panels):
            ax = axes[r][c]
            pstat = stat.get("regret" if col == "regret_pct" else "mape") \
                if isinstance(stat, dict) else stat
            y = d[col].clip(upper=clip) if clip is not None else d[col]
            ax.plot(d["chunk_index"], y, color=COLOR[k], linewidth=PER_CHUNK_LW,
                    alpha=PER_CHUNK_ALPHA, solid_joinstyle="round")
            x, m = binned(d, col, bin_n, clip, pstat)
            ax.plot(x, m, color=COLOR[k], linewidth=LINE_W, solid_capstyle="round")
            if clip is None:   # uncapped regret spans 0 to >1000%: linear to 10%, log above
                ax.set_yscale("symlog", linthresh=REGRET_LINTHRESH, linscale=0.6)
                ax.set_ylim(0, max(REGRET_LINTHRESH, float(np.nanmax(y)) * 1.3))
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            else:
                ax.set_ylim(0, clip)
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytick_step(clip)))
            ax.set_xlim(0, max(1.0, float(d["chunk_index"].max())))
            ax.set_ylabel(ylabel, fontsize=FS_AXIS)
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(**GRID)
            ax.set_axisbelow(True)
            if r == len(keys) - 1:
                ax.set_xlabel("Chunk index", fontsize=FS_AXIS)
        axes[r][0].set_title(f"{LABEL[k]}  ({len(d)} chunks)", loc="left", fontsize=FS_AXIS)
    names = sorted({(stat.get(k) or STAT) for k in ("regret", "mape")}) \
        if isinstance(stat, dict) else [stat or STAT]
    handles = [Line2D([], [], color="0.35", linewidth=PER_CHUNK_LW * 2, alpha=0.6, label="Per chunk"),
               Line2D([], [], color="0.35", linewidth=LINE_W,
                      label=f"{'/'.join(n.capitalize() for n in names)} of {bin_n} chunks")]
    # Below the bottom row's "Chunk index" labels, in figure fractions of the row count.
    h = fig.get_figheight()
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=FS_LEG, frameon=False,
               bbox_to_anchor=(0.5, -0.10 / h))
    mnote = f"Cost MAPE capped at {mclip:g}%." if mclip is not None else "Cost MAPE uncapped."
    note = (f"Regret capped at {rclip:g}%.  {mnote}"
            if rclip is not None else
            f"Regret axis: linear to {REGRET_LINTHRESH:g}%, logarithmic above.  {mnote}")
    fig.text(0.5, -0.36 / h, note, ha="center", va="top", fontsize=FS_LEG - 0.5, color="0.35")
    return fig


def summary(data):
    lines = [f"{'workload':<8}{'chunks':>8}{'regret first 200':>18}{'regret last 20%':>17}"
             f"{'cost MAPE last 20%':>20}{'regret last 20%, unfloored':>28}"]
    for k in KEYS:
        d = data.get(k)
        if d is None or d.empty:
            lines.append(f"{LABEL[k]:<8}{'no data':>8}")
            continue
        d = d.sort_values("chunk_index")
        n = len(d)
        tail = d.iloc[int(np.floor(n * (1.0 - SUMMARY_LAST_FRAC))):]
        first = d[d["chunk_index"] < d["chunk_index"].min() + SUMMARY_FIRST_N]
        raw = (f"{tail['regret_raw_pct'].mean():>27.0f}%" if "regret_raw_pct" in d
               and tail["regret_raw_pct"].notna().any() else f"{'--':>28}")
        lines.append(f"{LABEL[k]:<8}{n:>8}{first['regret_pct'].mean():>17.2f}%"
                     f"{tail['regret_pct'].mean():>16.2f}%{tail['cost_mape_pct'].mean():>19.2f}%{raw}")
    print("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", action="append", default=[], metavar="PATH",
                    help="per-chunk CSV (repeatable, globs allowed)")
    ap.add_argument("--trace", action="append", default=[], metavar="PATH:WORKLOAD",
                    help="per-(chunk, configuration) trace CSV (repeatable)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output directory (default: paper-benchmark/figures/fig8)")
    ap.add_argument("--bin", type=int, default=DEFAULT_BIN,
                    help=f"chunks per plotted bin (default: {DEFAULT_BIN})")
    ap.add_argument("--max-chunks", type=int, default=0, metavar="N",
                    help="plot only each workload's first N chunks (0 = all). Workloads whose "
                         "data is shorter are unaffected, so the x-axis is comparable.")
    ap.add_argument("--regret-clip", type=float, default=REGRET_CLIP, metavar="PCT",
                    help=f"cap each chunk's regret before binning (default: {REGRET_CLIP:g}; 0 = uncapped)")
    ap.add_argument("--mape-clip", type=float, default=MAPE_CLIP, metavar="PCT",
                    help=f"cap each chunk's cost MAPE before binning (default: {MAPE_CLIP:g}; 0 = uncapped)")
    ap.add_argument("--mape-floor", type=float, default=None, metavar="MS",
                    help="re-score cost MAPE at this time floor (needs the component "
                         "columns); regret is unchanged")
    ap.add_argument("--stat", choices=("mean", "median"), default=None,
                    help=f"per-bin statistic for BOTH panels (default: {STAT}); "
                         "--regret-stat and --mape-stat override it per panel")
    ap.add_argument("--regret-stat", choices=("mean", "median"), default=None)
    ap.add_argument("--mape-stat", choices=("mean", "median"), default=None)
    ap.add_argument("--mape-raw", dest="mape_raw", action="store_true",
                    help="score cost MAPE with no time floor (the *_raw columns)")
    ap.add_argument("--chunks-out", metavar="DIR",
                    help="also write DIR/<workload>/chunks.csv and DIR/summary.txt")
    a = ap.parse_args()
    if a.bin < 1:
        ap.error("--bin must be >= 1")
    if not a.chunks and not a.trace:
        ap.error("give --chunks and/or --trace")

    stat = {"regret": a.regret_stat or a.stat or STAT,
            "mape": a.mape_stat or a.stat or STAT}
    frames = read_chunks(a.chunks) + [read_trace(s, a.mape_raw, a.mape_floor) for s in a.trace]
    allrows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["workload", "chunk_index", "regret_pct", "cost_mape_pct"])
    if a.max_chunks > 0:
        allrows = allrows[allrows["chunk_index"] < a.max_chunks]
    data = {k: g for k, g in allrows.groupby("workload")}

    font = set_fonts()
    os.makedirs(a.out, exist_ok=True)
    written = []
    for panel, stem in (("regret", "fig8a_regret"), ("mape", "fig8b_cost_mape")):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H[panel]))
        draw(ax, data, panel, a.bin, a.regret_clip, a.mape_clip, stat)
        path = os.path.join(a.out, stem + ".png")
        fig.savefig(path, dpi=PREVIEW_DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H["regret"] + FIG_H["mape"] + 0.9),
                             gridspec_kw=dict(height_ratios=[FIG_H["regret"], FIG_H["mape"]],
                                              hspace=0.95))
    draw(axes[0], data, "regret", a.bin, a.regret_clip, a.mape_clip, stat)
    draw(axes[1], data, "mape", a.bin, a.regret_clip, a.mape_clip, stat)
    path = os.path.join(a.out, "fig8_preview.png")
    fig.savefig(path, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig = draw_per_chunk(data, a.bin, a.regret_clip, a.mape_clip, stat)
    if fig is not None:
        path = os.path.join(a.out, "fig8c_per_chunk.png")
        fig.savefig(path, dpi=PREVIEW_DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    table = summary(data)
    lim = f"   first {a.max_chunks} chunks" if a.max_chunks > 0 else ""
    cap = f"{a.regret_clip:g}%" if a.regret_clip and a.regret_clip > 0 else "off"
    mcap = f"{a.mape_clip:g}%" if a.mape_clip and a.mape_clip > 0 else "off"
    print(f"font: {font}   bin: {a.bin} chunks (regret {stat['regret']}, MAPE {stat['mape']})   plotted caps: regret {cap}, MAPE {mcap} "
          f"(summary above is uncapped){lim}")
    if a.chunks_out:
        for k, d in data.items():
            os.makedirs(os.path.join(a.chunks_out, k), exist_ok=True)
            path = os.path.join(a.chunks_out, k, "chunks.csv")
            d.sort_values("chunk_index").dropna(axis=1, how="all").to_csv(path, index=False)
            written.append(path)
        path = os.path.join(a.chunks_out, "summary.txt")
        with open(path, "w") as fh:
            fh.write(f"regret: F = {TRACE_FLOOR} unless unfloored; last 20% = the last "
                     f"{SUMMARY_LAST_FRAC:.0%} of each workload's chunks\n")
            fh.write(table + "\n")
            if a.trace:
                fh.write("traces:\n" + "".join(f"  {s}\n" for s in a.trace))
        written.append(path)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

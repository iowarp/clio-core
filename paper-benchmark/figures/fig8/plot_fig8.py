#!/usr/bin/env python3
"""Figure 8: regret and cost MAPE per chunk, one line per workload.

  (a) Regret (%)     (cost_selected - cost_optimal) / cost_optimal x 100
  (b) Cost MAPE (%)  |cost_predicted - cost_measured| / cost_measured x 100

Both use the trace's real_cost / predicted_cost; --mape-raw and --mape-floor
rescore the MAPE. Lines are per-bin means.

THIS DIRECTORY KEEPS ONE PLOTTER, so figure 8's four jobs are four
subcommands of this file rather than four scripts that drift apart. Each
one's flags belong to it alone -- `plot_fig8.py <mode> --help`:

  ./plot_fig8.py trace <wl>/explore.csv --out trace.csv
  ./plot_fig8.py panels --trace trace.csv:nyx --out DIR
  ./plot_fig8.py models --chunks fig8d_chunks.csv --out DIR
  ./plot_fig8.py split --src fig8d_models.png

`panels` takes --trace PATH:WORKLOAD (a `trace` output) and/or --chunks PATH
(workload,chunk_index,regret_pct,cost_mape_pct), and writes fig8a_regret.png,
fig8b_cost_mape.png, fig8_preview.png and fig8c_per_chunk.png into --out,
which defaults to this script's own directory. --chunks-out DIR also writes
DIR/<workload>/chunks.csv (every chunk's metrics) and DIR/summary.txt.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
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
HERE = os.path.dirname(os.path.abspath(__file__))   # figures/fig8
DEFAULT_OUT = HERE                    # the plots land beside this script

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
          f"{'unfloored (raw times)' if raw else f'at the {TRACE_FLOOR} floor the trace was written with'}")
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


def draw(ax, data, panel, bin_n, regret_clip=REGRET_CLIP, mape_clip=MAPE_CLIP, stat=None,
         eb=None):
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
              handletextpad=0.4, borderaxespad=0.0,
              title=(rf"NeuroPress, error bound $\varepsilon = {eb:g}$" if eb else None),
              title_fontsize=FS_LEG)


def draw_per_chunk(data, bin_n, regret_clip=REGRET_CLIP, mape_clip=MAPE_CLIP, stat=None,
                   eb=None):
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
    if eb:
        note = rf"NeuroPress, error bound $\varepsilon = {eb:g}$.  " + note
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


# ============================================================================
# THE TRACE  (`trace`): a run's explore.csv -> one row per (chunk,
# configuration), which is what `panels --trace` reads. It was fig8_trace.py
# until this directory went to one plotter.
# ============================================================================
RATIO_CAP = 100.0
#: Reporting floor, upstream's trace mape_cost (diagnostics_store.hpp:318).
#: Upstream's regret uses its unfloored real_cost (H5VLgpucompress.cu:2734),
#: which is real_cost_raw here.
MIN_TIME_MS = 5.0
#: How the panels describe that floor when they print what they scored.
TRACE_FLOOR = f"{MIN_TIME_MS:g} ms"


def write_trace(explore_csv, out, bw=5e6, chunks=0):
    """Score every measured configuration of every chunk, both sides floored.

    Prints the checks that say whether the run is usable: configurations per
    chunk, one pick per chunk, how much decompression was measured, and the
    penalty rows (never measured, so dropped).

    @param explore_csv a run that measured all 32 configurations per chunk
    @param out the trace csv to write
    @param bw bytes per ms in the cost model
    @param chunks keep only the first N chunks (0 = all)
    @return 0
    """
    e = pd.read_csv(explore_csv, usecols=["seq", "blob", "chunk_bytes", "role", "lib_name",
                                          "quantize", "shuffle", "pred_ratio", "pred_ct_ms",
                                          "pred_dt_ms", "ratio", "ct_ms", "dt_ms", "cost",
                                          "adopted"])
    picks = e[e["role"] == "primary"].sort_values("seq")
    if chunks:
        picks = picks.head(chunks)
    order = {b: i for i, b in enumerate(picks["blob"])}
    e = e[e["blob"].isin(order)].copy()

    penalty = (e["ct_ms"] <= 0) & (e["dt_ms"] < 0) & (e["cost"] >= 1e5)
    e["config_id"] = e["lib_name"] + "|q" + e["quantize"].astype(str) + "|s" + e["shuffle"].astype(str)
    per_chunk = e.groupby("blob").agg(rows=("config_id", "size"), configs=("config_id", "nunique"),
                                      picks=("role", lambda s: int((s == "primary").sum())))
    m = e[~penalty]
    print(f"chunks {len(order)} | configurations per chunk {per_chunk.configs.min()}..{per_chunk.configs.max()} "
          f"(rows {per_chunk.rows.min()}..{per_chunk.rows.max()}) | picks per chunk "
          f"{sorted(per_chunk.picks.unique().tolist())}")
    print(f"penalty rows dropped {int(penalty.sum())} (picks {int((penalty & (e.role == 'primary')).sum())}) | "
          f"decompression measured on {100 * (m.dt_ms >= 0).mean():.1f}% of rows | "
          f"alternative adopted on {int(e[(e.role != 'primary') & (e.adopted == 1)].blob.nunique())} chunk(s)")

    # An unmeasured decompression (dt < 0) is scored at the compression time, as
    # the runtime's own cost model does; below the floor that substitution shows.
    dt_meas = m["dt_ms"].where(m["dt_ms"] > 0, m["ct_ms"])
    io = m["chunk_bytes"] / (bw * m["ratio"].clip(lower=0.1, upper=RATIO_CAP))
    io_pred = m["chunk_bytes"] / (bw * m["pred_ratio"].clip(lower=0.1, upper=RATIO_CAP))
    m = m.assign(
        chunk_id=m["blob"].map(order),
        chosen=(m["role"] == "primary").astype(int),
        predicted_cost=m["pred_ct_ms"].clip(lower=MIN_TIME_MS) + m["pred_dt_ms"].clip(lower=MIN_TIME_MS)
        + io_pred,
        # Recomputed, not the runtime's `cost` column, so both sides share a floor.
        real_cost=m["ct_ms"].clip(lower=MIN_TIME_MS) + dt_meas.clip(lower=MIN_TIME_MS) + io,
        real_cost_raw=m["ct_ms"].clip(lower=0.0) + dt_meas.clip(lower=0.0) + io,
        predicted_cost_raw=m["pred_ct_ms"].clip(lower=0.0) + m["pred_dt_ms"].clip(lower=0.0) + io_pred,
    ).sort_values(["chunk_id", "chosen"], ascending=[True, False])
    # The six components as well, so the panels can re-score the cost MAPE at
    # any reporting floor without a rerun (--mape-floor).
    cols = ["chunk_id", "config_id", "real_cost", "chosen", "predicted_cost",
            "real_cost_raw", "predicted_cost_raw", "chunk_bytes", "cost",
            "ct_ms", "dt_ms", "ratio", "pred_ct_ms", "pred_dt_ms", "pred_ratio"]
    m[cols].rename(columns={"cost": "runtime_ranked_cost"}).to_csv(
        out, index=False, float_format="%.9g")
    print(f"wrote {out}")
    return 0


#: Legend order, colours (matching hcompress-accuracy/plot_accuracy.py's
#: MODELS), line style and
#: draw order. Two models can pick identically for a whole workload -- XGBoost
#: and HCompress seed both collapse to bitcomp|q0|s0 on VPIC and LAMMPS -- so a
#: hidden line would otherwise read as a missing one. The static models are
#: dashed, and the deployed model is drawn last, on top of everything.
MODELS = [
    ("NeuroPress (online)",   "#17becf", "-",  5),
    ("NeuroPress (static)",   "#1f77b4", "--", 4),
    ("XGBoost",               "#ff7f0e", "--", 3),
    ("HCompress CCP (+fb)",   "#d62728", "-",  2),
    ("HCompress CCP (seed)",  "#8c564b", ":",  1),
]
MODEL_WORKLOADS = [("vpic", "VPIC"), ("nyx", "Nyx"), ("lammps", "LAMMPS"),
             ("warpx", "WarpX"), ("ai", "AI")]
ROW_H = 1.45
DEFAULT_CAP = 100.0

#: The uncapped cost-MAPE axis is symmetric-log: linear up to the threshold,
#: logarithmic above. A plain log axis cannot draw an exact zero, and a bin
#: whose every chunk was predicted perfectly is a real value, not a missing one.
#: Regret uses REGRET_LINTHRESH above for the same reason.
MAPE_LINTHRESH = 1.0


def symlog_axis(ax, y, linthresh):
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.6)
    top = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else linthresh
    ax.set_ylim(0, max(linthresh, top * 1.4))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def draw_models(data, bin_n, stat, models, cap=None, floor=None):
    rows = [(k, lbl) for k, lbl in MODEL_WORKLOADS if k in data]
    fig, axes = plt.subplots(len(rows), 2, squeeze=False,
                             figsize=(PER_CHUNK_W, ROW_H * len(rows) + 0.7),
                             gridspec_kw=dict(hspace=0.70, wspace=0.22))
    panels = (("regret_pct", "Regret (%)"), ("cost_mape_pct", "Cost MAPE (%)"))
    for r, (k, lbl) in enumerate(rows):
        d = data[k]
        for c, (col, ylabel) in enumerate(panels):
            ax, allv = axes[r][c], []
            for name, colour, style, z in models:
                s = d[d.model == name]
                if s.empty:
                    continue
                # The cap goes through binned()'s own clip, so there is one
                # place where a per-chunk ceiling can be applied.
                x, y = binned(s, col, bin_n, cap, stat)
                if len(x) == 0:
                    continue
                ax.plot(x, y, color=colour, linewidth=LINE_W, linestyle=style,
                        zorder=z, solid_capstyle="round")
                allv.append(y)
            if cap is not None:
                ax.set_ylim(0, cap)
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytick_step(cap)))
            else:
                symlog_axis(ax, np.concatenate(allv) if allv else np.array([1.0]),
                            REGRET_LINTHRESH if col == "regret_pct" else MAPE_LINTHRESH)
            ax.set_xlim(0, max(1.0, float(d["chunk_index"].max())))
            ax.set_ylabel(ylabel, fontsize=FS_AXIS)
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(**GRID)
            ax.set_axisbelow(True)
            if r == len(rows) - 1:
                ax.set_xlabel("Chunk index", fontsize=FS_AXIS)
        axes[r][0].set_title(f"{lbl}  ({d.chunk_index.nunique()} chunks)",
                             loc="left", fontsize=FS_AXIS)
    handles = [Line2D([], [], color=c, linewidth=LINE_W, linestyle=st, label=n)
               for n, c, st, _ in models]
    # Anchored to the bottom ROW, not to the figure: the figure's bottom margin
    # scales with the row count, so a fixed figure-fraction leaves a growing gap.
    h = fig.get_figheight()
    y0 = min(ax.get_position().y0 for ax in axes[-1])
    fig.legend(handles=handles, loc="upper center", ncol=len(models), fontsize=FS_LEG,
               frameon=False, handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.5, y0 - 0.52 / h))
    scale = (f"Each chunk is capped at {cap:g}% before binning; both axes are linear "
             f"to {cap:g}%." if cap is not None else
             f"Uncapped.  Both axes are linear to {REGRET_LINTHRESH:g}% (regret) and "
             f"{MAPE_LINTHRESH:g}% (cost MAPE) and logarithmic above.")
    note = f"{stat.capitalize()} of {bin_n} chunks.  {scale}"
    if floor is not None:
        note = f"Reported at a {floor:g} ms time floor.  " + note
    note += "  Clamped-equal candidates are tie-broken at random."
    fig.text(0.5, y0 - 0.80 / h, note, ha="center", va="top",
             fontsize=FS_LEG - 0.5, color="0.35")
    return fig


def summary_models(data, models, cap=None):
    w = 22
    head = (f"{'workload':<9}{'model':<{w}}{'chunks':>7}{'regret first 200':>18}"
            f"{'regret last 20%':>17}{'cost MAPE last 20%':>20}{'top-1 pick':>12}"
            f"{'mean tied':>11}")
    if cap is not None:
        head += f"{'regret <=cap':>14}{'MAPE <=cap':>12}"
    # Every mean below is UNCAPPED, so the table still carries the numbers that
    # only exist above the figure's ceiling; the two <=cap columns are what the
    # capped panel actually draws.
    lines = [head]
    for k, lbl in MODEL_WORKLOADS:
        if k not in data:
            continue
        d = data[k]
        for name, *_ in models:
            s = d[d.model == name].sort_values("chunk_index")
            if s.empty:
                continue
            tail = s.iloc[int(np.floor(len(s) * (1.0 - SUMMARY_LAST_FRAC))):]
            first = s[s.chunk_index < s.chunk_index.min() + SUMMARY_FIRST_N]
            top1 = 100.0 * float((s.regret_pct <= 1e-7).mean())
            tied = f"{s.n_tied.mean():>11.1f}" if "n_tied" in s else f"{'--':>11}"
            line = (f"{lbl:<9}{name:<{w}}{len(s):>7}{first.regret_pct.mean():>17.2f}%"
                    f"{tail.regret_pct.mean():>16.2f}%{tail.cost_mape_pct.mean():>19.2f}%"
                    f"{top1:>11.1f}%{tied}")
            if cap is not None:
                line += (f"{tail.regret_pct.clip(upper=cap).mean():>13.2f}%"
                         f"{tail.cost_mape_pct.clip(upper=cap).mean():>11.2f}%")
            lines.append(line)
        lines.append("")
    lines.append("Means are UNCAPPED" + (f"; the two <=cap columns clip each chunk at "
                                         f"{cap:g}% first, as the figure does."
                                         if cap is not None else "."))
    out = "\n".join(lines).rstrip()
    print(out)
    return out


# ============================================================================
# THE SPLIT  (`split`): panel (d) as one figure per workload. It recombines
# bands of a rendered raster rather than re-plotting, because the per-chunk
# data behind it is not in the repo. The offsets are for `models` output at
# its default geometry (five workloads, --bin 25); another model set or row
# count needs them re-measured.
# ============================================================================
SPLIT_ROWS = [("vpic", 30, 369), ("nyx", 430, 769), ("lammps", 831, 1169),
              ("warpx", 1231, 1570), ("ai", 1632, 1970)]
SPLIT_XLAB = (1985, 2035)     # "Chunk index" axis labels
SPLIT_LEGEND = (2095, 2140)   # the five model series
SPLIT_FOOT = (2150, 2195)     # method footnote
SPLIT_PAD = 12


def split_models(src, out=""):
    """Cut a rendered panel (d) into one figure per workload.

    @param src the fig8d_models.png to split
    @param out where to write [fig8d_split/ beside src]
    @return 0
    """
    from PIL import Image          # only this mode needs Pillow
    out = out or os.path.join(os.path.dirname(os.path.abspath(src)), "fig8d_split")
    im = Image.open(src).convert("RGB")
    W, H = im.size
    os.makedirs(out, exist_ok=True)
    strips = [im.crop((0, t, W, b)) for t, b in (SPLIT_XLAB, SPLIT_LEGEND, SPLIT_FOOT)]
    for name, top, bot in SPLIT_ROWS:
        parts = [im.crop((0, max(0, top - SPLIT_PAD), W, min(H, bot + SPLIT_PAD)))] + strips
        canvas = Image.new("RGB", (W, sum(p.size[1] for p in parts) + SPLIT_PAD * 3), "white")
        y = 0
        for i, p in enumerate(parts):
            canvas.paste(p, (0, y))
            y += p.size[1] + (SPLIT_PAD if i < len(parts) - 1 else 0)
        path = os.path.join(out, f"fig8d_{name}.png")
        canvas.save(path, dpi=(300, 300))
        print("wrote", path)
    return 0


def draw_panels(a):
    """Panels (a)-(c): regret and cost MAPE per chunk, one line per workload.

    @param a the parsed `panels` arguments
    @return 0
    """

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
        draw(ax, data, panel, a.bin, a.regret_clip, a.mape_clip, stat, a.eb)
        path = os.path.join(a.out, stem + ".png")
        fig.savefig(path, dpi=PREVIEW_DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H["regret"] + FIG_H["mape"] + 0.9),
                             gridspec_kw=dict(height_ratios=[FIG_H["regret"], FIG_H["mape"]],
                                              hspace=0.95))
    draw(axes[0], data, "regret", a.bin, a.regret_clip, a.mape_clip, stat, a.eb)
    draw(axes[1], data, "mape", a.bin, a.regret_clip, a.mape_clip, stat, a.eb)
    path = os.path.join(a.out, "fig8_preview.png")
    fig.savefig(path, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig = draw_per_chunk(data, a.bin, a.regret_clip, a.mape_clip, stat, a.eb)
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


def run_models(a):
    """Panel (d): the same two metrics for every model, not just the deployed one.

    @param a the parsed `models` arguments
    @return 0
    """
    cap = a.cap if a.cap and a.cap > 0 else None
    df = pd.read_csv(a.chunks)
    models = MODELS
    if a.models:
        want = [m.strip() for m in a.models.split(",")]
        known = {n: (n, c, st, z) for n, c, st, z in MODELS}
        missing = [m for m in want if m not in known]
        if missing:
            raise SystemExit(f"unknown model(s): {', '.join(missing)}\n"
                             f"known: {', '.join(n for n, *_ in MODELS)}")
        models = [known[m] for m in want]
    have = set(df.model.unique())
    models = [m for m in models if m[0] in have]
    if not models:
        raise SystemExit("none of the requested models are in the CSV")

    data = {k: df[df.workload == k].copy()
            for k, _ in MODEL_WORKLOADS if (df.workload == k).any()}
    if not data:
        raise SystemExit("no known workload in the CSV")

    os.makedirs(a.out, exist_ok=True)
    set_fonts()
    floor = (float(df.floor_ms.iloc[0])
             if "floor_ms" in df and df.floor_ms.nunique() == 1 else None)
    fig = draw_models(data, a.bin, a.stat, models, cap, floor)
    png = os.path.join(a.out, f"{a.name}.png")
    fig.savefig(png, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)
    txt = os.path.join(a.out, f"{a.name}_summary.txt")
    with open(txt, "w") as f:
        f.write(summary_models(data, models, cap) + "\n")
    print(f"\nplotted cap: {f'{cap:g}%' if cap is not None else 'off'}")
    print(f"wrote {png} and {txt}")
    return 0


def main():
    """Four jobs, one per subcommand, so each one's flags stay its own."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_subparsers(dest="mode", required=True, metavar="MODE")

    m = mode.add_parser("trace", help="a run's explore.csv -> the per-(chunk, "
                                      "configuration) trace the panels read")
    m.add_argument("explore_csv", metavar="EXPLORE.CSV")
    m.add_argument("--out", required=True, metavar="CSV")
    m.add_argument("--bw", type=float, default=5e6,
                   help="bytes per ms in the cost (default 5e6)")
    m.add_argument("--chunks", type=int, default=0, metavar="N",
                   help="keep only the first N chunks (0 = all)")

    m = mode.add_parser("panels", help="THE PAPER'S FIGURE: panels (a)-(c), "
                                       "regret and cost MAPE per chunk")
    m.add_argument("--chunks", action="append", default=[], metavar="PATH",
                   help="per-chunk CSV (repeatable, globs allowed)")
    m.add_argument("--trace", action="append", default=[], metavar="PATH:WORKLOAD",
                   help="a `trace` output, one per workload (repeatable)")
    m.add_argument("--out", default=DEFAULT_OUT,
                   help="output directory [this script's own]")
    m.add_argument("--bin", type=int, default=DEFAULT_BIN,
                   help=f"chunks per plotted bin (default: {DEFAULT_BIN})")
    m.add_argument("--max-chunks", type=int, default=0, metavar="N",
                   help="plot only each workload's first N chunks (0 = all). "
                        "Workloads whose data is shorter are unaffected, so "
                        "the x-axis is comparable.")
    m.add_argument("--regret-clip", type=float, default=REGRET_CLIP, metavar="PCT",
                   help=f"cap each chunk's regret before binning (default: "
                        f"{REGRET_CLIP:g}; 0 = uncapped)")
    m.add_argument("--mape-clip", type=float, default=MAPE_CLIP, metavar="PCT",
                   help=f"cap each chunk's cost MAPE before binning (default: "
                        f"{MAPE_CLIP:g}; 0 = uncapped)")
    m.add_argument("--mape-floor", type=float, default=None, metavar="MS",
                   help="re-score cost MAPE at this time floor (needs the "
                        "component columns); regret is unchanged")
    m.add_argument("--stat", choices=("mean", "median"), default=None,
                   help=f"per-bin statistic for BOTH panels (default: {STAT}); "
                        "--regret-stat and --mape-stat override it per panel")
    m.add_argument("--regret-stat", choices=("mean", "median"), default=None)
    m.add_argument("--mape-stat", choices=("mean", "median"), default=None)
    m.add_argument("--mape-raw", dest="mape_raw", action="store_true",
                   help="score cost MAPE with no time floor (the *_raw columns)")
    m.add_argument("--eb", type=float, default=0.05,
                   help="the runs' absolute error bound, shown with the legend "
                        "(default: 0.05, figure_8.sh's; 0 hides it)")
    m.add_argument("--chunks-out", metavar="DIR",
                   help="also write DIR/<workload>/chunks.csv and DIR/summary.txt")

    m = mode.add_parser("models", help="panel (d): the same metrics for every "
                                       "model, from fig8_model_chunks.py")
    m.add_argument("--chunks", required=True, help="fig8_model_chunks.py output")
    m.add_argument("--out", default=DEFAULT_OUT,
                   help="output directory [this script's own]")
    m.add_argument("--bin", type=int, default=DEFAULT_BIN)
    m.add_argument("--stat", default="mean", choices=["mean", "median"])
    m.add_argument("--cap", type=float, default=DEFAULT_CAP, metavar="PCT",
                   help=f"cap each chunk's regret and cost MAPE before binning, "
                        f"and draw linear axes to it (default: {DEFAULT_CAP:g}; "
                        "0 = uncapped, symlog axes). The summary's means stay "
                        "uncapped.")
    m.add_argument("--models", default="",
                   help="comma-separated subset, in draw order")
    m.add_argument("--name", default="fig8d_models", help="output basename")

    m = mode.add_parser("split", help="cut a rendered panel (d) into one "
                                      "figure per workload")
    m.add_argument("--src", required=True, help="the fig8d_models.png to split")
    m.add_argument("--out", default="", metavar="DIR",
                   help="output directory [fig8d_split/ beside --src]")

    a = ap.parse_args()
    if a.mode == "trace":
        return write_trace(a.explore_csv, a.out, a.bw, a.chunks)
    if a.mode == "split":
        return split_models(a.src, a.out)
    if a.bin < 1:
        ap.error("--bin must be >= 1")
    if a.mode == "models":
        return run_models(a)
    if not a.chunks and not a.trace:
        ap.error("give --chunks and/or --trace")
    return draw_panels(a)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Producer + k-means consumer runs (1 write, R reads): best single codec,
NeuroPress learning and oracle, side by side.

    compare_kmeans_runs.py DATASET [--tag km4] [--bw 520000] [--png PNG]

Reads runs/<DATASET>_{fixed,learn,oracle}_<tag> (run_v2_workloads.sh with
READS=R KMEANS=K). Per method:
  write_s      producer: stage + compress + store (stdout "stage+compress")
  read_s       get + decompress, summed over the R timed read-backs
  kmeans_s     consumer computation, summed over the R iterations
  e2e_s        write_s + read_s + kmeans_s (the application's total time)
  ratio        bytes / stored bytes
  compress_s, decompress_s (R passes), predict_s, learn_s   (phases.csv)
  model_pfs_s  compress (with shuffle) + decompress (R passes) +
               (1 + R) * stored bytes / bw + NeuroPress's prediction: the
               time on the PFS of the cost model, from this run's own
               measured times (as costmodel_whatif.py models it)
  centroids    the last iteration's centroids; they must agree across the
               methods (lossless data, same analysis result)
Output: runs/<DATASET>_<tag>_compare.csv and a PNG figure.
"""
import argparse
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = "/mnt/nvme0/v2-work/runs"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "../figures/new-workloads/nn-v2")
# Every option selects by the same cost model (user, 2026-10-05): the best
# single codec has the lowest total cost, NeuroPress and the oracle the lowest
# predicted / measured cost for each chunk. --fixed-label renames the fixed run
# when it is not the lowest-cost single codec.
BASE_NAME = "best single"   # the fixed run, in the "% vs ..." labels
METHODS = (("fixed", "best single codec (lowest cost)"),
           ("learn", "NeuroPress learning"),
           ("oracle", "oracle (each chunk's best by cost)"))

def app_dir(ds):
    """@return the figure directory of a workload's simulation:
    sim-tuning/vpic, sim-tuning/nyx, ... (created), or nn-v2 for other
    workloads."""
    for app in ("vpic", "nyx", "lammps", "warpx"):
        if app in ds:
            d = os.path.join(FIG, "..", "sim-tuning", app)
            os.makedirs(d, exist_ok=True)
            return d
    return FIG



def parse_stdout(run):
    """Write time, per-pass read / consumer times and centroids from stdout.log.

    @param run run directory
    @return dict with write_s, read_ms (list), kmeans_ms (list), centroids
            ({field: centroid values} of the last iteration; the field is
            "all" in the first output format, which had no field)
    """
    text = open(os.path.join(run, "stdout.log")).read()
    w = re.search(r"stage\+compress ([0-9.]+) s", text)
    passes = re.split(r"\nREAD \d+/\d+\n", text)[1:]
    read_ms, km_ms = [], []
    for p in passes:
        g = re.search(r"get\+decompress: ([0-9.]+) ms", p)
        k = re.search(r"consumer \(k-means\): ([0-9.]+) ms", p)
        if g and k:   # the untimed check pass has no consumer line
            read_ms.append(float(g.group(1)))
            km_ms.append(float(k.group(1)))
    found = re.findall(r"KMEANS iteration (\d+)/\d+(?: field (\S+))?:[^\n]*\n"
                       r"  centroids:([^\n]*)", text)
    last = max((int(i) for i, _, _ in found), default=0)
    cents = {f or "all": [float(x) for x in c.split()]
             for i, f, c in found if int(i) == last}
    return {"write_s": float(w.group(1)) if w else np.nan,
            "read_ms": read_ms, "kmeans_ms": km_ms, "centroids": cents}


def phase_totals(run, reads):
    """Phase sums (s) from phases.csv; read rows of the R timed passes only.

    @param run   run directory
    @param reads number of timed read-backs
    @return dict of compress_s, decompress_s, predict_s, learn_s, stored and
            raw bytes
    """
    ph = pd.read_csv(os.path.join(run, "phases.csv"))
    w, rd = ph[ph.path == "write"], ph[ph.path == "read"]
    rd = rd.iloc[:reads * len(w)]           # the last pass is the untimed check
    return {"compress_s": (w.compress_ms.sum() + w.preproc_ms.fillna(0).sum()) / 1e3,
            "decompress_s": rd.decompress_ms.sum() / 1e3,
            "predict_s": (w.nn_ms.sum() + w.convert_ms.sum()) / 1e3,
            "learn_s": w.sgd_ms.sum() / 1e3,
            "raw_bytes": float(w.chunk_bytes.sum()),
            "stored_bytes": float(w.stored_bytes.sum())}


def run_row(run, label, mode, bw):
    """One method's measured row and centroids.

    @param run   run directory
    @param label method name for tables and figures
    @param mode  fixed / learn / oracle, or ref for the reference run
    @param bw    PFS bandwidth of the cost model, bytes per ms
    @return (row dict, centroids by field)
    """
    s = parse_stdout(run)
    reads = len(s["read_ms"])
    p = phase_totals(run, reads)
    km_s = sum(s["kmeans_ms"]) / 1e3
    # A pass's time ("pass:" = get+decompress + consumer) is the measured
    # time of one read with its k-means; with the timing read (reads in
    # flight, 1 or more) "get+decompress" already holds the consumer, so the
    # read time is the pass minus the consumer.
    passes = [float(x) for x in re.findall(r"pass: ([0-9.]+) ms",
                                           open(os.path.join(run, "stdout.log")).read())]
    pass_s = sum(passes) / 1e3 if passes else sum(s["read_ms"]) / 1e3 + km_s
    read_s = pass_s - km_s
    io_s = (1 + reads) * p["stored_bytes"] / bw / 1e3
    row = {"method": label, "mode": mode, "reads": reads,
           "write_s": s["write_s"], "read_s": read_s, "kmeans_s": km_s,
           "e2e_s": s["write_s"] + pass_s,
           "read_pass_s": " ".join(f"{x / 1e3:.2f}" for x in s["read_ms"]),
           "kmeans_pass_s": " ".join(f"{x / 1e3:.3f}" for x in s["kmeans_ms"]),
           "ratio": p["raw_bytes"] / p["stored_bytes"],
           "compress_s": p["compress_s"], "decompress_s": p["decompress_s"],
           "predict_s": p["predict_s"], "learn_s": p["learn_s"],
           "pfs_io_s": io_s,
           "model_pfs_s": p["compress_s"] + p["decompress_s"] + io_s + p["predict_s"]}
    return row, s["centroids"]


def summarize(ds, tag, bw, ref_tag=None, ref_label=None):
    """One row per method; the reference run, if any, second.

    @param ds        dataset
    @param tag       run tag
    @param bw        PFS bandwidth of the cost model, bytes per ms
    @param ref_tag   tag of a fixed run to compare against as well (e.g. the
                     fastest single codec by time), or None
    @param ref_label its method name
    @return (DataFrame, centroids per method)
    """
    runs = [(os.path.join(RUNS, f"{ds}_{mode}_{tag}"), label, mode) for mode, label in METHODS]
    if ref_tag:
        runs.insert(1, (os.path.join(RUNS, f"{ds}_fixed_{ref_tag}"), ref_label, "ref"))
    rows, cents = [], {}
    for run, label, mode in runs:
        if not os.path.exists(os.path.join(run, "stdout.log")):
            continue
        row, cents[label] = run_row(run, label, mode, bw)
        rows.append(row)
    t = pd.DataFrame(rows)
    cols = ("e2e_s", "write_s", "read_s", "model_pfs_s", "ratio")
    for mode, suffix in (("fixed", "best_single"), ("ref", "ref")):
        if len(t) and (t["mode"] == mode).any():
            base = t[t["mode"] == mode].iloc[0]
            for c in cols:
                t[f"{c}_vs_{suffix}_pct"] = 100 * (t[c] / base[c] - 1)
    return t, cents


def plot(t, ds, reads, size_gib, bw, png, w, note=""):
    """Measured time split, ratio and modelled PFS time per method."""
    fig, ax = plt.subplots(1, 3, figsize=(5 * len(t) + 3, 5.2))
    x = np.arange(len(t))
    names = [textwrap.fill(m, 24) for m in t.method]
    parts = (("write_s", "producer write (compress + store)", "#4c72b0"),
             ("read_s", f"consumer reads ({reads} x get + decompress)", "#dd8452"),
             ("kmeans_s", f"k-means computation ({reads} iterations)", "#55a868"))
    bottom = np.zeros(len(t))
    for col, lab, c in parts:
        ax[0].bar(x, t[col], bottom=bottom, color=c, label=lab, width=0.6)
        bottom += t[col].to_numpy()

    def tag_bars(a, vals, col, fmt):
        for i, v in enumerate(vals):
            text = fmt.format(v)
            for suffix, who in (("best_single", BASE_NAME), ("ref", "ref")):
                c = f"{col}_vs_{suffix}_pct"
                if c in t and t["mode"].iat[i] != {"best_single": "fixed"}.get(suffix, suffix):
                    text += f"\n{t[c].iat[i]:+.1f}% vs {who}"
            a.text(i, v, text, ha="center", va="bottom", fontsize=8.5)

    tag_bars(ax[0], t.e2e_s, "e2e_s", "{:.1f} s")
    ax[0].set_title("Measured application time (local NVMe)")
    ax[0].set_ylabel("seconds")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[1].bar(x, t.ratio, color="#8172b3", width=0.6)
    tag_bars(ax[1], t.ratio, "ratio", "{:.3f}x")
    ax[1].set_title("Compression ratio (higher is better)")
    ax[2].bar(x, t.model_pfs_s, color="#937860", width=0.6)
    tag_bars(ax[2], t.model_pfs_s, "model_pfs_s", "{:.1f} s")
    ax[2].set_title(f"Modelled time on a PFS at {bw / 1e6:.2f} GB/s\n"
                    f"(compress + {reads} x decompress + {1 + reads} x stored bytes / bandwidth)",
                    fontsize=10)
    ax[2].set_ylabel("seconds")
    for a in ax:
        a.set_xticks(x, names, fontsize=9)
        a.set_ylim(0, a.get_ylim()[1] * 1.35)
    best = t.method.iat[0]
    fig.suptitle(f"{ds} ({size_gib:.2f} GiB{note}): producer writes once, k-means consumer "
                 f"reads {reads} times.\nNeuroPress selects with the cost weights compress "
                 f"{w[0]:g}, decompress {w[1]:g}, transfer {w[2]:g}; the oracle is each chunk's "
                 f"lowest-cost setting; the best single codec has the lowest total cost."
                 f"\nEvery measured time is wall clock and includes the compress "
                 f"time, whatever the weights."
                 + ("  'ref' = " + t[t["mode"] == "ref"].method.iat[0] + "."
                    if (t["mode"] == "ref").any() else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


def plot_iterations(t, ds, size_gib, png, note=""):
    """Each clustering iteration's read and k-means time, the running total and
    the ratio.

    @param t        summarize() table (best single codec first)
    @param ds       dataset
    @param size_gib workload size
    @param png      output path
    """
    colors = {"fixed": "#4c72b0", "learn": "#dd8452", "oracle": "#55a868", "ref": "#8c8c8c"}
    reads = int(t.reads.iat[0])
    rd = {r.mode: np.array([float(x) for x in r.read_pass_s.split()]) for r in t.itertuples()}
    km = {r.mode: np.array([float(x) for x in r.kmeans_pass_s.split()]) for r in t.itertuples()}
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.8),
                           gridspec_kw={"width_ratios": [1.7, 1, 0.9]})
    it = np.arange(1, reads + 1)
    width = 0.8 / len(t)
    base = rd[t["mode"].iat[0]] + km[t["mode"].iat[0]]
    for j, r in enumerate(t.itertuples()):
        x = it - 0.4 + width * (j + 0.5)
        ax[0].bar(x, rd[r.mode], width=width, color=colors[r.mode], label=r.method)
        ax[0].bar(x, km[r.mode], bottom=rd[r.mode], width=width, color=colors[r.mode],
                  alpha=0.45, hatch="//")
        for xi, v, b in zip(x, rd[r.mode] + km[r.mode], base):
            pct = "" if j == 0 else f"\n{100 * (v / b - 1):+.1f}%"
            ax[0].text(xi, v, f"{v:.{1 if v >= 10 else 2}f} s{pct}", ha="center",
                       va="bottom", fontsize=8)
        steps = np.concatenate([[0.0, r.write_s], r.write_s + np.cumsum(rd[r.mode] + km[r.mode])])
        ax[1].plot(np.arange(len(steps)), steps, "o-", color=colors[r.mode], label=r.method)
        ax[1].text(len(steps) - 1, steps[-1], f" {steps[-1]:.1f} s", va="center", fontsize=8.5,
                   color=colors[r.mode])
    ax[0].set_xticks(it, [f"iteration {i}" for i in it])
    ax[0].set_ylabel("seconds")
    ax[0].set_title("Each clustering iteration: one full read\n(solid: get + decompress, "
                    f"hatched: k-means computation); % = against {BASE_NAME}", fontsize=10)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.35)
    ax[0].legend(fontsize=8, loc="upper center", ncol=3)
    ax[1].set_xticks(np.arange(reads + 2), ["start", "write"] + [f"read {i}" for i in it])
    ax[1].set_ylabel("seconds since the start")
    ax[1].set_title("Running total of the application time\n(write, then one read per "
                    "iteration)", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_xlim(-0.3, reads + 2.2)
    xr = np.arange(len(t))
    ax[2].bar(xr, t.ratio, color=[colors[m] for m in t["mode"]], width=0.6)
    for i, r in enumerate(t.itertuples()):
        text = f"{r.ratio:.3f}x"
        if r.mode != "fixed":
            text += f"\n{r.ratio_vs_best_single_pct:+.1f}% vs {BASE_NAME}"
        if "ratio_vs_ref_pct" in t and r.mode != "ref":
            text += f"\n{t.ratio_vs_ref_pct.iat[i]:+.1f}% vs ref"
        ax[2].text(i, r.ratio, text, ha="center", va="bottom", fontsize=8)
    ax[2].set_xticks(xr, [textwrap.fill(m, 13) for m in t.method], fontsize=7.5)
    ax[2].set_ylim(0, ax[2].get_ylim()[1] * 1.35)
    ax[2].set_ylabel("ratio")
    ax[2].set_title("Compression ratio (higher is better):\nfewer bytes in every read",
                    fontsize=10)
    fig.suptitle(f"{ds} ({size_gib:.2f} GiB{note}): measured through Clio on the local NVMe; the "
                 f"k-means consumer reads every chunk once per iteration ({reads} iterations)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--tag", default="km4")
    ap.add_argument("--bw", type=float, default=520000.0,
                    help="PFS bandwidth of the cost model, bytes per ms")
    ap.add_argument("--png", default=None)
    ap.add_argument("--w", default=None,
                    help="selection cost weights W_CT,W_DT,W_IO, for the title "
                         "[1,R,1+R: the advisor's rule]")
    ap.add_argument("--note", default="", help="text after the size in the titles")
    ap.add_argument("--fixed-label", default=None,
                    help="method name of the fixed run when it is not the lowest-cost "
                         "single codec under the run's weights")
    ap.add_argument("--base-name", default="the fixed run",
                    help="short name of the fixed run in the '% vs ...' labels (with --fixed-label)")
    ap.add_argument("--ref-tag", default=None,
                    help="tag of a fixed run to compare against too (e.g. km4e-ans)")
    ap.add_argument("--ref-label", default="fastest single codec by time",
                    help="method name of the --ref-tag run")
    a = ap.parse_args()
    if a.fixed_label:
        global METHODS, BASE_NAME
        METHODS = ((METHODS[0][0], a.fixed_label),) + METHODS[1:]
        BASE_NAME = a.base_name
    t, cents = summarize(a.dataset, a.tag, a.bw, a.ref_tag, a.ref_label)
    if t.empty:
        print("no runs yet")
        return
    out = os.path.join(RUNS, f"{a.dataset}_{a.tag}_compare.csv")
    t.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.drop(columns=["mode"]).round(3).to_string(index=False))
    ref_name, ref = next(iter(cents.items()))
    for m, by_field in cents.items():
        d = max((abs(u - v) / max(abs(u), 1e-300)
                 for f, c in by_field.items() for u, v in zip(c, ref.get(f, []))),
                default=np.nan)
        print(f"  centroids ({len(by_field)} field(s)), {m}: largest relative "
              f"difference to {ref_name}: {d:.2e}")
    ph = pd.read_csv(os.path.join(RUNS, f"{a.dataset}_{t['mode'].iat[0]}_{a.tag}",
                                  "phases.csv"), usecols=["path", "chunk_bytes"])
    size = ph[ph.path == "write"].chunk_bytes.sum() / 2**30
    reads = int(t.reads.iat[0])
    w = tuple(float(x) for x in a.w.split(",")) if a.w else (1.0, reads, 1.0 + reads)
    note = f", {a.note}" if a.note else ""
    tag = "" if a.tag == "km4" else f"_{a.tag}"
    png = a.png or os.path.join(app_dir(a.dataset), f"v2_{a.dataset}_kmeans_{reads}reads{tag}.png")
    plot(t, a.dataset, reads, size, a.bw, png, w, note)
    plot_iterations(t, a.dataset, size, png.replace(".png", "_per_iteration.png"), note)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

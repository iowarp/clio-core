#!/usr/bin/env python3
"""Tables and figure of run_single_codecs.sh: each codec of a list run alone (Clio's fixed mode) on
every final workload in the option-A setup, to check the best single codec by hand.

    single_codecs.py collect

Per workload, for every finished run in SC_DIR/<workload>/s<setting>_<name>/ (the option-A setup:
1 process x 1 chunk in flight, 1 write + 10 timed reads, page cache dropped before each read):
  e2e_s            end-to-end time: the write window + the timed read passes (as option A's app_s)
  write_s, read_s  the write window and the sum of the timed read passes
  compress_s       sum over the chunks of the measured compress time (write path)
  decompress_s     sum over the chunks and timed reads of the measured decompress time
  write_io_s, read_io_s   the same for the storage I/O
  ratio            raw bytes / stored bytes
  cost_s           the workload's cost model on the measured times: sum over the chunks of
                   w_ct x compress + w_dt x mean decompress + w_io x stored bytes / cost bandwidth
  exh_cost_s, exh_rank_of_45   the same cost model on the stored exhaustive search the best static
                   was chosen from (result-archive/exhaustive-baselines), and that setting's rank
                   among all 45; the summary compares the two cost rankings
  ratio_mismatch_chunks, raw_fallback_chunks   chunks whose ratio differs from the exhaustive search
                   (repo archive) / chunks stored raw by a codec fallback
next to the option-A best static (its setting, its option-A time and ratio, and how the same
setting measures here). Writes SC_DIR/single_codecs.csv, SC_DIR/single_codecs_summary.txt and
SC_DIR/single_codecs.png.
"""
import os
import re
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd

import compare_parallel_runs as cp
import eval_v2_workloads as ev
import final_config as fc
import ground_truth as gt
from plot_workload_summary import GRID, INK, INK_2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-work/runs"
SC_DIR = os.environ.get("SC_DIR", "/mnt/nvme0/v2-work/single-codecs")
ARCHIVE = os.path.join(HERE, "..", "result-archive", "exhaustive-search-2026-10-07")
# the stored exhaustive search the option-A best static was chosen from (run_kmeans_parallel.sh)
BASELINES = os.path.join(HERE, "..", "result-archive", "exhaustive-baselines")
OPTION_A = os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning", "ipdps-results", "option_a_1x1_runs.csv")
STATIC_COLOR, WRITE_COLOR, READ_COLOR = "#2a7bd5", "#8f86d0", "#f0a5b0"


def run_row(wl, run, setting, name, exh):
    """@return the measured numbers of one finished fixed-setting run (see the module doc)."""
    _, w, bw = fc.WORKLOADS[wl]
    x = cp.parse_proc(run)
    passes = [(b - a) / 1e9 for a, b in x["passes_ns"]]
    write = (x["end_ns"] - x["start_ns"]) / 1e9
    wr, rd = gt.phases(run)
    reads = max(len(passes), 1)
    dt_mean = rd.groupby("chunk_id").decompress_ms.mean().reindex(wr.chunk_id).fillna(0).to_numpy()
    cost = (w[0] * wr.compress_ms.fillna(0).to_numpy() + w[1] * dt_mean
            + w[2] * wr.stored_bytes.to_numpy() / bw).sum() / 1e3
    log = open(os.path.join(run, "stdout.log")).read()
    km = len(re.findall(rf"^KMEANS iteration \d+/{reads} done: \d+ chunk\(s\) skipped, 0 failed", log, re.M))
    mism, raw = gt.codec_check(run, setting, exh)
    return {"workload": wl, "setting": setting, "name": name, "e2e_s": write + sum(passes), "write_s": write,
            "read_s": sum(passes), "reads": len(passes), "compress_s": wr.compress_ms.sum() / 1e3,
            "decompress_s": rd.decompress_ms.sum() / 1e3, "decompress_per_read_s": rd.decompress_ms.sum() / 1e3 / reads,
            "write_io_s": wr.io_ms.sum() / 1e3, "read_io_s": rd.io_ms.sum() / 1e3, "ratio": x["raw"] / x["stored"],
            "stored_GB": x["stored"] / 1e9, "cost_s": cost, "digest_ok": x["digest_ok"], "timed_ok": x["timed_ok"],
            "kmeans_done": km, "ratio_mismatch_chunks": mism, "raw_fallback_chunks": raw, "run": run}


def option_a_static(wl):
    """@return (setting, app s, ratio) of the option-A best-static run of a workload, or None."""
    if not os.path.exists(OPTION_A):
        return None
    a = pd.read_csv(OPTION_A)
    a = a[(a.workload == wl) & (a.option == "best static")]
    if a.empty:
        return None
    run = os.path.join(RUNS, a.run.iloc[0])
    first = pd.read_csv(os.path.join(run, "v2_measured.csv"), nrows=1)
    return int(first.setting.iloc[0]), float(a.app_s.iloc[0]), float(a.ratio.iloc[0])


def exhaustive_costs(wl):
    """@return (total cost in s per setting over the complete chunks, the lowest-cost setting): the
    workload's cost model on the stored exhaustive search, computed exactly as
    eval_v2_workloads.load_truth and run_kmeans_parallel.sh do when they choose the best static
    (a setting that does not shrink a chunk stores it raw; 'store' is raw I/O only)."""
    ds, w, bw = fc.WORKLOADS[wl]
    d = os.path.join(BASELINES, ds)
    names, store = ev.settings_list()
    pred = pd.read_csv(os.path.join(d, "v2_pred.csv.xz"), usecols=["blob", "bytes"])
    m = pd.read_csv(os.path.join(d, "v2_measured.csv.xz"))
    m = m[~ev.raw_primary(m)].drop_duplicates(["blob", "setting"])
    m = m[m.decomp_ms > 0]
    idx = {b: i for i, b in enumerate(pred.blob)}
    cost = np.full((len(pred), len(names)), np.nan)
    raw_io = pred.set_index("blob").bytes.reindex(m.blob).to_numpy(float) / bw
    kept = m.ratio.to_numpy() > 1
    c = np.where(kept, w[0] * m.comp_ms + w[1] * m.decomp_ms + w[2] * raw_io / m.ratio, w[0] * m.comp_ms + w[2] * raw_io)
    cost[m.blob.map(idx).to_numpy(), m.setting.to_numpy()] = c
    cost[:, store] = w[2] * pred.bytes.to_numpy(float) / bw
    ok = ~np.isnan(cost).any(axis=1)
    total = ev.for_selection(cost[ok]).sum(axis=0) / 1e3
    return total, int(np.argmin(total))


def workload_table(wl):
    """@return the table of one workload's finished runs (ranked by end-to-end time) and the
    summary lines."""
    ds = fc.WORKLOADS[wl][0]
    d = os.path.join(SC_DIR, wl.lower())
    runs = sorted(x for x in (os.listdir(d) if os.path.isdir(d) else []) if re.match(r"s\d+_", x))
    if not runs:
        return None, []
    f = os.path.join(ARCHIVE, ds, "v2_measured.csv.xz")
    exh = pd.read_csv(f, usecols=["blob", "setting", "ratio"]) if os.path.exists(f) else None
    t = pd.DataFrame([run_row(wl, os.path.join(d, x), int(x[1:].split("_", 1)[0]), x.split("_", 1)[1].replace("_", " "),
                              exh) for x in runs]).sort_values("e2e_s").reset_index(drop=True)
    t["rank_e2e"] = np.arange(1, len(t) + 1)
    t["e2e_vs_fastest_pct"] = 100 * (t.e2e_s / t.e2e_s.min() - 1)
    t["rank_cost"] = t.cost_s.rank(method="min").astype(int)
    ref = option_a_static(wl)
    t["option_a_best_static"] = t.setting == (ref[0] if ref else -1)
    exh, exh_best = exhaustive_costs(wl)
    t["exh_cost_s"] = t.setting.map(lambda x: exh[x])
    t["exh_rank_of_45"] = t.setting.map(lambda x: int((exh < exh[x]).sum()) + 1)
    t["exh_rank"] = t.exh_cost_s.rank(method="min").astype(int)
    fast, cheap = t.iloc[0], t.loc[t.cost_s.idxmin()]
    lines = [f"{wl} (cost model {fc.model_name(fc.WORKLOADS[wl][1])} at {fc.WORKLOADS[wl][2] / 1e6:g} GB/s): "
             f"{len(t)} codecs, all bit-exact: {bool(t.digest_ok.all() and t.timed_ok.all())}",
             f"  fastest end to end: {fast['name']} {fast.e2e_s:.1f} s, ratio {fast.ratio:.2f}",
             f"  lowest cost (measured times): {cheap['name']} (cost rank 1, e2e rank {cheap.rank_e2e})"]
    if ref:
        s = t[t.setting == ref[0]]
        if s.empty:
            lines.append(f"  option-A best static: setting {ref[0]} not in this list")
        else:
            s = s.iloc[0]
            lines += [f"  option-A best static: {s['name']}; option A {ref[1]:.1f} s ratio {ref[2]:.3f}, here "
                      f"{s.e2e_s:.1f} s ({100 * (s.e2e_s / ref[1] - 1):+.1f} %) ratio {s.ratio:.3f}",
                      f"  option-A best static is the lowest-cost codec here: {s.setting == cheap.setting}; "
                      f"it is {s.e2e_vs_fastest_pct:.1f} % slower than the fastest (e2e rank {s.rank_e2e})"]
    names = ev.settings_list()[0]
    by_exh = ", ".join(t.sort_values("exh_cost_s")["name"])
    by_meas = ", ".join(t.sort_values("cost_s")["name"])
    lines += [f"  lowest cost over all {len(exh)} settings in the exhaustive search: {names[exh_best]}"
              + (f" (= the option-A best static: {exh_best == ref[0]})" if ref else ""),
              f"  cost ranking of these codecs, exhaustive search: {by_exh}",
              f"  cost ranking of these codecs, measured runs:     {by_meas}",
              f"  same order: {by_exh == by_meas}; lowest-cost codec the same: "
              f"{t.loc[t.exh_cost_s.idxmin()].setting == cheap.setting}"]
    odd = t[t.ratio_mismatch_chunks.fillna(0) > 0]
    lines.append("  chunks whose ratio differs from the exhaustive search: "
                 + (", ".join(f"{r['name']} {int(r.ratio_mismatch_chunks)} (raw {int(r.raw_fallback_chunks)})"
                              for _, r in odd.iterrows()) or "none"))
    return t, lines


def panel(ax, t, wl):
    """One workload: end-to-end time per codec (fastest first) split into the write window and the
    timed reads (the two add up to the end-to-end time); the compress and decompress time (they can
    overlap the I/O, so they are not stacked), the ratio and the cost rank are written at the bar."""
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.write_s, 0.62, color=WRITE_COLOR, label="write (compress + write I/O)")
    ax.barh(y, t.read_s, 0.62, left=t.write_s, color=READ_COLOR, label="timed reads (decompress + read I/O)")
    for yy, (_, r) in zip(y, t.iterrows()):
        tag = "   <- option-A best static" if r.option_a_best_static else ""
        ax.text(r.e2e_s + 0.01 * t.e2e_s.max(), yy,
                f"{r.e2e_s:.0f} s | compress {r.compress_s:.1f} s | decompress {r.decompress_s:.1f} s | "
                f"ratio {r.ratio:.2f} | cost rank {r.rank_cost}{tag}", va="center", fontsize=8,
                color=STATIC_COLOR if r.option_a_best_static else INK,
                fontweight="bold" if r.option_a_best_static or r.rank_e2e == 1 else "normal")
    ax.set_yticks(y, t["name"], fontsize=8.5, color=INK)
    ax.set_xlim(0, t.e2e_s.max() * 2.15)
    ds, w, bw = fc.WORKLOADS[wl]
    ax.set_title(f"{wl}: cost model {fc.model_name(w)} at {bw / 1e6:g} GB/s", loc="left", fontsize=11,
                 fontweight="bold", color=INK)
    ax.set_xlabel(f"end-to-end time (s), 1 write + {int(t.reads.max())} reads", fontsize=8.5, color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)


def plot(tables, out):
    """2 x 2 panels, one per workload."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (wl, t) in zip(axes.flat, tables.items()):
        panel(ax, t, wl)
    for ax in list(axes.flat)[len(tables):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=2, frameon=False, fontsize=9.5)
    reads = int(max(t.reads.max() for t in tables.values()))
    fig.text(0.01, 0.995, "Single codecs, each run alone on every workload: end-to-end time, its parts and the ratio",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.962, textwrap.fill(
        f"Measured in Clio, option-A setup: 1 process, 1 chunk in flight, 1 write + {reads} reads each followed by one "
        "k-means iteration (8 clusters, not in the time), page cache dropped before each read, 1 run per codec; every "
        "chunk verified bit-exact. Compress / decompress: summed over the chunks (and reads); they can overlap the I/O. "
        "Cost rank: the workload's cost model (w_ct x compress + w_dt x decompress + w_io x stored bytes / bandwidth) "
        "on the measured times; the option-A best static was chosen as the lowest-cost setting.", 190),
        fontsize=9.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def collect():
    """Write the tables, the summary and the figure of every finished run in SC_DIR."""
    tables, lines = {}, []
    for wl in fc.WORKLOADS:
        t, ln = workload_table(wl)
        if t is not None:
            tables[wl] = t
            lines += ln
    if not tables:
        raise SystemExit(f"no finished run in {SC_DIR}")
    pd.concat(tables.values(), ignore_index=True).to_csv(os.path.join(SC_DIR, "single_codecs.csv"), index=False)
    with open(os.path.join(SC_DIR, "single_codecs_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    plot(tables, os.path.join(SC_DIR, "single_codecs.png"))
    print("\n".join(lines))


def main():
    if sys.argv[1:] != ["collect"]:
        raise SystemExit(__doc__.split("\n\n")[1])
    collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())

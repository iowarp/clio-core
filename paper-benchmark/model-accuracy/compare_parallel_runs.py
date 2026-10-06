#!/usr/bin/env python3
"""Producer + k-means runs with several processes and chunks in flight
(run_kmeans_parallel.sh) against the serial runs (1 process, 1 chunk in
flight; run_kmeans_benchmark.sh): best single codec, NeuroPress learning and
oracle.

    compare_parallel_runs.py DATASET --procs 2 --tag km4p2i8w0-1-1 --w 0,1,1
                             [--serial-tag km4w0-1-1] [--best NAME] [--fig-dir DIR]

Per option of the parallel runs (runs/DATASET-p<i>_<mode>_<tag>):
  app_s     the measured application time: the length of the union of every
            process's timed intervals (its write window and each timed read
            pass, k-means included), in the driver's steady clock, the same in
            every process; with 1 process it is write + the read passes
  write_s   the joint write window: first process's start to the last one's
            end
  read_s    app_s - write_s: the time spent in reads only
  span_s    first write start to last timed read end, untimed work between
            the phases included (source digests, k-means seeding)
  wall_s    the whole option, process start to exit of every process
            (run_kmeans_parallel.sh's runs/DATASET_<tag>_walls.csv)
  ratio     all processes' bytes / stored bytes
  checks    every process's last read (untimed, sequential) bit-exact; the
            timed reads check return codes only
  prewarm_per_setting, codec_builds_in_timed
            codec objects warmed per setting at start, and codec objects
            built on demand inside a timed interval (runtime.log; 0 = no
            codec setup in the timed work)
Per option of the serial runs (runs/DATASET_<mode>_<serial-tag>): the same
columns from compare_kmeans_runs.run_row (write, reads + k-means, e2e).
Output: runs/DATASET_<tag>_compare.csv and <fig-dir>/v2_DATASET_<tag>.png.
"""
import argparse
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_kmeans_runs as ck
import plot_style as style

MODES = (("fixed", "best single codec"), ("learn", "NeuroPress learning"),
         ("oracle", "oracle"), ("hcompress", "HCompress"))


def has_runs(ds, procs, mode, tag):
    """@return True when every process of one option has a finished run log."""
    return all(os.path.exists(os.path.join(ck.RUNS, f"{ds}-p{i}_{mode}_{tag}", "stdout.log"))
               for i in range(procs))


def parse_proc(run):
    """One process's write window, read passes, bytes and checks.

    @param run run directory
    @return dict of start_ns, end_ns, write_s, pass_s (list), raw, stored,
            chunks, digest_ok (last read bit-exact), timed_ok (timed reads)
    """
    t = open(os.path.join(run, "stdout.log")).read()
    win = re.search(r"window: start_ns (\d+)\s+end_ns (\d+)", t)
    st = re.search(r"stored (\d+) blob\(s\), (\d+) B in -> (\d+) B", t)
    check = t.split("READ check", 1)
    ver = re.findall(r"VERIFIED: (\d+) of (\d+) ", check[0])
    pass_s = [float(x) / 1e3 for x in re.findall(r"pass: ([0-9.]+) ms", t)]
    rt = os.path.join(run, "runtime.log")
    rlog = open(rt, errors="replace").read() if os.path.exists(rt) else ""
    builds = [int(x) for x in re.findall(r"build on demand: .*? steady_ns (\d+)", rlog)]
    warm = re.search(r"prewarm: \d+ codec objects, (\d+) per setting", rlog)
    starts = [int(x) for x in re.findall(r"pass window: start_ns (\d+)", t)]
    return {"start_ns": int(win.group(1)), "end_ns": int(win.group(2)),
            "write_s": float(re.search(r"stage\+compress ([0-9.]+) s", t).group(1)),
            "pass_s": pass_s,
            # each timed pass as [start, start + its timed length]: a sequential
            # pass's window also holds its untimed per-chunk check
            "passes_ns": [(s0, s0 + int(p * 1e9)) for s0, p in zip(starts, pass_s)],
            "builds_ns": builds, "prewarm": int(warm.group(1)) if warm else 0,
            "chunks": int(st.group(1)), "raw": int(st.group(2)), "stored": int(st.group(3)),
            "digest_ok": len(check) == 2 and bool(re.search(r"VERIFIED: (\d+) of \1 ", check[1])),
            "timed_ok": len(ver) > 0 and all(a == b for a, b in ver)}


def union_s(intervals):
    """Length (s) of the union of [start_ns, end_ns] intervals."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return (total + (cur_e - cur_s if cur_e is not None else 0)) / 1e9


def walls(ds, tag):
    """{mode: whole-option wall clock (s)} from run_kmeans_parallel.sh, or {}."""
    f = os.path.join(ck.RUNS, f"{ds}_{tag}_walls.csv")
    return pd.read_csv(f).set_index("mode").wall_s.to_dict() if os.path.exists(f) else {}


def config_label(procs, inflight):
    """@return the label of a configuration, e.g. "2 processes, 8 in flight each"."""
    return (f"{procs} process{'es' if procs > 1 else ''}, {inflight} chunk"
            f"{'s' if inflight > 1 else ''} in flight{' each' if procs > 1 else ''}")


def parallel_row(ds, procs, mode, tag):
    """One option of the parallel runs, all processes together.

    @param procs processes; the chunks in flight come from the tag (p<P>i<I>)
    @return row dict
    """
    m = re.search(r"p(\d+)i(\d+)", tag)
    inflight = int(m.group(2)) if m else 1
    p = [parse_proc(os.path.join(ck.RUNS, f"{ds}-p{i}_{mode}_{tag}")) for i in range(procs)]
    write = (max(x["end_ns"] for x in p) - min(x["start_ns"] for x in p)) / 1e9
    app = union_s([(x["start_ns"], x["end_ns"]) for x in p]
                  + [iv for x in p for iv in x["passes_ns"]])
    last = max(iv[1] for x in p for iv in x["passes_ns"])
    timed = [(x["start_ns"], x["end_ns"]) for x in p] + [iv for x in p for iv in x["passes_ns"]]
    in_timed = sum(1 for x in p for b in x["builds_ns"] if any(a <= b <= e for a, e in timed))
    reads = [sum(x["pass_s"]) for x in p]
    return {"mode": mode, "config": config_label(procs, inflight),
            "write_s": write, "read_s": app - write, "app_s": app,
            "span_s": (last - min(x["start_ns"] for x in p)) / 1e9,
            "wall_s": walls(ds, tag).get(mode, float("nan")),
            "prewarm_per_setting": min(x["prewarm"] for x in p),
            "codec_builds_in_timed": in_timed,
            "ratio": sum(x["raw"] for x in p) / sum(x["stored"] for x in p),
            "chunks": sum(x["chunks"] for x in p),
            "start_spread_ms": (max(x["start_ns"] for x in p)
                                - min(x["start_ns"] for x in p)) / 1e6,
            "process_write_s": " ".join(f"{x['write_s']:.3f}" for x in p),
            "process_read_s": " ".join(f"{r:.3f}" for r in reads),
            "digest_ok": all(x["digest_ok"] for x in p),
            "timed_ok": all(x["timed_ok"] for x in p)}


def serial_row(ds, mode, tag, bw):
    """One option of the serial runs (1 process, 1 chunk in flight), or None."""
    run = os.path.join(ck.RUNS, f"{ds}_{mode}_{tag}")
    if not os.path.exists(os.path.join(run, "stdout.log")):
        return None
    r, _ = ck.run_row(run, mode, mode, bw)
    return {"mode": mode, "config": "1 process, 1 chunk in flight (serial)", "write_s": r["write_s"],
            "read_s": r["read_s"] + r["kmeans_s"], "app_s": r["e2e_s"], "ratio": r["ratio"]}


def TitleWrap(text, width=150):
    """@return text with each line wrapped to `width` characters."""
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def bar_options(ax, t, configs, names, reads):
    """Draw the three options of every configuration side by side.

    Top panel: application time, the write window as the darker lower part
    and the reads + k-means above it; bottom panel: the ratio. Labels:
    seconds or ratio over the best single codec, % against it over the other
    two; a bar whose last read was not bit-exact gets a red edge and says so.

    @param ax      the two axes (time, ratio)
    @param t       one row per (config, mode), with the *_vs_best_single_pct columns
    @param configs the configurations, in x order
    @param names   {mode: legend name}
    @param reads   the timed reads per process, for the legend
    """
    x = np.arange(len(configs))
    modes = [m for m, _ in MODES if m in set(t["mode"])]
    width = 0.78 / len(modes)   # the bars of one configuration leave a gap to the next
    for k, mode in enumerate(modes):
        s = t[t["mode"] == mode].set_index("config").reindex(configs)
        pos = x + (k - (len(modes) - 1) / 2) * width
        color = style.OPTION_COLORS[mode]
        ok = s["digest_ok"] if "digest_ok" in s else pd.Series(True, index=s.index)
        bad = ~ok.fillna(True).astype(bool).to_numpy()
        edge = np.where(bad, "red", "white")
        lw = np.where(bad, 2.2, 0.8)
        ax[0].bar(pos, s.write_s, width, color=style.darker(color), edgecolor="white",
                  linewidth=0.8)
        ax[0].bar(pos, s.read_s, width, bottom=s.write_s, color=color, label=names[mode],
                  edgecolor=edge, linewidth=lw)
        ax[1].bar(pos, s.ratio, width, color=color, label=names[mode], edgecolor=edge,
                  linewidth=lw)
        for p, v, d, r, rd, b in zip(pos, s.app_s, s.app_s_vs_best_single_pct, s.ratio,
                                     s.ratio_vs_best_single_pct, bad):
            if not np.isfinite(v):
                continue
            note = "\nNOT\nbit-exact" if b else ""
            for a, y, lab in ((ax[0], v, f"{v:.1f} s" if k == 0 else style.pct(d)),
                              (ax[1], r, f"{r:.2f}x" if k == 0 else style.pct(rd))):
                a.annotate(lab + note, (p, y), xytext=(0, 2), textcoords="offset points",
                           ha="center", va="bottom", fontsize=8,
                           color="red" if b else ("#333333" if k == 0 else color),
                           fontweight="normal" if k == 0 else "bold")
    for a in ax:
        a.set_ylim(0, a.get_ylim()[1] * 1.12)
        a.set_xlim(-0.6, len(configs) - 0.4)
    handles, labels = ax[0].get_legend_handles_labels()
    handles.append(matplotlib.patches.Patch(facecolor=style.darker("#9aa0a6")))
    labels.append(f"darker lower part: write\n(compress + store);\nupper part: {reads} reads + k-means")
    ax[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.005, 1.0))   # outside: no bar under it


def plot(t, ds, procs, inflight, w, best, png, reads=4, bw=520000.0):
    """Application time and ratio per configuration, the three options of one
    configuration side by side."""
    style.apply()
    configs = list(dict.fromkeys(t.config))   # reference(s) first
    fig, ax = plt.subplots(2, 1, figsize=(max(11, 2.6 * len(configs)), 9.5), sharex=True)
    names = {"fixed": f"best single codec ({best})", "learn": "NeuroPress learning",
             "oracle": "oracle (each chunk's best)", "hcompress": "HCompress"}
    bar_options(ax, t, configs, names, reads)
    ax[1].set_xticks(np.arange(len(configs)), [c.replace(", ", "\n") for c in configs])
    ax[0].set_ylabel("application time (s)")
    ax[0].set_title("Application time (lower is better)")
    ax[1].set_ylabel("compression ratio")
    ax[1].set_title("Compression ratio (higher is better)")
    style.titles(fig, f"{ds}: best single codec, NeuroPress learning and oracle",
                 TitleWrap(f"Producer writes once, a k-means consumer reads {reads} times. All "
                           f"options select by the cost weights {w} (compress / decompress / "
                           f"transfer) at {bw / 1e6:g} GB/s. % = change against the best single "
                           f"codec of the same configuration. With P processes, each process "
                           f"handles one chunk in P with its own Clio runtime. Application time "
                           f"= time when at least one process writes or does a timed read.", 150))
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print("wrote", png)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--w", required=True, help="the cost weights, for the title")
    ap.add_argument("--serial-tag", default=None,
                    help="tag of serial runs named runs/DATASET_<mode>_<tag>")
    ap.add_argument("--ref-tag", default=None,
                    help="tag of reference runs of run_kmeans_parallel.sh (e.g. PROCS=1 INFLIGHT=1)")
    ap.add_argument("--ref-procs", type=int, default=1)
    ap.add_argument("--best", default="", help="the best single codec's name")
    ap.add_argument("--bw", type=float, default=520000.0, help="bytes per ms")
    ap.add_argument("--fig-dir", default=None)
    a = ap.parse_args()
    rows = [parallel_row(a.dataset, a.procs, m, a.tag) for m, _ in MODES
            if has_runs(a.dataset, a.procs, m, a.tag)]
    if a.ref_tag:   # one or more tags, comma-separated; processes from p<P> in each
        refs = []
        for rt in a.ref_tag.split(","):
            rp = int(re.search(r"p(\d+)i\d+", rt).group(1)) if re.search(r"p(\d+)i\d+", rt) \
                else a.ref_procs
            refs += [parallel_row(a.dataset, rp, m, rt) for m, _ in MODES
                     if has_runs(a.dataset, rp, m, rt)]
        rows = refs + rows
    elif a.serial_tag:
        rows = [r for r in (serial_row(a.dataset, m, a.serial_tag, a.bw) for m, _ in MODES)
                if r] + rows
    t = pd.DataFrame(rows)
    for c in ("app_s", "write_s", "read_s", "ratio"):
        base = t.groupby("config")[c].transform("first")
        t[f"{c}_vs_best_single_pct"] = 100 * (t[c] / base - 1)
    out = os.path.join(ck.RUNS, f"{a.dataset}_{a.tag}_compare.csv")
    t.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(3).to_string(index=False))
    inflight = int(re.search(r"p\d+i(\d+)", a.tag).group(1)) if re.search(r"p\d+i(\d+)", a.tag) else 0
    fig_dir = a.fig_dir or ck.app_dir(a.dataset)
    os.makedirs(fig_dir, exist_ok=True)
    reads = len(parse_proc(os.path.join(ck.RUNS, f"{a.dataset}-p0_fixed_{a.tag}"))["pass_s"])
    plot(t, a.dataset, a.procs, inflight, a.w.replace(",", " / "), a.best,
         os.path.join(fig_dir, f"v2_{a.dataset}_{a.tag}.png"), reads, a.bw)
    print("wrote", out)


if __name__ == "__main__":
    main()

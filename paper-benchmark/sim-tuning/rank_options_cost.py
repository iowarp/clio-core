#!/usr/bin/env python3
"""Score every selection option of a workload with one cost model, from the
stored CSV files only (no run).

    rank_options_cost.py DATASET [--w 1,4,5] [--bw 520000] [--tag km4e]
                         [--run-w 0,1,1] [--out-dir DIR]

Truth: the stored exhaustive search (every setting's measured compress time,
decompress time and ratio on every chunk). With the weights w = 1 / R / R+1
the cost of a setting on a chunk is the modelled runtime of 1 write + R reads
on one tier of bandwidth BW: 1 x compress + R x decompress + (1 + R) x stored
bytes / BW (a chunk with ratio <= 1 is stored raw: no decompress).

Options (rows):
  measured runs <ds>_{fixed,learn,oracle}_<tag> (Clio, selected with --run-w):
    best single codec (the fixed run's setting), NeuroPress learning, oracle
  what-if (offline replay of Clio's selection and learning, costmodel_whatif):
    NeuroPress as trained and NeuroPress learning (1 pass), selecting with --w;
    the oracle under --w (each chunk's lowest --w cost: rank 1 by definition)
For each option: the --w cost split (compress / decompress / transfer), the
rank of the selected setting among all settings of each chunk under the --w
cost (1 = cheapest), and the codec mix. The measured runs' own --w cost
(their measured compress, decompress and stored bytes; compare_kmeans_runs.py)
is shown next to the truth-based one.
Output: <out-dir>/rank_options_<ds>_w<W>.csv and _per_chunk.csv.gz, and the
figure v2_<ds>_kmeans_<R>reads_w<W>_whatif.png in the format of the k-means
figures (best single codec, NeuroPress learning and oracle under --w).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import costmodel_whatif as cw  # noqa: E402

rl = cw.rl
RUNS = rl.RUNS
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning")
RANK_BINS = ((1, 1, "rank 1 (cheapest)"), (2, 2, "rank 2"), (3, 3, "rank 3"),
             (4, 5, "rank 4-5"), (6, 10, "rank 6-10"), (11, 99, "rank > 10"))
RANK_COLORS = ("#1baf7a", "#8fd5b4", "#f2d06b", "#f2a65a", "#e0613a", "#9c2b2b")


def measured_picks(ds, tag, mode, blobs):
    """The setting each measured run stored per chunk, in `blobs` order."""
    m = pd.read_csv(os.path.join(RUNS, f"{ds}_{mode}_{tag}", "v2_measured.csv"))
    prim = m[m.role == "primary"].drop_duplicates("blob").set_index("blob").setting
    return prim.reindex(blobs).to_numpy()


def rank_of(cost, picks):
    """1 + the number of candidate settings strictly cheaper than the pick, per
    chunk (a pick of a dropped setting is ranked by its measured cost)."""
    c = np.where(np.isnan(cost), np.inf, cost)
    chosen = c[np.arange(len(c)), picks]
    return 1 + (cw.ev.for_selection(c) < chosen[:, None]).sum(axis=1)


def option_row(name, kind, picks, mats, w, names, reads, measured=None):
    """Totals, rank shares and codec mix of one option.

    @param mats (ct, dt, stored, cost, bw, nbytes, ok) over all chunks
    @param measured the measured run's own cost split (dict) or None
    """
    ct, dt, stored, cost, bw, nbytes, ok = mats
    r = np.arange(len(picks))[ok]
    p = picks[ok]
    row = {"option": name, "source": kind,
           "compress_s": w[0] * ct[r, p].sum() / 1e3,
           "decompress_s": w[1] * dt[r, p].sum() / 1e3,
           "transfer_s": w[2] * (stored[r, p] / bw[r]).sum() / 1e3}
    row["cost_s"] = row["compress_s"] + row["decompress_s"] + row["transfer_s"]
    # The modelled runtime of 1 write + R reads, whatever the weights: every
    # part counted (the cost with w ignores what a zero weight drops).
    row["rt_compress_s"] = ct[r, p].sum() / 1e3
    row["rt_decompress_s"] = reads * dt[r, p].sum() / 1e3
    row["rt_transfer_s"] = (1 + reads) * (stored[r, p] / bw[r]).sum() / 1e3
    row["runtime_s"] = row["rt_compress_s"] + row["rt_decompress_s"] + row["rt_transfer_s"]
    row["ratio"] = nbytes[r].sum() / stored[r, p].sum()
    rk = rank_of(cost[ok], p)
    for lo, hi, lab in RANK_BINS:
        row[lab] = 100 * np.mean((rk >= lo) & (rk <= hi))
    row["mean_rank"] = rk.mean()
    best = np.nanmin(cw.ev.for_selection(cost[ok]), axis=1).sum() / 1e3
    row["vs_per_chunk_best_pct"] = 100 * (row["cost_s"] / best - 1)
    mix = pd.Series([names[k] for k in p]).value_counts(normalize=True) * 100
    row["codec_mix"] = "; ".join(f"{k} {v:.1f}%" for k, v in mix.items())
    if measured is not None:
        row.update({f"measured_{k}": v for k, v in measured.items()})
    return row, rk, mix


def plot_like_kmeans(t, ds, gib, w, bw, reads, predict_s, out):
    """The what-if under w in the format of the k-means figures
    (compare_kmeans_runs.py): best single codec, NeuroPress learning
    (selected with w) and the oracle under w, as vertical bars.

    @param t         option rows (best single first; the what-if rows by name)
    @param gib       workload size in GiB
    @param predict_s NeuroPress's prediction time (s), from the measured run
    @param out       PNG path
    """
    pick = [t.iloc[0], t[t.option.str.startswith("NeuroPress learning (1 pass)")].iloc[0],
            t[t.option.str.startswith("oracle") & (t.source == "what-if")].iloc[0]]
    names = [f"best single codec\n(lowest cost)", "NeuroPress learning\n(1 pass)",
             "oracle\n(each chunk's best by cost)"]
    x = np.arange(3)
    write = np.array([r.rt_compress_s + r.rt_transfer_s / (1 + reads) for r in pick])
    rd = np.array([r.rt_decompress_s + r.rt_transfer_s * reads / (1 + reads) for r in pick])
    pr = np.array([0.0, predict_s, 0.0])
    total = write + rd + pr
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    ax[0].bar(x, write, color="#4c72b0", width=0.6, label="producer write (compress + transfer)")
    ax[0].bar(x, rd, bottom=write, color="#dd8452", width=0.6,
              label=f"consumer reads ({reads} x transfer + decompress)")
    ax[0].bar(x, pr, bottom=write + rd, color="#333333", width=0.6, label="NeuroPress prediction")

    def tag(a, vals, fmt):
        for i, v in enumerate(vals):
            extra = "" if i == 0 else f"\n{100 * (v / vals[0] - 1):+.1f}% vs best single"
            a.text(i, v, fmt.format(v) + extra, ha="center", va="bottom", fontsize=9)
    tag(ax[0], total, "{:.1f} s")
    ax[0].set_title(f"Modelled runtime on a PFS at {bw / 1e6:.2f} GB/s (compress + {reads} x "
                    f"decompress\n+ {1 + reads} x stored bytes / bandwidth; NeuroPress adds its "
                    f"prediction time)", fontsize=10)
    ax[0].set_ylabel("seconds")
    ax[0].legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    ratio = np.array([r.ratio for r in pick])
    ax[1].bar(x, ratio, color="#8172b3", width=0.6)
    tag(ax[1], ratio, "{:.3f}x")
    ax[1].set_title("Compression ratio (higher is better)", fontsize=11)
    bottom = np.zeros(3)
    for (lo, hi, lab), c in zip(RANK_BINS, RANK_COLORS):
        v = np.array([r[lab] for r in pick])
        ax[2].bar(x, v, bottom=bottom, color=c, width=0.6, label=lab)
        for i in range(3):
            if v[i] >= 5:
                ax[2].text(i, bottom[i] + v[i] / 2, f"{v[i]:.0f}%", ha="center", va="center",
                           fontsize=8.5)
        bottom += v
    ax[2].set_ylim(0, 100)
    ax[2].set_ylabel("share of chunks, %")
    ax[2].set_title(f"Rank of the selected setting among all settings\nof the chunk under the "
                    f"{w[0]:g}/{w[1]:g}/{w[2]:g} cost", fontsize=10)
    ax[2].legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    for a in ax:
        a.set_xticks(x, names, fontsize=9)
    for a in ax[:2]:
        a.set_ylim(0, a.get_ylim()[1] * 1.35)
    fig.suptitle(
        f"{ds} ({gib:.2f} GiB), what-if: all options select by the cost weights {w[0]:g} / "
        f"{w[1]:g} / {w[2]:g} (compress / decompress / transfer); "
        f"the producer writes once, a k-means consumer reads {reads} times (one PFS tier).\n"
        f"Offline replay from the stored CSV files (exhaustive search: measured times of every "
        f"setting on every chunk); no new run.\nBest single "
        f"codec = the setting with the lowest total cost ({t.option.iat[0].split(': ')[-1]}); "
        f"NeuroPress = each chunk's lowest predicted cost; oracle = each chunk's lowest "
        f"measured cost. Percent = change against the best single codec.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", os.path.abspath(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--w", default="1,4,5")
    ap.add_argument("--reads", type=int, default=None,
                    help="reads of the runtime model [round(w_dt)]: give it when the weights "
                         "are not the runtime weights 1 / R / R + 1")
    ap.add_argument("--bw", type=float, default=520000.0, help="bytes per ms")
    ap.add_argument("--tag", default="km4e",
                    help="tag of the measured runs, or none (what-if rows only)")
    ap.add_argument("--predict-s", type=float, default=None,
                    help="NeuroPress's prediction time (s) [the measured learning run's]")
    ap.add_argument("--run-w", default="0,1,1", help="weights the measured runs selected with")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    ds = a.dataset
    w = tuple(float(x) for x in a.w.split(","))
    reads = a.reads if a.reads is not None else int(round(w[1]))
    names, store, order, meas, _ = rl.load(ds)
    nbytes = order.bytes.to_numpy(float)
    bw = np.full(len(order), a.bw)
    ct, dt, stored = cw.outcomes(meas, nbytes, store)
    cost = cw.truth_cost(meas, nbytes, store, w, bw)
    ok = ~np.isnan(cost).any(axis=1)
    mats = (ct, dt, stored, cost, bw, nbytes, ok)
    blobs = order.blob.to_numpy()
    have_runs = a.tag != "none"
    comp = (pd.read_csv(os.path.join(RUNS, f"{ds}_{a.tag}_compare.csv")).set_index("mode")
            if have_runs else None)
    rw = "/".join(f"{float(x):g}" for x in a.run_w.split(","))
    ws = "/".join(f"{x:g}" for x in w)
    # The best single codec selects by the same cost as NeuroPress and the
    # oracle: the one setting with the lowest total cost under w.
    sel = cw.ev.for_selection(np.where(np.isnan(cost), np.inf, cost))   # candidates only
    best_k = int(np.argmin(np.where(ok[:, None], sel, 0.0).sum(axis=0)))
    opts = [(f"best single codec (lowest {ws} cost): {names[best_k]}", "what-if",
             np.full(len(blobs), best_k), None)]
    if have_runs:
        fixed = measured_picks(ds, a.tag, "fixed", blobs)
        opts += [(f"single codec of the measured runs: "
                  f"{names[int(pd.Series(fixed).mode().iat[0])]}", "measured run", fixed, "fixed"),
                 (f"NeuroPress learning, selected with {rw}", "measured run",
                  measured_picks(ds, a.tag, "learn", blobs), "learn"),
                 (f"oracle: each chunk's best by the {rw} cost", "measured run",
                  measured_picks(ds, a.tag, "oracle", blobs), "oracle")]
    sc = {"w": w, "bw": a.bw}
    for method, picks, _ in cw.replays(ds, (names, store, order, meas, cost), sc, 1):
        label = ("NeuroPress as trained (no learning)" if method == cw.FROZEN
                 else "NeuroPress learning (1 pass)")
        opts.append((f"{label}, selected with {ws}", "what-if", picks, None))
    opts.append((f"oracle: each chunk's best by the {ws} cost", "what-if",
                 np.argmin(sel, axis=1), None))
    rows, ranks = [], {}
    for name, kind, picks, mode in opts:
        measured = None
        if mode is not None:
            m = comp.loc[mode]
            measured = {"compress_s": m.compress_s, "decompress_s": m.decompress_s,
                        "transfer_s": m.pfs_io_s,
                        "cost_s": m.compress_s + m.decompress_s + m.pfs_io_s,
                        "ratio": m.ratio, "e2e_s": m.e2e_s}
        row, rk, _ = option_row(name, kind, picks.astype(int), mats, w, names, reads, measured)
        rows.append(row)
        ranks[name] = (picks.astype(int), rk)
    t = pd.DataFrame(rows)
    t["vs_best_single_pct"] = 100 * (t.cost_s / t.cost_s.iat[0] - 1)
    out_dir = a.out_dir or os.path.join(FIGS, next((x for x in ("vpic", "nyx", "lammps", "warpx")
                                                    if x in ds), ""))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"rank_options_{ds}_w{ws.replace('/', '')}")
    t.to_csv(stem + ".csv", index=False)
    per = pd.DataFrame({"blob": blobs[ok]})
    for name, (picks, rk) in ranks.items():
        per[f"{name} | setting"] = [names[k] for k in picks[ok]]
        per[f"{name} | cost_ms"] = cost[ok][np.arange(ok.sum()), picks[ok]]
        per[f"{name} | rank"] = rk
    per.to_csv(stem + "_per_chunk.csv.gz", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(t.drop(columns=["codec_mix"]).round(2).to_string(index=False))
    predict_s = a.predict_s if a.predict_s is not None else (
        comp.loc["learn"].predict_s if have_runs else 0.0)
    plot_like_kmeans(t, ds, nbytes.sum() / 2**30, w, a.bw, reads, predict_s,
                     os.path.join(out_dir, f"v2_{ds}_kmeans_{reads}reads_w{ws.replace('/', '')}"
                                           f"_whatif.png"))
    print("wrote", stem + ".csv", "and", stem + "_per_chunk.csv.gz")


if __name__ == "__main__":
    main()

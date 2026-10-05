#!/usr/bin/env python3
"""How well does the NeuroPress v2 model predict on the real workloads?

    pred_error_real.py [DATASET ...] [--out PNG] [--pdf]

Truth: the stored exhaustive search (baselines/<ds>/exhaustive), which
measured compress time, decompress time and ratio of every setting except
store on every chunk. Two sets of predictions are scored against it:
  frozen   the model as trained (the exhaustive run's v2_pred.csv; no
           learning in that run)
  learned  NeuroPress learning (runs/<ds>_learn_nolog/v2_pred.csv): the
           prediction made for each chunk when it was chosen, with the
           weights updated by every earlier chunk
Per workload: absolute % error |pred - meas| / meas (median and mean, no
clamping) of compress time, decompress time and ratio, and the bias (median
pred / meas); how often the predicted-cheapest setting is the truly cheapest
by the balanced 4-tier cost, where the truly cheapest ranks in the
prediction, and the cost of the predicted pick over the true best. Also
broken down by element type. Writes runs/pred_error_real.csv and
runs/pred_error_by_type.csv.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_v2_workloads as ev

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/mnt/nvme0/v2-work/baselines"
RUNS = "/mnt/nvme0/v2-work/runs"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
FULL = ["nyx-full", "omics-pbmc", "gnn-igbh", "analytics-tpch", "climate-era5"]
# The model on the synthetic float32 corpus it was trained on (nn_v2_cv.py,
# model B, 5-fold cross-validation, results/nn-v2/cv_*.csv): median absolute
# % error, and the share of files whose predicted cheapest setting is the
# cheapest, at the 1:3:3:3 mix of 12 / 1 / 0.5 / 0.25 GB/s.
CV_MEDIAN_APE = {"ct": 3.6, "dt": 4.0, "ratio": 4.8}
CV_TOP1 = 0.1 * 97.1 + 0.3 * (85.6 + 82.9 + 83.7)
INK, INK2 = "#1f2328", "#57606a"
FROZEN, LEARNED = "#8c959f", "#d1495b"


def measured(ds):
    """Long table of measured (blob, setting, ct, dt, ratio) from the exhaustive run."""
    m = pd.read_csv(os.path.join(STORE, ds, "exhaustive", "v2_measured.csv"))
    m = m[~ev.raw_primary(m) & (m.decomp_ms > 0)].drop_duplicates(["blob", "setting"])
    return m.rename(columns={"comp_ms": "ct", "decomp_ms": "dt"})[
        ["blob", "setting", "spec", "ct", "dt", "ratio"]]


def predicted(path, n):
    """Long table of predicted (blob, setting, ct, dt, ratio) from a v2_pred.csv."""
    p = pd.read_csv(path)
    rows = []
    for k in range(n):
        rows.append(pd.DataFrame({"blob": p.blob, "setting": k, "ct": p[f"ct{k}"],
                                  "dt": p[f"dt{k}"], "ratio": p[f"r{k}"]}))
    return pd.concat(rows, ignore_index=True), p.set_index("blob")[["bytes", "tier_bw"]]


def dtype_of(blob):
    """Element type from the chunk name (__dt-<type>), else f32."""
    return blob.split("__dt-")[1].split("/")[0] if "__dt-" in blob else "f32"


def score(ds, which, names, store, blobs, truth, meas):
    """Error and ranking scores of one set of predictions on one workload."""
    path = (os.path.join(STORE, ds, "exhaustive", "v2_pred.csv") if which == "frozen"
            else os.path.join(RUNS, f"{ds}_learn_nolog", "v2_pred.csv"))
    pred, io = predicted(path, len(names))
    j = meas.merge(pred, on=["blob", "setting"], suffixes=("_m", "_p"))
    out = {"workload": ds, "model": which, "chunks": len(blobs), "pairs": len(j)}
    for q in ("ct", "dt", "ratio"):
        ape = 100 * (j[f"{q}_p"] - j[f"{q}_m"]).abs() / j[f"{q}_m"]
        out[f"{q}_ape_median"] = ape.median()
        out[f"{q}_ape_mean"] = ape.mean()
        out[f"{q}_bias"] = (j[f"{q}_p"] / j[f"{q}_m"]).median()
        j[f"{q}_ape"] = ape
    # Ranking by the predicted 4-tier cost, as the runtime ranks (RankKernel).
    pc = np.full(truth.shape, np.nan)
    idx = {b: i for i, b in enumerate(blobs)}
    r = pred.blob.map(idx).to_numpy()
    b = io.bytes.reindex(pred.blob).to_numpy(float)
    bw = io.tier_bw.reindex(pred.blob).to_numpy(float)
    pc[r, pred.setting.to_numpy()] = (pred.ct + pred.dt + b / (pred.ratio * bw)).to_numpy()
    ok = ~np.isnan(truth).any(axis=1) & ~np.isnan(pc).any(axis=1)
    t, p = truth[ok], pc[ok]
    pick, best = p.argmin(axis=1), t.argmin(axis=1)
    rank_of_best = (p < p[np.arange(len(p)), best][:, None]).sum(axis=1) + 1
    out["top1_hit_pct"] = 100 * (pick == best).mean()
    out["best_in_top3_pct"] = 100 * (rank_of_best <= 3).mean()
    out["rank_of_best_median"] = np.median(rank_of_best)
    out["pick_cost_over_best_pct"] = 100 * (t[np.arange(len(t)), pick].sum()
                                            / t[np.arange(len(t)), best].sum() - 1)
    j["dtype"] = j.blob.map(dtype_of)
    by = j.groupby("dtype").agg(chunks=("blob", "nunique"), ct=("ct_ape", "median"),
                                dt=("dt_ape", "median"), ratio=("ratio_ape", "median"))
    by.insert(0, "model", which)
    by.insert(0, "workload", ds)
    return out, by.reset_index(), j


def bars(ax, t, cols, title, ref=None, fmt="{:.0f}%"):
    """Grouped horizontal bars, frozen vs learned, one group per workload."""
    y = np.arange(len(t))[::-1]
    for k, (col, colour, lab) in enumerate(cols):
        v = t[col].to_numpy()
        yy = y + (0.2 if k == 0 else -0.2)
        ax.barh(yy, v, 0.38, color=colour, label=lab)
        for y1, x in zip(yy, v):
            ax.text(x, y1, " " + fmt.format(x), va="center", fontsize=7.5, color=colour)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls="--")
        ax.text(ref, len(t) - 0.45, f" synthetic {ref:.0f}%", fontsize=7.5,
                color=INK2, va="bottom")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color="#e6e9ed", lw=0.6)
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, max(t[[c for c, _, _ in cols]].to_numpy().max() * 1.3,
                       (ref or 0) * 1.6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--out", default=os.path.join(FIGS, "v2_prediction_error.png"))
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    a = ap.parse_args()
    dss = a.datasets or [d for d in FULL if os.path.exists(
        os.path.join(RUNS, f"{d}_learn_nolog", "v2_pred.csv"))]
    names, store = ev.settings_list()
    rows, types, worst = [], [], []
    for ds in dss:
        blobs, truth = ev.load_truth(os.path.join(STORE, ds, "exhaustive"), len(names), store)
        meas = measured(ds)
        for which in ("frozen", "learned"):
            o, by, j = score(ds, which, names, store, blobs, truth, meas)
            rows.append(o)
            types.append(by)
            if which == "frozen":
                w = j.groupby("spec")[["ct_ape", "dt_ape", "ratio_ape"]].median()
                w.insert(0, "workload", ds)
                worst.append(w.reset_index())
    t = pd.DataFrame(rows)
    ty = pd.concat(types, ignore_index=True)
    t.to_csv(os.path.join(RUNS, "pred_error_real.csv"), index=False)
    ty.to_csv(os.path.join(RUNS, "pred_error_by_type.csv"), index=False)
    pd.concat(worst).to_csv(os.path.join(RUNS, "pred_error_by_setting.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(1).to_string(index=False))
    print(ty.round(1).to_string(index=False))
    plot(t, a.out, a.pdf)


def plot(t, out, pdf):
    """Error of compress time, decompress time and ratio, and ranking quality."""
    wide = t.pivot(index="workload", columns="model")
    wide.columns = [f"{c}_{m}" for c, m in wide.columns]
    wide = wide.reindex([d for d in FULL if d in wide.index])
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, q, lab in zip(axes[:3], ("ct", "dt", "ratio"),
                          ("compress time", "decompress time", "compression ratio")):
        bars(ax, wide, [(f"{q}_ape_median_frozen", FROZEN, "as trained (no learning)"),
                        (f"{q}_ape_median_learned", LEARNED, "with online learning")],
             f"Error of predicted {lab}\n(median |pred - measured| / measured)",
             CV_MEDIAN_APE[q])
    bars(axes[3], wide, [("top1_hit_pct_frozen", FROZEN, "as trained (no learning)"),
                         ("top1_hit_pct_learned", LEARNED, "with online learning")],
         "Chunks where the predicted cheapest\nsetting is truly the cheapest (%)", CV_TOP1)
    axes[3].set_xlim(0, 115)
    axes[3].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_yticks(np.arange(len(wide))[::-1])
    axes[0].set_yticklabels([f"{d}\n({int(n)} chunks)" for d, n in
                             zip(wide.index, wide.chunks_frozen)], fontsize=9, color=INK)
    fig.suptitle("How accurate are NeuroPress v2's predictions on real workloads?",
                 x=0.01, ha="left", fontsize=14, color=INK, y=0.99)
    fig.text(0.01, 0.905, "Each chunk's prediction for all 44 compressing settings vs "
             "the exhaustive search's measurement of the same chunk and setting. Dashed "
             "line: the same measure on the synthetic float32 data the model was trained "
             "on (5-fold cross-validation).\nRight: the cheapest by the balanced 4-tier cost "
             "(compress + decompress + I/O at the chunk's tier) using predicted vs "
             "measured values. Online learning: prediction made when the chunk was "
             "chosen, after updates from all earlier chunks.",
             fontsize=9, color=INK2, va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.07, wspace=0.12)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    if pdf:
        fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()

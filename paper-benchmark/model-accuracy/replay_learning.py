#!/usr/bin/env python3
"""Replay NeuroPress v2 online learning offline, to compare learning rules.

    replay_learning.py check DATASET            (replay vs the Clio run)
    replay_learning.py compare [DATASET ...] [--passes P]   (rules side by side)

The replay runs what Clio's compressor does per chunk, in arrival order (the
learning run's v2_pred.csv): the network (model_v2.nnwt, float32) predicts
compress time, decompress time and ratio of all 45 settings from the chunk's
four inputs; the cheapest by the balanced 4-tier cost (compress + decompress
+ bytes / (ratio x tier bandwidth), as RankKernel) is picked; unless it is
store, its measured outcome is the label, and when the cost error
|measured - predicted| / measured exceeds the threshold the weights are
updated (TrainKernel). Labels come from the stored exhaustive search, which
measured every setting on every chunk, so any rule can be replayed and every
setting's prediction scored over time.

`check` replays the current rule with the Clio run's own labels and
compares the replayed predictions with the ones the run logged.

Rules (all only see the picked setting's outcome; no exploration):
  nlms         Clio today: normalised LMS on the picked setting's three
               output rows of the last layer
  nlms+shared  also moves every setting's output by the same log-space
               correction at this input (errors are shared across settings)
  backprop     SGD through all layers on the picked setting's outputs
--passes P replays the workload P times in a row, as more time steps of
the same data, to see whether a rule converges given longer. Writes
runs/replay_summary_<P>pass.csv (per workload, rule and pass).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import eval_v2_workloads as ev
from train_nn_v2 import read_nnwt

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/mnt/nvme0/v2-work/baselines"
RUNS = "/mnt/nvme0/v2-work/runs"
MODEL = os.path.join(HERE, "..", "..", "context-transport-primitives", "src", "compress",
                     "model", "weights", "v2", "model_v2.nnwt")
FULL = ev.FULL_WORKLOADS


class Net:
    """The v2 network with writable float32 weights."""

    def __init__(self, path=MODEL):
        dims, xm, xs, ym, ys, layers, settings = read_nnwt(path)
        self.xm, self.xs = xm.copy(), xs.copy()
        self.ym, self.ys = ym.astype(np.float64), ys.astype(np.float64)
        self.W = [w.copy() for w, _ in layers]
        self.b = [b.copy() for _, b in layers]
        self.settings = settings

    def forward(self, x_raw):
        """Activations of every layer (input first) and standardised outputs.
        ReLU is fmax(0, z) as on the GPU (fmaxf), so a NaN input (a chunk
        whose statistics are NaN, e.g. NaN-masked land in ERA5 SST) zeroes the
        first hidden layer instead of propagating."""
        a = [((np.asarray(x_raw, np.float32) - self.xm) / self.xs).astype(np.float32)]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = w @ a[-1] + b
            a.append(np.fmax(z, np.float32(0)) if i < len(self.W) - 1 else z)
        return a

    def log_pred(self, y_std):
        """(n_settings, 3) natural-log ct, dt, ratio."""
        return (y_std.astype(np.float64) * self.ys + self.ym).reshape(-1, 3)


def step_nlms(net, a, s, target_log, lr, shared):
    """Normalised LMS on setting s's output rows; `shared` also moves every
    setting by the same log-space correction (weighted by `shared`)."""
    h = a[-2].astype(np.float64)
    h2 = 1.0 + h @ h
    W, b = net.W[-1], net.b[-1]
    pred_log = net.log_pred(a[-1])[s]
    for k in range(3):
        if not np.isfinite(target_log[k]):
            continue
        o = 3 * s + k
        sd = net.ys[o]
        err = (pred_log[k] - target_log[k]) / sd
        st = lr * err / h2
        b[o] -= np.float32(st)
        W[o] -= (st * h).astype(np.float32)
        if shared > 0:
            # The same log-space shift for every other setting's output k.
            others = np.arange(k, W.shape[0], 3)
            others = others[others != o]
            st_o = shared * (pred_log[k] - target_log[k]) / net.ys[others] / h2
            b[others] -= st_o.astype(np.float32)
            W[others] -= np.outer(st_o, h).astype(np.float32)


def step_backprop(net, a, s, target_log, lr):
    """One SGD step through all layers on setting s's (up to) three outputs."""
    rows = [3 * s + k for k in range(3) if np.isfinite(target_log[k])]
    g = np.zeros(net.W[-1].shape[0])
    pred_log = net.log_pred(a[-1])[s]
    for k in range(3):
        if np.isfinite(target_log[k]):
            g[3 * s + k] = (pred_log[k] - target_log[k]) / net.ys[3 * s + k]
    if not rows:
        return
    for i in range(len(net.W) - 1, -1, -1):
        gw = np.outer(g, np.nan_to_num(a[i]))
        gb = g
        if i > 0:
            g = (net.W[i].T.astype(np.float64) @ g) * (a[i] > 0)
        net.W[i] -= (lr * gw).astype(np.float32)
        net.b[i] -= (lr * gb).astype(np.float32)


def load(ds):
    """Arrival order, inputs and every setting's measured outcome for one workload."""
    names, store = ev.settings_list()
    # Arrival order and inputs from the learning run, or, for a workload with
    # only an exhaustive search (a tuning probe), from that run: same chunks,
    # same order, same inputs and tiers.
    src = os.path.join(RUNS, f"{ds}_learn_nolog", "v2_pred.csv")
    if not os.path.exists(src):
        src = os.path.join(STORE, ds, "exhaustive", "v2_pred.csv")
    order = pd.read_csv(src, usecols=["blob", "bytes", "tier_bw", "x0", "x1", "x2", "x3"])
    m = pd.read_csv(os.path.join(STORE, ds, "exhaustive", "v2_measured.csv"))
    m = m[~ev.raw_primary(m) & (m.decomp_ms > 0)].drop_duplicates(["blob", "setting"])
    idx = {b: i for i, b in enumerate(order.blob)}
    meas = np.full((len(order), len(names), 3), np.nan)
    r = m.blob.map(idx)
    keep = r.notna()
    meas[r[keep].astype(int), m.setting[keep], 0] = m.comp_ms[keep]
    meas[r[keep].astype(int), m.setting[keep], 1] = m.decomp_ms[keep]
    meas[r[keep].astype(int), m.setting[keep], 2] = m.ratio[keep]
    blobs, truth = ev.load_truth(os.path.join(STORE, ds, "exhaustive"), len(names), store)
    t_idx = {b: i for i, b in enumerate(blobs)}
    cost = truth[[t_idx[b] for b in order.blob]]
    return names, store, order, meas, cost


# Clio's learning gate (neuropress_mape_threshold, user 2026-10-05: 0.30) and
# what Clio learns from: the write only. NeuroPress does no work on reads, so
# it never gets a measured decompress time (no_dt=True by default).
MAPE_THRESHOLD = 0.30


def replay(ds, rule, lr, shared=0.0, thr=MAPE_THRESHOLD, labels=None, data=None, no_dt=True,
           w=(1.0, 1.0, 1.0), bw=None):
    """Replay one workload under one rule; per-chunk record and predictions.

    @param w   cost weights (w_ct, w_dt, w_io) of the selection and of the
               learning trigger, as CLIO_NEUROPRESS_COST_W_CT/_W_DT/_W_IO;
               the default is the balanced cost
    @param bw  one bandwidth (bytes per ms) for every chunk, as
               CLIO_NEUROPRESS_COST_BW with no tiers; None (default) uses
               each chunk's tier_bw
    The record's cost_pick / cost_best come from data's cost matrix, so pass
    data with the truth cost of the same weights and bandwidth.
    """
    names, store, order, meas, cost = data or load(ds)
    net = Net()
    n, ns = len(order), len(names)
    X = order[["x0", "x1", "x2", "x3"]].to_numpy(np.float64)
    nbytes = order.bytes.to_numpy(float)
    bw = order.tier_bw.to_numpy(float) if bw is None else np.full(n, float(bw))
    w_ct, w_dt, w_io = (float(v) for v in w)
    preds = np.empty((n, ns, 3))
    rec = []
    updates = 0
    for i in range(n):
        a = net.forward(X[i])
        lp = net.log_pred(a[-1])
        preds[i] = lp
        ct, dt, r = np.exp(lp[:, 0]), np.exp(lp[:, 1]), np.exp(lp[:, 2])
        pc = w_ct * ct + w_dt * dt + w_io * nbytes[i] / (r * bw[i])
        s = int(np.argmin(pc))
        lab = labels[i] if labels is not None else meas[i, s]
        trained = False
        if s != store and np.isfinite(lab[0]) and lab[2] > 0:
            # As NeuroPressV2LearnPrimary: an unmeasured decompress time
            # counts as predicted in the cost and is not a label.
            # no_dt: no decompress on the write path, so no measured
            # decompress time while writing (Clio learns it only at read).
            dt_ok = (not no_dt) and np.isfinite(lab[1]) and lab[1] > 0
            act = (w_ct * lab[0] + w_dt * (lab[1] if dt_ok else dt[s])
                   + w_io * nbytes[i] / (lab[2] * bw[i]))
            err = abs(act - pc[s]) / act
            if err > thr:
                tl = np.log(np.where([True, dt_ok, True], lab, np.nan))
                if rule == "backprop":
                    step_backprop(net, a, s, tl, lr)
                else:
                    step_nlms(net, a, s, tl, lr, shared)
                updates += 1
                trained = True
        sel = ev.for_selection(cost[i])   # the oracle selects from the candidates
        best = int(np.nanargmin(sel)) if np.isfinite(sel).any() else -1
        rec.append({"blob": order.blob[i], "pick": s, "best": best,
                    "cost_pick": cost[i, s], "cost_best": cost[i, best] if best >= 0 else np.nan,
                    "trained": trained, "updates": updates})
    return pd.DataFrame(rec), preds, meas


def check(ds):
    """Replay the current rule with the Clio run's own labels; compare."""
    names, store = ev.settings_list()
    run = os.path.join(RUNS, f"{ds}_learn_nolog")
    logged = pd.read_csv(os.path.join(run, "v2_pred.csv"))
    prim = pd.read_csv(os.path.join(run, "v2_measured.csv"))
    prim = prim[prim.role == "primary"].drop_duplicates("blob").set_index("blob")
    labels = np.full((len(logged), 3), np.nan)
    run_pick = np.full(len(logged), -1)
    for i, b in enumerate(logged.blob):
        if b in prim.index:
            p = prim.loc[b]
            labels[i] = [p.comp_ms, p.decomp_ms, p.ratio]
            run_pick[i] = p.setting
    data = load(ds)
    rec, preds, _ = replay(ds, "nlms", 0.5, labels=labels, data=data)
    lp = np.stack([np.log(logged[[f"ct{k}", f"dt{k}", f"r{k}"]].to_numpy(float))
                   for k in range(len(names))], axis=1)
    diff = np.abs(preds - lp)
    same = (rec.pick.to_numpy() == run_pick) | (run_pick < 0)
    first = int(np.argmin(same)) if not same.all() else len(same)
    print(f"{ds}: replayed picks equal Clio's on {same.mean() * 100:.1f}% of "
          f"{len(same)} chunks (first difference at chunk {first}); "
          f"updates {rec.updates.iloc[-1]} replayed vs {logged.updates.iloc[-1]} logged "
          f"before the last chunk")
    print(f"  |replayed - logged| log prediction: median {np.median(diff):.2e}, "
          f"max over chunks before the first difference "
          f"{diff[:first].max() if first > 0 else float('nan'):.2e}")


RULES = [("frozen", None, 0.0, 0.0), ("nlms lr 0.5 (Clio)", "nlms", 0.5, 0.0),
         ("nlms lr 0.2", "nlms", 0.2, 0.0), ("nlms lr 1.0", "nlms", 1.0, 0.0)]
# Tried and dropped (2026-10-05, 5 passes of omics / tpch / era5): a shared
# log-space correction of every setting ("nlms" with shared > 0) raised the
# time error to 40-50 % and lost to the best single codec by 13-73 %; full
# backpropagation (step_backprop, lr 1e-3 / 1e-2) diverged on omics and tpch.


def summarize(ds, label, rec, preds, meas, n1, fixed):
    """Scores of one replay, per pass over the workload (n1 chunks each)."""
    ape = np.abs(np.exp(preds) - meas) / meas * 100  # chunks x settings x 3
    rows = []
    for p in range(len(rec) // n1):
        sl = slice(p * n1, (p + 1) * n1)
        r = rec.iloc[sl]
        rows.append({"workload": ds, "rule": label, "pass": p + 1, "chunks": n1,
                     "updates_in_pass": int(r.trained.sum()),
                     "ct_err": np.nanmedian(ape[sl, :, 0]),
                     "dt_err": np.nanmedian(ape[sl, :, 1]),
                     "ratio_err": np.nanmedian(ape[sl, :, 2]),
                     "top1": 100 * (r.pick == r.best).mean(),
                     "settings_picked": r.pick.nunique(),
                     "cost_vs_best_single_pct": 100 * (r.cost_pick.sum() / fixed - 1),
                     "oracle_vs_best_single_pct": 100 * (r.cost_best.sum() / fixed - 1)})
    return rows


def tile(data, passes):
    """The workload replayed `passes` times in a row (more time steps)."""
    names, store, order, meas, cost = data
    return (names, store, pd.concat([order] * passes, ignore_index=True),
            np.concatenate([meas] * passes), np.concatenate([cost] * passes))


def cache_path(ds, passes, label):
    """Per-workload, per-rule replay result kept between calls."""
    slug = "".join(c if c.isalnum() else "_" for c in label)
    return os.path.join(RUNS, "replay_cache", f"{ds}_{passes}pass_{slug}.csv")


def cache_fresh(path, ds):
    """True when a cached result is newer than the workload's inputs (the
    exhaustive baseline and the learning run)."""
    if not os.path.exists(path):
        return False
    inputs = [os.path.join(STORE, ds, "exhaustive", "meta.json"),
              os.path.join(RUNS, f"{ds}_learn_nolog", "v2_pred.csv")]
    return all(os.path.getmtime(path) > os.path.getmtime(i) for i in inputs
               if os.path.exists(i))


def compare(dss, passes, only=()):
    """Every rule (or those named in `only`) on every workload, `passes` each.
    A workload/rule already replayed with the same inputs is read from
    runs/replay_cache instead of replayed again."""
    rows = []
    rules = [r for r in RULES if not only or r[0] in only]
    os.makedirs(os.path.join(RUNS, "replay_cache"), exist_ok=True)
    for ds in dss:
        todo = [r for r in rules if not cache_fresh(cache_path(ds, passes, r[0]), ds)]
        if todo:
            data = load(ds)
            names, store, order, meas, cost = data
            n1 = len(order)
            fixed = np.nanmin(np.nansum(ev.for_selection(cost), axis=0))
            big = tile(data, passes)
        for label, rule, lr, sh in rules:
            cp = cache_path(ds, passes, label)
            if (label, rule, lr, sh) in todo:
                rec, preds, _ = replay(ds, rule or "nlms", lr if rule else 0.0, sh,
                                       thr=MAPE_THRESHOLD if rule else np.inf, data=big)
                pd.DataFrame(summarize(ds, label, rec, preds, big[3], n1, fixed)).to_csv(
                    cp, index=False)
            out = pd.read_csv(cp).to_dict("records")
            rows += out
            f, l = out[0], out[-1]
            print(f"{ds:15s} {label:24s} pass 1 -> {len(out)}: ratio err {f['ratio_err']:6.1f}% -> "
                  f"{l['ratio_err']:6.1f}%  ct err {f['ct_err']:5.1f}% -> {l['ct_err']:5.1f}%  "
                  f"top-1 {f['top1']:4.0f}% -> {l['top1']:4.0f}%  cost vs best single "
                  f"{f['cost_vs_best_single_pct']:+6.1f}% -> {l['cost_vs_best_single_pct']:+6.1f}% "
                  f"(oracle {f['oracle_vs_best_single_pct']:+.1f}%)  updates "
                  f"{f['updates_in_pass']} -> {l['updates_in_pass']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RUNS, f"replay_summary_{passes}pass.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["check", "compare"])
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--rules", default="",
                    help="comma-separated rule labels to run (default: all), "
                         "e.g. 'frozen,nlms lr 0.5 (Clio)'")
    ap.add_argument("--passes", type=int, default=1,
                    help="replay each workload this many times in a row")
    a = ap.parse_args()
    dss = a.datasets or [d for d in FULL if ev.run_finished(
        os.path.join(RUNS, f"{d}_learn_nolog"))]
    if a.action == "check":
        for ds in dss:
            check(ds)
    else:
        compare(dss, a.passes, [r for r in a.rules.split(",") if r])


if __name__ == "__main__":
    sys.exit(main())

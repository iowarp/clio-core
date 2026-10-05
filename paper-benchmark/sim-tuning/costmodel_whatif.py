#!/usr/bin/env python3
"""What if the data is written once and read R times from one storage tier?

    costmodel_whatif.py [WORKLOAD] [--scenario NAME:W_CT,W_DT,W_IO:BW:READS:LABEL ...]
                        [--passes P] [--csv PATH] [--png PATH] [--replot]

An offline what-if (CPU only) from stored measurements: no GPU, no Clio run.

Inputs (read only):
  baselines/<ws>/exhaustive/v2_measured.csv  every compressing setting measured
                                            on every chunk (ct, dt, ratio)
  runs/<ws>_learn_nolog/v2_pred.csv          arrival order and NeuroPress's
                                            four inputs per chunk (else the
                                            exhaustive run's v2_pred.csv)
  runs/<ws>_learn_nolog/phases.csv           nn_ms: NeuroPress prediction ms
                                            per chunk on the write path
All loaded by model-accuracy/replay_learning.load.

Cost of storing a chunk with a setting (the formula Clio's v2 selection uses,
with CLIO_NEUROPRESS_COST_W_CT/_W_DT/_W_IO and CLIO_NEUROPRESS_COST_BW):
  w_ct * ct + w_dt * dt + w_io * bytes / (ratio * BW)   ratio > 1
  w_ct * ct + w_io * bytes / BW                          ratio <= 1 (Clio
                                                         stores the chunk raw)
  w_io * bytes / BW                                      store (raw)
BW is one bandwidth for every chunk (bytes per ms), or 'tiers' for each
chunk's tier_bw from the files (the old balanced 4-tier model). 1 write + R
reads is w = (1, R, R + 1). NeuroPress's selection and its learning trigger
use the same weights and BW (replay_learning.replay with w= and bw=).

Per scenario and method (best single setting, per-chunk oracle, NeuroPress as
trained, NeuroPress learning lr 0.5 / threshold 0.30 / no measured dt on the
write path, pass 1 and pass P of the workload replayed P times; for the
'tiers' scenario also learning with the measured dt, as a check): total cost
and % vs the best single, whole-workload compression ratio (original bytes /
stored bytes), modeled time write + R x read in s (write = ct + stored / BW,
read = dt + stored / BW, dt = 0 for raw-stored chunks) plus, for NeuroPress,
the learning run's summed nn_ms, the settings picked, and per field the cost
share and % vs the best single. Before anything else, the 'tiers' (1, 1, 1)
replay is checked against replay_learning.replay's default.

Writes runs/whatif_<ws>.csv (one row per scenario and method) and
figures/new-workloads/sim-tuning/whatif_<ws>.png.
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model-accuracy"))
import replay_learning as rl  # noqa: E402
import eval_v2_workloads as ev  # noqa: E402

RUNS = rl.RUNS
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning")
PFS_BW = 0.52e6  # bytes per ms (0.52 GB/s)
# NAME:W_CT,W_DT,W_IO:BW (bytes per ms, or 'tiers'):READS (for the modeled time):LABEL
SCENARIOS = [
    "A:1,4,5:0.52e6:4:1 write + 4 reads",
    "B:1,1,2:0.52e6:1:1 write + 1 read",
    "C:1,1,1:tiers:1:old balanced 4-tier model",
    "D:1,3,3:0.52e6:3:weights 1/3/3",
    "E:0,1,1:0.52e6:1:weights 0/1/1",
    "F:1,1,1:0.52e6:1:weights 1/1/1 (PFS only)",
    "G:0,0,1:0.52e6:1:weights 0/0/1 (transfer only)",
]
LR, THR = 0.5, rl.MAPE_THRESHOLD
BEST, ORACLE = "best single setting", "oracle (per-chunk best)"
FROZEN, LEARN1 = "NeuroPress as trained", "NeuroPress learning, pass 1"
CHECK = "NeuroPress learning, pass 1, dt measured (check)"
INK, INK2, GRID = "#1f2328", "#57606a", "#e6e9ed"
COLORS = {BEST: "#57606a", FROZEN: "#2a78d6", LEARN1: "#eb6834", ORACLE: "#c9ccd1"}
PASS_COLOR = "#1baf7a"


def parse_scenario(text):
    """One scenario from 'NAME:W_CT,W_DT,W_IO:BW:READS:LABEL'.

    @param text  the scenario string; BW in bytes per ms or 'tiers'
    @return dict with name, w (3 floats), bw (float, or None for tiers),
            reads (int) and label
    """
    name, w, bw, reads, label = text.split(":", 4)
    return {"name": name, "w": tuple(float(v) for v in w.split(",")),
            "bw": None if bw == "tiers" else float(bw), "reads": int(reads),
            "label": label}


def field(blob):
    """Field name of a chunk ('..._comp00_density.f32__...' -> 'density')."""
    m = re.search(r"comp\d+_([A-Za-z_]+)\.f32", blob)
    return m.group(1) if m else "?"


def chunk_bw(order, bw):
    """Per-chunk bandwidth (bytes per ms): one value, or each chunk's tier.

    @param order  arrival-order DataFrame with tier_bw
    @param bw     bytes per ms, or None for the files' tier_bw
    """
    return order.tier_bw.to_numpy(float) if bw is None else np.full(len(order), bw)


def outcomes(meas, nbytes, store):
    """Per chunk and setting: compress ms, decompress ms paid at read, stored bytes.

    A setting with ratio <= 1 is stored raw (its compress time is still
    paid, nothing is decompressed); store costs no compute.

    @param meas    chunk x setting x (ct, dt, ratio), NaN where unmeasured
    @param nbytes  chunk sizes in bytes
    @param store   index of the store setting
    @return (ct, dt_read, stored_bytes), each chunk x setting
    """
    ct, dt, r = meas[..., 0].copy(), meas[..., 1].copy(), meas[..., 2]
    kept = r > 1
    raw = np.broadcast_to(nbytes[:, None], r.shape)
    stored = np.where(kept, raw / np.where(kept, r, 1.0), raw)
    stored[np.isnan(r)] = np.nan
    dt = np.where(kept, dt, np.where(np.isnan(r), np.nan, 0.0))
    ct[:, store], dt[:, store], stored[:, store] = 0.0, 0.0, nbytes
    return ct, dt, stored


def truth_cost(meas, nbytes, store, w, bw):
    """Chunk x setting truth cost under weights w and per-chunk bandwidth bw.

    @param meas    chunk x setting x (ct, dt, ratio)
    @param nbytes  chunk sizes in bytes
    @param store   index of the store setting
    @param w       (w_ct, w_dt, w_io)
    @param bw      per-chunk bandwidth in bytes per ms
    @return cost[chunk, setting] in ms, NaN where unmeasured
    """
    w_ct, w_dt, w_io = w
    ct, dt, r = meas[..., 0], meas[..., 1], meas[..., 2]
    raw_io = (nbytes / bw)[:, None]
    cost = np.where(r > 1, w_ct * ct + w_dt * dt + w_io * raw_io / r,
                    w_ct * ct + w_io * raw_io)
    cost[np.isnan(r)] = np.nan
    cost[:, store] = w_io * nbytes / bw
    return cost


def verify(ds, data):
    """Check the generic cost and replay against replay_learning's defaults.

    With weights (1, 1, 1) and the files' tier_bw, truth_cost must equal
    load()'s cost and the weighted replay must pick exactly what the
    default replay picks (no_dt False and True). Raises on a mismatch.

    @param ds    workload name
    @param data  replay_learning.load(ds)
    @return {no_dt: cost vs best single %} of the default replay
    """
    names, store, order, meas, cost = data
    nbytes = order.bytes.to_numpy(float)
    mine = truth_cost(meas, nbytes, store, (1.0, 1.0, 1.0), chunk_bw(order, None))
    diff = np.nanmax(np.abs(mine - cost) / cost)
    same_nan = np.array_equal(np.isnan(mine), np.isnan(cost))
    print(f"check: truth cost vs load(): max relative difference {diff:.3e}, "
          f"same unmeasured cells {same_nan}")
    if not (same_nan and diff < 1e-12):
        raise SystemExit("truth cost does not match replay_learning.load()")
    fixed = np.nanmin(np.nansum(ev.for_selection(cost), axis=0))
    out = {}
    for no_dt in (False, True):
        ref, _, _ = rl.replay(ds, "nlms", LR, thr=THR, data=data, no_dt=no_dt)
        new, _, _ = rl.replay(ds, "nlms", LR, thr=THR, no_dt=no_dt, w=(1.0, 1.0, 1.0),
                              bw=None, data=(names, store, order, meas, mine))
        same = (ref.pick.to_numpy() == new.pick.to_numpy()).mean()
        out[no_dt] = 100 * (ref.cost_pick.sum() / fixed - 1)
        print(f"check: replay default (no_dt={no_dt}) {out[no_dt]:+.12f}% vs best single, "
              f"weighted replay {100 * (new.cost_pick.sum() / fixed - 1):+.12f}%, "
              f"same picks on {100 * same:.4f}% of chunks")
        if same < 1:
            raise SystemExit("weighted replay does not reproduce the default replay")
    return out


def predict_seconds(ds):
    """Summed NeuroPress prediction time (s) on the learning run's write path,
    NaN when the workload has no learning run."""
    p = os.path.join(RUNS, f"{ds}_learn_nolog", "phases.csv")
    if not os.path.exists(p):
        return np.nan
    ph = pd.read_csv(p, usecols=["path", "nn_ms"])
    return ph[ph.path == "write"].nn_ms.sum() / 1e3


def score(picks, ctx, method):
    """One CSV row: cost, ratio, modeled time, picks and per-field scores.

    @param picks   setting index per chunk (arrival order)
    @param ctx     scenario context (see run_scenario)
    @param method  method label
    """
    r = np.arange(len(picks))
    ok = ctx["ok"]
    c = ctx["cost"][r, picks]
    stored = ctx["stored"][r, picks]
    write_ms = ctx["ct"][r, picks] + stored / ctx["bw"]
    read_ms = ctx["dt"][r, picks] + stored / ctx["bw"]
    pred_s = ctx["predict_s"] if method.startswith("NeuroPress") else 0.0
    t = (write_ms[ok].sum() + ctx["reads"] * read_ms[ok].sum()) / 1e3 + pred_s
    names = ctx["names"]
    counts = pd.Series([names[s] for s in picks[ok]]).value_counts()
    row = {"method": method, "cost_ms": c[ok].sum(),
           "vs_best_single_pct": 100 * (c[ok].sum() / ctx["single"] - 1),
           "ratio": ctx["nbytes"][ok].sum() / stored[ok].sum(),
           "write_s": write_ms[ok].sum() / 1e3, "read_s": read_ms[ok].sum() / 1e3,
           "predict_s": pred_s, "time_s": t,
           "time_vs_best_single_pct": 100 * (t / ctx["single_time"] - 1)
           if "single_time" in ctx else 0.0,
           "top1_pct": 100 * np.mean(picks[ok] == ctx["best"][ok]),
           "settings_picked": len(counts),
           "top_picks": "; ".join(f"{k} {100 * v / ok.sum():.2f}%"
                                  for k, v in counts.head(5).items())}
    for f in ctx["field_names"]:
        sel = ok & (ctx["fields"] == f)
        row[f"field_{f}_cost_share_pct"] = 100 * c[sel].sum() / c[ok].sum()
        row[f"field_{f}_vs_best_single_pct"] = 100 * (
            c[sel].sum() / ctx["single_field"][f] - 1)
    return row


def replays(ds, data_s, sc, passes):
    """NeuroPress picks under one scenario's weights and bandwidth.

    @param ds      workload name
    @param data_s  load()'s tuple with the scenario's truth cost
    @param sc      parsed scenario
    @param passes  passes of the learning replay
    @return list of (method, picks, updates in that pass or 0)
    """
    n = len(data_s[2])
    kw = {"w": sc["w"], "bw": sc["bw"]}
    frozen, _, _ = rl.replay(ds, "nlms", 0.0, thr=np.inf, data=data_s, **kw)
    out = [(FROZEN, frozen.pick.to_numpy(), 0)]
    learn, _, _ = rl.replay(ds, "nlms", LR, thr=THR, no_dt=True,
                            data=rl.tile(data_s, passes), **kw)
    for p in sorted({1, passes}):
        sl = slice((p - 1) * n, p * n)
        out.append((f"NeuroPress learning, pass {p}", learn.pick.to_numpy()[sl],
                    int(learn.trained.iloc[sl].sum())))
    if sc["bw"] is None:
        chk, _, _ = rl.replay(ds, "nlms", LR, thr=THR, no_dt=False, data=data_s, **kw)
        out.append((CHECK, chk.pick.to_numpy(), int(chk.trained.sum())))
    return out


def run_scenario(ds, data, sc, passes, predict_s):
    """All methods' rows for one scenario.

    @param ds         workload name
    @param data       replay_learning.load(ds)
    @param sc         parsed scenario
    @param passes     passes of the learning replay
    @param predict_s  summed NeuroPress prediction time (s)
    @return list of row dicts
    """
    names, store, order, meas, _ = data
    nbytes = order.bytes.to_numpy(float)
    bw = chunk_bw(order, sc["bw"])
    cost = truth_cost(meas, nbytes, store, sc["w"], bw)
    ct, dt, stored = outcomes(meas, nbytes, store)
    ok = ~np.isnan(cost).any(axis=1)
    if not ok.all():
        print(f"  {sc['name']}: {int((~ok).sum())} chunk(s) not measured on every "
              f"setting are left out of the scores")
    fields = np.array([field(b) for b in order.blob])
    # The best single codec and the oracle select from the candidate settings
    # (eval_v2_workloads.DROPPED_SETTINGS left out).
    sel = ev.for_selection(np.where(np.isnan(cost), np.inf, cost))
    single = np.where(ok[:, None], sel, 0.0).sum(axis=0)
    bs = int(np.argmin(single))
    best = np.where(ok, np.argmin(sel, axis=1), bs)
    ctx = {"names": names, "cost": cost, "ct": ct, "dt": dt, "stored": stored,
           "bw": bw, "nbytes": nbytes, "ok": ok, "reads": sc["reads"],
           "predict_s": predict_s, "single": cost[ok, bs].sum(), "best": best,
           "fields": fields, "field_names": sorted(set(fields)),
           "single_field": {f: cost[ok & (fields == f), bs].sum() for f in set(fields)}}
    base = score(np.full(len(order), bs), ctx, BEST)
    ctx["single_time"] = base["time_s"]
    base["time_vs_best_single_pct"] = 0.0
    rows = [dict(base, setting=names[bs]), dict(score(best, ctx, ORACLE), setting="")]
    data_s = (names, store, order, meas, cost)
    for method, picks, upd in replays(ds, data_s, sc, passes):
        rows.append(dict(score(picks, ctx, method), setting="", updates=upd))
    opp = 100 * (1 - rows[1]["cost_ms"] / rows[0]["cost_ms"])
    head = {"workload": ds, "scenario": sc["name"], "label": sc["label"],
            "w_ct": sc["w"][0], "w_dt": sc["w"][1], "w_io": sc["w"][2],
            "bw_bytes_per_ms": "tiers" if sc["bw"] is None else sc["bw"],
            "reads": sc["reads"], "best_single": names[bs], "opportunity_pct": opp,
            "chunks_scored": int(ok.sum()), "bytes_scored": nbytes[ok].sum()}
    return [dict(head, **r) for r in rows]


def report(t):
    """Print the main numbers per scenario."""
    pd.set_option("display.width", 250, "display.max_columns", 40,
                  "display.max_colwidth", 200)
    for sc, g in t.groupby("scenario", sort=False):
        h = g.iloc[0]
        print(f"\n== {sc}: {h.label}  w = {h.w_ct:g}/{h.w_dt:g}/{h.w_io:g}, "
              f"BW {h.bw_bytes_per_ms}, R = {h.reads}; best single {h.best_single}; "
              f"opportunity {h.opportunity_pct:.4f}%")
        print(g[["method", "cost_ms", "vs_best_single_pct", "ratio", "write_s", "read_s",
                 "predict_s", "time_s", "time_vs_best_single_pct", "top1_pct",
                 "settings_picked", "updates"]].to_string(index=False))
        fcols = [c for c in g.columns if c.startswith("field_")]
        print(g[["method"] + fcols].to_string(index=False))
        for _, r in g[g.method.str.startswith("NeuroPress")].iterrows():
            print(f"  {r.method}: {r.top_picks}")


def plot(t, ds, size, out):
    """Cost, modeled runtime and compression ratio per scenario, against the
    best single setting. Every option selects by the scenario's cost (user,
    2026-10-05): the best single setting has the lowest total cost, NeuroPress
    the lowest predicted cost per chunk, the oracle the lowest measured cost
    per chunk.

    @param t     the rows of every scenario and method
    @param ds    workload name
    @param size  workload size for the title, e.g. '12816 chunks, 50.1 GiB'
    @param out   PNG path
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    last = [m for m in t.method.unique()
            if re.fullmatch(r"NeuroPress learning, pass \d+", m)][-1]
    methods = [FROZEN, LEARN1] + ([last] if last != LEARN1 else []) + [ORACLE]
    colors = dict(COLORS, **{last: PASS_COLOR})
    scen = t.drop_duplicates("scenario")
    y = np.arange(len(scen))[::-1]
    H = 2.4 + 1.05 * len(scen)
    fig, axes = plt.subplots(1, 3, figsize=(19, H), sharey=True)
    fig.patch.set_facecolor("white")
    panels = [("vs_best_single_pct", methods,
               "Total cost vs the best single setting (%)\nnegative = cheaper", "{:+.2f}%"),
              ("time_vs_best_single_pct", methods,
               "Modeled runtime (write + R reads) vs the best single setting (%)\n"
               "NeuroPress includes its prediction time", "{:+.2f}%"),
              ("ratio", [BEST] + methods, "Whole-workload compression ratio\n"
               "(original bytes / stored bytes)", "{:.3f}")]
    for ax, (col, ms, title, fmt) in zip(axes, panels):
        h = 0.8 / len(ms)
        for k, m in enumerate(ms):
            v = [t[(t.scenario == s) & (t.method == m)][col].iloc[0] for s in scen.scenario]
            yy = y + 0.4 - h * (k + 0.5)
            ax.barh(yy, v, h * 0.9, color=colors[m], label=m)
            for yv, vv in zip(yy, v):
                ax.text(vv, yv, " " + fmt.format(vv) + " ", va="center", fontsize=6.5,
                        color=INK, ha="left" if vv >= 0 or col == "ratio" else "right")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.18 * (hi - lo) if lo < 0 else lo, hi + 0.18 * (hi - lo))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [f"{r.scenario}: {r.label}\nw_ct/w_dt/w_io = {r.w_ct:g}/{r.w_dt:g}/{r.w_io:g}, "
         f"{'per-chunk tiers' if r.bw_bytes_per_ms == 'tiers' else f'BW {float(r.bw_bytes_per_ms) / 1e6:g} GB/s'}, "
         f"R = {r.reads}\nbest single (lowest cost): {r.best_single}\n"
         f"oracle saves {r.opportunity_pct:.2f}% of the cost"
         for r in scen.itertuples()],
        fontsize=7.5, color=INK)
    names = [BEST + " (lowest total cost)"] + methods[:-1] + ["oracle (each chunk's lowest cost)"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in [BEST] + methods]
    fig.legend(handles, names, loc="upper left", ncol=len(handles), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.005, 1 - 0.72 / H))
    fig.suptitle(f"{ds} ({size}): one codec setting for every chunk vs NeuroPress "
                 f"choosing per chunk, under different read/write cost weights",
                 x=0.01, ha="left", fontsize=12.5, color=INK, y=1 - 0.12 / H)
    fig.text(0.01, 1 - 0.45 / H,
             "cost = w_ct x compress ms + w_dt x decompress ms + w_io x stored bytes / BW "
             "(chunk stored raw when ratio <= 1). All options select by the same cost: best "
             "single = lowest total cost; NeuroPress = lowest predicted cost per chunk (lr 0.5, "
             "threshold 0.30); oracle = lowest measured cost per chunk.",
             fontsize=8, color=INK2, va="top")
    fig.text(0.01, 0.08 / H,
             "Modeled runtime = sum(compress + stored / BW) + R x sum(decompress + stored / BW), "
             "no decompress for chunks stored raw; NeuroPress adds its prediction time (learning "
             "run's nn_ms).\nIt equals the cost only when the weights are 1 / R / R+1 (A, B); "
             "E and G ignore the compress time in the cost but not in the runtime; C uses each "
             "chunk's tier bandwidth. Measured GPU times from the exhaustive search.",
             fontsize=8, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 1.45 / H, bottom=0.8 / H, wspace=0.12)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", os.path.abspath(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workload", nargs="?", default="nyx-multiphase-50g")
    ap.add_argument("--scenario", action="append", default=None,
                    help="NAME:W_CT,W_DT,W_IO:BW:READS:LABEL, BW in bytes per ms or "
                         "'tiers' (repeat; default: scenarios A-G)")
    ap.add_argument("--passes", type=int, default=5,
                    help="passes of the learning replay (reports pass 1 and the last)")
    ap.add_argument("--csv", default=None, help="default runs/whatif_<workload>.csv")
    ap.add_argument("--png", default=None,
                    help="default figures/new-workloads/sim-tuning/whatif_<workload>.png")
    ap.add_argument("--replot", action="store_true",
                    help="only redraw the figure from an existing CSV")
    a = ap.parse_args()
    ds = a.workload
    csv = a.csv or os.path.join(RUNS, f"whatif_{ds}.csv")
    png = a.png or os.path.join(FIGS, f"whatif_{ds}.png")
    if a.replot:
        t = pd.read_csv(csv)
        h = t.iloc[0]
        plot(t, ds, f"{h.chunks_scored} chunks, {h.bytes_scored / 2**30:.1f} GiB", png)
        return 0
    scs = [parse_scenario(s) for s in (a.scenario or SCENARIOS)]
    data = rl.load(ds)
    verify(ds, data)
    predict_s = predict_seconds(ds)
    print(f"NeuroPress prediction time on the write path: {predict_s:.6f} s")
    rows = []
    for sc in scs:
        rows += run_scenario(ds, data, sc, a.passes, predict_s)
        print(f"scenario {sc['name']} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(csv, index=False)
    print("wrote", csv)
    report(t)
    h = t.iloc[0]
    plot(t, ds, f"{h.chunks_scored} chunks, {h.bytes_scored / 2**30:.1f} GiB", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
